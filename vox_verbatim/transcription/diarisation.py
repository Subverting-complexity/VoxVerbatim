"""Who said it, kept separate from what was said and when.

ElevenLabs tells the speakers apart while it transcribes, and it is the
initial authority on them for the same reason it is the authority on timing:
it is the service that actually listened for them. But diarisation is the
part of a transcript that goes wrong most visibly. Two people with similar
voices become one speaker; one person on a bad microphone becomes three; a
single word answer lands with the wrong person and reverses the meaning of
the exchange around it. So there is a second opinion available, and this
module decides when it is worth asking for and what to do with it.

The rule that matters more than anything else here is that **a speaker
correction never changes the text or the timing**. They are three separate
answers with three separate sources and three separate confidences. A word
spelled the way OpenAI heard it, timed the way ElevenLabs heard it and
attributed to the speaker AssemblyAI decided on is an ordinary word, not a
broken one. Every function here that changes a speaker touches the speaker
fields and nothing else, and there is a test whose only job is to prove it.

The hard part is not deciding to ask; it is understanding the answer.
ElevenLabs labels its speakers with its own string identifiers and
AssemblyAI labels its own with letters, and neither has any idea what the
other called anybody. Two lists of labels in the same order are not two
lists of the same people: a service that missed the first speaker's opening
sentence starts its lettering somewhere else entirely, and matching them by
position would then attribute every word in the recording to the wrong
person while looking perfectly tidy. So the labels are matched by the audio
they cover. Whoever was speaking during the same seconds is the same person,
which is the only evidence either service actually offers about identity.

The expected speaker count runs through all of it. A recording is usually
one person, sometimes two and occasionally three, and a service left to
guess freely will happily find eight. Saying how many people are expected is
what keeps the speaker space to a size the recording can support, and it is
used here rather than left as a setting nobody reads.

Nothing here depends on Qt or on any provider library, so it can be tested
on its own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Sequence

from vox_verbatim.transcription.escalation import Dispute, EscalationReason
from vox_verbatim.transcription.model import (
    AudioSpan,
    Confidence,
    FinalToken,
    Language,
    Provider,
    ProviderResult,
    ReviewReason,
)

_log = logging.getLogger(__name__)

#: A turn shorter than this is a very short utterance: a "yes", a "mm", a
#: name called across a room. They are the hardest thing in diarisation,
#: because there is barely enough voice in them to identify anybody by.
DEFAULT_MINIMUM_TURN_SECONDS = 1.0

#: How many times in a row the speaker may flip between very short turns
#: before the flipping itself is the problem rather than the turns. Two
#: people interrupting each other looks like this and so does a diariser
#: that has lost the thread; only more listening tells them apart.
DEFAULT_SHORT_TURN_RUN = 3

#: How close to a speaker boundary a disputed word has to be before the
#: boundary becomes part of the dispute. Half a second either side covers a
#: word that straddles the join.
DEFAULT_BOUNDARY_SECONDS = 0.5


class DiarisationConcern(str, Enum):
    """Why the speaker attribution of a passage is worth a second opinion."""

    UNEXPECTED_SPEAKERS = "unexpected_speakers"
    """More people were found than the recording was said to contain."""

    FREQUENT_SPEAKER_CHANGES = "frequent_speaker_changes"
    """The identity changes far more often than a conversation would."""

    VERY_SHORT_UTTERANCE = "very_short_utterance"
    """There is barely enough voice here to identify anybody by."""

    DISPUTE_NEAR_BOUNDARY = "dispute_near_boundary"
    """A disputed word sits where the speaker changes."""

    OVERLAPPING_SPEECH = "overlapping_speech"
    """Two people are speaking at once."""

    @property
    def display_name(self) -> str:
        return _CONCERN_DISPLAY_NAMES[self]

    @property
    def escalation_reason(self) -> EscalationReason:
        """How this concern is named when it becomes an escalation window."""
        if self is DiarisationConcern.OVERLAPPING_SPEECH:
            return EscalationReason.OVERLAPPING_SPEECH
        return EscalationReason.SPEAKER_BOUNDARY_UNCERTAIN


_CONCERN_DISPLAY_NAMES: dict[DiarisationConcern, str] = {
    DiarisationConcern.UNEXPECTED_SPEAKERS: "More speakers than the recording expects",
    DiarisationConcern.FREQUENT_SPEAKER_CHANGES: "The speaker changes implausibly often",
    DiarisationConcern.VERY_SHORT_UTTERANCE: "The utterance is too short to be sure of",
    DiarisationConcern.DISPUTE_NEAR_BOUNDARY: "A disputed word sits on a speaker boundary",
    DiarisationConcern.OVERLAPPING_SPEECH: "People are speaking over each other",
}


@dataclass(frozen=True)
class SpeakerTurn:
    """One unbroken stretch of one person speaking.

    Turns are what identity is reasoned about, rather than words. A label on
    a single word says almost nothing; the same label across nine seconds is
    something two services can be compared on.
    """

    speaker: str
    span: AudioSpan
    token_ids: tuple[str, ...] = ()

    @property
    def duration(self) -> float:
        return self.span.duration

    def overlap_with(self, other: "SpeakerTurn") -> float:
        """How many seconds of audio these two turns both cover."""
        return max(0.0, min(self.span.end, other.span.end) - max(self.span.start, other.span.start))


@dataclass(frozen=True)
class DiarisationOptions:
    """What a person can change about how speakers are judged."""

    expected_speaker_count: int = 1
    minimum_turn_seconds: float = DEFAULT_MINIMUM_TURN_SECONDS
    short_turn_run: int = DEFAULT_SHORT_TURN_RUN
    boundary_seconds: float = DEFAULT_BOUNDARY_SECONDS

    @classmethod
    def from_configuration(cls, configuration: Any = None) -> "DiarisationOptions":
        """Take the expected speaker count off the recording's configuration.

        Read by name rather than by type, so this module stays testable on
        its own and does not depend on the shape of the settings.
        """
        expected = getattr(configuration, "expected_speaker_count", None)
        if not isinstance(expected, int) or isinstance(expected, bool) or expected < 1:
            expected = cls.expected_speaker_count
        return cls(expected_speaker_count=expected)


@dataclass(frozen=True)
class SpeakerConcern:
    """One passage whose speaker attribution deserves a second opinion."""

    concern: DiarisationConcern
    span: AudioSpan
    speakers: tuple[str, ...] = ()
    token_ids: tuple[str, ...] = ()

    @property
    def description(self) -> str:
        people = ", ".join(self.speakers) or "nobody in particular"
        return (
            f"{self.span.start:.1f}s to {self.span.end:.1f}s: "
            f"{self.concern.display_name} ({people})."
        )

    def as_dispute(self, language: Language = Language.UNKNOWN) -> Dispute:
        """Turn this concern into something escalation can group and send.

        Speaker concerns and text disputes go into the same windows on
        purpose. A sentence where the services disagree about a word and
        about who said it is one passage of audio, and asking about it twice
        would cost twice as much and give each question less to work with.
        """
        return Dispute(
            span=self.span,
            reasons=(self.concern.escalation_reason,),
            token_ids=self.token_ids,
            language=language,
        )


@dataclass(frozen=True)
class SpeakerMapping:
    """Which of one service's speaker labels is which of another's.

    ``coverage`` is the share of each matched label's speech that really fell
    on its partner, between nothing and one. It is what separates a confident
    match from a coincidence: two labels that share nine seconds out of ten
    are the same person, and two that share half a second are not.
    """

    mapping: dict[str, str]
    coverage: dict[str, float]
    unmatched: tuple[str, ...] = ()
    """Labels with nobody to match, which is a speaker one service alone heard."""

    def label_for(self, speaker: str | None) -> str | None:
        if speaker is None:
            return None
        return self.mapping.get(speaker)

    def confidence_for(self, speaker: str | None, threshold: float = 0.6) -> Confidence:
        """How much a mapped label may be trusted, as a category.

        A label whose speech mostly fell on one partner is a firm match. One
        that was spread across several is a guess, and the words that depend
        on it belong in front of a person.
        """
        if speaker is None or speaker not in self.mapping:
            return Confidence.UNRESOLVED
        return (
            Confidence.HIGH
            if self.coverage.get(speaker, 0.0) >= threshold
            else Confidence.REVIEW_SUGGESTED
        )


@dataclass(frozen=True)
class SpeakerDecision:
    """What to do about one word's speaker, and on whose authority."""

    token_id: str
    speaker: str | None
    source: Provider | None
    confidence: Confidence
    changed: bool
    review_reason: ReviewReason | None = None


# -- Reading the speakers out of what a service returned -----------------


def speaker_turns(items: Iterable[Any]) -> list[SpeakerTurn]:
    """Gather consecutive words by the same speaker into turns.

    It reads ``speaker``, ``start`` and ``end``, which both the provider
    evidence and the finished words carry, so one function serves the
    backbone's answer, the second opinion's answer and the transcript itself.
    Words with no time behind them are skipped: a turn is a stretch of audio,
    and a word that is not anywhere cannot be part of one.
    """
    turns: list[SpeakerTurn] = []
    speaker: str | None = None
    start: float | None = None
    end: float | None = None
    ids: list[str] = []

    for item in items:
        label = getattr(item, "speaker", None)
        first, last = getattr(item, "start", None), getattr(item, "end", None)
        if label is None or first is None or last is None:
            continue
        if label != speaker and speaker is not None and start is not None and end is not None:
            turns.append(SpeakerTurn(speaker, AudioSpan(start, end), tuple(ids)))
            start, end, ids = None, None, []
        speaker = label
        start = first if start is None else min(start, first)
        end = last if end is None or last > end else end
        identifier = getattr(item, "id", None)
        if isinstance(identifier, str):
            ids.append(identifier)

    if speaker is not None and start is not None and end is not None:
        turns.append(SpeakerTurn(speaker, AudioSpan(start, end), tuple(ids)))
    return turns


def speakers_to_expect(expected_count: int, observed_labels: Sequence[str] = ()) -> int:
    """How many speakers to allow a service to find.

    The user's expectation is the anchor, because a service left free to
    invent speakers will do exactly that, and every extra one it invents
    splits a real person's words in two. The number is raised only where the
    backbone itself heard more people than the user expected, and then only
    as far as the backbone went, so the space is always bounded by somebody's
    evidence rather than by a service's imagination.
    """
    observed = len({label for label in observed_labels if label})
    return max(1, expected_count, observed)


# -- Deciding when to ask again ------------------------------------------


def find_speaker_concerns(
    turns: Sequence[SpeakerTurn],
    disputed_spans: Sequence[AudioSpan] = (),
    options: DiarisationOptions | None = None,
) -> list[SpeakerConcern]:
    """Find the passages where the speaker attribution cannot be trusted.

    The five tests are the ones the specification names, and each of them
    describes a situation where more listening is the only thing that helps.
    Everything else is left alone: a long, clean turn by a voice the service
    has heard for ten minutes is not made better by a second opinion, and
    asking for one costs money and time for nothing.
    """
    options = options or DiarisationOptions()
    concerns: list[SpeakerConcern] = []
    concerns.extend(_unexpected_speakers(turns, options))
    concerns.extend(_overlapping_speech(turns))
    concerns.extend(_short_turns(turns, options))
    concerns.extend(_disputes_on_boundaries(turns, disputed_spans, options))
    return sorted(concerns, key=lambda concern: (concern.span.start, concern.span.end))


def _unexpected_speakers(
    turns: Sequence[SpeakerTurn], options: DiarisationOptions
) -> list[SpeakerConcern]:
    """Flag the people the recording was not supposed to contain.

    The speakers are ranked by how much they say, and the ones past the
    expected count are the surplus. Ranking by speaking time rather than by
    first appearance matters: the extra speaker a diariser invents is almost
    always the one with the least speech, and flagging the person who talks
    most because they happened to speak last would be exactly backwards.
    """
    totals: dict[str, float] = {}
    for turn in turns:
        totals[turn.speaker] = totals.get(turn.speaker, 0.0) + turn.duration
    if len(totals) <= max(1, options.expected_speaker_count):
        return []
    ranked = sorted(totals, key=lambda speaker: totals[speaker], reverse=True)
    surplus = set(ranked[max(1, options.expected_speaker_count):])
    return [
        SpeakerConcern(
            concern=DiarisationConcern.UNEXPECTED_SPEAKERS,
            span=turn.span,
            speakers=(turn.speaker,),
            token_ids=turn.token_ids,
        )
        for turn in turns
        if turn.speaker in surplus
    ]


def _overlapping_speech(turns: Sequence[SpeakerTurn]) -> list[SpeakerConcern]:
    """Find where two people are talking at the same time.

    Two turns by different people whose spans overlap in time is the only
    direct evidence of overlap any of these services gives us, and overlap is
    where diarisation is least reliable and where a second opinion is worth
    most.
    """
    concerns: list[SpeakerConcern] = []
    for earlier, later in zip(turns, turns[1:]):
        if earlier.speaker == later.speaker or not earlier.span.overlaps(later.span):
            continue
        concerns.append(
            SpeakerConcern(
                concern=DiarisationConcern.OVERLAPPING_SPEECH,
                span=AudioSpan(later.span.start, min(earlier.span.end, later.span.end)),
                speakers=(earlier.speaker, later.speaker),
                token_ids=earlier.token_ids + later.token_ids,
            )
        )
    return concerns


def _short_turns(
    turns: Sequence[SpeakerTurn], options: DiarisationOptions
) -> list[SpeakerConcern]:
    """Flag utterances too short to identify, and runs of them.

    A single short turn is a "yes" that might belong to either person. A run
    of them is something else: the identity is flipping faster than people
    take turns, which usually means the diariser has lost the thread rather
    than that the conversation really moved that quickly. The two are told
    apart because the second is one question about a passage, not four
    questions about four words.
    """
    concerns: list[SpeakerConcern] = []
    run: list[SpeakerTurn] = []

    def close(group: list[SpeakerTurn]) -> None:
        if not group:
            return
        if len(group) >= max(2, options.short_turn_run):
            concerns.append(
                SpeakerConcern(
                    concern=DiarisationConcern.FREQUENT_SPEAKER_CHANGES,
                    span=AudioSpan(group[0].span.start, group[-1].span.end),
                    speakers=tuple(dict.fromkeys(turn.speaker for turn in group)),
                    token_ids=tuple(
                        token_id for turn in group for token_id in turn.token_ids
                    ),
                )
            )
            return
        concerns.extend(
            SpeakerConcern(
                concern=DiarisationConcern.VERY_SHORT_UTTERANCE,
                span=turn.span,
                speakers=(turn.speaker,),
                token_ids=turn.token_ids,
            )
            for turn in group
        )

    for turn in turns:
        if turn.duration < options.minimum_turn_seconds:
            run.append(turn)
            continue
        close(run)
        run = []
    close(run)
    return concerns


def _disputes_on_boundaries(
    turns: Sequence[SpeakerTurn],
    disputed_spans: Sequence[AudioSpan],
    options: DiarisationOptions,
) -> list[SpeakerConcern]:
    """Flag a disputed word that sits where the speaker changes.

    These two doubts feed each other. A word nobody agrees on, at the moment
    the speaker changes, is often a word that has been given to the wrong
    person and misheard because of it, and the same few seconds of audio
    answers both questions at once.
    """
    if not disputed_spans:
        return []
    margin = max(0.0, options.boundary_seconds)
    concerns: list[SpeakerConcern] = []
    for earlier, later in zip(turns, turns[1:]):
        if earlier.speaker == later.speaker:
            continue
        boundary = later.span.start
        for span in disputed_spans:
            if span.start - margin > boundary or span.end + margin < boundary:
                continue
            concerns.append(
                SpeakerConcern(
                    concern=DiarisationConcern.DISPUTE_NEAR_BOUNDARY,
                    span=AudioSpan(min(span.start, boundary), max(span.end, boundary)),
                    speakers=(earlier.speaker, later.speaker),
                )
            )
    return concerns


# -- Understanding the answer --------------------------------------------


def map_speaker_labels(
    backbone: Sequence[SpeakerTurn], other: Sequence[SpeakerTurn]
) -> SpeakerMapping:
    """Work out which of the second service's speakers is which of the first's.

    Solved by audio coverage, and deliberately not by anything else. Two
    services number their speakers from their own first sentence, so the
    orders agree only by luck, and the moment one of them misses somebody's
    opening line the orders disagree for the rest of the recording. Matching
    by position would then hand every word to the wrong person while looking
    entirely reasonable.

    So each foreign label is measured against each home label by how many
    seconds of audio they both cover, and the strongest pairs are taken
    first, one for one. A foreign label with nothing to match is not forced
    onto anybody: it is reported as unmatched, which is the honest reading of
    a service that heard somebody the backbone did not.
    """
    overlaps: dict[tuple[str, str], float] = {}
    totals: dict[str, float] = {}
    for foreign in other:
        totals[foreign.speaker] = totals.get(foreign.speaker, 0.0) + foreign.duration
        for home in backbone:
            shared = foreign.overlap_with(home)
            if shared <= 0.0:
                continue
            key = (foreign.speaker, home.speaker)
            overlaps[key] = overlaps.get(key, 0.0) + shared

    mapping: dict[str, str] = {}
    coverage: dict[str, float] = {}
    taken: set[str] = set()
    # Strongest pair first, so that a label with one clear partner is settled
    # before a label that overlaps a little of everybody can take that
    # partner away from it.
    for (foreign, home), shared in sorted(
        overlaps.items(), key=lambda item: item[1], reverse=True
    ):
        if foreign in mapping or home in taken:
            continue
        mapping[foreign] = home
        taken.add(home)
        total = totals.get(foreign, 0.0)
        coverage[foreign] = shared / total if total > 0 else 0.0

    unmatched = tuple(label for label in totals if label not in mapping)
    if unmatched:
        _log.info(
            "The second opinion heard %d speaker or speakers the backbone did not: %s.",
            len(unmatched),
            ", ".join(unmatched),
        )
    return SpeakerMapping(mapping=mapping, coverage=coverage, unmatched=unmatched)


def turns_from_result(result: ProviderResult) -> list[SpeakerTurn]:
    """The speaker turns in what one service returned, spoken words only."""
    return speaker_turns(result.word_tokens)


def reconcile_speakers(
    tokens: Sequence[FinalToken],
    other: Sequence[SpeakerTurn],
    mapping: SpeakerMapping,
    source: Provider = Provider.ASSEMBLYAI,
) -> list[SpeakerDecision]:
    """Decide each word's speaker from the two services, and nothing else.

    Every decision here is about the speaker fields alone. Where the second
    opinion agrees with the backbone the word is confirmed, which is worth
    saying because agreement between two services that listened separately is
    the best evidence of identity this application ever gets. Where it
    disagrees, the word takes the second opinion's answer and goes in front
    of a person, because two services that heard different people cannot both
    be right and neither of them can be checked from here.

    Where the second opinion heard somebody the backbone never had, the words
    keep the speaker they have and are flagged. Renaming them to a label that
    exists in only one service's world would put an identifier in the
    transcript that nothing else can explain.
    """
    decisions: list[SpeakerDecision] = []
    for token in tokens:
        span = token.audible_span
        if span is None:
            continue
        turn = _turn_covering(other, span)
        if turn is None:
            continue
        if turn.speaker in mapping.unmatched:
            decisions.append(
                SpeakerDecision(
                    token_id=token.id,
                    speaker=token.speaker,
                    source=token.speaker_source,
                    confidence=Confidence.REVIEW_REQUIRED,
                    changed=False,
                    review_reason=ReviewReason.SPEAKER_UNCERTAIN,
                )
            )
            continue
        mapped = mapping.label_for(turn.speaker)
        if mapped is None:
            continue
        if mapped == token.speaker:
            decisions.append(
                SpeakerDecision(
                    token_id=token.id,
                    speaker=mapped,
                    source=token.speaker_source,
                    confidence=mapping.confidence_for(turn.speaker),
                    changed=False,
                )
            )
            continue
        decisions.append(
            SpeakerDecision(
                token_id=token.id,
                speaker=mapped,
                source=source,
                confidence=Confidence.REVIEW_SUGGESTED,
                changed=True,
                review_reason=ReviewReason.SPEAKER_UNCERTAIN,
            )
        )
    return decisions


def _turn_covering(turns: Sequence[SpeakerTurn], span: AudioSpan) -> SpeakerTurn | None:
    """The turn that shares the most audio with this word.

    Most words sit inside one turn and any test would find it. A word on a
    boundary sits in two, and the one it shares more of itself with is the
    better answer than whichever happens to be looked at first.
    """
    best: SpeakerTurn | None = None
    best_shared = 0.0
    for turn in turns:
        shared = max(0.0, min(turn.span.end, span.end) - max(turn.span.start, span.start))
        if shared > best_shared:
            best, best_shared = turn, shared
    return best


def apply_speaker(token: FinalToken, decision: SpeakerDecision) -> None:
    """Write a speaker decision onto a word, and touch nothing else.

    This is the rule the whole module exists to keep. The text stays exactly
    as reconciliation settled it, the start and the end stay exactly as they
    were measured, and only the speaker, its source and its confidence
    change. A person told that a word was reattributed must be able to trust
    that the word itself did not move and did not change.
    """
    if decision.speaker is not None:
        token.speaker = decision.speaker
    token.speaker_source = decision.source
    token.speaker_confidence = decision.confidence
    if decision.review_reason is not None:
        token.flag(decision.review_reason)


def apply_speakers(tokens: Sequence[FinalToken], decisions: Sequence[SpeakerDecision]) -> int:
    """Apply a set of decisions to the words they name, and count the changes."""
    by_id = {token.id: token for token in tokens}
    changed = 0
    for decision in decisions:
        token = by_id.get(decision.token_id)
        if token is None:
            continue
        if decision.changed and token.speaker != decision.speaker:
            changed += 1
        apply_speaker(token, decision)
    return changed
