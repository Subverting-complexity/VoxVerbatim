"""Deciding what was actually said, without ever quietly guessing.

Several services have listened to the same recording and written down what
they heard. Alignment has laid their words side by side so that each column
is one place in the audio. This module is what turns that table into a
transcript: for every word, which spelling to print, whose timing to keep,
who was speaking, how far any of it can be trusted, and whether a person has
to look at it.

The order of work comes from section 12 of the specification and is followed
as written, because each step exists to keep the step after it from being
asked a question it would answer badly.

**Exact agreement first.** Where every service wrote the same thing, there is
nothing to decide and the word is accepted with high confidence.

**Then formatting.** "twenty five" and "25", "Jurgen" and "Jürgen", "data
base" and "database" are one piece of spoken evidence written three ways.
Treating them as disagreements would fill the review queue with differences
that are not differences, and the real disagreements would be lost in the
noise. So they are one candidate with several display forms, and choosing
between the forms is a question about spelling rather than about hearing.
Because nothing acoustic changed, the span the timing service measured still
fits, and the word inherits it.

**Then the vocabulary.** A candidate that matches a name the user wrote down,
or a correction they have actually made, has outside evidence behind it that
no service can offer. A candidate that matches a known *misrecognition* has
the opposite: the user has already told us that services produce this
spelling and that it is wrong. That is evidence against the candidate, and
treating a known mistake as merely another opinion would throw away the most
useful thing the user has told us.

**Then scoring, and never counting heads.** The specification says twice not
to implement simple majority voting, and this module goes out of its way to
make that impossible rather than merely not doing it. Each service's opinion
is worth the product of six factors: how far that service is believed, how
far it is believed about this language, how sure it was of what it heard,
what the vocabulary says about the spelling, how confident the alignment is
that it is talking about this word at all, and how it has actually performed
on the user's own corrections. Several services supporting one reading are
then combined the way independent evidence combines, not by addition: each
adds what is left of the doubt rather than a share of the total, and each
one after the first counts for less, because services trained on overlapping
data are not independent witnesses. The arithmetic has the property the
specification is asking for: agreement adds evidence but can never exceed
certainty, so two weak opinions cannot outvote one strong, well-evidenced
one.

**Then, where the evidence is genuinely split, refuse.** A word whose leading
candidate is barely ahead of the next is left unresolved with both readings
preserved. A high-risk value where the candidates differ is left unresolved
whatever the scores say, because "15,000" against "50,000" is exactly the
case where the winner reads perfectly well and is out by a factor of three
in somebody's contract. The exporters write ``[UNCERTAIN: 15,000 / 50,000]``
and the person decides.

Two structural points are worth stating because they are easy to get wrong.

Text, timing and speaker are decided separately, each with its own source
and its own confidence, and the timing decision is a function of its own so
that the team working on section 19 can build on it without touching
anything else here. Timing is inherited where nothing acoustic changed, and
where several final words come out of one measured span they share it and no
sub-word boundary is invented, because a boundary nobody measured is a
number the rest of the application would believe.

And provider evidence is never destroyed. A reading that lost is kept as a
candidate; a word that was dropped is kept as a rejected token. That is what
lets a person be shown why a word says what it says.

Nothing here depends on Qt or on any provider library, so it can be tested
on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Container, Iterable, Mapping, Sequence

from vox_verbatim.transcription.alignment import (
    AlignedRow,
    AlignedTable,
    AlignmentColumn,
)
from vox_verbatim.transcription.confidence import (
    ASSUMED_ACOUSTIC_CONFIDENCE,
    HIGH_CONFIDENCE_THRESHOLD,
    ConfidenceAssessment,
    TextSignals,
    assess_text,
    best_acoustic_confidence,
    speaker_confidence,
    timing_confidence,
)
from vox_verbatim.transcription.language import (
    LanguageReading,
    provider_language_weight,
    read_languages,
)
from vox_verbatim.transcription.model import (
    AlignmentStatus,
    AudioSpan,
    Candidate,
    Confidence,
    FinalToken,
    Language,
    LanguageEvidence,
    Provider,
    ProviderResult,
    ProviderToken,
    RecordingConfiguration,
    ReviewReason,
    RiskCategory,
    TimingStatus,
    TokenReference,
)
from vox_verbatim.transcription.normalise import (
    EquivalenceKind,
    are_equivalent,
    equivalence_kind,
    is_filler,
    normalise,
    spelled_as,
)
from vox_verbatim.transcription.risk import RISK_WINDOW, numbers_disagree, risk_at
from vox_verbatim.transcription.vocabulary import TermCategory, VocabularyIndex

#: How far each service is believed before anything about the word is known.
#:
#: These are the version 1 defaults and every one of them is an explicit rule
#: rather than a measurement, which is what section 24 asks for: start with
#: rules, collect real corrections, then calibrate. They are passed in rather
#: than read from here by anything that matters, so the calibration work can
#: replace them without touching this module.
#:
#: OpenAI and ElevenLabs are both named "very strong" in section 33 and sit
#: at the top. OpenAI is placed a little above ElevenLabs on text alone,
#: because ElevenLabs is carrying the timing and the diarisation as well and
#: its text is a verbatim stream rather than a considered reading; that is a
#: judgement about their different jobs, not a claim that one hears better.
#: Being the backbone earns ElevenLabs nothing here, deliberately: section 33
#: says not to designate any one service as the text authority, and a bonus
#: for holding the timeline would do exactly that by the back door.
#:
#: Microsoft is "very strong for English and German" but is a default
#: *additional* opinion rather than something the transcript depends on, so
#: it sits just below. Its Afrikaans exclusion is not expressed here; it is a
#: property of a span rather than of the service, and it lives in
#: :mod:`vox_verbatim.transcription.language` where the span evidence is.
#:
#: AssemblyAI is escalation-only in version 1. When it speaks at all it was
#: called precisely because the case was hard, so it is trusted, but it has
#: not earned a place above the services that heard the whole recording.
#: Deepgram is an optional challenger that version 1 does not require, and
#: nothing has yet shown it earns more than the lowest weight that still lets
#: it win when it is clearly right.
DEFAULT_PROVIDER_RELIABILITY: dict[Provider, float] = {
    Provider.OPENAI: 1.00,
    Provider.ELEVENLABS: 0.95,
    Provider.MICROSOFT: 0.90,
    Provider.ASSEMBLYAI: 0.85,
    Provider.DEEPGRAM: 0.75,
}


@dataclass(frozen=True)
class ReliabilityWeights:
    """How far each service is believed, as a table that can be replaced."""

    by_provider: Mapping[Provider, float] = field(
        default_factory=lambda: dict(DEFAULT_PROVIDER_RELIABILITY)
    )

    unknown_provider: float = 0.70
    """What a service not in the table is worth.

    Below every named default, because a service nobody has written a weight
    for is a service nobody has thought about yet.
    """

    def for_provider(self, provider: Provider) -> float:
        return self.by_provider.get(provider, self.unknown_provider)


DEFAULT_RELIABILITY = ReliabilityWeights()


@dataclass(frozen=True)
class WordFacts:
    """The few things about one word that a performance record can use.

    This is the seam for the calibration work. It is a small flat record
    rather than the word itself, so that this module does not have to know
    what the statistics are keyed on and the statistics do not have to know
    what a reconciled word looks like.
    """

    provider: Provider
    language: Language = Language.UNKNOWN
    speaker: str | None = None
    is_proper_noun: bool = False
    is_numeric: bool = False


#: A function that says how far one service has actually earned being
#: believed about a word like this one. Version 1 supplies none, and every
#: service is then treated as having performed as expected.
HistoricalWeight = Callable[[WordFacts], float]

#: A function that measures a span against the audio again after the text
#: changed. It answers ``None`` when it cannot justify an answer, which is a
#: result rather than a failure and is reported as one.
ForcedAlignment = Callable[[str, AudioSpan], AudioSpan | None]


@dataclass(frozen=True)
class ReconciliationOptions:
    """The thresholds the rules turn on, gathered so they can be tuned."""

    minimum_decision_margin: float = 0.12
    """How far ahead the leading candidate must be to settle the word.

    Two readings within this of each other are a genuine dispute however
    high both scores are, and the honest answer is that we do not know.
    """

    agreement_independence: float = 0.60
    """How much each additional service agreeing is worth, cumulatively.

    Services are not independent witnesses: they are trained on overlapping
    data and make correlated mistakes. Counting the second and third
    supporter at full value would be counting one piece of evidence three
    times, which is majority voting wearing a probability's clothes.
    """

    maximum_factor: float = 0.99
    """The ceiling on one service's contribution, so nothing is certain."""

    vocabulary_exact: float = 1.50
    """What it is worth to be the user's term, spelled exactly as they wrote it."""

    vocabulary_equivalent: float = 1.20
    """What it is worth to be the user's term in a different spelling.

    Less, because it argues that the word is right and says nothing about
    which of the rival spellings of it to print.
    """

    vocabulary_misrecognition: float = 0.35
    """What it costs to be a mistake the user has already told us about."""

    insertion_keep_threshold: float = 0.50
    """How strong a word no other service heard must be to enter the transcript.

    Set so that one trusted service, reasonably sure of what it heard, is
    enough to have the word kept and flagged, while a weak service that was
    itself unsure is not. Neither outcome is a silent one: a kept word is
    marked as having no acoustic evidence, and a dropped one stays in
    provenance where a service that keeps inventing words can be seen doing
    it.
    """

    hallucination_deletion_count: int = 2
    """How many services must have failed to hear a backbone word before it
    is treated as something the backbone imagined rather than something the
    others missed."""

    hallucination_acoustic_ceiling: float = 0.50
    """And how unsure the backbone must have been of it as well.

    Both conditions have to hold. Dropping a word the backbone was sure of,
    merely because the other services were silent, would let two services
    that stopped early delete real speech.
    """

    maximum_merge_span: int = 3
    """How many backbone words one decision may cover, matching alignment."""

    maximum_number_merge_span: int = 10
    """How many backbone words one decision may cover where they are a
    spoken number, matching alignment. "one hundred and fifty thousand" is
    six words that one service may have written as "150,000"."""

    absence_reach: int = 20
    """How far, in backbone words, a service may be silent before it is
    treated as absent rather than as disagreeing.

    A service that returned only part of the recording, because a chunk
    failed or a request timed out after partial output, is not in the
    table's list of missing services: it did say something. But it said
    nothing at all about the rest of the recording, and counting every word
    after the point where it stopped as a word it failed to hear would let
    two such services, with a little doubt from the backbone, delete real
    speech. So a service with no word within this many positions either
    side is treated exactly like one that never answered: its silence here
    is not evidence about this word.
    """

    alignment_floor: float = 0.60
    """What a service is still worth when the alignment lined it up poorly.

    The column quality is not a probability that a service is wrong. It says
    how confident the matching is that the service was talking about this
    word at all, which is a real reason to discount its opinion and not a
    reason to discard it. Used raw it would be close to a veto: an ordinary
    substitution scores about a half by construction, so every service that
    disagreed with the backbone would lose half its weight for the crime of
    disagreeing, and the backbone would become the text authority by
    arithmetic that nobody chose. The floor keeps the effect proportionate.
    """


DEFAULT_OPTIONS = ReconciliationOptions()


# -- What the timing rules answer ----------------------------------------


@dataclass(frozen=True)
class TimingDecision:
    """When a word was said, how that was arrived at, and how far to trust it."""

    start: float | None = None
    end: float | None = None
    status: TimingStatus = TimingStatus.UNALIGNED
    source: Provider | None = None
    confidence: Confidence = Confidence.UNRESOLVED
    source_audio_span: AudioSpan | None = None
    reasons: tuple[ReviewReason, ...] = ()


def decide_timing(
    backbone_tokens: Sequence[ProviderToken],
    equivalence: EquivalenceKind,
    shared: bool = False,
    quality: float = 1.0,
    containing_span: AudioSpan | None = None,
    forced_alignment: ForcedAlignment | None = None,
    text: str = "",
) -> TimingDecision:
    """Decide the span of one final word from the span its evidence sat in.

    This is section 19 in one place, and it is a function of its own so that
    the team refining the timing rules can build on it, replace it, or call
    it from a forced-alignment pass without disturbing anything else.

    The principle is that a span may only be inherited when the sounds
    underneath it did not change. A difference in how a word was written
    leaves the same sounds in the same place, so the measured span still
    fits exactly and the word keeps it. A genuinely different word in the
    same place is a defensible span rather than a measured one, and is
    marked as such so that nothing downstream navigates to it as though it
    were exact.

    Where several final words come out of one measured span they share it,
    and no boundary is invented between them. That is the rule the
    specification is most insistent about, and the reason is that an
    invented boundary is indistinguishable from a measured one once it has
    been written down: the review window would play the wrong audio, and
    nobody would be able to tell why.
    """
    timed = [token for token in backbone_tokens if token.has_timing]
    if not timed:
        return TimingDecision(
            status=TimingStatus.UNALIGNED,
            confidence=timing_confidence(TimingStatus.UNALIGNED, quality),
            source_audio_span=containing_span,
            reasons=(ReviewReason.UNALIGNED_WORD,),
        )

    start = min(token.start for token in timed if token.start is not None)
    end = max(token.end for token in timed if token.end is not None)
    source = timed[0].provider
    span = AudioSpan(start, end)
    reasons: list[ReviewReason] = []

    if shared:
        status = TimingStatus.SHARED_PHRASE_SPAN
    elif equivalence is EquivalenceKind.IDENTICAL:
        status = TimingStatus.EXACT_PROVIDER_TIME
    elif equivalence.is_equivalent:
        status = TimingStatus.MAPPED_FORMAT_EQUIVALENT
    else:
        status = TimingStatus.MAPPED_SUBSTITUTION
        if forced_alignment is not None:
            measured = forced_alignment(text, span)
            if measured is not None:
                start, end = measured.start, measured.end
                span = measured
                status = TimingStatus.FORCED_ALIGNED
            else:
                # A refusal is an answer. The word keeps the best defensible
                # span it has, and the person is told that the span was not
                # measured again rather than being left to assume it was.
                reasons.append(ReviewReason.FORCED_ALIGNMENT_FAILED)

    return TimingDecision(
        start=start,
        end=end,
        status=status,
        source=source,
        confidence=timing_confidence(status, quality),
        source_audio_span=span,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class SpeakerDecision:
    """Who said a word, on whose authority, and how far that can be trusted."""

    speaker: str | None = None
    source: Provider | None = None
    confidence: Confidence = Confidence.UNRESOLVED
    reasons: tuple[ReviewReason, ...] = ()


def decide_speaker(
    position: int,
    backbone_token: ProviderToken | None,
    backbone_boundaries: Container[int],
    other_boundaries: Mapping[Provider, Container[int]],
    overlapping: bool = False,
    diarised: bool = True,
) -> SpeakerDecision:
    """Decide who was speaking, using the backbone as the initial authority.

    The one thing this must not do is compare the services' speaker labels
    with each other. ElevenLabs calls somebody "speaker_0" and AssemblyAI
    calls somebody "A", and nothing whatever says those are the same person;
    an application that compared them would report a disagreement on every
    word of every recording with more than one speaker.

    What can honestly be compared is *where* each service puts a change of
    speaker. A service that hears a new voice starting at this word, where
    the authority hears the same voice continuing, is disagreeing about
    something real, and that is what raises the doubt here. Section 14 then
    sends the disputed region to AssemblyAI with a padded window, and a
    speaker correction changes only the speaker.

    Where no service reported a speaker anywhere in the recording, the
    question was never asked and there is nothing to be uncertain about. A
    word then carries no speaker and high confidence in that, rather than
    dragging every word of an undiarised transcript into the review queue
    over a dimension the transcript does not have.
    """
    if not diarised:
        return SpeakerDecision(confidence=Confidence.HIGH)
    if backbone_token is None or backbone_token.speaker is None:
        return SpeakerDecision(
            confidence=Confidence.UNRESOLVED,
            reasons=(ReviewReason.SPEAKER_UNCERTAIN,),
        )

    here = position in backbone_boundaries
    disputed = any(
        (position in boundaries) != here for boundaries in other_boundaries.values()
    )
    reasons: list[ReviewReason] = []
    if overlapping:
        reasons.append(ReviewReason.OVERLAPPING_SPEECH)
    if disputed:
        reasons.append(ReviewReason.SPEAKER_UNCERTAIN)
    return SpeakerDecision(
        speaker=backbone_token.speaker,
        source=backbone_token.provider,
        confidence=speaker_confidence(True, disputed, overlapping),
        reasons=tuple(reasons),
    )


# -- One service's reading of one place in the recording -----------------


@dataclass(frozen=True)
class _Reading:
    """What one service put at one place, with everything needed to score it."""

    provider: Provider
    text: str
    tokens: tuple[ProviderToken, ...]
    positions: tuple[int, ...]
    quality: float
    status: AlignmentStatus
    equivalence: EquivalenceKind
    is_backbone: bool = False
    """The backbone's own reading, which was not aligned against anything.

    Its quality is one by definition rather than by measurement, so it is
    marked and left out wherever an alignment quality is being judged.
    """


@dataclass
class _Group:
    """Readings that are the same spoken evidence, however they are written."""

    key: str
    readings: list[_Reading] = field(default_factory=list)
    score: float = 0.0
    raw_total: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    display: str = ""
    in_vocabulary: bool = False

    @property
    def providers(self) -> tuple[Provider, ...]:
        seen: list[Provider] = []
        for reading in self.readings:
            if reading.provider not in seen:
                seen.append(reading.provider)
        return tuple(seen)

    @property
    def texts(self) -> tuple[str, ...]:
        found: list[str] = []
        for reading in self.readings:
            if reading.text not in found:
                found.append(reading.text)
        return tuple(found)

    @property
    def token_references(self) -> tuple[TokenReference, ...]:
        return tuple(
            TokenReference(token.provider, token.index)
            for reading in self.readings
            for token in reading.tokens
        )


# -- The whole job -------------------------------------------------------


def reconcile(
    table: AlignedTable,
    configuration: RecordingConfiguration | None = None,
    vocabulary: VocabularyIndex | None = None,
    languages: LanguageReading | None = None,
    results: Iterable[ProviderResult] = (),
    weights: ReliabilityWeights = DEFAULT_RELIABILITY,
    historical: HistoricalWeight | None = None,
    options: ReconciliationOptions = DEFAULT_OPTIONS,
    forced_alignment: ForcedAlignment | None = None,
    escalated: Container[int] = (),
    adjudications: Mapping[int, str | None] | None = None,
) -> list[FinalToken]:
    """Turn an aligned table into the finished sequence of words.

    ``languages`` may be supplied by a caller that has already worked out
    the language evidence, which is the normal case once escalation has run
    and produced localised readings. Where it is not supplied it is worked
    out here from the whole-file readings in ``results``.

    ``escalated`` is the set of backbone positions that have already been
    sent for a second opinion, so that one still in dispute afterwards can be
    reported as a second opinion that did not settle it rather than as an
    ordinary disagreement. ``adjudications`` is what the language model
    answered, keyed the same way, with ``None`` where it declined to decide.
    """
    configuration = configuration or RecordingConfiguration()
    if languages is None:
        languages = read_languages(table, results, configuration)

    coverage = _coverage(table)
    extents = _extents(coverage)
    backbone_texts = [token.text for token in table.backbone_tokens]
    boundaries = _speaker_boundaries(table, coverage)
    overlaps = _overlapping_positions(table)

    tokens: list[FinalToken] = []
    rejected: list[TokenReference] = []
    consumed: set[int] = set()
    word_rows = {row.position: row for row in table.rows if not row.is_insertion_row}

    for row in table.rows:
        if row.is_insertion_row:
            emitted = _insertion_token(
                row, table, languages, vocabulary, weights, historical, options
            )
            if emitted is None:
                rejected.extend(
                    TokenReference(token.provider, token.index)
                    for column in row.columns
                    for token in column.aligned_tokens
                )
                continue
            _attach_rejected(emitted, rejected)
            tokens.append(emitted)
            continue

        if row.position in consumed:
            continue
        scope = _scope(row.position, coverage, len(table.backbone_tokens), options)
        consumed.update(scope)
        produced = _decide_scope(
            scope=scope,
            table=table,
            coverage=coverage,
            extents=extents,
            backbone_texts=backbone_texts,
            languages=languages,
            vocabulary=vocabulary,
            weights=weights,
            historical=historical,
            options=options,
            forced_alignment=forced_alignment,
            boundaries=boundaries,
            overlaps=overlaps,
            escalated=escalated,
            adjudications=adjudications or {},
        )
        before, after = _audio_events(scope, word_rows)
        tokens.extend(before)
        if not produced:
            rejected.extend(_backbone_references(table, scope))
            tokens.extend(after)
            continue
        _attach_rejected(produced[0], rejected)
        tokens.extend(produced)
        tokens.extend(after)

    if rejected and tokens:
        tokens[-1].rejected_tokens.extend(rejected)
    return tokens


def _audio_events(
    scope: tuple[int, ...], word_rows: Mapping[int, AlignedRow]
) -> tuple[list[FinalToken], list[FinalToken]]:
    """The backbone's audio events in this scope, split by side of its words.

    Alignment keeps audio events such as "(laughter)" out of the words it
    matches and hangs them on a neighbouring word, the way it does with
    punctuation. Nothing downstream reads that attached punctuation, so
    without this the laughter would silently vanish from the transcript.
    It happened in the room, so it is put back where it occurred, settled
    and unflagged: no other service writes events down, and their silence
    about one is not a disagreement.

    An event whose index is below its word's came before that word, which
    only happens before the first word of the recording; every other event
    follows the word it hangs on. Where a scope covers several words (a
    merge), an event between two of them comes out after the whole scope,
    because the merged words are emitted together.

    Only the backbone's events are kept. Another service's events hang on
    its own columns and are dropped, which is accepted because ElevenLabs
    is the one service that reports events and it is the preferred
    backbone.
    """
    before: list[FinalToken] = []
    after: list[FinalToken] = []
    for position in scope:
        row = word_rows.get(position)
        if row is None or row.backbone_token is None:
            continue
        word_index = row.backbone_token.index
        for token in row.punctuation:
            if not token.is_audio_event:
                continue
            target = before if token.index < word_index else after
            target.append(_audio_event_token(token))
    return before, after


def _audio_event_token(token: ProviderToken) -> FinalToken:
    """A transcript entry for an audio event, exactly as the service gave it."""
    final = FinalToken(
        text=token.text,
        normalised_text=normalise(token.text),
        text_source=token.provider,
        text_confidence=Confidence.HIGH,
        speaker=token.speaker,
        speaker_source=token.provider if token.speaker is not None else None,
        language=token.language,
        source_tokens=[TokenReference(token.provider, token.index)],
        alignment_status=AlignmentStatus.UNALIGNED,
    )
    if token.has_timing:
        final.start = token.start
        final.end = token.end
        final.timing_source = token.provider
        final.timing_status = TimingStatus.EXACT_PROVIDER_TIME
        final.timing_confidence = Confidence.HIGH
    return final


def _attach_rejected(token: FinalToken, rejected: list[TokenReference]) -> None:
    """Hang everything thrown away since the last word onto this one.

    Rejected evidence has to live somewhere it can be found again. Hanging it
    on the next word that survived keeps it in the right part of the
    recording without letting it into the transcript, which is what section
    19 asks for when a hallucinated word is removed.
    """
    if not rejected:
        return
    token.rejected_tokens.extend(rejected)
    rejected.clear()


def _backbone_references(table: AlignedTable, scope: tuple[int, ...]) -> list[TokenReference]:
    tokens = [
        table.backbone_tokens[position]
        for position in scope
        if 0 <= position < len(table.backbone_tokens)
    ]
    return [TokenReference(token.provider, token.index) for token in tokens]


def _coverage(table: AlignedTable) -> dict[Provider, dict[int, AlignmentColumn]]:
    """Which column of each service covers each backbone word.

    A merge covers several backbone words with one column and appears in the
    table only once, under the first of them. Without this map, every word
    after the first of a merge would look as though the service had said
    nothing there, and a service that wrote "database" would be recorded as
    having failed to hear the second half of "data base".
    """
    found: dict[Provider, dict[int, AlignmentColumn]] = {}
    for alignment in table.alignments:
        per_position = found.setdefault(alignment.provider, {})
        for column in alignment.columns:
            if column.is_insertion:
                continue
            for position in column.backbone_indices:
                per_position[position] = column
    return found


def _extents(
    coverage: Mapping[Provider, Mapping[int, AlignmentColumn]],
) -> dict[Provider, tuple[int, int]]:
    """The first and last backbone word each service put anything against.

    A service that stopped half way through the recording has an extent
    that ends there, and every word after it lies outside. Services that
    put nothing anywhere are left out of the result.
    """
    found: dict[Provider, tuple[int, int]] = {}
    for provider, per_position in coverage.items():
        covered = [
            position for position, column in per_position.items() if column.aligned_tokens
        ]
        if covered:
            found[provider] = (min(covered), max(covered))
    return found


def _scope(
    position: int,
    coverage: Mapping[Provider, Mapping[int, AlignmentColumn]],
    count: int,
    options: ReconciliationOptions,
) -> tuple[int, ...]:
    """The backbone words that have to be decided together.

    Normally one. Where a service merged several backbone words into one,
    all of them are decided at once, because "database" can only be compared
    with "data base" as a whole. Comparing it with "data" alone would report
    a disagreement between two services that heard exactly the same thing.

    The bound is the wider of the two merge spans. Alignment only merges
    past three words where every word is a number word, so the wider bound
    changes nothing for ordinary words, and it keeps a long number in one
    scope where two services merged overlapping parts of it.
    """
    limit = max(options.maximum_merge_span, options.maximum_number_merge_span)
    last = position
    changed = True
    while changed and last - position + 1 < limit:
        changed = False
        for per_position in coverage.values():
            for index in range(position, last + 1):
                column = per_position.get(index)
                if column is None:
                    continue
                reach = max(column.backbone_indices, default=index)
                if reach > last and reach < count:
                    last = reach
                    changed = True
    return tuple(range(position, min(last, count - 1) + 1))


# -- Deciding one place --------------------------------------------------


def _decide_scope(
    scope: tuple[int, ...],
    table: AlignedTable,
    coverage: Mapping[Provider, Mapping[int, AlignmentColumn]],
    extents: Mapping[Provider, tuple[int, int]],
    backbone_texts: Sequence[str],
    languages: LanguageReading,
    vocabulary: VocabularyIndex | None,
    weights: ReliabilityWeights,
    historical: HistoricalWeight | None,
    options: ReconciliationOptions,
    forced_alignment: ForcedAlignment | None,
    boundaries: Mapping[Provider, set[int]],
    overlaps: Container[int],
    escalated: Container[int],
    adjudications: Mapping[int, str | None],
) -> list[FinalToken]:
    """Settle one place in the recording and produce the words it comes to."""
    position = scope[0]
    backbone_tokens = tuple(table.backbone_tokens[index] for index in scope)
    evidence = languages.for_position(position)
    readings = _gather(scope, table, coverage, backbone_tokens)
    deletions = _deletions(scope, coverage, table, extents, options)

    speaker = backbone_tokens[0].speaker if backbone_tokens else None
    groups = _group(readings)
    for group in groups:
        _score_group(
            group,
            evidence=evidence,
            afrikaans_enabled=languages.afrikaans_enabled,
            language=languages.language_at(position),
            speaker=speaker,
            vocabulary=vocabulary,
            weights=weights,
            historical=historical,
            options=options,
        )
    groups.sort(key=lambda group: (group.score, group.raw_total), reverse=True)

    if not groups:
        return []

    winner = groups[0]
    runner_up = groups[1] if len(groups) > 1 else None

    if _is_hallucination(winner, deletions, readings, backbone_tokens, options):
        return []

    risk = _risk_for(scope, backbone_texts, groups)
    counted = _contesting_silence(deletions, winner, runner_up, scope, backbone_texts, risk)
    unresolved, reasons = _settle(
        winner=winner,
        runner_up=runner_up,
        risk=risk,
        position=position,
        scope=scope,
        vocabulary=vocabulary,
        languages=languages,
        escalated=escalated,
        options=options,
    )

    decision = adjudications.get(position, _NOT_ASKED)
    llm_decision: str | None = None
    if decision is not _NOT_ASKED:
        winner, unresolved, llm_decision, adjudged = _apply_adjudication(
            decision, groups, winner, unresolved
        )
        if not adjudged:
            reasons.append(ReviewReason.ADJUDICATION_DECLINED)

    signals = TextSignals(
        support=winner.score,
        runner_up=runner_up.score if runner_up else 0.0,
        acoustic=_group_acoustic(winner),
        acoustic_source=_group_acoustic_source(winner),
        alignment_quality=_group_quality(winner),
        agreeing_providers=len(winner.providers),
        contesting_providers=len(_contesting(groups, winner)) + len(counted),
        language_support=_language_support(winner, evidence, languages),
        in_vocabulary=winner.in_vocabulary,
        is_high_risk=bool(risk),
        unresolved=unresolved,
    )
    assessment = assess_text(signals)
    reasons.extend(assessment.reasons)
    category = assessment.category
    if _everybody_agreed(groups, winner, counted) and not assessment.reasons:
        # Section 12.1, applied as written: where every service that spoke
        # said the same thing once formatting is set aside, the word is
        # accepted with high confidence. Running unanimous agreement back
        # through the strength arithmetic would mark it down for the sole
        # reason that two of the three services do not report a confidence
        # figure, which is a fact about their APIs and not about the audio.
        category = Confidence.HIGH
    if unresolved:
        # A refusal to decide outranks everything above it, the unanimity
        # rule included. This test used to live inside _emit; it is here now
        # so that the category the word actually gets is settled in one
        # place, and so that the strength saved beside it can be worked out
        # from that same answer rather than from an earlier draft of it.
        category = Confidence.UNRESOLVED

    return _emit(
        scope=scope,
        table=table,
        backbone_tokens=backbone_tokens,
        groups=groups,
        winner=winner,
        assessment_category=category,
        assessment_strength=_recorded_strength(assessment, category),
        risk=risk,
        reasons=reasons,
        evidence=evidence,
        language=languages.language_at(position),
        forced_alignment=forced_alignment,
        boundaries=boundaries,
        overlaps=overlaps,
        llm_decision=llm_decision,
    )


def _recorded_strength(assessment: ConfidenceAssessment, category: Confidence) -> float:
    """The strength to save beside the category the word was actually given.

    Nearly always this is simply the strength the assessment worked out. It
    differs in one case: where a rule above has moved the word *up* into
    high confidence, as the unanimous-agreement rule does, saving the raw
    figure would put a number below the high-confidence threshold next to a
    category that says there is nothing to look at here. The two would then
    contradict each other, and worse, they would contradict each other
    invisibly: the review window would leave the word alone while the
    low-confidence sweep pulled the same word up as weak, and nobody reading
    either screen could tell why.

    So a promoted word is recorded at the bottom of the band it was promoted
    into, and no higher. The unanimous-agreement rule argues that the word is
    good enough to leave alone; it does not argue that the word is certain,
    and rounding it up to one would claim something nobody established.

    A category moved the other way, *down* below what its strength says, is
    left alone. That happens where the rules refused to settle the word, and
    nothing goes wrong: the word is already in the queue for its own stated
    reasons, and an honest strength beside it tells the reader the useful
    truth that the arithmetic was content and the refusal came from
    somewhere else. Only an inflated strength can hide a word from the person
    who ought to see it, so only inflation is corrected here.
    """
    if category is Confidence.HIGH and assessment.strength < HIGH_CONFIDENCE_THRESHOLD:
        return HIGH_CONFIDENCE_THRESHOLD
    return assessment.strength


#: A value that cannot be confused with "the model declined", which is what
#: ``None`` means in the adjudication mapping.
_NOT_ASKED = object()


def _gather(
    scope: tuple[int, ...],
    table: AlignedTable,
    coverage: Mapping[Provider, Mapping[int, AlignmentColumn]],
    backbone_tokens: tuple[ProviderToken, ...],
) -> list[_Reading]:
    """Collect every service's reading of this place, the backbone's included."""
    backbone_text = " ".join(token.text for token in backbone_tokens)
    readings: list[_Reading] = [
        _Reading(
            provider=table.backbone_provider,
            text=backbone_text,
            tokens=backbone_tokens,
            positions=scope,
            quality=1.0,
            status=AlignmentStatus.EXACT,
            equivalence=EquivalenceKind.IDENTICAL,
            is_backbone=True,
        )
    ]
    for provider, per_position in coverage.items():
        columns: list[AlignmentColumn] = []
        for position in scope:
            column = per_position.get(position)
            if column is not None and column not in columns:
                columns.append(column)
        tokens = tuple(token for column in columns for token in column.aligned_tokens)
        if not tokens:
            continue
        text = " ".join(token.text for token in tokens)
        quality = min((column.quality for column in columns), default=0.0)
        status = columns[0].status if len(columns) == 1 else AlignmentStatus.MERGE
        readings.append(
            _Reading(
                provider=provider,
                text=text,
                tokens=tokens,
                positions=scope,
                quality=quality,
                status=status,
                equivalence=equivalence_kind(backbone_text, text),
            )
        )
    return readings


def _deletions(
    scope: tuple[int, ...],
    coverage: Mapping[Provider, Mapping[int, AlignmentColumn]],
    table: AlignedTable,
    extents: Mapping[Provider, tuple[int, int]] | None = None,
    options: ReconciliationOptions = DEFAULT_OPTIONS,
) -> tuple[Provider, ...]:
    """The services that heard nothing where the backbone heard a word.

    A service that returned nothing at all is not counted. It is not
    disagreeing about this word; it never spoke about any word, and letting
    a failed request argue against every word of the recording would turn
    one outage into a transcript full of doubt.

    Neither is a service that returned nothing anywhere near here. A service
    whose output stops half way through the recording is in the same
    position, for every word after the stop, as one that never answered,
    and is treated the same way: outside the first and last word it put
    anything against, or with no word of its own within reach of this
    place, its silence is not evidence. See
    :attr:`ReconciliationOptions.absence_reach`.
    """
    if extents is None:
        extents = _extents(coverage)
    found: list[Provider] = []
    count = len(table.backbone_tokens)
    for provider, per_position in coverage.items():
        if provider in table.missing_providers:
            continue
        columns = [per_position.get(position) for position in scope]
        if not all(column is not None and not column.aligned_tokens for column in columns):
            continue
        if _is_absent_near(per_position, scope, count, extents.get(provider), options.absence_reach):
            continue
        found.append(provider)
    return tuple(found)


def _is_absent_near(
    per_position: Mapping[int, AlignmentColumn],
    scope: tuple[int, ...],
    count: int,
    extent: tuple[int, int] | None,
    reach: int,
) -> bool:
    """Whether a service put no word at all near this place.

    True outside the stretch of the recording the service covered at all,
    and true inside it where the nearest word the service put down is more
    than ``reach`` backbone positions away.
    """
    if extent is None:
        return True
    first, last = extent
    if scope[-1] < first or scope[0] > last:
        return True
    start = max(0, scope[0] - reach)
    end = min(count, scope[-1] + reach + 1)
    for position in range(start, end):
        column = per_position.get(position)
        if column is not None and column.aligned_tokens:
            return False
    return True


def _group(readings: Sequence[_Reading]) -> list[_Group]:
    """Fold readings that are the same spoken evidence into one candidate.

    The comparison form settles nearly every case in one lookup. It does not
    settle all of them: "Jurgen" reaches "Jürgen" only through the full
    comparison, which has to know that one of the two texts contains a
    German letter. A miss therefore falls back to comparing against the
    groups already made, which is a short walk over at most a handful of
    them.
    """
    groups: list[_Group] = []
    for reading in readings:
        key = normalise(reading.text)
        found: _Group | None = None
        for group in groups:
            if group.key == key or are_equivalent(group.readings[0].text, reading.text):
                found = group
                break
        if found is None:
            found = _Group(key=key)
            groups.append(found)
        found.readings.append(reading)
    return groups


def _score_group(
    group: _Group,
    evidence: LanguageEvidence,
    afrikaans_enabled: bool,
    language: Language,
    speaker: str | None,
    vocabulary: VocabularyIndex | None,
    weights: ReliabilityWeights,
    historical: HistoricalWeight | None,
    options: ReconciliationOptions,
) -> None:
    """Work out what one candidate is worth, and keep every part of the sum.

    The six factors are the ones section 12 names, multiplied as it sets
    out. They are kept in :attr:`_Group.components` under the name of the
    service that contributed them, because the review window has to be able
    to tell a person why a candidate won, and a single number cannot.
    """
    per_provider: list[float] = []
    for reading in group.readings:
        reliability = weights.for_provider(reading.provider)
        language_weight = provider_language_weight(
            reading.provider, evidence, afrikaans_enabled
        )
        measured, _source = best_acoustic_confidence(reading.tokens)
        acoustic = ASSUMED_ACOUSTIC_CONFIDENCE if measured is None else measured
        vocabulary_factor, in_vocabulary = _vocabulary_factor(
            reading.text, vocabulary, options
        )
        measured_alignment = max(0.0, min(1.0, reading.quality))
        alignment = options.alignment_floor + (1.0 - options.alignment_floor) * (
            1.0 if reading.is_backbone else measured_alignment
        )
        performance = 1.0
        if historical is not None:
            performance = max(
                0.0,
                min(
                    1.0,
                    historical(
                        WordFacts(
                            provider=reading.provider,
                            language=language,
                            speaker=speaker,
                            is_proper_noun=_looks_like_name(reading.text),
                            is_numeric=any(
                                character.isdigit() for character in reading.text
                            ),
                        )
                    ),
                ),
            )
        raw = (
            reliability
            * language_weight
            * acoustic
            * vocabulary_factor
            * alignment
            * performance
        )
        name = reading.provider.value
        group.components[f"{name}.provider_reliability"] = round(reliability, 4)
        group.components[f"{name}.language_reliability"] = round(language_weight, 4)
        group.components[f"{name}.acoustic_confidence"] = round(acoustic, 4)
        group.components[f"{name}.vocabulary_evidence"] = round(vocabulary_factor, 4)
        group.components[f"{name}.alignment_quality"] = round(alignment, 4)
        group.components[f"{name}.alignment_measured"] = round(measured_alignment, 4)
        group.components[f"{name}.historical_performance"] = round(performance, 4)
        group.components[f"{name}.score"] = round(raw, 4)
        per_provider.append(raw)
        group.in_vocabulary = group.in_vocabulary or in_vocabulary

    group.raw_total = sum(per_provider)
    group.score = _combine(per_provider, options)
    group.components["combined"] = round(group.score, 4)
    group.components["supporting_providers"] = float(len(group.readings))
    group.display = _preferred_text(group, vocabulary, language)


def _combine(scores: Sequence[float], options: ReconciliationOptions) -> float:
    """Put several services' support for one reading together.

    Not by adding, and not by counting. Each service removes a share of the
    doubt that is left, which is how independent evidence actually
    accumulates, and which has the property the specification is asking for:
    the total approaches certainty and never reaches it, so no number of
    weak opinions can pass a strong one. Each supporter after the strongest
    counts for less again, because services trained on overlapping data make
    overlapping mistakes and pretending otherwise would let correlated
    agreement masquerade as independent confirmation.
    """
    doubt = 1.0
    for index, score in enumerate(sorted(scores, reverse=True)):
        contribution = min(options.maximum_factor, max(0.0, score))
        doubt *= 1.0 - contribution * (options.agreement_independence**index)
    return 1.0 - doubt


def _vocabulary_factor(
    text: str,
    vocabulary: VocabularyIndex | None,
    options: ReconciliationOptions,
) -> tuple[float, bool]:
    """What the user's own knowledge says about this spelling.

    Three answers, and the third is the one that earns this its place. A
    candidate the user has recorded as a mistake the services make is not
    merely unsupported; it is a spelling they have already rejected, and
    scoring it as though it were an ordinary rival would throw away the most
    specific thing they have told us.
    """
    if vocabulary is None:
        return 1.0, False
    match = vocabulary.match(text)
    if match is not None:
        if match.is_exact:
            return options.vocabulary_exact, True
        return options.vocabulary_equivalent, True
    if vocabulary.misrecognition_of(text) is not None:
        return options.vocabulary_misrecognition, False
    return 1.0, False


#: The spelling each language gives a word that is one spoken word written
#: in two languages, as in ``normalise._SAME_WORD_ACROSS_LANGUAGES``.
_SPELLING_BY_LANGUAGE: tuple[dict[Language, str], ...] = (
    {Language.ENGLISH: "yeah", Language.AFRIKAANS: "ja", Language.GERMAN: "ja"},
)


def _readings_for_language(
    readings: Sequence[_Reading], language: Language
) -> Sequence[_Reading]:
    """Keep the readings spelled for the language spoken at this place.

    Only where the readings hold more than one spelling of a word in
    ``_SPELLING_BY_LANGUAGE``: "Ja" and "Yeah" are then the same word, and
    the transcript should show the one the speaker's language writes. Where
    the language is not known, or no reading has its spelling, every
    reading is kept and the usual ranking decides.
    """
    for spellings in _SPELLING_BY_LANGUAGE:
        wanted = spellings.get(language)
        if wanted is None:
            continue
        present = {
            spelling
            for spelling in spellings.values()
            if any(spelled_as(reading.text, spelling) for reading in readings)
        }
        if len(present) < 2:
            continue
        chosen = [reading for reading in readings if spelled_as(reading.text, wanted)]
        if chosen:
            return chosen
    return readings


def _preferred_text(
    group: _Group,
    vocabulary: VocabularyIndex | None,
    language: Language,
) -> str:
    """Choose which spelling of one candidate to print.

    The user's own spelling wins, because that is the entire point of having
    asked them for it. Failing that, a spelling that kept a German letter
    beats one that dropped it: a service that wrote "Jürgen" heard something
    a service that wrote "Jurgen" did not write down, and printing the
    poorer of the two would lose information nobody can recover later.
    Otherwise the strongest service's spelling is used, with the backbone's
    preferred on a tie so that the same recording reconciles the same way
    twice. Where the readings are one word written in two languages, as
    "Ja" and "Yeah", only those spelled for the language spoken here are
    ranked; the chosen reading keeps its own capitals and punctuation.
    """
    texts = group.texts
    if not texts:
        return ""
    if vocabulary is not None:
        for text in texts:
            match = vocabulary.match(text)
            if match is not None:
                return match.term.text
    ranked = sorted(
        _readings_for_language(group.readings, language),
        key=lambda reading: (
            group.components.get(f"{reading.provider.value}.score", 0.0),
            _has_german_letter(reading.text),
        ),
        reverse=True,
    )
    accented = [reading for reading in ranked if _has_german_letter(reading.text)]
    if accented:
        return accented[0].text
    return ranked[0].text


_GERMAN_LETTERS = frozenset("üöäÜÖÄßẞéèêáàâíìîóòôúùû")


def _has_german_letter(text: str) -> bool:
    return any(character in _GERMAN_LETTERS for character in text)


def _looks_like_name(text: str) -> bool:
    """A rough test for a proper noun, used only to raise a review reason.

    Nothing is decided on this. It exists so that a disagreement over what
    is probably somebody's name can be told apart from a disagreement over
    an ordinary word in the review queue, which is one of the filters
    section 22 asks for.
    """
    words = [word for word in text.split() if word]
    if not words:
        return False
    return any(word[:1].isupper() and not word.isupper() for word in words)


# -- Settling, or refusing to ---------------------------------------------


def _settle(
    winner: _Group,
    runner_up: _Group | None,
    risk: tuple[RiskCategory, ...],
    position: int,
    scope: tuple[int, ...],
    vocabulary: VocabularyIndex | None,
    languages: LanguageReading,
    escalated: Container[int],
    options: ReconciliationOptions,
) -> tuple[bool, list[ReviewReason]]:
    """Decide whether this word may be settled at all, and say why not.

    The two rules that leave a word unresolved are different in kind. A
    narrow margin means the evidence did not separate the candidates, which
    is a statement about this recording. A high-risk value where the
    candidates differ is left unresolved however wide the margin, which is a
    statement about what a wrong answer would cost: the specification is
    explicit that a two-to-one result on a monetary amount is meaningful
    evidence and still not a reason to write the amount down as though it
    were known. Where no service heard a different value there is nothing
    to decide between, so a high-risk value alone does not stop it settling.
    """
    reasons: list[ReviewReason] = []
    if languages.is_uncertain(position):
        reasons.append(ReviewReason.LANGUAGE_UNCERTAIN)

    if runner_up is None:
        return False, reasons

    margin = winner.score - runner_up.score
    unresolved = margin < options.minimum_decision_margin

    if risk:
        reasons.append(ReviewReason.HIGH_RISK_ENTITY)
        unresolved = True
    if numbers_disagree(winner.display, runner_up.display):
        reasons.append(ReviewReason.NUMERIC_DISAGREEMENT)
    if _name_dispute(winner, runner_up, vocabulary):
        reasons.append(ReviewReason.PROPER_NAME_DISAGREEMENT)
    if unresolved and any(index in escalated for index in scope):
        reasons.append(ReviewReason.ESCALATION_UNRESOLVED)
    return unresolved, reasons


def _name_dispute(
    winner: _Group,
    runner_up: _Group,
    vocabulary: VocabularyIndex | None,
) -> bool:
    """Whether the two readings differ over something that looks like a name."""
    if vocabulary is not None:
        for group in (winner, runner_up):
            match = vocabulary.match(group.display)
            if match is not None and match.term.category in _NAME_CATEGORIES:
                return True
    return _looks_like_name(winner.display) or _looks_like_name(runner_up.display)


_NAME_CATEGORIES = (
    TermCategory.PERSON,
    TermCategory.COMPANY,
    TermCategory.PLACE,
    TermCategory.PRODUCT,
)


def _risk_for(
    scope: tuple[int, ...],
    backbone_texts: Sequence[str],
    groups: Sequence[_Group],
) -> tuple[RiskCategory, ...]:
    """The high-risk categories covering this place, over all its candidates.

    Each candidate is put into the surrounding words in turn and the results
    are pooled. That matters because the risk of a word can depend on which
    reading of it is right: if either "fifteen" or "fifty" would make the
    phrase an amount of money, the phrase is an amount of money and must not
    be settled quietly.
    """
    position = scope[0]
    found: list[RiskCategory] = []
    # Only the words within the risk window either side can bear on the
    # answer, so only those are copied. Copying the whole recording once per
    # candidate per word made this quadratic in the length of the recording,
    # for no difference in the result.
    start = max(0, position - RISK_WINDOW)
    end = min(len(backbone_texts), position + len(scope) + RISK_WINDOW)
    local = position - start
    for group in groups or ():
        words = list(backbone_texts[start:end])
        if 0 <= position < len(backbone_texts):
            words[local : local + len(scope)] = [group.display]
        for category in risk_at(words, local):
            if category not in found:
                found.append(category)
    return tuple(found)


def _is_hallucination(
    winner: _Group,
    deletions: tuple[Provider, ...],
    readings: Sequence[_Reading],
    backbone_tokens: tuple[ProviderToken, ...],
    options: ReconciliationOptions,
) -> bool:
    """Whether the backbone appears to have written down something nobody said.

    Both halves have to hold before a word is removed. Several services must
    have heard nothing here, and the backbone itself must have been unsure
    of what it heard. Either on its own is far too weak: services drop words
    at the end of a chunk all the time, and a service that was sure of a
    word is usually right even when the others missed it. Removing real
    speech is worse than keeping a doubtful word, because a kept word is
    visible and a removed one is not.
    """
    if len(deletions) < options.hallucination_deletion_count:
        return False
    if len(winner.readings) > 1 or len(readings) > 1:
        return False
    measured, _source = best_acoustic_confidence(backbone_tokens)
    if measured is None:
        return False
    return measured < options.hallucination_acoustic_ceiling


def _contesting_silence(
    deletions: tuple[Provider, ...],
    winner: _Group,
    runner_up: _Group | None,
    scope: tuple[int, ...],
    backbone_texts: Sequence[str],
    risk: tuple[RiskCategory, ...],
) -> tuple[Provider, ...]:
    """The silent services that count as arguing against the winning word.

    Normally every one of them. A service that heard nothing where others
    heard a word is evidence that the word may not be there.

    That stops being true for a filler such as "um" and for a word that
    repeats the word beside it, as in "we, we". Several services tidy speech
    and leave both out as a matter of style, so their silence says nothing
    about the audio. Counting it put every such word in the review list as
    a disagreement even when every service that wrote it agreed. Where no
    service heard a different word, the silence is set aside.

    The same holds for a high-risk value such as a date or an amount that
    at least two services heard the same way. A service that heard nothing
    there did not hear a different value, so its silence is not a dispute
    about which value was said. A value only one service heard keeps every
    silence against it, because nobody corroborated it.

    Only the confidence of the word uses this. Removing a word that nobody
    else heard still counts every silent service; see
    :func:`_is_hallucination`.
    """
    if runner_up is not None:
        return deletions
    if is_filler(winner.display) or _is_repeated_beside(scope, backbone_texts, winner.display):
        return ()
    if risk and len(winner.providers) >= 2:
        return ()
    return deletions


def _is_repeated_beside(
    scope: tuple[int, ...],
    backbone_texts: Sequence[str],
    text: str,
) -> bool:
    """Whether this text repeats the backbone word just before or after it.

    Both sides are checked because alignment does not choose which copy of
    a stutter a service left out. Given "so we we should" against "so we
    should", it puts the gap on the first "we", which only matches the word
    after it. Punctuation the backbone returned as its own entry is
    skipped, so "we , we" is still a repeat.
    """
    form = normalise(text)
    if not form:
        return False
    before = range(scope[0] - 1, -1, -1)
    after = range(scope[-1] + 1, len(backbone_texts))
    return any(
        _nearest_word(backbone_texts, positions) == form for positions in (before, after)
    )


def _nearest_word(backbone_texts: Sequence[str], positions: range) -> str:
    """The comparison form of the first real word at these positions, or ""."""
    for position in positions:
        if 0 <= position < len(backbone_texts):
            form = normalise(backbone_texts[position])
            if form:
                return form
    return ""


def _apply_adjudication(
    decision: object,
    groups: Sequence[_Group],
    winner: _Group,
    unresolved: bool,
) -> tuple[_Group, bool, str | None, bool]:
    """Take the language model's answer, but only where it chose an option.

    The model is allowed to pick between the readings the services offered.
    It is not allowed to introduce a reading of its own, and an answer that
    matches none of the candidates is therefore treated exactly as a refusal
    would be. Section 17 forbids the model rewriting the transcript, and the
    only way to enforce that is to refuse anything that is not already on
    the table.
    """
    if not isinstance(decision, str) or not decision.strip():
        return winner, True, None, False
    for group in groups:
        if are_equivalent(group.display, decision) or group.key == normalise(decision):
            return group, False, decision, True
    return winner, True, decision, False


def _everybody_agreed(
    groups: Sequence[_Group],
    winner: _Group,
    deletions: tuple[Provider, ...],
) -> bool:
    """Whether every service that spoke here said the same thing.

    Two services are required rather than one. A word only the backbone
    returned is not something the services agree about; it is something
    nobody else was asked about, and presenting it as corroborated would
    make a transcript from a degraded run look exactly like one from a
    complete run.
    """
    return len(groups) == 1 and len(winner.providers) >= 2 and not deletions


def _contesting(groups: Sequence[_Group], winner: _Group) -> tuple[Provider, ...]:
    """The services that put something else here."""
    found: list[Provider] = []
    for group in groups:
        if group is winner:
            continue
        for provider in group.providers:
            if provider not in found:
                found.append(provider)
    return tuple(found)


def _group_acoustic(group: _Group) -> float | None:
    measured, _source = best_acoustic_confidence(
        token for reading in group.readings for token in reading.tokens
    )
    return measured


def _group_acoustic_source(group: _Group) -> Provider | None:
    _measured, source = best_acoustic_confidence(
        token for reading in group.readings for token in reading.tokens
    )
    return source


def _language_support(
    group: _Group,
    evidence: LanguageEvidence,
    languages: LanguageReading,
) -> float:
    """How far the services behind the winning reading are trusted here.

    This is section 16's language support, and it is deliberately not a
    measure of how certain the language is. A word in a recording whose
    language nobody has established is not a less reliable word; it is a
    word about which one further thing is unknown, and that is reported as
    its own review reason. What does bear on the decision is a reading
    resting on a service that this span's language excludes, which is why
    the strongest supporter's eligibility is what comes back.
    """
    if not group.readings:
        return 1.0
    return max(
        provider_language_weight(
            reading.provider, evidence, languages.afrikaans_enabled
        )
        for reading in group.readings
    )


def _group_quality(group: _Group) -> float:
    """The best alignment quality behind this candidate.

    The best rather than the average, because the question is whether
    anything lined this reading up confidently, and one service that did is
    a better answer than three that half did. The backbone's own reading is
    left out: it was not aligned against anything, so its quality is a
    convention rather than a measurement and would drown the real ones.
    """
    qualities = [reading.quality for reading in group.readings if not reading.is_backbone]
    return max(qualities, default=1.0)


# -- Building the words ---------------------------------------------------


def _emit(
    scope: tuple[int, ...],
    table: AlignedTable,
    backbone_tokens: tuple[ProviderToken, ...],
    groups: Sequence[_Group],
    winner: _Group,
    assessment_category: Confidence,
    assessment_strength: float,
    risk: tuple[RiskCategory, ...],
    reasons: Sequence[ReviewReason],
    evidence: LanguageEvidence,
    language: Language,
    forced_alignment: ForcedAlignment | None,
    boundaries: Mapping[Provider, set[int]],
    overlaps: Container[int],
    llm_decision: str | None,
) -> list[FinalToken]:
    """Turn one settled place into the final words it produces.

    How many words come out is decided by the spelling that won, not by the
    backbone. A merged spelling produces one word covering the span of
    several; a split spelling produces several words sharing one span. The
    three cases are kept apart here rather than in the timing rules, because
    only here is it known which spelling won.
    """
    words = winner.display.split()
    if not words:
        words = [winner.display]
    shared = len(words) != len(scope) and len(words) > 1
    candidates = _candidates(groups)
    backbone_text = " ".join(token.text for token in backbone_tokens)

    tokens: list[FinalToken] = []
    for index, word in enumerate(words):
        if len(words) == len(scope):
            covered = (backbone_tokens[index],)
            equivalence = equivalence_kind(covered[0].text, word)
        else:
            covered = backbone_tokens
            # The span being inherited is the backbone's, so the question
            # is how the winner relates to what the backbone wrote there.
            # Comparing the winner with its own first reading answered a
            # different question, and called a substitution over a merged
            # span an exact provider time whenever the backbone's reading
            # was not in the winning group.
            equivalence = equivalence_kind(backbone_text, winner.display)
        timing = decide_timing(
            backbone_tokens=covered,
            equivalence=equivalence,
            shared=shared,
            quality=_group_quality(winner),
            forced_alignment=forced_alignment,
            text=word,
        )
        speaker = decide_speaker(
            position=scope[0],
            backbone_token=covered[0] if covered else None,
            backbone_boundaries=boundaries.get(table.backbone_provider, set()),
            other_boundaries={
                provider: positions
                for provider, positions in boundaries.items()
                if provider is not table.backbone_provider
            },
            overlapping=scope[0] in overlaps,
            diarised=bool(boundaries),
        )
        token = FinalToken(
            text=word,
            normalised_text=normalise(word),
            text_source=_source_of(winner),
            text_confidence=assessment_category,
            confidence_strength=assessment_strength,
            start=timing.start,
            end=timing.end,
            timing_source=timing.source,
            timing_status=timing.status,
            timing_confidence=timing.confidence,
            speaker=speaker.speaker,
            speaker_source=speaker.source,
            speaker_confidence=speaker.confidence,
            language=language,
            language_evidence=evidence,
            source_tokens=list(winner.token_references),
            source_audio_span=timing.source_audio_span,
            alignment_status=_status_for(winner, words, scope),
            candidates=candidates,
            risk_categories=list(risk),
            llm_decision=llm_decision,
        )
        for reason in (*reasons, *timing.reasons, *speaker.reasons):
            token.flag(reason)
        tokens.append(token)
    return tokens


def _candidates(groups: Sequence[_Group]) -> list[Candidate]:
    """Every reading that was on the table, strongest first and none thrown away."""
    return [
        Candidate(
            text=group.display,
            providers=group.providers,
            score=group.score,
            components=dict(group.components),
            source_tokens=group.token_references,
            in_vocabulary=group.in_vocabulary,
        )
        for group in groups
    ]


def _source_of(group: _Group) -> Provider | None:
    """Which service is credited with the winning spelling."""
    best: _Reading | None = None
    best_score = -1.0
    for reading in group.readings:
        score = group.components.get(f"{reading.provider.value}.score", 0.0)
        if reading.text == group.display and score > best_score:
            best, best_score = reading, score
    if best is not None:
        return best.provider
    return group.readings[0].provider if group.readings else None


def _status_for(
    winner: _Group,
    words: Sequence[str],
    scope: tuple[int, ...],
) -> AlignmentStatus:
    """How the winning reading stood against the backbone."""
    if len(words) < len(scope):
        return AlignmentStatus.MERGE
    if len(words) > len(scope):
        return AlignmentStatus.SPLIT
    kinds = {reading.equivalence for reading in winner.readings}
    if kinds == {EquivalenceKind.IDENTICAL}:
        return AlignmentStatus.EXACT
    if all(kind.is_equivalent for kind in kinds):
        return AlignmentStatus.FORMAT_EQUIVALENT
    return AlignmentStatus.SUBSTITUTION


# -- Words the backbone never had -----------------------------------------


def _insertion_token(
    row: AlignedRow,
    table: AlignedTable,
    languages: LanguageReading,
    vocabulary: VocabularyIndex | None,
    weights: ReliabilityWeights,
    historical: HistoricalWeight | None,
    options: ReconciliationOptions,
) -> FinalToken | None:
    """Decide what to do with a word the backbone never heard.

    Section 19 is clear that such a word has to be verified as actually
    spoken before it enters the verbatim transcript, and nothing available
    here can verify it. So the rule is a threshold rather than a judgement:
    a strongly supported insertion is kept, marked as having no acoustic
    evidence of its own and put in front of a person, and a weak one is
    dropped into provenance where a pattern of one service inventing words
    can still be seen. Either way the word is never quietly accepted and
    never quietly lost.

    A row can hold the same word from several services, because alignment
    puts services that inserted the same word in the same gap into one row.
    Each of them is scored on its own and the scores are put together the
    way agreeing readings of a backbone word are, by :func:`_combine`, so a
    second service hearing the word counts as further evidence for it and
    not as a second word. The text comes from the service with the best
    score, and every service in the row is listed as support.
    """
    tokens = tuple(token for column in row.columns for token in column.aligned_tokens)
    if not tokens:
        return None
    evidence = languages.for_position(row.position)
    components: dict[str, float] = {}
    scored: list[tuple[float, AlignmentColumn]] = []
    in_vocabulary = False
    for column in row.columns:
        if not column.aligned_tokens:
            continue
        reliability = weights.for_provider(column.provider)
        language_weight = provider_language_weight(
            column.provider, evidence, languages.afrikaans_enabled
        )
        measured, _source = best_acoustic_confidence(column.aligned_tokens)
        acoustic = ASSUMED_ACOUSTIC_CONFIDENCE if measured is None else measured
        vocabulary_factor, known = _vocabulary_factor(column.text, vocabulary, options)
        # An inserted word scores low on alignment quality by construction:
        # there was nothing on the other side for it to line up with. Using
        # that raw would mean no insertion could ever clear any threshold,
        # which would be a decision taken by arithmetic rather than by the
        # rule below.
        alignment = options.alignment_floor + (1.0 - options.alignment_floor) * column.quality
        raw = reliability * language_weight * acoustic * vocabulary_factor * alignment
        if historical is not None:
            raw *= max(0.0, min(1.0, historical(WordFacts(provider=column.provider))))
        name = column.provider.value
        components[f"{name}.provider_reliability"] = round(reliability, 4)
        components[f"{name}.language_reliability"] = round(language_weight, 4)
        components[f"{name}.acoustic_confidence"] = round(acoustic, 4)
        components[f"{name}.vocabulary_evidence"] = round(vocabulary_factor, 4)
        components[f"{name}.alignment_quality"] = round(alignment, 4)
        components[f"{name}.alignment_measured"] = round(column.quality, 4)
        scored.append((raw, column))
        in_vocabulary = in_vocabulary or known

    # One service on its own keeps exactly the score it always had. Putting a
    # lone score through the combination would cap it just below one, which
    # changes nothing about the threshold but would change the number a
    # person sees in the review window for no reason.
    if len(scored) == 1:
        score = scored[0][0]
    else:
        score = _combine([raw for raw, _column in scored], options)
        for raw, column in scored:
            components[f"{column.provider.value}.score"] = round(raw, 4)
        components["supporting_providers"] = float(len(scored))
    components["combined"] = round(score, 4)
    if score < options.insertion_keep_threshold:
        return None
    # The strongest supporter gives the text, which matters only where the
    # services wrote the same word differently. On a tie, max keeps the
    # first, which is the service alignment saw first.
    lead = max(scored, key=lambda pair: pair[0])[1]
    providers = tuple(column.provider for _raw, column in scored)

    token = FinalToken(
        text=lead.text,
        normalised_text=normalise(lead.text),
        text_source=lead.provider,
        text_confidence=Confidence.REVIEW_REQUIRED,
        timing_status=TimingStatus.UNALIGNED,
        timing_confidence=Confidence.UNRESOLVED,
        language=languages.language_at(row.position),
        language_evidence=evidence,
        source_tokens=[TokenReference(token.provider, token.index) for token in tokens],
        source_audio_span=_gap_span(table, row.position),
        alignment_status=AlignmentStatus.INSERTION,
        candidates=[
            Candidate(
                text=lead.text,
                providers=providers,
                score=score,
                components=components,
                source_tokens=tuple(
                    TokenReference(token.provider, token.index) for token in tokens
                ),
                in_vocabulary=in_vocabulary,
            )
        ],
    )
    token.flag(ReviewReason.UNALIGNED_WORD)
    token.flag(ReviewReason.PROVIDER_DISAGREEMENT)
    return token


def _gap_span(table: AlignedTable, position: int) -> AudioSpan | None:
    """The stretch of audio an inserted word must lie in, if it lies anywhere.

    It is the gap between the timed words either side. Wider than a word,
    deliberately: the point is to give a person something they can listen to,
    and pretending to know a narrower span would be inventing the very
    boundary this avoids.
    """
    tokens = table.backbone_tokens
    before = None
    for index in range(position - 1, -1, -1):
        if tokens[index].end is not None:
            before = tokens[index].end
            break
    after = None
    for index in range(position, len(tokens)):
        if tokens[index].start is not None:
            after = tokens[index].start
            break
    if before is None and after is None:
        return None
    start = before if before is not None else after
    end = after if after is not None else before
    if start is None or end is None:
        return None
    return AudioSpan(min(start, end), max(start, end))


# -- Speakers -------------------------------------------------------------


def _speaker_boundaries(
    table: AlignedTable,
    coverage: Mapping[Provider, Mapping[int, AlignmentColumn]],
) -> dict[Provider, set[int]]:
    """Where each service hears the speaker change, in backbone positions.

    Expressed as positions rather than as labels on purpose. See
    :func:`decide_speaker` for why the labels themselves cannot be compared.
    """
    boundaries: dict[Provider, set[int]] = {}
    backbone: set[int] = set()
    previous: str | None = None
    labelled = False
    for position, token in enumerate(table.backbone_tokens):
        if position and token.speaker is not None and token.speaker != previous:
            backbone.add(position)
        if token.speaker is not None:
            previous, labelled = token.speaker, True
    if labelled:
        # A service that reported no speaker at all is left out entirely, so
        # that a caller can tell "nobody was asked" from "everybody agreed".
        boundaries[table.backbone_provider] = backbone

    for provider, per_position in coverage.items():
        found: set[int] = set()
        seen: str | None = None
        started = False
        for position in range(len(table.backbone_tokens)):
            column = per_position.get(position)
            if column is None or not column.aligned_tokens:
                continue
            speaker = column.aligned_tokens[0].speaker
            if speaker is None:
                continue
            if started and speaker != seen:
                found.add(position)
            seen, started = speaker, True
        if started:
            boundaries[provider] = found
    return boundaries


def _overlapping_positions(table: AlignedTable) -> set[int]:
    """Backbone words whose span runs into a neighbour's under another speaker.

    Two words that overlap in time and are attributed to different people is
    what overlapping speech looks like in a timed transcript, and section 12
    lists it as a reason to escalate. Words of the same speaker overlapping
    slightly is ordinary run-on speech and is not interesting.
    """
    found: set[int] = set()
    tokens = table.backbone_tokens
    for index in range(1, len(tokens)):
        previous, token = tokens[index - 1], tokens[index]
        if not previous.has_timing or not token.has_timing:
            continue
        if previous.speaker is None or token.speaker is None:
            continue
        if previous.speaker == token.speaker:
            continue
        if token.start is not None and previous.end is not None and token.start < previous.end:
            found.add(index - 1)
            found.add(index)
    return found
