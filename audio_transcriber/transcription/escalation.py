"""Deciding what deserves a second opinion, and getting one.

Most words in a recording are heard the same way by every service, and
sending those anywhere again would be paying twice for an answer already in
hand. A few words are not: the services split three ways over a name, a
number comes back as two different numbers, a speaker boundary lands in the
middle of a sentence. Those are the moments worth asking about again, and
this module is what finds them and what asks.

Three ideas shape everything here.

The first is that a second opinion is about a passage of audio, never about
a word on its own. A service handed one word has no sentence to make sense
of, no way to tell which language it is in and nobody to identify as the
speaker, so it does worse on that word than it would have done on the whole
recording. Every question this module asks is therefore about a window with
several seconds of ordinary speech either side of the disputed part, and the
padding is clamped to the recording so that a dispute in the opening seconds
does not ask for audio from before the beginning.

The second is that disputes travel in groups. Three words argued over in the
same sentence are one question, not three. Grouping them into a single
window costs one request instead of three, takes a third of the time, and
gives the service more of the sentence to work with, so the answer is better
as well as cheaper. This is the most valuable thing in the module, and it is
the reason :func:`group_into_windows` exists as its own step rather than
being buried inside the sending.

The third is that a window has two timebases and only one of them means
anything outside this module. A service answers in the local time of
whatever audio it was given, and canonical time is that local time plus the
offset of the audio it was given:

    canonical_time = local_clip_time + canonical_offset

That single line is what the whole transcript rests on, and the way it is
kept honest here is by never guessing the offset. Where the recording is
uploaded once and asked about by time window, the service still holds the
whole canonical recording, so its local time already is canonical time and
the offset is zero. Where a clip has to be cut instead, the offset is the
one :func:`~audio_transcriber.transcription.canonical.cut_window` reports
having actually cut, not the one it was asked for, because a window that
runs off either end of the recording comes back shorter than requested and
an offset describing audio that was never cut would move every word in it.
The adapters add the offset as the times pass through them, so this module's
job is to hand them the right number and never to add it a second time.

Of the two ways to ask, the first is much the better. AssemblyAI takes a
start and an end time on an already-uploaded recording, so the audio goes up
once and a hundred questions can be asked about it for the price of the
questions. Cutting clips is the fallback for services that cannot do that:
it means encoding, writing and uploading a file for every question. Both
paths must produce identical canonical times, which is a thing worth proving
rather than assuming, and the tests prove it.

Two rules about what comes back. AssemblyAI hears Afrikaans only on its
older model, and its own documentation files that as moderately accurate,
which is a polite way of saying that between a quarter and a half of the
words may be wrong. An Afrikaans answer from it is therefore evidence that
informs a decision rather than an authority that settles one, and it is
marked as such here so that nothing downstream can mistake it for an English
answer. And escalation is charged per request, so a recording that confuses
every service is stopped at the ceiling in Settings. Reaching that ceiling
is not a failure and is never silent: the remaining passages are named in
the transcript's warnings and go to the review queue for a person to settle.

If AssemblyAI cannot be reached at all, the transcript is finished without
it and the affected passages are flagged for review. Losing a second opinion
must never cost the user their transcript.

Nothing here depends on Qt, so it can be tested on its own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from audio_transcriber.transcription.canonical import AudioClip, cut_window
from audio_transcriber.transcription.model import (
    AudioSpan,
    Candidate,
    CanonicalAudio,
    Confidence,
    FinalToken,
    Language,
    Provider,
    ProviderRequestRecord,
    ProviderResult,
    ProviderToken,
    ReviewReason,
    ReviewStatus,
    RiskCategory,
    TimingStatus,
    TokenReference,
    Transcript,
)
from audio_transcriber.transcription.normalise import are_equivalent, normalise, read_number
from audio_transcriber.transcription.providers.base import (
    CancelCheck,
    ProviderError,
    TranscriptionRequest,
)

_log = logging.getLogger(__name__)

#: What a request made by this module is called in provenance. Escalation
#: sends many small requests and they must not be mistaken for full passes
#: over the recording when the cost of a run is being explained afterwards.
ESCALATION_PURPOSE = "escalation"

#: How long a single window may grow to as disputes are joined into it.
#: Grouping saves money only while the window stays short; a window that has
#: swallowed a minute of speech is no longer a question about a disputed
#: word, and the service's attention is spread across everything else in it.
DEFAULT_MAXIMUM_WINDOW_SECONDS = 60.0

#: Below this, an ElevenLabs word is one the backbone itself was unsure of.
#: It is a starting point rather than a measurement, and it should be
#: calibrated against real corrections once there are enough of them.
DEFAULT_LOW_LOG_PROBABILITY = -0.5

#: How sure the language evidence has to be before a span counts as settled.
#: It matches the default in :meth:`LanguageEvidence.is_confidently`, and
#: below it an Afrikaans-enabled recording is at a language boundary that
#: only a second opinion can resolve.
DEFAULT_LANGUAGE_CERTAINTY = 0.75

#: The risk categories whose disagreements are numeric in the sense that
#: matters: getting one wrong produces a wrong value that reads perfectly
#: well, which is exactly the damage that must never be settled by
#: plausibility alone.
#: The confidence levels at which reconciliation has left a word open.
_UNSETTLED = frozenset({Confidence.UNRESOLVED, Confidence.REVIEW_REQUIRED})

_NUMERIC_RISKS = frozenset(
    {
        RiskCategory.MONEY,
        RiskCategory.DATE,
        RiskCategory.TIME,
        RiskCategory.PERCENTAGE,
        RiskCategory.TELEPHONE,
        RiskCategory.ACCOUNT_NUMBER,
        RiskCategory.VERSION_NUMBER,
        RiskCategory.QUANTITY,
        RiskCategory.MEDICAL_MEASUREMENT,
        RiskCategory.PRODUCT_CODE,
        RiskCategory.LEGAL_IDENTIFIER,
    }
)


class EscalationReason(str, Enum):
    """Why a passage was sent for a second opinion.

    These are the material disagreements the specification lists. They are
    kept apart from :class:`~audio_transcriber.transcription.model.ReviewReason`
    because they answer different questions: this one says why a service was
    asked again, and the review reason says why a person is being asked. A
    passage that escalation could not settle turns one into the other.
    """

    ALL_PROVIDERS_DISAGREE = "all_providers_disagree"
    UNSETTLED_DISAGREEMENT = "unsettled_disagreement"
    PROPER_NOUN_DIFFERS = "proper_noun_differs"
    NUMBER_DIFFERS = "number_differs"
    TECHNICAL_TERM_DIFFERS = "technical_term_differs"
    LANGUAGE_BOUNDARY_UNCLEAR = "language_boundary_unclear"
    LOW_PROBABILITY_BACKBONE_WORD = "low_probability_backbone_word"
    SPEAKER_BOUNDARY_UNCERTAIN = "speaker_boundary_uncertain"
    OVERLAPPING_SPEECH = "overlapping_speech"
    WEAK_ALIGNMENT = "weak_alignment"

    @property
    def display_name(self) -> str:
        return _ESCALATION_REASON_DISPLAY_NAMES[self]

    @property
    def review_reason(self) -> ReviewReason:
        """What to tell a person when a second opinion did not settle this."""
        return _REVIEW_REASON_FOR_ESCALATION[self]


_ESCALATION_REASON_DISPLAY_NAMES: dict[EscalationReason, str] = {
    EscalationReason.ALL_PROVIDERS_DISAGREE: "Every service heard something different",
    EscalationReason.UNSETTLED_DISAGREEMENT: "The services disagree and the evidence did not settle it",
    EscalationReason.PROPER_NOUN_DIFFERS: "A name differs between services",
    EscalationReason.NUMBER_DIFFERS: "A number, date or amount differs between services",
    EscalationReason.TECHNICAL_TERM_DIFFERS: "A known term differs between services",
    EscalationReason.LANGUAGE_BOUNDARY_UNCLEAR: "The language of this passage is unclear",
    EscalationReason.LOW_PROBABILITY_BACKBONE_WORD: "The backbone was unsure of this word",
    EscalationReason.SPEAKER_BOUNDARY_UNCERTAIN: "The speaker changes near this word",
    EscalationReason.OVERLAPPING_SPEECH: "People are speaking over each other",
    EscalationReason.WEAK_ALIGNMENT: "The services' words could not be matched up",
}

_REVIEW_REASON_FOR_ESCALATION: dict[EscalationReason, ReviewReason] = {
    EscalationReason.ALL_PROVIDERS_DISAGREE: ReviewReason.PROVIDER_DISAGREEMENT,
    EscalationReason.UNSETTLED_DISAGREEMENT: ReviewReason.PROVIDER_DISAGREEMENT,
    EscalationReason.PROPER_NOUN_DIFFERS: ReviewReason.PROPER_NAME_DISAGREEMENT,
    EscalationReason.NUMBER_DIFFERS: ReviewReason.NUMERIC_DISAGREEMENT,
    EscalationReason.TECHNICAL_TERM_DIFFERS: ReviewReason.PROVIDER_DISAGREEMENT,
    EscalationReason.LANGUAGE_BOUNDARY_UNCLEAR: ReviewReason.LANGUAGE_UNCERTAIN,
    EscalationReason.LOW_PROBABILITY_BACKBONE_WORD: ReviewReason.LOW_ACOUSTIC_CONFIDENCE,
    EscalationReason.SPEAKER_BOUNDARY_UNCERTAIN: ReviewReason.SPEAKER_UNCERTAIN,
    EscalationReason.OVERLAPPING_SPEECH: ReviewReason.OVERLAPPING_SPEECH,
    EscalationReason.WEAK_ALIGNMENT: ReviewReason.WEAK_ALIGNMENT,
}


class EvidenceStrength(str, Enum):
    """How much weight an escalation answer is entitled to.

    A second opinion in English or German comes from the strongest model
    AssemblyAI has and can settle a dispute on its own. The same service
    hearing Afrikaans is running its older model, which its own
    documentation files as moderately accurate. Both answers arrive in the
    same shape, and without this distinction the weaker one would be weighed
    as though it were the stronger, which is precisely the mistake that
    produces a confident transcript in the wrong words.
    """

    DECIDING = "deciding"
    """Strong enough to settle a disagreement between the other services."""

    INFORMING = "informing"
    """Worth having as evidence, but never enough to settle anything alone."""

    @property
    def display_name(self) -> str:
        return _EVIDENCE_STRENGTH_DISPLAY_NAMES[self]

    @property
    def confidence_ceiling(self) -> Confidence:
        """The best a word may be called on the strength of this answer alone.

        An informing answer cannot lift a word out of the review queue, so
        the ceiling is what stops one from doing so by accident.
        """
        if self is EvidenceStrength.DECIDING:
            return Confidence.HIGH
        return Confidence.REVIEW_SUGGESTED


_EVIDENCE_STRENGTH_DISPLAY_NAMES: dict[EvidenceStrength, str] = {
    EvidenceStrength.DECIDING: "Strong enough to settle the disagreement",
    EvidenceStrength.INFORMING: "Weak evidence, not enough to settle it",
}


@dataclass(frozen=True)
class Dispute:
    """One passage of audio worth asking about again.

    It is deliberately small: a region of the canonical recording, the
    reasons it is in doubt, and the identifiers of the final words it covers
    so that an answer can be put back where it came from. It carries no
    provider evidence of its own, because everything that decided it was a
    dispute has already been recorded on the words themselves.
    """

    span: AudioSpan
    reasons: tuple[EscalationReason, ...]
    token_ids: tuple[str, ...] = ()
    language: Language = Language.UNKNOWN
    candidates: tuple[str, ...] = ()
    """What the services offered here, for the record and for the log."""

    @property
    def description(self) -> str:
        """One sentence naming when this is and what is wrong with it."""
        reasons = "; ".join(reason.display_name for reason in self.reasons)
        return f"{self.span.start:.1f}s to {self.span.end:.1f}s: {reasons}."


@dataclass(frozen=True)
class EscalationWindow:
    """The audio one request will ask about, and everything it is asking.

    ``span`` is the padded window that will be sent and ``target`` is the
    region the disputes themselves cover. Both are needed afterwards: the
    padding is context the service was given and not something it was asked
    about, so an answer outside the target is background rather than a
    correction.
    """

    span: AudioSpan
    target: AudioSpan
    disputes: tuple[Dispute, ...]
    language: Language = Language.UNKNOWN
    clamped: bool = False
    """Whether the recording ran out before the padding did, at either end."""

    @property
    def reasons(self) -> tuple[EscalationReason, ...]:
        """Every reason in this window, each named once, in order."""
        found: list[EscalationReason] = []
        for dispute in self.disputes:
            for reason in dispute.reasons:
                if reason not in found:
                    found.append(reason)
        return tuple(found)

    @property
    def token_ids(self) -> tuple[str, ...]:
        found: list[str] = []
        for dispute in self.disputes:
            for token_id in dispute.token_ids:
                if token_id not in found:
                    found.append(token_id)
        return tuple(found)

    @property
    def is_afrikaans(self) -> bool:
        return self.language is Language.AFRIKAANS

    @property
    def description(self) -> str:
        count = len(self.disputes)
        word = "dispute" if count == 1 else "disputes"
        return (
            f"{count} {word} between {self.target.start:.1f}s and "
            f"{self.target.end:.1f}s, sent as {self.span.start:.1f}s to {self.span.end:.1f}s."
        )


@dataclass(frozen=True)
class EscalationOptions:
    """Everything about escalation that a person can change.

    The margins and the ceiling come from Settings, and the language and
    speaker expectations come from the recording's own configuration.
    :meth:`from_settings` puts the two together, so that nothing above this
    module has to know which setting lives where.
    """

    context_seconds_before: float = 5.0
    context_seconds_after: float = 5.0

    join_gap_seconds: float | None = None
    """How close two disputes have to be to share a window.

    Left unset it is the sum of the two margins, which is the distance at
    which their padded windows would touch anyway. Joining them at that
    point costs no extra audio at all and saves a whole request.
    """

    maximum_window_seconds: float = DEFAULT_MAXIMUM_WINDOW_SECONDS
    maximum_escalations: int = 200
    """A ceiling on requests for one recording, because they are charged for."""

    afrikaans_enabled: bool = False
    expected_speaker_count: int = 1
    languages: tuple[Language, ...] = (Language.ENGLISH, Language.GERMAN)
    """The languages the recording may contain, for a window whose own
    language is not known. Naming one of several would push the service away
    from the others, so several means the service is left to detect."""

    low_log_probability: float = DEFAULT_LOW_LOG_PROBABILITY
    language_certainty: float = DEFAULT_LANGUAGE_CERTAINTY

    primary_model: str = "universal-3-5-pro"
    afrikaans_model: str = "universal-2"

    vocabulary_terms: tuple[str, ...] = ()
    context_prompt: str = ""

    @property
    def effective_join_gap_seconds(self) -> float:
        if self.join_gap_seconds is not None:
            return max(0.0, self.join_gap_seconds)
        return max(0.0, self.context_seconds_before) + max(0.0, self.context_seconds_after)

    @classmethod
    def from_settings(
        cls,
        processing: Any = None,
        assemblyai: Any = None,
        configuration: Any = None,
        vocabulary_terms: Iterable[str] = (),
    ) -> "EscalationOptions":
        """Build the options from the settings objects, without importing them.

        The attributes are read by name rather than by type, which keeps this
        module testable on its own and stops a change to the shape of the
        settings from reaching into escalation. Anything missing keeps the
        default, because a partly-configured application should still be able
        to ask a question.
        """
        options = cls()
        values: dict[str, Any] = {
            "context_seconds_before": _number(
                processing, "escalation_context_seconds_before", options.context_seconds_before
            ),
            "context_seconds_after": _number(
                processing, "escalation_context_seconds_after", options.context_seconds_after
            ),
            "maximum_escalations": int(
                _number(
                    processing,
                    "maximum_escalations_per_recording",
                    options.maximum_escalations,
                )
            ),
            "expected_speaker_count": int(
                _number(
                    configuration, "expected_speaker_count", options.expected_speaker_count
                )
            ),
            "primary_model": str(getattr(assemblyai, "primary_model", options.primary_model)),
            "afrikaans_model": str(
                getattr(assemblyai, "afrikaans_model", options.afrikaans_model)
            ),
            "vocabulary_terms": tuple(term for term in vocabulary_terms if term.strip()),
            "context_prompt": str(getattr(configuration, "recording_context", "") or ""),
        }
        afrikaans = getattr(configuration, "afrikaans_enabled", None)
        if afrikaans is None:
            afrikaans = getattr(processing, "default_afrikaans_enabled", False)
        values["afrikaans_enabled"] = bool(afrikaans)
        languages = list(options.languages)
        if values["afrikaans_enabled"] and Language.AFRIKAANS not in languages:
            languages.append(Language.AFRIKAANS)
        values["languages"] = tuple(languages)
        return cls(**values)


@dataclass(frozen=True)
class EscalationResult:
    """What one window produced, whether or not the service answered.

    ``tokens`` are already on the canonical timeline, because the adapter
    added ``canonical_offset`` to everything the service reported. The offset
    and the audio actually sent are kept here too, so that a suspect answer
    can be traced back to the exact piece of audio that produced it.
    """

    window: EscalationWindow
    model: str
    strength: EvidenceStrength
    audio_span: AudioSpan
    """The audio the service really heard, after any clamping."""

    canonical_offset: float
    used_time_window: bool
    """Whether the recording was asked about in place rather than cut up."""

    tokens: tuple[ProviderToken, ...] = ()
    request: ProviderRequestRecord | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def settles_disputes(self) -> bool:
        """Whether this answer is allowed to settle the disputes on its own."""
        return self.succeeded and self.strength is EvidenceStrength.DECIDING

    def tokens_in_target(self) -> tuple[ProviderToken, ...]:
        """The words inside the disputed region, without the padding.

        The padding was context, not a question. A word from it is a
        perfectly good word that nobody asked about, and treating it as an
        answer would let a second opinion quietly rewrite speech that was
        never in dispute.
        """
        target = self.window.target
        found: list[ProviderToken] = []
        for token in self.tokens:
            if token.start is None or token.end is None:
                continue
            if AudioSpan(token.start, max(token.start, token.end)).overlaps(target):
                found.append(token)
        return tuple(found)


@dataclass(frozen=True)
class EscalationOutcome:
    """Everything that happened when a recording's disputes were escalated."""

    results: tuple[EscalationResult, ...] = ()
    skipped: tuple[EscalationWindow, ...] = ()
    warnings: tuple[str, ...] = ()
    reached_limit: bool = False

    @property
    def requests(self) -> tuple[ProviderRequestRecord, ...]:
        return tuple(result.request for result in self.results if result.request is not None)

    @property
    def failed(self) -> tuple[EscalationResult, ...]:
        return tuple(result for result in self.results if not result.succeeded)

    @property
    def unsettled_token_ids(self) -> tuple[str, ...]:
        """The words a second opinion did not reach or did not settle.

        A window that was never sent and a window whose request failed are
        the same thing from the point of view of the person reviewing the
        transcript: nobody looked again, so they have to.
        """
        found: list[str] = []
        for window in self.skipped:
            found.extend(window.token_ids)
        for result in self.failed:
            found.extend(result.window.token_ids)
        return tuple(dict.fromkeys(found))


# -- Finding what is in dispute ------------------------------------------


def find_disputes(
    tokens: Sequence[FinalToken],
    results: dict[Provider, ProviderResult] | None = None,
    options: EscalationOptions | None = None,
) -> list[Dispute]:
    """Work out which words deserve a second opinion, and why.

    The reasons are the material disagreements from the specification, and a
    word can have several of them at once. Everything needed is already on
    the word except the backbone's own log probability, which lives on the
    provider evidence, so ``results`` is followed back through the word's
    source references to find it.

    A word with no audio behind it at all cannot be escalated, because there
    is nothing to send. It stays in the review queue instead, where it was
    already put by whatever noticed that it had no span.
    """
    options = options or EscalationOptions()
    disputes: list[Dispute] = []
    for token in tokens:
        reasons = reasons_for(token, log_probability=_backbone_probability(token, results),
                              options=options)
        if not reasons:
            continue
        span = token.audible_span
        if span is None:
            _log.debug(
                "The word %r is in dispute but has no audio behind it, so it goes "
                "straight to review rather than to a second opinion.",
                token.text,
            )
            continue
        disputes.append(
            Dispute(
                span=span,
                reasons=reasons,
                token_ids=(token.id,),
                language=token.language,
                candidates=tuple(dict.fromkeys(candidate.text for candidate in token.candidates)),
            )
        )
    return disputes


def reasons_for(
    token: FinalToken,
    log_probability: float | None = None,
    options: EscalationOptions | None = None,
) -> tuple[EscalationReason, ...]:
    """Every reason this one word is worth asking about again.

    The tests for a name and for a number are deliberately generous. Sending
    a passage that turns out to have been fine costs one short request;
    failing to send one that was not costs a wrong name or a wrong amount in
    a finished transcript, which is far more expensive and much harder to
    notice.
    """
    options = options or EscalationOptions()
    texts = _distinct_texts(token.candidates)
    disagreement = len(texts) > 1
    reasons: list[EscalationReason] = []

    if disagreement and _every_service_differs(token.candidates, texts):
        reasons.append(EscalationReason.ALL_PROVIDERS_DISAGREE)
    if disagreement and token.text_confidence in _UNSETTLED:
        # Two services against one is evidence the scoring rules weigh on
        # their own, and where they could weigh it the word is settled. Where
        # they could not, which is what the confidence says, a fourth voice
        # is exactly the evidence that is missing. Without this rule the
        # commonest dispute of all, an ordinary word heard two ways and left
        # unresolved, would never be asked about.
        reasons.append(EscalationReason.UNSETTLED_DISAGREEMENT)
    if disagreement and any(_looks_like_a_name(text) for text in texts):
        reasons.append(EscalationReason.PROPER_NOUN_DIFFERS)
    if (disagreement and any(_looks_numeric(text) for text in texts)) or _is_numeric_risk(token):
        reasons.append(EscalationReason.NUMBER_DIFFERS)
    if disagreement and any(candidate.in_vocabulary for candidate in token.candidates):
        reasons.append(EscalationReason.TECHNICAL_TERM_DIFFERS)
    if options.afrikaans_enabled and _language_is_unclear(token, options.language_certainty):
        reasons.append(EscalationReason.LANGUAGE_BOUNDARY_UNCLEAR)
    if (
        disagreement
        and log_probability is not None
        and log_probability <= options.low_log_probability
    ):
        reasons.append(EscalationReason.LOW_PROBABILITY_BACKBONE_WORD)
    if ReviewReason.SPEAKER_UNCERTAIN in token.review_reasons:
        reasons.append(EscalationReason.SPEAKER_BOUNDARY_UNCERTAIN)
    if ReviewReason.OVERLAPPING_SPEECH in token.review_reasons:
        reasons.append(EscalationReason.OVERLAPPING_SPEECH)
    if ReviewReason.WEAK_ALIGNMENT in token.review_reasons:
        reasons.append(EscalationReason.WEAK_ALIGNMENT)
    return tuple(reasons)


def _distinct_texts(candidates: Sequence[Candidate]) -> tuple[str, ...]:
    """The candidate texts that are genuinely different from each other.

    Comparison is done on the normalised forms, so "twenty-five" and "25" are
    one candidate rather than two. Escalating those would fill the queue with
    differences that are not differences, and hide the real ones.
    """
    seen: dict[str, str] = {}
    for candidate in candidates:
        if not candidate.text.strip():
            continue
        seen.setdefault(normalise(candidate.text), candidate.text)
    return tuple(seen.values())


def _every_service_differs(candidates: Sequence[Candidate], texts: Sequence[str]) -> bool:
    """Whether no two services agreed on anything here.

    Two services agreeing against a third is ordinary evidence and the
    scoring rules can weigh it. Nobody agreeing with anybody is a different
    situation: there is nothing to weigh, and only more listening will help.
    """
    providers: set[Provider] = set()
    for candidate in candidates:
        providers.update(candidate.providers)
    return len(providers) >= 2 and len(texts) >= len(providers)


def _looks_like_a_name(text: str) -> bool:
    """Whether this could be a proper noun, judged only by how it is written.

    A capital letter followed by small ones is all there is to go on without
    a parser, and it is enough: the cost of being wrong is one short request.
    """
    stripped = text.strip().strip(".,;:!?\"'()[]")
    if len(stripped) < 2 or not stripped[0].isupper():
        return False
    return any(character.islower() for character in stripped[1:])


def _looks_numeric(text: str) -> bool:
    """Whether this candidate is a number, in digits or in words."""
    if any(character.isdigit() for character in text):
        return True
    return read_number(text) is not None


def _is_numeric_risk(token: FinalToken) -> bool:
    return any(category in _NUMERIC_RISKS for category in token.risk_categories)


def _language_is_unclear(token: FinalToken, certainty: float) -> bool:
    """Whether the language of this word is in doubt.

    Only asked when Afrikaans is enabled for the recording. With Afrikaans
    off there is no code-switch to find and no fallback model to route to, so
    the question does not arise and asking it anyway would escalate half of a
    perfectly ordinary English recording.
    """
    if ReviewReason.LANGUAGE_UNCERTAIN in token.review_reasons:
        return True
    evidence = token.language_evidence
    if not evidence.scores:
        return False
    return not evidence.is_confidently(evidence.best, certainty)


def _backbone_probability(
    token: FinalToken, results: dict[Provider, ProviderResult] | None
) -> float | None:
    """The log probability the timing service gave the word behind this one.

    ElevenLabs is asked first because it is normally the backbone and its
    probability is the one the specification names. Where it is not in the
    evidence at all, because it failed and another service took over, the
    lowest probability among the sources is used, since a word is only as
    safe as the least sure service that heard it.
    """
    if not results:
        return None
    found: list[float] = []
    for reference in token.source_tokens:
        result = results.get(reference.provider)
        if result is None:
            continue
        source = result.token_at(reference.index)
        if source is None or source.log_probability is None:
            continue
        if reference.provider is Provider.ELEVENLABS:
            return source.log_probability
        found.append(source.log_probability)
    return min(found) if found else None


# -- Grouping disputes into windows --------------------------------------


def group_into_windows(
    disputes: Sequence[Dispute],
    duration: float | None = None,
    options: EscalationOptions | None = None,
) -> list[EscalationWindow]:
    """Gather nearby disputes into as few windows as honestly possible.

    This is where escalation stops being expensive. Disputes arrive one to a
    word, and a difficult sentence can hold four of them within two seconds.
    Sent separately that is four requests, four uploads on the services that
    need one, and four answers each of which saw only its own word. Sent
    together it is one request covering the whole sentence, which is cheaper,
    quicker, and better, because the service gets the sentence rather than a
    fragment of it.

    Two things stop a group from growing for ever. Disputes only join when
    the gap between them is no wider than the padding they would each be
    given anyway, so joining never asks for audio that was not going to be
    sent. And a window that has grown past
    :attr:`EscalationOptions.maximum_window_seconds` is closed, because past
    that point the disputed words are a small part of what the service is
    listening to.

    ``duration`` is the length of the canonical recording and is what the
    padding is clamped against. Without it the padding is clamped only at the
    start, which is the honest answer when the length is not known.
    """
    options = options or EscalationOptions()
    ordered = sorted(disputes, key=lambda dispute: (dispute.span.start, dispute.span.end))
    windows: list[EscalationWindow] = []
    group: list[Dispute] = []

    for dispute in ordered:
        if not group:
            group = [dispute]
            continue
        target = _target_span(group)
        gap = dispute.span.start - target.end
        joined = AudioSpan(target.start, max(target.end, dispute.span.end))
        padded, _ = _pad(joined, duration, options)
        if gap <= options.effective_join_gap_seconds and (
            padded.duration <= options.maximum_window_seconds
        ):
            group.append(dispute)
            continue
        windows.append(_window_for(group, duration, options))
        group = [dispute]

    if group:
        windows.append(_window_for(group, duration, options))
    return windows


def plan_escalation(
    tokens: Sequence[FinalToken],
    results: dict[Provider, ProviderResult] | None = None,
    duration: float | None = None,
    options: EscalationOptions | None = None,
) -> list[EscalationWindow]:
    """Find the disputes in a transcript and gather them into windows."""
    options = options or EscalationOptions()
    return group_into_windows(find_disputes(tokens, results, options), duration, options)


def _target_span(group: Sequence[Dispute]) -> AudioSpan:
    return AudioSpan(
        min(dispute.span.start for dispute in group),
        max(dispute.span.end for dispute in group),
    )


def _pad(target: AudioSpan, duration: float | None, options: EscalationOptions) -> tuple[
    AudioSpan, bool
]:
    """Add the context margins to a target region, clamped to the recording.

    The clamping is the whole reason this is not a subtraction and an
    addition written inline. A dispute two seconds into a recording cannot be
    given five seconds of context in front of it, and a window that claimed
    to start at minus three seconds would either fail or, worse, come back
    describing audio that does not exist.
    """
    padded = target.padded(
        max(0.0, options.context_seconds_before),
        max(0.0, options.context_seconds_after),
        limit=duration,
    )
    wanted_start = target.start - max(0.0, options.context_seconds_before)
    wanted_end = target.end + max(0.0, options.context_seconds_after)
    clamped = padded.start > wanted_start or padded.end < wanted_end
    return padded, clamped


def _window_for(
    group: Sequence[Dispute], duration: float | None, options: EscalationOptions
) -> EscalationWindow:
    target = _target_span(group)
    padded, clamped = _pad(target, duration, options)
    return EscalationWindow(
        span=padded,
        target=target,
        disputes=tuple(group),
        language=_language_of(group),
        clamped=clamped,
    )


def _language_of(group: Sequence[Dispute]) -> Language:
    """The language to treat a whole window as being in.

    Afrikaans wins over anything else in the group, because it is the one
    language here that changes which model is asked. A window holding one
    Afrikaans word and four English ones still has to go to the model that
    can hear Afrikaans at all.
    """
    languages = {dispute.language for dispute in group if dispute.language is not Language.UNKNOWN}
    if Language.AFRIKAANS in languages:
        return Language.AFRIKAANS
    if len(languages) == 1:
        return next(iter(languages))
    return Language.UNKNOWN


# -- Asking the question -------------------------------------------------


def choose_model(window: EscalationWindow, options: EscalationOptions) -> str:
    """Which AssemblyAI model to put at the front of the chain.

    The strong model normally, and the older one only where Afrikaans is
    enabled for the recording *and* this window is Afrikaans. Both conditions
    matter: routing a German span to the Afrikaans model because somebody
    switched Afrikaans on would make the transcript worse for no reason.
    """
    if options.afrikaans_enabled and window.is_afrikaans:
        return options.afrikaans_model
    return options.primary_model


def strength_of(window: EscalationWindow, options: EscalationOptions) -> EvidenceStrength:
    """How far an answer about this window may be trusted.

    Afrikaans runs on the older model, which AssemblyAI documents as
    moderately accurate: somewhere between a quarter and a half of the words
    may be wrong. That is worth having as evidence and is not worth treating
    as an answer, so it comes back marked as informing rather than deciding
    and nothing downstream can promote it by mistake.
    """
    if options.afrikaans_enabled and window.is_afrikaans:
        return EvidenceStrength.INFORMING
    return EvidenceStrength.DECIDING


def canonical_time(local_time: float, canonical_offset: float) -> float:
    """Turn a time inside a piece of sent audio into a time in the recording.

    The one conversion the whole transcript rests on. It is trivial
    arithmetic and it is written down here so that there is exactly one
    place where it happens and one name to look for when a transcript's
    words point at the wrong moment.
    """
    return local_time + canonical_offset


def prefers_time_window(provider: Any) -> bool:
    """Whether this service can be asked about part of a recording in place.

    A service that can is asked that way, because the audio then goes up once
    for the whole recording rather than once for every question. Everything
    else has clips cut for it, which works just as well and costs a great
    deal more time.
    """
    capabilities = getattr(provider, "capabilities", None)
    return bool(
        capabilities is not None
        and getattr(capabilities, "time_window", False)
        and hasattr(provider, "transcribe_window")
    )


#: Told how many windows have been answered out of how many will be sent.
#: Each window is a separate request that is submitted and polled, so a
#: recording with a hundred of them spends many minutes here, and a progress
#: bar that does not move for that long is indistinguishable from a hang.
EscalationProgress = Callable[[int, int], None]


def escalate(
    windows: Sequence[EscalationWindow],
    provider: Any,
    canonical: CanonicalAudio,
    options: EscalationOptions | None = None,
    folder: Path | None = None,
    cancelled: CancelCheck | None = None,
    progress: EscalationProgress | None = None,
) -> EscalationOutcome:
    """Ask a service about each window, and never fail the transcript.

    Where the service can be asked about a time window of an already-uploaded
    recording, the recording is uploaded once and every window is a cheap
    question against it. Where it cannot, or where the upload fails, a clip
    is cut for each window instead. The two paths differ only in what is sent
    and in which offset converts the answer back, and they produce identical
    canonical times.

    Nothing raises. A service that cannot be reached costs the affected
    passages their second opinion, and they are named in the warnings and
    left for a person to settle.
    """
    options = options or EscalationOptions()
    results: list[EscalationResult] = []
    skipped: list[EscalationWindow] = []
    warnings: list[str] = []

    allowed = max(0, options.maximum_escalations)
    if allowed == 0 and windows:
        skipped.extend(windows)
        warnings.append(_limit_warning(0, len(windows)))
        return EscalationOutcome(
            results=(), skipped=tuple(skipped), warnings=tuple(warnings), reached_limit=True
        )

    audio_url = _upload_once(provider, canonical, windows, warnings)
    reached_limit = False

    for index, window in enumerate(windows):
        if cancelled is not None and cancelled():
            skipped.extend(windows[index:])
            warnings.append(
                "The run was stopped before every disputed passage had a second opinion."
            )
            break
        if index >= allowed:
            skipped.extend(windows[index:])
            warnings.append(_limit_warning(allowed, len(windows)))
            reached_limit = True
            break
        results.append(
            _escalate_one(
                window,
                provider=provider,
                canonical=canonical,
                options=options,
                audio_url=audio_url,
                folder=folder,
                index=index,
                cancelled=cancelled,
            )
        )
        if progress is not None:
            progress(index + 1, min(len(windows), allowed))

    failures = [result for result in results if not result.succeeded]
    if failures:
        warnings.append(
            f"{len(failures)} of {len(results)} second opinions did not come back, so those "
            "passages are in the review queue for a person to settle."
        )
    return EscalationOutcome(
        results=tuple(results),
        skipped=tuple(skipped),
        warnings=tuple(warnings),
        reached_limit=reached_limit,
    )


def _limit_warning(allowed: int, wanted: int) -> str:
    """The sentence the user reads when the ceiling stopped the escalation.

    Written as plain fact rather than as an error, because the ceiling did
    exactly what it was set to do. What matters is that the passages it
    stopped are named as still needing attention instead of quietly
    disappearing.
    """
    return (
        f"Only {allowed} of {wanted} disputed passages were sent for a second opinion, "
        f"because the limit for one recording is {allowed}. The rest are in the review "
        "queue for a person to settle."
    )


def _upload_once(
    provider: Any,
    canonical: CanonicalAudio,
    windows: Sequence[EscalationWindow],
    warnings: list[str],
) -> str | None:
    """Put the recording on the service once, if that is how it likes to work.

    Returning ``None`` is not a failure. It means the questions will be asked
    by cutting clips instead, which is slower and dearer but gives the same
    answers, so a service that will not take an upload costs time rather than
    accuracy.
    """
    if not windows or not prefers_time_window(provider) or not hasattr(provider, "upload"):
        return None
    try:
        return provider.upload(Path(canonical.path))
    except ProviderError as error:
        _log.warning("The recording could not be uploaded for escalation: %s", error)
        warnings.append(
            "The recording could not be uploaded once for the second opinions, so a short "
            "clip was cut for each of them instead. The answers are the same; it takes longer."
        )
        return None
    except Exception as error:  # noqa: BLE001 - a broken upload must not stop the run
        _log.exception("Uploading the recording for escalation failed unexpectedly.")
        warnings.append(f"The recording could not be uploaded for the second opinions: {error}")
        return None


def _escalate_one(
    window: EscalationWindow,
    provider: Any,
    canonical: CanonicalAudio,
    options: EscalationOptions,
    audio_url: str | None,
    folder: Path | None,
    index: int,
    cancelled: CancelCheck | None,
) -> EscalationResult:
    model = choose_model(window, options)
    strength = strength_of(window, options)
    if audio_url is not None:
        return _ask_by_time_window(
            window, provider, canonical, options, audio_url, model, strength, index, cancelled
        )
    return _ask_by_clip(
        window, provider, canonical, options, model, strength, folder, index, cancelled
    )


def _ask_by_time_window(
    window: EscalationWindow,
    provider: Any,
    canonical: CanonicalAudio,
    options: EscalationOptions,
    audio_url: str,
    model: str,
    strength: EvidenceStrength,
    index: int,
    cancelled: CancelCheck | None,
) -> EscalationResult:
    """Ask about part of the recording the service already holds.

    The offset here is zero, and that is a statement about the audio rather
    than a shortcut. The service is holding the canonical recording itself,
    so the times it reports are already counted from the start of the
    canonical recording and adding anything to them would move every word.
    """
    request = _request_for(
        window,
        options,
        audio_path=Path(canonical.path),
        canonical_offset=0.0,
        window_span=window.span,
        as_time_window=True,
        model=model,
        index=index,
    )
    ask = lambda: provider.transcribe_window(  # noqa: E731 - a thunk for the retry loop
        request, audio_url=audio_url, window=window.span, cancelled=cancelled
    )
    try:
        # Through the adapter's own retry loop where it has one, so that a
        # rate limit on the fortieth of two hundred short questions costs a
        # pause rather than that passage's second opinion.
        retrying = getattr(provider, "call_with_retries", None)
        result = retrying(ask, cancelled) if callable(retrying) else ask()
    except ProviderError as error:
        result = ProviderResult(provider=_provider_of(provider), error=str(error))
    except Exception as error:  # noqa: BLE001 - a second opinion never fails a transcript
        _log.exception("Asking for a second opinion failed unexpectedly.")
        result = ProviderResult(provider=_provider_of(provider), error=str(error))

    return EscalationResult(
        window=window,
        model=model,
        strength=strength,
        audio_span=window.span,
        canonical_offset=0.0,
        used_time_window=True,
        tokens=tuple(result.tokens),
        request=result.request,
        error=result.error,
    )


def _ask_by_clip(
    window: EscalationWindow,
    provider: Any,
    canonical: CanonicalAudio,
    options: EscalationOptions,
    model: str,
    strength: EvidenceStrength,
    folder: Path | None,
    index: int,
    cancelled: CancelCheck | None,
) -> EscalationResult:
    """Cut the window out of the recording and send it as its own file.

    Everything after the cut is arithmetic on what was actually cut. A window
    that ran off the end of the recording comes back shorter than it was
    asked for, and its offset describes where the audio in the file really
    begins, so those are the numbers used here. Using the numbers that were
    asked for instead would shift every word in the clip by the difference,
    which nothing downstream could detect.
    """
    destination = None if folder is None else folder / f"escalation-{index:04d}.wav"
    try:
        # Speech services work at 16 kHz mono internally, so a clip at the
        # recording's own rate is a bigger upload carrying nothing extra.
        clip = cut_window(
            canonical, window.span, destination=destination, sample_rate=16000, channels=1
        )
    except Exception as error:  # noqa: BLE001 - a clip that cannot be cut is a failure like any
        _log.exception("A window could not be cut out of the recording.")
        return EscalationResult(
            window=window,
            model=model,
            strength=strength,
            audio_span=window.span,
            canonical_offset=window.span.start,
            used_time_window=False,
            error=f"The audio for this passage could not be prepared: {error}",
        )

    request = _request_for(
        window,
        options,
        audio_path=clip.path,
        canonical_offset=clip.canonical_offset,
        window_span=clip.span,
        # Deliberately not sent as a time window. The audio in this file has
        # already been cut down to the window, and a service that also
        # accepts a start and an end would take those canonical seconds as an
        # instruction to trim the clip again, cutting the wrong part of the
        # wrong audio.
        as_time_window=False,
        model=model,
        index=index,
    )
    try:
        result = provider.transcribe(request, cancelled)
    finally:
        if destination is None:
            # The clip was a temporary file made only to carry the question,
            # so it is removed whatever happened. Where a folder was named the
            # clip is kept, because then it is provenance.
            clip.path.unlink(missing_ok=True)

    return EscalationResult(
        window=window,
        model=model,
        strength=strength,
        audio_span=_clip_span(clip),
        canonical_offset=clip.canonical_offset,
        used_time_window=False,
        tokens=tuple(result.tokens),
        request=result.request,
        error=result.error,
    )


def _clip_span(clip: AudioClip) -> AudioSpan:
    """Where the audio in a clip really sits in the canonical recording.

    Built from the offset and the length of what was written, through the one
    conversion, rather than copied from the window that was requested. The
    two differ whenever the recording ran out first.
    """
    start = canonical_time(0.0, clip.canonical_offset)
    return AudioSpan(start, canonical_time(clip.duration, clip.canonical_offset))


def _request_for(
    window: EscalationWindow,
    options: EscalationOptions,
    audio_path: Path,
    canonical_offset: float,
    window_span: AudioSpan,
    as_time_window: bool,
    model: str,
    index: int,
) -> TranscriptionRequest:
    """Build the one request this window will be asked as.

    ``as_time_window`` decides whether the region is part of the question or
    only part of the record. A service holding the whole recording is being
    asked about a region of it, so the region belongs on the request. A
    service holding a clip has already been given nothing else, so naming the
    region again would ask it to trim what is already trimmed.
    """
    return TranscriptionRequest(
        audio_path=audio_path,
        canonical_offset=canonical_offset,
        duration=window_span.duration,
        window=window_span if as_time_window else None,
        languages=_languages_for(window, options),
        expected_speaker_count=max(1, options.expected_speaker_count),
        diarise=True,
        vocabulary_terms=options.vocabulary_terms,
        context_prompt=options.context_prompt,
        model_override=model,
        purpose=ESCALATION_PURPOSE,
        chunk_index=index,
    )


def _languages_for(
    window: EscalationWindow, options: EscalationOptions
) -> tuple[Language, ...]:
    """Which languages to allow for this window.

    A window whose language is known is sent as that language, which on
    AssemblyAI also decides which model runs. A window whose language is not
    known is sent with the recording's whole set, which leaves the service to
    detect rather than pushing it towards one answer.
    """
    if window.language is not Language.UNKNOWN:
        return (window.language,)
    return options.languages


def _provider_of(provider: Any) -> Provider:
    found = getattr(provider, "provider", None)
    return found if isinstance(found, Provider) else Provider.ASSEMBLYAI


# -- Putting the outcome back into the transcript ------------------------



# -- Choosing which windows to send when there are too many ----------------


#: The reasons that mean the services genuinely heard different words. A
#: window carrying one of these is where a second opinion changes the
#: transcript; a window that is only there because a number appeared, with
#: every service agreeing on it, is insurance rather than a dispute.
_DISAGREEMENT_REASONS = frozenset(
    {
        EscalationReason.ALL_PROVIDERS_DISAGREE,
        EscalationReason.UNSETTLED_DISAGREEMENT,
        EscalationReason.PROPER_NOUN_DIFFERS,
        EscalationReason.TECHNICAL_TERM_DIFFERS,
        EscalationReason.LOW_PROBABILITY_BACKBONE_WORD,
    }
)


def window_priority(window: EscalationWindow) -> int:
    """How much a second opinion on this window is worth, higher first.

    The ceiling on requests is applied to a list, and whatever order the
    list is in decides which passages are asked about and which are left
    for a person. In time order the ceiling cuts off the end of the
    recording, so a long interview would have every agreed-upon number in
    its first hour checked and every three-way disagreement in its last
    hour ignored. Ranking by what the window is about sends the money
    where it buys a correction.
    """
    reasons = set(window.reasons)
    contested = any(len(dispute.candidates) > 1 for dispute in window.disputes)
    if reasons & _DISAGREEMENT_REASONS:
        return 3
    if EscalationReason.NUMBER_DIFFERS in reasons and contested:
        return 3
    if EscalationReason.LANGUAGE_BOUNDARY_UNCLEAR in reasons:
        return 2
    if contested:
        return 2
    return 1


def prioritise(windows: Sequence[EscalationWindow]) -> list[EscalationWindow]:
    """Order the windows so that the most valuable are sent first.

    Windows of equal worth are kept in time order. The caller still applies
    the ceiling by taking the list from the front; this only decides what
    the front holds.
    """
    return sorted(windows, key=lambda window: (-window_priority(window), window.span.start))


# -- Putting the answers back into the transcript ---------------------------


#: The review reasons that are about the text of a word. These are the ones
#: a second opinion on the text can answer. A reason about the speaker or
#: the timing is left exactly where it was, because the second opinion was
#: not asked about those and must not be allowed to clear them.
_TEXT_REVIEW_REASONS = frozenset(
    {
        ReviewReason.PROVIDER_DISAGREEMENT,
        ReviewReason.PROPER_NAME_DISAGREEMENT,
        ReviewReason.NUMERIC_DISAGREEMENT,
        ReviewReason.LOW_ACOUSTIC_CONFIDENCE,
        ReviewReason.ESCALATION_UNRESOLVED,
    }
)


@dataclass(frozen=True)
class EscalationApplication:
    """What applying the second opinions did to the transcript."""

    settled: int = 0
    """Words whose text the second opinion decided."""

    confirmed: int = 0
    """Words the second opinion agreed with as they already stood."""

    unsettled: int = 0
    """Words that were asked about and still need a person."""

    evidence: ProviderResult | None = None
    """Everything the escalation service said, as one result, so that the
    review window and the exports can show it beside the other services."""

    @property
    def sentence(self) -> str:
        parts = []
        if self.settled:
            parts.append(f"{self.settled} corrected")
        if self.confirmed:
            parts.append(f"{self.confirmed} confirmed")
        if self.unsettled:
            parts.append(f"{self.unsettled} still waiting for you")
        if not parts:
            return "The second opinions changed nothing."
        return "Second opinions: " + ", ".join(parts) + "."


def apply_answers(
    tokens: Sequence[FinalToken],
    outcome: EscalationOutcome,
) -> EscalationApplication:
    """Weigh what the second opinions said against the words in dispute.

    This is the step that turns an escalation from an expense into a
    correction, and it is deliberately conservative in three ways.

    The second opinion may only choose between the readings the other
    services already offered. A reading nobody else heard is kept as
    evidence, for the review window and for the language model, and is
    never written into the transcript: a service asked about fifteen
    seconds of audio has less context than the services that heard the
    whole recording, and an answer that matches none of them is as likely
    to be its mistake as theirs.

    Only an answer strong enough to settle a dispute settles one. An
    Afrikaans answer comes from a weaker model and is marked as informing,
    so it is added to the evidence and lifts nothing out of the review
    queue. The ceiling on :class:`EvidenceStrength` is what enforces that.

    And a value that must not be guessed is never settled here, however
    clear the answer. Two services against one on an amount of money is
    evidence, and the specification is explicit that it is still not a
    reason to write the amount down as though it were known. Those words
    keep the answer as evidence and go to a person.

    Settling changes the text and nothing else. Timing and speaker come
    from their own sources, which is the idea the whole design rests on; a
    word whose spelling changed is marked as a substitution so that the
    timing stage knows to look at it again.
    """
    by_id = {token.id: token for token in tokens}
    settled = confirmed = unsettled = 0
    heard_everywhere: list[ProviderToken] = []

    # Every answer is renumbered into one running sequence first, so that
    # the references written onto the words point into the one result that
    # is kept, rather than into a per-window numbering nobody will have.
    for result in outcome.results:
        if not result.succeeded:
            continue
        for token in result.tokens:
            if token.is_spoken_word:
                heard_everywhere.append(_renumbered(token, len(heard_everywhere)))

    evidence = None
    if heard_everywhere:
        evidence = ProviderResult(
            provider=heard_everywhere[0].provider,
            tokens=heard_everywhere,
            speakers=list(
                dict.fromkeys(word.speaker for word in heard_everywhere if word.speaker)
            ),
        )

    for result in outcome.results:
        if not result.succeeded:
            continue
        target = result.window.target
        in_target = [word for word in heard_everywhere if _span_of(word).overlaps(target)]
        used: set[int] = set()
        for dispute in result.window.disputes:
            heard = _words_for(dispute, in_target, used)
            for token_id in dispute.token_ids:
                token = by_id.get(token_id)
                if token is None or token.human_corrected:
                    continue
                verdict = _apply_one(token, heard, result)
                if verdict == "settled":
                    settled += 1
                elif verdict == "confirmed":
                    confirmed += 1
                else:
                    unsettled += 1

    return EscalationApplication(
        settled=settled, confirmed=confirmed, unsettled=unsettled, evidence=evidence
    )


def _words_for(
    dispute: Dispute,
    candidates: Sequence[ProviderToken],
    used: set[int],
) -> list[ProviderToken]:
    """The service's words that answer this dispute, and only this one.

    A dispute is one word's span as the backbone measured it, and the
    service measures its own boundaries. Adjacent words from a service abut,
    so the neighbour of the disputed word overlaps the dispute's span
    whenever the two services' boundaries differ by a few milliseconds,
    which they always do. Matching on any overlap would therefore hand back
    "fifty dollars" for a question about "fifty", which matches nothing and
    settles nothing. So a word answers a dispute when its middle falls
    inside the dispute's span, and as many words are taken as the dispute
    has tokens, nearest the middle first.

    A word is used once. A single service word that straddles two disputed
    words is one word where the transcript has two, which is a disagreement
    about the word count rather than a confirmation of both.
    """
    span = dispute.span
    wanted = max(1, len(dispute.token_ids))
    inside = [
        word
        for word in candidates
        if id(word) not in used and span.start <= _midpoint(word) <= span.end
    ]
    if not inside:
        return []
    centre = (span.start + span.end) / 2.0
    inside.sort(key=lambda word: abs(_midpoint(word) - centre))
    chosen = inside[:wanted]
    chosen.sort(key=_midpoint)
    used.update(id(word) for word in chosen)
    return chosen


def _midpoint(word: ProviderToken) -> float:
    span = _span_of(word)
    return (span.start + span.end) / 2.0


def _span_of(word: ProviderToken) -> AudioSpan:
    """A word's span, tolerating a service that puts the end before the start."""
    start = word.start if word.start is not None else 0.0
    end = word.end if word.end is not None else start
    return AudioSpan(start, max(start, end))


def _apply_one(
    token: FinalToken,
    heard: Sequence[ProviderToken],
    result: EscalationResult,
) -> str:
    """Weigh one answer against one word. Returns what became of it."""
    if not heard:
        # The service was asked about this moment and reported no word in
        # it. That is not an answer a person can act on, so the word stays
        # where it was and is marked as still needing one.
        token.flag(ReviewReason.ESCALATION_UNRESOLVED)
        return "unsettled"

    provider = heard[0].provider
    heard_text = " ".join(word.text for word in heard if word.text).strip()
    references = tuple(TokenReference(provider, word.index) for word in heard)
    matched = _attach_evidence(token, heard_text, provider, references)

    if matched is None or not result.settles_disputes or token.risk_categories:
        token.flag(ReviewReason.ESCALATION_UNRESOLVED)
        return "unsettled"

    changed = not are_equivalent(matched.text, token.text)
    if changed:
        token.text = matched.text
        token.normalised_text = normalise(matched.text)
        token.text_source = provider
        if token.start is not None and token.end is not None:
            # The span still belongs to the word that was measured; a
            # different word now sits in it. Saying so is what lets the
            # timing stage decide whether to measure again.
            token.timing_status = TimingStatus.MAPPED_SUBSTITUTION
    token.text_confidence = result.strength.confidence_ceiling
    for reason in list(token.review_reasons):
        if reason in _TEXT_REVIEW_REASONS:
            token.review_reasons.remove(reason)
    if not token.review_reasons and token.review_status is ReviewStatus.PENDING:
        token.review_status = ReviewStatus.SETTLED
    return "settled" if changed else "confirmed"


def _attach_evidence(
    token: FinalToken,
    heard_text: str,
    provider: Provider,
    references: tuple[TokenReference, ...],
) -> Candidate | None:
    """Record what the second opinion heard among the word's candidates.

    Returns the candidate it agreed with, or ``None`` where it agreed with
    nobody, in which case what it heard is added as a candidate of its own
    so that the review window can show it, and so that the language model
    sees it, without it ever being taken as the answer.
    """
    if not heard_text:
        return None
    for position, candidate in enumerate(token.candidates):
        if are_equivalent(candidate.text, heard_text):
            if provider in candidate.providers:
                return candidate
            updated = replace(
                candidate,
                providers=(*candidate.providers, provider),
                source_tokens=(*candidate.source_tokens, *references),
            )
            token.candidates[position] = updated
            return updated
    token.candidates.append(
        Candidate(text=heard_text, providers=(provider,), source_tokens=references)
    )
    return None


def _renumbered(token: ProviderToken, index: int) -> ProviderToken:
    return replace(token, index=index)


def apply_outcome(transcript: Transcript, outcome: EscalationOutcome) -> None:
    """Write what happened into the transcript, plainly.

    Two things are recorded. The warnings are put where the user reads them
    at the end of a run, so a ceiling that stopped half the second opinions
    is a sentence they see rather than something they would only find by
    counting requests. And every word a second opinion did not reach is
    flagged for review, because from a reader's point of view a window that
    was never sent and one whose request failed are the same thing: nobody
    listened again.

    It never touches text, timing or speakers. Deciding what an answer means
    is reconciliation's work, and doing any of it here would put two modules
    in charge of the same words.
    """
    for warning in outcome.warnings:
        if warning not in transcript.warnings:
            transcript.warnings.append(warning)
    unsettled = set(outcome.unsettled_token_ids)
    if not unsettled:
        return
    for token in transcript.tokens:
        if token.id in unsettled:
            token.flag(ReviewReason.ESCALATION_UNRESOLVED)


def _number(source: Any, name: str, default: float) -> float:
    value = getattr(source, name, None)
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
