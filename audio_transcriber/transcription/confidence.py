"""Turning many different kinds of evidence into one honest category.

Section 16 of the specification is emphatic about what not to do here. Do
not reduce all the confidence information to a single provider score, and do
not present provider confidences as though they were comparable. They are
not comparable, and the reason is worth stating plainly: ElevenLabs reports
a log probability from its own decoder, AssemblyAI reports a number it calls
a confidence, Deepgram reports a different number it also calls a
confidence, and OpenAI and Microsoft report nothing at all. Averaging those
would produce a figure with a decimal point and no meaning, and worse, it
would look authoritative while having none.

So this module never averages across services. It takes the best acoustic
evidence there is for the reading that won, from whichever single service
actually measured it, and treats every other signal as its own separate
question: did the services agree, how clear was the decision between the
leading candidate and the next one, how well did the words line up, is the
language of this stretch known, is the word one the user has told us about,
and is it the kind of content where being nearly right is dangerous.

Those questions combine into one strength between nought and one, and the
strength falls into one of the four categories the specification names. The
combination is multiplicative, and every factor has a floor. That is
deliberate. A weak signal should pull a word towards review, but no single
missing measurement should be able to push a well-evidenced word all the way
to unresolved on its own, because the services that report nothing report
nothing about every word they return, and a rule that punished silence would
quietly mark down two of the three main services everywhere.

The thresholds are named constants rather than numbers buried in an
expression, because section 16 asks for them to be calibrated against real
human corrections later. The calibration work needs to be able to find them,
compare a transcript made today against one made last month, and say what
changed. A magic number inside a comparison could not be found, and moving
one silently would make two transcripts incomparable without anybody
knowing.

Nothing here depends on Qt or on any provider library, so it can be tested
on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from audio_transcriber.transcription.model import (
    AlignmentStatus,
    Confidence,
    Provider,
    ProviderToken,
    ReviewReason,
    TimingStatus,
)

#: Where the four categories begin. A word at or above the first is left
#: alone; below the last there is not enough evidence to claim anything and
#: the word is unresolved rather than guessed at.
HIGH_CONFIDENCE_THRESHOLD = 0.78
REVIEW_SUGGESTED_THRESHOLD = 0.55
REVIEW_REQUIRED_THRESHOLD = 0.30

#: Below this, a service that does measure its own certainty is telling us
#: it was unsure, which section 12 lists as a reason to escalate.
LOW_ACOUSTIC_THRESHOLD = 0.55

#: Below this, the alignment is not confident that the services were even
#: talking about the same word, which is a different problem from their
#: disagreeing about what it was.
WEAK_ALIGNMENT_THRESHOLD = 0.50

#: What acoustic confidence is assumed for a service that reports none.
#: OpenAI and Microsoft never report one, and an absent measurement is not a
#: low measurement. Treating silence as doubt would mark down two of the
#: three default services on every word of every recording, which would be a
#: statement about the design of their APIs rather than about the audio.
ASSUMED_ACOUSTIC_CONFIDENCE = 0.80

#: How much a high-risk word is marked down when the services disagreed
#: about it. Agreement on a high-risk word is still agreement; it is the
#: combination of money and doubt that section 13 wants raised in priority.
HIGH_RISK_PENALTY = 0.75

#: How far a decision that was nearly a tie can be trusted, as against one
#: where the leading reading was well clear of the next. The floor is the
#: first number and the span above it the second.
DECISIVENESS_FLOOR = 0.60
DECISIVENESS_SPAN = 0.40

#: The same shape for the remaining signals. Each is a multiplier of the
#: form floor + span × signal, so a signal that is entirely absent costs the
#: word something but never everything.
ACOUSTIC_FLOOR, ACOUSTIC_SPAN = 0.70, 0.30
ALIGNMENT_FLOOR, ALIGNMENT_SPAN = 0.70, 0.30
LANGUAGE_FLOOR, LANGUAGE_SPAN = 0.85, 0.15

#: What a vocabulary hit is worth to confidence, as against to the score. A
#: word the user has written down and confirmed is a word we have outside
#: evidence for, so it may pull a marginal decision up into high confidence.
VOCABULARY_BONUS = 1.10


def acoustic_confidence(token: ProviderToken) -> float | None:
    """The service's own certainty about this word, as a number from nought to one.

    A log probability is turned back into a probability, which is what it
    was before the logarithm and is directly meaningful. A service that
    reports a plain confidence is taken at its word. A service that reports
    neither gets ``None``, which means "did not say" and must not be
    confused with "was unsure".
    """
    if token.log_probability is not None:
        return max(0.0, min(1.0, math.exp(token.log_probability)))
    if token.confidence is not None:
        return max(0.0, min(1.0, token.confidence))
    return None


def best_acoustic_confidence(
    tokens: Iterable[ProviderToken],
) -> tuple[float | None, Provider | None]:
    """The strongest measured certainty among these words, and who measured it.

    The strongest rather than the average, and from one service rather than
    from several, because the numbers come off different scales and cannot
    be added up. What this answers is "did anybody who measures this actually
    hear it clearly?", which is a question with a meaningful answer.
    """
    best: float | None = None
    source: Provider | None = None
    for token in tokens:
        value = acoustic_confidence(token)
        if value is None:
            continue
        if best is None or value > best:
            best, source = value, token.provider
    return best, source


@dataclass(frozen=True)
class TextSignals:
    """Everything known about a decision, before it is turned into a category.

    Each field is one of the separate pieces of evidence section 16 asks to
    be kept apart. They arrive here already worked out by the reconciliation
    rules, so that this module has one job and can be tested by handing it
    numbers rather than a whole recording.
    """

    support: float = 0.0
    """The winning candidate's combined score, from nought to one."""

    runner_up: float = 0.0
    """The next best candidate's score, or nought where there was none."""

    acoustic: float | None = None
    """The best measured acoustic certainty, or ``None`` if nobody measured."""

    acoustic_source: Provider | None = None

    alignment_quality: float = 1.0
    """How confident the alignment is that the services meant this word."""

    agreeing_providers: int = 1
    contesting_providers: int = 0

    language_support: float = 1.0
    """How far the services behind this reading are trusted on this language.

    Section 16 calls this language support, and it is a different question
    from whether the language is certain. With Afrikaans disabled every
    service is valid on every span, so this is one and stays out of the way.
    Where it bites is a reading resting on a service that is not trusted on
    the language of the span it sits in, which is a real reason to look
    again, and it is the only language question that changes what may be
    believed. Whether the language itself is in doubt is reported separately,
    as a review reason, because it is a fact about the recording rather than
    about this decision.
    """

    in_vocabulary: bool = False
    is_high_risk: bool = False
    adjudicated: bool = False
    """Whether a language model was asked and gave an answer."""

    human_corrected: bool = False
    unresolved: bool = False
    """Set when the rules refused to decide, which no signal can undo."""


@dataclass(frozen=True)
class ConfidenceAssessment:
    """A category, the strength behind it, and why it came out that way."""

    category: Confidence
    strength: float
    components: dict[str, float] = field(default_factory=dict)
    reasons: tuple[ReviewReason, ...] = ()


def category_for(strength: float) -> Confidence:
    """Which of the four categories a strength falls into."""
    if strength >= HIGH_CONFIDENCE_THRESHOLD:
        return Confidence.HIGH
    if strength >= REVIEW_SUGGESTED_THRESHOLD:
        return Confidence.REVIEW_SUGGESTED
    if strength >= REVIEW_REQUIRED_THRESHOLD:
        return Confidence.REVIEW_REQUIRED
    return Confidence.UNRESOLVED


def assess_text(signals: TextSignals) -> ConfidenceAssessment:
    """Turn the evidence about one decision into a category and its reasons.

    A person's own correction ends the argument: they listened to their own
    audio and typed what they heard, which is stronger evidence than
    anything a service can offer, so the word is high confidence whatever
    the signals said before.

    A decision the rules refused to make ends it the other way. No amount of
    supporting evidence can promote a word that was left unresolved, because
    unresolved is a statement that the evidence was not sufficient, and
    letting a strong signal override it would be exactly the silent guess the
    specification forbids.
    """
    components: dict[str, float] = {}
    reasons: list[ReviewReason] = []

    if signals.human_corrected:
        return ConfidenceAssessment(Confidence.HIGH, 1.0, {"human_corrected": 1.0}, ())

    decisiveness = max(0.0, min(1.0, signals.support - signals.runner_up))
    acoustic = ASSUMED_ACOUSTIC_CONFIDENCE if signals.acoustic is None else signals.acoustic
    alignment = max(0.0, min(1.0, signals.alignment_quality))
    language = max(0.0, min(1.0, signals.language_support))

    components["support"] = round(signals.support, 4)
    components["decisiveness"] = round(decisiveness, 4)
    components["acoustic_confidence"] = round(acoustic, 4)
    components["alignment_quality"] = round(alignment, 4)
    components["language_support"] = round(language, 4)
    components["agreeing_providers"] = float(signals.agreeing_providers)
    components["contesting_providers"] = float(signals.contesting_providers)

    strength = max(0.0, min(1.0, signals.support))
    strength *= DECISIVENESS_FLOOR + DECISIVENESS_SPAN * decisiveness
    strength *= ACOUSTIC_FLOOR + ACOUSTIC_SPAN * acoustic
    strength *= ALIGNMENT_FLOOR + ALIGNMENT_SPAN * alignment
    strength *= LANGUAGE_FLOOR + LANGUAGE_SPAN * language
    if signals.in_vocabulary:
        components["vocabulary_bonus"] = VOCABULARY_BONUS
        strength *= VOCABULARY_BONUS
    if signals.is_high_risk and signals.contesting_providers:
        components["high_risk_penalty"] = HIGH_RISK_PENALTY
        strength *= HIGH_RISK_PENALTY
    strength = max(0.0, min(1.0, strength))
    components["strength"] = round(strength, 4)

    if signals.acoustic is not None and signals.acoustic < LOW_ACOUSTIC_THRESHOLD:
        reasons.append(ReviewReason.LOW_ACOUSTIC_CONFIDENCE)
    if signals.alignment_quality < WEAK_ALIGNMENT_THRESHOLD:
        reasons.append(ReviewReason.WEAK_ALIGNMENT)
    if signals.contesting_providers:
        reasons.append(ReviewReason.PROVIDER_DISAGREEMENT)

    category = Confidence.UNRESOLVED if signals.unresolved else category_for(strength)
    return ConfidenceAssessment(category, strength, components, tuple(reasons))


#: What each timing state is worth before the alignment quality is taken into
#: account. The three exact states are the ones a word may be navigated to;
#: the mapped ones put the right word in a span measured for another word,
#: which is defensible but not exact; the last three are various kinds of
#: not knowing.
_TIMING_STRENGTH: dict[TimingStatus, float] = {
    TimingStatus.EXACT_PROVIDER_TIME: 1.0,
    TimingStatus.MAPPED_FORMAT_EQUIVALENT: 0.95,
    TimingStatus.FORCED_ALIGNED: 0.95,
    TimingStatus.MAPPED_SUBSTITUTION: 0.75,
    TimingStatus.SHARED_PHRASE_SPAN: 0.6,
    TimingStatus.APPROXIMATE_SPAN: 0.45,
    TimingStatus.UNCERTAIN: 0.25,
    TimingStatus.UNALIGNED: 0.0,
}


def timing_confidence(status: TimingStatus, alignment_quality: float = 1.0) -> Confidence:
    """How far the span on a word can be trusted.

    Kept separate from the text confidence on purpose. A word whose spelling
    is certain can sit in a span nobody measured, and a word nobody can
    settle can sit in a span measured exactly. Collapsing the two would lose
    whichever of them was the better answer.
    """
    strength = _TIMING_STRENGTH.get(status, 0.0)
    strength *= ALIGNMENT_FLOOR + ALIGNMENT_SPAN * max(0.0, min(1.0, alignment_quality))
    return category_for(strength)


def speaker_confidence(
    has_speaker: bool,
    disputed_boundary: bool = False,
    overlapping: bool = False,
) -> Confidence:
    """How far the speaker on a word can be trusted.

    The three cases are genuinely different. A word with no speaker at all is
    unresolved, because nothing has been decided. A word where another
    service puts a change of speaker and the authority does not is a real
    dispute about attribution. Overlapping speech is worse than either,
    because diarisation is at its least reliable exactly where two people
    talk at once.

    Note that the *labels* the services use are never compared with each
    other. "speaker_0" and "A" are two services' names for whoever was
    talking, and nothing says they mean the same person. What can be compared
    is where each service puts a change, and that is what this rests on.
    """
    if not has_speaker:
        return Confidence.UNRESOLVED
    if overlapping:
        return Confidence.REVIEW_REQUIRED
    if disputed_boundary:
        return Confidence.REVIEW_SUGGESTED
    return Confidence.HIGH


def alignment_reasons(
    status: AlignmentStatus,
    quality: float,
) -> tuple[ReviewReason, ...]:
    """The review reasons that follow from how the words lined up.

    A word no service could be matched to has no acoustic evidence behind it
    at all, which is a different and more serious thing than a word matched
    weakly, so the two raise different reasons.
    """
    reasons: list[ReviewReason] = []
    if status is AlignmentStatus.UNALIGNED or status is AlignmentStatus.INSERTION:
        reasons.append(ReviewReason.UNALIGNED_WORD)
    if quality < WEAK_ALIGNMENT_THRESHOLD:
        reasons.append(ReviewReason.WEAK_ALIGNMENT)
    return tuple(reasons)


def weakest_of(values: Sequence[Confidence]) -> Confidence:
    """The least certain of several categories, for callers with a list."""
    return Confidence.weakest(*values)
