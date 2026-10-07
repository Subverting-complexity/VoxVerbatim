"""What a transcript is made of.

Everything in this package reads or writes the types defined here, so they
are the one place to look to understand the shape of the whole thing.

Three rules run through the design.

The first is that text, timing and speaker are separate. Each has its own
value, its own source and its own confidence. A word spelled the way OpenAI
heard it, timed the way ElevenLabs heard it and attributed to the speaker
AssemblyAI decided on is an ordinary word, not a broken one.

The second is that time is always canonical time: seconds from the start of
the one canonical audio file, as a float. Services are sent chunks and
windows of that file and answer in their own local time, and every adapter
converts back before its answer reaches anything here. Nothing downstream
should ever have to ask which timebase a number is in.

The third is that provider evidence is never destroyed. Reconciliation adds
a decision on top of what the services said; it does not overwrite it. That
is what lets a transcript be explained afterwards, reprocessed when a
service improves, and argued with by the person reviewing it.

Nothing here depends on Qt or on any provider library, so it can be tested
on its own.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from vox_verbatim.transcription.normalise import normalise

#: The version of the reconciliation rules that produced a transcript. It is
#: written into provenance so that a transcript produced today can still be
#: explained after the rules have moved on. Bump it whenever a change would
#: make the same inputs produce a different transcript.
RECONCILIATION_VERSION = 1

#: The version of the transcript file format on disk.
TRANSCRIPT_FORMAT_VERSION = 1


class Provider(str, Enum):
    """The speech-to-text services the application can call.

    The values are the keys used in settings, in file names and in
    provenance, so they are written down once here and never spelled out
    again as loose strings.
    """

    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"
    MICROSOFT = "microsoft"
    ASSEMBLYAI = "assemblyai"
    DEEPGRAM = "deepgram"

    @property
    def display_name(self) -> str:
        """The name to show a person, in full rather than as a key."""
        return _PROVIDER_DISPLAY_NAMES[self]


_PROVIDER_DISPLAY_NAMES: dict[Provider, str] = {
    Provider.ELEVENLABS: "ElevenLabs Scribe",
    Provider.OPENAI: "OpenAI",
    Provider.MICROSOFT: "Microsoft MAI",
    Provider.ASSEMBLYAI: "AssemblyAI",
    Provider.DEEPGRAM: "Deepgram",
}


class Language(str, Enum):
    """The languages a recording may contain.

    Only the three the specification names are listed. Anything else a
    service reports becomes ``UNKNOWN`` rather than being carried around as
    a code nothing else understands.
    """

    ENGLISH = "en"
    GERMAN = "de"
    AFRIKAANS = "af"
    UNKNOWN = "unknown"

    @property
    def display_name(self) -> str:
        return _LANGUAGE_DISPLAY_NAMES[self]

    @classmethod
    def from_code(cls, code: str | None) -> "Language":
        """Turn whatever a service called the language into one of ours.

        Services differ in what they return: "en", "eng", "en-GB", "English".
        The prefix is enough to tell the three apart, and anything else is
        honestly unknown rather than guessed at.
        """
        if not code:
            return cls.UNKNOWN
        text = code.strip().lower().replace("_", "-")
        for language, prefixes in _LANGUAGE_PREFIXES.items():
            if any(text == prefix or text.startswith(f"{prefix}-") for prefix in prefixes):
                return language
        return cls.UNKNOWN


_LANGUAGE_DISPLAY_NAMES: dict[Language, str] = {
    Language.ENGLISH: "English",
    Language.GERMAN: "German",
    Language.AFRIKAANS: "Afrikaans",
    Language.UNKNOWN: "Unknown",
}

_LANGUAGE_PREFIXES: dict[Language, tuple[str, ...]] = {
    Language.ENGLISH: ("en", "eng", "english"),
    Language.GERMAN: ("de", "deu", "ger", "german", "deutsch"),
    Language.AFRIKAANS: ("af", "afr", "afrikaans"),
}


class TimingStatus(str, Enum):
    """How much the start and end times of a word can be trusted.

    These are the states from the specification. They exist because a
    reconciled word does not always sit exactly where a service put one. A
    corrected spelling can honestly inherit the span of the word it
    replaced; two words merged into one honestly cover both spans; but a
    word split in two cannot honestly claim to know where the join falls,
    and inventing that boundary would be a lie the rest of the application
    would then believe.
    """

    EXACT_PROVIDER_TIME = "exact_provider_time"
    """A service timed this exact word and nothing has changed it."""

    MAPPED_FORMAT_EQUIVALENT = "mapped_format_equivalent"
    """Only the formatting changed, so the original span still fits."""

    MAPPED_SUBSTITUTION = "mapped_substitution"
    """A different word now sits in the span of the word it replaced."""

    SHARED_PHRASE_SPAN = "shared_phrase_span"
    """Several words share one span because the join between them is unknown."""

    FORCED_ALIGNED = "forced_aligned"
    """The span was measured again against the audio after the text changed."""

    APPROXIMATE_SPAN = "approximate_span"
    """The span is the right region but its edges are not exact."""

    UNALIGNED = "unaligned"
    """No acoustic evidence maps to this word at all."""

    UNCERTAIN = "uncertain"
    """Alignment was attempted and could not be justified."""

    @property
    def is_exact(self) -> bool:
        """Whether the span may be used for word-level navigation."""
        return self in (
            TimingStatus.EXACT_PROVIDER_TIME,
            TimingStatus.MAPPED_FORMAT_EQUIVALENT,
            TimingStatus.FORCED_ALIGNED,
        )


class AlignmentStatus(str, Enum):
    """What happened when a service's word was matched to the backbone word."""

    EXACT = "exact"
    """The same word, character for character."""

    FORMAT_EQUIVALENT = "format_equivalent"
    """The same word once case, punctuation and spelling habits are set aside."""

    SUBSTITUTION = "substitution"
    """A genuinely different word in the same place."""

    MERGE = "merge"
    """Several backbone words correspond to one final word."""

    SPLIT = "split"
    """One backbone word corresponds to several final words."""

    INSERTION = "insertion"
    """The service heard a word the backbone does not have."""

    DELETION = "deletion"
    """The backbone has a word the service did not hear."""

    UNALIGNED = "unaligned"
    """No correspondence could be established."""


class Confidence(str, Enum):
    """How much a value can be trusted, as a category rather than a number.

    Services report confidence on scales that mean different things and
    cannot be compared with each other, so turning them all into one
    percentage would invent a precision that is not there. Four categories
    are enough to drive the review queue, and they can be calibrated later
    against real corrections.

    The order matters: they are declared from most to least certain, and
    :meth:`weakest` relies on that.
    """

    HIGH = "high"
    REVIEW_SUGGESTED = "review_suggested"
    REVIEW_REQUIRED = "review_required"
    UNRESOLVED = "unresolved"

    @property
    def display_name(self) -> str:
        return _CONFIDENCE_DISPLAY_NAMES[self]

    @property
    def needs_review(self) -> bool:
        return self is not Confidence.HIGH

    @classmethod
    def weakest(cls, *values: "Confidence") -> "Confidence":
        """Return the least certain of the given values.

        A word is only as trustworthy as its weakest part. Text we are sure
        of, sitting at a time we are not, is not a word that can be left out
        of the review queue.
        """
        order = list(cls)
        found = [value for value in values if value is not None]
        if not found:
            return cls.UNRESOLVED
        return max(found, key=order.index)


_CONFIDENCE_DISPLAY_NAMES: dict[Confidence, str] = {
    Confidence.HIGH: "High confidence",
    Confidence.REVIEW_SUGGESTED: "Review suggested",
    Confidence.REVIEW_REQUIRED: "Review required",
    Confidence.UNRESOLVED: "Unresolved",
}


class ReviewReason(str, Enum):
    """Why a word was put in front of a person.

    Every reason is also a filter in the review window, which is why they
    are named for what the person would look for rather than for the rule
    inside the code that raised them.
    """

    PROVIDER_DISAGREEMENT = "provider_disagreement"
    PROPER_NAME_DISAGREEMENT = "proper_name_disagreement"
    NUMERIC_DISAGREEMENT = "numeric_disagreement"
    HIGH_RISK_ENTITY = "high_risk_entity"
    LOW_ACOUSTIC_CONFIDENCE = "low_acoustic_confidence"
    SPEAKER_UNCERTAIN = "speaker_uncertain"
    OVERLAPPING_SPEECH = "overlapping_speech"
    LANGUAGE_UNCERTAIN = "language_uncertain"
    WEAK_ALIGNMENT = "weak_alignment"
    FORCED_ALIGNMENT_FAILED = "forced_alignment_failed"
    UNALIGNED_WORD = "unaligned_word"
    ESCALATION_UNRESOLVED = "escalation_unresolved"
    ADJUDICATION_DECLINED = "adjudication_declined"

    @property
    def display_name(self) -> str:
        return _REVIEW_REASON_DISPLAY_NAMES[self]


_REVIEW_REASON_DISPLAY_NAMES: dict[ReviewReason, str] = {
    ReviewReason.PROVIDER_DISAGREEMENT: "The services disagree",
    ReviewReason.PROPER_NAME_DISAGREEMENT: "A name differs between services",
    ReviewReason.NUMERIC_DISAGREEMENT: "A number differs between services",
    ReviewReason.HIGH_RISK_ENTITY: "A value that must not be guessed",
    ReviewReason.LOW_ACOUSTIC_CONFIDENCE: "The service was unsure of what it heard",
    ReviewReason.SPEAKER_UNCERTAIN: "The speaker is uncertain",
    ReviewReason.OVERLAPPING_SPEECH: "People are speaking over each other",
    ReviewReason.LANGUAGE_UNCERTAIN: "The language is uncertain",
    ReviewReason.WEAK_ALIGNMENT: "The words could not be matched up confidently",
    ReviewReason.FORCED_ALIGNMENT_FAILED: "The timing could not be measured again",
    ReviewReason.UNALIGNED_WORD: "This word has no audio behind it",
    ReviewReason.ESCALATION_UNRESOLVED: "A second opinion did not settle it",
    ReviewReason.ADJUDICATION_DECLINED: "The language model would not decide",
}


class RiskCategory(str, Enum):
    """Kinds of content that must never be settled by plausibility alone.

    Getting a common word wrong makes a transcript slightly worse. Getting
    an amount, a date or an account number wrong makes it dangerous, and the
    mistake is invisible because the wrong value reads perfectly well. These
    categories are what stops the scoring rules and the language model from
    quietly picking one.
    """

    MONEY = "money"
    DATE = "date"
    TIME = "time"
    PERCENTAGE = "percentage"
    TELEPHONE = "telephone"
    ACCOUNT_NUMBER = "account_number"
    VERSION_NUMBER = "version_number"
    ADDRESS = "address"
    QUANTITY = "quantity"
    LEGAL_IDENTIFIER = "legal_identifier"
    MEDICAL_MEASUREMENT = "medical_measurement"
    PRODUCT_CODE = "product_code"

    @property
    def display_name(self) -> str:
        return self.value.replace("_", " ").capitalize()


class ReviewStatus(str, Enum):
    """Where a word stands in the review workflow."""

    SETTLED = "settled"
    """Nothing about this word needs a person."""

    PENDING = "pending"
    """Waiting in the review queue."""

    CORRECTED = "corrected"
    """A person changed it."""

    CONFIRMED = "confirmed"
    """A person looked at it and left it as it was."""


# -- What a service said -------------------------------------------------


@dataclass(frozen=True)
class ProviderToken:
    """One word as one service reported it, converted to canonical time.

    This is evidence, not a decision. It is never edited: reconciliation
    builds :class:`FinalToken` objects that point back at these, and the
    review window shows them so a person can see what each service actually
    heard.

    ``start`` and ``end`` are seconds on the canonical timebase, and are
    ``None`` for a service that does not time its words at all, which is the
    normal case for OpenAI and Microsoft.
    """

    provider: Provider
    index: int
    """Position in that service's own sequence of words, counted from zero."""

    text: str
    """Exactly what the service returned, punctuation and casing included."""

    start: float | None = None
    end: float | None = None

    speaker: str | None = None
    """The service's own speaker label, before any renaming."""

    log_probability: float | None = None
    """A log probability where the service reports one, as ElevenLabs does."""

    confidence: float | None = None
    """A zero-to-one confidence where the service reports one instead."""

    language: Language = Language.UNKNOWN
    """The language of this word where the service says, per word or per segment."""

    chunk_index: int = 0
    """Which provider-specific chunk this word came back in.

    Zero when the whole recording went in one request. It is kept so that a
    word can be traced to the exact request that produced it, which matters
    when a chunk boundary is suspected of causing a problem.
    """

    is_punctuation: bool = False
    """Whether the service returned this as punctuation rather than a word.

    ElevenLabs returns punctuation and spacing as their own entries. They
    are kept, because dropping them would lose the timing they carry, but
    alignment and scoring skip them.
    """

    is_audio_event: bool = False
    """Whether this is a sound rather than a word, such as laughter.

    Kept apart from punctuation even though both are skipped by alignment,
    because they are skipped for opposite reasons. A comma is a written mark
    that no service can disagree about. Laughter is something that actually
    happened in the room, and the exports and the review window may
    reasonably want to show it where they would never show a stray comma.
    Folding the two together would make that distinction unrecoverable.
    """

    @property
    def is_spoken_word(self) -> bool:
        """Whether this token is a word that services can be compared on.

        The one test alignment and scoring should use. Punctuation and audio
        events are both excluded, so neither has to be remembered separately
        at each of the places that walks a token list.
        """
        return not self.is_punctuation and not self.is_audio_event

    @property
    def has_timing(self) -> bool:
        return self.start is not None and self.end is not None

    @property
    def duration(self) -> float | None:
        if self.start is None or self.end is None:
            return None
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class TokenReference:
    """A pointer from a final word back to the service word it came from."""

    provider: Provider
    index: int


@dataclass(frozen=True)
class AudioSpan:
    """A region of the canonical recording, in seconds from its start."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"An audio span cannot end before it starts: {self.start}-{self.end}")

    @property
    def duration(self) -> float:
        return self.end - self.start

    def padded(self, before: float, after: float, limit: float | None = None) -> "AudioSpan":
        """Return this span with context added either side.

        Used to build the window sent to a service for a second opinion. A
        service given only the disputed word has no sentence to make sense
        of, no way to tell the language, and nothing to identify the speaker
        by, so it does worse than it would on the whole recording. The
        padding is clamped to the recording, so a dispute in the first
        seconds does not ask for audio before the beginning.
        """
        start = max(0.0, self.start - max(0.0, before))
        end = self.end + max(0.0, after)
        if limit is not None:
            end = min(end, limit)
        return AudioSpan(start, max(start, end))

    def overlaps(self, other: "AudioSpan") -> bool:
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True)
class LanguageEvidence:
    """How likely each language is over some span of audio.

    Kept as several numbers rather than one answer because a code-switch
    boundary is genuinely uncertain, and because which services are allowed
    to weigh in on a span depends on how sure we are of its language rather
    than on a decision already taken. Microsoft, for instance, is excluded
    from a span that is confidently Afrikaans but only reduced in weight
    where the language is in doubt.
    """

    scores: dict[Language, float] = field(default_factory=dict)

    @property
    def best(self) -> Language:
        if not self.scores:
            return Language.UNKNOWN
        return max(self.scores.items(), key=lambda item: item[1])[0]

    @property
    def best_score(self) -> float:
        if not self.scores:
            return 0.0
        return max(self.scores.values())

    def score_for(self, language: Language) -> float:
        return self.scores.get(language, 0.0)

    def is_confidently(self, language: Language, threshold: float = 0.75) -> bool:
        return self.best is language and self.best_score >= threshold


@dataclass(frozen=True)
class Candidate:
    """One possible text for a word, and what argues for it.

    The score is worked out by the reconciliation rules, but the parts that
    went into it are kept beside it. A person in the review window needs to
    know *why* a candidate won, and a number on its own does not tell them.
    """

    text: str
    providers: tuple[Provider, ...] = ()
    score: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    """The named factors that multiplied together to make the score."""

    source_tokens: tuple[TokenReference, ...] = ()
    in_vocabulary: bool = False
    """Whether this candidate matches a known name, term or past correction."""


# -- What the application decided ----------------------------------------


@dataclass(frozen=True)
class StatisticsNote:
    """What the service statistics were told about one settled word.

    The statistics are updated every time the review window saves, and a
    save carries the whole transcript. Without a note on each word, every
    save would count every settled word again. The note is what lets a save
    count only what changed: a word with no note is new, a word whose result
    differs from its note has its old result taken out and the new one put
    in, and a word that matches its note is left alone.

    The facts are written once, the first time a person settles the word,
    and never again. They have to be taken then, because a correction
    clears :attr:`FinalToken.text_source`: afterwards the transcript no
    longer says which service was believed, and the statistics are about
    exactly that.

    The vocabulary category is kept as its stored value rather than as the
    vocabulary module's enumeration, because that module imports this one.
    """

    provider: Provider | None = None
    """The service whose word was believed, where one was."""

    language: Language = Language.UNKNOWN
    is_proper_noun: bool = False
    is_numeric: bool = False
    vocabulary_category: str | None = None
    confidence: Confidence = Confidence.UNRESOLVED
    """The word's category before the person looked at it."""

    service_text: str = ""
    """What the services said, which a correction is measured against."""

    rejected_providers: tuple[Provider, ...] = ()
    speaker_provider: Provider | None = None
    service_speaker: str | None = None
    """The speaker label before review, to tell a changed speaker from an unchanged one.

    It stays in the transcript, which already carries every speaker label,
    and is never copied into the statistics.
    """

    counted: bool = False
    """Whether this word's result is in the statistics now."""

    text_corrected: bool = False
    """The counted result: the text was changed from what the services said."""

    speaker_corrected: bool = False
    """The counted result: the speaker was changed from what the service said."""


@dataclass
class FinalToken:
    """One word of the finished transcript.

    The three questions are answered independently, and each answer carries
    its own source and its own confidence. That is the whole point of the
    design, and the reason this class has as many fields as it does.

    It is mutable, unlike the evidence, because it is built up in stages:
    alignment fills in the sources, reconciliation settles the text,
    escalation and adjudication may change it, forced alignment may correct
    the timing, and a person may correct any of it. Each stage leaves the
    provenance behind it intact.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    # -- What was said
    text: str = ""
    normalised_text: str = ""
    """The comparison-only form. Never shown to anyone and never exported."""

    text_source: Provider | None = None
    text_confidence: Confidence = Confidence.UNRESOLVED

    confidence_strength: float | None = None
    """The application's own combined estimate for this word, nought to one.

    This does not retreat from the argument in :class:`Confidence` above.
    That argument is about the numbers the *services* report: ElevenLabs,
    AssemblyAI and Deepgram each publish a figure they each call a
    confidence, the three are measured on scales that do not correspond, two
    of the services publish nothing at all, and averaging any of that would
    manufacture a precision that was never there. All of that still holds,
    and none of those numbers is what this field holds.

    What this field holds is the application's own conclusion, worked out by
    ``confidence.assess_text`` from evidence the application gathered itself:
    how far the services agreed, how clearly the winning reading beat the
    next one, how well the words lined up, whether the language of the span
    is one the deciding service is trusted on, and whether the user has told
    us about the word. It is therefore never to be shown or described as a
    service's confidence, and it is not comparable with one. It *is*
    comparable with the same field on another word, because both were
    reached by the same rules from the same kinds of evidence, and that one
    comparison is the whole reason to save it.

    It is kept beside the category rather than in place of it, because the
    two answer different questions. The category is the decision: leave this
    alone, look at it, look at it now. The strength is where inside that
    decision the word actually fell, which is what lets a thousand words be
    sorted weakest first, and what lets two spellings of the same surname be
    recognised as about equally shaky. Neither can be recovered from the
    other, so both are written down.

    ``None`` means the strength was never worked out: the transcript was
    made before this field existed, or the word came out of a path that does
    no text assessment at all. It emphatically does not mean the word is
    weak. Anything that compares against a threshold must leave such words
    out rather than reading the absence as a low number, because a word
    nobody measured is not a word we know to be bad.
    """

    # -- When it was said
    start: float | None = None
    end: float | None = None
    timing_source: Provider | None = None
    timing_status: TimingStatus = TimingStatus.UNALIGNED
    timing_confidence: Confidence = Confidence.UNRESOLVED

    # -- Who said it
    speaker: str | None = None
    speaker_source: Provider | None = None
    speaker_confidence: Confidence = Confidence.UNRESOLVED

    # -- Language
    language: Language = Language.UNKNOWN
    language_evidence: LanguageEvidence = field(default_factory=LanguageEvidence)

    # -- Where it came from and what else was on offer
    source_tokens: list[TokenReference] = field(default_factory=list)
    source_audio_span: AudioSpan | None = None
    """The region of audio this word is believed to sit in.

    Wider than start-to-end when the exact boundaries are not defensible,
    which is what lets a person hear an unaligned or approximate word
    without the application pretending to know where it begins.
    """

    alignment_status: AlignmentStatus = AlignmentStatus.UNALIGNED
    candidates: list[Candidate] = field(default_factory=list)
    rejected_tokens: list[TokenReference] = field(default_factory=list)
    """Service words that reconciliation threw out, kept but never shown.

    A service that hallucinates a word leaves it here rather than in the
    transcript, so that a pattern of hallucination can be seen later.
    """

    # -- Risk, review and decisions taken
    risk_categories: list[RiskCategory] = field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.SETTLED
    review_reasons: list[ReviewReason] = field(default_factory=list)
    llm_decision: str | None = None
    """What the adjudicating model decided here, if it was consulted."""

    human_corrected: bool = False
    """A person decided something about this word.

    Deliberately broad. Correcting the text sets it, correcting the speaker
    sets it, and confirming or rejecting the timing sets it, because all three
    are a person putting their judgement on the word and all three are reasons
    to stop treating it as something the services alone produced.

    It is emphatically **not** the answer to "was the text replaced". Use
    :attr:`text_corrected` for that. The two were once the same field, and the
    damage that did is worth remembering: confirming a clock made the word
    report that somebody had rewritten it, and left it permanently exempt from
    every replacement rule the project would ever learn, which is precisely
    the case those rules exist for.
    """

    text_corrected: bool = False
    """The text of this word was replaced, by a person or by a rule they accepted.

    True only where :meth:`Transcript.with_correction` was actually given a
    new text and that text differed from what the word said. A correction to
    the speaker alone does not set it, and neither does a decision about the
    timing.

    This is what anything asking "does this word still say what the services
    said" must read. A replacement rule refuses to touch a word that is
    already ``text_corrected``, because a person typing into this transcript
    is a newer and stronger statement than a rule; and re-matching keeps the
    saved strength for such a word, because the 1.0 that a correction confers
    is not a measurement of anything.
    """

    original_text: str | None = None
    """What the word said before a person changed it."""

    statistics_note: StatisticsNote | None = None
    """What the service statistics have counted for this word, if anything."""

    @property
    def confidence(self) -> Confidence:
        """The weakest of the three, which is what the review queue sorts on.

        A missing time counts as "review suggested" here, not "unresolved".
        "Unresolved" tells the reader that nothing was chosen, and a word
        every service heard the same has had its text chosen; only where it
        sits is unknown. Letting the missing time through unchanged rated
        every word of a recording with no word times as unresolved, and hid
        the few real disputes among them. The word still needs review, so it
        stays in the queue, and its text rating still wins when it is worse.

        The test is on the missing start, not on the timing rating alone. A
        person who rejects a word's time keeps its numbers and marks the
        timing unresolved, and that judgement must still count in full.
        """
        timing = self.timing_confidence
        if timing is Confidence.UNRESOLVED and self.start is None:
            timing = Confidence.REVIEW_SUGGESTED
        return Confidence.weakest(self.text_confidence, timing, self.speaker_confidence)

    @property
    def needs_review(self) -> bool:
        return self.review_status is ReviewStatus.PENDING or bool(self.review_reasons)

    @property
    def span(self) -> AudioSpan | None:
        """The word's own span, where it has one."""
        if self.start is None or self.end is None:
            return None
        return AudioSpan(self.start, self.end)

    @property
    def audible_span(self) -> AudioSpan | None:
        """The best region to play to hear this word.

        Falls back to the wider containing span when the word's own
        boundaries are not known, which is exactly when a person most needs
        to hear it.
        """
        return self.span or self.source_audio_span

    def flag(self, reason: ReviewReason) -> None:
        """Mark this word for review, without repeating a reason already there."""
        if reason not in self.review_reasons:
            self.review_reasons.append(reason)
        if self.review_status is ReviewStatus.SETTLED:
            self.review_status = ReviewStatus.PENDING

    def is_uncertain_between(self) -> tuple[str, ...]:
        """The candidate texts still in contention, for an uncertain word.

        Empty unless the word is genuinely unresolved. The exporters use it
        to write ``[UNCERTAIN: 15,000 / 50,000]`` rather than choosing.
        """
        if self.text_confidence is not Confidence.UNRESOLVED:
            return ()
        texts: list[str] = []
        for candidate in self.candidates:
            if candidate.text not in texts:
                texts.append(candidate.text)
        return tuple(texts)


@dataclass
class CleanSegment:
    """A run of readable prose in the optional tidied-up transcript.

    It keeps the identifiers of the verbatim words behind it. Without that
    link the tidy version would be prose floating free of the recording,
    which is precisely what the specification forbids.
    """

    text: str
    source_token_ids: list[str] = field(default_factory=list)
    speaker: str | None = None
    start: float | None = None
    end: float | None = None


@dataclass(frozen=True)
class Speaker:
    """A speaker in the recording, and the name a person gave them."""

    id: str
    """The label the backbone service used, such as "speaker_0"."""

    name: str | None = None
    """What a person called them, once they have said."""

    @property
    def display_name(self) -> str:
        return self.name or f"Speaker {self.id}"


# -- The whole thing -----------------------------------------------------


@dataclass
class RecordingConfiguration:
    """What the user chose for this recording before it was transcribed."""

    afrikaans_enabled: bool = False
    """Whether Afrikaans may occur.

    Off by default, and deliberately explicit. Off means no Afrikaans
    detection is attempted at all and no Afrikaans fallback path exists,
    which removes a whole class of wrong language guesses from recordings
    that never contained any.
    """

    expected_speaker_count: int = 1
    known_speakers: list[str] = field(default_factory=list)
    recording_context: str = ""
    """A sentence or two about the recording, given to the services that
    accept context, and to the adjudicating model."""

    vocabulary_profile_ids: list[str] = field(default_factory=list)


@dataclass
class CanonicalAudio:
    """The one file every timestamp in the transcript refers to.

    Services are sent copies, chunks and windows cut from this. Recording
    exactly what it is matters because a transcript is only meaningful
    against the audio it was made from, and a re-encoded or trimmed file
    would silently shift every timestamp.
    """

    path: str
    original_path: str
    duration: float
    sample_rate: int
    channels: int
    size_bytes: int
    container: str
    is_copy: bool = False
    """Whether this is a converted copy rather than the original file."""


@dataclass(frozen=True)
class ChunkRecord:
    """One request-sized piece of audio sent to one service.

    Chunking is per service, never global: OpenAI's 25 MB limit is no
    reason to cut up what ElevenLabs would happily accept whole. The offset
    is what keeps that from mattering, because it maps every word the
    service returns back onto the canonical timeline.
    """

    provider: Provider
    chunk_index: int
    canonical_start: float
    canonical_end: float
    canonical_offset: float
    """Added to a service's local time to get canonical time."""

    encoded_size_bytes: int
    path: str | None = None
    provider_request_id: str | None = None
    overlap_before: float = 0.0
    """How much of the previous chunk this one repeats, so the duplicated
    words can be dropped deterministically rather than appearing twice."""


@dataclass(frozen=True)
class ProviderRequestRecord:
    """An immutable note of one request to one service, and what came back.

    Written for every request, including failed ones. It is what makes a
    later reprocessing run explainable: services change their models
    underneath a fixed name, and without this there would be no way to tell
    a changed service from a changed application.

    It never holds an API key. Adapters are responsible for stripping
    credentials out of the parameters before they arrive here.
    """

    provider: Provider
    model_identifier: str
    model_version: str | None = None
    request_parameters: dict[str, Any] = field(default_factory=dict)
    language_configuration: str | None = None
    vocabulary_terms: tuple[str, ...] = ()
    chunks: tuple[ChunkRecord, ...] = ()
    raw_response_file: str | None = None
    """The file inside the recording's folder holding the untouched response."""

    processing_seconds: float = 0.0
    started_at: str = ""
    """When the request was made, as an ISO 8601 string in local time."""

    succeeded: bool = True
    error: str | None = None
    purpose: str = "transcription"
    """What the request was for: a full pass, an escalation window, forced
    alignment or adjudication. Escalation sends many small requests, and
    they should not be mistaken for full passes."""


@dataclass
class ProviderResult:
    """Everything one service returned for one recording.

    A failed result is still a result. It carries the error and an empty
    list of words, and the pipeline carries on with the services that did
    answer, because losing Microsoft must not cost the user their
    transcript.
    """

    provider: Provider
    tokens: list[ProviderToken] = field(default_factory=list)
    request: ProviderRequestRecord | None = None
    detected_language: Language = Language.UNKNOWN
    speakers: list[str] = field(default_factory=list)
    error: str | None = None

    raw_response: Any | None = None
    """Exactly what the service sent back, before we touched it.

    The specification requires every provider response to be preserved
    unchanged, for debugging, for reprocessing, for auditing, and for
    telling a changed service from a changed application when a later run
    produces a different answer. An adapter that parses a response into
    tokens and throws the original away makes all of that impossible, and
    the loss is invisible until the day somebody needs it.

    This field is a carrier, not storage. It holds the payload just long
    enough for the pipeline to hand it to the transcript store, which writes
    it to its own file and puts that file's name in the request record. It
    is deliberately **not** written into the transcript itself: the response
    from a long recording is large, and holding a second copy inside a file
    that is loaded every time the review window opens would be wasteful.
    Serialisation therefore skips it, and the file named in
    ``ProviderRequestRecord.raw_response_file`` is where it actually lives.
    """

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def word_tokens(self) -> list[ProviderToken]:
        """The spoken words, without punctuation, spacing or audio events."""
        return [token for token in self.tokens if token.is_spoken_word]

    _by_index: tuple[list[ProviderToken], int, dict[int, ProviderToken]] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    """A lookup from ``ProviderToken.index`` to the token, built when first
    needed. See :meth:`token_at` for why it is keyed the way it is."""

    def token_at(self, index: int) -> ProviderToken | None:
        """The token this service numbered ``index``, or ``None``.

        Escalation planning, dispute building and the review report each
        ask this once for every reference in every column, and a scan of
        the token list for each of those took eight seconds per caller on
        a three-hour recording. The answer comes from a dictionary instead,
        built the first time it is needed.

        This is a mutable dataclass, so the dictionary has to notice when
        the tokens change underneath it. It remembers the very list it was
        built from and that list's length, and is rebuilt when either
        differs: a new list assigned to ``tokens``, or a token added to or
        removed from the old one. Holding the list itself rather than its
        identity means a list freed and replaced by another at the same
        address cannot pass for it. What it cannot notice is a token being
        replaced in place, ``tokens[3] = other``, with the length unchanged;
        nothing in the application does that, and a caller that does must
        assign a fresh list.

        Where two tokens carry the same index, the first in the list wins,
        as it did when this was a scan.
        """
        tokens = self.tokens
        cached = self._by_index
        if cached is None or cached[0] is not tokens or cached[1] != len(tokens):
            by_index: dict[int, ProviderToken] = {}
            for token in tokens:
                by_index.setdefault(token.index, token)
            self._by_index = (tokens, len(tokens), by_index)
            cached = self._by_index
        return cached[2].get(index)


@dataclass
class Transcript:
    """A finished transcript, with everything needed to explain it.

    This is what is written to disk beside the recording, what the review
    window reads and writes, and what the exporters render.
    """

    recording_name: str
    canonical_audio: CanonicalAudio | None = None
    configuration: RecordingConfiguration = field(default_factory=RecordingConfiguration)

    tokens: list[FinalToken] = field(default_factory=list)
    clean_segments: list[CleanSegment] = field(default_factory=list)
    speakers: list[Speaker] = field(default_factory=list)

    provider_results: dict[Provider, ProviderResult] = field(default_factory=dict)
    requests: list[ProviderRequestRecord] = field(default_factory=list)

    reconciliation_version: int = RECONCILIATION_VERSION
    created_at: str = ""
    completed_at: str = ""
    effective_settings: dict[str, Any] = field(default_factory=dict)
    """The non-secret settings this run used, so it can be reproduced."""

    warnings: list[str] = field(default_factory=list)
    """Things that went wrong without stopping the run, such as a service
    that did not answer. The user is told these plainly at the end."""

    stopped: bool = False
    """Whether the run that made this transcript was stopped part way.

    This describes the run in hand only and is never saved: a transcript read
    back from its file is always ``False`` here. It exists so that the runner
    can report an incomplete transcript without matching the wording of the
    warning that says so."""

    @property
    def review_tokens(self) -> list[FinalToken]:
        return [token for token in self.tokens if token.needs_review]

    @property
    def verbatim_text(self) -> str:
        """The plain running text, without speaker labels.

        Words are joined with single spaces, and punctuation returned as its
        own token is attached to the word before it, which is how the
        services that separate them expect it to be put back together.
        """
        parts: list[str] = []
        for token in self.tokens:
            text = token.text
            if not text:
                continue
            if parts and _attaches_to_previous(text):
                parts[-1] = parts[-1] + text
            else:
                parts.append(text)
        return " ".join(parts)

    def speaker_for(self, speaker_id: str | None) -> Speaker | None:
        if speaker_id is None:
            return None
        for speaker in self.speakers:
            if speaker.id == speaker_id:
                return speaker
        return None

    def token_by_id(self, token_id: str) -> FinalToken | None:
        for token in self.tokens:
            if token.id == token_id:
                return token
        return None

    def counts_by_confidence(self) -> dict[Confidence, int]:
        counts = {value: 0 for value in Confidence}
        for token in self.tokens:
            counts[token.confidence] += 1
        return counts

    def with_correction(
        self,
        token_id: str,
        text: str | None = None,
        speaker: str | None = None,
    ) -> "Transcript":
        """Return a copy with one word corrected by a person.

        A person's decision is the strongest evidence there is, so the word
        becomes high confidence and leaves the review queue. What it said
        before is kept, both so the change can be undone and so the
        correction can teach the vocabulary and the provider statistics.

        Correcting the text does not touch the timing or the speaker, and
        correcting the speaker does not touch the text. That separation is
        the same one the whole design rests on.

        Two flags come out of this rather than one, and the difference is the
        whole reason :attr:`FinalToken.text_corrected` exists.
        ``human_corrected`` is set for any correction at all, including one
        that only names the speaker, because in every case a person has put
        their judgement on the word. ``text_corrected`` is set only on the
        branch below that actually replaces the text, so that anything asking
        "does this word still say what the services said" gets an answer to
        the question it asked rather than to a wider one.
        """
        tokens = list(self.tokens)
        for position, token in enumerate(tokens):
            if token.id != token_id:
                continue
            corrected = replace(token)
            if text is not None and text != token.text:
                corrected.original_text = token.original_text or token.text
                corrected.text = text
                corrected.text_confidence = Confidence.HIGH
                # The strength has to move with the category, or the two
                # would tell a person different stories: the word would read
                # as settled and still be dragged back out by anything that
                # thresholds on the number. One is the figure
                # ``confidence.assess_text`` reaches by its own shortcut for
                # a word a person corrected, and it is written out here
                # rather than imported because that module already imports
                # this one.
                corrected.confidence_strength = 1.0
                corrected.text_source = None
                # The comparison form has to be rebuilt from the new text.
                # Left as it was, it would go on describing the word the
                # person has just replaced, and every rule, lookup and
                # grouping decision that compares on this field would match
                # the old spelling on a word that no longer says it. The
                # visible effect is a correction that is quietly made again
                # every time the project is reprocessed.
                corrected.normalised_text = normalise(text)
                # Set here, inside the branch that genuinely rewrote the
                # word, rather than beside ``human_corrected`` below. A word
                # reached by this method for its speaker alone has not been
                # rewritten, and saying it had would put it beyond the reach
                # of every replacement rule for the rest of the project's
                # life.
                corrected.text_corrected = True
            if speaker is not None and speaker != token.speaker:
                corrected.speaker = speaker
                corrected.speaker_confidence = Confidence.HIGH
                corrected.speaker_source = None
            corrected.human_corrected = True
            corrected.review_status = ReviewStatus.CORRECTED
            corrected.review_reasons = []
            tokens[position] = corrected
            break
        return replace(self, tokens=tokens)


#: Punctuation that belongs on the end of the word before it, rather than
#: standing on its own with a space in front.
_TRAILING_PUNCTUATION = frozenset(".,;:!?)]}%…”’")


def _attaches_to_previous(text: str) -> bool:
    return len(text) > 0 and all(character in _TRAILING_PUNCTUATION for character in text)
