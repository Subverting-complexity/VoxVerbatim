"""What the application learns when a person corrects a transcript.

A person changing a word while listening to their own recording is the best
evidence this application will ever get. It was produced by somebody who
knows what was said, checking against the audio, with nothing to gain by
being careless. Everything in this module exists to make sure that evidence
is not thrown away the moment the review window closes.

The work is in three parts. The first is reading a transcript before and
after review and working out what actually changed: which word, what it said
before, what it says now, which service produced the wrong one, what
language it was in, whether it was a name or a number, and whether it was
one of the values that must never be guessed at. The second is handing that
to the two places that can use it: the learned corrections the vocabulary
module already keeps, and the provider statistics that the weighting is
built from. The statistics are fed by :func:`note_settled_words` and
:func:`count_settled_words`, which count each word a person settles once
however many times the transcript is saved. The third, and the only one that involves a judgement, is
deciding which corrections should be taught back to the services as terms
to listen out for.

That third part is where this module earns its keep, because teaching the
wrong thing is worse than teaching nothing. Every service accepts a list of
words to listen for, and being told to listen for a word makes it more
likely to write that word down. Told to listen for a client's surname, a
service stops mangling it. Told to listen for "there", because somebody once
corrected "their" to "there", a service starts writing "there" where the
speaker said "their", and the correction has made every future transcript
worse. The list is not free either: it is capped by every service, so a
grammar fix on it displaces a name that would have earned its place.

So the rule is that a correction teaches a term only when the *right text
names something* rather than being an ordinary word of the language. Two
things have to hold.

Nothing may disqualify it. It must not be a number or a high-risk value: an
amount, a date or an account number is a value that happened to be said
once, and priming a service with it invites that value to reappear in a
recording where it was never spoken. It must be at most
:data:`MAXIMUM_TERM_WORDS` words, because a whole rewritten phrase is a
correction to the wording rather than a word the service could not hear.
Every word must be at least two characters, and none of them may be one of
the everyday function words of English, German or Afrikaans, which is where
the homophone confusions live: their and there, its and it's, dan and dann.

And something must positively identify it as a name or a specialised term.
That is one of four signals: it is written in capitals, as an acronym is; it
carries a capital inside it, as "McDonald" and "iPhone" do; it starts with a
capital somewhere other than the start of a sentence, which is what
distinguishes a name from a word that merely began a sentence; or the
pipeline itself flagged that word as a disagreement over a proper name.

The rule has one honest weakness, and it is better written down than
discovered. German capitalises every noun, so in German the capital
distinguishes a name from a verb but not a name from an ordinary noun, and
an everyday German noun will occasionally be taught as a term. The cost of
that is small, because the word genuinely was in the recording and the
service will hear it again; the cost of the opposite mistake, dropping every
German surname, would be much larger. Where a wider signal becomes
available, such as a part-of-speech tagger or the user marking a category
themselves, it belongs here as a fifth signal rather than as a change to
the rest.

One distinction runs underneath all of this and is worth stating on its
own, because getting it backwards has already caused one real bug in the
vocabulary module. Asking whether a service said a word is a question about
*sound*, and the spoken-equivalence comparison answers it: "Mueller" and
"Müller" are the same thing heard. Asking whether a person changed a word,
or whether a term is already on a list, is a question about *text*, and
only an exact comparison answers it: "Mueller" corrected to "Müller" is
precisely the kind of correction this module exists to catch, and a test
built on equivalence would silently discard every one of them. Each
comparison in this module says which of the two questions it is asking.

Corrections that do not become terms are not wasted. They are still recorded
as corrections, still counted against the service that produced them, and
still grouped into the categories section 23 asks for, so that a service
that keeps mishearing numbers or keeps mixing up two speakers can be seen
doing it.

Nothing here depends on Qt or on any provider library, so it can be tested
on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from vox_verbatim.transcription.calibration import Observation, ProviderStatistics
from vox_verbatim.transcription.model import (
    Confidence,
    FinalToken,
    Language,
    Provider,
    ReviewReason,
    ReviewStatus,
    RiskCategory,
    StatisticsNote,
    Transcript,
)
from vox_verbatim.transcription.normalise import read_number
from vox_verbatim.transcription.vocabulary import (
    MINIMUM_CORRECTION_OCCURRENCES,
    LearnedCorrection,
    TermCategory,
    Vocabulary,
    VocabularyIndex,
    VocabularyLevel,
    VocabularyProfile,
    VocabularyTerm,
)

#: The most words a correction may run to and still be taught as a term.
#: Beyond this it is a rewritten phrase rather than a word the services
#: could not hear, and the services are asking for terms, not sentences.
MAXIMUM_TERM_WORDS = 3

#: The profile the vetted terms are collected in. It is one named list
#: rather than a scattering, so the user can open it, see what the
#: application has decided to teach the services, and delete anything they
#: disagree with.
LEARNED_PROFILE_ID = "learned"
LEARNED_PROFILE_NAME = "Learned from corrections"

#: What ends a sentence, for the purpose of telling a name from a word that
#: happened to come first. Closing quotes and brackets are stripped before
#: the test, because a sentence can end inside them.
_SENTENCE_ENDINGS = ".!?…"
_SENTENCE_CLOSERS = "\"')]}»”’"

#: The everyday words of the three supported languages. A correction between
#: two of these is a correction to the grammar or to a homophone, never a
#: word the services failed to hear, and teaching one to a service would
#: push it towards the mistake rather than away from it.
#:
#: The list is short on purpose. It only has to cover the words that get
#: confused with each other, and a long list of ordinary vocabulary would
#: start refusing to teach real terms that happen to look ordinary.
_FUNCTION_WORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does for from had has have
    he her here his how i if in into is it its me my no not of off on or our out she
    should so than that the their theirs them then there these they this those to too
    two us was we were what when where which who whom why will with would you your
    aber am auf aus bei das dass dein den denn der des dem die dir doch du ein eine
    einen er es euch für hier ich ihm ihn ihr im in ist ja kein mehr mein mit nicht
    noch nur oder sein sich sie sind so über um und uns von vor war waren was wer wehr
    wie wir wo zu
    daar dat deur die dit ek en haar het hom hulle hy in is jou jy kan maar met my nie
    om ons op sy te toe uit van vir was wat wie word wors
    """.split()
)


class MistakeCategory(str, Enum):
    """The kinds of mistake a correction records, as section 23 lists them.

    The category does not change how a correction is stored. It is what lets
    a person be told that a service keeps getting numbers wrong, or that two
    speakers keep being swapped, rather than being shown one long list in
    which those patterns are invisible.
    """

    HIGH_RISK = "high_risk"
    """An amount, a date, an identifier: a value that must never be guessed."""

    NUMBER = "number"
    NAME = "name"
    """A name, a company, an acronym or a specialised term."""

    SPEAKER = "speaker"
    """The wrong person was said to be talking."""

    WORDING = "wording"
    """An ordinary word: a grammar fix, a homophone, a different choice of word."""

    @property
    def display_name(self) -> str:
        return _CATEGORY_DISPLAY_NAMES[self]


_CATEGORY_DISPLAY_NAMES: dict[MistakeCategory, str] = {
    MistakeCategory.HIGH_RISK: "A value that must not be guessed",
    MistakeCategory.NUMBER: "A number",
    MistakeCategory.NAME: "A name or specialised term",
    MistakeCategory.SPEAKER: "The wrong speaker",
    MistakeCategory.WORDING: "Ordinary wording",
}


# -- What a review actually changed --------------------------------------


@dataclass(frozen=True)
class TextCorrection:
    """One word a person changed, with everything known about it at the time."""

    token_id: str
    wrong_text: str
    right_text: str
    provider: Provider | None = None
    """The service whose candidate became the wrong word, where that is known."""

    language: Language = Language.UNKNOWN
    speaker: str | None = None
    is_proper_noun: bool = False
    """Whether the word looks like a name, by the rule described in this module."""

    is_numeric: bool = False
    at_sentence_start: bool = False
    """Whether the word opened a sentence, which is why its capital means less."""

    risk_categories: tuple[RiskCategory, ...] = ()
    vocabulary_category: TermCategory | None = None
    """The category of the user's own term this word matches, where it matches one."""

    @property
    def is_high_risk(self) -> bool:
        return bool(self.risk_categories)

    @property
    def category(self) -> MistakeCategory:
        """Which of section 23's categories this correction belongs in.

        High risk is tested before numeric because an amount is both, and a
        person told "a number was corrected" would not act on it in the way
        that "an amount was corrected" deserves.
        """
        if self.is_high_risk:
            return MistakeCategory.HIGH_RISK
        if self.is_numeric:
            return MistakeCategory.NUMBER
        if teaches_a_term(self):
            return MistakeCategory.NAME
        return MistakeCategory.WORDING

    def observation(self) -> Observation | None:
        """This word as something the provider statistics can count."""
        if self.provider is None:
            return None
        return Observation(
            provider=self.provider,
            language=self.language,
            is_proper_noun=self.is_proper_noun,
            is_numeric=self.is_numeric,
            vocabulary_category=self.vocabulary_category,
        )


@dataclass(frozen=True)
class SpeakerCorrection:
    """One word a person moved from one speaker to another."""

    token_id: str
    wrong_speaker: str | None
    right_speaker: str | None
    provider: Provider | None = None
    language: Language = Language.UNKNOWN

    @property
    def category(self) -> MistakeCategory:
        return MistakeCategory.SPEAKER


@dataclass(frozen=True)
class ChoiceOutcome:
    """One word a service was believed for, and whether that held up."""

    observation: Observation
    corrected: bool


@dataclass(frozen=True)
class ReviewedWord:
    """One word a person actually looked at, and what it was rated beforehand.

    This is what the confidence categories are calibrated against. Only the
    words somebody looked at can be counted, because a word nobody checked
    is not evidence that it was right.
    """

    token_id: str
    confidence: Confidence
    corrected: bool


@dataclass(frozen=True)
class CorrectionReport:
    """Everything one review taught, before any of it has been stored."""

    text_corrections: tuple[TextCorrection, ...] = ()
    speaker_corrections: tuple[SpeakerCorrection, ...] = ()
    choices: tuple[ChoiceOutcome, ...] = ()
    rejections: tuple[Observation, ...] = ()
    reviewed: tuple[ReviewedWord, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (self.text_corrections or self.speaker_corrections or self.reviewed)

    def by_category(self) -> dict[MistakeCategory, int]:
        """How many corrections of each kind this review produced."""
        counts = {category: 0 for category in MistakeCategory}
        for correction in self.text_corrections:
            counts[correction.category] += 1
        counts[MistakeCategory.SPEAKER] += len(self.speaker_corrections)
        return counts


@dataclass(frozen=True)
class LearningResult:
    """What a review changed in the stored knowledge."""

    report: CorrectionReport
    recorded: tuple[LearnedCorrection, ...] = ()
    """The corrections as they now stand, counts included."""

    taught: tuple[VocabularyTerm, ...] = ()
    """The terms the services will be told to listen for next time."""


# -- Reading a review ----------------------------------------------------


def extract_corrections(
    before: Transcript,
    after: Transcript,
    vocabulary: VocabularyIndex | None = None,
) -> CorrectionReport:
    """Work out what a person changed between two versions of a transcript.

    Words are matched by their identifier rather than by position, because a
    review changes what a word says and never which word it is. A word that
    appears in only one of the two transcripts is ignored: there is no pair
    to compare, and guessing at one would invent a correction nobody made.

    Any change to the text counts, including one that changes only the
    spelling. See :func:`_text_changed` for why that has to be so.
    """
    before_tokens = {token.id: token for token in before.tokens}
    sentence_starts = _sentence_starts(before.tokens)

    text_corrections: list[TextCorrection] = []
    speaker_corrections: list[SpeakerCorrection] = []
    choices: list[ChoiceOutcome] = []
    rejections: list[Observation] = []
    reviewed: list[ReviewedWord] = []

    for token in after.tokens:
        original = before_tokens.get(token.id)
        if original is None:
            continue

        language = token.language if token.language is not Language.UNKNOWN else original.language
        at_sentence_start = sentence_starts.get(token.id, False)
        category = _vocabulary_category(vocabulary, token.text)
        text_changed = _text_changed(original.text, token.text)

        if text_changed:
            text_corrections.append(
                TextCorrection(
                    token_id=token.id,
                    wrong_text=original.text,
                    right_text=token.text,
                    provider=original.text_source,
                    language=language,
                    speaker=token.speaker or original.speaker,
                    is_proper_noun=_looks_like_a_name(token.text, at_sentence_start)
                    or ReviewReason.PROPER_NAME_DISAGREEMENT in original.review_reasons,
                    is_numeric=_is_numeric(token.text) or _is_numeric(original.text),
                    at_sentence_start=at_sentence_start,
                    risk_categories=tuple(original.risk_categories or token.risk_categories),
                    vocabulary_category=category,
                )
            )

        speaker_changed = (token.speaker or None) != (original.speaker or None)
        if speaker_changed:
            speaker_corrections.append(
                SpeakerCorrection(
                    token_id=token.id,
                    wrong_speaker=original.speaker,
                    right_speaker=token.speaker,
                    provider=original.speaker_source,
                    language=language,
                )
            )

        observation = _observation(original, token, language, at_sentence_start, category)
        if observation is not None:
            choices.append(ChoiceOutcome(observation=observation, corrected=text_changed))
        for reference in original.rejected_tokens:
            rejections.append(
                Observation(
                    provider=reference.provider,
                    language=language,
                    is_proper_noun=_looks_like_a_name(original.text, at_sentence_start),
                    is_numeric=_is_numeric(original.text),
                    vocabulary_category=category,
                )
            )

        if token.review_status in (ReviewStatus.CORRECTED, ReviewStatus.CONFIRMED):
            reviewed.append(
                ReviewedWord(
                    token_id=token.id,
                    confidence=original.confidence,
                    corrected=text_changed or speaker_changed,
                )
            )

    return CorrectionReport(
        text_corrections=tuple(text_corrections),
        speaker_corrections=tuple(speaker_corrections),
        choices=tuple(choices),
        rejections=tuple(rejections),
        reviewed=tuple(reviewed),
    )


# -- The judgement: which corrections are worth teaching -----------------


def teaches_a_term(correction: TextCorrection) -> bool:
    """Whether this correction should be taught to the services as a term.

    The reasoning is set out in full in the module docstring. In short: only
    when the corrected text names something, and never when it is an
    ordinary word, a number or a value that must not be guessed.
    """
    words = correction.right_text.split()
    if not words or len(words) > MAXIMUM_TERM_WORDS:
        return False
    if correction.is_numeric or correction.is_high_risk:
        return False
    if not _is_teachable_shape(words):
        return False
    return _looks_like_a_name(correction.right_text, correction.at_sentence_start) or (
        correction.is_proper_noun and not correction.at_sentence_start
    )


def term_for(correction: TextCorrection) -> VocabularyTerm | None:
    """Turn a correction into a term, or return ``None`` if it should not be one.

    The wrong text is kept on the term as a known misrecognition, which is
    the shape the vocabulary module already understands: it lets
    reconciliation recognise the wrong spelling as a mistake it has seen
    before rather than as an ordinary rival.

    Only the acronym category is filled in. Telling a person's surname from a
    company name from a product name cannot be done from one word, and a
    guess written into the category would look like something the
    application knows.
    """
    if not teaches_a_term(correction):
        return None
    text = correction.right_text.strip()
    return VocabularyTerm(
        text=text,
        category=TermCategory.ACRONYM if _is_acronym(text) else correction.vocabulary_category,
        language=correction.language if correction.language is not Language.UNKNOWN else None,
        common_misrecognitions=(correction.wrong_text.strip(),),
        confirmation_count=1,
    )


# -- Storing what was learned --------------------------------------------


def learn_from_review(
    before: Transcript,
    after: Transcript,
    vocabulary: Vocabulary,
    when: str | None = None,
    profile_id: str = LEARNED_PROFILE_ID,
) -> LearningResult:
    """Read a review and store everything it taught.

    The corrections go into the learned corrections the vocabulary module
    already keeps, so that there is one record of them and not two. The
    vetted terms go into a profile of their own, which is what carries the
    category and the language that the vocabulary's own automatic pass
    cannot know.

    That profile is the reason a caller who uses it should ask for its terms
    with ``include_corrections=False``. The vocabulary module offers every
    recurring correction to the services on its own, grammar fixes included,
    which is the behaviour this module exists to improve on; taking both
    would put back exactly what the rule above filtered out.

    It does not touch the provider statistics. It compares two whole
    transcripts, so it would count every word in them, including the ones
    nobody looked at; :func:`count_settled_words` counts only what a person
    settled.
    """
    index = VocabularyIndex(_profile_terms(vocabulary))
    report = extract_corrections(before, after, index)

    recorded: list[LearnedCorrection] = []
    for correction in report.text_corrections:
        stored = vocabulary.corrections.record(
            wrong_text=correction.wrong_text,
            right_text=correction.right_text,
            provider=correction.provider,
            when=when,
        )
        if stored is not None:
            recorded.append(stored)

    taught = _teach(vocabulary, report, profile_id)
    return LearningResult(report=report, recorded=tuple(recorded), taught=tuple(taught))


# -- Counting each settled word once -------------------------------------

#: The review states in which a person has decided about a word.
_SETTLED = (ReviewStatus.CORRECTED, ReviewStatus.CONFIRMED)


@dataclass(frozen=True)
class StatisticsChange:
    """What one save adds to and takes out of the provider statistics.

    It is a change rather than a finished set of statistics, so that it can
    be applied to the file as it stands on disk at the moment of writing.
    See :meth:`~vox_verbatim.transcription.calibration.CalibrationStore.apply`.
    """

    removed: tuple[StatisticsNote, ...] = ()
    """Results counted earlier that no longer hold, to be taken out."""

    added: tuple[StatisticsNote, ...] = ()
    """Results to be counted now."""

    @property
    def is_empty(self) -> bool:
        return not (self.removed or self.added)

    def apply(self, statistics: ProviderStatistics) -> None:
        for note in self.removed:
            _count(statistics, note, -1)
        for note in self.added:
            _count(statistics, note, 1)


def note_settled_words(
    before: Transcript,
    after: Transcript,
    excluded: frozenset[str] | set[str] = frozenset(),
    vocabulary: VocabularyIndex | None = None,
) -> Transcript:
    """Return ``after`` with a statistics note on each newly settled word.

    ``before`` is the transcript as it was saved last time and ``after`` the
    one about to be saved. A word gets a note in the save where it goes from
    unsettled to corrected or confirmed. That is the one moment both facts
    the statistics need are still available: ``before`` says which service
    was believed and how confident the word was, and ``after`` says what
    the person decided.

    Three kinds of word get no note, and so are never counted:

    * a word that is not settled, such as one still pending;
    * a word whose identifier is in ``excluded``, which the caller uses for
      the words a folder replacement rule answered: nobody listened to them,
      and the decision behind the rule was counted when it was made;
    * a word already settled in ``before`` with no note, which is a word
      settled before notes existed. Its service is no longer known, and a
      guess would be counted as evidence.

    A word keeps its note from then on. :func:`count_settled_words` reads it.
    """
    before_tokens = {token.id: token for token in before.tokens}
    sentence_starts = _sentence_starts(before.tokens)
    tokens = list(after.tokens)
    changed = False
    for position, token in enumerate(tokens):
        if (
            token.statistics_note is not None
            or token.review_status not in _SETTLED
            or token.id in excluded
        ):
            continue
        original = before_tokens.get(token.id)
        if original is None or original.review_status in _SETTLED:
            continue
        noted = replace(token)
        noted.statistics_note = _note_for(
            original, token, sentence_starts.get(token.id, False), vocabulary
        )
        tokens[position] = noted
        changed = True
    return replace(after, tokens=tokens) if changed else after


def count_settled_words(transcript: Transcript) -> tuple[StatisticsChange, Transcript]:
    """Work out what the statistics must change to match this transcript.

    Each noted word is compared with what its note says was counted. A word
    not counted yet is added. A word whose result has changed has its old
    result taken out and the new one put in, so a word corrected and then
    changed back to what the service said ends up counted once, as right.
    A word that is no longer settled has its result taken out. A word whose
    result is unchanged is left alone, which is why saving the same
    transcript twice changes nothing.

    Returns the change and the transcript with every note brought up to
    date. The caller saves that transcript only if the change was saved:
    otherwise it saves the one it passed in, and the words stay uncounted
    until a save that works.
    """
    removed: list[StatisticsNote] = []
    added: list[StatisticsNote] = []
    tokens = list(transcript.tokens)
    changed = False
    for position, token in enumerate(tokens):
        note = token.statistics_note
        if note is None:
            continue
        if token.review_status in _SETTLED:
            current = replace(
                note,
                counted=True,
                text_corrected=_text_changed(note.service_text, token.text),
                speaker_corrected=(token.speaker or None) != (note.service_speaker or None),
            )
        else:
            current = replace(note, counted=False, text_corrected=False, speaker_corrected=False)
        if current == note:
            continue
        if note.counted:
            removed.append(note)
        if current.counted:
            added.append(current)
        updated = replace(token)
        updated.statistics_note = current
        tokens[position] = updated
        changed = True
    change = StatisticsChange(removed=tuple(removed), added=tuple(added))
    return change, (replace(transcript, tokens=tokens) if changed else transcript)


def _note_for(
    original: FinalToken,
    settled: FinalToken,
    at_sentence_start: bool,
    vocabulary: VocabularyIndex | None,
) -> StatisticsNote:
    """The facts about one word that the statistics are keyed on."""
    language = settled.language if settled.language is not Language.UNKNOWN else original.language
    category = _vocabulary_category(vocabulary, settled.text)
    return StatisticsNote(
        provider=original.text_source,
        language=language,
        is_proper_noun=_looks_like_a_name(settled.text, at_sentence_start),
        is_numeric=_is_numeric(settled.text) or _is_numeric(original.text),
        vocabulary_category=category.value if category is not None else None,
        confidence=original.confidence,
        service_text=original.original_text or original.text,
        rejected_providers=tuple(reference.provider for reference in original.rejected_tokens),
        speaker_provider=original.speaker_source,
        service_speaker=original.speaker,
    )


def _count(statistics: ProviderStatistics, note: StatisticsNote, amount: int) -> None:
    """Add one word's counted result to the statistics, or take it out."""
    category = _term_category(note.vocabulary_category)
    if note.provider is not None:
        statistics.record_choice(
            _note_observation(note, note.provider, category),
            corrected=note.text_corrected,
            amount=amount,
        )
    for provider in note.rejected_providers:
        statistics.record_rejection(_note_observation(note, provider, category), amount=amount)
    if note.speaker_corrected and note.speaker_provider is not None:
        statistics.record_speaker_correction(note.speaker_provider, amount=amount)
    statistics.record_review(
        note.confidence, note.text_corrected or note.speaker_corrected, amount=amount
    )


def _note_observation(
    note: StatisticsNote, provider: Provider, category: TermCategory | None
) -> Observation:
    return Observation(
        provider=provider,
        language=note.language,
        is_proper_noun=note.is_proper_noun,
        is_numeric=note.is_numeric,
        vocabulary_category=category,
    )


def _term_category(value: str | None) -> TermCategory | None:
    if value is None:
        return None
    try:
        return TermCategory(value)
    except ValueError:
        return None


def repeated_mistakes(
    corrections: list[LearnedCorrection],
    minimum: int = MINIMUM_CORRECTION_OCCURRENCES,
) -> dict[MistakeCategory, list[LearnedCorrection]]:
    """Group the recurring corrections into section 23's categories.

    This works from the stored corrections rather than from a live review,
    so it has the two texts and nothing else: no sentence position, no risk
    categories, no language. A word is therefore judged as though it were
    standing mid-sentence, which is the reading that matters for a word seen
    repeatedly. It is a report for a person to read, not evidence anything
    is weighed on.
    """
    grouped: dict[MistakeCategory, list[LearnedCorrection]] = {
        category: [] for category in MistakeCategory
    }
    for correction in corrections:
        if correction.occurrences < minimum:
            continue
        standing = TextCorrection(
            token_id="",
            wrong_text=correction.wrong_text,
            right_text=correction.right_text,
            provider=correction.provider,
            is_numeric=_is_numeric(correction.right_text) or _is_numeric(correction.wrong_text),
        )
        grouped[standing.category].append(correction)
    return grouped


# -- The details ---------------------------------------------------------


def _teach(
    vocabulary: Vocabulary,
    report: CorrectionReport,
    profile_id: str,
) -> list[VocabularyTerm]:
    """Add the vetted terms to the learned profile, merging repeats.

    A term already in the profile has its confirmation count raised rather
    than being added again, because the count is what decides which terms
    survive when a service will not take the whole list, and a term the user
    has now corrected twice deserves to outrank one they corrected once.
    """
    terms = [term for term in (term_for(c) for c in report.text_corrections) if term is not None]
    if not terms:
        return []
    profile = vocabulary.profile(profile_id)
    if profile is None:
        profile = VocabularyProfile(
            id=profile_id,
            level=VocabularyLevel.GLOBAL,
            name=LEARNED_PROFILE_NAME,
        )
        vocabulary.profiles.append(profile)

    taught: list[VocabularyTerm] = []
    for term in terms:
        position = _position_of(profile.terms, term.text)
        if position is None:
            profile.terms.append(term)
            taught.append(term)
            continue
        existing = profile.terms[position]
        misrecognitions = list(existing.common_misrecognitions)
        for wrong in term.common_misrecognitions:
            if wrong and wrong not in misrecognitions:
                misrecognitions.append(wrong)
        merged = VocabularyTerm(
            text=existing.text,
            category=existing.category or term.category,
            language=existing.language or term.language,
            pronunciation_hints=existing.pronunciation_hints,
            common_misrecognitions=tuple(misrecognitions),
            confirmation_count=existing.confirmation_count + 1,
        )
        profile.terms[position] = merged
        taught.append(merged)
    return taught


def _position_of(terms: list[VocabularyTerm], text: str) -> int | None:
    """Where this exact term already sits on the list, if it sits there at all.

    Another question about text rather than about sound. Two terms that
    sound the same and are spelled differently are two terms: a user whose
    list holds both "Müller" and "Mueller" has drawn a distinction, and
    merging them here would quietly undo it.
    """
    wanted = _stored_form(text)
    for position, term in enumerate(terms):
        if _stored_form(term.text) == wanted:
            return position
    return None


def _stored_form(text: str) -> str:
    """The form two stored strings are compared on: space and capitals aside.

    Deliberately the same rule the vocabulary module uses for its own stored
    strings. Capitals are set aside because a term written "bosch" and one
    written "Bosch" are one term written twice, and casefolding leaves the
    German letters exactly where they are, which is what keeps "Müller" and
    "Mueller" apart.
    """
    return text.strip().casefold()


def _profile_terms(vocabulary: Vocabulary) -> list[VocabularyTerm]:
    """Every term the user has written down, whichever list it sits on.

    The learned corrections are left out. What this feeds is the lookup that
    labels a corrected word with the category the user chose for it, and a
    correction has no category to give.
    """
    return [term for profile in vocabulary.profiles for term in profile.terms]


def _vocabulary_category(index: VocabularyIndex | None, text: str) -> TermCategory | None:
    """The category of the user's own term this word matches, where it matches one.

    This one *is* a question about sound, which is why the index is asked
    rather than the strings compared: a word a service wrote as "van der
    merwe" belongs to the term the user wrote as "Van Der Merwe", and the
    category the user gave it applies either way.
    """
    if index is None:
        return None
    found = index.match(text)
    return found.term.category if found is not None else None


def _observation(
    original: FinalToken,
    corrected: FinalToken,
    language: Language,
    at_sentence_start: bool,
    category: TermCategory | None,
) -> Observation | None:
    """What one word looked like to the service that was believed for it.

    The provider comes from the transcript as it was before review. A
    corrected word no longer names a service as its source, and rightly so,
    but the statistics need to know which service was believed when the
    mistake was made.
    """
    if original.text_source is None:
        return None
    return Observation(
        provider=original.text_source,
        language=language,
        is_proper_noun=_looks_like_a_name(corrected.text, at_sentence_start),
        is_numeric=_is_numeric(corrected.text) or _is_numeric(original.text),
        vocabulary_category=category,
    )


def _text_changed(before_text: str, after_text: str) -> bool:
    """Whether a person really changed this word.

    This asks a question about text, not about sound, and the two must not
    be confused. The spoken-equivalence comparison exists to decide whether
    two services heard the same thing, and by that comparison "Mueller" and
    "Müller" are identical. Using it here would throw away every correction
    of a spelling, which is the single thing this module exists to learn.

    So the test is exact once the surrounding space is trimmed, which is the
    same test the learned corrections apply before storing one. Anything
    looser would report corrections that the store then silently refuses,
    and the two halves would disagree about what had been learned.
    """
    return before_text.strip() != after_text.strip()


def _sentence_starts(tokens: list[FinalToken]) -> dict[str, bool]:
    """Which words open a sentence, so their capital can be discounted.

    Punctuation returned as its own word is skipped when looking backwards,
    but a full stop still ends the sentence, which is why the test is on the
    text rather than on the position.
    """
    starts: dict[str, bool] = {}
    opening = True
    for token in tokens:
        text = token.text.strip()
        if not text:
            starts[token.id] = opening
            continue
        starts[token.id] = opening
        opening = _ends_a_sentence(text)
    return starts


def _ends_a_sentence(text: str) -> bool:
    stripped = text.rstrip(_SENTENCE_CLOSERS)
    return bool(stripped) and stripped[-1] in _SENTENCE_ENDINGS


def _is_numeric(text: str) -> bool:
    """Whether this word is a number, written as digits or as words."""
    if any(character.isdigit() for character in text):
        return True
    return read_number(text) is not None


def _is_acronym(text: str) -> bool:
    letters = [character for character in text if character.isalpha()]
    return len(letters) >= 2 and all(character.isupper() for character in letters)


def _has_internal_capital(word: str) -> bool:
    return any(character.isupper() for character in word[1:])


def _looks_like_a_name(text: str, at_sentence_start: bool) -> bool:
    """Whether the shape of this text says it names something.

    A capital at the start of a sentence says nothing at all, which is the
    whole reason the sentence position is carried this far.
    """
    words = text.split()
    if not words:
        return False
    if _is_acronym(text):
        return True
    if any(_has_internal_capital(word) for word in words):
        return True
    if at_sentence_start:
        # Only a capital on a later word can mean anything here. The first
        # word would be capitalised whatever it was.
        return any(word[:1].isupper() for word in words[1:])
    return words[0][:1].isupper()


def _is_teachable_shape(words: list[str]) -> bool:
    """Whether these words could be a term at all, before asking what they look like.

    The words are compared with the function-word list by case alone, and
    not through the comparison module. That module reads "two" as the digit
    2 and expands "its" to "it is", both of which are right for comparing
    what two services heard and wrong for asking whether a word is an
    everyday one.
    """
    for word in words:
        stripped = "".join(character for character in word if character.isalnum())
        if len(stripped) < 2:
            return False
        if stripped.casefold() in _FUNCTION_WORDS:
            return False
    return True
