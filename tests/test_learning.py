"""Tests for what the application learns when a person corrects a transcript."""

from __future__ import annotations

import json
from dataclasses import replace

from vox_verbatim.transcription.calibration import Dimension, ProviderStatistics
from vox_verbatim.transcription.learning import (
    LEARNED_PROFILE_ID,
    MistakeCategory,
    count_settled_words,
    extract_corrections,
    learn_from_review,
    note_settled_words,
    repeated_mistakes,
    teaches_a_term,
    term_for,
)
from vox_verbatim.transcription.model import (
    Confidence,
    FinalToken,
    Language,
    Provider,
    ReviewStatus,
    RiskCategory,
    TokenReference,
    Transcript,
)
from vox_verbatim.transcription.vocabulary import (
    TermCategory,
    Vocabulary,
    VocabularyLevel,
    VocabularyProfile,
    VocabularyTerm,
)


_next_id = 0


def word(
    text: str,
    provider: Provider | None = Provider.OPENAI,
    speaker: str | None = "speaker_0",
    language: Language = Language.ENGLISH,
    **extra,
) -> FinalToken:
    global _next_id
    _next_id += 1
    return FinalToken(
        id=f"token-{_next_id}",
        text=text,
        text_source=provider,
        speaker=speaker,
        speaker_source=provider,
        language=language,
        **extra,
    )


def sentence(*texts: str, **extra) -> Transcript:
    """A transcript of one sentence, each word from OpenAI unless said otherwise."""
    return Transcript(recording_name="meeting", tokens=[word(text, **extra) for text in texts])


def corrected(before: Transcript, text: str, into: str) -> Transcript:
    """The same transcript with one word changed, as the review window would."""
    for token in before.tokens:
        if token.text == text:
            return before.with_correction(token.id, text=into)
    raise AssertionError(f"There is no word {text!r} in this transcript.")


def rate(token: FinalToken, level: Confidence) -> None:
    """Give a word one confidence for all three of its parts.

    A token's confidence is the weakest of the three, so setting only the
    text one would leave the word unresolved and the test would be measuring
    the default rather than what it meant to set.
    """
    token.text_confidence = level
    token.timing_confidence = level
    token.speaker_confidence = level


def only_correction(before: Transcript, after: Transcript):
    report = extract_corrections(before, after)
    assert len(report.text_corrections) == 1
    return report.text_corrections[0]


# -- Reading a review ----------------------------------------------------


def test_a_correction_is_read_out_of_the_two_transcripts():
    before = sentence("We", "met", "Fermeulen", "yesterday.")
    after = corrected(before, "Fermeulen", "Vermeulen")

    correction = only_correction(before, after)

    assert correction.wrong_text == "Fermeulen"
    assert correction.right_text == "Vermeulen"
    assert correction.provider is Provider.OPENAI
    assert correction.language is Language.ENGLISH
    assert correction.speaker == "speaker_0"
    assert correction.is_proper_noun is True
    assert correction.at_sentence_start is False


def test_a_word_nobody_touched_is_not_a_correction():
    before = sentence("We", "met", "Vermeulen", "yesterday.")

    report = extract_corrections(before, before)

    assert report.text_corrections == ()
    assert report.is_empty is True


def test_a_change_of_spelling_alone_still_counts_as_a_correction():
    """The bug the vocabulary team hit: spelling is what this exists to learn."""
    before = sentence("We", "met", "Mueller", "yesterday.")
    after = corrected(before, "Mueller", "Müller")

    correction = only_correction(before, after)

    assert (correction.wrong_text, correction.right_text) == ("Mueller", "Müller")


def test_a_number_written_another_way_is_still_a_correction():
    """The two texts sound alike, but the page changed, and a person changed it."""
    before = sentence("about", "twenty-five", "people")
    after = corrected(before, "twenty-five", "25")

    correction = only_correction(before, after)

    assert correction.right_text == "25"
    assert correction.is_numeric is True


def test_the_language_and_the_risk_of_a_word_travel_with_the_correction():
    before = Transcript(
        recording_name="meeting",
        tokens=[
            word("Das", language=Language.GERMAN),
            word("kostet", language=Language.GERMAN),
            word("15,000", language=Language.GERMAN, risk_categories=[RiskCategory.MONEY]),
        ],
    )
    after = corrected(before, "15,000", "50,000")

    correction = only_correction(before, after)

    assert correction.language is Language.GERMAN
    assert correction.risk_categories == (RiskCategory.MONEY,)
    assert correction.is_high_risk is True
    assert correction.category is MistakeCategory.HIGH_RISK


def test_a_change_of_speaker_is_its_own_kind_of_correction():
    before = sentence("Yes", "of", "course")
    after = before.with_correction(before.tokens[0].id, speaker="speaker_2")

    report = extract_corrections(before, after)

    assert report.text_corrections == ()
    assert len(report.speaker_corrections) == 1
    speaker_correction = report.speaker_corrections[0]
    assert speaker_correction.wrong_speaker == "speaker_0"
    assert speaker_correction.right_speaker == "speaker_2"
    assert speaker_correction.provider is Provider.OPENAI
    assert speaker_correction.category is MistakeCategory.SPEAKER


def test_every_word_a_service_was_believed_for_is_counted():
    before = sentence("We", "met", "Fermeulen", "yesterday.")
    after = corrected(before, "Fermeulen", "Vermeulen")

    report = extract_corrections(before, after)

    assert len(report.choices) == 4
    assert sum(1 for choice in report.choices if choice.corrected) == 1


def test_a_candidate_that_lost_is_recorded_as_rejected():
    before = sentence("We", "met", "Vermeulen")
    before.tokens[2].rejected_tokens = [TokenReference(Provider.MICROSOFT, 2)]

    report = extract_corrections(before, before)

    assert len(report.rejections) == 1
    assert report.rejections[0].provider is Provider.MICROSOFT


def test_the_confidence_a_word_carried_is_remembered_for_calibration():
    before = sentence("We", "met", "Fermeulen")
    rate(before.tokens[2], Confidence.REVIEW_REQUIRED)
    rate(before.tokens[0], Confidence.HIGH)
    after = corrected(before, "Fermeulen", "Vermeulen")
    # A word the person looked at and left exactly as it was.
    after.tokens[0].review_status = ReviewStatus.CONFIRMED

    report = extract_corrections(before, after)

    by_id = {found.token_id: found for found in report.reviewed}
    assert len(by_id) == 2
    assert by_id[after.tokens[2].id].confidence is Confidence.REVIEW_REQUIRED
    assert by_id[after.tokens[2].id].corrected is True
    assert by_id[after.tokens[0].id].corrected is False


# -- The judgement: which corrections become terms -----------------------


def test_a_name_becomes_a_term_the_services_will_listen_for():
    before = sentence("We", "met", "Fermeulen", "yesterday.")
    after = corrected(before, "Fermeulen", "Vermeulen")

    correction = only_correction(before, after)

    assert teaches_a_term(correction) is True
    term = term_for(correction)
    assert term is not None
    assert term.text == "Vermeulen"
    assert term.common_misrecognitions == ("Fermeulen",)
    assert correction.category is MistakeCategory.NAME


def test_a_grammar_fix_never_becomes_a_term():
    """Teaching "there" would push the services towards writing it for "their"."""
    before = sentence("Put", "it", "over", "their")
    after = corrected(before, "their", "there")

    correction = only_correction(before, after)

    assert teaches_a_term(correction) is False
    assert term_for(correction) is None
    assert correction.category is MistakeCategory.WORDING


def test_a_capital_at_the_start_of_a_sentence_proves_nothing():
    before = sentence("Write", "the", "answer", "down.")
    after = corrected(before, "Write", "Right")

    correction = only_correction(before, after)

    assert correction.at_sentence_start is True
    assert teaches_a_term(correction) is False


def test_the_same_word_in_the_middle_of_a_sentence_does_prove_something():
    before = sentence("The", "answer", "was", "Write", "down.")
    after = corrected(before, "Write", "Right")

    correction = only_correction(before, after)

    assert correction.at_sentence_start is False
    assert teaches_a_term(correction) is True


def test_an_acronym_becomes_a_term_and_is_labelled_as_one():
    before = sentence("Send", "it", "to", "sars", "today.")
    after = corrected(before, "sars", "SARS")

    term = term_for(only_correction(before, after))

    assert term is not None
    assert term.category is TermCategory.ACRONYM


def test_a_number_is_never_taught_as_a_term():
    before = sentence("about", "fifteen", "thousand")
    after = corrected(before, "fifteen", "50")

    correction = only_correction(before, after)

    assert correction.is_numeric is True
    assert term_for(correction) is None
    assert correction.category is MistakeCategory.NUMBER


def test_an_amount_is_never_taught_as_a_term():
    before = Transcript(
        recording_name="meeting",
        tokens=[word("costs"), word("15,000", risk_categories=[RiskCategory.MONEY])],
    )
    after = corrected(before, "15,000", "50,000")

    assert term_for(only_correction(before, after)) is None


def test_a_rewritten_phrase_is_not_a_term():
    before = sentence("He", "said", "Thefour", "wecan")
    after = corrected(before, "Thefour", "Therefore Malan and Botha")

    assert term_for(only_correction(before, after)) is None


def test_a_corrected_spelling_of_a_name_is_taught_with_its_own_spelling():
    before = sentence("We", "met", "Mueller", "yesterday.")
    after = corrected(before, "Mueller", "Müller")

    term = term_for(only_correction(before, after))

    assert term is not None
    assert term.text == "Müller"
    assert term.common_misrecognitions == ("Mueller",)


def test_the_language_of_the_word_travels_onto_the_term():
    before = Transcript(
        recording_name="meeting",
        tokens=[
            word("Der", language=Language.GERMAN),
            word("Bericht", language=Language.GERMAN),
            word("von", language=Language.GERMAN),
            word("Muehler", language=Language.GERMAN),
        ],
    )
    after = corrected(before, "Muehler", "Müller")

    term = term_for(only_correction(before, after))

    assert term is not None
    assert term.language is Language.GERMAN


# -- Storing what was learned --------------------------------------------


def test_a_review_feeds_the_corrections_the_vocabulary_module_keeps():
    before = sentence("We", "met", "Fermeulen", "yesterday.")
    after = corrected(before, "Fermeulen", "Vermeulen")
    vocabulary = Vocabulary()

    result = learn_from_review(before, after, vocabulary, when="2026-08-17T09:00:00")

    assert len(vocabulary.corrections.corrections) == 1
    stored = vocabulary.corrections.corrections[0]
    assert (stored.wrong_text, stored.right_text) == ("Fermeulen", "Vermeulen")
    assert stored.provider is Provider.OPENAI
    assert vocabulary.corrections.mistakes_by_provider() == {Provider.OPENAI: 1}
    assert [term.text for term in result.taught] == ["Vermeulen"]


def test_the_same_correction_twice_is_counted_rather_than_listed_twice():
    before = sentence("We", "met", "Fermeulen", "yesterday.")
    after = corrected(before, "Fermeulen", "Vermeulen")
    vocabulary = Vocabulary()

    learn_from_review(before, after, vocabulary, when="2026-08-17T09:00:00")
    learn_from_review(before, after, vocabulary, when="2026-08-18T09:00:00")

    assert len(vocabulary.corrections.corrections) == 1
    assert vocabulary.corrections.corrections[0].occurrences == 2
    profile = vocabulary.profile(LEARNED_PROFILE_ID)
    assert profile is not None
    assert len(profile.terms) == 1
    assert profile.terms[0].confirmation_count == 2


def test_the_vetted_terms_are_collected_in_a_list_the_user_can_see():
    before = sentence("Put", "it", "over", "their")
    after = corrected(before, "their", "there")
    vocabulary = Vocabulary()

    result = learn_from_review(before, after, vocabulary)

    # The correction is still evidence, but it is not taught to the services.
    assert len(vocabulary.corrections.corrections) == 1
    assert result.taught == ()
    assert vocabulary.profile(LEARNED_PROFILE_ID) is None


def test_a_word_the_user_already_categorised_keeps_that_category():
    vocabulary = Vocabulary(
        profiles=[
            VocabularyProfile(
                id="acme",
                level=VocabularyLevel.CLIENT,
                name="Acme",
                terms=[VocabularyTerm(text="Vermeulen", category=TermCategory.PERSON)],
            )
        ]
    )
    before = sentence("We", "met", "Fermeulen", "yesterday.")
    after = corrected(before, "Fermeulen", "Vermeulen")

    result = learn_from_review(before, after, vocabulary)

    assert [term.category for term in result.taught] == [TermCategory.PERSON]


def save(
    before: Transcript,
    after: Transcript,
    statistics: ProviderStatistics,
    excluded: frozenset[str] = frozenset(),
) -> Transcript:
    """Count one save the way the review window does, and return what it saves."""
    noted = note_settled_words(before, after, excluded)
    change, counted = count_settled_words(noted)
    change.apply(statistics)
    return counted


def confirmed(before: Transcript, text: str) -> Transcript:
    """The same transcript with one word confirmed as it stands."""
    tokens = []
    for token in before.tokens:
        if token.text == text:
            token = replace(token)
            token.review_status = ReviewStatus.CONFIRMED
        tokens.append(token)
    return replace(before, tokens=tokens)


def test_a_correction_counts_the_settled_word_against_the_service_that_said_it():
    before = sentence("We", "met", "Fermeulen", "yesterday.")
    rate(before.tokens[2], Confidence.REVIEW_REQUIRED)
    statistics = ProviderStatistics()

    save(before, corrected(before, "Fermeulen", "Vermeulen"), statistics)

    counts = statistics.counts_for(Provider.OPENAI)
    # Only the word a person settled is counted. The other three were never
    # looked at, so they are not evidence that the service was right.
    assert (counts.chosen, counts.corrected) == (1, 1)
    assert statistics.counts_for(Provider.OPENAI, Dimension.PROPER_NOUN).corrected == 1
    assert statistics.confidence[Confidence.REVIEW_REQUIRED].corrected == 1


def test_saving_the_same_transcript_again_changes_nothing():
    before = sentence("We", "met", "Fermeulen", "yesterday.")
    statistics = ProviderStatistics()
    saved = save(before, corrected(before, "Fermeulen", "Vermeulen"), statistics)
    first = statistics.to_dict()

    saved_again = save(saved, saved, statistics)

    assert statistics.to_dict() == first
    assert saved_again is saved


def test_a_word_changed_back_to_what_the_service_said_counts_as_right():
    before = sentence("We", "met", "Fermeulen", "yesterday.")
    statistics = ProviderStatistics()
    saved = save(before, corrected(before, "Fermeulen", "Vermeulen"), statistics)

    save(saved, corrected(saved, "Vermeulen", "Fermeulen"), statistics)

    counts = statistics.counts_for(Provider.OPENAI)
    assert (counts.chosen, counts.corrected) == (1, 0)
    assert statistics.confidence[Confidence.UNRESOLVED].reviewed == 1
    assert statistics.confidence[Confidence.UNRESOLVED].corrected == 0


def test_a_confirmed_word_counts_as_right():
    before = sentence("We", "met", "Fermeulen", "yesterday.")
    statistics = ProviderStatistics()

    save(before, confirmed(before, "met"), statistics)

    counts = statistics.counts_for(Provider.OPENAI)
    assert (counts.chosen, counts.corrected) == (1, 0)


def test_pending_words_and_words_a_folder_rule_answered_add_nothing():
    before = sentence("We", "met", "Fermeulen", "yesterday.")
    before.tokens[0].review_status = ReviewStatus.PENDING
    after = corrected(before, "Fermeulen", "Vermeulen")
    statistics = ProviderStatistics()

    saved = save(before, after, statistics, excluded=frozenset({before.tokens[2].id}))

    assert statistics.to_dict() == ProviderStatistics().to_dict()
    assert all(token.statistics_note is None for token in saved.tokens)


def test_a_word_settled_before_notes_existed_stays_uncounted():
    before = sentence("We", "met", "Fermeulen", "yesterday.")
    earlier = corrected(before, "Fermeulen", "Vermeulen")
    statistics = ProviderStatistics()

    # The word was already corrected in the transcript last saved, and it has
    # no note, so which service said it is no longer known.
    save(earlier, earlier, statistics)

    assert statistics.total_words == 0


def test_a_word_sent_back_to_the_queue_has_its_count_taken_out():
    before = sentence("We", "met", "Fermeulen", "yesterday.")
    statistics = ProviderStatistics()
    saved = save(before, confirmed(before, "met"), statistics)
    reopened = replace(saved, tokens=[replace(token) for token in saved.tokens])
    reopened.tokens[1].review_status = ReviewStatus.PENDING

    save(saved, reopened, statistics)

    assert statistics.counts_for(Provider.OPENAI).chosen == 0


def test_a_rejected_candidate_is_counted_once():
    before = sentence("We", "met")
    before.tokens[1].rejected_tokens = [TokenReference(Provider.DEEPGRAM, 0)]
    statistics = ProviderStatistics()

    saved = save(before, confirmed(before, "met"), statistics)
    save(saved, saved, statistics)

    assert statistics.counts_for(Provider.DEEPGRAM).rejected == 1


def test_a_speaker_mistake_is_one_total_with_no_label():
    before = sentence("Yes", "of", "course")
    after = before.with_correction(before.tokens[0].id, speaker="Mrs Smith")
    statistics = ProviderStatistics()

    save(before, after, statistics)

    assert statistics.repeated_speaker_mistakes(minimum=1) == {Provider.OPENAI: 1}
    # It is not counted as a misheard word, because no word was misheard.
    assert statistics.counts_for(Provider.OPENAI).corrected == 0
    written = json.dumps(statistics.to_dict())
    assert "speaker_0" not in written
    assert "Mrs Smith" not in written


def test_the_recurring_corrections_are_grouped_the_way_section_23_asks():
    before = sentence("We", "met", "Fermeulen", "yesterday.")
    after = corrected(before, "Fermeulen", "Vermeulen")
    numbers_before = sentence("about", "fifteen", "people")
    numbers_after = corrected(numbers_before, "fifteen", "50")
    words_before = sentence("Put", "it", "over", "their")
    words_after = corrected(words_before, "their", "there")
    vocabulary = Vocabulary()

    for _ in range(2):
        learn_from_review(before, after, vocabulary)
        learn_from_review(numbers_before, numbers_after, vocabulary)
        learn_from_review(words_before, words_after, vocabulary)

    grouped = repeated_mistakes(vocabulary.corrections.corrections)

    assert [found.right_text for found in grouped[MistakeCategory.NAME]] == ["Vermeulen"]
    assert [found.right_text for found in grouped[MistakeCategory.NUMBER]] == ["50"]
    assert [found.right_text for found in grouped[MistakeCategory.WORDING]] == ["there"]


def test_a_correction_seen_only_once_is_not_yet_a_pattern():
    before = sentence("We", "met", "Fermeulen", "yesterday.")
    after = corrected(before, "Fermeulen", "Vermeulen")
    vocabulary = Vocabulary()

    learn_from_review(before, after, vocabulary)
    grouped = repeated_mistakes(vocabulary.corrections.corrections)

    assert grouped[MistakeCategory.NAME] == []


def test_a_review_that_changed_nothing_teaches_nothing():
    before = sentence("We", "met", "Vermeulen", "yesterday.")
    vocabulary = Vocabulary()

    result = learn_from_review(before, before, vocabulary)

    assert result.recorded == ()
    assert result.taught == ()
    assert vocabulary.corrections.corrections == []
