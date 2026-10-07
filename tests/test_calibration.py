"""Tests for the provider statistics, the weights built on them and their store."""

from __future__ import annotations

import json

from vox_verbatim.transcription.calibration import (
    CALIBRATION_FILE_NAME,
    DEFAULT_WEIGHT,
    MINIMUM_DIMENSION_EVIDENCE,
    CalibrationStore,
    Dimension,
    Observation,
    ProviderStatistics,
    evidence_phrase,
    reliability_weights,
)
from vox_verbatim.transcription.model import Confidence, Language, Provider
from vox_verbatim.transcription.reconcile import DEFAULT_PROVIDER_RELIABILITY
from vox_verbatim.transcription.vocabulary import TermCategory


def word(
    provider: Provider = Provider.OPENAI,
    language: Language = Language.ENGLISH,
    **extra,
) -> Observation:
    return Observation(provider=provider, language=language, **extra)


def record_words(
    statistics: ProviderStatistics,
    chosen: int,
    corrected: int,
    observation: Observation | None = None,
) -> None:
    """Record ``chosen`` words, of which ``corrected`` were changed by a person."""
    subject = observation or word()
    for index in range(chosen):
        statistics.record_choice(subject, corrected=index < corrected)


def test_a_service_nobody_has_used_sits_at_the_documented_default():
    statistics = ProviderStatistics()

    assert statistics.weight_for(Provider.DEEPGRAM) == DEFAULT_WEIGHT
    assert statistics.weight_for_word(word(Provider.DEEPGRAM)) == DEFAULT_WEIGHT


def test_a_handful_of_corrections_does_not_swing_the_weight():
    """Three bad words is three hard words, not a verdict on the service."""
    statistics = ProviderStatistics()

    record_words(statistics, chosen=3, corrected=3)

    weight = statistics.weight_for(Provider.OPENAI)
    assert weight < DEFAULT_WEIGHT
    assert DEFAULT_WEIGHT - weight < 0.06


def test_a_real_body_of_evidence_does_move_the_weight():
    statistics = ProviderStatistics()

    record_words(statistics, chosen=600, corrected=300)

    assert statistics.weight_for(Provider.OPENAI) < 0.6


def test_being_right_for_a_long_time_raises_the_weight():
    statistics = ProviderStatistics()

    record_words(statistics, chosen=600, corrected=6)

    assert statistics.weight_for(Provider.OPENAI) > DEFAULT_WEIGHT


def test_the_weight_moves_further_the_more_evidence_arrives():
    """The same correction rate, seen more often, is believed more."""
    thin = ProviderStatistics()
    record_words(thin, chosen=10, corrected=5)
    thick = ProviderStatistics()
    record_words(thick, chosen=400, corrected=200)

    assert thick.weight_for(Provider.OPENAI) < thin.weight_for(Provider.OPENAI)


def test_evidence_is_counted_separately_for_each_dimension():
    statistics = ProviderStatistics()

    record_words(statistics, chosen=100, corrected=0, observation=word(language=Language.ENGLISH))
    record_words(
        statistics,
        chosen=100,
        corrected=90,
        observation=word(language=Language.AFRIKAANS),
    )

    english = statistics.weight_for(Provider.OPENAI, Dimension.LANGUAGE, Language.ENGLISH.value)
    afrikaans = statistics.weight_for(Provider.OPENAI, Dimension.LANGUAGE, Language.AFRIKAANS.value)
    assert english > 0.9
    assert afrikaans < 0.5


def test_a_word_is_weighed_by_the_narrowest_dimension_with_evidence():
    statistics = ProviderStatistics()
    numbers = word(is_numeric=True)

    # A service that is fine in general and poor at numbers.
    record_words(statistics, chosen=400, corrected=4)
    record_words(
        statistics,
        chosen=MINIMUM_DIMENSION_EVIDENCE * 2,
        corrected=50,
        observation=numbers,
    )

    assert statistics.weight_for_word(numbers) < statistics.weight_for(Provider.OPENAI)


def test_a_thin_dimension_falls_back_to_what_is_known_overall():
    statistics = ProviderStatistics()
    numbers = word(is_numeric=True)

    record_words(statistics, chosen=400, corrected=4)
    record_words(statistics, chosen=2, corrected=2, observation=numbers)

    assert statistics.weight_for_word(numbers) == statistics.weight_for(Provider.OPENAI)


def test_a_rejected_candidate_is_counted_but_never_weighed():
    """Nobody checked a rejected candidate, so it cannot mark a service down."""
    statistics = ProviderStatistics()

    for _ in range(50):
        statistics.record_rejection(word(Provider.MICROSOFT))

    assert statistics.counts_for(Provider.MICROSOFT).rejected == 50
    assert statistics.weight_for(Provider.MICROSOFT) == DEFAULT_WEIGHT


def test_the_confidence_categories_are_measured_against_what_happened():
    statistics = ProviderStatistics()

    for index in range(100):
        statistics.record_review(Confidence.HIGH, corrected=index < 2)
    for index in range(20):
        statistics.record_review(Confidence.REVIEW_REQUIRED, corrected=index < 12)

    by_category = {found.category: found for found in statistics.confidence_reliability()}
    assert by_category[Confidence.HIGH].correction_rate == 0.02
    assert by_category[Confidence.REVIEW_REQUIRED].correction_rate == 0.6
    assert by_category[Confidence.UNRESOLVED].correction_rate is None
    # The sample size travels with the rate wherever it is reported.
    assert "100 words" in by_category[Confidence.HIGH].summary


def test_repeated_speaker_mistakes_are_kept_as_their_own_category():
    statistics = ProviderStatistics()

    statistics.record_speaker_correction(Provider.ELEVENLABS)
    statistics.record_speaker_correction(Provider.ELEVENLABS)
    statistics.record_speaker_correction(Provider.OPENAI)

    repeated = statistics.repeated_speaker_mistakes()
    assert repeated == {Provider.ELEVENLABS: 2}
    # A speaker mistake is not a misheard word and does not pretend to be one.
    assert statistics.counts_for(Provider.ELEVENLABS).corrected == 0


def test_repeated_numeric_errors_are_kept_as_their_own_category():
    statistics = ProviderStatistics()

    record_words(statistics, chosen=4, corrected=3, observation=word(is_numeric=True))

    assert statistics.repeated_numeric_mistakes() == {Provider.OPENAI: 3}


def test_the_vocabulary_category_of_a_word_is_one_of_the_dimensions():
    statistics = ProviderStatistics()

    statistics.record_choice(word(vocabulary_category=TermCategory.PERSON), corrected=True)

    counts = statistics.counts_for(Provider.OPENAI, Dimension.VOCABULARY, TermCategory.PERSON.value)
    assert counts.chosen == 1
    assert counts.corrected == 1


def test_a_word_with_nothing_known_about_it_lands_only_in_the_overall_record():
    statistics = ProviderStatistics()

    statistics.record_choice(Observation(provider=Provider.OPENAI))

    assert statistics.counts_for(Provider.OPENAI).chosen == 1
    assert statistics.providers[Provider.OPENAI].dimensions == {}


def test_every_reported_rate_carries_the_number_of_words_behind_it():
    statistics = ProviderStatistics()

    record_words(statistics, chosen=3, corrected=0)

    sentences = " ".join(statistics.summary_sentences())
    assert "3 words" in sentences
    assert "very little evidence" in sentences


def test_an_empty_record_says_so_rather_than_showing_a_perfect_score():
    sentences = " ".join(ProviderStatistics().summary_sentences())

    assert "Nothing has been learned yet" in sentences


def test_the_evidence_phrase_grows_with_the_sample():
    assert evidence_phrase(0) == "no evidence at all"
    assert "very little" in evidence_phrase(3)
    assert "good body" in evidence_phrase(5_000)


def test_the_statistics_survive_being_written_and_read_back(tmp_path):
    statistics = ProviderStatistics()
    record_words(statistics, chosen=40, corrected=4, observation=word(is_proper_noun=True))
    statistics.record_rejection(word(Provider.DEEPGRAM))
    statistics.record_speaker_correction(Provider.ELEVENLABS)
    statistics.record_review(Confidence.REVIEW_SUGGESTED, corrected=True)
    store = CalibrationStore(tmp_path / CALIBRATION_FILE_NAME)

    assert store.save(statistics) is True
    loaded = store.load()

    assert loaded.counts_for(Provider.OPENAI).chosen == 40
    assert loaded.counts_for(Provider.OPENAI, Dimension.PROPER_NOUN).corrected == 4
    assert loaded.counts_for(Provider.DEEPGRAM).rejected == 1
    assert loaded.repeated_speaker_mistakes(minimum=1) == {Provider.ELEVENLABS: 1}
    assert loaded.confidence[Confidence.REVIEW_SUGGESTED].corrected == 1
    assert loaded.weight_for(Provider.OPENAI) == statistics.weight_for(Provider.OPENAI)


def test_a_damaged_file_is_ignored_rather_than_stopping_anything(tmp_path):
    path = tmp_path / CALIBRATION_FILE_NAME
    path.write_text("{ this is not json", encoding="utf-8")

    statistics = CalibrationStore(path).load()

    assert statistics.providers == {}
    assert statistics.weight_for(Provider.OPENAI) == DEFAULT_WEIGHT


def test_a_missing_file_simply_means_nothing_has_been_learned(tmp_path):
    statistics = CalibrationStore(tmp_path / "nothing-here.json").load()

    assert statistics.total_words == 0


def test_nonsense_inside_a_readable_file_is_dropped_piece_by_piece(tmp_path):
    path = tmp_path / CALIBRATION_FILE_NAME
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {
                    "openai": {"overall": {"chosen": 10, "corrected": "many", "rejected": -4}},
                    "a-service-we-never-had": {"overall": {"chosen": 5}},
                },
                "confidence": {"high": {"reviewed": 4, "corrected": 1}, "sideways": {}},
            }
        ),
        encoding="utf-8",
    )

    statistics = CalibrationStore(path).load()

    assert list(statistics.providers) == [Provider.OPENAI]
    counts = statistics.counts_for(Provider.OPENAI)
    assert (counts.chosen, counts.corrected, counts.rejected) == (10, 0, 0)
    assert list(statistics.confidence) == [Confidence.HIGH]


def test_a_file_claiming_more_corrections_than_words_is_brought_back_to_earth(tmp_path):
    path = tmp_path / CALIBRATION_FILE_NAME
    path.write_text(
        json.dumps({"providers": {"openai": {"overall": {"chosen": 5, "corrected": 500}}}}),
        encoding="utf-8",
    )

    statistics = CalibrationStore(path).load()

    assert statistics.counts_for(Provider.OPENAI).corrected == 5
    assert 0.0 <= statistics.weight_for(Provider.OPENAI) <= 1.0


def test_a_count_can_be_taken_back_out_without_going_below_zero():
    statistics = ProviderStatistics()
    statistics.record_choice(word(is_numeric=True), corrected=True)

    statistics.record_choice(word(is_numeric=True), corrected=True, amount=-1)
    statistics.record_choice(word(is_numeric=True), corrected=True, amount=-1)

    counts = statistics.counts_for(Provider.OPENAI)
    assert (counts.chosen, counts.corrected) == (0, 0)
    assert statistics.counts_for(Provider.OPENAI, Dimension.NUMERIC).chosen == 0


def test_an_older_file_loses_its_speaker_labels_and_keeps_the_total(tmp_path):
    path = tmp_path / CALIBRATION_FILE_NAME
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {
                    "elevenlabs": {
                        "overall": {"chosen": 3, "corrected": 1, "rejected": 0},
                        "dimensions": {
                            "speaker:Mrs Smith": {"chosen": 3, "corrected": 1, "rejected": 0},
                            "language:en": {"chosen": 3, "corrected": 1, "rejected": 0},
                        },
                        "speaker_mistakes": {"Mrs Smith": 2, "speaker_1": 1},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    store = CalibrationStore(path)

    loaded = store.load()
    store.save(loaded)

    assert loaded.repeated_speaker_mistakes(minimum=1) == {Provider.ELEVENLABS: 3}
    assert loaded.counts_for(Provider.ELEVENLABS, Dimension.LANGUAGE, "en").chosen == 3
    written = path.read_text(encoding="utf-8")
    assert "Mrs Smith" not in written
    assert "speaker_1" not in written


def test_two_stores_writing_to_one_file_keep_both_changes(tmp_path):
    path = tmp_path / CALIBRATION_FILE_NAME
    first = CalibrationStore(path)
    second = CalibrationStore(path)

    assert first.apply(lambda statistics: statistics.record_choice(word(Provider.OPENAI)))
    assert second.apply(lambda statistics: statistics.record_choice(word(Provider.DEEPGRAM)))

    loaded = first.load()
    assert loaded.counts_for(Provider.OPENAI).chosen == 1
    assert loaded.counts_for(Provider.DEEPGRAM).chosen == 1


def test_reliability_weights_keep_the_defaults_where_there_is_no_evidence():
    weights = reliability_weights(ProviderStatistics())

    assert dict(weights.by_provider) == DEFAULT_PROVIDER_RELIABILITY


def test_reliability_weights_move_a_service_by_its_learned_weight():
    statistics = ProviderStatistics()
    record_words(statistics, chosen=100, corrected=50, observation=word(Provider.MICROSOFT))

    weights = reliability_weights(statistics)

    learned = statistics.weight_for(Provider.MICROSOFT)
    expected = DEFAULT_PROVIDER_RELIABILITY[Provider.MICROSOFT] * learned / DEFAULT_WEIGHT
    assert weights.for_provider(Provider.MICROSOFT) == expected
    assert weights.for_provider(Provider.MICROSOFT) < DEFAULT_PROVIDER_RELIABILITY[Provider.MICROSOFT]
    assert weights.for_provider(Provider.OPENAI) == DEFAULT_PROVIDER_RELIABILITY[Provider.OPENAI]


def test_a_file_that_cannot_be_read_is_not_overwritten_by_one_change(tmp_path):
    path = tmp_path / CALIBRATION_FILE_NAME
    path.write_text('{"providers": {"openai": {"overall": {"chosen": 400', encoding="utf-8")
    store = CalibrationStore(path)

    saved = store.apply(lambda statistics: statistics.record_choice(word(Provider.OPENAI)))

    assert saved is False
    assert path.read_text(encoding="utf-8") == '{"providers": {"openai": {"overall": {"chosen": 400'


def test_a_missing_file_is_created_by_the_first_change(tmp_path):
    store = CalibrationStore(tmp_path / CALIBRATION_FILE_NAME)

    assert store.apply(lambda statistics: statistics.record_choice(word(Provider.OPENAI)))
    assert store.load().counts_for(Provider.OPENAI).chosen == 1
