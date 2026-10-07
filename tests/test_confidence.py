"""Tests for turning the separate confidence signals into one category.

These are all pure: signals go in, a category and its reasons come out. That
is the point of keeping this module apart from the reconciliation rules, and
it is what will let the calibration work compare the thresholds against real
corrections without having to build a whole recording first.
"""

from __future__ import annotations

from vox_verbatim.transcription.confidence import (
    ASSUMED_ACOUSTIC_CONFIDENCE,
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_ACOUSTIC_THRESHOLD,
    REVIEW_REQUIRED_THRESHOLD,
    REVIEW_SUGGESTED_THRESHOLD,
    STRENGTH_NOT_MEASURED,
    WEAK_ALIGNMENT_THRESHOLD,
    TextSignals,
    acoustic_confidence,
    assess_text,
    best_acoustic_confidence,
    category_for,
    speaker_confidence,
    strength_display,
    strength_percentage,
    timing_confidence,
)
from vox_verbatim.transcription.model import (
    AlignmentStatus,
    Confidence,
    Provider,
    ProviderToken,
    ReviewReason,
    TimingStatus,
)


def token(
    provider: Provider = Provider.ELEVENLABS,
    log_probability: float | None = None,
    confidence: float | None = None,
) -> ProviderToken:
    return ProviderToken(
        provider=provider,
        index=0,
        text="contract",
        log_probability=log_probability,
        confidence=confidence,
    )


def test_a_log_probability_becomes_the_probability_it_came_from() -> None:
    value = acoustic_confidence(token(log_probability=-0.05))
    assert value is not None
    assert 0.94 < value < 0.96


def test_a_plain_confidence_is_taken_as_it_stands() -> None:
    assert acoustic_confidence(token(confidence=0.4)) == 0.4


def test_a_service_that_measures_nothing_says_nothing() -> None:
    # Not zero. OpenAI and Microsoft report no confidence at all, and
    # treating their silence as doubt would mark down two of the three
    # default services on every word of every recording.
    assert acoustic_confidence(token(provider=Provider.OPENAI)) is None


def test_the_best_measurement_is_used_and_its_owner_named() -> None:
    tokens = [
        token(provider=Provider.OPENAI),
        token(provider=Provider.DEEPGRAM, confidence=0.6),
        token(provider=Provider.ELEVENLABS, log_probability=-0.02),
    ]
    value, source = best_acoustic_confidence(tokens)
    assert source is Provider.ELEVENLABS
    assert value is not None and value > 0.9


def test_nothing_measured_at_all_is_reported_as_such() -> None:
    assert best_acoustic_confidence([token(provider=Provider.OPENAI)]) == (None, None)


def test_the_four_categories_sit_where_the_thresholds_say() -> None:
    assert category_for(HIGH_CONFIDENCE_THRESHOLD) is Confidence.HIGH
    assert category_for(REVIEW_SUGGESTED_THRESHOLD) is Confidence.REVIEW_SUGGESTED
    assert category_for(REVIEW_REQUIRED_THRESHOLD) is Confidence.REVIEW_REQUIRED
    assert category_for(REVIEW_REQUIRED_THRESHOLD - 0.01) is Confidence.UNRESOLVED


def test_strong_undisputed_evidence_is_high_confidence() -> None:
    signals = TextSignals(
        support=0.95,
        runner_up=0.0,
        acoustic=0.93,
        alignment_quality=1.0,
        agreeing_providers=3,
        language_support=1.0,
    )
    assert assess_text(signals).category is Confidence.HIGH


def test_a_near_tie_is_not_high_confidence_however_strong_both_readings_are() -> None:
    signals = TextSignals(
        support=0.95,
        runner_up=0.93,
        acoustic=0.9,
        alignment_quality=1.0,
        contesting_providers=1,
    )
    assert assess_text(signals).category is not Confidence.HIGH


def test_a_missing_acoustic_measurement_is_assumed_rather_than_punished() -> None:
    measured = assess_text(
        TextSignals(support=0.9, acoustic=ASSUMED_ACOUSTIC_CONFIDENCE)
    )
    unmeasured = assess_text(TextSignals(support=0.9, acoustic=None))
    assert measured.strength == unmeasured.strength


def test_a_service_saying_it_was_unsure_raises_that_reason() -> None:
    assessment = assess_text(
        TextSignals(support=0.9, acoustic=LOW_ACOUSTIC_THRESHOLD - 0.1)
    )
    assert ReviewReason.LOW_ACOUSTIC_CONFIDENCE in assessment.reasons


def test_weak_alignment_raises_its_own_reason() -> None:
    assessment = assess_text(
        TextSignals(support=0.9, alignment_quality=WEAK_ALIGNMENT_THRESHOLD - 0.1)
    )
    assert ReviewReason.WEAK_ALIGNMENT in assessment.reasons


def test_a_contested_word_raises_provider_disagreement() -> None:
    assessment = assess_text(TextSignals(support=0.8, runner_up=0.4, contesting_providers=2))
    assert ReviewReason.PROVIDER_DISAGREEMENT in assessment.reasons


def test_a_high_risk_word_the_services_disagreed_about_is_marked_down() -> None:
    plain = assess_text(TextSignals(support=0.9, runner_up=0.2, contesting_providers=1))
    risky = assess_text(
        TextSignals(support=0.9, runner_up=0.2, contesting_providers=1, is_high_risk=True)
    )
    assert risky.strength < plain.strength


def test_agreement_on_a_high_risk_word_is_still_agreement() -> None:
    plain = assess_text(TextSignals(support=0.9, acoustic=0.9))
    risky = assess_text(TextSignals(support=0.9, acoustic=0.9, is_high_risk=True))
    assert risky.strength == plain.strength


def test_a_word_the_user_told_us_about_is_helped() -> None:
    plain = assess_text(TextSignals(support=0.8, acoustic=0.8))
    known = assess_text(TextSignals(support=0.8, acoustic=0.8, in_vocabulary=True))
    assert known.strength > plain.strength


def test_a_refusal_to_decide_cannot_be_talked_up_by_a_strong_signal() -> None:
    assessment = assess_text(
        TextSignals(support=0.99, acoustic=0.99, alignment_quality=1.0, unresolved=True)
    )
    assert assessment.category is Confidence.UNRESOLVED


def test_a_person_who_corrected_the_word_ends_the_argument() -> None:
    assessment = assess_text(
        TextSignals(support=0.1, acoustic=0.1, unresolved=True, human_corrected=True)
    )
    assert assessment.category is Confidence.HIGH
    # The strength has to move with the category. Left at what the signals
    # argued, it would keep pulling a word its owner has already settled back
    # into anything that sorts or filters on the number.
    assert assessment.strength == 1.0


def test_the_category_is_the_one_the_strength_actually_falls_into() -> None:
    """The saved number and the saved category must not contradict each other.

    Both are written on to the word and both are read afterwards, so a word
    marked high confidence carrying a strength below the high-confidence
    threshold would be a word the review window left alone and the
    low-confidence sweep pulled up, with nothing on either screen to explain
    the difference. The two shortcuts that can break this are checked
    elsewhere: the person's own correction just above, and the
    unanimous-agreement rule in the reconciliation tests.
    """
    cases = [
        TextSignals(support=0.95, runner_up=0.05, acoustic=0.95, alignment_quality=1.0),
        TextSignals(support=0.8, runner_up=0.3, acoustic=0.6, alignment_quality=0.9),
        TextSignals(support=0.6, runner_up=0.5, acoustic=0.4, alignment_quality=0.6),
        TextSignals(support=0.3, runner_up=0.28, acoustic=0.2, alignment_quality=0.3),
        TextSignals(support=0.05, runner_up=0.04, acoustic=0.1, alignment_quality=0.1),
    ]
    for signals in cases:
        assessment = assess_text(signals)
        assert assessment.category is category_for(assessment.strength)


# -- Saying a strength out loud and showing it in a cell -----------------


def test_a_strength_is_spoken_as_words_and_shown_as_a_symbol() -> None:
    assert strength_percentage(0.47) == "47 percent"
    assert strength_display(0.47) == "47%"


def test_both_forms_of_the_same_strength_carry_the_same_number() -> None:
    """A cell and its announcement disagreeing would be a bug nobody could see.

    The person reading the screen and the person listening to it are often
    the same person at different magnifications, and a figure that changed
    between the two would be impossible to make sense of.
    """
    for strength in (0.0, 0.004, 0.126, 0.5, 0.666, 0.999, 1.0):
        spoken = strength_percentage(strength).split(" ")[0]
        assert strength_display(strength) == f"{spoken}%"


def test_a_strength_is_rounded_to_whole_percentage_points() -> None:
    # Not truncated, and not carried to a decimal place. The strength is a
    # weighing of evidence rather than a measurement, so a tenth of a point
    # would be precision the inputs never had.
    assert strength_display(0.4712) == "47%"
    assert strength_display(0.4759) == "48%"
    assert strength_percentage(0.0) == "0 percent"
    assert strength_percentage(1.0) == "100 percent"


def test_a_word_nobody_measured_says_so_rather_than_showing_a_number() -> None:
    """"Not measured" and "nought percent" are opposite statements.

    One says there is no evidence either way, the other says the evidence is
    as bad as it gets, and only the second belongs at the top of a review
    queue. They must never be shown as the same thing.
    """
    assert strength_percentage(None) == STRENGTH_NOT_MEASURED
    assert strength_display(None) == STRENGTH_NOT_MEASURED
    assert strength_display(0.0) != STRENGTH_NOT_MEASURED
    assert strength_percentage(0.0) != STRENGTH_NOT_MEASURED


def test_the_unmeasured_text_is_a_phrase_a_screen_reader_can_read() -> None:
    # Not a dash, not a blank and not a symbol. A blank cell is silent, and a
    # mark conveys the meaning by appearance alone, which the accessibility
    # rules treat as a functional failure rather than a cosmetic one.
    assert STRENGTH_NOT_MEASURED.strip() == STRENGTH_NOT_MEASURED
    assert any(character.isalpha() for character in STRENGTH_NOT_MEASURED)


def test_every_factor_is_kept_so_the_answer_can_be_explained() -> None:
    assessment = assess_text(
        TextSignals(support=0.8, runner_up=0.3, acoustic=0.7, alignment_quality=0.9)
    )
    for name in (
        "support",
        "decisiveness",
        "acoustic_confidence",
        "alignment_quality",
        "language_support",
        "strength",
    ):
        assert name in assessment.components


# -- Timing and speaker are answered separately --------------------------


def test_an_exactly_timed_word_is_high_confidence() -> None:
    assert timing_confidence(TimingStatus.EXACT_PROVIDER_TIME) is Confidence.HIGH


def test_a_formatting_change_keeps_the_span_it_inherited() -> None:
    assert timing_confidence(TimingStatus.MAPPED_FORMAT_EQUIVALENT) is Confidence.HIGH


def test_a_shared_span_is_not_high_confidence() -> None:
    assert timing_confidence(TimingStatus.SHARED_PHRASE_SPAN) is not Confidence.HIGH


def test_a_word_with_no_acoustic_evidence_has_no_timing_to_trust() -> None:
    assert timing_confidence(TimingStatus.UNALIGNED) is Confidence.UNRESOLVED


def test_weak_alignment_pulls_the_timing_down_as_well() -> None:
    assert timing_confidence(TimingStatus.MAPPED_SUBSTITUTION, 0.2) is not Confidence.HIGH


def test_a_word_with_no_speaker_is_unresolved_rather_than_assumed() -> None:
    assert speaker_confidence(False) is Confidence.UNRESOLVED


def test_a_disputed_boundary_lowers_the_speaker_confidence() -> None:
    assert speaker_confidence(True) is Confidence.HIGH
    assert speaker_confidence(True, disputed_boundary=True) is Confidence.REVIEW_SUGGESTED


def test_overlapping_speech_is_worse_than_a_disputed_boundary() -> None:
    assert speaker_confidence(True, overlapping=True) is Confidence.REVIEW_REQUIRED


def test_provider_confidences_are_never_averaged_together() -> None:
    """The rule section 16 states outright, checked as behaviour.

    One service that heard the word clearly and one that reports nothing must
    not produce something halfway between the two, because the second number
    does not exist and inventing it would drag a clear reading down.
    """
    tokens = [
        token(provider=Provider.ELEVENLABS, log_probability=-0.01),
        token(provider=Provider.OPENAI),
        token(provider=Provider.MICROSOFT),
    ]
    value, source = best_acoustic_confidence(tokens)
    assert source is Provider.ELEVENLABS
    assert value is not None and value > 0.98


def test_alignment_status_names_the_right_kind_of_problem() -> None:
    from vox_verbatim.transcription.confidence import alignment_reasons

    assert ReviewReason.UNALIGNED_WORD in alignment_reasons(AlignmentStatus.INSERTION, 1.0)
    assert ReviewReason.WEAK_ALIGNMENT in alignment_reasons(AlignmentStatus.SUBSTITUTION, 0.1)
    assert alignment_reasons(AlignmentStatus.EXACT, 1.0) == ()
