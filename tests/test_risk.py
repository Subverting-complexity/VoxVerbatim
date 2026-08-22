"""Tests for finding the values that must not be guessed at.

The bias of these tests matches the bias of the module: they check that
risky content is found, and they check that the one thing which must never
happen does not happen, which is a number being settled quietly because its
separators were read with the wrong language in mind.
"""

from __future__ import annotations

import pytest

from audio_transcriber.transcription.model import RiskCategory
from audio_transcriber.transcription.risk import (
    find_risks,
    has_ambiguous_separator,
    is_numeric,
    numbers_disagree,
    risk_at,
    risk_categories,
    risk_details_at,
)


def _categories(*words: str) -> set[RiskCategory]:
    found: set[RiskCategory] = set()
    for categories in risk_categories(list(words)):
        found.update(categories)
    return found


def test_a_number_beside_a_currency_word_is_money() -> None:
    assert RiskCategory.MONEY in _categories("fifteen", "thousand", "rand")


def test_the_whole_amount_is_protected_not_only_its_ends() -> None:
    words = ["pay", "fifteen", "thousand", "five", "hundred", "rand", "today"]
    for position in range(1, 6):
        assert RiskCategory.MONEY in risk_at(words, position)


def test_a_currency_symbol_needs_no_keyword() -> None:
    assert RiskCategory.MONEY in _categories("it", "cost", "$1500")


def test_the_rand_written_against_its_digits_is_money() -> None:
    assert RiskCategory.MONEY in _categories("about", "R2500", "each")


def test_a_month_is_a_date_on_its_own() -> None:
    assert RiskCategory.DATE in _categories("on", "the", "fifth", "of", "January")


def test_german_and_afrikaans_months_are_recognised() -> None:
    assert RiskCategory.DATE in _categories("am", "fünften", "März")
    assert RiskCategory.DATE in _categories("op", "vyf", "Maart")


def test_a_clock_time_is_a_time() -> None:
    assert RiskCategory.TIME in _categories("at", "14:30", "sharp")


def test_a_percentage_is_found_by_its_word_and_by_its_sign() -> None:
    assert RiskCategory.PERCENTAGE in _categories("twenty", "percent")
    assert RiskCategory.PERCENTAGE in _categories("20%")
    assert RiskCategory.PERCENTAGE in _categories("zwanzig", "Prozent")


def test_a_long_run_of_digits_could_be_a_telephone_or_an_account() -> None:
    found = _categories("call", "0821234567")
    assert RiskCategory.TELEPHONE in found
    assert RiskCategory.ACCOUNT_NUMBER in found


def test_digits_read_out_one_at_a_time_are_an_identifier() -> None:
    found = _categories("the", "number", "is", "four", "seven", "two", "nine")
    assert RiskCategory.TELEPHONE in found


def test_an_account_keyword_claims_the_number_beside_it() -> None:
    assert RiskCategory.ACCOUNT_NUMBER in _categories("account", "4471")


def test_a_version_is_found_by_shape_and_by_keyword() -> None:
    assert RiskCategory.VERSION_NUMBER in _categories("running", "v2.1")
    assert RiskCategory.VERSION_NUMBER in _categories("version", "eleven")


def test_a_street_number_is_an_address() -> None:
    assert RiskCategory.ADDRESS in _categories("at", "12", "Church", "Street")


def test_a_measured_amount_is_a_quantity() -> None:
    assert RiskCategory.QUANTITY in _categories("about", "thirty", "kilograms")


def test_a_bare_number_is_still_a_quantity() -> None:
    assert RiskCategory.QUANTITY in risk_at(["we", "need", "twelve", "more"], 2)


def test_a_case_number_is_a_legal_identifier() -> None:
    assert RiskCategory.LEGAL_IDENTIFIER in _categories("case", "4471", "of", "2024")


def test_a_dose_is_a_medical_measurement() -> None:
    assert RiskCategory.MEDICAL_MEASUREMENT in _categories("take", "500", "mg")


def test_letters_and_digits_together_look_like_a_product_code() -> None:
    assert RiskCategory.PRODUCT_CODE in _categories("part", "AB12X")


def test_an_ordinary_sentence_is_not_flagged() -> None:
    assert _categories("we", "should", "talk", "about", "the", "proposal") == set()


# -- Dates and fractions that carry one separator or none -----------------


@pytest.mark.parametrize("text", ["12/05", "05/2024", "1/12"])
def test_a_day_and_month_with_one_slash_is_a_date(text: str) -> None:
    """The written-date rule wanted two separators, so "12/05" slipped past."""
    assert RiskCategory.DATE in risk_at(["on", text, "we", "meet"], 1)


@pytest.mark.parametrize("text", ["1/2", "3/4", "24/7", "7/8"])
def test_a_fraction_no_rule_claims_is_still_a_number(text: str) -> None:
    """A token with a digit in it that nobody claimed is a quantity.

    These cannot be read as one number, and the catch-all used to insist on
    that, so a disputed half could be settled on plausibility alone.
    """
    assert RiskCategory.QUANTITY in risk_at(["about", text, "of", "them"], 1)


def test_a_single_full_stop_between_digits_is_left_to_the_separator_rule() -> None:
    """"12.05" is a decimal to a German reader, so it is not called a date."""
    assert RiskCategory.DATE not in risk_at(["about", "12.05", "each"], 1)
    assert RiskCategory.QUANTITY in risk_at(["about", "12.05", "each"], 1)


# -- Days of the month named in words --------------------------------------


@pytest.mark.parametrize(
    "words",
    [
        ["on", "the", "fifteenth"],
        ["by", "the", "twentieth"],
        ["the", "twenty-first"],
        ["am", "vierten"],
        ["am", "fünfundzwanzigsten"],
        ["am", "fuenfundzwanzigsten"],
        ["am", "zwanzigsten"],
        ["die", "vyfde"],
        ["die", "een-en-twintigste"],
    ],
)
def test_an_ordinal_from_the_fourth_up_is_a_date_on_its_own(words: list[str]) -> None:
    assert RiskCategory.DATE in risk_at(words, len(words) - 1)
    assert "a day of the month" in risk_details_at(words, len(words) - 1)


@pytest.mark.parametrize(
    "words",
    [
        ["the", "first", "of", "March"],
        ["on", "the", "first"],
        ["am", "ersten"],
        ["am", "ersten", "Mai"],
        ["op", "die", "eerste"],
        ["die", "derde", "Junie"],
        ["the", "second", "week"],
    ],
)
def test_a_low_ordinal_is_a_date_when_the_calendar_is_nearby(words: list[str]) -> None:
    position = next(
        index for index, word in enumerate(words)
        if word in {"first", "second", "ersten", "eerste", "derde"}
    )
    assert RiskCategory.DATE in risk_at(words, position)


def test_the_whole_of_first_of_march_is_protected() -> None:
    words = ["the", "first", "of", "March", "then"]
    for position in range(1, 4):
        assert RiskCategory.DATE in risk_at(words, position)
    assert RiskCategory.DATE not in risk_at(words, 4)


@pytest.mark.parametrize(
    "words",
    [
        ["the", "first", "one", "is", "fine"],
        ["the", "second", "item"],
        ["die", "erste", "Frage"],
        ["die", "eerste", "keer"],
    ],
)
def test_a_low_ordinal_in_an_ordinary_phrase_is_not_a_date(words: list[str]) -> None:
    assert RiskCategory.DATE not in risk_at(words, 1)


def test_a_digit_ordinal_is_still_a_date() -> None:
    assert RiskCategory.DATE in risk_at(["on", "the", "1st"], 2)
    assert RiskCategory.DATE in risk_at(["on", "the", "15th"], 2)


# -- The decimal separator, which is the dangerous one -------------------


def test_a_separator_between_digits_is_never_read_quietly() -> None:
    assert has_ambiguous_separator("1,5")
    assert has_ambiguous_separator("1.500")
    assert has_ambiguous_separator("15,000")


def test_a_plain_number_carries_no_separator_ambiguity() -> None:
    assert not has_ambiguous_separator("1500")
    assert not has_ambiguous_separator("fifteen")


def test_an_ambiguous_number_is_high_risk_on_that_ground_alone() -> None:
    assert RiskCategory.QUANTITY in _categories("1,5")
    details = risk_details_at(["it", "was", "1,5"], 2)
    assert any("separator" in detail for detail in details)


def test_the_same_number_written_two_ways_is_not_a_disagreement() -> None:
    assert not numbers_disagree("twenty-five", "25")
    assert not numbers_disagree("25", "25")


def test_different_numbers_disagree() -> None:
    assert numbers_disagree("15000", "50000")


def test_two_readings_of_an_ambiguous_separator_disagree() -> None:
    # "15,000" and "15.000" are the same characters read under two
    # conventions and are a thousand-fold apart. Nothing here may decide
    # which was meant.
    assert numbers_disagree("15,000", "15.000")


def test_numbers_are_recognised_in_all_three_languages() -> None:
    assert is_numeric("fifteen")
    assert is_numeric("fünfzehn")
    assert is_numeric("vyftien")
    assert not is_numeric("contract")


# -- The shape of what comes back ----------------------------------------


def test_a_finding_names_its_reason_in_plain_words() -> None:
    findings = find_risks(["fifteen", "thousand", "rand"])
    money = [finding for finding in findings if finding.category is RiskCategory.MONEY]
    assert money and money[0].detail == "an amount of money"


def test_asking_outside_the_sequence_is_answered_rather_than_raising() -> None:
    assert risk_at(["one"], 5) == ()
    assert risk_details_at(["one"], -1) == ()


def test_a_keyword_out_of_reach_does_not_claim_a_number() -> None:
    words = ["rand", "and", "then", "we", "spoke", "for", "a", "while", "about", "twelve"]
    assert RiskCategory.MONEY not in risk_at(words, 9)
