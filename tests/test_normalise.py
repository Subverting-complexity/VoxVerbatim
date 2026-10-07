"""Tests for the comparison-only forms of transcript text.

Everything here is a pair of texts and a question: are these two services
saying the same thing? The tests are written as those pairs, because that
is exactly how the module is used, and because a table of pairs is the
only readable way to cover a set of equivalences this wide.

The refusals matter as much as the equivalences. A module that called
everything equivalent would pass half of these tests and be useless, so
the pairs that must *not* match are tested just as carefully as the ones
that must.
"""

from __future__ import annotations

import pytest

from vox_verbatim.transcription.normalise import (
    EquivalenceKind,
    are_equivalent,
    equivalence_kind,
    is_number_word,
    is_punctuation_only,
    normalise,
    normalise_phrase,
    normalise_sequence,
    read_number,
    spelled_as,
)


# -- Case, Unicode and whitespace ---------------------------------------


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Contract", "contract"),
        ("CONTRACT", "contract"),
        ("ﬁnal", "final"),  # a ligature, which Unicode normalisation takes apart
        ("２５", "25"),  # full-width digits, as some services return them
        ("the  contract", "the contract"),
        (" contract ", "contract"),
    ],
)
def test_case_unicode_and_whitespace_are_set_aside(first, second):
    assert are_equivalent(first, second)


def test_a_difference_of_capitals_is_reported_as_exactly_that():
    assert equivalence_kind("Contract", "contract") is EquivalenceKind.CASE_ONLY


# -- Punctuation and apostrophes ----------------------------------------


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("contract.", "contract"),
        ("contract,", "contract"),
        ("(contract)", "contract"),
        ("don't", "don’t"),  # a straight apostrophe against a curly one
        ("don't", "dont"),  # and against no apostrophe at all
        ("O'Brien", "OBrien"),
    ],
)
def test_punctuation_and_apostrophes_are_set_aside(first, second):
    assert equivalence_kind(first, second) is EquivalenceKind.PUNCTUATION_ONLY


@pytest.mark.parametrize(
    ("contraction", "word"),
    [
        ("we're", "were"),
        ("we'll", "well"),
        ("I'll", "ill"),
        ("she'd", "shed"),
        ("he'll", "hell"),
        ("I'd", "id"),
        ("she'll", "shell"),
    ],
)
def test_a_contraction_that_spells_another_word_stays_apart(contraction, word):
    assert equivalence_kind(contraction, word) is EquivalenceKind.DIFFERENT
    assert not are_equivalent(contraction, word)


@pytest.mark.parametrize(
    ("first", "second"),
    [("we're", "we’re"), ("I'll", "I’ll"), ("we’ll", "weʼll")],
)
def test_the_shape_of_the_apostrophe_in_a_contraction_does_not_matter(first, second):
    assert equivalence_kind(first, second) is EquivalenceKind.PUNCTUATION_ONLY


def test_quote_marks_around_a_word_are_not_a_contraction():
    assert equivalence_kind("'were'", "were") is EquivalenceKind.PUNCTUATION_ONLY
    assert equivalence_kind("‘we're’", "we're") is EquivalenceKind.PUNCTUATION_ONLY


@pytest.mark.parametrize(
    ("first", "second"),
    [("a ' b", "a b"), ("so ’ then", "so then"), ("the ` word", "the word")],
)
def test_an_apostrophe_standing_alone_is_only_punctuation(first, second):
    assert equivalence_kind(first, second) is EquivalenceKind.PUNCTUATION_ONLY


def test_a_phrase_with_a_contraction_does_not_meet_the_ordinary_word():
    assert normalise("we're going") != normalise("were going")
    assert not are_equivalent("we're going", "were going")


def test_punctuation_on_its_own_has_no_comparison_form():
    assert normalise(",") == ""
    assert is_punctuation_only(".")
    assert is_punctuation_only("")
    assert not is_punctuation_only("contract")


# -- German characters, in both directions ------------------------------


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Jürgen", "Juergen"),
        ("Juergen", "Jürgen"),
        ("schön", "schoen"),
        ("schoen", "schön"),
        ("Bär", "Baer"),
        ("Baer", "Bär"),
        ("Straße", "Strasse"),
        ("Strasse", "Straße"),
    ],
)
def test_german_letters_match_their_written_out_form_both_ways(first, second):
    assert are_equivalent(first, second)


@pytest.mark.parametrize(("first", "second"), [("Jürgen", "Jurgen"), ("Jurgen", "Jürgen")])
def test_a_dropped_umlaut_is_still_the_same_name(first, second):
    """The specification's own example: Scribe writes Jurgen, we mean Jürgen."""
    assert equivalence_kind(first, second) is EquivalenceKind.SPELLING_VARIANT


@pytest.mark.parametrize(("first", "second"), [("toe", "to"), ("doe", "do"), ("blue", "blu")])
def test_ordinary_words_are_not_dragged_into_the_german_equivalence(first, second):
    """The second direction is only tried when a German letter is present.

    Collapsing "oe" into "o" for every word would make "toe" and "to" the
    same word, which is the sort of quiet damage that would be very hard to
    find again once it was in a transcript.
    """
    assert not are_equivalent(first, second)


@pytest.mark.parametrize(
    ("first", "second"),
    [("Ja.", "Yeah."), ("ja", "yeah"), ("Yeah,", "JA"), ("Ja, yeah", "yeah ja")],
)
def test_ja_and_yeah_are_the_same_word(first, second):
    """Afrikaans and German write "ja" where English writes "yeah"."""
    assert are_equivalent(first, second)
    assert equivalence_kind(first, second) is EquivalenceKind.SPELLING_VARIANT


@pytest.mark.parametrize(("first", "second"), [("jam", "yeah"), ("Ja.", "yes"), ("ja", "yea")])
def test_only_ja_and_yeah_are_joined(first, second):
    assert not are_equivalent(first, second)


def test_spelled_as_ignores_capitals_and_punctuation_only():
    assert spelled_as("Ja.", "ja")
    assert spelled_as("Yeah,", "yeah")
    assert not spelled_as("Ja.", "yeah")
    assert not spelled_as("Jam", "ja")


# -- Numbers ------------------------------------------------------------


@pytest.mark.parametrize(
    ("words", "digits"),
    [
        # English
        ("twenty five", "25"),
        ("twenty-five", "25"),
        ("seven", "7"),
        ("nineteen", "19"),
        ("three hundred forty seven", "347"),
        ("one hundred and one", "101"),
        ("two thousand twenty four", "2024"),
        ("five thousand", "5000"),
        # German, including the compounds it writes as one word
        ("fünfundzwanzig", "25"),
        ("fuenfundzwanzig", "25"),
        ("einundzwanzig", "21"),
        ("dreihundertsiebenundvierzig", "347"),
        ("zwölf", "12"),
        ("dreißig", "30"),
        ("zwei tausend", "2000"),
        # Afrikaans
        ("vyfentwintig", "25"),
        ("vyf en twintig", "25"),
        ("sewentien", "17"),
        ("negentig", "90"),
        ("drie honderd", "300"),
        ("nul", "0"),
    ],
)
def test_numbers_written_in_words_match_the_same_number_in_digits(words, digits):
    assert equivalence_kind(words, digits) is EquivalenceKind.NUMBER_FORMAT
    assert are_equivalent(digits, words)


def test_the_german_articles_ein_and_eine_are_words_not_the_number_one():
    """"ein" and "eine" are "a" and "an" far more often than they are 1.

    Reading them as the number would make one service hearing "ein" and
    another hearing "eine" agree, when the two are different words.
    """
    assert equivalence_kind("ein", "eine") is EquivalenceKind.DIFFERENT
    assert not are_equivalent("ein", "eine")
    assert normalise("ein") == "ein"
    assert normalise("eine") == "eine"
    assert normalise("Eine") == "eine"
    assert read_number("ein") is None
    assert read_number("eine") is None
    assert not are_equivalent("ein", "1")
    assert not are_equivalent("eine", "eins")


@pytest.mark.parametrize(
    ("words", "digits"),
    [
        ("eins", "1"),
        ("einundzwanzig", "21"),
        ("einunddreißig", "31"),
        ("einhundert", "100"),
        ("eintausend", "1000"),
        ("einhunderteins", "101"),
        ("eintausendeinhundert", "1100"),
        ("zwei", "2"),
        ("neunzehn", "19"),
    ],
)
def test_eins_and_compounds_that_start_with_ein_are_still_numbers(words, digits):
    """Inside a joined compound, "ein" is unmistakably the number.

    "einhundert" and "eintausend" are in here on purpose: they carry the
    same value as the bare words "hundert" and "tausend", which are left as
    words, so they need to be recognised as compounds rather than by value.
    """
    assert normalise(words) == digits
    assert read_number(words) == int(digits)
    assert equivalence_kind(words, digits) is EquivalenceKind.NUMBER_FORMAT


@pytest.mark.parametrize(
    ("words", "digits"),
    [
        ("eine Million", "1 Million"),
        ("eine Milliarde", "1 Milliarde"),
        ("ein tausend", "1000"),
        ("ein hundert", "100"),
    ],
)
def test_ein_and_eine_in_front_of_a_scale_word_are_the_number_one(words, digits):
    """"eine Million" is the ordinary German way to write a million.

    A scale word cannot follow the article, so here "ein" and "eine" are the
    number, and a service writing the words agrees with one writing digits.
    """
    assert equivalence_kind(words, digits) is EquivalenceKind.NUMBER_FORMAT


def test_ein_and_eine_in_front_of_any_other_word_stay_the_article():
    assert equivalence_kind("ein Haus", "1 Haus") is EquivalenceKind.DIFFERENT
    assert equivalence_kind("eine Frau", "1 Frau") is EquivalenceKind.DIFFERENT


def test_ein_uhr_is_one_o_clock_and_eine_uhr_is_a_clock():
    """"Uhr" is feminine, so "ein Uhr" can only be the time."""
    assert equivalence_kind("um ein Uhr", "um 1 Uhr") is EquivalenceKind.NUMBER_FORMAT
    assert equivalence_kind("eine Uhr", "1 Uhr") is EquivalenceKind.DIFFERENT


def test_the_bare_scale_words_are_still_not_numbers():
    assert read_number("hundert") is None
    assert read_number("tausend") is None
    assert read_number("thousand") is None


def test_a_thousands_comma_does_not_change_the_number():
    assert equivalence_kind("1,000", "1000") is EquivalenceKind.NUMBER_FORMAT
    assert are_equivalent("15,000", "fifteen thousand")



class TestAmountsWithPrefixesAndDecimals:
    """A currency prefix or a decimal tail does not hide a thousands comma.

    "R15,000" and "R15 000" are the same amount written two ways, so they
    must not be shown to the user as a disagreement. The comma grouping
    stays strict, so a different amount still reads as different.
    """

    def test_a_currency_prefix_does_not_stop_the_comma_being_a_separator(self):
        assert equivalence_kind("R15,000", "R15 000").is_equivalent
        assert equivalence_kind("$1,000", "$1000") is EquivalenceKind.NUMBER_FORMAT

    def test_a_decimal_tail_after_a_comma_grouping_is_kept(self):
        assert equivalence_kind("1,000.50", "1000.50") is EquivalenceKind.NUMBER_FORMAT
        assert are_equivalent("R1,000.50", "R1000.50")

    def test_a_different_amount_with_the_same_prefix_stays_different(self):
        assert equivalence_kind("R15,000", "R50,000") is EquivalenceKind.DIFFERENT

    def test_the_fraction_is_kept_exactly_as_written(self):
        assert not are_equivalent("1,000.50", "1000.5")

    def test_german_decimal_order_is_still_not_guessed(self):
        assert not are_equivalent("1.000,50", "1000.50")

def test_a_full_stop_between_digits_is_left_exactly_where_it_is():
    """A full stop means one thing in English and the opposite in German.

    "1.500" is one and a half to an English reader and fifteen hundred to a
    German one. Either guess produces a number that reads perfectly well
    and is out by a factor of a thousand, so neither guess is made.
    """
    assert not are_equivalent("1.500", "1500")
    assert not are_equivalent("3.141", "3141")


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("-5", "5"),
        ("-1,200", "1,200"),
        ("(-5)", "5"),
        ('"-5', "5"),
        ("50%", "50"),
        ("10:30", "1030"),
    ],
)
def test_a_minus_a_per_cent_sign_or_a_time_colon_changes_the_value(first, second):
    """A dropped sign reads perfectly well and names a different value.

    These are the values a person has to see, so a service that left the
    sign out does not agree with one that kept it.
    """
    assert equivalence_kind(first, second) is EquivalenceKind.DIFFERENT
    assert not are_equivalent(first, second)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("10%-20%", "10% 20%"),
        ("(10)-20", "10 20"),
    ],
)
def test_a_dash_after_a_closing_mark_is_still_a_hyphen(first, second):
    assert are_equivalent(first, second)


def test_every_shape_of_minus_sign_is_the_same_minus():
    assert are_equivalent("−5", "-5")
    assert are_equivalent("–5", "-5")
    assert are_equivalent("-1,200", "-1200")
    assert read_number("-5") is None


def test_a_hyphen_between_words_or_numbers_is_still_a_hyphen():
    assert are_equivalent("well-known", "well known")
    assert normalise("well-known") == normalise("well known")
    assert normalise("10-20") == normalise("10 20")
    assert normalise("10-20") != normalise("-10 20")


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("25", 25),
        ("1,000", 1000),
        ("twenty five", 25),
        ("fünfundzwanzig", 25),
        ("vyfentwintig", 25),
        ("two thousand twenty four", 2024),
    ],
)
def test_a_number_can_be_read_back_as_a_number(text, value):
    assert read_number(text) == value


@pytest.mark.parametrize("text", ["contract", "25 people", "", "thousand"])
def test_things_that_are_not_numbers_come_back_as_nothing(text):
    assert read_number(text) is None


def test_a_year_is_not_added_up():
    """"Nineteen eighty four" is a year, not nineteen plus eighty-four.

    The numeral grammar refuses it, so both services saying it still agree
    with each other, and neither is quietly turned into 1984.
    """
    assert normalise("nineteen eighty four") == "nineteeneightyfour"
    assert not are_equivalent("nineteen eighty four", "1984")


def test_digits_read_out_one_by_one_are_left_as_they_were():
    """A telephone number read aloud must not become the sum of its digits."""
    assert not are_equivalent("one two three", "123")
    assert normalise("one two three") == "onetwothree"


def test_two_numbers_standing_next_to_each_other_stay_two_numbers():
    assert normalise("in 2024 25 people came") == normalise("in 2024 25 people came")
    assert not are_equivalent("2024 25", "2049")


def test_an_english_and_only_joins_a_hundred_to_what_follows_it():
    """"Five and ten" is a phrase, not fifteen."""
    assert not are_equivalent("five and ten", "15")
    assert are_equivalent("one hundred and fifty", "150")


@pytest.mark.parametrize(
    ("words", "value"),
    [
        ("two thousand and five", 2005),
        ("two thousand and twenty four", 2024),
        ("one thousand and one", 1001),
        ("zweitausendfünf", 2005),
        ("zweitausend fünf", 2005),
        ("tausend und eins", 1001),
        ("tweeduisend en vyf", 2005),
        ("honderd en een", 101),
        ("honderd en vyf en twintig", 125),
        ("zweihundert", 200),
        ("zweihundert tausend", 200000),
    ],
)
def test_an_and_after_a_thousand_reads_the_way_people_say_years(words, value):
    """South African English says "two thousand and five" routinely.

    Refusing that form turned a year every service agreed on into a
    numeric disagreement that could never be settled. German and Afrikaans
    put their own joiners in the same place, and a joined compound such as
    "zweitausend" carries its quantity inside the word.
    """
    assert read_number(words) == value
    assert are_equivalent(words, str(value))
    assert equivalence_kind(words, str(value)) is EquivalenceKind.NUMBER_FORMAT


def test_a_joiner_with_nothing_after_it_is_not_a_number():
    assert read_number("two thousand and") is None
    assert not are_equivalent("two thousand and", "2000")


def test_an_afrikaans_en_between_two_units_does_not_add_them():
    """"Vyf en ses" is five and six, not eleven."""
    assert not are_equivalent("vyf en ses", "11")
    assert read_number("vyf en ses") is None


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("1 2", "12"),
        ("twenty 5", "205"),
        ("1/2", "12"),
        ("24/7", "247"),
        ("1/2", "1 2"),
        ("2024 25", "202425"),
    ],
)
def test_two_numbers_are_never_run_together_into_one(first, second):
    """A half and twelve are not the same evidence.

    The compound stage runs words together so that "data base" meets
    "database". Letting it run digits together made "1 2" and "12" one
    candidate, so a service that heard two numbers and one that heard one
    were never seen to disagree.
    """
    assert not are_equivalent(first, second)
    assert equivalence_kind(first, second) is EquivalenceKind.DIFFERENT


def test_a_digit_next_to_a_letter_still_joins():
    assert are_equivalent("3 kg", "3kg")
    assert are_equivalent("data base", "database")


# -- Contractions and compounds -----------------------------------------


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("database", "data base"),
        ("database", "data-base"),
        ("data-base", "data base"),
        ("up-to-date", "up to date"),
        ("cannot", "can not"),
        ("can't", "can not"),
        ("don't", "do not"),
        ("it's", "it is"),
        ("I'm", "I am"),
        ("they're", "they are"),
    ],
)
def test_words_joined_or_split_differently_are_the_same_words(first, second):
    assert equivalence_kind(first, second) is EquivalenceKind.COMPOUND


def test_a_difference_in_how_many_words_there_are_outranks_the_rest():
    """"data-base" against "data base" is punctuation *and* one word against two.

    The kind that constrains the timing most is the one reported, because a
    caller told only about the punctuation would let the replacement keep a
    span it is not entitled to. Two words sharing one span means nobody
    knows where inside it the join between them falls.
    """
    assert equivalence_kind("data-base", "data base") is EquivalenceKind.COMPOUND
    assert equivalence_kind("Data-Base", "data base") is EquivalenceKind.COMPOUND
    # A number keeps its own kind, which says everything compound says and
    # adds that the content in dispute is a number.
    assert equivalence_kind("twenty five", "25") is EquivalenceKind.NUMBER_FORMAT
    # Nothing was joined or split here, so the smaller description stands.
    assert equivalence_kind("Contract.", "contract") is EquivalenceKind.PUNCTUATION_ONLY


def test_punctuation_standing_on_its_own_is_not_counted_as_a_word():
    """A service that returns "contract ." has still said one word."""
    assert equivalence_kind("contract .", "contract") is EquivalenceKind.PUNCTUATION_ONLY


def test_a_contraction_that_is_also_an_ordinary_word_is_left_alone():
    """"Were" and "we're" sound different and are different words.

    Expanding every apostrophe-free contraction would merge them, and the
    transcript would lose a real distinction. The table leaves them out on
    purpose.
    """
    assert not are_equivalent("were", "we are")
    assert not are_equivalent("well", "we will")


def test_genuinely_different_words_stay_different():
    assert equivalence_kind("contract", "contact") is EquivalenceKind.DIFFERENT
    assert equivalence_kind("fifteen thousand", "fifty thousand") is EquivalenceKind.DIFFERENT
    assert not are_equivalent("Müller", "Meier")


# -- The kinds themselves -----------------------------------------------


def test_identical_text_is_reported_as_identical():
    assert equivalence_kind("contract", "contract") is EquivalenceKind.IDENTICAL


def test_every_kind_but_different_counts_as_the_same_evidence():
    """The timing rules ask this one question of the kind, so it is tested."""
    for kind in EquivalenceKind:
        assert kind.is_equivalent == (kind is not EquivalenceKind.DIFFERENT)
        assert kind.display_name


def test_the_smallest_honest_description_of_a_difference_is_the_one_reported():
    """A word differing in case and in a full stop is not a spelling variant."""
    assert equivalence_kind("Contract.", "contract") is EquivalenceKind.PUNCTUATION_ONLY


# -- Sequences ----------------------------------------------------------


def test_a_sequence_keeps_its_positions_so_it_can_be_indexed_back():
    forms = normalise_sequence(["The", ",", "data", "base", "cost", "twenty-five"])

    assert forms == ("the", "", "data", "base", "cost", "25")


def test_several_words_are_compared_as_a_phrase_not_as_joined_forms():
    """This is why :func:`normalise_phrase` exists at all.

    Pasting the comparison forms of "twenty" and "five" together gives
    "205". Only normalising the phrase as a whole gives 25, which is what
    lets the alignment see one word on one side against two on the other.
    """
    forms = normalise_sequence(["twenty", "five"])
    assert "".join(forms) == "205"
    assert normalise_phrase(["twenty", "five"]) == "25"
    assert normalise_phrase(["data", "base"]) == normalise("database")


@pytest.mark.parametrize(
    ("spaced", "plain"),
    [
        ("10 000", "10,000"),
        ("R10 000", "R10000"),
        ("100 000 000", "100000000"),
        ("082 123 4567", "0821234567"),
    ],
)
def test_a_number_written_in_blocks_is_still_one_number(spaced: str, plain: str) -> None:
    """Thousands grouped by a space, and a telephone number read in blocks,
    are one number however a service chose to write it."""
    assert are_equivalent(spaced, plain)


@pytest.mark.parametrize("pair", [("1 2", "12"), ("20 5", "205"), ("2024 05", "202405")])
def test_two_numbers_beside_each_other_stay_two(pair: tuple[str, str]) -> None:
    assert not are_equivalent(*pair)


@pytest.mark.parametrize(
    "text",
    [
        "two", "hundred", "and", "fifty", "twenty-five", "250", "150,000",
        "einundzwanzig", "honderd", "en", "und", "tausend", "ein",
    ],
)
def test_a_word_that_can_belong_to_a_spoken_number_is_a_number_word(text):
    assert is_number_word(text)


@pytest.mark.parametrize("text", ["cost", "rand", "", "data", "often"])
def test_an_ordinary_word_is_not_a_number_word(text):
    assert not is_number_word(text)
