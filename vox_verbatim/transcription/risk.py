"""The words where being nearly right is worse than admitting we do not know.

Most of a transcript can absorb a mistake. A misheard adjective makes a
sentence slightly wrong and a reader usually notices. An amount, a date, an
account number or a dose is different in two ways at once: the mistake is
invisible, because the wrong value reads perfectly well, and the cost of it
falls on somebody who trusted the document. Section 13 of the specification
lists the kinds of content this is true of, and this module is what finds
them.

Its only job is to say "do not guess here". Nothing in it decides what a
word is; it decides whether the reconciliation rules are allowed to settle a
disagreement about that word on their own. That is why it is deliberately
generous. A false positive costs the user a few seconds of review on a word
that turned out to be fine. A false negative costs them a wrong amount in a
document they had no reason to doubt. Those two are not comparable, so the
rules below lean the same way every time they are in doubt.

Three things about the design are worth explaining.

**It reads words in context, never on their own.** "15" by itself is a
quantity. "15 000 rand" is money, "15 January" is a date, "15:30" is a time
and "case 15 of 2024" is a legal identifier. So detection walks a small
window of the surrounding words and lets a keyword on one side of a number
claim the number on the other. A single word is not enough evidence to
classify anything, and asking it to be would produce exactly the confident,
wrong answers this module exists to prevent.

**Number conventions differ between the three languages, and the difference
is the most dangerous thing here.** English writes fifteen hundred and a
half as "1,500.5"; German and Afrikaans write it "1.500,5". A separator
between digits therefore cannot be read without knowing the language, and
the language of a single spoken number is often exactly what is uncertain.
This module does not guess. Any number carrying a separator between its
digits is treated as high risk on that ground alone, because a wrong reading
is out by a factor of a thousand and looks entirely reasonable on the page.

**It answers about a position, not about a match.** The reconciliation rules
ask "may I settle the word at this position by score alone?", so what comes
back is the categories covering that position, including those claimed by a
keyword several words away. A finding that covers a phrase covers every
word of it, because leaving the middle of "fifteen thousand rand" settleable
while its ends are protected would protect nothing.

Nothing here depends on Qt or on any provider library, so it can be tested
on its own.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

from vox_verbatim.transcription.model import RiskCategory
from vox_verbatim.transcription.normalise import MINUS_SIGNS, read_number

#: How far a keyword may reach to claim a number, in words. Three covers
#: "fifteen thousand five hundred rand" from either end, and stops short of
#: joining two sentences that merely happen to be near each other.
RISK_WINDOW = 3

#: How many digits in one token are enough to be an identifier rather than a
#: quantity. Seven is the shortest telephone number in ordinary use, and
#: nothing anybody says as a quantity runs that long without a separator.
LONG_DIGIT_RUN = 7

#: How many separate all-digit words in a row mean somebody is reading a
#: number out digit by digit, as people do with telephone and account
#: numbers. Four is short enough to catch "oh eight two one" and long enough
#: that ordinary counting does not trip it.
SPOKEN_DIGIT_RUN = 4


@dataclass(frozen=True)
class RiskFinding:
    """One stretch of words that must not be settled by plausibility alone.

    ``detail`` is written for a person rather than for the code: it appears
    in the review window beside the word, and "an amount of money" is worth
    more to somebody deciding what to listen for than the name of the rule
    that fired.
    """

    category: RiskCategory
    positions: tuple[int, ...]
    detail: str

    def covers(self, position: int) -> bool:
        return position in self.positions


# -- The words that give a number its meaning ----------------------------
#
# Every table below is written out in all three languages, in full, because
# a shorter clever version would be unreadable and this is a list anybody
# should be able to check by eye. Entries are compared against a lightly
# cleaned form of the word: lower case, surrounding punctuation removed,
# accents kept. Number words are deliberately absent; a numeral is found by
# reading it, not by looking it up here.

_MONEY_WORDS = frozenset(
    {
        "rand", "rands", "cent", "cents", "sent", "dollar", "dollars",
        "euro", "euros", "pound", "pounds", "pence", "franc", "francs",
        "yen", "zar", "usd", "eur", "gbp", "chf", "geld", "money",
        "amount", "bedrag", "betrag", "price", "prys", "preis", "cost",
        "koste", "kosten", "invoice", "faktuur", "rechnung", "salary",
        "salaris", "gehalt", "deposit", "deposito", "fee", "fooi",
        "gebuehr", "gebühr", "million", "millionen", "miljoen", "billion",
        "milliarde", "miljard", "thousand", "tausend", "duisend",
        "vat", "btw", "mwst", "tax", "belasting", "steuer",
    }
)

#: Currency marks, kept as characters rather than words because a service
#: writes them attached to the digits.
_CURRENCY_MARKS = "$€£¥₹"

_PERCENT_WORDS = frozenset({"percent", "percentage", "prozent", "persent", "persentasie"})

_MONTH_WORDS = frozenset(
    {
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
        "januar", "februar", "maerz", "märz", "mai", "juni", "juli",
        "oktober", "dezember",
        "januarie", "februarie", "maart", "mei", "junie", "julie",
        "augustus", "oktober", "desember",
    }
)

_WEEKDAY_WORDS = frozenset(
    {
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
        "sunday", "montag", "dienstag", "mittwoch", "donnerstag", "freitag",
        "samstag", "sonntag", "maandag", "dinsdag", "woensdag", "donderdag",
        "vrydag", "saterdag", "sondag",
    }
)

_DATE_WORDS = frozenset(
    {
        "date", "datum", "year", "jahr", "jaar", "month", "monat", "maand",
        "week", "woche", "deadline", "sperfrist", "sperdatum", "birthday",
        "geburtstag", "verjaarsdag", "anniversary", "quarter", "quartal",
        "kwartaal",
    }
)

_TIME_WORDS = frozenset(
    {
        "time", "zeit", "tyd", "clock", "uhr", "uur", "hour", "hours",
        "stunde", "stunden", "minute", "minutes", "minuten", "minute",
        "minuut", "minute", "second", "seconds", "sekunde", "sekunden",
        "am", "pm", "midday", "midnight", "mittag", "mitternacht",
        "middag", "middernag", "past", "quarter", "kwart", "halb", "half",
        "voormiddag", "namiddag",
    }
)

_TELEPHONE_WORDS = frozenset(
    {
        "telephone", "phone", "mobile", "cell", "cellphone", "fax",
        "telefon", "handy", "telefoon", "selfoon", "faks", "extension",
        "durchwahl", "dial", "call", "bel", "nommer", "nummer",
        "whatsapp", "landline",
    }
)

_ACCOUNT_WORDS = frozenset(
    {
        "account", "accounts", "konto", "kontonummer", "rekening",
        "rekeningnommer", "iban", "swift", "bic", "bank", "branch",
        "takkode", "sortcode", "routing", "reference", "referenz",
        "verwysing", "ref", "policy", "polis", "police", "invoice",
        "order", "bestellung", "bestelling", "card", "karte", "kaart",
        "pin", "otp", "customer", "kunde", "klient",
    }
)

_VERSION_WORDS = frozenset(
    {
        "version", "versie", "release", "build", "revision", "patch",
        "firmware", "ausgabe", "uitgawe",
    }
)

_ADDRESS_WORDS = frozenset(
    {
        "street", "straat", "strasse", "straße", "road", "weg", "pad",
        "avenue", "laan", "allee", "drive", "rylaan", "lane", "close",
        "crescent", "singel", "boulevard", "platz", "plein", "square",
        "suite", "apartment", "flat", "wohnung", "woonstel", "unit",
        "eenheid", "floor", "stock", "vloer", "block", "blok", "box",
        "posbus", "postfach", "postal", "poskode", "postcode", "plz",
        "postleitzahl", "address", "adresse", "adres", "erf", "stand",
        "gate", "hek",
    }
)

_QUANTITY_WORDS = frozenset(
    {
        "kilogram", "kilograms", "kilo", "kilos", "kg", "gram", "grams",
        "gramm", "g", "ton", "tons", "tonne", "tonnen", "pound", "pounds",
        "metre", "metres", "meter", "meters", "m", "km", "kilometre",
        "kilometres", "kilometer", "kilometers", "centimetre", "cm", "mm",
        "millimetre", "inch", "inches", "foot", "feet", "yard", "mile",
        "miles", "myl", "litre", "litres", "liter", "liters", "l",
        "millilitre", "gallon", "gallons", "square", "cubic", "hectare",
        "hektaar", "hektar", "acre", "degrees", "grad", "grade",
        "people", "persons", "mense", "personen", "items", "stuks",
        "stueck", "stück", "boxes", "kartons", "dose", "doses", "pallets",
        "units", "eenhede", "einheiten", "copies", "kopieë", "pages",
        "seiten", "bladsye", "times", "keer", "mal",
    }
)

_LEGAL_WORDS = frozenset(
    {
        "case", "fall", "saak", "docket", "claim", "eis", "anspruch",
        "contract", "kontrak", "vertrag", "clause", "klausel", "klousule",
        "section", "artikel", "abschnitt", "afdeling", "paragraph",
        "paragraaf", "act", "wet", "gesetz", "regulation", "verordnung",
        "regulasie", "schedule", "annexure", "aanhangsel", "anlage",
        "identity", "identiteit", "id", "passport", "reisepass",
        "paspoort", "licence", "license", "lisensie", "registration",
        "registrasie", "registrierung", "company", "maatskappy",
        "court", "hof", "gericht", "judgment", "vonnis", "urteil",
        "matter", "file",
    }
)

_MEDICAL_WORDS = frozenset(
    {
        "mg", "milligram", "milligrams", "milligramm", "mcg", "microgram",
        "ml", "millilitre", "milliliter", "mmol", "mmhg", "iu", "bpm",
        "dose", "dosage", "dosis", "dosering", "tablet", "tablets",
        "tablette", "pil", "pille", "capsule", "kapsel", "kapsule",
        "insulin", "insulien", "blood", "bloed", "blut", "pressure",
        "druk", "systolic", "diastolic", "sistolies", "temperature",
        "temperatuur", "pulse", "pols", "puls", "heart", "hart", "rate",
        "haemoglobin", "glucose", "glukose", "cholesterol", "creatinine",
        "prescription", "voorskrif", "rezept", "millimol", "microgram",
    }
)

_PRODUCT_WORDS = frozenset(
    {
        "model", "modell", "part", "teil", "onderdeel", "sku", "code",
        "kode", "katalog", "catalogue", "catalog", "serial", "serien",
        "reeks", "product", "produkt", "produk", "item", "artikelnummer",
        "batch", "charge", "lot", "barcode", "streepkode", "imei", "isbn",
    }
)

#: Every keyword table again, against the category it argues for and
#: whether it needs a number nearby before it means anything. A month name
#: is a date on its own; the word "account" is only interesting when there
#: is something numeric near it.
_KEYWORDS: tuple[tuple[frozenset[str], RiskCategory, bool, str], ...] = (
    (_MONEY_WORDS, RiskCategory.MONEY, True, "an amount of money"),
    (_PERCENT_WORDS, RiskCategory.PERCENTAGE, True, "a percentage"),
    (_MONTH_WORDS, RiskCategory.DATE, False, "a date"),
    (_WEEKDAY_WORDS, RiskCategory.DATE, False, "a day of the week"),
    (_DATE_WORDS, RiskCategory.DATE, True, "a date"),
    (_TIME_WORDS, RiskCategory.TIME, True, "a time of day"),
    (_TELEPHONE_WORDS, RiskCategory.TELEPHONE, True, "a telephone number"),
    (_ACCOUNT_WORDS, RiskCategory.ACCOUNT_NUMBER, True, "an account or reference number"),
    (_VERSION_WORDS, RiskCategory.VERSION_NUMBER, True, "a version number"),
    (_ADDRESS_WORDS, RiskCategory.ADDRESS, False, "an address"),
    (_QUANTITY_WORDS, RiskCategory.QUANTITY, True, "a measured quantity"),
    (_LEGAL_WORDS, RiskCategory.LEGAL_IDENTIFIER, True, "a legal or official identifier"),
    (_MEDICAL_WORDS, RiskCategory.MEDICAL_MEASUREMENT, True, "a medical measurement"),
    (_PRODUCT_WORDS, RiskCategory.PRODUCT_CODE, True, "a product or part code"),
)


# -- Ordinals, which name a day of the month ------------------------------
#
# "On the fifteenth" and "am ersten" are dates with no digit anywhere in
# them, so neither the number reader nor the shapes below can see them. The
# words are written out in full, one language at a time, for the same
# reason the keyword tables are.
#
# The first three are kept apart. "The second item", "die erste Frage" and
# "die eerste keer" are everyday phrases that have nothing to do with a
# calendar, so those ordinals only count as a date with a month, a weekday,
# a year or a date word within reach. From the fourth upwards, an ordinal
# standing on its own is nearly always a day of the month, and the module's
# bias says to flag it.

_LOW_ORDINAL_WORDS = frozenset(
    {
        "first", "second", "third",
        "eerste", "tweede", "derde",
    }
)

_HIGH_ORDINAL_WORDS = frozenset(
    {
        "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
        "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
        "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth",
        "thirtieth",
        "vierde", "vyfde", "sesde", "sewende", "agste", "agtste", "negende",
        "tiende", "elfde", "twaalfde", "dertiende", "veertiende", "vyftiende",
        "sestiende", "sewentiende", "agtiende", "negentiende", "twintigste",
        "dertigste",
    }
)

#: German ordinals decline, so "erste", "ersten", "erster" and "erstes" are
#: all the same word. The stem is matched and the ending allowed to vary.
#: "acht" is with the low ordinals rather than the high ones, although
#: eighth is not a low number, because "achte", "achten" and "achter" are
#: also forms of the everyday verb "achten", to pay attention to. On its own
#: the word is far more often the verb; next to a month or a lead-in such as
#: "am" it is the date.
_GERMAN_LOW_ORDINAL = re.compile(r"^(erst|zweit|dritt|acht)(e|en|er|es|em)$")
_GERMAN_HIGH_ORDINAL = re.compile(
    r"^(viert|f(ü|ue)nft|sechst|siebt|neunt|zehnt|elft|zw(ö|oe)lft"
    r"|(drei|vier|f(ü|ue)nf|sech|sieb|acht|neun)zehnt"
    r"|\w+und(zwanzig|drei(ß|ss)ig)st|(zwanzig|drei(ß|ss)ig)st)(e|en|er|es|em)$"
)

#: "twenty-first" and "een-en-twintigste" arrive as one word with hyphens in
#: it. The tens in front settle the question: nobody says "the twenty-first"
#: about anything but a day or a century.
_HYPHENATED_ORDINAL = re.compile(
    r"^(twenty|thirty)-(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)$"
    r"|^\w+-en-(twintig|dertig)ste$"
)

_ORDINAL_DATE_CONTEXT = _MONTH_WORDS | _WEEKDAY_WORDS | _DATE_WORDS

#: The short phrases that put a low ordinal on the calendar with nothing
#: else said: "am ersten", "on the first", "op die eerste". German "am" is
#: "an dem", and "am ersten" is a date however the sentence goes on.
_ORDINAL_DATE_LEAD_INS: tuple[tuple[str, ...], ...] = (
    ("am",), ("vom",), ("zum",), ("bis",),
    ("on", "the"), ("by", "the"), ("from", "the"), ("until", "the"),
    ("op", "die"), ("teen", "die"), ("vanaf", "die"), ("tot", "die"),
)


def _ordinal_rank(cleaned: str) -> str | None:
    """Whether this word is an ordinal, and whether it is a date on its own.

    Returns ``"high"`` for an ordinal that is a date by itself, ``"low"`` for
    one that needs a date word nearby, and ``None`` for anything else.
    """
    if cleaned in _HIGH_ORDINAL_WORDS or _HYPHENATED_ORDINAL.match(cleaned):
        return "high"
    if _GERMAN_HIGH_ORDINAL.match(cleaned):
        return "high"
    if cleaned in _LOW_ORDINAL_WORDS or _GERMAN_LOW_ORDINAL.match(cleaned):
        return "low"
    return None


# -- Shapes that speak for themselves ------------------------------------

_CLOCK = re.compile(r"^\d{1,2}[:h.]\d{2}(:\d{2})?$")
_ISO_DATE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$")
_WRITTEN_DATE = re.compile(r"^\d{1,2}[-/.]\d{1,2}([-/.]\d{2,4})?$")
#: A day and a month, or a month and a year, with nothing else: "12/05" and
#: "05/2024". Only the slash is accepted here. "12.05" with one full stop is
#: a decimal number to a German reader, and is left to the separator rule.
_SHORT_DATE = re.compile(r"^\d{1,2}/(\d{2}|\d{4})$")
_YEAR = re.compile(r"^(19|20)\d{2}$")
_ORDINAL_DAY = re.compile(r"^\d{1,2}(st|nd|rd|th|\.)$")
_VERSION = re.compile(r"^v\d+(\.\d+)*$|^\d+(\.\d+){2,}$", re.IGNORECASE)
_SEPARATED_NUMBER = re.compile(r"^\d{1,3}([.,]\d{3})*([.,]\d+)?$|^\d+[.,]\d+$")
_ALL_DIGITS = re.compile(r"^\d+$")
_PRODUCT_CODE = re.compile(r"^(?=.*[0-9])(?=.*[a-z])[a-z0-9]+([-/][a-z0-9]+)*$", re.IGNORECASE)
_UNIT_SUFFIX = re.compile(
    r"^\d+([.,]\d+)?\s*(kg|g|mg|mcg|ml|l|km|cm|mm|m|kb|mb|gb|tb|hz|khz|mhz|ghz"
    r"|mmol|mmhg|bpm|iu)$",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    """The form keywords are matched against: lower case, edges trimmed.

    Only punctuation at the edges is removed, and nothing inside the word is
    touched. A comma inside "1,500" is the whole question this module is
    careful about, so it must survive to be looked at. A minus sign in front
    of a digit survives too, written as a plain "-", because "-5" and "5"
    are different numbers.
    """
    stripped = unicodedata.normalize("NFKC", text).strip().casefold()
    while stripped and _is_trimmable(stripped[0]):
        if stripped[0] in MINUS_SIGNS and stripped[1:2].isdigit():
            break
        stripped = stripped[1:]
    if stripped[:1] in MINUS_SIGNS and stripped[1:2].isdigit():
        stripped = "-" + stripped[1:]
    while stripped and _is_trimmable(stripped[-1]):
        stripped = stripped[:-1]
    return stripped


#: Marks that carry meaning even at the edge of a word and must survive the
#: trimming. The per-cent sign counts as punctuation to Unicode, and trimming
#: it would turn a percentage into a bare number, which is the difference
#: between twenty per cent and twenty of something.
_MEANINGFUL_EDGES = frozenset("%" + _CURRENCY_MARKS)


def _is_trimmable(character: str) -> bool:
    if character in _MEANINGFUL_EDGES:
        return False
    return unicodedata.category(character).startswith("P")


def has_ambiguous_separator(text: str) -> bool:
    """Whether this number's separators cannot be read without the language.

    "1,500" is fifteen hundred to an English speaker and one and a half to a
    German or Afrikaans one, and "1.500" is the same disagreement the other
    way round. Nothing in a transcript of speech says which convention the
    person writing it down had in mind, so a number carrying a separator is
    never settled quietly, whatever else is or is not known about it.
    """
    cleaned = _clean(text)
    if not any(character in ",." for character in cleaned):
        return False
    return bool(_SEPARATED_NUMBER.match(cleaned) or _WRITTEN_DATE.match(cleaned))


def is_unit_word(text: str) -> bool:
    """Whether this word is a unit of measure, such as "kg" or "mm".

    Some units are spelled like a filler sound: "mm" is millimetres after a
    number and a hum anywhere else. Reconciliation asks this so that a filler
    standing inside an amount is still treated as a filler while the unit of
    a measurement is not.
    """
    return _clean(text) in _QUANTITY_WORDS


def is_numeric(text: str) -> bool:
    """Whether this word carries a number, written as digits or as words.

    Anything with a digit in it counts, because "R50", "3kg" and "v2" are all
    numbers wearing something else. Number words in the three languages count
    as well, which is what makes "fifteen thousand rand" money rather than
    two ordinary words next to a currency.
    """
    cleaned = _clean(text)
    if not cleaned:
        return False
    if any(character.isdigit() for character in cleaned):
        return True
    return read_number(cleaned) is not None


def numbers_disagree(first: str, second: str) -> bool:
    """Whether two candidate spellings name different numbers.

    Only a difference in the number itself counts. "twenty-five" against "25"
    is one number written two ways and is not a disagreement about anything,
    which is precisely the distinction the review queue must not lose.
    Where either text is not a plain number, or where the separators make it
    unreadable, the answer is that they do disagree, because the alternative
    is to decide quietly that two unreadable numbers were the same.
    """
    if _clean(first) == _clean(second):
        return False
    if has_ambiguous_separator(first) or has_ambiguous_separator(second):
        return True
    left = read_number(first)
    right = read_number(second)
    if left is None or right is None:
        return is_numeric(first) or is_numeric(second)
    return left != right


def _self_evident(cleaned: str) -> list[tuple[RiskCategory, str]]:
    """What one word says about itself, before its neighbours are consulted."""
    found: list[tuple[RiskCategory, str]] = []
    if not cleaned:
        return found
    if "%" in cleaned:
        found.append((RiskCategory.PERCENTAGE, "a percentage"))
    if any(mark in cleaned for mark in _CURRENCY_MARKS):
        found.append((RiskCategory.MONEY, "an amount of money"))
    elif cleaned.startswith("r") and cleaned[1:2].isdigit():
        # The rand is written without a space in South African usage, and it
        # is the currency most of these recordings are in.
        found.append((RiskCategory.MONEY, "an amount of money"))
    if _CLOCK.match(cleaned):
        found.append((RiskCategory.TIME, "a time of day"))
    if _ISO_DATE.match(cleaned):
        found.append((RiskCategory.DATE, "a date"))
    elif _WRITTEN_DATE.match(cleaned) and cleaned.count(".") + cleaned.count("/") >= 2:
        found.append((RiskCategory.DATE, "a date"))
    elif _SHORT_DATE.match(cleaned):
        found.append((RiskCategory.DATE, "a date"))
    if _ORDINAL_DAY.match(cleaned) or _YEAR.match(cleaned):
        found.append((RiskCategory.DATE, "a date"))
    if _VERSION.match(cleaned):
        found.append((RiskCategory.VERSION_NUMBER, "a version number"))
    if _UNIT_SUFFIX.match(cleaned):
        found.append((RiskCategory.QUANTITY, "a measured quantity"))
    if _ALL_DIGITS.match(cleaned) and len(cleaned) >= LONG_DIGIT_RUN:
        found.append((RiskCategory.TELEPHONE, "a long run of digits"))
        found.append((RiskCategory.ACCOUNT_NUMBER, "a long run of digits"))
    if has_ambiguous_separator(cleaned):
        found.append((RiskCategory.QUANTITY, "a number whose decimal separator is ambiguous"))
    if (
        not found
        and _PRODUCT_CODE.match(cleaned)
        and not _ORDINAL_DAY.match(cleaned)
        and any(character.isdigit() for character in cleaned)
        and any(character.isalpha() for character in cleaned)
    ):
        found.append((RiskCategory.PRODUCT_CODE, "letters and digits together, as codes are"))
    return found


#: The words for "one" that are as often a pronoun as a number: English
#: "one" and Afrikaans "een". They are not a quantity on their own. German
#: "ein" and "eine" are articles and are not read as numbers at all. German
#: "eins" is kept as a quantity, because it is only ever the number counted
#: or named, as in "Zimmer eins".
_PRONOUN_ONES = frozenset(("one", "een"))


def find_risks(words: Sequence[str]) -> tuple[RiskFinding, ...]:
    """Every high-risk stretch in a sequence of words, in order.

    The sequence is normally a handful of words around the position being
    reconciled rather than a whole transcript, because a keyword only reaches
    :data:`RISK_WINDOW` words and looking further would join unrelated
    sentences together.
    """
    cleaned = [_clean(word) for word in words]
    numeric = [is_numeric(word) for word in words]
    findings: list[RiskFinding] = []

    for index, word in enumerate(cleaned):
        for category, detail in _self_evident(word):
            findings.append(RiskFinding(category, (index,), detail))

    for index, word in enumerate(cleaned):
        for table, category, needs_number, detail in _KEYWORDS:
            if word not in table:
                continue
            reach = _numbers_near(numeric, index)
            if needs_number and not reach:
                continue
            positions = tuple(sorted({index, *reach, *_between(index, reach)}))
            findings.append(RiskFinding(category, positions, detail))

    findings.extend(_spoken_digit_runs(cleaned))
    findings.extend(_ordinal_dates(cleaned))
    # A number nobody else claimed is still a number somebody said, and a
    # wrong one is still wrong. Quantity is the honest description of it.
    # Anything with a digit in it counts, whether or not it can be read as
    # one number: "1/2", "3/4" and "24/7" could not be read and used to slip
    # through here, and a wrong fraction is as wrong as a wrong integer.
    # "One" on its own does not count. In "the one who taught me" or "That
    # one" it is a pronoun, not a value, and flagging every such word buried
    # the real values in the review list. Every other number word still
    # counts: "twelve" in "we need twelve more" is a value, and leaving it
    # out would let a second opinion overwrite "fifteen" with "fifty". "One"
    # next to a unit or a currency is claimed above already.
    claimed = {position for finding in findings for position in finding.positions}
    for index, word in enumerate(cleaned):
        if index in claimed or not numeric[index] or word in _PRONOUN_ONES:
            continue
        if any(character.isdigit() for character in word) or read_number(word) is not None:
            findings.append(RiskFinding(RiskCategory.QUANTITY, (index,), "a number"))
    return tuple(findings)


def _ordinal_dates(cleaned: Sequence[str]) -> list[RiskFinding]:
    """Find days of the month named by an ordinal word rather than a digit.

    "The fifteenth" is a date on its own. "The first" is only a date when a
    month, a weekday, a year or a date word is within reach, or when it is
    led in by a phrase such as "on the" or German "am". The finding then
    covers the ordinal and everything up to that word, so that "first of
    March" is protected in the middle as well as at its ends.
    """
    findings: list[RiskFinding] = []
    for index, word in enumerate(cleaned):
        rank = _ordinal_rank(word)
        if rank is None:
            continue
        if rank == "high":
            findings.append(RiskFinding(RiskCategory.DATE, (index,), "a day of the month"))
            continue
        start = max(0, index - RISK_WINDOW)
        end = min(len(cleaned), index + RISK_WINDOW + 1)
        context = tuple(
            position
            for position in range(start, end)
            if position != index
            and (cleaned[position] in _ORDINAL_DATE_CONTEXT or _YEAR.match(cleaned[position]))
        )
        if not context:
            for lead_in in _ORDINAL_DATE_LEAD_INS:
                before = tuple(cleaned[max(0, index - len(lead_in)) : index])
                if before == lead_in:
                    context = (index - len(lead_in),)
                    break
        if context:
            positions = tuple(sorted({index, *context, *_between(index, context)}))
            findings.append(RiskFinding(RiskCategory.DATE, positions, "a day of the month"))
    return findings


def _numbers_near(numeric: Sequence[bool], index: int) -> tuple[int, ...]:
    """The positions within reach of ``index`` that carry a number."""
    start = max(0, index - RISK_WINDOW)
    end = min(len(numeric), index + RISK_WINDOW + 1)
    return tuple(position for position in range(start, end) if numeric[position])


def _between(index: int, reach: Sequence[int]) -> tuple[int, ...]:
    """Everything lying between a keyword and the numbers it claimed.

    "fifteen thousand five hundred rand" has to be protected in the middle as
    well as at its ends, or the rules would be free to settle "hundred" while
    refusing to settle the words either side of it.
    """
    if not reach:
        return ()
    low = min(index, min(reach))
    high = max(index, max(reach))
    return tuple(range(low, high + 1))


def _spoken_digit_runs(cleaned: Sequence[str]) -> list[RiskFinding]:
    """Find numbers being read out digit by digit, as people read out codes."""
    findings: list[RiskFinding] = []
    run: list[int] = []
    for index, word in enumerate([*cleaned, ""]):
        digits = _ALL_DIGITS.match(word) and len(word) <= 3
        number = read_number(word) if word else None
        if digits or (number is not None and number < 100 and word.isalpha()):
            run.append(index)
            continue
        if len(run) >= SPOKEN_DIGIT_RUN:
            positions = tuple(run)
            detail = "a number being read out digit by digit"
            findings.append(RiskFinding(RiskCategory.TELEPHONE, positions, detail))
            findings.append(RiskFinding(RiskCategory.ACCOUNT_NUMBER, positions, detail))
        run = []
    return findings


def _findings_at(words: Sequence[str], position: int) -> tuple[RiskFinding, ...]:
    """Every finding covering one position, found from its neighbourhood alone.

    Only a window either side of the position is examined, which is both
    faster and more honest than scanning a whole recording: a keyword four
    sentences away is not evidence about this word.
    """
    if not 0 <= position < len(words):
        return ()
    start = max(0, position - RISK_WINDOW)
    end = min(len(words), position + RISK_WINDOW + 1)
    local = position - start
    return tuple(
        finding for finding in find_risks(words[start:end]) if finding.covers(local)
    )


def risk_at(words: Sequence[str], position: int) -> tuple[RiskCategory, ...]:
    """The categories covering one position, given the words around it.

    This is what the reconciliation rules call before they decide whether
    they are allowed to settle a disagreement on their own.
    """
    found: list[RiskCategory] = []
    for finding in _findings_at(words, position):
        if finding.category not in found:
            found.append(finding.category)
    return tuple(found)


def risk_details_at(words: Sequence[str], position: int) -> tuple[str, ...]:
    """The plain-English reasons this position is high risk, for the review window."""
    details: list[str] = []
    for finding in _findings_at(words, position):
        if finding.detail not in details:
            details.append(finding.detail)
    return tuple(details)


def risk_categories(words: Sequence[str]) -> tuple[tuple[RiskCategory, ...], ...]:
    """The categories covering every position in a sequence, in order."""
    findings = find_risks(words)
    per_position: list[list[RiskCategory]] = [[] for _ in words]
    for finding in findings:
        for position in finding.positions:
            if 0 <= position < len(words) and finding.category not in per_position[position]:
                per_position[position].append(finding.category)
    return tuple(tuple(found) for found in per_position)
