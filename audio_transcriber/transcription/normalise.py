"""Recognising two spellings of the same spoken thing.

Several services listen to the same recording and each writes down what it
heard in its own house style. One writes "twenty-five", another "25". One
writes "Jürgen", another "Jurgen". One writes "data base", another
"database". One writes "don't", another "dont". None of them misheard
anything. If the application compared those strings directly it would call
every one of them a disagreement, send them all for a second opinion, and
put them in front of a person to settle. The review queue would fill up
with differences that are not differences, and the real disagreements
would be lost in the noise.

So this module builds a second form of a piece of text, used for nothing
except comparison. Two texts whose comparison forms match are the same
spoken evidence, whatever they look like on the page.

**The original provider text is never modified.** Not here and not
anywhere downstream. The comparison form is a separate value that exists
alongside the text a service returned, and it is never shown to anyone and
never exported. That rule is what keeps the transcript explainable: a
person reviewing a word must see exactly what each service said, not a
tidied-up version of it that this module invented.

The forms are built in stages, each one more invasive than the last:

1. Case and Unicode. "Word" and "word" and a full-width "２５" are settled
   here.
2. Punctuation, apostrophes and hyphens. "don't" and "dont" meet here.
3. German character equivalence, in both directions: "ü" and "ue" are the
   same letter, and so are "ö"/"oe", "ä"/"ae" and "ß"/"ss".
4. Numbers. "twenty five", "twenty-five" and "25" become the same, and so
   do "1,000" and "1000".
5. Word boundaries and contractions. "data base" and "database" meet here,
   and so do "cannot" and "can not".

:func:`equivalence_kind` reports the stage at which two texts meet, because
the rest of the application needs to know *how* they differ, not only that
they are the same. A difference that is only in how the word was written
leaves the original timing fitting perfectly; a genuinely different word in
the same place does not, and has to be treated as a substitution.

A pair of texts can qualify at more than one stage, and which one is
reported is decided by how much it constrains the timing rather than by
which came first. "data-base" against "data base" is a difference in
punctuation, but it is also one word against two, and a caller told only
about the punctuation would give the replacement a span it is not entitled
to. Where the number of words changes, nobody knows where inside the span
the join between them falls, and the answer has to say so.

Two decisions in here are deliberately conservative, because being wrong
about a number is far worse than failing to notice that two numbers are
the same.

The first is that a full stop between digits is left alone. In English
"1.500" is one and a half; in German it is fifteen hundred. Guessing gives
a plausible, wrong number that reads perfectly well, which is exactly the
sort of silent damage the specification forbids. A comma between digits is
treated as a thousands separator, because no supported language uses it as
a decimal point in a transcript of speech.

The second is that a run of number words is only read as a number when it
is grammatical. "twenty five" is twenty-five, but "one two three" is a
telephone number being read out and is left as words, and "nineteen eighty
four" is a year and is left alone as well. Reading those as sums would
invent numbers nobody said.

Nothing here depends on Qt or on any provider library, so it can be tested
on its own.
"""

from __future__ import annotations

import unicodedata
from enum import Enum
from functools import lru_cache
from typing import Callable, Iterable, Sequence


class EquivalenceKind(str, Enum):
    """How two pieces of text are the same, if they are the same at all.

    The members are declared in the order the stages are applied, from the
    smallest difference to the largest, and :func:`equivalence_kind`
    reports the first one that fits.
    """

    IDENTICAL = "identical"
    """Character for character the same."""

    CASE_ONLY = "case_only"
    """The same once case and Unicode form are set aside."""

    PUNCTUATION_ONLY = "punctuation_only"
    """The same once punctuation, apostrophes and hyphens are set aside."""

    SPELLING_VARIANT = "spelling_variant"
    """The same word spelled another way, such as "Jürgen" and "Jurgen"."""

    NUMBER_FORMAT = "number_format"
    """The same number written another way, such as "twenty five" and "25"."""

    COMPOUND = "compound"
    """The same words, joined or split differently, or contracted."""

    DIFFERENT = "different"
    """Not the same spoken evidence."""

    @property
    def is_equivalent(self) -> bool:
        """Whether the two texts are the same spoken evidence.

        Everything except :attr:`DIFFERENT` is cosmetic in the sense that
        matters to the timing rules: the words were heard in the same place
        and the span a service measured still covers them. Only
        :attr:`DIFFERENT` means a genuinely other word, whose span may not
        fit.
        """
        return self is not EquivalenceKind.DIFFERENT

    @property
    def display_name(self) -> str:
        return _EQUIVALENCE_DISPLAY_NAMES[self]


_EQUIVALENCE_DISPLAY_NAMES: dict[EquivalenceKind, str] = {
    EquivalenceKind.IDENTICAL: "Identical",
    EquivalenceKind.CASE_ONLY: "A difference in capitals only",
    EquivalenceKind.PUNCTUATION_ONLY: "A difference in punctuation only",
    EquivalenceKind.SPELLING_VARIANT: "The same word spelled another way",
    EquivalenceKind.NUMBER_FORMAT: "The same number written another way",
    EquivalenceKind.COMPOUND: "The same words joined or split differently",
    EquivalenceKind.DIFFERENT: "A different word",
}


# -- The characters that get unified -------------------------------------

#: Every shape an apostrophe arrives in. Services differ, and so do the
#: keyboards of the people who wrote the vocabulary lists. They are all
#: dropped rather than replaced, so "don't" and "dont" meet.
_APOSTROPHES = frozenset("'’‘ʼʹ‛′`´")

#: Every shape a hyphen or a joining slash arrives in. These become spaces
#: rather than disappearing, so that "twenty-five" becomes two words and
#: can be read as a number, and "data-base" waits until the last stage to
#: meet "database". A slash standing between two digits is the exception:
#: it is kept, because "1/2" is a half and "24/7" is round the clock, and
#: neither is the number the digits spell when run together.
_DASHES = frozenset("-‐‑‒–—―−/_")

#: The German letters and what they are also written as. Note that
#: ``str.casefold`` already turns "ß" into "ss", so by the time this table
#: is used the eszett has usually gone; it is listed anyway so that the
#: table is complete when read.
_GERMAN_FOLD: dict[str, str] = {
    "ü": "ue",
    "ö": "oe",
    "ä": "ae",
    "ß": "ss",
}

#: The same letters as they appear when somebody simply drops the dots, as
#: in "Jurgen" for "Jürgen". This is the second direction, and it is only
#: ever applied when one of the two texts being compared actually contains
#: a German letter. Applying it always would be a disaster: "toe" and "to"
#: would become the same word, and so would "doe" and "do".
_DROPPED_DIGRAPHS: tuple[tuple[str, str], ...] = (
    ("ue", "u"),
    ("oe", "o"),
    ("ae", "a"),
)

_GERMAN_LETTERS = frozenset("üöäÜÖÄßẞ")


# -- Number words --------------------------------------------------------
#
# The tables are written out in full, one language at a time, because a
# clever generator would be shorter and much harder to check. Anyone can
# read these and see whether "sewentien" is in the right place.

_ENGLISH_UNITS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}

_GERMAN_UNITS: dict[str, int] = {
    "null": 0, "ein": 1, "eine": 1, "eins": 1, "zwei": 2, "drei": 3,
    "vier": 4, "fünf": 5, "sechs": 6, "sieben": 7, "acht": 8, "neun": 9,
    "zehn": 10, "elf": 11, "zwölf": 12, "dreizehn": 13, "vierzehn": 14,
    "fünfzehn": 15, "sechzehn": 16, "siebzehn": 17, "achtzehn": 18,
    "neunzehn": 19,
}

#: Afrikaans "ag" for eight is deliberately left out. It is also the
#: everyday interjection, and turning every sighed "ag" in a recording into
#: the number 8 would be worse than missing the odd numeral. The standard
#: spelling "agt" is here and does the work.
_AFRIKAANS_UNITS: dict[str, int] = {
    "nul": 0, "een": 1, "twee": 2, "drie": 3, "vier": 4, "vyf": 5,
    "ses": 6, "sewe": 7, "agt": 8, "nege": 9, "tien": 10, "elf": 11,
    "twaalf": 12, "dertien": 13, "veertien": 14, "vyftien": 15,
    "sestien": 16, "sewentien": 17, "agtien": 18, "negentien": 19,
}

_ENGLISH_TENS: dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}

_GERMAN_TENS: dict[str, int] = {
    "zwanzig": 20, "dreißig": 30, "vierzig": 40, "fünfzig": 50,
    "sechzig": 60, "siebzig": 70, "achtzig": 80, "neunzig": 90,
}

_AFRIKAANS_TENS: dict[str, int] = {
    "twintig": 20, "dertig": 30, "veertig": 40, "vyftig": 50,
    "sestig": 60, "sewentig": 70, "tagtig": 80, "negentig": 90,
}

_HUNDRED_WORDS = ("hundred", "hundert", "honderd")
_THOUSAND_WORDS = ("thousand", "tausend", "duisend")

#: "und" and "en" join a unit to a tens word, as in "fünfundzwanzig" and
#: "vyfentwintig". English "and" never does that in speech people actually
#: use, so it is kept apart: it may only follow a hundred or a thousand, as
#: in "one hundred and one" and "two thousand and five". Letting English
#: "and" join freely would turn "five and ten" into fifteen.
#:
#: "und" and "en" also do the job "and" does after a scale word, as in
#: "tausend und eins" and "honderd en vyf". That is the one place a unit may
#: follow them, and :func:`_accumulate` allows it only there, so that "vyf
#: en ses" stays two numbers rather than becoming eleven.
_COMPOUND_JOINERS = ("und", "en")
_SCALE_JOINERS = ("and",)

# The kinds an atom of a numeral can be. They are plain strings because
# they are only ever compared with each other.
_UNIT = "unit"
_TENS = "tens"
_HUNDRED = "hundred"
_THOUSAND = "thousand"
_JOIN_COMPOUND = "join_compound"
_JOIN_SCALE = "join_scale"

#: What may follow what. This is the grammar that keeps "one two three"
#: from being read as six.
_MAY_FOLLOW: dict[str, frozenset[str]] = {
    _UNIT: frozenset({_HUNDRED, _THOUSAND, _JOIN_COMPOUND}),
    _TENS: frozenset({_UNIT, _THOUSAND}),
    _HUNDRED: frozenset({_UNIT, _TENS, _THOUSAND, _JOIN_SCALE, _JOIN_COMPOUND}),
    _THOUSAND: frozenset({_UNIT, _TENS, _HUNDRED, _JOIN_SCALE, _JOIN_COMPOUND}),
    _JOIN_COMPOUND: frozenset({_TENS}),
    _JOIN_SCALE: frozenset({_UNIT, _TENS}),
}

_SCALE_KINDS = frozenset({_HUNDRED, _THOUSAND})

_JOINER_KINDS = frozenset({_JOIN_COMPOUND, _JOIN_SCALE})


def _german_fold(text: str) -> str:
    for letter, replacement in _GERMAN_FOLD.items():
        text = text.replace(letter, replacement)
    return text


def _drop_german_letters(text: str) -> str:
    """Turn "fünf", "fuenf" and "funf" all into the same thing.

    The dots are removed by decomposing the letter and throwing the
    combining mark away, which leaves ordinary accented text alone in every
    other respect.
    """
    stripped = "".join(
        character
        for character in unicodedata.normalize("NFD", text)
        if not unicodedata.combining(character)
    )
    stripped = unicodedata.normalize("NFC", stripped)
    for digraph, replacement in _DROPPED_DIGRAPHS:
        stripped = stripped.replace(digraph, replacement)
    return stripped


def _atom_table(dropped: bool) -> dict[str, tuple[str, int]]:
    """Build the numeral lookup in the same spelling the text will be in.

    The tables above are written with real German letters because that is
    how a person reads them. By the time a numeral is looked up, the text
    has already been through the German folding, so the keys have to go
    through exactly the same folding or nothing would ever match.
    """
    fold = _drop_german_letters if dropped else _german_fold

    def key(word: str) -> str:
        return fold(unicodedata.normalize("NFKC", word).casefold())

    table: dict[str, tuple[str, int]] = {}
    for words in (_ENGLISH_UNITS, _GERMAN_UNITS, _AFRIKAANS_UNITS):
        for word, value in words.items():
            table[key(word)] = (_UNIT, value)
    for words in (_ENGLISH_TENS, _GERMAN_TENS, _AFRIKAANS_TENS):
        for word, value in words.items():
            table[key(word)] = (_TENS, value)
    for word in _HUNDRED_WORDS:
        table[key(word)] = (_HUNDRED, 100)
    for word in _THOUSAND_WORDS:
        table[key(word)] = (_THOUSAND, 1000)
    for word in _COMPOUND_JOINERS:
        table[key(word)] = (_JOIN_COMPOUND, 0)
    for word in _SCALE_JOINERS:
        table[key(word)] = (_JOIN_SCALE, 0)
    return table


_ATOMS = _atom_table(dropped=False)
_DROPPED_ATOMS = _atom_table(dropped=True)


# -- Contractions --------------------------------------------------------
#
# The apostrophe has already gone by the time these are used, so the keys
# are the apostrophe-free spellings. That also means "dont" written without
# its apostrophe is handled by the same entry, which is the point.

_CONTRACTIONS: dict[str, str] = {
    "cannot": "can not",
    "cant": "can not",
    "shant": "shall not",
    "wont": "will not",
    "dont": "do not",
    "doesnt": "does not",
    "didnt": "did not",
    "isnt": "is not",
    "arent": "are not",
    "wasnt": "was not",
    "werent": "were not",
    "havent": "have not",
    "hasnt": "has not",
    "hadnt": "had not",
    "wouldnt": "would not",
    "couldnt": "could not",
    "shouldnt": "should not",
    "mustnt": "must not",
    "its": "it is",
    "thats": "that is",
    "whats": "what is",
    "whos": "who is",
    "theres": "there is",
    "heres": "here is",
    "hes": "he is",
    "shes": "she is",
    "im": "i am",
    "ive": "i have",
    "youre": "you are",
    "youve": "you have",
    "theyre": "they are",
    "theyve": "they have",
    "weve": "we have",
    "lets": "let us",
}

#: Contractions that are left alone, because without their apostrophe they
#: are ordinary English words and expanding them would make two genuinely
#: different words compare equal: "were" and "we're", "well" and "we'll",
#: "shed" and "she'd", "ill" and "I'll", "hell" and "he'll", "id" and
#: "I'd". The list is here so that nobody adds them back by accident.
_LEFT_ALONE = ("were", "well", "shed", "hed", "wed", "id", "ill", "hell", "shell")

#: "its" is in the table above even though it is also the possessive. That
#: is on purpose: "its" and "it's" sound identical, so as *acoustic
#: evidence* they are the same thing, and which one to print is a question
#: for the reconciliation rules rather than for this module.


# -- Building the comparison forms ---------------------------------------


@lru_cache(maxsize=100_000)
def _case_form(text: str) -> str:
    """Case and Unicode settled, and nothing else."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(folded.split())


def _between_digits(text: str, index: int) -> bool:
    return (
        index > 0
        and index + 1 < len(text)
        and text[index - 1].isdigit()
        and text[index + 1].isdigit()
    )


@lru_cache(maxsize=100_000)
def _plain_form(text: str) -> str:
    """Punctuation, apostrophes and hyphens settled as well.

    A comma or a full stop between two digits survives this stage. They are
    not punctuation there, they are part of a number, and the number stage
    below is the only place that has any business deciding what they mean.
    """
    source = _case_form(text)
    characters: list[str] = []
    for index, character in enumerate(source):
        if character in ",./" and _between_digits(source, index):
            # The slash is kept for the same reason as the comma and the
            # full stop: "1/2" and "12" are different things, and turning
            # the slash into a space would make the first look like the
            # digits one and two read out separately. This test comes
            # before the dash test because the slash is also a dash.
            characters.append(character)
        elif character in _DASHES:
            characters.append(" ")
        elif character in _APOSTROPHES:
            continue
        elif unicodedata.category(character).startswith("P"):
            continue
        else:
            characters.append(character)
    return " ".join("".join(characters).split())


@lru_cache(maxsize=100_000)
def _spelling_form(text: str, dropped: bool) -> str:
    """German characters settled, in one direction or the other."""
    plain = _plain_form(text)
    return _drop_german_letters(plain) if dropped else _german_fold(plain)


@lru_cache(maxsize=100_000)
def _number_form(text: str, dropped: bool) -> str:
    """Numbers settled, whether they were written in words or in digits."""
    return _read_numbers(_spelling_form(text, dropped), dropped)


@lru_cache(maxsize=100_000)
def _compound_form(text: str, dropped: bool) -> str:
    """Contractions and word boundaries settled: the most permissive form.

    The words are run together so that "data base" meets "database". One
    boundary survives: the one between a word ending in a digit and a word
    starting with one. "1 2" is two numbers and "12" is one, and running
    them together would make a half and twelve, or "twenty 5" and "205",
    the same evidence. A digit next to a letter still joins, so "3kg" and
    "3 kg" meet as before.

    Two shapes of digit run are one number written with spaces rather than
    two numbers, and keep joining. Thousands grouped by a space, "10 000"
    or "100 000 000", which is how South African, German and ISO writing
    groups them and which a service writes as "10,000" or "10000" as often
    as not; and a telephone number read out in blocks, "082 123 4567".
    """
    words = _number_form(text, dropped).split()
    expanded: list[str] = []
    for word in words:
        expanded.extend(_CONTRACTIONS.get(word, word).split())
    joined: list[str] = []
    position = 0
    while position < len(expanded):
        run = _digit_run(expanded, position)
        # A run that does not fit as a whole may still begin with a number
        # that does, "1 000 000" followed by "5", so it is shortened from the
        # end until what is left is one number, or too short to be one.
        while len(run) >= 2 and not _is_one_number_in_blocks(run):
            run = run[:-1]
        if len(run) >= 2:
            if joined and joined[-1][-1:].isdigit():
                joined.append(" ")
            joined.append("".join(run))
            position += len(run)
            continue
        word = expanded[position]
        if joined and joined[-1][-1:].isdigit() and word[:1].isdigit():
            joined.append(" ")
        joined.append(word)
        position += 1
    return "".join(joined)


def _digit_run(words: list[str], start: int) -> list[str]:
    """The digit blocks beginning at ``start``, if there are two or more.

    The first block may carry a currency sign or other letters in front of
    its digits, "R10" or "$5", because an amount is written that way; every
    later block is digits only.
    """
    first = words[start]
    if not first[-1:].isdigit():
        return []
    run = [first]
    for word in words[start + 1 :]:
        if not word.isdigit():
            break
        run.append(word)
    return run if len(run) >= 2 else []


def _is_one_number_in_blocks(run: list[str]) -> bool:
    """Whether these digit words are one number written in blocks.

    Grouped thousands have one to three digits first and exactly three in
    every later block. A telephone number begins with a zero and is three or
    more blocks of three or four digits. "1 2", "20 5" and "2024 05" fit
    neither and stay apart, and so do three years or three amounts read in
    a row, which a looser telephone rule would run together.
    """
    leading = len(run[0]) - len(run[0].rstrip("0123456789"))
    if 1 <= leading <= 3 and all(len(block) == 3 for block in run[1:]):
        return True
    return (
        len(run) >= 3
        and run[0].isdigit()
        and run[0].startswith("0")
        and all(3 <= len(block) <= 4 for block in run)
    )


@lru_cache(maxsize=100_000)
def _has_german_letter(text: str) -> bool:
    return any(character in _GERMAN_LETTERS for character in unicodedata.normalize("NFKC", text))


# -- Reading numbers -----------------------------------------------------


def _split_into_atoms(word: str, table: dict[str, tuple[str, int]]) -> tuple[str, ...] | None:
    """Break a joined numeral such as "fünfundzwanzig" into its parts.

    German and Afrikaans write their compound numerals as one word, so
    there is nothing to split on. The word is covered from the front with
    the longest known part that still leaves a coverable remainder, which
    finds "fünf", "und", "zwanzig" without a dictionary.

    Every letter has to be accounted for. That requirement is what stops
    ordinary words being mistaken for numerals: "often" contains "ten" but
    nothing can cover the "of" in front of it, so the whole attempt fails.
    """
    length = len(word)
    covered: list[tuple[str, ...] | None] = [None] * (length + 1)
    covered[length] = ()
    for start in range(length - 1, -1, -1):
        for end in range(length, start, -1):
            piece = word[start:end]
            if piece in table and covered[end] is not None:
                covered[start] = (piece,) + covered[end]
                break
    return covered[0]


def _accumulate(items: Sequence[tuple[str, int]]) -> int | None:
    """Turn a grammatical run of numeral parts into the number it names."""
    total = 0
    current = 0
    previous: str | None = None
    before_previous: str | None = None
    for kind, value in items:
        if previous is None:
            if kind in _JOINER_KINDS:
                return None
        elif kind not in _MAY_FOLLOW[previous]:
            # "und" and "en" normally lead to a tens word. Straight after a
            # hundred or a thousand they may lead to a unit as well, as in
            # "honderd en een", and nowhere else, because "vyf en ses" is
            # not eleven.
            scale_joiner = (
                kind == _UNIT
                and previous == _JOIN_COMPOUND
                and before_previous in _SCALE_KINDS
            )
            if not scale_joiner:
                return None
        if kind in (_UNIT, _TENS):
            current += value
        elif kind == _HUNDRED:
            # A scale atom normally carries its own value, 100 or 1000. A
            # joined compound such as "zweihundert" or "tweeduisend" arrives
            # as one atom carrying the whole product, and that product is
            # used as it is, so that "zweitausend fünf" reads as 2005 rather
            # than as a thousand and five.
            current = current * 100 if current else value
        elif kind == _THOUSAND:
            total += current * 1000 if current else value
            current = 0
        before_previous = previous
        previous = kind
    if previous is None or previous in _JOINER_KINDS:
        return None
    return total + current


@lru_cache(maxsize=20_000)
def _word_atom(word: str, dropped: bool) -> tuple[str, int] | None:
    """What one word contributes to a number, if it contributes anything.

    A joined compound counts as whatever its last part was, so that
    "fünfundzwanzig tausend" still reads as twenty-five thousand.
    """
    table = _DROPPED_ATOMS if dropped else _ATOMS
    direct = table.get(word)
    if direct is not None:
        return direct
    if not word.isalpha():
        return None
    parts = _split_into_atoms(word, table)
    if parts is None or len(parts) < 2:
        return None
    items = [table[part] for part in parts]
    value = _accumulate(items)
    if value is None:
        return None
    return (items[-1][0], value)


def _digit_group(word: str) -> str | None:
    """Return a digit string with its thousands separators taken out.

    Only commas are treated as separators, and only where they group three
    digits. A full stop is left where it is, because it means one thing in
    English and the opposite in German, and a wrong guess produces a number
    that looks entirely reasonable and is out by a factor of a thousand.
    """
    if word.isdigit():
        return word
    if "," not in word:
        return None
    parts = word.split(",")
    if len(parts) < 2 or not parts[0].isdigit() or not 1 <= len(parts[0]) <= 3:
        return None
    if not all(part.isdigit() and len(part) == 3 for part in parts[1:]):
        return None
    return "".join(parts)


def _read_numbers(form: str, dropped: bool) -> str:
    """Replace runs of number words, and grouped digits, with plain digits.

    Digits never join a run of words. Two numbers standing next to each
    other, as in "in 2024 25 people came", are two numbers, and adding them
    together would produce a number nobody said.
    """
    words = form.split()
    atoms = [_word_atom(word, dropped) for word in words]
    output: list[str] = []
    start = 0
    while start < len(words):
        digits = _digit_group(words[start])
        if digits is not None:
            output.append(digits)
            start += 1
            continue
        if atoms[start] is None:
            output.append(words[start])
            start += 1
            continue
        end = start
        while end < len(words) and atoms[end] is not None:
            end += 1
        run = [atom for atom in atoms[start:end] if atom is not None]
        value = _read_run(run)
        if value is None:
            output.extend(words[start:end])
        else:
            output.append(str(value))
        start = end
    return " ".join(output)


def _read_run(run: Sequence[tuple[str, int]]) -> int | None:
    """Read a whole run of numeral words, or refuse to read it at all.

    Nothing partial is accepted. Reading "nineteen eighty four" as a
    nineteen followed by an eighty-four would drop the fact that the three
    words belong together, and the words are left exactly as they are
    instead. A run also has to name a quantity: a lone "thousand" with
    nothing in front of it is the ordinary word, not the number 1000. A
    joined compound such as "zweihundert" has its quantity inside it, and
    shows that by carrying a value other than the bare scale.
    """
    if not any(
        kind in (_UNIT, _TENS) or (kind in _SCALE_KINDS and value not in (100, 1000))
        for kind, value in run
    ):
        return None
    return _accumulate(run)


# -- What the rest of the application calls ------------------------------


def normalise(text: str) -> str:
    """Return the comparison form of one word or short phrase.

    This is the most permissive of the forms: case, punctuation, German
    spelling, number format, contractions and word boundaries have all been
    settled. Two texts with the same comparison form are certainly the same
    spoken evidence.

    The reverse does not quite hold. A dropped umlaut, as in "Jurgen" for
    "Jürgen", is only recognised by :func:`are_equivalent`, which knows to
    look for it when one of the two texts actually contains a German
    letter. Doing it here would mean folding "ue" into "u" for every text,
    which would quietly make "toe" and "to" the same word.

    Punctuation on its own normalises to an empty string.
    """
    return _compound_form(text, False)


def normalise_sequence(texts: Sequence[str]) -> tuple[str, ...]:
    """Return the comparison form of each text, keeping their positions.

    Punctuation comes back as an empty string rather than being dropped, so
    that the result can still be indexed against the sequence that went in.

    These forms cannot be pasted together to compare a phrase. "twenty" and
    "five" normalise to "20" and "5", and joining those gives "205" rather
    than "25". Use :func:`normalise_phrase` when several words on one side
    have to be compared with one word on the other.
    """
    return tuple(normalise(text) for text in texts)


def normalise_phrase(texts: Iterable[str]) -> str:
    """Return the comparison form of several words taken together.

    This is what makes a many-to-one difference visible: "data base" and
    "database" both give "database", and "twenty five" and "25" both give
    "25".
    """
    return normalise(" ".join(texts))


def is_punctuation_only(text: str) -> bool:
    """Whether this text carries no word for alignment to work with.

    True for punctuation, for spacing, and for an empty string. Services
    that return punctuation as its own entry rely on this: the entries are
    kept, because they carry timing, but they take no part in alignment.
    """
    return not normalise(text)


def read_number(text: str) -> int | None:
    """Return the whole number this text names, or ``None`` if it is not one.

    Works for digits and for number words in English, German and Afrikaans,
    including the joined compounds those last two write, such as
    "fünfundzwanzig". Anything with more to it than a single number, such
    as "25 people", comes back as ``None``.
    """
    form = _number_form(text, False)
    if form.isdigit():
        return int(form)
    if _has_german_letter(text):
        dropped = _number_form(text, True)
        if dropped.isdigit():
            return int(dropped)
    return None


def _same_form(
    first: str,
    second: str,
    builder: Callable[[str, bool], str],
    allow_dropped: bool,
) -> bool:
    if builder(first, False) == builder(second, False):
        return True
    if not allow_dropped:
        return False
    # The second direction of the German equivalence is only tried when one
    # of the two texts really has a German letter in it. See the comment on
    # _DROPPED_DIGRAPHS for what goes wrong when it is tried on everything.
    if not (_has_german_letter(first) or _has_german_letter(second)):
        return False
    return builder(first, True) == builder(second, True)


def _ignore_dropped(builder: Callable[[str], str]) -> Callable[[str, bool], str]:
    """Give the early stages the same signature as the later ones."""

    def build(text: str, _dropped: bool) -> str:
        return builder(text)

    return build


#: The stages, in the order they are tried. Each one is more invasive than
#: the one before it, so the first that matches is the smallest honest
#: description of how the two texts differ.
_LADDER: tuple[tuple[EquivalenceKind, Callable[[str, bool], str], bool], ...] = (
    (EquivalenceKind.CASE_ONLY, _ignore_dropped(_case_form), False),
    (EquivalenceKind.PUNCTUATION_ONLY, _ignore_dropped(_plain_form), False),
    (EquivalenceKind.SPELLING_VARIANT, _spelling_form, True),
    (EquivalenceKind.NUMBER_FORMAT, _number_form, True),
    (EquivalenceKind.COMPOUND, _compound_form, True),
)

#: The kinds again, this time in the order of how much each one constrains
#: the timing of the word it describes, most constraining first. **This
#: order is deliberate**, and it decides the answer whenever a pair of
#: texts qualifies as more than one kind. A caller that acted on a weaker
#: classification would give a word a span it is not entitled to, and the
#: transcript would then claim to know when a word was said when it does
#: not, so the strongest claim always wins.
#:
#: The first two are the kinds that can change how many words there are.
#: Where that happens, several words share one span and nobody knows where
#: inside it the join falls, which is what the specification calls a shared
#: phrase span. :attr:`EquivalenceKind.NUMBER_FORMAT` comes before
#: :attr:`EquivalenceKind.COMPOUND` because it says everything compound
#: says and adds that the content in dispute is a number, which the
#: high-risk rules have to know. The remaining three leave the words
#: standing where they were, so the original span still fits and the order
#: between them only decides how the difference is described.
#:
#: Anyone adding a kind has to decide where it belongs here.
_BY_TIMING_CONSTRAINT: tuple[EquivalenceKind, ...] = (
    EquivalenceKind.NUMBER_FORMAT,
    EquivalenceKind.COMPOUND,
    EquivalenceKind.SPELLING_VARIANT,
    EquivalenceKind.PUNCTUATION_ONLY,
    EquivalenceKind.CASE_ONLY,
    EquivalenceKind.IDENTICAL,
)


@lru_cache(maxsize=20_000)
def _word_count(text: str) -> int:
    """How many actual words a service put in this text.

    Punctuation standing on its own does not count, so a service that
    returns "contract ." has still said one word. This is the number the
    timing question turns on: two words against one means one span covering
    both, however the difference came about.
    """
    return sum(1 for part in text.split() if normalise(part))


@lru_cache(maxsize=100_000)
def equivalence_kind(first: str, second: str) -> EquivalenceKind:
    """Say whether two texts are the same spoken evidence, and how.

    The timing rules need the answer to the second question as much as the
    first. A word that differs only in how it was written can keep the span
    the service measured for it, because the same sounds are underneath it.
    A word that is genuinely different cannot be assumed to fit that span,
    and the reconciliation rules treat it as a substitution.
    """
    if first == second:
        return EquivalenceKind.IDENTICAL
    smallest: EquivalenceKind | None = None
    for kind, builder, allow_dropped in _LADDER:
        if _same_form(first, second, builder, allow_dropped):
            smallest = kind
            break
    if smallest is None:
        return EquivalenceKind.DIFFERENT
    if _word_count(first) == _word_count(second):
        return smallest
    # The words were joined or split, whatever else also changed, so the
    # answer has to be at least as strong as that.
    return min(
        (smallest, EquivalenceKind.COMPOUND),
        key=_BY_TIMING_CONSTRAINT.index,
    )


def are_equivalent(first: str, second: str) -> bool:
    """Whether two texts are the same spoken evidence.

    This is the cheap question, and alignment asks it for every pair of
    words it considers, so it answers without working out which stage the
    two texts met at. Use :func:`equivalence_kind` when the answer to that
    matters.
    """
    if first == second:
        return True
    if _compound_form(first, False) == _compound_form(second, False):
        return True
    if not (_has_german_letter(first) or _has_german_letter(second)):
        return False
    return _compound_form(first, True) == _compound_form(second, True)
