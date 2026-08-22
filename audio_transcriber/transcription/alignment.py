"""Putting every service's words in the same order as the timed ones.

Only one service in the ensemble says when each word was spoken. The
others hand back a stream of text with nothing to hang it on. Before any
of them can be compared, argued with or scored, their words have to be
laid against the timed sequence so that the application knows which word
each service is talking about. That sequence is called the backbone, and
this module is what maps everything else onto it.

ElevenLabs Scribe is normally the backbone, but nothing here assumes it.
The backbone is a parameter, and :func:`choose_backbone` will hand the job
to AssemblyAI, or to whoever else came back with timings, when Scribe
fails. Hard-wiring one service would mean that losing it cost the user
their whole transcript, and the specification is explicit that the
pipeline degrades instead.

Matching is done with the usual dynamic-programming sequence alignment: a
matrix of costs, the cheapest path through it, and a walk back along that
path. What matters is the cost of each step, and the costs here are chosen
around one particular failure. An exact match is free and a
formatting-equivalent match, such as "twenty-five" against "25", is nearly
free. A substitution costs about one, and an insertion or a deletion costs
about one each. That gap is the whole point: if a substitution cost more
than an insertion plus a deletion, the cheapest path through a pair of
words that are merely written differently would be to delete one and
insert the other, and the correspondence between them would be lost.
Whoever tunes these numbers has to keep that inequality, and
:class:`AlignmentOptions` refuses to be built any other way.

The matrix also has steps that consume several words on one side at once,
because "data base" against "database" is one word written two ways, not a
match followed by a stray insertion. Those steps are only allowed when the
words either side really do say the same thing, which the normalisation
module decides.

**Windowing is the most important decision in this module.** Sequence
alignment costs time and memory proportional to the product of the two
lengths. An hour of speech is tens of thousands of words, so aligning a
recording as one sequence would mean a matrix of hundreds of millions of
cells: slow enough to look broken, and large enough to exhaust memory on
an ordinary machine. It would also be worse, not just slower, because a
single bad match in the middle of an hour can drag the path off course for
hundreds of words.

So the recording is aligned in pieces. First the two sequences are pinned
together at anchors: words that appear exactly once in each sequence and
match exactly, kept only where they run in the same order in both. A word
that occurs once in an hour of speech and was heard identically by two
services is about as safe a landmark as this problem offers, and pinning
to them is what the "patience" diff algorithm does for source code. The
alignment then runs between consecutive anchors, on regions that are
usually a handful of words long.

Where two anchors are still too far apart, the region between them is cut
again at the most natural place near its middle: the end of a sentence, a
change of speaker, or a silence. Only when there is no such place is the
region cut arbitrarily, and columns from an arbitrary cut carry a reduced
quality score, because a cut in the wrong place can misalign the words
either side of the seam and the rest of the application should know that.

The result is one :class:`AlignmentColumn` per correspondence, gathered
into a :class:`ProviderAlignment` for each service and then into an
:class:`AlignedTable` keyed by backbone word, which is what the
reconciliation rules read. Every column carries a quality score between
zero and one. Weak alignment is one of the reasons a word goes in front of
a person, so the score is part of the output rather than something thrown
away once the matching is done.

Nothing here depends on Qt or on any provider library, so it can be tested
on its own.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Iterable, Sequence

from audio_transcriber.transcription.model import (
    AlignmentStatus,
    Provider,
    ProviderResult,
    ProviderToken,
)
from audio_transcriber.transcription.normalise import (
    EquivalenceKind,
    _compound_form,
    _has_german_letter,
    equivalence_kind,
    is_punctuation_only,
    normalise,
)

#: Who takes the backbone when the service before them is unavailable.
#: Scribe first, because it is the only one designed for this. AssemblyAI
#: next, which is what the specification says to fall back to for timing
#: and speaker attribution. OpenAI is last because it does not time its
#: words at all, and a backbone without timings can still order the words
#: but gives the transcript nothing to play back against.
BACKBONE_PREFERENCE: tuple[Provider, ...] = (
    Provider.ELEVENLABS,
    Provider.ASSEMBLYAI,
    Provider.DEEPGRAM,
    Provider.MICROSOFT,
    Provider.OPENAI,
)


@dataclass(frozen=True)
class AlignmentOptions:
    """The costs and the sizes, in one place so they can be tuned.

    The defaults are set so that a formatting difference is almost free, a
    real substitution is affordable, and deleting a word and inserting
    another in its place is the dearest thing on offer. See the module
    docstring for why that last ordering is not negotiable.
    """

    exact_cost: float = 0.0
    equivalent_cost: float = 0.05
    """A match that is only a matter of spelling, case or number format."""

    merge_cost: float = 0.15
    """One word on one side against several on the other, meaning the same.

    Slightly dearer than a plain equivalent match, so that where both
    readings are possible the simple one wins.
    """

    minimum_substitution_cost: float = 0.8
    maximum_substitution_cost: float = 1.4
    """Substituting a word costs more the less it looks like the original.

    "cat" for "cot" is a likely mishearing and a safe correspondence.
    "cat" for "elephant" is neither, and should only be chosen when the
    words around it leave no better path.
    """

    insertion_cost: float = 1.0
    deletion_cost: float = 1.0

    maximum_merge_span: int = 3
    """How many words may be folded into one. Three covers "up to date"
    against "up-to-date" and stops well short of matching whole phrases by
    accident."""

    maximum_window_words: int = 200
    """The largest region aligned in one matrix, on either side."""

    minimum_anchor_length: int = 3
    """Short words make poor landmarks even when they happen to be unique."""

    utterance_gap_seconds: float = 0.6
    """A silence at least this long is treated as a natural place to cut."""

    forced_cut_quality: float = 0.85
    """What columns are multiplied by when their region was cut arbitrarily."""

    def __post_init__(self) -> None:
        if self.maximum_substitution_cost >= self.insertion_cost + self.deletion_cost:
            raise ValueError(
                "A substitution must cost less than an insertion plus a deletion, "
                "or two spellings of the same word will be aligned as two errors "
                "and their correspondence will be lost."
            )
        if self.minimum_substitution_cost <= self.equivalent_cost:
            raise ValueError(
                "A substitution must cost more than an equivalent match, or a "
                "genuinely different word will be preferred to the right one."
            )
        if self.maximum_merge_span < 2:
            raise ValueError("Merging fewer than two words is not merging.")


DEFAULT_OPTIONS = AlignmentOptions()


# -- What comes out ------------------------------------------------------


@dataclass(frozen=True)
class AlignmentColumn:
    """One correspondence between the backbone and one other service.

    Both sides are tuples rather than single tokens, because a
    correspondence is not always one word to one word. "data base" heard by
    the backbone and "database" heard by another service is two tokens
    against one, and calling that a match plus an insertion would throw
    away the fact that they are the same word. Where there is exactly one
    token on a side, :attr:`backbone_token` and :attr:`aligned_token` give
    it directly.

    An empty ``aligned_tokens`` means the service did not hear this word at
    all; an empty ``backbone_tokens`` means it heard a word the backbone
    does not have.
    """

    provider: Provider
    position: int
    """Where this sits in the backbone word sequence.

    For an insertion, this is the position of the backbone word it comes
    before, which is one past the last word when it comes at the end.
    """

    backbone_tokens: tuple[ProviderToken, ...] = ()
    aligned_tokens: tuple[ProviderToken, ...] = ()
    punctuation: tuple[ProviderToken, ...] = ()
    """Punctuation this service returned after these words.

    Alignment ignores punctuation, because services disagree about it
    constantly and it would only add noise. It is carried here rather than
    dropped, because these entries hold timing of their own.
    """

    status: AlignmentStatus = AlignmentStatus.UNALIGNED
    equivalence: EquivalenceKind = EquivalenceKind.DIFFERENT
    """How the two texts are the same, where they are the same at all.

    The timing rules need this: a word that differs only in how it was
    written can keep the span the backbone measured for it.
    """

    quality: float = 0.0
    """How much this correspondence can be trusted, from zero to one.

    In one sentence: how confident the alignment is that these two services
    are talking about the same word here. Three things go into it. The kind
    of correspondence matters most, and an exact match of the same word
    scores best. How much of the surrounding stretch matched matters next,
    because a disagreement among agreements is a disagreement about one
    word, while the same disagreement in a stretch where nothing matches
    usually means the alignment has lost its place. For a substitution, how
    alike the two words look matters as well, since "contact" for
    "contract" is what a mishearing looks like and "now" for "today" is
    not.

    It says nothing about whether either service is *right*. That is the
    reconciliation rules' question, and this number is one of the things
    they weigh when they answer it.
    """

    @property
    def backbone_token(self) -> ProviderToken | None:
        return self.backbone_tokens[0] if len(self.backbone_tokens) == 1 else None

    @property
    def aligned_token(self) -> ProviderToken | None:
        return self.aligned_tokens[0] if len(self.aligned_tokens) == 1 else None

    @property
    def backbone_indices(self) -> tuple[int, ...]:
        """Positions in the backbone word sequence that this column covers."""
        return tuple(range(self.position, self.position + len(self.backbone_tokens)))

    @property
    def text(self) -> str:
        """What this service said here, as it said it."""
        return " ".join(token.text for token in self.aligned_tokens)

    @property
    def backbone_text(self) -> str:
        return " ".join(token.text for token in self.backbone_tokens)

    @property
    def is_insertion(self) -> bool:
        return not self.backbone_tokens

    @property
    def is_deletion(self) -> bool:
        return not self.aligned_tokens

    @property
    def is_agreement(self) -> bool:
        """Whether this service said the same thing as the backbone.

        True for an exact match and for one that differs only in how it was
        written, which are the two cases the reconciliation rules can settle
        without asking anybody.
        """
        return self.status in (AlignmentStatus.EXACT, AlignmentStatus.FORMAT_EQUIVALENT)


@dataclass(frozen=True)
class ProviderAlignment:
    """Everything one service's words turned into against the backbone."""

    provider: Provider
    backbone_provider: Provider
    columns: tuple[AlignmentColumn, ...] = ()

    @property
    def quality(self) -> float:
        """The average column quality, as one rough number for the service.

        Treat this as a rough signal and nothing more. It is an average
        over columns that mean different things, so a service that
        disagrees once about a word that sounds similar can score above one
        that disagrees once about a word that does not, and neither of them
        is thereby the better service. A small difference between two
        services carries no meaning at all, and this number will not
        support a strict ordering between them.

        What it is good for is telling a service that mostly lined up from
        one that mostly did not. Where the question is about a particular
        word, read :attr:`AlignmentColumn.quality` for that word instead,
        which is the number the review rules are meant to act on.
        """
        if not self.columns:
            return 0.0
        return sum(column.quality for column in self.columns) / len(self.columns)

    @property
    def agreement(self) -> float:
        """The share of columns where this service and the backbone agree.

        Provider weighting and the review rules both want this: a service
        that agrees with the backbone nine times in ten is telling a
        different story from one that agrees half the time.
        """
        if not self.columns:
            return 0.0
        agreed = sum(1 for column in self.columns if column.is_agreement)
        return agreed / len(self.columns)

    @property
    def is_empty(self) -> bool:
        """Whether this service contributed no words at all."""
        return not any(column.aligned_tokens for column in self.columns)

    def columns_by_backbone_position(self) -> dict[int, list[AlignmentColumn]]:
        """Group the columns by the backbone word they sit at.

        A merge appears once, under the first of the backbone words it
        covers, so that a consumer walking the backbone does not emit the
        same word several times.
        """
        grouped: dict[int, list[AlignmentColumn]] = {}
        for column in self.columns:
            grouped.setdefault(column.position, []).append(column)
        return grouped


@dataclass(frozen=True)
class AlignedRow:
    """One backbone word, and what every other service put in its place.

    A row with no backbone token holds words that one or more services
    heard where the backbone heard nothing. Those rows sit immediately
    before the row of the backbone word they came in front of.
    """

    position: int
    backbone_token: ProviderToken | None = None
    punctuation: tuple[ProviderToken, ...] = ()
    """Punctuation the backbone returned after this word."""

    columns: tuple[AlignmentColumn, ...] = ()

    def column_for(self, provider: Provider) -> AlignmentColumn | None:
        for column in self.columns:
            if column.provider == provider:
                return column
        return None

    @property
    def is_insertion_row(self) -> bool:
        return self.backbone_token is None

    @property
    def quality(self) -> float:
        """The weakest column here, which is what the review rules look at."""
        if not self.columns:
            return 0.0
        return min(column.quality for column in self.columns)

    @property
    def texts(self) -> dict[Provider, str]:
        """What each service said here, keyed by service."""
        return {column.provider: column.text for column in self.columns if column.aligned_tokens}


@dataclass(frozen=True)
class AlignedTable:
    """Every backbone word beside what all the other services said there.

    This is the whole point of the module and the thing reconciliation
    reads. The rows follow the backbone in order, so walking them is
    walking the recording.
    """

    backbone_provider: Provider
    backbone_tokens: tuple[ProviderToken, ...] = ()
    """The backbone's words, without its punctuation entries."""

    rows: tuple[AlignedRow, ...] = ()
    alignments: tuple[ProviderAlignment, ...] = ()
    missing_providers: tuple[Provider, ...] = ()
    """Services that returned nothing, so every row is a deletion for them."""

    @property
    def providers(self) -> tuple[Provider, ...]:
        return tuple(alignment.provider for alignment in self.alignments)

    def alignment_for(self, provider: Provider) -> ProviderAlignment | None:
        for alignment in self.alignments:
            if alignment.provider == provider:
                return alignment
        return None

    def quality_for(self, provider: Provider) -> float:
        """How well this service lined up overall, as a rough signal.

        See :attr:`ProviderAlignment.quality` for what this number can and
        cannot be asked to do. It will not carry a strict ordering between
        two services that lined up about equally well.
        """
        alignment = self.alignment_for(provider)
        return alignment.quality if alignment is not None else 0.0

    def row_for_backbone(self, position: int) -> AlignedRow | None:
        for row in self.rows:
            if not row.is_insertion_row and row.position == position:
                return row
        return None


# -- Choosing the backbone -----------------------------------------------


def choose_backbone(
    results: Iterable[ProviderResult],
    preference: Sequence[Provider] = BACKBONE_PREFERENCE,
) -> Provider | None:
    """Decide which service the others will be aligned against.

    Timings are what make a backbone useful, so a service that timed its
    words is always preferred to one that did not, whatever the order of
    preference says. Only when nothing came back with timings does this
    fall through to the service with the most words, which still gives a
    sequence to reconcile against even though the transcript will have no
    playable spans until forced alignment fills them in.
    """
    usable = [result for result in results if result.succeeded and result.word_tokens]
    if not usable:
        return None

    def rank(result: ProviderResult) -> int:
        try:
            return preference.index(result.provider)
        except ValueError:
            return len(preference)

    timed = [
        result for result in usable if any(token.has_timing for token in result.word_tokens)
    ]
    if timed:
        return min(timed, key=rank).provider
    return max(usable, key=lambda result: (len(result.word_tokens), -rank(result))).provider


# -- Aligning ------------------------------------------------------------


def align_sequences(
    backbone_tokens: Sequence[ProviderToken],
    aligned_tokens: Sequence[ProviderToken],
    provider: Provider | None = None,
    backbone_provider: Provider | None = None,
    options: AlignmentOptions = DEFAULT_OPTIONS,
) -> ProviderAlignment:
    """Map one service's words onto the backbone's words.

    Both sequences are given as the tokens the services returned, including
    their punctuation entries, which are set aside for the matching and put
    back in the result.
    """
    provider = _provider_of(aligned_tokens, provider, "aligned")
    backbone_provider = _provider_of(backbone_tokens, backbone_provider, "backbone")

    backbone = _build_stream(backbone_tokens, options)
    other = _build_stream(aligned_tokens, options)

    columns: list[AlignmentColumn] = []
    factors: list[float] = []
    for window in _windows(backbone, other, options):
        for column in _align_window(backbone, other, window, provider, options):
            columns.append(column)
            factors.append(options.forced_cut_quality if window.forced else 1.0)

    scored = _score_columns(columns, factors)
    return ProviderAlignment(
        provider=provider,
        backbone_provider=backbone_provider,
        columns=scored,
    )


def align_result(
    backbone: ProviderResult,
    other: ProviderResult,
    options: AlignmentOptions = DEFAULT_OPTIONS,
) -> ProviderAlignment:
    """Map one service's result onto the backbone's result.

    A service that failed is aligned exactly like one that returned
    nothing: every backbone word becomes a deletion. That is honest, and it
    keeps the shape of the table the same however many services answered,
    so nothing downstream has to special-case a missing service.
    """
    return align_sequences(
        backbone.tokens,
        other.tokens,
        provider=other.provider,
        backbone_provider=backbone.provider,
        options=options,
    )


def build_aligned_table(
    backbone: ProviderResult,
    others: Iterable[ProviderResult],
    options: AlignmentOptions = DEFAULT_OPTIONS,
) -> AlignedTable:
    """Align every other service against the backbone and lay them side by side."""
    stream = _build_stream(backbone.tokens, options)
    alignments = [
        align_result(backbone, other, options)
        for other in others
        if other.provider != backbone.provider
    ]
    missing = tuple(
        alignment.provider for alignment in alignments if alignment.is_empty
    )
    return AlignedTable(
        backbone_provider=backbone.provider,
        backbone_tokens=stream.words,
        rows=_build_rows(stream, alignments),
        alignments=tuple(alignments),
        missing_providers=missing,
    )


def _build_rows(
    stream: "_Stream",
    alignments: Sequence[ProviderAlignment],
) -> tuple[AlignedRow, ...]:
    """Turn one alignment per service into one row per backbone word."""
    at_position: dict[int, list[AlignmentColumn]] = {}
    inserted_before: dict[int, list[AlignmentColumn]] = {}
    for alignment in alignments:
        for column in alignment.columns:
            target = inserted_before if column.is_insertion else at_position
            target.setdefault(column.position, []).append(column)

    rows: list[AlignedRow] = []
    for position in range(len(stream.words) + 1):
        for column in inserted_before.get(position, []):
            # Each inserted word gets its own row rather than being merged
            # with its neighbours, because two services inserting different
            # words in the same gap are not agreeing with each other.
            rows.append(AlignedRow(position=position, columns=(column,)))
        if position == len(stream.words):
            break
        rows.append(
            AlignedRow(
                position=position,
                backbone_token=stream.words[position],
                punctuation=stream.punctuation[position],
                columns=tuple(at_position.get(position, [])),
            )
        )
    return tuple(rows)


def _provider_of(
    tokens: Sequence[ProviderToken],
    given: Provider | None,
    role: str,
) -> Provider:
    if given is not None:
        return given
    if tokens:
        return tokens[0].provider
    raise ValueError(
        f"The {role} service cannot be worked out from an empty list of words, "
        "so it has to be given."
    )


# -- The word streams ----------------------------------------------------


#: Everything :func:`~audio_transcriber.transcription.normalise.are_equivalent`
#: looks at when it compares two texts, worked out once per text so that
#: the matrix can compare two words, or a word and a phrase, by comparing
#: two tuples. In order: the text itself, its comparison form, whether it
#: has a German letter in it, and its comparison form with the umlauts
#: dropped. See :func:`_keys_agree`.
_Key = tuple[str, str, bool, str]


def _key_of(text: str) -> _Key:
    return (text, normalise(text), _has_german_letter(text), _compound_form(text, True))


def _keys_agree(first: _Key, second: _Key) -> bool:
    """Whether two texts are equivalent, decided from their keys alone.

    This is :func:`~audio_transcriber.transcription.normalise.are_equivalent`
    step for step: the same text, or the same comparison form, or -- only
    when one of the two has a German letter in it -- the same form with the
    umlauts dropped. Identical texts have identical comparison forms, so
    the first step needs no test of its own. The matrix asks this question
    several times for every cell, and asking it of two tuples that are
    already worked out is what keeps a long, anchor-poor recording from
    spending its whole time joining strings and normalising them again.
    """
    if first[1] == second[1]:
        return True
    if not (first[2] or second[2]):
        return False
    return first[3] == second[3]


@dataclass(frozen=True)
class _Stream:
    """One service's words, with everything alignment needs about them.

    Punctuation is taken out of the sequence that gets matched and hung on
    the word in front of it instead. Services disagree about punctuation
    far more than they disagree about words, and leaving it in would mean
    aligning a full stop against a comma and calling it a substitution.
    """

    words: tuple[ProviderToken, ...] = ()
    punctuation: tuple[tuple[ProviderToken, ...], ...] = ()
    forms: tuple[str, ...] = ()
    keys: tuple[_Key, ...] = ()
    """The :class:`_Key` of each word, worked out once for the whole stream."""

    boundary_after: tuple[bool, ...] = ()
    """Whether a sentence, a speaker or a silence ends at this word."""


_SENTENCE_ENDS = frozenset(".!?…")


def _build_stream(tokens: Sequence[ProviderToken], options: AlignmentOptions) -> _Stream:
    words: list[ProviderToken] = []
    punctuation: list[list[ProviderToken]] = []
    pending: list[ProviderToken] = []
    for token in tokens:
        if token.is_punctuation or is_punctuation_only(token.text):
            pending.append(token)
            continue
        words.append(token)
        # Punctuation that arrived before any word is hung on the first
        # word instead, so that nothing is silently lost.
        if len(words) == 1:
            punctuation.append(list(pending))
        else:
            punctuation[-1].extend(pending)
            punctuation.append([])
        pending = []
    if punctuation:
        punctuation[-1].extend(pending)

    keys = tuple(_key_of(token.text) for token in words)
    boundaries = tuple(
        _ends_region(words, punctuation, index, options) for index in range(len(words))
    )
    return _Stream(
        words=tuple(words),
        punctuation=tuple(tuple(group) for group in punctuation),
        forms=tuple(key[1] for key in keys),
        keys=keys,
        boundary_after=boundaries,
    )


def _ends_region(
    words: Sequence[ProviderToken],
    punctuation: Sequence[Sequence[ProviderToken]],
    index: int,
    options: AlignmentOptions,
) -> bool:
    """Whether this word is a natural place to cut a region in two."""
    token = words[index]
    if token.text and token.text[-1] in _SENTENCE_ENDS:
        return True
    if any(
        character in _SENTENCE_ENDS
        for mark in punctuation[index]
        for character in mark.text
    ):
        return True
    if index + 1 >= len(words):
        return False
    following = words[index + 1]
    if token.speaker is not None and following.speaker != token.speaker:
        return True
    if token.end is not None and following.start is not None:
        return following.start - token.end >= options.utterance_gap_seconds
    return False


# -- Windows -------------------------------------------------------------


@dataclass(frozen=True)
class _Window:
    """A region of the backbone and the region of the other service it faces."""

    backbone_start: int
    backbone_end: int
    aligned_start: int
    aligned_end: int
    forced: bool = False

    @property
    def backbone_length(self) -> int:
        return self.backbone_end - self.backbone_start

    @property
    def aligned_length(self) -> int:
        return self.aligned_end - self.aligned_start


def _windows(
    backbone: _Stream,
    other: _Stream,
    options: AlignmentOptions,
) -> list[_Window]:
    """Cut both sequences into regions small enough to align one by one."""
    anchors = _anchor_pairs(backbone.forms, other.forms, options)
    cuts = [(0, 0), *anchors, (len(backbone.words), len(other.words))]
    windows: list[_Window] = []
    for (start_b, start_a), (end_b, end_a) in zip(cuts, cuts[1:]):
        if end_b == start_b and end_a == start_a:
            continue
        windows.extend(
            _subdivide(_Window(start_b, end_b, start_a, end_a), backbone, options)
        )
    return windows


def _anchor_pairs(
    backbone_forms: Sequence[str],
    other_forms: Sequence[str],
    options: AlignmentOptions,
) -> list[tuple[int, int]]:
    """Find the words that can safely pin the two sequences together.

    A word qualifies only if it occurs exactly once in each sequence and is
    long enough to be worth trusting. Uniqueness is what makes it safe: a
    word that occurs once in the whole recording cannot be confused with
    another occurrence of itself, which is the mistake that would otherwise
    drag an alignment off course.

    The pairs that survive still have to run in the same order in both
    sequences, and where they do not, the longest run that does is kept.
    Two services can genuinely swap a phrase around, and pinning to a pair
    that crosses over would force everything between it into nonsense.
    """
    counted_backbone = Counter(backbone_forms)
    counted_other = Counter(other_forms)
    positions = {form: index for index, form in enumerate(other_forms)}

    candidates: list[tuple[int, int]] = []
    for index, form in enumerate(backbone_forms):
        if len(form) < options.minimum_anchor_length:
            continue
        if counted_backbone[form] != 1 or counted_other.get(form) != 1:
            continue
        candidates.append((index, positions[form]))
    return _longest_rising_run(candidates)


def _longest_rising_run(pairs: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Keep the longest subset of the pairs whose second halves also rise.

    The pairs already rise on the backbone side, so this is the classic
    longest-increasing-subsequence problem, solved with the patience
    method: lay each value on the leftmost pile it fits, and read the
    answer back along the trail of predecessors.
    """
    if not pairs:
        return []
    tails: list[int] = []
    tail_index: list[int] = []
    previous: list[int] = [-1] * len(pairs)
    for position, (_backbone, other) in enumerate(pairs):
        pile = bisect_left(tails, other)
        if pile == len(tails):
            tails.append(other)
            tail_index.append(position)
        else:
            tails[pile] = other
            tail_index[pile] = position
        previous[position] = tail_index[pile - 1] if pile else -1
    result: list[tuple[int, int]] = []
    position = tail_index[-1]
    while position != -1:
        result.append(pairs[position])
        position = previous[position]
    result.reverse()
    return result


def _subdivide(
    window: _Window,
    backbone: _Stream,
    options: AlignmentOptions,
) -> list[_Window]:
    """Cut a region that is still too large, preferring natural places.

    The backbone is cut at the sentence end, speaker change or silence
    nearest the middle of the region. The other service has no timings and
    may have a different number of words, so its cut is placed at the
    matching fraction of its own region. That is a guess, and it is only
    ever made when there was no anchor to use instead, which is why the
    columns that come out of a forced cut are marked down.
    """
    limit = options.maximum_window_words
    if window.backbone_length <= limit and window.aligned_length <= limit:
        return [window]
    if window.backbone_length < 2 or window.aligned_length < 2:
        # Nothing useful is left to cut: one side is a single word or empty,
        # and the matrix is a strip rather than a square.
        return [window]

    middle = window.backbone_start + window.backbone_length // 2
    cut_backbone = _nearest_boundary(backbone, window, middle)
    fraction = (cut_backbone - window.backbone_start) / window.backbone_length
    cut_aligned = window.aligned_start + max(
        1, min(window.aligned_length - 1, round(fraction * window.aligned_length))
    )
    first = _Window(window.backbone_start, cut_backbone, window.aligned_start, cut_aligned, True)
    second = _Window(cut_backbone, window.backbone_end, cut_aligned, window.aligned_end, True)
    return _subdivide(first, backbone, options) + _subdivide(second, backbone, options)


def _nearest_boundary(backbone: _Stream, window: _Window, middle: int) -> int:
    """The natural cutting point closest to the middle of a region."""
    best = middle
    best_distance = window.backbone_length
    for index in range(window.backbone_start, window.backbone_end - 1):
        if not backbone.boundary_after[index]:
            continue
        distance = abs(index + 1 - middle)
        if distance < best_distance:
            best = index + 1
            best_distance = distance
    return max(window.backbone_start + 1, min(window.backbone_end - 1, best))


# -- The matrix ----------------------------------------------------------


@lru_cache(maxsize=100_000)
def _similarity(first: str, second: str) -> float:
    """How alike two comparison forms look, from zero to one."""
    if not first or not second:
        return 0.0
    return SequenceMatcher(None, first, second).ratio()


def _align_window(
    backbone: _Stream,
    other: _Stream,
    window: _Window,
    provider: Provider,
    options: AlignmentOptions,
) -> list[AlignmentColumn]:
    """Find the cheapest correspondence across one region.

    The matrix has a row for every backbone word and a column for every
    word of the other service, and each cell asks four questions of the
    words around it: do these two words correspond, and do two or three
    words on either side say the same as one word on the other. The
    questions are the same ones whatever the cell, so everything they need
    is worked out once per row and once per column before the matrix is
    filled, and the cell itself only compares tuples and looks up lists.
    Working it out inside the cell instead meant joining and normalising
    the same phrase once for every column it was tried against, which is
    where an anchor-poor recording spent nearly all of its time.
    """
    backbone_words = backbone.words[window.backbone_start : window.backbone_end]
    other_words = other.words[window.aligned_start : window.aligned_end]
    backbone_forms = backbone.forms[window.backbone_start : window.backbone_end]
    other_forms = other.forms[window.aligned_start : window.aligned_end]
    rows = len(backbone_words)
    columns_count = len(other_words)

    if rows and rows == columns_count and backbone_forms == other_forms:
        # Most of a transcript is two services saying the same thing. Taking
        # that case straight out is what keeps a whole recording linear
        # rather than quadratic in the regions where nothing is in dispute.
        moves = [(index, index + 1, index, index + 1) for index in range(rows)]
        return _columns_from_moves(backbone, other, window, moves, provider)

    backbone_keys = backbone.keys[window.backbone_start : window.backbone_end]
    other_keys = other.keys[window.aligned_start : window.aligned_end]
    spans = range(2, options.maximum_merge_span + 1)
    # backbone_merges[span][row] lists the columns whose single word says
    # the same as the ``span`` backbone words ending at ``row``; the other
    # direction likewise. Both are in matrix coordinates, where a row or
    # column of nought is the empty prefix, so a phrase ending at ``row``
    # is the words ``row - span`` up to ``row``.
    backbone_singles = _SingleWords(backbone_keys)
    other_singles = _SingleWords(other_keys)
    backbone_merges = {
        span: _merge_partners(backbone_words, other_singles, span) for span in spans
    }
    other_merges = {
        span: _merge_partners(other_words, backbone_singles, span) for span in spans
    }
    pair_costs = [
        [
            _pair_cost(backbone_keys[row], other_keys[column], options)
            for column in range(columns_count)
        ]
        for row in range(rows)
    ]

    infinity = float("inf")
    deletion_cost = options.deletion_cost
    insertion_cost = options.insertion_cost
    merge_cost = options.merge_cost
    cost = [[infinity] * (columns_count + 1) for _ in range(rows + 1)]
    step: list[list[tuple[int, int] | None]] = [
        [None] * (columns_count + 1) for _ in range(rows + 1)
    ]
    cost[0][0] = 0.0
    for row in range(rows + 1):
        cost_row = cost[row]
        step_row = step[row]
        above = cost[row - 1] if row else None
        pair_row = pair_costs[row - 1] if row else None
        for column in range(columns_count + 1):
            if row == 0 and column == 0:
                continue
            best = infinity
            taken: tuple[int, int] | None = None
            if row and column:
                candidate = above[column - 1] + pair_row[column - 1]
                if candidate < best:
                    best, taken = candidate, (1, 1)
            if row:
                candidate = above[column] + deletion_cost
                if candidate < best:
                    best, taken = candidate, (1, 0)
            if column:
                candidate = cost_row[column - 1] + insertion_cost
                if candidate < best:
                    best, taken = candidate, (0, 1)
            for span in spans:
                if row >= span and column:
                    if column in backbone_merges[span][row]:
                        candidate = cost[row - span][column - 1] + merge_cost
                        if candidate < best:
                            best, taken = candidate, (span, 1)
                if column >= span and row:
                    if row in other_merges[span][column]:
                        candidate = above[column - span] + merge_cost
                        if candidate < best:
                            best, taken = candidate, (1, span)
            cost_row[column] = best
            step_row[column] = taken

    moves: list[tuple[int, int, int, int]] = []
    row, column = rows, columns_count
    while row or column:
        taken = step[row][column]
        if taken is None:
            break
        back, across = taken
        moves.append((row - back, row, column - across, column))
        row, column = row - back, column - across
    moves.reverse()
    return _columns_from_moves(backbone, other, window, moves, provider)


class _SingleWords:
    """One side of a window, indexed by the keys a phrase could match.

    Answers "which words here say the same as this phrase" with three
    dictionary lookups instead of a comparison against every word, and
    gives exactly the answer :func:`_keys_agree` would give word by word:
    the words whose comparison form is the phrase's, plus, through the
    dropped-umlaut form, every word when the phrase has a German letter
    and only the words that have one when it does not.
    """

    def __init__(self, keys: Sequence[_Key]) -> None:
        self._by_form: dict[str, set[int]] = {}
        self._by_dropped: dict[str, set[int]] = {}
        self._by_dropped_german: dict[str, set[int]] = {}
        for position, (_text, form, german, dropped) in enumerate(keys):
            # One-based, as the matrix counts rows and columns.
            self._by_form.setdefault(form, set()).add(position + 1)
            self._by_dropped.setdefault(dropped, set()).add(position + 1)
            if german:
                self._by_dropped_german.setdefault(dropped, set()).add(position + 1)

    def agreeing_with(self, phrase: _Key) -> frozenset[int]:
        _text, form, german, dropped = phrase
        found = set(self._by_form.get(form, ()))
        if german:
            found.update(self._by_dropped.get(dropped, ()))
        else:
            found.update(self._by_dropped_german.get(dropped, ()))
        return frozenset(found)


def _merge_partners(
    several: Sequence[ProviderToken],
    singles: _SingleWords,
    span: int,
) -> list[frozenset[int]]:
    """For each end position on one side, the single words it may fold into.

    Entry ``end`` holds the positions, one-based as the matrix counts them,
    of every word on the other side that says the same as the ``span``
    words ending just before ``end``. The first ``span`` entries are empty
    because no phrase of that length ends there yet.

    Every phrase is joined and keyed exactly once here, however many
    columns it is then tried against.
    """
    partners: list[frozenset[int]] = [frozenset()] * min(span, len(several) + 1)
    for end in range(span, len(several) + 1):
        phrase = _key_of(" ".join(token.text for token in several[end - span : end]))
        partners.append(singles.agreeing_with(phrase))
    return partners


def _pair_cost(backbone: _Key, other: _Key, options: AlignmentOptions) -> float:
    """What it costs to say that these two words are each other.

    The comparison forms are compared first because that answers the
    question for nearly every pair without any further work, and this
    function is called once for every cell of every matrix.
    """
    if backbone[0] == other[0]:
        return options.exact_cost
    if _keys_agree(backbone, other):
        return options.equivalent_cost
    likeness = _similarity(backbone[1], other[1])
    spread = options.maximum_substitution_cost - options.minimum_substitution_cost
    return options.maximum_substitution_cost - spread * likeness



def _columns_from_moves(
    backbone: _Stream,
    other: _Stream,
    window: _Window,
    moves: Sequence[tuple[int, int, int, int]],
    provider: Provider,
) -> list[AlignmentColumn]:
    """Turn the path through the matrix into columns, and name each step."""
    columns: list[AlignmentColumn] = []
    for backbone_from, backbone_to, other_from, other_to in moves:
        position = window.backbone_start + backbone_from
        backbone_tokens = backbone.words[
            window.backbone_start + backbone_from : window.backbone_start + backbone_to
        ]
        other_slice = slice(window.aligned_start + other_from, window.aligned_start + other_to)
        other_tokens = other.words[other_slice]
        punctuation = tuple(
            mark for group in other.punctuation[other_slice] for mark in group
        )
        status, equivalence = _describe(backbone_tokens, other_tokens)
        columns.append(
            AlignmentColumn(
                provider=provider,
                position=position,
                backbone_tokens=backbone_tokens,
                aligned_tokens=other_tokens,
                punctuation=punctuation,
                status=status,
                equivalence=equivalence,
            )
        )
    return columns


def _describe(
    backbone_tokens: Sequence[ProviderToken],
    other_tokens: Sequence[ProviderToken],
) -> tuple[AlignmentStatus, EquivalenceKind]:
    """Say what kind of correspondence one step of the path is."""
    if not other_tokens:
        return AlignmentStatus.DELETION, EquivalenceKind.DIFFERENT
    if not backbone_tokens:
        return AlignmentStatus.INSERTION, EquivalenceKind.DIFFERENT
    backbone_text = " ".join(token.text for token in backbone_tokens)
    other_text = " ".join(token.text for token in other_tokens)
    kind = equivalence_kind(backbone_text, other_text)
    if len(backbone_tokens) > 1:
        return AlignmentStatus.MERGE, kind
    if len(other_tokens) > 1:
        return AlignmentStatus.SPLIT, kind
    if kind is EquivalenceKind.IDENTICAL:
        return AlignmentStatus.EXACT, kind
    if kind.is_equivalent:
        return AlignmentStatus.FORMAT_EQUIVALENT, kind
    return AlignmentStatus.SUBSTITUTION, kind


# -- Scoring -------------------------------------------------------------

#: What each kind of correspondence is worth before its surroundings are
#: taken into account. An exact match of the same word is as good as this
#: gets; an inserted or deleted word is barely evidence of anything.
_BASE_QUALITY: dict[AlignmentStatus, float] = {
    AlignmentStatus.EXACT: 1.0,
    AlignmentStatus.FORMAT_EQUIVALENT: 0.9,
    AlignmentStatus.MERGE: 0.8,
    AlignmentStatus.SPLIT: 0.8,
    AlignmentStatus.SUBSTITUTION: 0.5,
    AlignmentStatus.INSERTION: 0.3,
    AlignmentStatus.DELETION: 0.3,
    AlignmentStatus.UNALIGNED: 0.0,
}

#: How many columns either side count as the neighbourhood of a column.
_CONTEXT_COLUMNS = 4


def _score_columns(
    columns: Sequence[AlignmentColumn],
    factors: Sequence[float],
) -> tuple[AlignmentColumn, ...]:
    """Give every column a quality score between zero and one.

    The kind of correspondence is only half of it. What the columns around
    it look like is the other half, because a substitution standing among
    exact matches is a confident, local disagreement about one word,
    whereas the same substitution in the middle of a stretch where nothing
    matches is a sign that the alignment itself has lost its place. The
    second of those is the one a person needs to see.
    """
    scored: list[AlignmentColumn] = []
    for index, column in enumerate(columns):
        start = max(0, index - _CONTEXT_COLUMNS)
        end = min(len(columns), index + _CONTEXT_COLUMNS + 1)
        neighbourhood = columns[start:end]
        support = sum(1 for other in neighbourhood if other.is_agreement) / len(neighbourhood)
        quality = _BASE_QUALITY[column.status] * (0.5 + 0.5 * support)
        if column.status is AlignmentStatus.SUBSTITUTION:
            # A substitution between words that look alike is a likely
            # mishearing and a sound correspondence; one between words with
            # nothing in common is a guess about where the words line up.
            likeness = _similarity(
                normalise(column.backbone_text), normalise(column.text)
            )
            quality *= 0.6 + 0.4 * likeness
        quality *= factors[index]
        scored.append(_with_quality(column, max(0.0, min(1.0, quality))))
    return tuple(scored)


def _with_quality(column: AlignmentColumn, quality: float) -> AlignmentColumn:
    return AlignmentColumn(
        provider=column.provider,
        position=column.position,
        backbone_tokens=column.backbone_tokens,
        aligned_tokens=column.aligned_tokens,
        punctuation=column.punctuation,
        status=column.status,
        equivalence=column.equivalence,
        quality=quality,
    )
