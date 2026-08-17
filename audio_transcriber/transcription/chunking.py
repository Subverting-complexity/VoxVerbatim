"""Cutting a recording up for the services that cannot take it whole.

The services differ enormously in what they will accept in one request.
ElevenLabs takes files up to five gigabytes; AssemblyAI takes five gigabytes
and up to ten hours; Microsoft takes three hundred megabytes; OpenAI takes
twenty-five. So OpenAI needs an hour of audio broken into pieces, and the
others do not.

The mistake this module exists to prevent is treating the smallest of those
limits as though it applied to everything. It would be much simpler to cut
the recording into twenty-five megabyte pieces once and send the same pieces
to everyone, and it would be worse in every way that matters. Every chunk
boundary is a place where a service loses the context either side of it and
where two answers have to be stitched back together, so a boundary that is
not needed is damage taken for nothing. Chunking is therefore worked out per
service, from that service's own limits, and a service that will take the
whole recording is sent the whole recording.

Where boundaries are needed, they are placed at the quietest moment that can
be found near the wanted position rather than at the wanted position itself.
That is not tidiness. A boundary in the middle of a word leaves each side
holding half of it, and half a word matches nothing on the other side, which
is exactly what the stitching afterwards needs in order to work.

The stitching is the other half of this module, and it has to work in two
quite different situations. Where a service returns times for its words, the
overlap between two chunks is resolved on time: the earlier chunk owns
everything up to the middle of the overlap and the later chunk owns
everything after it. The reason for choosing the middle is that a word near
the edge of a chunk was heard with context on one side and silence on the
other, so each word is taken from the chunk that heard it better.

But the one service that genuinely needs chunking is the one service that
returns no times at all. OpenAI's ``gpt-transcribe`` gives words and nothing
else: no word times, no segment times, no confidences, and the specification
forbids dropping back to an older model to get them. So for OpenAI the
overlap has to be resolved on the words themselves, by finding the longest
run of words that ends the earlier chunk and begins the later one and
joining there. Where no convincing run can be found, the join is made at the
position the overlap implies and is reported as uncertain, so that the
pipeline can put those words in front of a person rather than quietly
emitting a seam nobody knows about.

Nothing here depends on Qt, so it can be tested on its own.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

from audio_transcriber.audio.enhance import OUTPUT_FORMATS, OutputFormat
from audio_transcriber.transcription.canonical import (
    cut_window,
    estimate_lossless_bytes,
    lossless_bytes_per_second,
)
from audio_transcriber.transcription.model import (
    AudioSpan,
    CanonicalAudio,
    ChunkRecord,
    Provider,
    ProviderToken,
)
from audio_transcriber.transcription.providers.base import ProviderCapabilities

_log = logging.getLogger(__name__)

#: Chunks are written as FLAC. It is lossless, every service that needs
#: chunking accepts it, and it holds about half as much audio per byte as
#: PCM, which directly halves the number of boundaries a long recording needs.
DEFAULT_CHUNK_FORMAT: OutputFormat = OUTPUT_FORMATS[2]

#: The fastest anyone speaks, in words a second, used only to work out how
#: many words could possibly fall inside an overlap. Being generous here
#: costs a slightly wider search; being mean would make the search miss the
#: match it was looking for.
_FASTEST_WORDS_PER_SECOND = 6.0

#: How many words in a row have to agree before a text join is believed. One
#: word is no evidence at all, because "the" appears everywhere and would
#: happily join two chunks in the wrong place.
MINIMUM_MATCHED_WORDS = 3

#: What the level meter reports for a window holding nothing. It is
#: substituted for negative infinity so that quiet points can be compared
#: with ordinary arithmetic.
_SILENT_LEVEL_DB = -120.0


@dataclass(frozen=True)
class ChunkingOptions:
    """The knobs on how a recording is cut up."""

    output_format: OutputFormat = DEFAULT_CHUNK_FORMAT

    safety_fraction: float = 0.8
    """How much of a service's limit a chunk may use.

    Aiming at the limit and hoping is how a run fails on the last chunk of a
    long recording. A fifth of the limit held back covers the difference
    between what FLAC was estimated at and what it turned out to be.
    """

    overlap_seconds: float = 3.0
    """How much of the previous chunk each chunk repeats.

    Long enough to hold several words, because the words are what the join
    is made on where a service gives no times.
    """

    minimum_chunk_seconds: float = 10.0
    """Shorter than this is not worth a request of its own, so a scrap left
    at the end is folded into the chunk before it."""

    boundary_search_seconds: float = 10.0
    """How far either side of a wanted boundary a quiet moment is looked for."""

    measurement_seconds: float = 0.1
    """The length of the windows the level is measured over."""


DEFAULT_CHUNKING_OPTIONS = ChunkingOptions()


@dataclass(frozen=True)
class ChunkJoin:
    """What happened where two chunks were stitched back together.

    Kept and returned rather than logged, because an uncertain join is a
    fact about the transcript: the words around it may be duplicated or
    missing, and the only honest thing to do is say so and let a person look.
    """

    earlier_chunk_index: int
    later_chunk_index: int
    canonical_time: float
    """Where the two chunks were joined, in canonical time."""

    method: str
    """``time``, ``text``, ``nominal`` or ``abutting``."""

    matched_words: int = 0
    """The length of the run of words that agreed across the overlap.

    It is zero for every join not made on the words, which includes both the
    joins made on timings and the joins where nothing matched.
    """

    certain: bool = True

    @property
    def description(self) -> str:
        """One sentence about this join, written to be read aloud."""
        minutes, seconds = divmod(self.canonical_time, 60.0)
        at = f"{int(minutes)} minutes {seconds:.1f} seconds in"
        if self.method == "time":
            return f"The chunks were joined on their timings at {at}."
        if self.method == "text":
            return (
                f"The chunks were joined at {at}, where {self.matched_words} words "
                "matched across the overlap."
            )
        if self.method == "abutting":
            return f"The chunks meet at {at} and do not overlap."
        return (
            f"The chunks were joined at {at}, but no words matched across the overlap, "
            "so a few words there may be repeated or missing."
        )


@dataclass(frozen=True)
class MergedTokens:
    """One service's words for the whole recording, with the seams noted."""

    tokens: tuple[ProviderToken, ...]
    joins: tuple[ChunkJoin, ...] = ()

    @property
    def uncertain_joins(self) -> tuple[ChunkJoin, ...]:
        return tuple(join for join in self.joins if not join.certain)


# -- Planning -------------------------------------------------------------


def plan_chunks(
    canonical: CanonicalAudio,
    provider: Provider,
    capabilities: ProviderCapabilities,
    options: ChunkingOptions = DEFAULT_CHUNKING_OPTIONS,
) -> list[ChunkRecord]:
    """Work out what to send one service, without writing anything yet.

    The common answer, for every service except OpenAI, is one chunk covering
    the whole recording and pointing at the canonical file itself. Nothing is
    cut, nothing is copied, and the offset is zero.

    Where the recording will not fit, boundaries are chosen against this
    service's own limits, quiet moments are preferred, and each chunk is
    given the exact region of canonical time it covers. Sizes are estimated
    rather than measured, which is the whole point of planning separately
    from writing: a run can tell the user it is about to make eleven requests
    before it has spent a minute of disc writing them.
    """
    duration = max(0.0, canonical.duration)
    accepted = frozenset(name.lower() for name in capabilities.accepted_containers)
    container_accepted = canonical.container.lower() in accepted

    fits_whole = (
        container_accepted
        and canonical.size_bytes <= capabilities.maximum_file_bytes
        and (
            capabilities.maximum_duration_seconds is None
            or duration <= capabilities.maximum_duration_seconds
        )
    )
    if fits_whole:
        return [
            ChunkRecord(
                provider=provider,
                chunk_index=0,
                canonical_start=0.0,
                canonical_end=duration,
                canonical_offset=0.0,
                encoded_size_bytes=canonical.size_bytes,
                path=canonical.path,
            )
        ]

    hard_seconds = _longest_acceptable_chunk(canonical, capabilities, options)
    converted_whole = estimate_lossless_bytes(
        duration, canonical.sample_rate, canonical.channels, options.output_format
    )
    fits_once_converted = (
        duration <= hard_seconds and converted_whole <= capabilities.maximum_file_bytes
    )
    if fits_once_converted:
        # The recording is short enough; it is only in the wrong format. One
        # chunk covering all of it, written out in a format this service
        # takes, and the offset is still zero.
        return [
            ChunkRecord(
                provider=provider,
                chunk_index=0,
                canonical_start=0.0,
                canonical_end=duration,
                canonical_offset=0.0,
                encoded_size_bytes=converted_whole,
            )
        ]

    boundaries = _choose_boundaries(canonical, hard_seconds, options)
    return _records_for(canonical, provider, boundaries, hard_seconds, options)


def _longest_acceptable_chunk(
    canonical: CanonicalAudio,
    capabilities: ProviderCapabilities,
    options: ChunkingOptions,
) -> float:
    """The longest chunk this service would accept, in seconds.

    Both of the service's limits are turned into a length in seconds so that
    one number governs the plan. The size limit becomes a length because the
    chunks are written losslessly, at a fixed number of bytes a second.
    """
    per_second = lossless_bytes_per_second(
        canonical.sample_rate, canonical.channels, options.output_format
    )
    seconds = capabilities.maximum_file_bytes / max(1.0, per_second)
    if capabilities.maximum_duration_seconds is not None:
        seconds = min(seconds, capabilities.maximum_duration_seconds)
    return max(1.0, seconds)


def _choose_boundaries(
    canonical: CanonicalAudio,
    hard_seconds: float,
    options: ChunkingOptions,
) -> list[float]:
    """Pick the points at which one chunk hands over to the next.

    The arithmetic that keeps every chunk inside the limit is worth setting
    out, because it is what makes this safe rather than merely likely to
    work. The overlap, the search margin and the minimum tail are each capped
    at a tenth of the limit, and the target length is the safety fraction of
    the limit with all three subtracted. A chunk is then at most the overlap
    plus the target plus the search margin plus the tail it may absorb, which
    adds back up to exactly the safety fraction of the limit. No chunk can
    exceed it, whatever the silence search decides.
    """
    duration = max(0.0, canonical.duration)
    tenth = hard_seconds * 0.1
    overlap = min(options.overlap_seconds, tenth)
    search = min(options.boundary_search_seconds, tenth)
    minimum = min(options.minimum_chunk_seconds, tenth)
    target = hard_seconds * options.safety_fraction - overlap - search - minimum
    if target <= 0.0:
        target = hard_seconds * 0.5

    levels = _measure_levels(Path(canonical.path), options) if search > 0.0 else []

    boundaries = [0.0]
    position = 0.0
    while duration - position > target + minimum:
        ideal = position + target
        boundary = _quietest_near(levels, ideal, search, options.measurement_seconds)
        # However attractive a quiet moment is, a boundary may not shorten a
        # chunk below the minimum, push it past the search margin, or leave a
        # tail too short to be worth a request of its own.
        boundary = min(max(boundary, position + minimum), ideal + search, duration - minimum)
        if boundary <= position:
            break
        boundaries.append(boundary)
        position = boundary
    boundaries.append(duration)
    return boundaries


def _records_for(
    canonical: CanonicalAudio,
    provider: Provider,
    boundaries: Sequence[float],
    hard_seconds: float,
    options: ChunkingOptions,
) -> list[ChunkRecord]:
    """Turn a list of hand-over points into chunk plans with their overlaps.

    The overlap is put on the front of each chunk rather than the back, which
    is what :class:`ChunkRecord` means by ``overlap_before``. Chunk two
    therefore starts a few seconds before chunk one ends, and the region they
    share is the region the join is made in.
    """
    tenth = hard_seconds * 0.1
    overlap = min(options.overlap_seconds, tenth)
    records: list[ChunkRecord] = []
    for index in range(len(boundaries) - 1):
        handover = boundaries[index]
        start = max(0.0, handover - overlap) if index > 0 else 0.0
        end = boundaries[index + 1]
        records.append(
            ChunkRecord(
                provider=provider,
                chunk_index=index,
                canonical_start=start,
                canonical_end=end,
                canonical_offset=start,
                encoded_size_bytes=estimate_lossless_bytes(
                    end - start, canonical.sample_rate, canonical.channels, options.output_format
                ),
                overlap_before=handover - start,
            )
        )
    return records


# -- Finding the quiet places --------------------------------------------


def _measure_levels(path: Path, options: ChunkingOptions) -> list[tuple[float, float]]:
    """Measure the level of the whole recording in short, even windows.

    FFmpeg is asked to hand out frames of a fixed length and to report the
    level of each one, so nothing here has to open sample data or work out
    what a decibel is. What comes back is a list of window start times and
    levels, quietest last being the only thing anyone asks of it.

    A recording that cannot be measured comes back as an empty list rather
    than as an error, because failing to find a quiet moment is not a reason
    to abandon a transcript. Boundaries then fall where the arithmetic put
    them.
    """
    import av
    from av.filter import Graph

    levels: list[tuple[float, float]] = []
    try:
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            window = max(1, int(options.measurement_seconds * int(stream.sample_rate)))
            graph = Graph()
            chain = [graph.add_abuffer(template=stream)]
            # Folding to one channel first means one level per window rather
            # than one per channel, and the level of a stereo recording of a
            # room is the same in both channels anyway.
            chain.append(graph.add("aformat", "channel_layouts=mono:sample_fmts=fltp"))
            chain.append(graph.add("asetnsamples", f"n={window}:p=0"))
            chain.append(
                graph.add(
                    "astats",
                    "metadata=1:reset=1:measure_perchannel=RMS_level:measure_overall=RMS_level",
                )
            )
            chain.append(graph.add("abuffersink"))
            for upstream, downstream in zip(chain, chain[1:]):
                upstream.link_to(downstream)
            graph.configure()

            def drain() -> None:
                while True:
                    try:
                        frame = graph.pull()
                    except (av.BlockingIOError, av.EOFError):
                        return
                    levels.append((frame.time or 0.0, _level_from(frame)))

            for frame in container.decode(stream):
                graph.push(frame)
                drain()
            graph.push(None)
            drain()
    except (OSError, IndexError, av.FFmpegError) as error:
        # Failing here costs only the preference for quiet boundaries, so it
        # is noted and the plan carries on with boundaries where the
        # arithmetic put them. Letting it escape would turn a cosmetic
        # problem into a lost transcript.
        _log.debug("The levels of %s could not be measured: %s", path, error)
        return []
    return levels


def _level_from(frame) -> float:
    """The level of one measured window, in decibels."""
    metadata = frame.metadata or {}
    text = metadata.get("lavfi.astats.Overall.RMS_level")
    if text is None:
        text = metadata.get("lavfi.astats.1.RMS_level")
    if text is None:
        return _SILENT_LEVEL_DB
    try:
        value = float(text)
    except ValueError:
        return _SILENT_LEVEL_DB
    return value if math.isfinite(value) else _SILENT_LEVEL_DB


def _quietest_near(
    levels: Sequence[tuple[float, float]],
    ideal: float,
    search: float,
    measurement_seconds: float,
) -> float:
    """The best place to cut near ``ideal``, which is the quietest moment there.

    Ties are broken first by nearness to the wanted position and then by
    which comes first, so the same recording always gives the same
    boundaries. Levels are compared rounded to a tenth of a decibel, because
    two windows of the same silence differ in the sixth decimal place and
    nobody should be choosing a boundary on that.
    """
    if not levels:
        return ideal
    lowest = ideal - search
    highest = ideal + search
    best: tuple[float, float, float] | None = None
    best_time = ideal
    for start, level in levels:
        centre = start + measurement_seconds / 2.0
        if centre < lowest or centre > highest:
            continue
        key = (round(level, 1), abs(centre - ideal), centre)
        if best is None or key < best:
            best = key
            best_time = centre
    return best_time


# -- Writing the chunks ---------------------------------------------------


def write_chunks(
    canonical: CanonicalAudio,
    plans: Sequence[ChunkRecord],
    folder: Path,
    options: ChunkingOptions = DEFAULT_CHUNKING_OPTIONS,
    capabilities: ProviderCapabilities | None = None,
) -> list[ChunkRecord]:
    """Write the audio for a plan, and correct the plan to match what was written.

    A plan that already points at a file, which is what a whole recording
    that fits looks like, is left alone. Everything else is cut from the
    canonical file, and the record that comes back carries the span the clip
    really covers and the size it really is, not the span and size that were
    planned. Those are almost always the same; where they are not, it is
    because the recording ended sooner than its header claimed, and a chunk
    that reported the planned span would push every word in it out of place.
    """
    folder.mkdir(parents=True, exist_ok=True)
    written: list[ChunkRecord] = []
    for plan in plans:
        if plan.path:
            written.append(plan)
            continue
        destination = folder / (
            f"{plan.provider.value}-chunk-{plan.chunk_index:03d}{options.output_format.extension}"
        )
        clip = cut_window(
            canonical,
            AudioSpan(plan.canonical_start, plan.canonical_end),
            destination,
            options.output_format,
        )
        # A clip that starts later than planned has that much less of the
        # previous chunk in front of it, and the overlap has to shrink with
        # it or the join would be made in a region one of the chunks does not
        # contain.
        lost_at_the_front = max(0.0, clip.span.start - plan.canonical_start)
        record = replace(
            plan,
            canonical_start=clip.span.start,
            canonical_end=clip.span.end,
            canonical_offset=clip.canonical_offset,
            encoded_size_bytes=clip.size_bytes,
            path=str(clip.path),
            overlap_before=max(0.0, plan.overlap_before - lost_at_the_front),
        )
        if capabilities is not None and record.encoded_size_bytes > capabilities.maximum_file_bytes:
            _log.warning(
                "Chunk %d of %s came out at %d bytes, over the %d byte limit for %s.",
                record.chunk_index,
                canonical.path,
                record.encoded_size_bytes,
                capabilities.maximum_file_bytes,
                plan.provider.display_name,
            )
        written.append(record)
    return written


def plan_and_write_chunks(
    canonical: CanonicalAudio,
    provider: Provider,
    capabilities: ProviderCapabilities,
    folder: Path,
    options: ChunkingOptions = DEFAULT_CHUNKING_OPTIONS,
) -> list[ChunkRecord]:
    """Plan the chunks for one service and write whatever has to be written."""
    plans = plan_chunks(canonical, provider, capabilities, options)
    return write_chunks(canonical, plans, folder, options, capabilities)


# -- Putting the answers back together ------------------------------------


def merge_overlapping_tokens(
    tokens: Sequence[ProviderToken],
    chunks: Sequence[ChunkRecord],
    minimum_matched_words: int = MINIMUM_MATCHED_WORDS,
) -> MergedTokens:
    """Join the words from several chunks into one transcript, without duplicates.

    Chunks overlap on purpose, so the words in the overlap come back twice
    and one of the two copies has to go. Which one goes is decided the same
    way every time, because a transcript that changed between two runs over
    the same audio would be impossible to argue with.

    Where the words carry times, the rule is that the earlier chunk owns
    everything up to the middle of the overlap and the later chunk owns
    everything after it. A word is placed by its own midpoint rather than by
    its start, so a word lying across the line is claimed by exactly one
    side. The middle is the right place to divide because a word at the edge
    of a chunk was heard with context on one side only, and the middle of the
    overlap is the point at which the two chunks stop being better than each
    other.

    Where the words carry no times, which is what OpenAI returns, the join is
    made on the words themselves. The longest run of words that ends the
    earlier chunk and begins the later one is found, and the later chunk
    starts after it. A run has to be several words long to be believed, since
    one common word matches almost anywhere.

    Where no run can be found, the join falls where the length of the overlap
    suggests it should, and the join is reported as uncertain. That is
    deliberately not silent: a handful of words there may be repeated or
    missing, and the pipeline is expected to put them in front of a person
    rather than let a seam through unremarked.

    The words come back renumbered from zero, because what is returned is one
    service's transcript of the whole recording and its indexes are what
    everything downstream points at.
    """
    ordered_chunks = sorted(chunks, key=lambda chunk: (chunk.canonical_start, chunk.chunk_index))
    grouped = _group_by_chunk(tokens, ordered_chunks)
    if not ordered_chunks:
        return MergedTokens(tokens=tuple(_renumbered(list(tokens))))

    normalise, are_equivalent = _text_comparison()
    merged: list[ProviderToken] = list(grouped.get(ordered_chunks[0].chunk_index, []))
    previous = ordered_chunks[0]
    joins: list[ChunkJoin] = []

    for chunk in ordered_chunks[1:]:
        later = list(grouped.get(chunk.chunk_index, []))
        overlap_start = chunk.canonical_start
        overlap_end = min(previous.canonical_end, chunk.canonical_end)
        if overlap_end - overlap_start <= 1e-6:
            joins.append(
                ChunkJoin(
                    earlier_chunk_index=previous.chunk_index,
                    later_chunk_index=chunk.chunk_index,
                    canonical_time=overlap_start,
                    method="abutting",
                )
            )
            merged.extend(later)
            previous = chunk
            continue

        earlier_tail = _tokens_of(merged, previous.chunk_index)
        if _all_timed(earlier_tail) and _all_timed(later):
            kept_earlier, kept_later, join = _join_on_time(
                earlier_tail, later, previous, chunk, overlap_start, overlap_end
            )
        else:
            kept_earlier, kept_later, join = _join_on_text(
                earlier_tail,
                later,
                previous,
                chunk,
                overlap_start,
                overlap_end,
                minimum_matched_words,
                normalise,
                are_equivalent,
            )
        merged = _without(merged, previous.chunk_index) + kept_earlier + kept_later
        joins.append(join)
        previous = chunk

    return MergedTokens(tokens=tuple(_renumbered(merged)), joins=tuple(joins))


def _join_on_time(
    earlier: list[ProviderToken],
    later: list[ProviderToken],
    earlier_chunk: ChunkRecord,
    later_chunk: ChunkRecord,
    overlap_start: float,
    overlap_end: float,
) -> tuple[list[ProviderToken], list[ProviderToken], ChunkJoin]:
    """Divide an overlap at its midpoint, by the words' own times."""
    midpoint = (overlap_start + overlap_end) / 2.0
    kept_earlier = [token for token in earlier if _midpoint_of(token) < midpoint]
    kept_later = [token for token in later if _midpoint_of(token) >= midpoint]
    return (
        kept_earlier,
        kept_later,
        ChunkJoin(
            earlier_chunk_index=earlier_chunk.chunk_index,
            later_chunk_index=later_chunk.chunk_index,
            canonical_time=midpoint,
            method="time",
            certain=True,
        ),
    )


def _join_on_text(
    earlier: list[ProviderToken],
    later: list[ProviderToken],
    earlier_chunk: ChunkRecord,
    later_chunk: ChunkRecord,
    overlap_start: float,
    overlap_end: float,
    minimum_matched_words: int,
    normalise: Callable[[str], str],
    are_equivalent: Callable[[str, str], bool],
) -> tuple[list[ProviderToken], list[ProviderToken], ChunkJoin]:
    """Divide an overlap by finding the words the two chunks share.

    Only real words are compared. Punctuation that a service returns as its
    own entry, and anything that normalises away to nothing, is skipped for
    the comparison and then carried along with whichever side its words end
    up on.
    """
    overlap_seconds = max(0.0, overlap_end - overlap_start)
    most = max(minimum_matched_words, int(math.ceil(overlap_seconds * _FASTEST_WORDS_PER_SECOND)))

    earlier_words = _comparable_words(earlier, normalise)
    later_words = _comparable_words(later, normalise)
    reach = min(most, len(earlier_words), len(later_words))

    for length in range(reach, minimum_matched_words - 1, -1):
        tail = earlier_words[-length:]
        head = later_words[:length]
        if all(are_equivalent(a.text, b.text) for a, b in zip(tail, head)):
            # Everything in the later chunk up to and including the last
            # matched word is the same speech the earlier chunk already has.
            cut = head[-1].position + 1
            return (
                earlier,
                later[cut:],
                ChunkJoin(
                    earlier_chunk_index=earlier_chunk.chunk_index,
                    later_chunk_index=later_chunk.chunk_index,
                    canonical_time=overlap_end,
                    method="text",
                    matched_words=length,
                    certain=True,
                ),
            )

    # Nothing matched. The overlap is a known fraction of the later chunk, so
    # the words it probably holds are dropped from the later chunk and the
    # join is reported as uncertain rather than a duplicated or missing
    # passage being left for somebody to find.
    later_duration = max(1e-6, later_chunk.canonical_end - later_chunk.canonical_start)
    share = min(1.0, overlap_seconds / later_duration)
    nominal = round(share * len(later_words))
    cut = later_words[nominal - 1].position + 1 if 0 < nominal <= len(later_words) else 0
    return (
        earlier,
        later[cut:],
        ChunkJoin(
            earlier_chunk_index=earlier_chunk.chunk_index,
            later_chunk_index=later_chunk.chunk_index,
            canonical_time=overlap_end,
            method="nominal",
            matched_words=0,
            certain=False,
        ),
    )


@dataclass(frozen=True)
class _ComparableWord:
    """One word of a chunk, with where it sits in that chunk's own list."""

    position: int
    text: str


def _comparable_words(
    tokens: Sequence[ProviderToken], normalise: Callable[[str], str]
) -> list[_ComparableWord]:
    words: list[_ComparableWord] = []
    for position, token in enumerate(tokens):
        if token.is_punctuation:
            continue
        if not normalise(token.text).strip():
            continue
        words.append(_ComparableWord(position=position, text=token.text))
    return words


def _text_comparison() -> tuple[Callable[[str], str], Callable[[str, str], bool]]:
    """The two text functions used to compare words across a seam.

    They belong to the normalisation module, which knows about casing,
    hyphenation, German transliteration and the rest. It is imported here
    rather than at the top of the file so that this module keeps working, on
    a plainer comparison, while that one is still being written.
    """
    try:
        from audio_transcriber.transcription.normalise import (  # type: ignore[attr-defined]
            are_equivalent,
            normalise,
        )
    except (ImportError, AttributeError):
        return _plain_normalise, _plain_equivalent
    return normalise, are_equivalent


#: Punctuation and quoting that a word may be wearing on either side. It is
#: taken off before two words are compared, so that "Bosch," and "Bosch" are
#: the same word arriving at a chunk boundary from two directions.
_EDGE_CHARACTERS = " \t\n.,;:!?\"'()[]{}…-–—“”‘’"


def _plain_normalise(text: str) -> str:
    """A modest stand-in for the real normaliser: case and edge punctuation."""
    return text.strip(_EDGE_CHARACTERS).casefold()


def _plain_equivalent(first: str, second: str) -> bool:
    return _plain_normalise(first) == _plain_normalise(second)


def _group_by_chunk(
    tokens: Sequence[ProviderToken], chunks: Sequence[ChunkRecord]
) -> dict[int, list[ProviderToken]]:
    """Sort the words into the chunk each came back in, in their own order."""
    known = {chunk.chunk_index for chunk in chunks}
    grouped: dict[int, list[ProviderToken]] = {index: [] for index in known}
    for token in sorted(tokens, key=lambda token: (token.chunk_index, token.index)):
        grouped.setdefault(token.chunk_index, []).append(token)
    return grouped


def _tokens_of(tokens: Sequence[ProviderToken], chunk_index: int) -> list[ProviderToken]:
    return [token for token in tokens if token.chunk_index == chunk_index]


def _without(tokens: Sequence[ProviderToken], chunk_index: int) -> list[ProviderToken]:
    return [token for token in tokens if token.chunk_index != chunk_index]


def _all_timed(tokens: Sequence[ProviderToken]) -> bool:
    return bool(tokens) and all(token.has_timing for token in tokens)


def _midpoint_of(token: ProviderToken) -> float:
    return ((token.start or 0.0) + (token.end or 0.0)) / 2.0


def _renumbered(tokens: Iterable[ProviderToken]) -> list[ProviderToken]:
    """Number the words from zero again, keeping everything else about them."""
    return [replace(token, index=position) for position, token in enumerate(tokens)]
