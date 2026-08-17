"""Tests for cutting a recording up per service, and stitching it back together.

Two quite different things are being checked here.

The planning and writing tests work on real audio, because the question they
ask is whether a time inside a chunk still means what it should. The
recording is the one built out of a different tone every second, so a chunk
can be asked which second of the original it holds and made to answer.

The stitching tests work on made-up words, because there the audio is beside
the point: what matters is which of two copies of a word survives, and that
the answer is the same every time. Both routes through the join are covered,
the one for services that time their words and the one for OpenAI, which
does not time anything and has to be joined on the words themselves.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcriber.audio.enhance import OUTPUT_FORMATS
from audio_transcriber.transcription.canonical import (
    cut_window,
    prepare_canonical_audio,
    probe_audio,
)
from audio_transcriber.transcription.chunking import (
    ChunkingOptions,
    merge_overlapping_tokens,
    plan_and_write_chunks,
    plan_chunks,
    write_chunks,
)
from audio_transcriber.transcription.model import (
    AudioSpan,
    CanonicalAudio,
    ChunkRecord,
    Provider,
    ProviderToken,
)
from audio_transcriber.transcription.providers.base import ProviderCapabilities

from tests.test_canonical import MARKER_TONES, dominant_tone, write_marker_audio

WAV16, _WAV24, FLAC = OUTPUT_FORMATS

#: Roomy limits, standing in for ElevenLabs and AssemblyAI, which take a
#: recording of any length this application will ever see.
GENEROUS = ProviderCapabilities(maximum_file_bytes=5 * 1024 * 1024 * 1024)

#: A limit small enough that a six-second recording has to be cut up. It
#: stands in for OpenAI, whose twenty-five megabytes does the same thing to a
#: recording of an hour.
TIGHT = ProviderCapabilities(maximum_file_bytes=60_000)


def write_gapped_audio(
    path: Path, seconds: int = 6, silent_from: float = 1.3, silent_until: float = 1.5
) -> None:
    """Write a steady tone with one stretch of silence in the middle of it.

    It stands in for the pause between two sentences, which is where a chunk
    boundary belongs.
    """
    import array
    import math

    import av

    rate = 16000
    total = rate * seconds
    samples = array.array("h", (0,)) * total
    amplitude = 0.5 * 32767.0
    for index in range(total):
        moment = index / rate
        if silent_from <= moment < silent_until:
            continue
        samples[index] = int(amplitude * math.sin(2.0 * math.pi * 440.0 * index / rate))

    with av.open(str(path), "w") as container:
        stream = container.add_stream("pcm_s16le", rate=rate, layout="mono")
        frame = av.AudioFrame(format="s16", layout="mono", samples=total)
        frame.planes[0].update(samples.tobytes())
        frame.sample_rate = rate
        frame.pts = 0
        queue = av.audio.fifo.AudioFifo()
        queue.write(frame)
        while True:
            block = queue.read(1024, partial=True)
            if block is None:
                break
            for packet in stream.encode(block):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


@pytest.fixture
def canonical(tmp_path) -> CanonicalAudio:
    """Six seconds of marker audio, settled as the canonical recording."""
    source = tmp_path / "marker.wav"
    write_marker_audio(source)
    return prepare_canonical_audio(source, tmp_path / "work", {"wav"})


# -- Planning -------------------------------------------------------------


def test_a_service_that_takes_the_whole_recording_is_sent_the_whole_recording(canonical):
    """The common case, and the one the specification is most insistent on.

    OpenAI's limit is no reason to cut up what ElevenLabs would happily
    accept whole, and every boundary avoided is a boundary that cannot go
    wrong.
    """
    chunks = plan_chunks(canonical, Provider.ELEVENLABS, GENEROUS)

    assert len(chunks) == 1
    assert chunks[0].canonical_start == 0.0
    assert chunks[0].canonical_end == pytest.approx(canonical.duration)
    assert chunks[0].canonical_offset == 0.0
    assert chunks[0].overlap_before == 0.0
    assert chunks[0].path == canonical.path, "nothing should be cut or copied"
    assert chunks[0].encoded_size_bytes == canonical.size_bytes


def test_nothing_is_written_for_a_service_that_needs_no_chunks(canonical, tmp_path):
    folder = tmp_path / "chunks"

    written = write_chunks(canonical, plan_chunks(canonical, Provider.ELEVENLABS, GENEROUS), folder)

    assert written[0].path == canonical.path
    assert list(folder.iterdir()) == []


def test_a_service_with_a_tight_limit_gets_several_chunks(canonical):
    chunks = plan_chunks(canonical, Provider.OPENAI, TIGHT)

    assert len(chunks) > 1
    assert chunks[0].canonical_start == 0.0
    assert chunks[-1].canonical_end == pytest.approx(canonical.duration)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.path is None for chunk in chunks), "planning must not write anything"


def test_every_chunk_carries_the_offset_that_is_its_own_start(canonical):
    """The offset is what converts a service's local time to canonical time.

    It is the same number as the chunk's start by definition, and the two
    drifting apart is the fault that would put every word in a chunk in the
    wrong place.
    """
    for chunk in plan_chunks(canonical, Provider.OPENAI, TIGHT):
        assert chunk.canonical_offset == chunk.canonical_start


def test_the_chunks_cover_the_recording_and_overlap_rather_than_leaving_gaps(canonical):
    chunks = plan_chunks(canonical, Provider.OPENAI, TIGHT)

    for earlier, later in zip(chunks, chunks[1:]):
        assert later.canonical_start < earlier.canonical_end, "a gap would lose speech"
        assert later.overlap_before == pytest.approx(earlier.canonical_end - later.canonical_start)
    covered = chunks[-1].canonical_end - chunks[0].canonical_start
    assert covered == pytest.approx(canonical.duration)


def test_no_planned_chunk_comes_near_the_size_the_service_refuses(canonical):
    """Aiming at the limit is how a run fails on its last chunk."""
    for chunk in plan_chunks(canonical, Provider.OPENAI, TIGHT):
        assert chunk.encoded_size_bytes <= TIGHT.maximum_file_bytes


def test_a_limit_on_length_chunks_a_recording_that_is_small_enough(canonical):
    """AssemblyAI takes five gigabytes but only ten hours, so length can bite."""
    capabilities = ProviderCapabilities(
        maximum_file_bytes=5 * 1024 * 1024 * 1024, maximum_duration_seconds=2.5
    )

    chunks = plan_chunks(canonical, Provider.ASSEMBLYAI, capabilities)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.canonical_end - chunk.canonical_start <= 2.5


def test_a_boundary_moves_to_the_quiet_moment_near_it(tmp_path):
    """Cutting in a gap is what makes the two halves joinable afterwards.

    A boundary that falls in the middle of a word leaves each chunk holding
    half of it, and half a word matches nothing on the other side. For
    OpenAI, whose words come back with no times on them, that match is the
    only thing the join has to work with, so a quiet moment near the wanted
    boundary is taken in preference to the boundary itself.
    """
    source = tmp_path / "gapped.wav"
    write_gapped_audio(source, seconds=6, silent_from=1.3, silent_until=1.5)
    canonical = prepare_canonical_audio(source, tmp_path / "work", {"wav"})

    chunks = plan_chunks(canonical, Provider.OPENAI, TIGHT)

    # The arithmetic would put the first hand-over at 1.25 seconds, in the
    # middle of the tone. The silence a little after it wins instead.
    handover = chunks[1].canonical_start + chunks[1].overlap_before
    assert 1.3 <= handover <= 1.5


def test_the_boundaries_are_the_same_every_time_the_same_recording_is_planned(tmp_path):
    source = tmp_path / "gapped.wav"
    write_gapped_audio(source, seconds=6, silent_from=1.3, silent_until=1.5)
    canonical = prepare_canonical_audio(source, tmp_path / "work", {"wav"})

    first = plan_chunks(canonical, Provider.OPENAI, TIGHT)
    second = plan_chunks(canonical, Provider.OPENAI, TIGHT)

    assert [chunk.canonical_start for chunk in first] == [
        chunk.canonical_start for chunk in second
    ]


def test_a_recording_in_a_format_a_service_will_not_take_is_converted_whole(canonical):
    """One chunk, covering everything, but written out rather than pointed at."""
    capabilities = ProviderCapabilities(accepted_containers=frozenset({"mp3"}))

    chunks = plan_chunks(canonical, Provider.MICROSOFT, capabilities)

    assert len(chunks) == 1
    assert chunks[0].canonical_offset == 0.0
    assert chunks[0].canonical_end == pytest.approx(canonical.duration)
    assert chunks[0].path is None, "it has to be written before it can be sent"


# -- Writing --------------------------------------------------------------


def test_the_chunk_files_are_written_and_are_as_long_as_they_claim(canonical, tmp_path):
    chunks = plan_and_write_chunks(canonical, Provider.OPENAI, TIGHT, tmp_path / "chunks")

    for chunk in chunks:
        assert chunk.path is not None
        written = probe_audio(Path(chunk.path))
        assert written.duration == pytest.approx(
            chunk.canonical_end - chunk.canonical_start, abs=0.01
        )
        assert chunk.encoded_size_bytes == written.size_bytes
        assert chunk.encoded_size_bytes <= TIGHT.maximum_file_bytes


def test_a_time_inside_a_chunk_adds_up_to_the_right_canonical_time(canonical, tmp_path):
    """The whole reason chunks carry an offset, checked against the audio.

    Every half second of the recording is found in the chunk that holds it,
    converted to that chunk's local time, cut back out of the chunk file, and
    asked which second of the original it came from. An offset out by even a
    fraction of a second gives the wrong tone.
    """
    chunks = plan_and_write_chunks(canonical, Provider.OPENAI, TIGHT, tmp_path / "chunks")

    for second in range(len(MARKER_TONES)):
        canonical_time = second + 0.5
        holder = next(
            chunk
            for chunk in chunks
            if chunk.canonical_start <= canonical_time - 0.05
            and chunk.canonical_end >= canonical_time + 0.05
        )
        chunk_audio = _as_canonical(Path(holder.path))
        local = canonical_time - holder.canonical_offset

        clip = cut_window(
            chunk_audio, AudioSpan(local - 0.05, local + 0.05), tmp_path / f"probe{second}.wav"
        )

        assert dominant_tone(clip.path) == MARKER_TONES[second], (
            f"canonical {canonical_time}s came back as the wrong second of audio"
        )


def test_the_written_chunks_between_them_hold_all_of_the_recording(canonical, tmp_path):
    chunks = plan_and_write_chunks(canonical, Provider.OPENAI, TIGHT, tmp_path / "chunks")

    assert chunks[0].canonical_start == 0.0
    assert chunks[-1].canonical_end == pytest.approx(canonical.duration, abs=0.01)
    for earlier, later in zip(chunks, chunks[1:]):
        assert later.canonical_start <= earlier.canonical_end


def _as_canonical(path: Path) -> CanonicalAudio:
    """Treat a chunk file as a recording in its own right, so it can be cut."""
    probe = probe_audio(path)
    return CanonicalAudio(
        path=str(path),
        original_path=str(path),
        duration=probe.duration,
        sample_rate=probe.sample_rate,
        channels=probe.channels,
        size_bytes=probe.size_bytes,
        container=probe.container,
    )


# -- Stitching the answers back together ----------------------------------


def chunk_record(index: int, start: float, end: float, overlap: float = 0.0) -> ChunkRecord:
    return ChunkRecord(
        provider=Provider.OPENAI,
        chunk_index=index,
        canonical_start=start,
        canonical_end=end,
        canonical_offset=start,
        encoded_size_bytes=1000,
        overlap_before=overlap,
    )


def words(
    texts: str, chunk_index: int, first_start: float | None = None, step: float = 1.0
) -> list[ProviderToken]:
    """Turn a sentence into tokens, timed or untimed.

    Leaving ``first_start`` out gives words with no times at all, which is
    what OpenAI returns and what the text join has to cope with.
    """
    tokens: list[ProviderToken] = []
    for index, text in enumerate(texts.split()):
        start = None if first_start is None else first_start + index * step
        tokens.append(
            ProviderToken(
                provider=Provider.OPENAI,
                index=index,
                text=text,
                start=start,
                end=None if start is None else start + step * 0.8,
                chunk_index=chunk_index,
            )
        )
    return tokens


def texts_of(merged) -> list[str]:
    return [token.text for token in merged.tokens]


def test_a_timed_overlap_is_divided_at_its_midpoint():
    """Each word goes to the chunk that heard it with more around it.

    The chunks share the seconds from 4 to 6, so the line falls at 5. The
    earlier chunk keeps what it heard before then and gives up the rest,
    because a word at the very edge of a chunk was heard with context on one
    side only.
    """
    chunks = [chunk_record(0, 0.0, 6.0), chunk_record(1, 4.0, 10.0, overlap=2.0)]
    first = words("one two three four five six", 0, first_start=0.0)
    second = words("five six seven eight", 1, first_start=4.0)

    merged = merge_overlapping_tokens(first + second, chunks)

    assert texts_of(merged) == ["one", "two", "three", "four", "five", "six", "seven", "eight"]
    assert merged.joins[0].method == "time"
    assert merged.joins[0].canonical_time == pytest.approx(5.0)
    assert not merged.uncertain_joins


def test_a_word_lying_across_the_line_is_kept_by_exactly_one_chunk():
    """It is placed by its own middle, so it cannot be dropped or doubled."""
    chunks = [chunk_record(0, 0.0, 6.0), chunk_record(1, 4.0, 10.0, overlap=2.0)]
    straddling = ProviderToken(
        provider=Provider.OPENAI, index=0, text="hello", start=4.8, end=5.4, chunk_index=0
    )
    again = ProviderToken(
        provider=Provider.OPENAI, index=0, text="hello", start=4.8, end=5.4, chunk_index=1
    )

    merged = merge_overlapping_tokens([straddling, again], chunks)

    assert texts_of(merged) == ["hello"]


def test_the_words_come_back_numbered_from_zero_again():
    """What comes out is one service's transcript, so its numbering is its own."""
    chunks = [chunk_record(0, 0.0, 6.0), chunk_record(1, 4.0, 10.0, overlap=2.0)]
    merged = merge_overlapping_tokens(
        words("one two three", 0, first_start=0.0) + words("four five", 1, first_start=6.0),
        chunks,
    )

    assert [token.index for token in merged.tokens] == [0, 1, 2, 3, 4]
    assert [token.chunk_index for token in merged.tokens] == [0, 0, 0, 1, 1]


def test_untimed_words_are_joined_where_the_words_agree():
    """OpenAI returns no times at all, so the words are all there is to go on.

    The last four words of the earlier chunk and the first four of the later
    one are the same speech heard twice. The join is made after them, and the
    later chunk contributes only what the earlier one did not have.
    """
    chunks = [chunk_record(0, 0.0, 6.0), chunk_record(1, 4.0, 10.0, overlap=2.0)]
    first = words("the meeting began at nine on Tuesday morning", 0)
    second = words("nine on Tuesday morning and ran for an hour", 1)

    merged = merge_overlapping_tokens(first + second, chunks)

    assert texts_of(merged) == (
        "the meeting began at nine on Tuesday morning and ran for an hour".split()
    )
    assert merged.joins[0].method == "text"
    assert merged.joins[0].matched_words == 4
    assert not merged.uncertain_joins


def test_a_word_spelt_differently_on_each_side_still_joins():
    """The same speech, transcribed twice, is not transcribed identically.

    A word at the end of one request and the start of the next can come back
    capitalised in one and not the other, or with a comma attached. The join
    is made on normalised words for exactly that reason, so a difference of
    punctuation does not cost the seam its match.
    """
    chunks = [chunk_record(0, 0.0, 6.0), chunk_record(1, 4.0, 10.0, overlap=2.0)]
    first = words("we spoke to Doctor Bosch about", 0)
    second = words("Doctor bosch, about the report", 1)

    merged = merge_overlapping_tokens(first + second, chunks)

    assert texts_of(merged) == ["we", "spoke", "to", "Doctor", "Bosch", "about", "the", "report"]
    assert merged.joins[0].method == "text"
    assert merged.joins[0].matched_words == 3


def test_a_single_common_word_is_not_enough_to_join_on():
    """"The" matches almost anywhere, and joining on it would join in the
    wrong place, which loses real speech without anybody noticing."""
    chunks = [chunk_record(0, 0.0, 6.0), chunk_record(1, 4.0, 10.0, overlap=2.0)]
    first = words("she opened the", 0)
    second = words("the window was open", 1)

    merged = merge_overlapping_tokens(first + second, chunks)

    assert merged.joins[0].method != "text"
    assert not merged.joins[0].certain


def test_a_join_with_nothing_matching_is_reported_rather_than_hidden():
    """Where the two chunks share no words, the join is a guess and says so.

    Something has to be done, because leaving both copies duplicates a
    passage and dropping both loses one. The overlap is a known fraction of
    the later chunk, so that fraction of its words is dropped and the join is
    flagged, which lets the pipeline put those words in front of a person
    instead of letting a silent seam through.
    """
    chunks = [chunk_record(0, 0.0, 6.0), chunk_record(1, 4.0, 10.0, overlap=2.0)]
    first = words("completely different words over here", 0)
    second = words("nothing at all like the other side of it", 1)

    merged = merge_overlapping_tokens(first + second, chunks)

    join = merged.joins[0]
    assert join.method == "nominal"
    assert not join.certain
    assert join.matched_words == 0
    assert merged.uncertain_joins == (join,)
    assert "may be repeated or missing" in join.description
    # A third of the later chunk is overlap, so about a third of its words go.
    assert texts_of(merged)[:5] == ["completely", "different", "words", "over", "here"]
    assert len(merged.tokens) < len(first) + len(second)


def test_chunks_that_do_not_overlap_are_simply_put_end_to_end():
    chunks = [chunk_record(0, 0.0, 5.0), chunk_record(1, 5.0, 10.0)]
    merged = merge_overlapping_tokens(words("one two", 0) + words("three four", 1), chunks)

    assert texts_of(merged) == ["one", "two", "three", "four"]
    assert merged.joins[0].method == "abutting"
    assert not merged.uncertain_joins


def test_one_chunk_needs_no_joining_at_all():
    chunks = [chunk_record(0, 0.0, 6.0)]

    merged = merge_overlapping_tokens(words("nothing to join here", 0), chunks)

    assert texts_of(merged) == ["nothing", "to", "join", "here"]
    assert merged.joins == ()


def test_the_same_words_always_give_the_same_transcript():
    """Determinism is not a nicety here.

    A transcript that changed between two runs over the same audio could not
    be argued with, reprocessed, or compared against an earlier version.
    """
    chunks = [
        chunk_record(0, 0.0, 6.0),
        chunk_record(1, 4.0, 10.0, overlap=2.0),
        chunk_record(2, 8.0, 14.0, overlap=2.0),
    ]
    tokens = (
        words("the first part of what was said", 0)
        + words("what was said in the middle of it", 1)
        + words("middle of it and then the end", 2)
    )

    first_run = texts_of(merge_overlapping_tokens(tokens, chunks))
    second_run = texts_of(merge_overlapping_tokens(list(reversed(tokens)), chunks))

    assert first_run == second_run
    assert first_run == (
        "the first part of what was said in the middle of it and then the end".split()
    )


def test_a_deliberately_short_overlap_still_joins(canonical, tmp_path):
    """Planning and stitching have to agree about where the overlap is.

    The chunks here come from the planner rather than from the test, so the
    overlap the join is made in is the overlap the planner actually chose.
    """
    options = ChunkingOptions(overlap_seconds=0.5)
    chunks = plan_chunks(canonical, Provider.OPENAI, TIGHT, options)
    assert len(chunks) > 1

    tokens: list[ProviderToken] = []
    for chunk in chunks:
        tokens.extend(words("shared words here and more", chunk.chunk_index))

    merged = merge_overlapping_tokens(tokens, chunks)

    assert len(merged.joins) == len(chunks) - 1
    assert [token.index for token in merged.tokens] == list(range(len(merged.tokens)))
