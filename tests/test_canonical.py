"""Tests for settling the canonical recording and cutting windows out of it.

Everything here works on real audio written by the tests themselves, because
the questions being asked are about files: how long is this one really, was
a copy made, and does the piece that came out start where it says it does.
None of those can be answered by a fake.

The window tests carry most of the weight. A window cut a few milliseconds
away from where it claims to be is the kind of fault that never announces
itself: the clip plays, the service answers, and every word in it lands
slightly out of place for ever after. So the clips are cut from a recording
built out of tones that change every second, and what comes out is measured
by listening for which tone is in it.
"""

from __future__ import annotations

import array
import math
from pathlib import Path

import pytest

from audio_transcriber.audio.enhance import OUTPUT_FORMATS
from audio_transcriber.transcription.canonical import (
    CanonicalAudioError,
    containers_accepted_by_all,
    cut_window,
    decide_canonical_copy,
    estimate_lossless_bytes,
    prepare_canonical_audio,
    probe_audio,
)
from audio_transcriber.transcription.model import AudioSpan
from audio_transcriber.transcription.providers.base import ProviderCapabilities

from tests.conftest import write_fake_audio, write_real_audio

WAV16, _WAV24, FLAC = OUTPUT_FORMATS

#: The tones the marker recording is built from, one a second. They are far
#: enough apart that the strongest one in a clip is unmistakable.
MARKER_TONES = (200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0)

MARKER_RATE = 16000


def write_marker_audio(path: Path, seconds: int = 6, rate: int = MARKER_RATE) -> None:
    """Write a recording whose every second holds a different tone.

    This makes a clip self-describing. Cut one second out of it and the tone
    inside says which second of the recording it came from, so a window that
    is off by even a fraction of a second shows up as two tones where there
    should be one.
    """
    import av

    total = rate * seconds
    samples = array.array("h", (0,)) * total
    amplitude = 0.5 * 32767.0
    for index in range(total):
        tone = MARKER_TONES[min(index // rate, len(MARKER_TONES) - 1)]
        samples[index] = int(amplitude * math.sin(2.0 * math.pi * tone * index / rate))

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
            block.pts = None
            for packet in stream.encode(block):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def dominant_tone(path: Path) -> float:
    """Which of the marker tones is loudest in a clip.

    A single correlation against each candidate tone is enough here, because
    the tones are the only thing in the recording and they are an octave
    apart. It avoids needing a Fourier transform, and therefore numpy, which
    this project does not depend on.
    """
    import av

    samples: list[int] = []
    rate = MARKER_RATE
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        rate = int(stream.sample_rate)
        # Whatever the file holds is turned into plain 16-bit mono first, so
        # this only ever has to read one kind of buffer.
        resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
        for frame in list(container.decode(stream)) + [None]:
            for converted in resampler.resample(frame):
                plane = memoryview(converted.planes[0]).cast("h")
                samples.extend(plane[: converted.samples])

    best_tone = MARKER_TONES[0]
    best_energy = -1.0
    for tone in MARKER_TONES:
        real = sum(
            value * math.cos(2.0 * math.pi * tone * index / rate)
            for index, value in enumerate(samples)
        )
        imaginary = sum(
            value * math.sin(2.0 * math.pi * tone * index / rate)
            for index, value in enumerate(samples)
        )
        energy = real * real + imaginary * imaginary
        if energy > best_energy:
            best_energy = energy
            best_tone = tone
    return best_tone


# -- Probing --------------------------------------------------------------


def test_a_recording_is_measured_for_what_it_actually_is(tmp_path):
    source = tmp_path / "talk.wav"
    write_real_audio(source, seconds=2.0, codec="pcm_s16le", rate=16000, layout="mono")

    probe = probe_audio(source)

    assert probe.container == "wav"
    assert probe.sample_rate == 16000
    assert probe.channels == 1
    assert probe.duration == pytest.approx(2.0, abs=0.1)
    assert probe.size_bytes == source.stat().st_size
    assert probe.sample_count == pytest.approx(probe.duration * 16000, abs=1)


def test_a_compressed_recording_is_named_by_its_extension(tmp_path):
    """Services ask what kind of file it is, and the extension is the answer."""
    source = tmp_path / "talk.m4a"
    write_real_audio(source, seconds=1.0)

    assert probe_audio(source).container == "m4a"


def test_a_file_that_is_not_audio_is_refused_in_words(tmp_path):
    source = tmp_path / "broken.m4a"
    write_fake_audio(source)

    with pytest.raises(CanonicalAudioError) as raised:
        probe_audio(source)

    assert "broken.m4a" in str(raised.value)


# -- Deciding whether to copy ---------------------------------------------


def test_a_recording_every_service_accepts_is_used_as_it_is(tmp_path):
    source = tmp_path / "talk.wav"
    write_real_audio(source, seconds=1.0, codec="pcm_s16le")

    decision = decide_canonical_copy(probe_audio(source), {"wav", "flac"})

    assert not decision.needed
    assert "as it is" in decision.reason


def test_a_container_no_service_takes_forces_a_copy(tmp_path):
    source = tmp_path / "talk.m4a"
    write_real_audio(source, seconds=1.0)

    decision = decide_canonical_copy(probe_audio(source), {"wav", "flac"})

    assert decision.needed
    assert decision.output_format is WAV16
    assert "m4a" in decision.reason


def test_a_compressed_recording_over_the_limit_is_left_alone(tmp_path):
    """Turning a compressed recording lossless makes it bigger, not smaller.

    Size is a reason to send a service smaller requests, which chunking does.
    It is not a reason to convert the canonical file, and doing so would make
    the very problem it was meant to solve worse.
    """
    source = tmp_path / "talk.m4a"
    write_real_audio(source, seconds=1.0)

    decision = decide_canonical_copy(
        probe_audio(source), {"wav", "flac", "m4a"}, maximum_bytes=1
    )

    assert not decision.needed
    assert "in pieces" in decision.reason


def test_an_uncompressed_recording_over_the_limit_is_copied_as_flac(tmp_path):
    source = tmp_path / "talk.wav"
    write_real_audio(source, seconds=2.0, codec="pcm_s16le")

    decision = decide_canonical_copy(probe_audio(source), {"wav", "flac"}, maximum_bytes=1000)

    assert decision.needed
    assert decision.output_format is FLAC


def test_the_accepted_containers_are_what_every_service_has_in_common():
    together = containers_accepted_by_all(
        [
            ProviderCapabilities(accepted_containers=frozenset({"wav", "mp3", "flac", "m4a"})),
            ProviderCapabilities(accepted_containers=frozenset({"wav", "flac"})),
        ]
    )

    assert together == frozenset({"wav", "flac"})


# -- Preparing the canonical audio ----------------------------------------


def test_a_recording_that_needs_nothing_becomes_canonical_untouched(tmp_path):
    source = tmp_path / "talk.wav"
    write_real_audio(source, seconds=1.0, codec="pcm_s16le", rate=16000, layout="mono")
    before = source.read_bytes()

    canonical = prepare_canonical_audio(source, tmp_path / "work", {"wav", "flac"})

    assert canonical.path == str(source)
    assert canonical.original_path == str(source)
    assert not canonical.is_copy
    assert canonical.sample_rate == 16000
    assert canonical.channels == 1
    assert source.read_bytes() == before, "the original must never be written to"
    assert not (tmp_path / "work").exists(), "nothing should be written when nothing is needed"


def test_a_copy_is_written_beside_the_transcript_and_the_original_is_kept(tmp_path):
    source = tmp_path / "talk.m4a"
    write_real_audio(source, seconds=1.0, rate=16000, layout="mono")
    before = source.read_bytes()
    folder = tmp_path / "work"

    canonical = prepare_canonical_audio(source, folder, {"wav", "flac"})

    assert canonical.is_copy
    assert canonical.path == str(folder / "talk-canonical.wav")
    assert Path(canonical.path).is_file()
    assert canonical.original_path == str(source)
    assert source.read_bytes() == before, "the original must never be written to"


def test_the_canonical_numbers_are_measured_from_the_copy(tmp_path):
    """The copy is the file every timestamp is read against, so it is the
    file that must be measured.

    A conversion can change the length by a few milliseconds. Carrying the
    original's duration over would describe one file while the services
    listened to another, and every word would sit slightly out of place with
    nothing to show why.
    """
    source = tmp_path / "talk.m4a"
    write_real_audio(source, seconds=2.0, rate=16000, layout="mono")
    folder = tmp_path / "work"

    canonical = prepare_canonical_audio(source, folder, {"wav", "flac"})
    copy = probe_audio(Path(canonical.path))

    assert canonical.duration == copy.duration
    assert canonical.sample_rate == copy.sample_rate
    assert canonical.channels == copy.channels
    assert canonical.size_bytes == copy.size_bytes
    assert canonical.container == "wav"


def test_the_copy_holds_the_same_audio_as_the_original(tmp_path):
    source = tmp_path / "marker.wav"
    write_marker_audio(source, seconds=3)
    folder = tmp_path / "work"

    canonical = prepare_canonical_audio(source, folder, {"flac"}, preferred_format=FLAC)

    assert canonical.is_copy
    assert canonical.container == "flac"
    assert canonical.duration == pytest.approx(3.0, abs=0.01)
    assert dominant_tone(Path(canonical.path)) == MARKER_TONES[0]


# -- Cutting windows ------------------------------------------------------


def test_a_window_holds_the_second_of_audio_it_asked_for(tmp_path):
    source = tmp_path / "marker.wav"
    write_marker_audio(source)
    canonical = prepare_canonical_audio(source, tmp_path / "work", {"wav"})

    clip = cut_window(canonical, AudioSpan(3.0, 4.0), tmp_path / "clip.wav")

    assert clip.canonical_offset == pytest.approx(3.0)
    assert clip.span == AudioSpan(3.0, 4.0)
    assert clip.duration == pytest.approx(1.0)
    assert not clip.clamped
    assert dominant_tone(clip.path) == MARKER_TONES[3]


def test_every_second_of_the_recording_can_be_cut_out_exactly(tmp_path):
    """The offset has to be right at every position, not only convenient ones."""
    source = tmp_path / "marker.wav"
    write_marker_audio(source)
    canonical = prepare_canonical_audio(source, tmp_path / "work", {"wav"})

    for second, tone in enumerate(MARKER_TONES):
        clip = cut_window(
            canonical, AudioSpan(second + 0.1, second + 0.9), tmp_path / f"clip{second}.wav"
        )
        assert clip.canonical_offset == pytest.approx(second + 0.1)
        assert dominant_tone(clip.path) == tone, f"second {second} came out wrong"


def test_a_window_that_starts_before_the_recording_says_so(tmp_path):
    """The offset must describe what was cut, not what was asked for.

    Ask for two seconds beginning one second before the recording, and one
    second is all there is. Returning the requested offset of minus one would
    move every word the service reports back by a second.
    """
    source = tmp_path / "marker.wav"
    write_marker_audio(source)
    canonical = prepare_canonical_audio(source, tmp_path / "work", {"wav"})

    clip = cut_window(canonical, AudioSpan(-1.0, 1.0), tmp_path / "clip.wav")

    assert clip.canonical_offset == pytest.approx(0.0)
    assert clip.span == AudioSpan(0.0, 1.0)
    assert clip.clamped
    assert dominant_tone(clip.path) == MARKER_TONES[0]


def test_a_window_that_runs_past_the_end_stops_at_the_end(tmp_path):
    source = tmp_path / "marker.wav"
    write_marker_audio(source)
    canonical = prepare_canonical_audio(source, tmp_path / "work", {"wav"})

    clip = cut_window(canonical, AudioSpan(5.5, 9.0), tmp_path / "clip.wav")

    assert clip.canonical_offset == pytest.approx(5.5)
    assert clip.span.end == pytest.approx(canonical.duration)
    assert clip.duration == pytest.approx(0.5)
    assert clip.clamped
    assert dominant_tone(clip.path) == MARKER_TONES[5]


def test_a_window_asked_for_beyond_the_end_comes_back_empty_rather_than_wrong(tmp_path):
    source = tmp_path / "marker.wav"
    write_marker_audio(source)
    canonical = prepare_canonical_audio(source, tmp_path / "work", {"wav"})

    clip = cut_window(canonical, AudioSpan(20.0, 25.0), tmp_path / "clip.wav")

    assert clip.canonical_offset == pytest.approx(canonical.duration)
    assert clip.duration == pytest.approx(0.0)
    assert clip.clamped


def test_the_padded_span_of_a_word_near_the_start_still_cuts_correctly(tmp_path):
    """The escalation path end to end: pad a disputed word, then cut it.

    The padding is clamped by the model, the cut is clamped again here, and
    the two have to agree. A word half a second into the recording cannot
    have five seconds of context in front of it, and both steps have to come
    to the same conclusion about that rather than one of them pretending.
    """
    source = tmp_path / "marker.wav"
    write_marker_audio(source)
    canonical = prepare_canonical_audio(source, tmp_path / "work", {"wav"})
    disputed = AudioSpan(0.4, 0.6)

    window = disputed.padded(5.0, 5.0, limit=canonical.duration)
    clip = cut_window(canonical, window, tmp_path / "clip.wav")

    assert window.start == pytest.approx(0.0)
    assert clip.canonical_offset == pytest.approx(0.0)
    # A time reported inside the clip converts straight back to canonical
    # time, which is the whole reason the offset exists.
    assert clip.canonical_offset + 0.5 == pytest.approx(0.5)


def test_a_window_cut_from_a_compressed_recording_lands_where_it_should(tmp_path):
    """The canonical file is not always one this application wrote.

    Where every service accepts the recording as it stands, no copy is made
    and windows are cut from the user's own compressed file. Timing a cut in
    that is harder than in a WAV, because the decoder has to be seeked to a
    packet boundary and the samples counted from there, so it is worth
    proving separately.
    """
    source = tmp_path / "marker.wav"
    write_marker_audio(source)
    compressed = tmp_path / "marker.mp3"
    _write_compressed(source, compressed, "mp3")
    canonical = prepare_canonical_audio(compressed, tmp_path / "work", {"mp3", "wav"})

    assert not canonical.is_copy
    clip = cut_window(canonical, AudioSpan(4.2, 4.8), tmp_path / "clip.wav")

    assert clip.canonical_offset == pytest.approx(4.2)
    assert dominant_tone(clip.path) == MARKER_TONES[4]


def test_a_window_cut_from_an_m4a_recording_can_be_read_back(tmp_path):
    """The exact combination the user meets first.

    Their recordings are ``.m4a``, every service accepts ``.m4a``, so no
    canonical copy is made and escalation cuts its windows straight out of
    it. The clip must therefore come out in a format this application can
    certainly write and then read again, which is why windows are written as
    WAV or FLAC and never back into the codec they came from. Some builds of
    FFmpeg can decode AAC without being able to encode it, and a window that
    tried to stay AAC would fail on the user's own recordings.
    """
    source = tmp_path / "marker.wav"
    write_marker_audio(source)
    compressed = tmp_path / "marker.m4a"
    _write_compressed(source, compressed, "aac")
    canonical = prepare_canonical_audio(compressed, tmp_path / "work", {"m4a", "wav", "flac"})

    assert not canonical.is_copy
    assert canonical.container == "m4a"

    clip = cut_window(canonical, AudioSpan(2.1, 2.9), tmp_path / "clip.flac", output_format=FLAC)

    assert clip.path.suffix == ".flac"
    read_back = probe_audio(clip.path)
    assert read_back.container == "flac"
    assert read_back.duration == pytest.approx(0.8, abs=0.02)
    assert clip.canonical_offset == pytest.approx(2.1)
    assert dominant_tone(clip.path) == MARKER_TONES[2]


def test_a_window_can_be_written_at_a_lower_rate_without_moving(tmp_path):
    """Resampling must change the samples in the clip and nothing about where it is.

    The window is chosen in the recording's own samples and only then
    converted, so the offset, the span and the tone inside are exactly what
    an unconverted clip would have had.
    """
    source = tmp_path / "marker.wav"
    write_marker_audio(source, rate=48000)
    canonical = prepare_canonical_audio(source, tmp_path / "work", {"wav"})

    for second, tone in enumerate(MARKER_TONES):
        clip = cut_window(
            canonical,
            AudioSpan(second + 0.1, second + 0.9),
            tmp_path / f"clip{second}.flac",
            FLAC,
            sample_rate=16000,
            channels=1,
        )
        written = probe_audio(clip.path)
        assert clip.sample_rate == 16000 and clip.channels == 1
        assert written.sample_rate == 16000 and written.channels == 1
        assert written.duration == pytest.approx(0.8, abs=0.001)
        assert clip.canonical_offset == pytest.approx(second + 0.1)
        assert clip.span == AudioSpan(second + 0.1, second + 0.9)
        assert dominant_tone(clip.path) == tone, f"second {second} came out wrong"


def test_a_window_is_folded_to_mono_when_asked(tmp_path):
    source = tmp_path / "stereo.wav"
    write_real_audio(source, seconds=2.0, codec="pcm_s16le", layout="stereo", rate=48000)
    canonical = prepare_canonical_audio(source, tmp_path / "work", {"wav"})

    clip = cut_window(canonical, AudioSpan(0.5, 1.5), tmp_path / "clip.wav", channels=1)

    written = probe_audio(clip.path)
    assert written.channels == 1
    assert written.sample_rate == 48000, "the rate was not asked to change"
    assert written.duration == pytest.approx(1.0, abs=0.001)


def test_a_window_is_never_upsampled_or_given_channels(tmp_path):
    source = tmp_path / "marker.wav"
    write_marker_audio(source)
    canonical = prepare_canonical_audio(source, tmp_path / "work", {"wav"})

    clip = cut_window(
        canonical, AudioSpan(1.0, 2.0), tmp_path / "clip.wav", sample_rate=48000, channels=2
    )

    written = probe_audio(clip.path)
    assert (clip.sample_rate, clip.channels) == (16000, 1)
    assert (written.sample_rate, written.channels) == (16000, 1)
    assert dominant_tone(clip.path) == MARKER_TONES[1]


def test_a_window_goes_to_a_temporary_file_when_no_home_is_given(tmp_path):
    source = tmp_path / "marker.wav"
    write_marker_audio(source, seconds=2)
    canonical = prepare_canonical_audio(source, tmp_path / "work", {"wav"})

    clip = cut_window(canonical, AudioSpan(1.0, 2.0))
    try:
        assert clip.path.is_file()
        assert clip.size_bytes > 0
        assert dominant_tone(clip.path) == MARKER_TONES[1]
    finally:
        clip.path.unlink(missing_ok=True)


# -- Estimating size ------------------------------------------------------


def test_the_estimated_size_of_uncompressed_audio_is_the_arithmetic(tmp_path):
    estimate = estimate_lossless_bytes(60.0, 16000, 1, WAV16)

    # A minute of 16 kHz mono at two bytes a sample, plus the header.
    assert estimate == pytest.approx(60 * 16000 * 2, rel=0.01)


def test_flac_is_estimated_pessimistically_so_a_plan_stays_inside_a_limit():
    """An underestimate costs a rejected request; an overestimate costs nothing."""
    pcm = estimate_lossless_bytes(60.0, 16000, 1, WAV16)
    flac = estimate_lossless_bytes(60.0, 16000, 1, FLAC)

    assert flac < pcm
    assert flac > pcm * 0.5


def _write_compressed(source: Path, destination: Path, codec: str) -> None:
    """Write a recording out again in a compressed codec, for the tests that need one.

    Which codecs FFmpeg can *encode* is a property of how it was built, and
    the copy that arrives inside PyAV is not the same everywhere. Some builds
    decode AAC perfectly well and cannot write it. A test that needs an
    encoder the machine has not got says so and steps aside, rather than
    failing and being mistaken for a fault in the application.
    """
    import av

    try:
        with av.open(str(source)) as reader, av.open(str(destination), "w") as writer:
            stream = reader.streams.audio[0]
            # A plain WAV does not record its channel arrangement, so the
            # name has to be supplied rather than copied from the source.
            layout = "mono" if int(stream.channels) == 1 else "stereo"
            output = writer.add_stream(codec, rate=stream.sample_rate, layout=layout)
            for frame in reader.decode(stream):
                frame.pts = None
                for packet in output.encode(frame):
                    writer.mux(packet)
            for packet in output.encode(None):
                writer.mux(packet)
    except av.FFmpegError as error:
        pytest.skip(f"The FFmpeg inside PyAV on this machine cannot encode {codec}: {error}")
