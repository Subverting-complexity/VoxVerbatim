"""Tests for the audio enhancement engine.

The gain calculation is tested on its own with numbers, because that is
where the judgement lives and it is exact. The rest is tested by enhancing
real recordings and measuring what came out, because the point of the
whole exercise is a file on disk at the right loudness, and nothing short
of measuring it proves that.
"""

from __future__ import annotations

import math

import pytest

from vox_verbatim.audio.enhance import (
    OUTPUT_FORMATS,
    EnhanceOptions,
    GainLimit,
    Outcome,
    enhance_file,
    enhance_files,
    output_format_for,
    plan_gain,
)

from tests.conftest import write_fake_audio, write_real_audio

WAV16, WAV24, FLAC = OUTPUT_FORMATS

#: A stereo tone measures 0.69 LUFS below its amplitude in dBFS, which
#: falls out of the loudness formula. Half a decibel of slack on top covers
#: what the lossy source codec does to it.
_TOLERANCE_DB = 1.2


@pytest.fixture
def options(tmp_path):
    return EnhanceOptions(output_folder=tmp_path / "enhanced")


# -- Working out the gain -----------------------------------------------


def test_a_quiet_recording_is_raised_to_the_target(options):
    """Nothing is in the way, so the recording lands exactly on the target."""
    plan = plan_gain(measured_lufs=-29.4, measured_dbtp=-20.0, options=options)

    assert plan.gain_db == pytest.approx(11.4)
    assert plan.limited_by == GainLimit.NONE


def test_the_last_decibel_of_headroom_is_left_alone(options):
    """A recording that would just clear the ceiling stops short of it.

    Going from -29.4 LUFS to the -18 target asks for 11.4 dB, which would
    put a peak of -11.7 dBTP at -0.3 dBTP, over the ceiling. It is held to
    10.7 dB instead, which lands the peak exactly on -1 dBTP and the
    loudness a little under target at -18.7 LUFS.
    """
    plan = plan_gain(measured_lufs=-29.4, measured_dbtp=-11.7, options=options)

    assert plan.gain_db == pytest.approx(10.7)
    assert plan.limited_by == GainLimit.CEILING


def test_the_peak_ceiling_holds_the_gain_back_and_says_so(options):
    """A recording with loud transients cannot be raised as far as it wants."""
    plan = plan_gain(measured_lufs=-40.0, measured_dbtp=-6.0, options=options)

    # It wants 22 dB, but 5 dB is all that fits under a ceiling of -1 dBTP.
    assert plan.gain_db == pytest.approx(5.0)
    assert plan.limited_by == GainLimit.CEILING


def test_the_maximum_gain_holds_a_very_quiet_recording_back(options):
    plan = plan_gain(measured_lufs=-60.0, measured_dbtp=-58.0, options=options)

    assert plan.gain_db == pytest.approx(options.maximum_gain_db)
    assert plan.limited_by == GainLimit.MAXIMUM


def test_the_limiter_lets_the_target_be_reached_despite_the_peaks(tmp_path):
    options = EnhanceOptions(output_folder=tmp_path / "enhanced", use_limiter=True)

    plan = plan_gain(measured_lufs=-40.0, measured_dbtp=-6.0, options=options)

    assert plan.gain_db == pytest.approx(22.0)
    assert plan.limited_by == GainLimit.NONE


def test_a_recording_louder_than_the_target_is_turned_down(options):
    plan = plan_gain(measured_lufs=-9.0, measured_dbtp=-3.0, options=options)

    assert plan.gain_db == pytest.approx(-9.0)
    assert plan.limited_by == GainLimit.NONE


def test_peaks_already_over_the_ceiling_are_pulled_under_it(options):
    """Protecting against clipping is not subject to the maximum gain.

    The maximum gain is a limit on how far a recording is dragged upwards.
    Turning one down is always allowed, however far it has to go, because
    the alternative is a file that distorts.
    """
    plan = plan_gain(measured_lufs=-18.0, measured_dbtp=0.5, options=options)

    assert plan.gain_db == pytest.approx(-1.5)
    assert plan.limited_by == GainLimit.CEILING


def test_an_unknown_output_format_falls_back_to_the_default():
    assert output_format_for("something-else") is WAV16
    assert output_format_for(None) is WAV16
    assert output_format_for("flac") is FLAC


# -- Enhancing real recordings -------------------------------------------


def test_a_quiet_recording_comes_out_at_the_target_loudness(tmp_path, options):
    source = tmp_path / "quiet.m4a"
    write_real_audio(source, level_db=-32.0)

    result = enhance_file(source, options)

    assert result.outcome == Outcome.ENHANCED
    assert result.output == options.output_folder / "quiet.wav"
    assert result.output.is_file()
    # The loudness is measured from the finished file, not predicted, so
    # this is a reading of what was actually written.
    assert result.output_lufs == pytest.approx(options.target_lufs, abs=_TOLERANCE_DB)
    assert result.output_dbtp <= options.ceiling_dbtp
    assert result.gain_db > 0


def test_the_true_peak_ceiling_is_respected_in_the_file_that_is_written(tmp_path):
    """A recording already close to full scale must not be pushed over it."""
    source = tmp_path / "loud.m4a"
    write_real_audio(source, level_db=-2.0)
    options = EnhanceOptions(output_folder=tmp_path / "enhanced", target_lufs=-6.0)

    result = enhance_file(source, options)

    assert result.outcome == Outcome.ENHANCED
    assert result.output_dbtp <= options.ceiling_dbtp


def test_the_report_says_what_was_measured_and_what_was_done(tmp_path, options):
    source = tmp_path / "quiet.m4a"
    write_real_audio(source, level_db=-32.0)

    result = enhance_file(source, options)

    assert "quiet.m4a" in result.message
    assert "LUFS" in result.message and "dBTP" in result.message
    assert "quiet.wav" in result.message


def test_each_output_format_is_written_with_its_own_extension(tmp_path):
    source = tmp_path / "quiet.m4a"
    write_real_audio(source, level_db=-30.0)

    for output_format, extension in ((WAV16, ".wav"), (WAV24, ".wav"), (FLAC, ".flac")):
        folder = tmp_path / output_format.key
        result = enhance_file(
            source, EnhanceOptions(output_folder=folder, output_format=output_format)
        )
        assert result.outcome == Outcome.ENHANCED, result.message
        assert result.output == folder / f"quiet{extension}"
        assert result.output.stat().st_size > 0


def test_the_limiter_keeps_the_peaks_under_the_ceiling_too(tmp_path):
    source = tmp_path / "quiet.m4a"
    write_real_audio(source, level_db=-30.0)
    options = EnhanceOptions(
        output_folder=tmp_path / "enhanced", target_lufs=-9.0, use_limiter=True
    )

    result = enhance_file(source, options)

    assert result.outcome == Outcome.ENHANCED
    assert result.output_dbtp <= options.ceiling_dbtp


def test_a_silent_recording_is_left_alone(tmp_path, options):
    source = tmp_path / "silence.m4a"
    write_real_audio(source, level_db=-200.0)

    result = enhance_file(source, options)

    assert result.outcome == Outcome.SKIPPED
    assert "silent" in result.message
    assert not (options.output_folder / "silence.wav").exists()


def test_a_file_that_is_not_really_audio_is_reported_rather_than_raised(tmp_path, options):
    source = tmp_path / "broken.m4a"
    write_fake_audio(source)

    result = enhance_file(source, options)

    assert result.outcome == Outcome.FAILED
    assert "broken.m4a" in result.message


def test_an_existing_file_is_kept_unless_replacing_is_asked_for(tmp_path):
    source = tmp_path / "quiet.m4a"
    write_real_audio(source, level_db=-30.0)
    folder = tmp_path / "enhanced"
    options = EnhanceOptions(output_folder=folder)

    assert enhance_file(source, options).outcome == Outcome.ENHANCED
    first_written = (folder / "quiet.wav").stat().st_mtime_ns

    again = enhance_file(source, options)
    assert again.outcome == Outcome.SKIPPED
    assert "already in the output folder" in again.message
    assert (folder / "quiet.wav").stat().st_mtime_ns == first_written

    replaced = enhance_file(source, EnhanceOptions(output_folder=folder, replace_existing=True))
    assert replaced.outcome == Outcome.ENHANCED


def test_a_run_that_would_overwrite_the_recording_itself_is_refused(tmp_path):
    """Writing WAV output into the folder a WAV recording came from."""
    source = tmp_path / "talk.wav"
    write_real_audio(source, level_db=-30.0, codec="pcm_s16le")
    options = EnhanceOptions(output_folder=tmp_path, replace_existing=True)

    result = enhance_file(source, options)

    assert result.outcome == Outcome.SKIPPED
    assert "replace the recording itself" in result.message


def test_another_recording_with_the_output_name_is_never_written_over(tmp_path):
    """The output folder is the recordings folder, and replacing is on.

    meeting.m4a would be written as meeting.wav, which is a different
    recording that happens to share the name.
    """
    source = tmp_path / "meeting.m4a"
    write_real_audio(source, level_db=-30.0)
    other_recording = tmp_path / "meeting.wav"
    write_real_audio(other_recording, level_db=-30.0, codec="pcm_s16le")
    original_bytes = other_recording.read_bytes()
    options = EnhanceOptions(output_folder=tmp_path, replace_existing=True)

    result = enhance_file(source, options)

    assert result.outcome == Outcome.SKIPPED
    assert "meeting.m4a" in result.message
    assert "meeting.wav is a recording in the same folder" in result.message
    assert other_recording.read_bytes() == original_bytes


def test_sources_that_share_an_output_name_are_all_skipped(tmp_path, options):
    """talk.m4a and talk.mp3 would both become talk.wav, the second over the first."""
    first = tmp_path / "talk.m4a"
    second = tmp_path / "talk.mp3"
    unrelated = tmp_path / "quiet.m4a"
    # Neither clashing file is ever decoded, so they need not hold audio.
    write_fake_audio(first)
    write_fake_audio(second)
    write_real_audio(unrelated, level_db=-30.0)
    reported: list = []
    seen: list[float] = []

    results = enhance_files(
        [first, unrelated, second],
        EnhanceOptions(output_folder=options.output_folder, replace_existing=True),
        progress=seen.append,
        on_result=reported.append,
    )

    by_source = {result.source: result for result in results}
    assert by_source[first].outcome == Outcome.SKIPPED
    assert by_source[second].outcome == Outcome.SKIPPED
    assert "talk.mp3" in by_source[first].message
    assert "talk.m4a" in by_source[second].message
    assert "talk.wav" in by_source[first].message
    assert not (options.output_folder / "talk.wav").exists()
    # The clashes are reported before anything is written.
    assert [result.source for result in reported[:2]] == [first, second]
    assert by_source[unrelated].outcome == Outcome.ENHANCED
    assert (options.output_folder / "quiet.wav").is_file()
    assert seen == sorted(seen), "progress must never go backwards"
    assert seen[-1] == pytest.approx(1.0)


def test_the_output_folder_is_created_when_it_is_not_there(tmp_path):
    source = tmp_path / "quiet.m4a"
    write_real_audio(source, level_db=-30.0)
    folder = tmp_path / "not" / "there" / "yet"

    result = enhance_file(source, EnhanceOptions(output_folder=folder))

    assert result.outcome == Outcome.ENHANCED
    assert folder.is_dir()


def test_one_bad_file_does_not_stop_the_others(tmp_path, options):
    good_first = tmp_path / "a-good.m4a"
    bad = tmp_path / "b-bad.m4a"
    good_last = tmp_path / "c-good.m4a"
    write_real_audio(good_first, level_db=-30.0)
    write_fake_audio(bad)
    write_real_audio(good_last, level_db=-30.0)

    results = enhance_files([good_first, bad, good_last], options)

    assert [result.outcome for result in results] == [
        Outcome.ENHANCED,
        Outcome.FAILED,
        Outcome.ENHANCED,
    ]


def test_progress_runs_from_nothing_to_everything_across_the_files(tmp_path, options):
    first = tmp_path / "one.m4a"
    second = tmp_path / "two.m4a"
    write_real_audio(first, level_db=-30.0)
    write_real_audio(second, level_db=-30.0)
    seen: list[float] = []

    enhance_files([first, second], options, progress=seen.append)

    assert seen
    assert all(0.0 <= fraction <= 1.0 for fraction in seen)
    assert seen == sorted(seen), "progress must never go backwards"
    assert seen[-1] == pytest.approx(1.0)


def test_cancelling_stops_the_run_and_leaves_no_half_written_file(tmp_path, options):
    source = tmp_path / "quiet.m4a"
    write_real_audio(source, level_db=-30.0, seconds=4.0)

    result = enhance_file(source, options, cancelled=lambda: True)

    assert result.outcome == Outcome.CANCELLED
    assert not (options.output_folder / "quiet.wav").exists()


def test_cancelling_a_replace_part_way_through_keeps_the_previous_copy(tmp_path):
    """The run is stopped while the new copy is being written, not before."""
    source = tmp_path / "quiet.m4a"
    write_real_audio(source, level_db=-30.0, seconds=4.0)
    folder = tmp_path / "enhanced"
    folder.mkdir()
    previous = folder / "quiet.wav"
    previous.write_bytes(b"the previous enhanced copy")
    options = EnhanceOptions(output_folder=folder, replace_existing=True)
    writing: list[float] = []

    def note(fraction: float) -> None:
        # Past halfway is the second pass, which writes the file.
        if 0.5 < fraction < 1.0:
            writing.append(fraction)

    result = enhance_file(source, options, progress=note, cancelled=lambda: bool(writing))

    assert writing, "the run must be cancelled while the file is being written"
    assert result.outcome == Outcome.CANCELLED
    assert previous.read_bytes() == b"the previous enhanced copy"
    assert sorted(path.name for path in folder.iterdir()) == ["quiet.wav"]


def test_a_failed_replace_keeps_the_previous_copy(tmp_path, monkeypatch):
    source = tmp_path / "quiet.m4a"
    write_real_audio(source, level_db=-30.0)
    folder = tmp_path / "enhanced"
    folder.mkdir()
    previous = folder / "quiet.wav"
    previous.write_bytes(b"the previous enhanced copy")
    options = EnhanceOptions(output_folder=folder, replace_existing=True)

    def refuse(_source, _destination):
        raise PermissionError("The file is open in another program")

    monkeypatch.setattr("vox_verbatim.audio.enhance.os.replace", refuse)
    result = enhance_file(source, options)

    assert result.outcome == Outcome.FAILED
    assert previous.read_bytes() == b"the previous enhanced copy"
    assert sorted(path.name for path in folder.iterdir()) == ["quiet.wav"]


def test_a_completed_replace_takes_the_place_of_the_previous_copy(tmp_path):
    source = tmp_path / "quiet.m4a"
    write_real_audio(source, level_db=-30.0)
    folder = tmp_path / "enhanced"
    folder.mkdir()
    previous = folder / "quiet.wav"
    previous.write_bytes(b"the previous enhanced copy")
    options = EnhanceOptions(output_folder=folder, replace_existing=True)

    result = enhance_file(source, options)

    assert result.outcome == Outcome.ENHANCED, result.message
    assert result.output == previous
    assert previous.read_bytes() != b"the previous enhanced copy"
    assert previous.read_bytes()[:4] == b"RIFF"
    assert sorted(path.name for path in folder.iterdir()) == ["quiet.wav"]


def test_a_mono_recording_keeps_its_single_channel(tmp_path, options):
    """The channel arrangement is not something this is allowed to change."""
    import av

    source = tmp_path / "mono.m4a"
    write_real_audio(source, level_db=-30.0, layout="mono")

    result = enhance_file(source, options)

    assert result.outcome == Outcome.ENHANCED
    with av.open(str(result.output)) as container:
        stream = container.streams.audio[0]
        assert stream.channels == 1
        assert stream.sample_rate == 48000


def test_the_measurement_of_a_quiet_tone_is_the_level_it_was_written_at(tmp_path, options):
    """A check that the meter is reading the recording, not something else.

    A steady stereo tone measures 0.69 LUFS below its own amplitude, which
    is arithmetic rather than opinion, so a reading far from that would
    mean the wrong thing is being measured.
    """
    source = tmp_path / "quiet.m4a"
    write_real_audio(source, level_db=-30.0)

    result = enhance_file(source, options)

    assert result.input_lufs == pytest.approx(-30.69, abs=_TOLERANCE_DB)
    assert math.isfinite(result.input_dbtp)
    assert result.input_dbtp == pytest.approx(-30.0, abs=_TOLERANCE_DB)
