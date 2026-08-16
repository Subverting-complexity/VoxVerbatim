"""Tests for finding audio files and reading their metadata."""

from __future__ import annotations

import struct
import wave

from audio_transcriber.audio.library import (
    DurationState,
    is_supported_audio_file,
    list_audio_files,
    read_duration,
)

from tests.conftest import write_fake_audio


def test_only_supported_extensions_are_listed(tmp_path):
    write_fake_audio(tmp_path / "interview.m4a")
    write_fake_audio(tmp_path / "notes.txt")
    write_fake_audio(tmp_path / "meeting.MP3")
    (tmp_path / "archive").mkdir()
    write_fake_audio(tmp_path / "archive" / "hidden.m4a")

    names = [audio_file.name for audio_file in list_audio_files(tmp_path)]

    assert names == ["interview.m4a", "meeting.MP3"]


def test_files_are_sorted_ignoring_case(tmp_path):
    for name in ("beta.m4a", "Alpha.m4a", "charlie.m4a"):
        write_fake_audio(tmp_path / name)

    names = [audio_file.name for audio_file in list_audio_files(tmp_path)]

    assert names == ["Alpha.m4a", "beta.m4a", "charlie.m4a"]


def test_size_is_read_and_duration_starts_out_pending(tmp_path):
    write_fake_audio(tmp_path / "recording.m4a", size_bytes=4096)

    audio_file = list_audio_files(tmp_path)[0]

    assert audio_file.size_bytes == 4096
    assert audio_file.duration_state == DurationState.PENDING
    assert audio_file.duration_seconds is None
    assert audio_file.key == "recording.m4a"


def test_setting_a_duration_records_whether_it_is_known(tmp_path):
    write_fake_audio(tmp_path / "recording.m4a")
    audio_file = list_audio_files(tmp_path)[0]

    audio_file.set_duration(12.5)
    assert audio_file.duration_state == DurationState.KNOWN
    assert audio_file.duration_seconds == 12.5

    audio_file.set_duration(None)
    assert audio_file.duration_state == DurationState.UNAVAILABLE
    assert audio_file.duration_seconds is None


def test_the_duration_of_a_real_recording_is_read_from_the_file(tmp_path):
    """Read a genuine audio file, so the metadata library is really exercised."""
    path = tmp_path / "tone.wav"
    rate = 8000
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<h", 0) * (rate * 3))

    assert read_duration(path) == 3.0


def test_a_file_that_is_not_really_audio_has_no_duration(tmp_path):
    path = tmp_path / "broken.m4a"
    write_fake_audio(path)

    assert read_duration(path) is None


def test_a_missing_file_has_no_duration(tmp_path):
    assert read_duration(tmp_path / "gone.m4a") is None


def test_extension_check_ignores_case():
    assert is_supported_audio_file("Recording.M4A")
    assert is_supported_audio_file("song.mp3")
    assert not is_supported_audio_file("notes.docx")
