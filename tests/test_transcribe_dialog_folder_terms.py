"""Tests for the Transcribe dialog sending a folder's learned names.

The names a folder's reviews taught live in that folder's project file. The
dialog reads them when a run starts and hands them to the pipeline, and asks
first when the file is there but cannot be read. The pipeline is replaced in
every test that starts a run, so nothing here reaches a service.
"""

from __future__ import annotations

import json

import pytest

from vox_verbatim import settings as settings_module
from vox_verbatim.audio.library import AudioFile
from vox_verbatim.transcription import pipeline
from vox_verbatim.transcription.model import Transcript
from vox_verbatim.transcription.project import PROJECT_FILE_NAME
from vox_verbatim.ui.transcribe_dialog import TranscribeDialog

from tests.conftest import wait_until
from tests.test_transcribe_dialog import configured_settings


@pytest.fixture(autouse=True)
def every_library_loads(monkeypatch) -> None:
    monkeypatch.setattr(settings_module, "_probe_library", lambda module, attribute: None)


@pytest.fixture(autouse=True)
def nothing_is_confirmed_by_hand(monkeypatch) -> None:
    monkeypatch.setattr(TranscribeDialog, "confirm_cost", lambda self, message: True)
    monkeypatch.setattr(TranscribeDialog, "_show_completion_message", lambda self, message: None)


def recordings_in(folder) -> list[AudioFile]:
    folder.mkdir(parents=True, exist_ok=True)
    return [AudioFile(path=folder / "alpha.m4a", size_bytes=1000, duration_seconds=60.0)]


def write_learned_names(folder, *names: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    data = {
        "learned_names": [
            {"text": name, "language": "unknown", "wrong_forms": []} for name in names
        ]
    }
    (folder / PROJECT_FILE_NAME).write_text(json.dumps(data), encoding="utf-8")


def record_options(monkeypatch) -> list:
    """Replace the pipeline and keep the options each recording was given."""
    seen: list = []

    def transcribe(recording, options, *args, **kwargs):
        seen.append(options)
        return Transcript(recording_name=recording.name)

    monkeypatch.setattr(pipeline, "transcribe_recording", transcribe)
    return seen


def run_and_wait(qapp, dialog: TranscribeDialog) -> None:
    assert dialog.start() is True
    assert wait_until(qapp, lambda: dialog.summary is not None)


def folder_texts(options) -> list[str]:
    return [term.text for term in options.folder_terms]


def test_a_folder_that_learned_bosch_sends_bosch(qapp, monkeypatch, tmp_path):
    folder = tmp_path / "client"
    write_learned_names(folder, "Bosch")
    seen = record_options(monkeypatch)
    monkeypatch.setattr(
        TranscribeDialog,
        "confirm_without_learned_names",
        lambda self: pytest.fail("A readable file is not asked about."),
    )

    dialog = TranscribeDialog(recordings_in(folder), configured_settings())
    try:
        assert folder_texts(dialog.chosen_options()) == ["Bosch"]
        run_and_wait(qapp, dialog)
    finally:
        dialog.close()

    assert [folder_texts(options) for options in seen] == [["Bosch"]]


def test_a_different_folder_is_not_sent_bosch(qapp, monkeypatch, tmp_path):
    write_learned_names(tmp_path / "client", "Bosch")
    other = tmp_path / "other"
    write_learned_names(other, "Smit")
    seen = record_options(monkeypatch)

    dialog = TranscribeDialog(recordings_in(other), configured_settings())
    try:
        run_and_wait(qapp, dialog)
    finally:
        dialog.close()

    assert [folder_texts(options) for options in seen] == [["Smit"]]


def test_a_name_learned_after_the_dialog_opened_is_still_sent(qapp, monkeypatch, tmp_path):
    folder = tmp_path / "client"
    seen = record_options(monkeypatch)

    dialog = TranscribeDialog(recordings_in(folder), configured_settings())
    try:
        write_learned_names(folder, "Bosch")
        run_and_wait(qapp, dialog)
    finally:
        dialog.close()

    assert [folder_texts(options) for options in seen] == [["Bosch"]]


def test_a_folder_with_no_project_file_runs_without_asking(qapp, monkeypatch, tmp_path):
    folder = tmp_path / "client"
    seen = record_options(monkeypatch)
    monkeypatch.setattr(
        TranscribeDialog,
        "confirm_without_learned_names",
        lambda self: pytest.fail("A missing file is not asked about."),
    )

    dialog = TranscribeDialog(recordings_in(folder), configured_settings())
    try:
        run_and_wait(qapp, dialog)
    finally:
        dialog.close()

    assert [options.folder_terms for options in seen] == [()]


def test_a_damaged_file_asks_and_cancelling_starts_nothing(qapp, monkeypatch, tmp_path):
    folder = tmp_path / "client"
    recordings = recordings_in(folder)
    (folder / PROJECT_FILE_NAME).write_text("{ this is not json", encoding="utf-8")
    seen = record_options(monkeypatch)
    asked: list[bool] = []

    def decline(self) -> bool:
        asked.append(True)
        return False

    monkeypatch.setattr(TranscribeDialog, "confirm_without_learned_names", decline)

    dialog = TranscribeDialog(recordings, configured_settings())
    try:
        assert dialog.start() is False
        assert not dialog.is_running
    finally:
        dialog.close()

    assert asked == [True]
    assert seen == []


def test_a_damaged_file_asks_and_going_on_runs_without_names(qapp, monkeypatch, tmp_path):
    folder = tmp_path / "client"
    recordings = recordings_in(folder)
    (folder / PROJECT_FILE_NAME).write_text("{ this is not json", encoding="utf-8")
    seen = record_options(monkeypatch)
    monkeypatch.setattr(TranscribeDialog, "confirm_without_learned_names", lambda self: True)

    dialog = TranscribeDialog(recordings, configured_settings())
    try:
        run_and_wait(qapp, dialog)
    finally:
        dialog.close()

    assert [options.folder_terms for options in seen] == [()]


def test_folder_names_count_as_terms_for_the_cost_estimate(qapp, tmp_path):
    folder = tmp_path / "client"
    write_learned_names(folder, "Bosch")

    dialog = TranscribeDialog(recordings_in(folder), configured_settings())
    try:
        assert dialog.vocabulary_terms_will_be_sent() is True
    finally:
        dialog.close()
