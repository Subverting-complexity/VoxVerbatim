"""Tests that the window holds the session together across restarts."""

from __future__ import annotations

import pytest

from audio_transcriber.session import SessionStore
from audio_transcriber.ui.main_window import MainWindow

from tests.conftest import wait_until, write_fake_audio


@pytest.fixture
def audio_folder(tmp_path):
    folder = tmp_path / "recordings"
    folder.mkdir()
    for name in ("alpha.m4a", "beta.m4a", "gamma.m4a"):
        write_fake_audio(folder / name)
    return folder


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "session.json")


def open_window(qapp, store, expected_files: int | None = None) -> MainWindow:
    window = MainWindow(store)
    if expected_files is not None:
        assert wait_until(qapp, lambda: window._model.rowCount() == expected_files)
    return window


def close_window(window: MainWindow) -> None:
    window.close()
    window.deleteLater()


def test_choosing_a_folder_lists_its_audio_files(qapp, store, audio_folder):
    window = open_window(qapp, store)
    try:
        window._folder_panel.folderChosen.emit(str(audio_folder))

        assert wait_until(qapp, lambda: window._model.rowCount() == 3)
        assert [f.name for f in window._model.files()] == [
            "alpha.m4a",
            "beta.m4a",
            "gamma.m4a",
        ]
        # The first file is highlighted so playback has something to work with.
        assert window._table.selected_row() == 0
    finally:
        close_window(window)


def test_the_folder_checked_files_and_highlight_survive_a_restart(qapp, store, audio_folder):
    window = open_window(qapp, store)
    try:
        window._folder_panel.folderChosen.emit(str(audio_folder))
        assert wait_until(qapp, lambda: window._model.rowCount() == 3)
        window._model.set_checked(0, True)
        window._model.set_checked(2, True)
        window._table.select_row(1)
    finally:
        close_window(window)

    restored = open_window(qapp, store, expected_files=3)
    try:
        assert restored._folder == audio_folder
        assert restored._model.checked_names() == ["alpha.m4a", "gamma.m4a"]
        assert restored._table.selected_row() == 1
    finally:
        close_window(restored)


def test_files_deleted_between_runs_are_forgotten(qapp, store, audio_folder):
    window = open_window(qapp, store)
    try:
        window._folder_panel.folderChosen.emit(str(audio_folder))
        assert wait_until(qapp, lambda: window._model.rowCount() == 3)
        window._model.set_checked_names(["alpha.m4a", "beta.m4a"])
        window._table.select_row(1)
    finally:
        close_window(window)

    (audio_folder / "beta.m4a").unlink()

    restored = open_window(qapp, store, expected_files=2)
    try:
        assert restored._model.checked_names() == ["alpha.m4a"]
        # The highlighted file is gone, so the first remaining file takes over.
        assert restored._table.selected_row() == 0
    finally:
        close_window(restored)


def test_a_folder_that_has_gone_away_is_reported_and_does_not_stop_the_window(
    qapp, store, audio_folder
):
    window = open_window(qapp, store)
    try:
        window._folder_panel.folderChosen.emit(str(audio_folder))
        assert wait_until(qapp, lambda: window._model.rowCount() == 3)
    finally:
        close_window(window)

    for child in audio_folder.iterdir():
        child.unlink()
    audio_folder.rmdir()

    restored = MainWindow(store)
    try:
        assert "not available" in restored._status_label.text()
        assert restored._model.rowCount() == 0
    finally:
        close_window(restored)


def test_an_empty_folder_says_so(qapp, store, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    window = open_window(qapp, store)
    try:
        window._folder_panel.folderChosen.emit(str(empty))

        assert wait_until(qapp, lambda: "No audio files were found" in window._status_label.text())
        assert window._model.rowCount() == 0
        assert window._table.selected_row() == -1
    finally:
        close_window(window)


def test_the_summary_counts_files_and_checked_files(qapp, store, audio_folder):
    window = open_window(qapp, store)
    try:
        window._folder_panel.folderChosen.emit(str(audio_folder))
        assert wait_until(qapp, lambda: window._model.rowCount() == 3)

        assert window._summary_label.text() == "3 files, 0 checked"

        window._model.set_checked(0, True)

        assert window._summary_label.text() == "3 files, 1 checked"
    finally:
        close_window(window)
