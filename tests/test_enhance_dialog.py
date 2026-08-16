"""Tests for the Enhance Audio dialog, the runner behind it, and the window.

The runs here use real recordings, so what the dialog reports is what the
enhancement engine actually did. They are kept short, because the point is
the behaviour of the dialog rather than the audio, which
:mod:`tests.test_enhance` covers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QDialog

from audio_transcriber.audio.enhance import EnhanceOptions, Outcome
from audio_transcriber.audio.enhance_runner import EnhanceRunner
from audio_transcriber.session import SessionStore
from audio_transcriber.settings import SETTINGS_FILE_NAME, EnhanceSettings, SettingsStore
from audio_transcriber.ui.enhance_dialog import EnhanceAudioDialog
from audio_transcriber.ui.main_window import MainWindow

from tests.conftest import wait_until, write_real_audio


@pytest.fixture
def recordings(tmp_path):
    """Two short, quiet recordings to work on."""
    folder = tmp_path / "recordings"
    folder.mkdir()
    paths = []
    for name in ("alpha.m4a", "beta.m4a"):
        path = folder / name
        write_real_audio(path, level_db=-30.0, seconds=1.0)
        paths.append(path)
    return paths


def run_and_wait(qapp, dialog, timeout_seconds: float = 60.0) -> None:
    """Start the dialog's run and let the event loop carry it to the end."""
    assert dialog.start()
    assert wait_until(qapp, lambda: dialog.summary is not None, timeout_seconds)


def silence_completion_message(monkeypatch) -> list[str]:
    """Catch the completion message box instead of letting it block the test."""
    shown: list[str] = []
    monkeypatch.setattr(
        EnhanceAudioDialog,
        "_show_completion_message",
        lambda self, message: shown.append(message),
    )
    return shown


# -- The runner ----------------------------------------------------------


def test_the_runner_reports_each_file_and_finishes(qapp, tmp_path, recordings):
    runner = EnhanceRunner()
    started: list[tuple[str, int, int]] = []
    finished: list = []
    summaries: list = []
    runner.fileStarted.connect(lambda name, number, total: started.append((name, number, total)))
    runner.fileFinished.connect(finished.append)
    runner.runFinished.connect(summaries.append)

    runner.start(recordings, EnhanceOptions(output_folder=tmp_path / "enhanced"))
    assert wait_until(qapp, lambda: bool(summaries), 60.0)

    assert started == [("alpha.m4a", 1, 2), ("beta.m4a", 2, 2)]
    assert [result.outcome for result in finished] == [Outcome.ENHANCED, Outcome.ENHANCED]
    assert summaries[0].enhanced == 2
    assert summaries[0].cancelled is False


def test_a_second_run_cannot_start_on_top_of_the_first(qapp, tmp_path, recordings):
    runner = EnhanceRunner()
    options = EnhanceOptions(output_folder=tmp_path / "enhanced")
    summaries: list = []
    runner.runFinished.connect(summaries.append)

    assert runner.start(recordings, options) is True
    assert runner.start(recordings, options) is False

    assert wait_until(qapp, lambda: bool(summaries), 60.0)


def test_cancelling_stops_the_run_and_says_it_was_cancelled(qapp, tmp_path):
    folder = tmp_path / "recordings"
    folder.mkdir()
    # Long enough that the cancellation lands part way through rather than
    # after everything has already been written.
    for index in range(4):
        write_real_audio(folder / f"take-{index}.m4a", level_db=-30.0, seconds=6.0)
    runner = EnhanceRunner()
    summaries: list = []
    runner.runFinished.connect(summaries.append)

    runner.start(sorted(folder.iterdir()), EnhanceOptions(output_folder=tmp_path / "enhanced"))
    runner.cancel()
    assert wait_until(qapp, lambda: bool(summaries), 60.0)

    assert summaries[0].cancelled is True
    assert summaries[0].enhanced < 4


# -- The dialog ----------------------------------------------------------


def test_the_dialog_opens_with_the_chosen_files_listed(qapp, recordings):
    dialog = EnhanceAudioDialog(recordings, EnhanceSettings())
    try:
        assert dialog._file_list.count() == 2
        assert dialog._file_list.item(0).text() == "alpha.m4a"
        assert "2 files" in dialog._file_group.title()
    finally:
        dialog.close()


def test_the_dialog_opens_with_the_settings_from_last_time(qapp, recordings, tmp_path):
    settings = EnhanceSettings(
        output_folder=str(tmp_path / "somewhere"),
        target_lufs=-14.0,
        ceiling_dbtp=-2.0,
        maximum_gain_db=12.0,
        use_limiter=True,
        output_format="flac",
        replace_existing=True,
    )

    dialog = EnhanceAudioDialog(recordings, settings)
    try:
        assert dialog._folder_edit.text() == str(tmp_path / "somewhere")
        assert dialog._target_box.value() == -14.0
        assert dialog._ceiling_box.value() == -2.0
        assert dialog._gain_box.value() == 12.0
        assert dialog._limiter_box.isChecked() is True
        assert dialog._replace_box.isChecked() is True
        assert dialog.chosen_settings() == settings
    finally:
        dialog.close()


def test_the_output_folder_starts_beside_the_recordings(qapp, recordings, tmp_path):
    """With nothing remembered, a sub-folder of the audio folder is offered."""
    folder = tmp_path / "recordings"

    dialog = EnhanceAudioDialog(recordings, EnhanceSettings(), source_folder=folder)
    try:
        assert Path(dialog._folder_edit.text()) == folder / "Enhanced"
    finally:
        dialog.close()


def test_the_progress_bar_appears_only_once_a_run_has_started(
    qapp, recordings, tmp_path, monkeypatch
):
    silence_completion_message(monkeypatch)
    dialog = EnhanceAudioDialog(
        recordings, EnhanceSettings(output_folder=str(tmp_path / "enhanced"))
    )
    try:
        dialog.show()
        assert dialog._progress_group.isVisible() is False
        assert dialog._report_group.isVisible() is False

        run_and_wait(qapp, dialog)

        assert dialog._progress_group.isVisible() is True
        assert dialog._progress_bar.value() == 100
        assert dialog._report_group.isVisible() is True
    finally:
        dialog.close()


def test_everything_but_cancel_is_switched_off_while_the_run_goes(
    qapp, recordings, tmp_path, monkeypatch
):
    """And the focus is moved to Cancel before anything is switched off.

    A control that is disabled while it holds the focus takes the focus
    with it, which leaves a keyboard or screen reader user nowhere.
    """
    silence_completion_message(monkeypatch)
    dialog = EnhanceAudioDialog(
        recordings, EnhanceSettings(output_folder=str(tmp_path / "enhanced"))
    )
    try:
        dialog.show()
        dialog._start_button.setFocus()

        assert dialog.start()

        assert dialog._parameters_group.isEnabled() is False
        assert dialog._folder_group.isEnabled() is False
        assert dialog._file_group.isEnabled() is False
        assert dialog._start_button.isEnabled() is False
        assert dialog._close_button.isEnabled() is True
        assert dialog.focusWidget() is dialog._close_button
        assert dialog._close_button.text() == "&Cancel"

        assert wait_until(qapp, lambda: dialog.summary is not None, 60.0)

        assert dialog._parameters_group.isEnabled() is True
        assert dialog._start_button.isEnabled() is True
        # With the run over, the same button now closes the dialog, and
        # says so rather than still offering to cancel something.
        assert dialog._close_button.text() == "&Close"
        assert dialog._close_button.accessibleName() == "Close"
    finally:
        dialog.close()


def test_the_completion_message_says_how_the_run_went(
    qapp, recordings, tmp_path, monkeypatch
):
    shown = silence_completion_message(monkeypatch)
    dialog = EnhanceAudioDialog(
        recordings, EnhanceSettings(output_folder=str(tmp_path / "enhanced"))
    )
    try:
        run_and_wait(qapp, dialog)

        assert shown == ["Finished. 2 of 2 files enhanced."]
        # The same words stay on the dialog, and the detail stays with them.
        assert dialog._progress_label.text() == shown[0]
        assert "alpha.m4a" in dialog._report_text.toPlainText()
        assert "beta.m4a" in dialog._report_text.toPlainText()
    finally:
        dialog.close()


def test_the_run_really_writes_the_files(qapp, recordings, tmp_path, monkeypatch):
    silence_completion_message(monkeypatch)
    folder = tmp_path / "enhanced"
    dialog = EnhanceAudioDialog(recordings, EnhanceSettings(output_folder=str(folder)))
    try:
        run_and_wait(qapp, dialog)

        assert sorted(path.name for path in folder.iterdir()) == ["alpha.wav", "beta.wav"]
    finally:
        dialog.close()


def test_a_run_with_no_output_folder_is_refused_and_explained(qapp, recordings):
    dialog = EnhanceAudioDialog(recordings, EnhanceSettings(output_folder=" "))
    try:
        dialog._folder_edit.setText("")

        assert dialog.start() is False

        assert "No output folder has been chosen" in dialog._progress_label.text()
        assert dialog.summary is None
    finally:
        dialog.close()


def test_an_output_folder_that_cannot_be_created_is_reported(qapp, recordings, tmp_path):
    """A file where the folder should go, which Windows will not open as one."""
    blocker = tmp_path / "blocked"
    blocker.write_text("not a folder", encoding="utf-8")
    dialog = EnhanceAudioDialog(recordings, EnhanceSettings(output_folder=str(blocker)))
    try:
        assert dialog.start() is False

        assert "could not be created" in dialog._progress_label.text()
        assert dialog.summary is None
    finally:
        dialog.close()


def test_escape_cancels_the_run_rather_than_closing_on_top_of_it(
    qapp, tmp_path, monkeypatch
):
    silence_completion_message(monkeypatch)
    folder = tmp_path / "recordings"
    folder.mkdir()
    write_real_audio(folder / "long.m4a", level_db=-30.0, seconds=6.0)
    dialog = EnhanceAudioDialog(
        [folder / "long.m4a"], EnhanceSettings(output_folder=str(tmp_path / "enhanced"))
    )
    try:
        dialog.show()
        assert dialog.start()

        dialog.reject()

        # The dialog stays where it is, so that the user is told what was
        # done before they close it.
        assert dialog.isVisible() is True
        assert wait_until(qapp, lambda: dialog.summary is not None, 60.0)
        assert dialog.summary.cancelled is True
        # The half-written file is thrown away, so nothing is left behind
        # that looks like a finished recording.
        assert not (tmp_path / "enhanced" / "long.wav").exists()

        # And now that it has stopped, the same key really does close it.
        dialog.reject()
        assert dialog.isVisible() is False
    finally:
        dialog._runner.stop()
        dialog.close()


# -- The window ----------------------------------------------------------


def open_window(qapp, tmp_path, folder, expected_files: int):
    window = MainWindow(SessionStore(tmp_path / "session.json"))
    window._folder_panel.folderChosen.emit(str(folder))
    assert wait_until(qapp, lambda: window._model.rowCount() == expected_files)
    return window


def test_the_checked_files_are_what_gets_enhanced(qapp, tmp_path, recordings):
    window = open_window(qapp, tmp_path, recordings[0].parent, 2)
    try:
        window._model.set_checked_names(["beta.m4a"])

        assert [f.name for f in window.files_to_enhance()] == ["beta.m4a"]
    finally:
        window.close()


def test_with_nothing_checked_the_highlighted_file_is_used(qapp, tmp_path, recordings):
    window = open_window(qapp, tmp_path, recordings[0].parent, 2)
    try:
        window._table.select_row(1)

        assert [f.name for f in window.files_to_enhance()] == ["beta.m4a"]
    finally:
        window.close()


def test_with_no_folder_open_the_window_says_there_is_nothing_to_enhance(
    qapp, tmp_path, monkeypatch
):
    window = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        opened: list = []
        monkeypatch.setattr(EnhanceAudioDialog, "exec", lambda self: opened.append(self))

        window.show_enhance_audio()

        # No dialog is opened at all, because there is nothing for it to
        # work on. The window says so instead.
        assert opened == []
        assert "nothing to enhance" in window._status_label.text()
    finally:
        window.close()


def test_what_the_dialog_was_left_on_is_remembered_for_next_time(
    qapp, tmp_path, recordings, monkeypatch
):
    window = open_window(qapp, tmp_path, recordings[0].parent, 2)
    try:
        def change_the_settings_and_close(dialog):
            dialog._target_box.setValue(-14.0)
            dialog._limiter_box.setChecked(True)
            dialog._format_box.setCurrentIndex(2)
            return QDialog.DialogCode.Rejected

        monkeypatch.setattr(EnhanceAudioDialog, "exec", change_the_settings_and_close)
        window._table.select_row(0)

        window.show_enhance_audio()

        assert window.settings.enhance.target_lufs == -14.0
        assert window.settings.enhance.use_limiter is True
        assert window.settings.enhance.output_format == "flac"
    finally:
        window.close()

    saved = SettingsStore(tmp_path / SETTINGS_FILE_NAME).load()
    assert saved.enhance.target_lufs == -14.0
    assert saved.enhance.output_format == "flac"
