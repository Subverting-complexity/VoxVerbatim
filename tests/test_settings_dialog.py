"""Tests for the Settings dialog and the way the window uses it."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog

from audio_transcriber.session import SessionStore
from audio_transcriber.settings import SETTINGS_FILE_NAME, Settings, SettingsStore
from audio_transcriber.ui.help_dialogs import keyboard_shortcuts_text
from audio_transcriber.ui.main_window import MainWindow
from audio_transcriber.ui.settings_dialog import SettingsDialog

from tests.conftest import wait_until, write_fake_audio


def test_the_dialog_opens_showing_the_settings_in_force(qapp):
    settings = Settings(
        reopen_last_folder=False,
        short_skip_seconds=10,
        medium_skip_seconds=45,
        long_skip_seconds=200,
    )

    dialog = SettingsDialog(settings)

    assert dialog._reopen_box.isChecked() is False
    assert dialog._short_box.value() == 10
    assert dialog._medium_box.value() == 45
    assert dialog._long_box.value() == 200


def test_the_dialog_hands_back_what_the_user_changed(qapp):
    dialog = SettingsDialog(Settings())

    dialog._reopen_box.setChecked(False)
    dialog._short_box.setValue(30)

    chosen = dialog.chosen_settings()
    assert chosen.reopen_last_folder is False
    assert chosen.short_skip_seconds == 30
    # Anything untouched keeps the value it had.
    assert chosen.medium_skip_seconds == 120


def test_the_dialog_refuses_a_skip_outside_what_the_buttons_are_for(qapp):
    dialog = SettingsDialog(Settings())

    dialog._short_box.setValue(0)
    assert dialog._short_box.value() >= 1

    dialog._long_box.setValue(999_999)
    assert dialog._long_box.value() <= 3600


def test_the_dialog_starts_on_a_setting_rather_than_the_ok_button(qapp):
    dialog = SettingsDialog(Settings())
    dialog.show()

    assert dialog.focusWidget() is dialog._reopen_box


def test_new_intervals_reach_the_buttons_the_menu_and_the_help(qapp, tmp_path):
    """A changed interval must change everything that quotes it."""
    window = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        window.apply_settings(Settings(short_skip_seconds=30, medium_skip_seconds=90))

        forward_short = window._player_panel._forward_buttons[0]
        assert forward_short.text() == "+ 30 sec"
        assert forward_short.accessibleName() == "Forward 30 seconds"

        back_medium = [
            action.text() for action, interval, direction in window._skip_actions
            if interval == 1 and direction == -1
        ]
        assert back_medium == ["Back 1 minute 30 seconds"]

        assert "Back 30 seconds" in keyboard_shortcuts_text(window.settings)
    finally:
        window.close()


def test_a_changed_interval_is_how_far_the_button_moves(qapp, tmp_path):
    window = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        requested: list[int] = []
        window._player_panel.skipRequested.connect(requested.append)
        # The transport controls are switched off until a file is loaded,
        # and a switched-off button does nothing when it is clicked.
        window._player_panel.set_media_loaded(True)
        window.apply_settings(Settings(short_skip_seconds=30))

        window._player_panel._forward_buttons[0].click()
        window._player_panel._back_buttons[-1].click()

        assert requested == [30_000, -30_000]
    finally:
        window.close()


def test_settings_are_saved_beside_the_session(qapp, tmp_path):
    window = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        window.apply_settings(Settings(short_skip_seconds=25))
    finally:
        window.close()

    assert SettingsStore(tmp_path / SETTINGS_FILE_NAME).load().short_skip_seconds == 25


def test_settings_survive_a_restart(qapp, tmp_path):
    store = SessionStore(tmp_path / "session.json")
    window = MainWindow(store)
    try:
        window.apply_settings(Settings(reopen_last_folder=False, long_skip_seconds=42))
    finally:
        window.close()

    reopened = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        assert reopened.settings.reopen_last_folder is False
        assert reopened.settings.long_skip_seconds == 42
        assert reopened._player_panel._back_buttons[0].accessibleName() == "Back 42 seconds"
    finally:
        reopened.close()


def test_switching_off_reopening_leaves_the_folder_closed_on_start(qapp, tmp_path):
    """The folder is still remembered; it is simply not opened again."""
    folder = tmp_path / "recordings"
    folder.mkdir()
    write_fake_audio(folder / "one.m4a")

    window = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        window._folder_panel.folderChosen.emit(str(folder))
        assert wait_until(qapp, lambda: window._model.rowCount() == 1)
        window.apply_settings(Settings(reopen_last_folder=False))
    finally:
        window.close()

    reopened = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        assert reopened._model.rowCount() == 0
        assert "switched off in the settings" in reopened._status_label.text()
        # The folder is remembered, so switching the setting back on brings
        # it straight back.
        assert reopened._session.folder == str(folder)
    finally:
        reopened.close()


def test_the_log_file_is_reported_rather_than_opened_when_it_is_missing(qapp, tmp_path):
    window = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        opened: list[str] = []
        window._open_with_windows = lambda path, description: opened.append(description)

        window.open_log_file()

        # There is no log file in a test run, so the user is told where one
        # would appear rather than being handed an error from Windows.
        assert opened == []
        assert "no log file yet" in window._status_label.text()
    finally:
        window.close()


def test_the_settings_folder_that_opens_is_the_one_actually_in_use(qapp, tmp_path):
    window = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        asked_for: list = []
        window._open_with_windows = lambda path, description: asked_for.append(path)

        window.open_settings_folder()

        assert asked_for == [tmp_path]
    finally:
        window.close()


def test_cancelling_the_dialog_changes_nothing(qapp, tmp_path, monkeypatch):
    window = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        before = window.settings
        monkeypatch.setattr(SettingsDialog, "exec", lambda self: QDialog.DialogCode.Rejected)

        window.show_settings()

        assert window.settings == before
        assert not (tmp_path / SETTINGS_FILE_NAME).exists()
    finally:
        window.close()


def test_accepting_the_dialog_applies_what_was_chosen(qapp, tmp_path, monkeypatch):
    window = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        def accept_with_a_longer_skip(dialog):
            dialog._short_box.setValue(20)
            return QDialog.DialogCode.Accepted

        monkeypatch.setattr(SettingsDialog, "exec", accept_with_a_longer_skip)

        window.show_settings()

        assert window.settings.short_skip_seconds == 20
        assert window._player_panel._forward_buttons[0].text() == "+ 20 sec"
    finally:
        window.close()
