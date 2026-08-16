"""Tests for the things a screen reader and a magnifier depend on.

These check what Qt exposes to Windows accessibility. They cannot prove
that JAWS or NVDA read something out, which needs a real screen reader, but
they do catch the case where the information never reaches them at all.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QAccessible, QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
)

from audio_transcriber.audio.library import AudioFile
from audio_transcriber.session import SessionStore
from audio_transcriber.settings import EnhanceSettings
from audio_transcriber.ui import enhance_dialog as enhance_dialog_module
from audio_transcriber.ui import main_window as main_window_module
from audio_transcriber.ui.enhance_dialog import EnhanceAudioDialog
from audio_transcriber.ui.file_info_panel import FileInfoPanel
from audio_transcriber.ui.file_table import (
    COLUMN_DURATION,
    COLUMN_NAME,
    AudioFileTableModel,
    AudioFileTableView,
)
from audio_transcriber.ui.main_window import MainWindow
from audio_transcriber.ui.player_panel import (
    STATUS_FINISHED,
    STATUS_NO_FILE,
    STATUS_PAUSED,
    STATUS_PLAYING,
    STATUS_STOPPED,
)

from tests.conftest import wait_until, write_fake_audio


def accessible_name(widget) -> str:
    interface = QAccessible.queryAccessibleInterface(widget)
    assert interface is not None
    return interface.text(QAccessible.Text.Name)


def open_window(qapp, tmp_path, folder=None, expected_files=None) -> MainWindow:
    window = MainWindow(SessionStore(tmp_path / "session.json"))
    window.show()
    if folder is not None:
        window._folder_panel.folderChosen.emit(str(folder))
    if expected_files is not None:
        assert wait_until(qapp, lambda: window._model.rowCount() == expected_files)
    return window


def test_a_label_carrying_a_message_reports_the_message_not_a_name(qapp, tmp_path):
    """A label has no accessible value: naming it hides what it says.

    Setting an accessible name on a status label replaced its text with
    that name, so every status message and every error was invisible to a
    screen reader.
    """
    window = open_window(qapp, tmp_path)
    try:
        window._set_status("The folder D:\\Recordings is not available.")

        assert accessible_name(window._status_label) == (
            "The folder D:\\Recordings is not available."
        )

        window._summary_label.setText("4 files, 2 checked")
        assert accessible_name(window._summary_label) == "4 files, 2 checked"
        assert accessible_name(window._player_panel._status_label) == STATUS_NO_FILE
    finally:
        window.close()


def test_every_control_worth_reaching_has_a_name(qapp, tmp_path):
    window = open_window(qapp, tmp_path)
    try:
        unnamed = [
            widget
            for kind in (QPushButton, QLineEdit, QTableView)
            for widget in window.findChildren(kind)
            if not widget.accessibleName()
        ]

        assert unnamed == []
    finally:
        window.close()


def test_the_panel_key_lands_on_a_control_rather_than_a_container(qapp, tmp_path):
    """F6 must not strand the focus on a container that answers no keys."""
    folder = tmp_path / "recordings"
    folder.mkdir()
    write_fake_audio(folder / "one.m4a")
    window = open_window(qapp, tmp_path, folder, expected_files=1)
    try:
        landed = []
        for _ in range(4):
            window.focus_next_panel()
            landed.append(window.focusWidget())

        assert all(widget is not None for widget in landed)
        for widget in landed:
            assert widget.focusPolicy() != Qt.FocusPolicy.NoFocus
        assert {type(widget) for widget in landed} == {
            QPushButton,
            AudioFileTableView,
            QLineEdit,
        }
    finally:
        window.close()


def test_the_space_bar_moves_to_the_cell_that_holds_the_check_box(qapp):
    """The state must change on the cell the user is pointed at.

    Selection is by row and Left and Right move across the columns, so the
    highlight can sit on Duration. Toggling from there without moving would
    change a check box on another cell, and the user would hear nothing.
    """
    model = AudioFileTableModel()
    model.set_files(
        [AudioFile(path=Path(f"C:/Audio/{name}.m4a"), size_bytes=1024) for name in "ab"]
    )
    view = AudioFileTableView()
    view.setModel(model)
    view.toggleCheckRequested.connect(model.toggle_checked)
    view.setCurrentIndex(model.index(1, COLUMN_DURATION))

    QApplication.sendEvent(
        view,
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier),
    )

    assert model.checked_names() == ["b.m4a"]
    assert view.currentIndex().column() == COLUMN_NAME


def test_reading_the_position_field_is_not_interrupted_by_playback(qapp):
    """The caret stays put while the value rewrites itself every second."""
    panel = FileInfoPanel()
    panel.show()
    panel.set_position(65_000)
    panel._position_edit.setFocus()
    # hasFocus is false while no window is active, which is the case in a
    # headless test run, so the window is asked directly instead.
    assert panel.window().focusWidget() is panel._position_edit
    panel._position_edit.setCursorPosition(3)

    panel.set_position(66_000)

    assert panel._position_edit.text() == "1:06"
    assert panel._position_edit.cursorPosition() == 3


def test_the_position_field_starts_at_the_beginning_when_nobody_is_reading_it(qapp):
    panel = FileInfoPanel()
    panel.set_position(65_000)

    assert panel._position_edit.cursorPosition() == 0


def test_only_playback_the_user_asked_for_is_announced(qapp, tmp_path, monkeypatch):
    """Arrowing down the list must not interrupt each file name with "Stopped"."""
    announced: list[str] = []
    monkeypatch.setattr(
        main_window_module,
        "announce",
        lambda widget, message, urgent=False: announced.append(message),
    )
    window = open_window(qapp, tmp_path)
    try:
        for status in (STATUS_PLAYING, STATUS_PAUSED, STATUS_FINISHED):
            window._on_playback_status_changed(status)
        said_for_commands = list(announced)

        announced.clear()
        for status in (STATUS_STOPPED, STATUS_NO_FILE):
            window._on_playback_status_changed(status)

        assert said_for_commands == [STATUS_PLAYING, STATUS_PAUSED, STATUS_FINISHED]
        assert announced == []
    finally:
        window.close()


def test_checking_a_file_is_said_out_loud(qapp, tmp_path, monkeypatch):
    announced: list[str] = []
    monkeypatch.setattr(
        main_window_module,
        "announce",
        lambda widget, message, urgent=False: announced.append(message),
    )
    folder = tmp_path / "recordings"
    folder.mkdir()
    write_fake_audio(folder / "one.m4a")
    window = open_window(qapp, tmp_path, folder, expected_files=1)
    try:
        announced.clear()
        window._model.set_checked(0, True)

        assert "one.m4a checked" in announced
    finally:
        window.close()


def test_the_focus_is_caught_when_the_transport_controls_switch_off(qapp, tmp_path):
    """Disabling a control that has focus must not drop the user somewhere random."""
    folder = tmp_path / "recordings"
    folder.mkdir()
    write_fake_audio(folder / "one.m4a")
    window = open_window(qapp, tmp_path, folder, expected_files=1)
    try:
        window._player_panel.focus_play_button()
        assert isinstance(window.focusWidget(), QPushButton)

        window._player_panel.set_media_loaded(False)

        assert window.focusWidget() is window._table
        assert "focus has moved" in window._status_label.text()
    finally:
        window.close()


# -- The Enhance Audio dialog -------------------------------------------


def enhance_dialog(tmp_path, files=None):
    folder = tmp_path / "recordings"
    folder.mkdir(exist_ok=True)
    paths = files if files is not None else [folder / "one.m4a"]
    return EnhanceAudioDialog(paths, EnhanceSettings(output_folder=str(tmp_path / "enhanced")))


def test_every_control_in_the_enhance_dialog_has_a_name(qapp, tmp_path):
    dialog = enhance_dialog(tmp_path)
    try:
        unnamed = [
            widget.__class__.__name__
            for widget in (
                dialog._file_list,
                dialog._folder_edit,
                dialog._browse_button,
                dialog._target_box,
                dialog._ceiling_box,
                dialog._gain_box,
                dialog._limiter_box,
                dialog._format_box,
                dialog._replace_box,
                dialog._progress_bar,
                dialog._report_text,
                dialog._start_button,
                dialog._close_button,
            )
            if not widget.accessibleName()
        ]

        assert unnamed == []
    finally:
        dialog.close()


def test_no_two_controls_in_the_enhance_dialog_answer_the_same_alt_key(qapp, tmp_path):
    """Two controls on one Alt letter means neither is reliably reachable."""
    dialog = enhance_dialog(tmp_path)
    try:
        letters = [
            text[text.index("&") + 1].casefold()
            for text in (
                widget.text()
                for kind in (QLabel, QPushButton, QCheckBox)
                for widget in dialog.findChildren(kind)
            )
            if "&" in text and not text.endswith("&")
        ]

        assert sorted(letters) == sorted(set(letters)), f"repeated Alt letters in {letters}"
    finally:
        dialog.close()


def test_each_field_in_the_enhance_dialog_is_reached_by_its_own_label(qapp, tmp_path):
    """A label with no buddy names nothing, and the field is read as blank."""
    dialog = enhance_dialog(tmp_path)
    try:
        labelled = {label.buddy() for label in dialog.findChildren(QLabel) if label.buddy()}

        for field in (
            dialog._folder_edit,
            dialog._target_box,
            dialog._ceiling_box,
            dialog._gain_box,
            dialog._format_box,
        ):
            assert field in labelled, f"{field.accessibleName()} has no label pointing at it"
    finally:
        dialog.close()


def test_the_progress_of_a_run_is_said_out_loud(qapp, tmp_path, monkeypatch):
    """A progress bar is only read if the user goes and looks at it.

    Each file is therefore announced as it starts, which is the pace at
    which something actually changes.
    """
    announced: list[str] = []
    monkeypatch.setattr(
        enhance_dialog_module,
        "announce",
        lambda widget, message, urgent=False: announced.append(message),
    )
    dialog = enhance_dialog(tmp_path)
    try:
        dialog._on_file_started("beta.m4a", 2, 3)

        assert announced == ["Enhancing beta.m4a. File 2 of 3."]
    finally:
        dialog.close()


def test_a_run_that_cannot_start_says_why_out_loud(qapp, tmp_path, monkeypatch):
    announced: list[str] = []
    monkeypatch.setattr(
        enhance_dialog_module,
        "announce",
        lambda widget, message, urgent=False: announced.append(message),
    )
    dialog = enhance_dialog(tmp_path)
    try:
        dialog._folder_edit.setText("")

        assert dialog.start() is False

        assert announced
        assert "No output folder has been chosen" in announced[0]
    finally:
        dialog.close()
