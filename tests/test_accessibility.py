"""Tests for the things a screen reader and a magnifier depend on.

These check what Qt exposes to Windows accessibility. They cannot prove
that JAWS or NVDA read something out, which needs a real screen reader, but
they do catch the case where the information never reaches them at all.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QAccessible, QKeyEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
)

from vox_verbatim.audio.library import AudioFile
from vox_verbatim.session import SessionStore
from vox_verbatim.settings import EnhanceSettings
from vox_verbatim.ui import enhance_dialog as enhance_dialog_module
from vox_verbatim.ui import main_window as main_window_module
from vox_verbatim.ui import enhance_notes as notes
from vox_verbatim.ui.enhance_dialog import EnhanceAudioDialog
from vox_verbatim.ui.file_info_panel import FileInfoPanel
from vox_verbatim.ui.file_table import (
    COLUMN_DURATION,
    COLUMN_NAME,
    AudioFileTableModel,
    AudioFileTableView,
)
from vox_verbatim.ui.main_window import MainWindow
from vox_verbatim.ui.player_panel import (
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


def test_the_panel_key_passes_a_player_with_nothing_loaded(qapp, tmp_path):
    """With no file, Play is switched off and refuses the focus.

    F6 used to say "Audio player" and leave the focus on the file list, so
    the next F6 started from the list again and the details were never
    reached.
    """
    window = open_window(qapp, tmp_path)
    try:
        window._table.setFocus()
        window.focus_next_panel()

        assert window.focusWidget() is window._info_panel._name_edit
        assert window._status_label.text() == "Selected file"
    finally:
        window.close()


def test_the_panel_key_still_lands_on_play_with_a_file_loaded(qapp, tmp_path):
    folder = tmp_path / "recordings"
    folder.mkdir()
    write_fake_audio(folder / "one.m4a")
    window = open_window(qapp, tmp_path, folder, expected_files=1)
    try:
        window._table.setFocus()
        window.focus_next_panel()

        assert window.focusWidget() is window._player_panel._play_button
        assert window._status_label.text() == "Audio player"
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


def test_the_note_panel_follows_the_focus_onto_every_setting(qapp, tmp_path):
    """Including the spin boxes, which hand the focus to a field inside them.

    Watching each control for focus directly would miss those, because the
    widget that takes the focus is not the widget the user thinks they are
    on.
    """
    dialog = enhance_dialog(tmp_path)
    try:
        dialog.show()
        expected = [
            (dialog._file_list, notes.FILES),
            (dialog._folder_edit, notes.OUTPUT_FOLDER),
            (dialog._browse_button, notes.OUTPUT_FOLDER),
            (dialog._target_box, notes.TARGET_LOUDNESS),
            (dialog._ceiling_box, notes.CEILING),
            (dialog._gain_box, notes.MAXIMUM_GAIN),
            (dialog._limiter_box, notes.LIMITER),
            (dialog._format_box, notes.OUTPUT_FORMAT),
            (dialog._replace_box, notes.REPLACE_EXISTING),
        ]

        landed = []
        for widget, key in expected:
            widget.setFocus(Qt.FocusReason.TabFocusReason)
            qapp.processEvents()
            landed.append((key, dialog._showing_note, dialog._notes_text.toPlainText()))

        for key, showing, shown in landed:
            assert showing == key
            assert shown == notes.note_text(key)
    finally:
        dialog.close()


def test_reading_a_note_does_not_change_the_note(qapp, tmp_path):
    """The panel takes focus so it can be read, and must hold still when it does."""
    dialog = enhance_dialog(tmp_path)
    try:
        dialog.show()
        dialog._limiter_box.setFocus(Qt.FocusReason.TabFocusReason)
        qapp.processEvents()

        dialog._notes_text.setFocus(Qt.FocusReason.TabFocusReason)
        qapp.processEvents()

        assert dialog._showing_note == notes.LIMITER
        assert dialog._notes_text.toPlainText() == notes.note_text(notes.LIMITER)
    finally:
        dialog.close()


def test_the_note_panel_says_which_setting_it_is_talking_about(qapp, tmp_path):
    """Otherwise a block of prose appears with nothing tying it to anything."""
    dialog = enhance_dialog(tmp_path)
    try:
        dialog.show()
        dialog._ceiling_box.setFocus(Qt.FocusReason.TabFocusReason)
        qapp.processEvents()

        assert notes.note_for(notes.CEILING).title in dialog._notes_group.title()
    finally:
        dialog.close()


def test_a_note_starts_at_its_first_line_rather_than_where_the_last_one_stopped(
    qapp, tmp_path
):
    dialog = enhance_dialog(tmp_path)
    try:
        dialog.show()
        dialog._limiter_box.setFocus(Qt.FocusReason.TabFocusReason)
        qapp.processEvents()
        dialog._notes_text.verticalScrollBar().setValue(
            dialog._notes_text.verticalScrollBar().maximum()
        )

        dialog._gain_box.setFocus(Qt.FocusReason.TabFocusReason)
        qapp.processEvents()

        assert dialog._notes_text.verticalScrollBar().value() == 0
        assert dialog._notes_text.textCursor().position() == 0
    finally:
        dialog.close()


def test_the_short_summary_is_what_a_screen_reader_hears_on_focus(qapp, tmp_path):
    """The full note is far too long to hear every time the focus moves."""
    dialog = enhance_dialog(tmp_path)
    try:
        for widget, key in (
            (dialog._target_box, notes.TARGET_LOUDNESS),
            (dialog._limiter_box, notes.LIMITER),
            (dialog._format_box, notes.OUTPUT_FORMAT),
        ):
            assert widget.accessibleDescription() == notes.summary_of(key)
            assert widget.accessibleName() == notes.note_for(key).title
            # The same words are on the tooltip, so a mouse user and a
            # screen reader user are told the same thing.
            assert widget.toolTip() == notes.summary_of(key)
    finally:
        dialog.close()


def test_the_whole_guide_can_be_opened_from_the_dialog(qapp, tmp_path, monkeypatch):
    opened: list = []
    monkeypatch.setattr(
        enhance_dialog_module.EnhancementGuideDialog, "exec", lambda self: opened.append(self)
    )
    dialog = enhance_dialog(tmp_path)
    try:
        dialog.show()

        dialog._guide_button.click()

        assert len(opened) == 1
        assert opened[0]._text.toPlainText() == notes.guide_text()
    finally:
        dialog.close()


def test_the_guide_window_starts_on_the_text_rather_than_the_close_button(qapp):
    guide = enhance_dialog_module.EnhancementGuideDialog()
    try:
        guide.show()

        assert guide.focusWidget() is guide._text
        assert guide._text.isReadOnly()
        assert guide._text.accessibleName()
    finally:
        guide.close()


def test_the_dialog_scrolls_rather_than_running_off_the_bottom_of_the_screen(qapp, tmp_path):
    """At a large text size this dialog is taller than the screen it is on.

    Growing past the screen would put Start and Cancel out of reach, so it
    stops at the screen and the contents scroll instead.
    """
    dialog = enhance_dialog(tmp_path)
    try:
        dialog.show()
        room = dialog._room_to_grow()

        dialog._progress_group.setVisible(True)
        dialog._report_group.setVisible(True)
        dialog._grow_to_fit()

        assert dialog.height() <= room
        # And the buttons are outside the scrolling part, so they stay put.
        assert not dialog._scroll.isAncestorOf(dialog._start_button)
        assert not dialog._scroll.isAncestorOf(dialog._close_button)
    finally:
        dialog.close()


def test_f1_is_what_opens_the_guide(qapp, tmp_path, monkeypatch):
    """F1 asks for help on what is in front of you everywhere else in Windows."""
    opened: list = []
    monkeypatch.setattr(
        enhance_dialog_module.EnhancementGuideDialog, "exec", lambda self: opened.append(self)
    )
    dialog = enhance_dialog(tmp_path)
    try:
        dialog.show()
        assert dialog._guide_shortcut.key() == QKeySequence(Qt.Key.Key_F1)

        dialog._guide_shortcut.activated.emit()

        assert len(opened) == 1
    finally:
        dialog.close()


def test_a_closed_dialog_stops_watching_the_focus(qapp, tmp_path):
    """Otherwise every opening of Enhance Audio leaves a watcher behind.

    The watch is on the application, which outlives the dialog, and the
    dialog is owned by the main window rather than thrown away when it
    closes. Cancel and Escape both go through reject, which hides the
    dialog without raising a close event, so tidying up on close alone
    never happened at all in ordinary use.
    """
    dialogs = [enhance_dialog(tmp_path) for _ in range(3)]
    try:
        for dialog in dialogs:
            dialog.show()
            qapp.processEvents()
        assert [d._watching_focus for d in dialogs] == [True, True, True]

        # reject is what the Cancel button and Escape both do, and what
        # exec returns through.
        for dialog in dialogs:
            dialog.reject()
            qapp.processEvents()

        assert [d._watching_focus for d in dialogs] == [False, False, False]

        # A dialog that is opened again picks the watch back up, rather
        # than going quiet for the rest of its life.
        dialogs[0].show()
        qapp.processEvents()
        dialogs[0]._limiter_box.setFocus(Qt.FocusReason.TabFocusReason)
        qapp.processEvents()

        assert dialogs[0]._watching_focus is True
        assert dialogs[0]._showing_note == notes.LIMITER
    finally:
        for dialog in dialogs:
            dialog.close()
