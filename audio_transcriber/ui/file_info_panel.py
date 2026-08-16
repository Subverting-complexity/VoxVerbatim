"""Details of the file that is currently selected, shown under the player."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel, QLineEdit, QWidget

from audio_transcriber.audio.library import AudioFile, DurationState
from audio_transcriber.formatting import (
    UNKNOWN_TEXT,
    format_position,
    spoken_duration,
    spoken_size,
)
from audio_transcriber.ui.accessibility import describe

NO_FILE_TEXT = "No file selected"


class FileInfoPanel(QGroupBox):
    """Shows the name, length, size and playback position of the selected file.

    Each value sits in a read-only line edit rather than a label. A line
    edit can take focus, so the values can be reached with the Tab key and
    read or copied, and it shows a real text caret that ZoomText can track.

    The length and the size are written out in words here, as "4.2
    megabytes" rather than "4.2 MB". The file list keeps the compact forms
    for scanning down a column; this panel is where a value is read closely,
    including out loud, and "1:10:31" read aloud is a puzzle.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Selected file", parent)
        self._audio_file: AudioFile | None = None
        self._last_position_text = ""

        layout = QFormLayout(self)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        # Labels wrap onto their own line when the window is narrow or the
        # system font is large, instead of squeezing the values.
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._name_edit = self._add_field(layout, "File &name", "File name")
        self._duration_edit = self._add_field(layout, "&Duration", "Duration")
        self._size_edit = self._add_field(layout, "File si&ze", "File size")
        self._position_edit = self._add_field(layout, "Pos&ition", "Playback position")

        self.set_file(None)

    def _add_field(
        self,
        layout: QFormLayout,
        label_text: str,
        accessible_name: str,
    ) -> QLineEdit:
        label = QLabel(label_text, self)
        edit = QLineEdit(self)
        edit.setReadOnly(True)
        describe(edit, accessible_name)
        label.setBuddy(edit)
        layout.addRow(label, edit)
        return edit

    def focus_first_field(self) -> None:
        """Put the focus on the first value, for the F6 panel key."""
        self._name_edit.setFocus(Qt.FocusReason.TabFocusReason)

    def set_file(self, audio_file: AudioFile | None) -> None:
        """Show the details of a file, or clear the panel when given ``None``."""
        self._audio_file = audio_file
        if audio_file is None:
            self._set_value(self._name_edit, NO_FILE_TEXT)
            self._set_value(self._duration_edit, UNKNOWN_TEXT)
            self._set_value(self._size_edit, UNKNOWN_TEXT)
            self.set_position(None)
            return
        self._set_value(self._name_edit, audio_file.name)
        self.refresh_duration()
        self._set_value(self._size_edit, spoken_size(audio_file.size_bytes))
        self.set_position(0)

    def refresh_duration(self) -> None:
        """Show the duration again, after it was read in the background."""
        audio_file = self._audio_file
        if audio_file is None:
            return
        if audio_file.duration_state == DurationState.KNOWN:
            self._set_value(self._duration_edit, spoken_duration(audio_file.duration_seconds))
        elif audio_file.duration_state == DurationState.PENDING:
            self._set_value(self._duration_edit, "Still being read")
        else:
            self._set_value(self._duration_edit, UNKNOWN_TEXT)

    def set_position(self, milliseconds: int | None) -> None:
        """Show the playback position.

        Playback reports its position many times a second. The text is only
        replaced when the displayed second actually changes, so that a
        screen reader or magnifier is not chasing a field that rewrites
        itself constantly.
        """
        text = format_position(milliseconds) if milliseconds is not None else UNKNOWN_TEXT
        if text == self._last_position_text:
            return
        self._last_position_text = text
        self._set_value(self._position_edit, text)

    @staticmethod
    def _set_value(edit: QLineEdit, text: str) -> None:
        """Put a value in a field, without disturbing anyone reading it.

        The value is the text itself and nothing else. Putting a second,
        spoken form in the accessible description as well would have a
        screen reader read every field twice over.

        The caret is put back where the reader left it when the field has
        focus. Replacing the text moves the caret to the end on its own, and
        the playback position rewrites itself once a second, so anyone
        reading the field a word at a time, or following the caret with
        ZoomText, would be dragged away from their place every second.
        """
        if edit.text() == text:
            return
        # The window is asked which of its widgets has the focus, rather
        # than the widget itself, so that this still holds when the user has
        # switched to another window and left the caret here.
        if edit.window().focusWidget() is edit:
            caret = edit.cursorPosition()
            edit.setText(text)
            edit.setCursorPosition(min(caret, len(text)))
        else:
            edit.setText(text)
            edit.setCursorPosition(0)
