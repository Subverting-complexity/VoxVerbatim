"""Details of the file that is currently selected, shown under the player."""

from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel, QLineEdit, QWidget

from audio_transcriber.audio.library import AudioFile, DurationState
from audio_transcriber.formatting import (
    UNKNOWN_TEXT,
    format_duration,
    format_position,
    format_size,
    spoken_duration,
    spoken_position,
    spoken_size,
)
from audio_transcriber.ui.accessibility import describe

NO_FILE_TEXT = "No file selected"


class FileInfoPanel(QGroupBox):
    """Shows the name, length, size and playback position of the selected file.

    Each value sits in a read-only line edit rather than a label. A line
    edit can take focus, so the values can be reached with the Tab key and
    read or copied, and it shows a real text caret that ZoomText can track.
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
        self._position_edit = self._add_field(
            layout,
            "Position",
            "Playback position",
            "How far into the recording playback has reached.",
        )

        self.set_file(None)

    def _add_field(
        self,
        layout: QFormLayout,
        label_text: str,
        accessible_name: str,
        description: str | None = None,
    ) -> QLineEdit:
        label = QLabel(label_text, self)
        edit = QLineEdit(self)
        edit.setReadOnly(True)
        describe(edit, accessible_name, description)
        label.setBuddy(edit)
        layout.addRow(label, edit)
        return edit

    def set_file(self, audio_file: AudioFile | None) -> None:
        """Show the details of a file, or clear the panel when given ``None``."""
        self._audio_file = audio_file
        if audio_file is None:
            self._set_value(self._name_edit, NO_FILE_TEXT, NO_FILE_TEXT)
            self._set_value(self._duration_edit, UNKNOWN_TEXT, UNKNOWN_TEXT)
            self._set_value(self._size_edit, UNKNOWN_TEXT, UNKNOWN_TEXT)
            self.set_position(None)
            return
        self._set_value(self._name_edit, audio_file.name, audio_file.name)
        self.refresh_duration()
        self._set_value(
            self._size_edit,
            format_size(audio_file.size_bytes),
            spoken_size(audio_file.size_bytes),
        )
        self.set_position(0)

    def refresh_duration(self) -> None:
        """Show the duration again, after it was read in the background."""
        audio_file = self._audio_file
        if audio_file is None:
            return
        if audio_file.duration_state == DurationState.KNOWN:
            self._set_value(
                self._duration_edit,
                format_duration(audio_file.duration_seconds),
                spoken_duration(audio_file.duration_seconds),
            )
        elif audio_file.duration_state == DurationState.PENDING:
            self._set_value(self._duration_edit, "Reading...", "duration is still being read")
        else:
            self._set_value(self._duration_edit, UNKNOWN_TEXT, "duration is not known")

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
        spoken = spoken_position(milliseconds) if milliseconds is not None else UNKNOWN_TEXT
        self._set_value(self._position_edit, text, spoken)

    @staticmethod
    def _set_value(edit: QLineEdit, text: str, spoken_text: str) -> None:
        edit.setText(text)
        edit.setCursorPosition(0)
        # The name a screen reader reads stays put; the spoken form of the
        # value goes in the description so that "4.2 MB" is read as
        # "4.2 megabytes" rather than letter by letter.
        edit.setAccessibleDescription(spoken_text)
