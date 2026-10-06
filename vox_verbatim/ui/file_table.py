"""The list of audio files, as a standard Qt model and view.

A table model and a ``QTableView`` are used rather than a hand-drawn
control, because Qt already exposes a table to Windows accessibility with
proper rows, columns, headers and cell states. A custom widget would have
to reproduce all of that by hand and would almost certainly get it wrong.

Two ideas live side by side in this table and must not be confused:

* The **selected** row is the file the audio player is working with. There
  is always exactly one, and moving the highlight changes what plays.
* The **checked** files are the ones Enhance Audio and Transcribe act on.
  Any number can be checked, and checking one has no effect on playback.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableView, QWidget

from vox_verbatim.audio.library import AudioFile, DurationState
from vox_verbatim.formatting import (
    format_duration,
    format_size,
    spoken_duration,
    spoken_size,
)

COLUMN_NAME = 0
COLUMN_DURATION = 1
COLUMN_SIZE = 2
COLUMN_COUNT = 3

COLUMN_TITLES = ("File name", "Duration", "File size")

_PENDING_DURATION_TEXT = "Reading..."
_PENDING_DURATION_SPOKEN = "duration is still being read"
_UNKNOWN_DURATION_SPOKEN = "duration is not known"


class AudioFileTableModel(QAbstractTableModel):
    """Holds the audio files of the current folder and which of them are checked."""

    checkedFilesChanged = Signal()
    """The set of checked files changed, whoever changed it."""

    fileChecked = Signal(str, bool)
    """A single file was checked or cleared, by name and new state."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._files: list[AudioFile] = []
        self._checked_names: set[str] = set()

    # -- Contents -------------------------------------------------------

    def set_files(self, files: list[AudioFile]) -> None:
        """Replace the contents of the table.

        Checked files that are no longer in the folder are forgotten, which
        is what keeps deleted or renamed files from lingering in the saved
        session for ever.
        """
        self.beginResetModel()
        self._files = list(files)
        present = {audio_file.key for audio_file in self._files}
        self._checked_names &= present
        self.endResetModel()
        self.checkedFilesChanged.emit()

    def clear(self) -> None:
        self.set_files([])

    def files(self) -> list[AudioFile]:
        return list(self._files)

    def file_at(self, row: int) -> AudioFile | None:
        if 0 <= row < len(self._files):
            return self._files[row]
        return None

    def row_for_name(self, name: str | None) -> int:
        """Return the row holding ``name``, or ``-1`` if the file is not listed."""
        if not name:
            return -1
        for row, audio_file in enumerate(self._files):
            if audio_file.key == name:
                return row
        return -1

    def update_durations(self, durations: list[tuple[str, float | None]]) -> None:
        """Fill in durations that were read in the background."""
        if not durations:
            return
        by_name = dict(durations)
        first_changed: int | None = None
        last_changed = 0
        for row, audio_file in enumerate(self._files):
            if audio_file.key not in by_name:
                continue
            audio_file.set_duration(by_name[audio_file.key])
            if first_changed is None:
                first_changed = row
            last_changed = row
        if first_changed is None:
            return
        self.dataChanged.emit(
            self.index(first_changed, COLUMN_DURATION),
            self.index(last_changed, COLUMN_DURATION),
            [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.AccessibleTextRole],
        )

    def set_duration_for_name(self, name: str, seconds: float | None) -> None:
        """Record a duration for one file, for example one the player reported."""
        self.update_durations([(name, seconds)])

    # -- Checked files --------------------------------------------------

    def checked_names(self) -> list[str]:
        """The checked file names, in the order they appear in the table."""
        return [audio_file.key for audio_file in self.checked_files()]

    def checked_files(self) -> list[AudioFile]:
        """The checked files themselves, in the order they appear in the table."""
        return [
            audio_file for audio_file in self._files if audio_file.key in self._checked_names
        ]

    def set_checked_names(self, names: list[str] | set[str]) -> None:
        """Check exactly the named files, and clear every other one."""
        wanted = {name for name in names if self.row_for_name(name) >= 0}
        if wanted == self._checked_names:
            return
        self._checked_names = wanted
        if self._files:
            self.dataChanged.emit(
                self.index(0, COLUMN_NAME),
                self.index(len(self._files) - 1, COLUMN_NAME),
                [Qt.ItemDataRole.CheckStateRole],
            )
        self.checkedFilesChanged.emit()

    def is_checked(self, row: int) -> bool:
        audio_file = self.file_at(row)
        return audio_file is not None and audio_file.key in self._checked_names

    def set_checked(self, row: int, checked: bool) -> bool:
        """Check or clear one row. Returns whether anything changed."""
        audio_file = self.file_at(row)
        if audio_file is None:
            return False
        if checked == (audio_file.key in self._checked_names):
            return False
        if checked:
            self._checked_names.add(audio_file.key)
        else:
            self._checked_names.discard(audio_file.key)
        index = self.index(row, COLUMN_NAME)
        self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
        self.fileChecked.emit(audio_file.key, checked)
        self.checkedFilesChanged.emit()
        return True

    def toggle_checked(self, row: int) -> bool:
        return self.set_checked(row, not self.is_checked(row))

    # -- QAbstractTableModel --------------------------------------------

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._files)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return COLUMN_COUNT

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if index.column() == COLUMN_NAME:
            flags |= Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ):
        if orientation != Qt.Orientation.Horizontal:
            return None
        if not 0 <= section < COLUMN_COUNT:
            return None
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.AccessibleTextRole):
            return COLUMN_TITLES[section]
        if role == Qt.ItemDataRole.ToolTipRole and section == COLUMN_NAME:
            return "The check box selects a file for Enhance Audio and Transcribe."
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        audio_file = self.file_at(index.row())
        if audio_file is None:
            return None
        column = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            return self._display_text(audio_file, column)

        # Screen readers prefer whole words to abbreviations, so the spoken
        # form of each cell is offered separately from what is shown.
        if role == Qt.ItemDataRole.AccessibleTextRole:
            return self._spoken_text(audio_file, column)

        if role == Qt.ItemDataRole.CheckStateRole and column == COLUMN_NAME:
            checked = audio_file.key in self._checked_names
            return Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked

        if role == Qt.ItemDataRole.ToolTipRole:
            return str(audio_file.path)

        if role == Qt.ItemDataRole.TextAlignmentRole and column in (
            COLUMN_DURATION,
            COLUMN_SIZE,
        ):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        if role == Qt.ItemDataRole.UserRole:
            return audio_file

        return None

    def setData(self, index: QModelIndex, value, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid() or role != Qt.ItemDataRole.CheckStateRole:
            return False
        if index.column() != COLUMN_NAME:
            return False
        checked = Qt.CheckState(value) == Qt.CheckState.Checked
        self.set_checked(index.row(), checked)
        return True

    # -- Cell text ------------------------------------------------------

    @staticmethod
    def _display_text(audio_file: AudioFile, column: int) -> str:
        if column == COLUMN_NAME:
            return audio_file.name
        if column == COLUMN_DURATION:
            if audio_file.duration_state == DurationState.PENDING:
                return _PENDING_DURATION_TEXT
            return format_duration(audio_file.duration_seconds)
        if column == COLUMN_SIZE:
            return format_size(audio_file.size_bytes)
        return ""

    @staticmethod
    def _spoken_text(audio_file: AudioFile, column: int) -> str:
        if column == COLUMN_NAME:
            return audio_file.name
        if column == COLUMN_DURATION:
            if audio_file.duration_state == DurationState.PENDING:
                return _PENDING_DURATION_SPOKEN
            if audio_file.duration_state == DurationState.UNAVAILABLE:
                return _UNKNOWN_DURATION_SPOKEN
            return spoken_duration(audio_file.duration_seconds)
        if column == COLUMN_SIZE:
            return spoken_size(audio_file.size_bytes)
        return ""


class AudioFileTableView(QTableView):
    """A table of audio files that can be worked entirely from the keyboard."""

    toggleCheckRequested = Signal(int)
    """The user asked to check or clear the file in this row."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setWordWrap(False)
        self.setSortingEnabled(False)
        self.verticalHeader().setVisible(False)

        # Tab should move on to the next control instead of walking across
        # the cells of this table, so that the tab order through the window
        # stays predictable.
        self.setTabKeyNavigation(False)

        header = self.horizontalHeader()
        header.setSectionsClickable(False)
        header.setHighlightSections(False)

    def setModel(self, model) -> None:
        """Attach a model, then size the columns.

        Column widths can only be set once the header knows how many
        sections it has, which is only true after a model is attached.
        """
        super().setModel(model)
        if model is None:
            return
        header = self.horizontalHeader()
        # The file name takes whatever room is left over, because it is the
        # column that varies in length and matters most.
        header.setSectionResizeMode(COLUMN_NAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(COLUMN_DURATION, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COLUMN_SIZE, QHeaderView.ResizeMode.ResizeToContents)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Let the space bar check or clear the file on the highlighted row.

        Qt only toggles a check box when the highlight is on the cell that
        holds it. Rows are selected whole here, so the space bar is handled
        directly and works no matter which column the highlight sits in.

        The highlight moves to the file name first. The check box lives on
        that cell, so a screen reader reading the focused cell is looking at
        the one whose state just changed. Toggling from the Duration cell
        without moving would change a state the user is not pointed at, and
        they would hear nothing at all.
        """
        is_toggle_key = event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Select)
        no_modifiers = event.modifiers() == Qt.KeyboardModifier.NoModifier
        index = self.currentIndex()
        if is_toggle_key and no_modifiers and index.isValid():
            if index.column() != COLUMN_NAME:
                self.setCurrentIndex(index.sibling(index.row(), COLUMN_NAME))
            self.toggleCheckRequested.emit(index.row())
            event.accept()
            return
        super().keyPressEvent(event)

    def selected_row(self) -> int:
        index = self.currentIndex()
        return index.row() if index.isValid() else -1

    def select_row(self, row: int) -> None:
        """Highlight a row and scroll it into view."""
        model = self.model()
        if model is None or not 0 <= row < model.rowCount():
            return
        self.setCurrentIndex(model.index(row, COLUMN_NAME))
        self.scrollTo(model.index(row, COLUMN_NAME), QAbstractItemView.ScrollHint.EnsureVisible)


def path_of(audio_file: AudioFile | None) -> Path | None:
    return audio_file.path if audio_file is not None else None
