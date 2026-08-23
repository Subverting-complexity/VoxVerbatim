"""The folder selector at the top of the left-hand side of the window."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vox_verbatim.ui.accessibility import describe

NO_FOLDER_TEXT = "No folder selected"


class FolderPanel(QWidget):
    """Shows the audio folder in use and lets the user pick a different one."""

    folderChosen = Signal(str)
    """The user picked a folder. Carries its absolute path."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._folder: Path | None = None

        # Every Alt shortcut in this window is different from every other
        # one. Alt+F already opens the File menu, so the folder box uses
        # Alt+O.
        self._label = QLabel("Audio f&older", self)

        # A read-only line edit rather than a plain label, because it can
        # take focus, exposes a real text caret for ZoomText to follow, and
        # lets the path be read a word at a time or copied out.
        self._path_edit = QLineEdit(self)
        self._path_edit.setReadOnly(True)
        self._path_edit.setText(NO_FOLDER_TEXT)
        self._path_edit.setCursorPosition(0)
        describe(
            self._path_edit,
            "Audio folder",
            "The folder the audio files are read from. Use the Browse button to change it.",
        )
        self._label.setBuddy(self._path_edit)

        self._browse_button = QPushButton("&Browse...", self)
        describe(
            self._browse_button,
            "Browse for audio folder",
            "Opens a dialog for choosing the folder that holds your audio files.",
        )
        self._browse_button.clicked.connect(self.browse)

        row = QHBoxLayout()
        row.addWidget(self._path_edit, 1)
        row.addWidget(self._browse_button, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)
        layout.addLayout(row)

    @property
    def folder(self) -> Path | None:
        return self._folder

    def set_folder(self, folder: Path | str | None) -> None:
        """Show a folder without asking anyone to load it."""
        if folder is None:
            self._folder = None
            self._path_edit.setText(NO_FOLDER_TEXT)
        else:
            self._folder = Path(folder)
            self._path_edit.setText(str(self._folder))
        # Put the caret at the start so a screen reader and ZoomText both
        # begin at the drive letter rather than the end of a long path.
        self._path_edit.setCursorPosition(0)

    def browse(self) -> None:
        """Ask the user for a folder and report it if they chose one."""
        start_at = str(self._folder) if self._folder is not None else ""
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select the folder containing your audio files",
            start_at,
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )
        if chosen:
            self.folderChosen.emit(chosen)

    def focus_browse_button(self) -> None:
        self._browse_button.setFocus()
