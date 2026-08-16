"""The Help menu dialogs: the keyboard shortcut list and the About box."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from audio_transcriber import APPLICATION_NAME, __version__
from audio_transcriber.ui.accessibility import describe

KEYBOARD_SHORTCUTS = """\
Getting around
  Tab and Shift+Tab   Move between the controls of the window.
  F6                  Move to the next panel.
  Alt                 Open the menu bar.

Files
  Ctrl+O              Choose the audio folder.
  F5                  Read the folder again.
  Up and Down         Move through the file list.
  Space               Check or clear the highlighted file.

Playback
  Ctrl+Space          Play, or pause if already playing.
  Alt+Left            Back 15 seconds.
  Alt+Right           Forward 15 seconds.
  Alt+Shift+Left      Back 2 minutes.
  Alt+Shift+Right     Forward 2 minutes.
  Alt+Ctrl+Left       Back 5 minutes.
  Alt+Ctrl+Right      Forward 5 minutes.

On the seek bar
  Left and Right      Move 5 seconds.
  Page Up, Page Down  Move 30 seconds.
  Home and End        Move to the start or the end of the recording.
"""


class KeyboardShortcutsDialog(QDialog):
    """Shows the keyboard shortcuts in a window that can be read and copied."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Keyboard shortcuts")

        self._text = QPlainTextEdit(self)
        self._text.setPlainText(KEYBOARD_SHORTCUTS)
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        describe(
            self._text,
            "Keyboard shortcuts",
            "The list of keyboard shortcuts. Use the arrow keys to read through it.",
        )

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(self._text)
        layout.addWidget(buttons)

        self.resize(560, 460)
        # The text is the point of this dialog, so it starts with the focus
        # and a screen reader lands directly on something worth reading.
        self._text.setFocus()


def show_about(parent: QWidget | None = None) -> None:
    """Show what this application is and which version is running."""
    QMessageBox.about(
        parent,
        f"About {APPLICATION_NAME}",
        f"{APPLICATION_NAME} version {__version__}.\n\n"
        "An accessible player for reviewing recordings. Transcription across "
        "several services is added in a later version.",
    )
