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
from audio_transcriber.settings import Settings
from audio_transcriber.ui.accessibility import describe
from audio_transcriber.ui.player_panel import skip_button_name

def keyboard_shortcuts_text(settings: Settings | None = None) -> str:
    """The list of shortcuts, with the skip distances the user has chosen.

    The six skip commands move as far as the settings say, so the list is
    written out from the settings rather than fixed, and never tells the
    user something the buttons no longer do.
    """
    settings = settings or Settings()
    short, medium, long = settings.skip_seconds
    return f"""\
Getting around
  Tab and Shift+Tab   Move between the controls of the window.
  F6                  Move to the next panel.
  Alt                 Open the menu bar.

Files
  Ctrl+O              Choose the audio folder.
  F5                  Read the folder again.
  Up and Down         Move through the file list.
  Space               Check or clear the highlighted file.
  Ctrl+E              Enhance the checked files, or the highlighted one
                      if none are checked.
  Ctrl+T              Transcribe the checked files, or the highlighted one
                      if none are checked.
  Ctrl+R              Review the transcript of the highlighted file.

In the Enhance Audio and Transcribe dialogs
  Tab                 Move between the settings. The panel at the bottom
                      explains whichever one you are on.
  F1                  Explain every setting, in one window.

In the review window
  F2                  Correct the text of the highlighted word.
  F3 and Shift+F3     Move to the next and previous word needing review.
  F4                  Confirm the highlighted word as correct.
  F5 and Shift+F5     Play the audio around it, and play a wider stretch.
  F6                  Move to the next panel.

Settings and closing
  Ctrl+comma          Open the Settings dialog.
  Ctrl+Q              Close the application.
  The File menu also opens the log file, and the folder that holds the
  settings, the session and the log.

Playback
  Ctrl+Space          Play, or pause if already playing.
  Alt+Left            {skip_button_name(-short)}.
  Alt+Right           {skip_button_name(short)}.
  Alt+Shift+Left      {skip_button_name(-medium)}.
  Alt+Shift+Right     {skip_button_name(medium)}.
  Alt+Ctrl+Left       {skip_button_name(-long)}.
  Alt+Ctrl+Right      {skip_button_name(long)}.

On the seek bar
  Left and Right      Move 5 seconds.
  Page Up, Page Down  Move 30 seconds.
  Home and End        Move to the start or the end of the recording.
"""


class KeyboardShortcutsDialog(QDialog):
    """Shows the keyboard shortcuts in a window that can be read and copied."""

    def __init__(self, settings: Settings | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Keyboard shortcuts")

        self._text = QPlainTextEdit(self)
        self._text.setPlainText(keyboard_shortcuts_text(settings))
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
