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

from vox_verbatim import APPLICATION_NAME, __version__
from vox_verbatim.settings import Settings
from vox_verbatim.ui.accessibility import describe
from vox_verbatim.ui.player_panel import skip_button_name

def keyboard_shortcuts_text(settings: Settings | None = None) -> str:
    """The list of shortcuts, with the skip distances the user has chosen.

    The six skip commands move as far as the settings say, so the list is
    written out from the settings rather than fixed, and never tells the
    user something the buttons no longer do.

    This text is not a documentation nicety. It is how somebody working by
    ear finds out what the application can do, so a line that describes
    behaviour the application no longer has is a defect of the same kind as a
    button that does the wrong thing. Two things here are worth guarding when
    this list is next edited.

    The review window's two lists are the thing most easily described wrongly.
    ``F3`` moves between the *occurrences* of one word, which is to say
    between moments in different recordings, and ``Ctrl+F3`` moves between
    words. Saying it the other way round -- which this list did say, before
    the review became a thing done to a whole folder -- leaves somebody
    pressing ``F3`` eight times, hearing the same surname eight times, and
    concluding the window is stuck.

    And the word is "word", not "group". The menus say "Next Word", the first
    list is named "Word Groups", and a screen reader user hears "word", so
    help text calling the same thing a group sends them looking for a control
    that is not there.
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
  Ctrl+R              Review the whole folder, not just the highlighted
                      file. The same word is often said in several
                      recordings, so a review covers all of them at once.
  Ctrl+Shift+M        Make the highlighted recording's smooth transcript
                      again, with every correction made so far.
  Ctrl+Shift+O        Open the highlighted recording's smooth transcript.

In the Enhance Audio and Transcribe dialogs
  Tab                 Move between the settings. The panel at the bottom
                      explains whichever one you are on.
  F1                  Explain every setting, in one window.

In the review window
  The review window holds two lists. The first is the words that need you.
  The second is the occurrences of the word you are on, which is every
  moment that word was said, in every recording in the folder. Deciding a
  word once settles all of its occurrences.

  The window opens in a simple form: the words, the choices for the word
  you are on, and playback. Each choice button keeps the word or uses what
  one service heard. Type a different word in the box and press Enter to
  use it instead.

  Ctrl+D              Show or hide the details: the occurrences, the
                      corrections, the review settings and the notes.

  Ctrl+F3             Go to the next word.
  Ctrl+Shift+F3       Go to the previous word.
  F3 and Shift+F3     Go to the next and previous occurrence of that word.
  F2                  Move to the Replacement box for the word.
  F4                  Confirm this one occurrence as correct.
  Ctrl+I              Isolate this occurrence, which takes it out of its
                      word and keeps it out when the folder is looked at
                      again.
  F5                  Play the word alone. In this window F5 plays a word;
                      in the file list it reads the folder again.
  Shift+F5            Play the word with a second and a half either side.
  Ctrl+F5             Play the word with twelve seconds either side.
  Ctrl+L              Process the low confidence words of the whole folder.
  Ctrl+G              Group the words again, using the minimum confidence
                      and the grouping tolerance as they now stand.
  Ctrl+Shift+M        Make the smooth transcript of the selected
                      occurrence's recording again, with your corrections.
  Ctrl+Shift+O        Open the smooth transcript of that recording.
  F6 and Shift+F6     Move to the next and previous panel.
  Ctrl+Shift+Left     Give the lists more of the window.
  Ctrl+Shift+Right    Give the details more of the window.
  Everything else is on the menu bar: applying the replacement to the word or
  to this occurrence only, marking a word correct as detected, applying the
  speaker, confirming or rejecting the timing, setting the minimum
  confidence, the grouping tolerance and the wait before playing, playing
  automatically, and showing reviewed words and other uncertainties.

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
        "An accessible application that turns folders of audio recordings into "
        "transcripts. It can make quiet recordings louder, transcribes each "
        "recording with several speech-to-text services at the same time, "
        "combines their answers, and lets you review the words they were "
        "unsure of.",
    )
