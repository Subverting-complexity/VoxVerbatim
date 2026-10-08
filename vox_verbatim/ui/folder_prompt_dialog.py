"""The editor for one folder's own smoothing prompt.

A folder may smooth its transcripts with a style prompt of its own instead
of the app's prompt from Settings, for example a folder of family interviews
in Afrikaans. This dialog edits that prompt. The review window opens it and
saves the answer with the folder's other review settings.

The dialog gives one of two answers when it is accepted: a prompt of the
folder's own, or none, which means the folder goes back to the app's prompt.
A prompt saved with no real text in it counts as none, because sending the
model an empty style prompt is never what a person meant.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vox_verbatim.ui.accessibility import announce, describe

#: Said when the editor is filled with the app's prompt.
APP_PROMPT_COPIED = "The app's prompt is now in the box, ready to edit."


class FolderPromptDialog(QDialog):
    """Edit, start from the app's prompt, or go back to the app's prompt."""

    def __init__(
        self,
        folder_name: str,
        app_prompt: str,
        folder_prompt: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._app_prompt = app_prompt
        self._use_app_prompt = False
        self.setWindowTitle(f"Smoothing Prompt for {folder_name}")

        layout = QVBoxLayout(self)

        # Says which prompt is in use now, in words, so the state is never
        # carried by an empty box alone.
        if folder_prompt.strip():
            state = "This folder has its own smoothing prompt."
        else:
            state = (
                "This folder uses the app's smoothing prompt from Settings. "
                "Write a prompt here to give it its own."
            )
        self._state_label = QLabel(
            f"{state} The format rules the app needs are added by the app and "
            "are not part of this prompt. Saving an empty box goes back to the "
            "app's prompt.",
            self,
        )
        self._state_label.setWordWrap(True)
        layout.addWidget(self._state_label)

        prompt_label = QLabel("Folder &prompt:", self)
        layout.addWidget(prompt_label)
        self._editor = QPlainTextEdit(self)
        self._editor.setPlainText(folder_prompt)
        # Tab leaves the box rather than typing a tab character, so the
        # dialog can be left from the keyboard without a trap.
        self._editor.setTabChangesFocus(True)
        prompt_label.setBuddy(self._editor)
        describe(
            self._editor,
            "Folder prompt",
            "The style prompt this folder's transcripts are smoothed with.",
        )
        layout.addWidget(self._editor, 1)

        self._start_button = QPushButton("&Start from the App's Prompt", self)
        describe(
            self._start_button,
            "Start from the app's prompt",
            "Replace the text in the box with the app's prompt, to edit it.",
        )
        self._start_button.clicked.connect(self.start_from_app_prompt)
        layout.addWidget(self._start_button)

        self._use_app_button = QPushButton("&Use the App's Prompt", self)
        describe(
            self._use_app_button,
            "Use the app's prompt",
            "Remove this folder's own prompt and close, so the app's prompt is used.",
        )
        self._use_app_button.clicked.connect(self.use_app_prompt)
        layout.addWidget(self._use_app_button)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.resize(640, 420)
        self._editor.setFocus()

    @property
    def editor(self) -> QPlainTextEdit:
        return self._editor

    def start_from_app_prompt(self) -> None:
        """Put the app's prompt in the box, to be edited into the folder's own."""
        self._editor.setPlainText(self._app_prompt)
        self._editor.setFocus()
        announce(self._editor, APP_PROMPT_COPIED)

    def use_app_prompt(self) -> None:
        """Remove the folder's own prompt and close the dialog."""
        self._use_app_prompt = True
        self.accept()

    def chosen_prompt(self) -> str:
        """The folder's own prompt after an accepted dialog, or empty for none."""
        if self._use_app_prompt:
            return ""
        text = self._editor.toPlainText()
        return text if text.strip() else ""
