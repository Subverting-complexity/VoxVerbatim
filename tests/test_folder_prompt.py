"""A folder's own smoothing prompt, set from the review window."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QLabel, QPushButton

from tests.test_review_window import open_window
from vox_verbatim.transcription.project import ProjectStore
from vox_verbatim.ui import review_window as review_window_module
from vox_verbatim.ui.folder_prompt_dialog import FolderPromptDialog


def _said(monkeypatch) -> list[str]:
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    return said


def test_saving_a_folder_prompt_keeps_it_with_the_folder_and_says_so(
    qapp, tmp_path, monkeypatch
):
    said = _said(monkeypatch)
    window = open_window(tmp_path)
    try:
        window.set_smoothing_prompt("Skryf in Afrikaans.\n")

        assert ProjectStore(tmp_path).load().settings.smoothing_prompt == "Skryf in Afrikaans.\n"
        assert said[-1] == "This folder's smoothing prompt is saved."
    finally:
        window.close()


def test_going_back_to_the_apps_prompt_removes_the_folder_prompt(
    qapp, tmp_path, monkeypatch
):
    said = _said(monkeypatch)
    window = open_window(tmp_path)
    try:
        window.set_smoothing_prompt("Skryf in Afrikaans.")
        window.set_smoothing_prompt("")

        assert ProjectStore(tmp_path).load().settings.smoothing_prompt == ""
        assert "uses the app's prompt" in said[-1]
    finally:
        window.close()


def test_the_editor_is_on_the_project_menu_and_opens_with_the_folders_prompt(
    qapp, tmp_path, monkeypatch
):
    opened: list[FolderPromptDialog] = []

    def fake_exec(dialog):
        opened.append(dialog)
        dialog.editor.setPlainText("Hou dit kort.")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(FolderPromptDialog, "exec", fake_exec)
    _said(monkeypatch)
    window = open_window(tmp_path)
    try:
        window.set_smoothing_prompt("Skryf in Afrikaans.")
        window._smoothing_prompt_action.trigger()

        assert len(opened) == 1
        assert ProjectStore(tmp_path).load().settings.smoothing_prompt == "Hou dit kort."
    finally:
        window.close()


def test_the_dialog_starts_from_the_apps_prompt_and_can_go_back_to_it(qapp):
    dialog = FolderPromptDialog("Family", "The app's prompt.", "Mine.")
    try:
        assert dialog.editor.toPlainText() == "Mine."
        dialog.start_from_app_prompt()
        assert dialog.editor.toPlainText() == "The app's prompt."

        dialog.use_app_prompt()
        assert dialog.result() == QDialog.DialogCode.Accepted
        assert dialog.chosen_prompt() == ""
    finally:
        dialog.close()


def test_the_dialog_is_labelled_and_a_blank_box_means_the_apps_prompt(qapp):
    dialog = FolderPromptDialog("Family", "The app's prompt.")
    try:
        labels = [label for label in dialog.findChildren(QLabel) if label.buddy() is dialog.editor]
        assert len(labels) == 1
        assert dialog.editor.accessibleName() == "Folder prompt"
        assert dialog.editor.tabChangesFocus()
        names = {button.accessibleName() for button in dialog.findChildren(QPushButton)}
        assert {"Start from the app's prompt", "Use the app's prompt"} <= names

        dialog.editor.setPlainText("   \n")
        assert dialog.chosen_prompt() == ""
    finally:
        dialog.close()
