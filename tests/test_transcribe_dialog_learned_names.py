"""Tests for the list of names a folder has learned, in the Transcribe dialog.

The list is read from the folder's project file on disk, and Remove takes one
name out of that file as it is now. These tests write the project file the
way the review window does, open the dialog on the same folder, and check
the file afterwards.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from vox_verbatim import settings as settings_module
from vox_verbatim.audio.library import AudioFile
from vox_verbatim.settings import TranscriptionSettings
from vox_verbatim.transcription.learning import names_learned
from vox_verbatim.transcription.model import FinalToken, Language, Provider, Transcript
from vox_verbatim.transcription.project import (
    PROJECT_FILE_NAME,
    LearnedName,
    ProjectState,
    ProjectStore,
    read_learned_names,
)
from vox_verbatim.ui import transcribe_dialog as transcribe_dialog_module
from vox_verbatim.ui.transcribe_dialog import TranscribeDialog


@pytest.fixture(autouse=True)
def every_library_loads(monkeypatch) -> None:
    monkeypatch.setattr(settings_module, "_probe_library", lambda module, attribute: None)


@pytest.fixture
def announced(monkeypatch) -> list[tuple[str, bool]]:
    """What the dialog asked a screen reader to say, and whether it was urgent."""
    spoken: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        transcribe_dialog_module,
        "announce",
        lambda widget, message, urgent=False: spoken.append((message, urgent)),
    )
    return spoken


@pytest.fixture
def recordings(tmp_path) -> list[AudioFile]:
    return [AudioFile(path=tmp_path / "alpha.m4a", size_bytes=1000, duration_seconds=60.0)]


def learned(text: str, wrong: str, language: str = "en") -> LearnedName:
    return LearnedName(text=text, language=language, wrong_forms=[wrong],
                       learned_at="2026-10-07T09:00:00")


def teach(folder, *names: LearnedName) -> None:
    """Save names into the folder's project file, as the review window does."""
    assert ProjectStore(folder).save_keeping_learned_names(ProjectState(), list(names))


def names_on_disk(folder) -> list[str]:
    names, _status = read_learned_names(folder)
    return [name.text for name in names]


def open_dialog(recordings) -> TranscribeDialog:
    return TranscribeDialog(recordings, TranscriptionSettings())


def rows(dialog: TranscribeDialog) -> list[str]:
    return [dialog._learned_list.item(row).text() for row in range(dialog._learned_list.count())]


def select(dialog: TranscribeDialog, row: int) -> None:
    dialog._learned_list.setCurrentRow(row)


# -- What the list shows -------------------------------------------------


def test_each_learned_name_is_listed_with_its_language_and_how_it_was_heard(
    qapp, tmp_path, recordings
):
    teach(tmp_path, learned("Bosch", "Bosh"), learned("Vermeulen", "Fermeulen", "af"))
    dialog = open_dialog(recordings)
    try:
        assert rows(dialog) == [
            "Bosch (English), also heard as Bosh",
            "Vermeulen (Afrikaans), also heard as Fermeulen",
        ]
    finally:
        dialog.close()


def test_the_list_is_named_and_its_label_points_at_it(qapp, tmp_path, recordings):
    dialog = open_dialog(recordings)
    try:
        assert dialog._learned_list.accessibleName() == "Names learned in this folder"
        assert dialog._learned_list.accessibleDescription()
        labels = [
            label for label in dialog.findChildren(transcribe_dialog_module.QLabel)
            if label.buddy() is dialog._learned_list
        ]
        assert [label.text() for label in labels] == ["Names &learned in this folder"]
        assert dialog._remove_name_button.accessibleName() == "Remove name"
    finally:
        dialog.close()


def test_a_folder_with_no_learned_names_says_so(qapp, tmp_path, recordings):
    dialog = open_dialog(recordings)
    try:
        assert rows(dialog) == ["No names have been learned in this folder yet."]
        select(dialog, 0)
        assert dialog._remove_name_button.isEnabled() is False
    finally:
        dialog.close()


def test_a_project_file_that_cannot_be_read_is_said_plainly(qapp, tmp_path, recordings):
    (tmp_path / PROJECT_FILE_NAME).write_text("{ not json", encoding="utf-8")
    dialog = open_dialog(recordings)
    try:
        assert rows(dialog) == [
            "The project file in this folder could not be read, so its learned names "
            "cannot be shown."
        ]
    finally:
        dialog.close()


def test_remove_is_off_until_a_name_is_selected(qapp, tmp_path, recordings):
    teach(tmp_path, learned("Bosch", "Bosh"))
    dialog = open_dialog(recordings)
    try:
        assert dialog._remove_name_button.isEnabled() is False
        select(dialog, 0)
        assert dialog._remove_name_button.isEnabled() is True
    finally:
        dialog.close()


# -- Removing a name -----------------------------------------------------


def test_remove_takes_the_name_out_of_the_file_and_the_list(
    qapp, tmp_path, recordings, announced
):
    teach(tmp_path, learned("Bosch", "Bosh"), learned("Vermeulen", "Fermeulen"))
    dialog = open_dialog(recordings)
    dialog.show()
    try:
        select(dialog, 0)
        QTest.mouseClick(dialog._remove_name_button, Qt.MouseButton.LeftButton)
        qapp.processEvents()

        assert names_on_disk(tmp_path) == ["Vermeulen"]
        assert rows(dialog) == ["Vermeulen (English), also heard as Fermeulen"]
        # The focus lands on the name that took the removed one's place, so a
        # screen reader reads where the person now is.
        assert dialog._learned_list.hasFocus()
        assert dialog._learned_list.currentRow() == 0
        assert dialog._learned_list.currentItem().isSelected()
        assert announced == [
            ("Bosch removed. It will be learned again if you correct it in a review.", False)
        ]
        assert "Bosch removed" in dialog._learned_status.text()
        # The removed name no longer counts towards the cost estimate.
        assert [found.text for found in dialog._folder_terms] == ["Vermeulen"]
    finally:
        dialog.close()


def test_the_delete_key_removes_the_selected_name(qapp, tmp_path, recordings, announced):
    teach(tmp_path, learned("Bosch", "Bosh"), learned("Vermeulen", "Fermeulen"))
    dialog = open_dialog(recordings)
    dialog.show()
    try:
        dialog._learned_list.setFocus()
        select(dialog, 1)
        QTest.keyClick(dialog._learned_list, Qt.Key.Key_Delete)
        qapp.processEvents()

        assert names_on_disk(tmp_path) == ["Bosch"]
        # The last row went, so the focus moves up to the one before it.
        assert dialog._learned_list.currentRow() == 0
        assert dialog._learned_list.hasFocus()
        assert announced[-1][0].startswith("Vermeulen removed.")
    finally:
        dialog.close()


def test_removing_the_last_name_leaves_the_list_saying_it_is_empty(
    qapp, tmp_path, recordings, announced
):
    teach(tmp_path, learned("Bosch", "Bosh"))
    dialog = open_dialog(recordings)
    dialog.show()
    try:
        select(dialog, 0)
        dialog._remove_name_button.click()
        qapp.processEvents()

        assert names_on_disk(tmp_path) == []
        assert rows(dialog) == ["No names have been learned in this folder yet."]
        assert dialog._remove_name_button.isEnabled() is False
        # Remove is now off, so the focus must not be left on it.
        assert dialog._learned_list.hasFocus()
    finally:
        dialog.close()


def test_a_removal_that_cannot_be_saved_is_announced_urgently(
    qapp, tmp_path, recordings, announced
):
    teach(tmp_path, learned("Bosch", "Bosh"))
    dialog = open_dialog(recordings)
    try:
        (tmp_path / PROJECT_FILE_NAME).write_text("{ damaged", encoding="utf-8")
        select(dialog, 0)

        assert dialog.remove_selected_name() is False

        assert rows(dialog) == ["Bosch (English), also heard as Bosh"]
        assert announced[-1][1] is True
        assert "could not be removed" in announced[-1][0]
        assert (tmp_path / PROJECT_FILE_NAME).read_text(encoding="utf-8") == "{ damaged"
    finally:
        dialog.close()


def test_a_name_learned_after_the_dialog_opened_survives_a_removal(
    qapp, tmp_path, recordings, announced
):
    """The review window may be saving on the same folder while the dialog is open."""
    teach(tmp_path, learned("Bosch", "Bosh"))
    dialog = open_dialog(recordings)
    try:
        teach(tmp_path, learned("Vermeulen", "Fermeulen"))
        select(dialog, 0)

        assert dialog.remove_selected_name() is True

        assert names_on_disk(tmp_path) == ["Vermeulen"]
        assert rows(dialog) == ["Vermeulen (English), also heard as Fermeulen"]
    finally:
        dialog.close()


def test_a_removed_name_is_learned_again_by_a_later_correction(
    qapp, tmp_path, recordings, announced
):
    teach(tmp_path, learned("Vermeulen", "Fermeulen"))
    dialog = open_dialog(recordings)
    try:
        select(dialog, 0)
        assert dialog.remove_selected_name() is True
        assert names_on_disk(tmp_path) == []

        tokens = [
            FinalToken(id=f"token-{number}", text=text, text_source=Provider.OPENAI,
                       speaker="speaker_0", speaker_source=Provider.OPENAI,
                       language=Language.ENGLISH)
            for number, text in enumerate(["We", "met", "Fermeulen", "yesterday."])
        ]
        before = Transcript(recording_name="meeting", tokens=tokens)
        after = before.with_correction("token-2", text="Vermeulen")
        # The review window still holds a project with the name in it. Saving
        # must not bring it back on its own; the new correction does.
        stale = ProjectState(learned_names=[learned("Vermeulen", "Fermeulen")])
        ProjectStore(tmp_path).save_keeping_learned_names(stale, [])
        assert names_on_disk(tmp_path) == []

        ProjectStore(tmp_path).save_keeping_learned_names(stale, names_learned(before, after))

        assert names_on_disk(tmp_path) == ["Vermeulen"]
    finally:
        dialog.close()
