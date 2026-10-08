"""Tests for exporting the reviewed recording from the review window.

The review window borrows the main window's export, so these open a real
main window for its settings and transcript folders, and a review window
whose transcripts are real files beside fake recordings. The dialog is
replaced where a test only needs its answer.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication, QDialog, QMenu

from vox_verbatim.session import SessionStore
from vox_verbatim.settings import ExportSettings
from vox_verbatim.transcription.folder_export import ExportKind
from vox_verbatim.transcription.project import ProjectStore
from vox_verbatim.transcription.store import TranscriptStore
from vox_verbatim.ui import export_flow as export_flow_module
from vox_verbatim.ui.export_dialog import ExportDialog
from vox_verbatim.ui.export_flow import ExportFlow
from vox_verbatim.ui.main_window import MainWindow
from vox_verbatim.ui.review_window import ReviewWindow

from tests.conftest import write_fake_audio
from tests.test_review_window import (
    FakePlayer,
    interactive_widgets,
    select_word,
    two_file_folder,
)


class FakeDialog:
    """Stands in for the export dialog, accepted with the given choices.

    ``while_open`` runs where the real dialog would be on screen, so a test
    can do what a person's dialog would do to the window behind it.
    """

    def __init__(self, settings: ExportSettings, while_open=None, remake: bool = False) -> None:
        self.settings = settings
        self.while_open = while_open
        self.remake = remake
        self.calls: list[dict] = []

    def __call__(self, count, settings, parent, **options):
        self.calls.append({"count": count, "parent": parent, **options})
        return self

    def exec(self):
        if self.while_open is not None:
            self.while_open()
        return QDialog.DialogCode.Accepted

    def chosen_settings(self):
        return self.settings

    def chosen_kinds(self):
        kinds = []
        if self.settings.transcript:
            kinds.append(ExportKind.TRANSCRIPT)
        if self.settings.smooth_transcript:
            kinds.append(ExportKind.SMOOTH_TRANSCRIPT)
        if self.settings.review_report:
            kinds.append(ExportKind.REVIEW_REPORT)
        return kinds

    def chosen_remake(self):
        return self.remake

    def deleteLater(self):
        pass


@pytest.fixture
def out(tmp_path):
    folder = tmp_path / "out"
    folder.mkdir()
    return folder


@pytest.fixture
def main(qapp, tmp_path):
    window = MainWindow(SessionStore(tmp_path / "session.json"))
    yield window
    window._smooth_runner.wait(5)
    window.close()
    window.deleteLater()


def open_review(tmp_path, main: MainWindow, export: bool = True) -> ReviewWindow:
    """A review window on two recordings whose transcripts are real files.

    Corrections are saved to the transcript files, as the main window's
    reader saves them, so an export reads what the window last saved.
    """
    audio = tmp_path / "audio"
    audio.mkdir(exist_ok=True)
    folder = two_file_folder()
    paths = {}
    for name, transcript in folder.transcripts.items():
        path = audio / name
        write_fake_audio(path)
        TranscriptStore(path).save(transcript)
        paths[name] = path
    window = ReviewWindow(
        audio,
        folder.names,
        lambda name: TranscriptStore(paths[name]).load(),
        paths,
        ProjectStore(audio),
        FakePlayer(),
        lambda name, transcript: TranscriptStore(paths[name]).save(transcript),
        transcript_store_for=lambda name: TranscriptStore(paths[name]),
        smooth_runner=main._smooth_runner,
        export_recordings=main.export_recordings if export else None,
    )
    window.show()
    window.process_low_confidence_words()
    return window


def test_export_is_on_the_project_menu_with_the_main_windows_key(qapp, tmp_path, main):
    review = open_review(tmp_path, main)
    try:
        assert review._export_action.text() == "&Export Transcripts..."
        assert review._export_action.shortcut() == QKeySequence("Ctrl+Shift+E")
        project_menu = review.menuBar().findChildren(QMenu)[0]
        assert review._export_action in project_menu.actions()
        keys = [
            action.shortcut().toString()
            for menu in review.menuBar().findChildren(QMenu)
            for action in menu.actions()
            if not action.shortcut().isEmpty()
        ]
        assert len(keys) == len(set(keys)), "two commands share a key"
    finally:
        review.close()
        review.deleteLater()


def test_a_correction_just_made_is_in_the_exported_transcript(
    qapp, tmp_path, main, out, monkeypatch
):
    review = open_review(tmp_path, main)
    fake = FakeDialog(ExportSettings(folder=str(out), smooth_transcript=False))
    monkeypatch.setattr(export_flow_module, "ExportDialog", fake)
    summaries = []
    monkeypatch.setattr(
        ExportFlow, "_show_export_summary", lambda _flow, m: summaries.append(m) or False
    )
    try:
        select_word(review, "Bosch")
        review._replacement_edit.setText("Bausch")
        assert review.apply_replacement_to_word() is True
        occurrence = review.current_occurrence()
        assert occurrence is not None
        name = occurrence.recording_name

        review._export_action.trigger()

        # Only the recording of the selected occurrence, and the dialog says so.
        assert fake.calls[0]["count"] == 1
        assert fake.calls[0]["parent"] is review
        assert fake.calls[0]["scope_text"] == f"{name} will be exported."
        stem = name.rsplit(".", 1)[0]
        assert sorted(path.name for path in out.iterdir()) == [f"{stem} - transcript.txt"]
        assert "Bausch" in (out / f"{stem} - transcript.txt").read_text(encoding="utf-8")
        assert summaries and summaries[0].startswith("Exported 1 file")
        # Said in the review window, where the person is working, and the
        # choices are the main window's, so both windows remember them.
        assert review._status_label.text().startswith("Exported 1 file")
        assert main._settings.export == fake.settings
    finally:
        review.close()
        review.deleteLater()


def test_the_dialog_opens_in_front_of_the_review_window_and_cancel_says_so(
    qapp, tmp_path, main
):
    """Qt gives the focus back to the parent window when a modal child closes.

    The offscreen platform has no window activation, so what is checked is
    what makes Qt return it: the dialog is a modal child of the review window.
    """
    review = open_review(tmp_path, main)
    seen = []

    def close_the_dialog():
        dialog = QApplication.activeModalWidget()
        seen.append((dialog, dialog.parent(), dialog.isModal(), dialog._count_label.text()))
        dialog.reject()

    try:
        select_word(review, "Bosch")
        name = review.current_occurrence().recording_name
        QTimer.singleShot(50, close_the_dialog)
        review.export_transcripts()

        dialog, parent, modal, scope = seen[0]
        assert isinstance(dialog, ExportDialog)
        assert parent is review and modal
        assert scope == f"{name} will be exported."
        assert "closed without exporting" in review._status_label.text()
    finally:
        review.close()
        review.deleteLater()


def test_the_focus_goes_back_where_it_was_when_the_export_ends(
    qapp, tmp_path, main, out, monkeypatch
):
    review = open_review(tmp_path, main)
    try:
        select_word(review, "Bosch")
        review._groups.setFocus()
        assert review.focusWidget() is review._groups
        elsewhere = next(
            widget
            for widget in interactive_widgets(review)
            if widget.isVisible() and widget is not review._groups
        )

        # A dialog, and the boxes after it, can leave the focus somewhere else
        # in the window behind them; this stands in for that.
        fake = FakeDialog(
            ExportSettings(folder=str(out), smooth_transcript=False),
            while_open=elsewhere.setFocus,
        )
        monkeypatch.setattr(export_flow_module, "ExportDialog", fake)
        monkeypatch.setattr(ExportFlow, "_show_export_summary", lambda _flow, _m: False)

        review.export_transcripts()

        assert review.focusWidget() is review._groups
    finally:
        review.close()
        review.deleteLater()


def test_with_no_word_chosen_or_no_export_given_it_says_why(qapp, tmp_path, main):
    review = open_review(tmp_path, main, export=False)
    try:
        select_word(review, "Bosch")
        review.export_transcripts()
        assert "cannot be exported from this window" in review._status_label.text()
    finally:
        review.close()
        review.deleteLater()
