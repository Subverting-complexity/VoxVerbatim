"""Tests for exporting transcripts to a folder the person chooses."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication, QDialog

from vox_verbatim.session import SessionStore
from vox_verbatim.settings import ExportSettings, Settings, SettingsStore
from vox_verbatim.transcription.exports import SMOOTH_EXPORT_NAME, TEXT_EXPORT_NAME
from vox_verbatim.transcription.folder_export import (
    ExportKind,
    export_file_name,
    plan_export,
    run_export,
    summary_text,
)
from vox_verbatim.transcription.model import FinalToken, Transcript
from vox_verbatim.transcription.store import TranscriptStore
from vox_verbatim.ui import export_flow as export_flow_module
from vox_verbatim.ui.export_dialog import NO_FOLDER_CHOSEN, NO_KIND_CHOSEN, ExportDialog
from vox_verbatim.ui.export_flow import ExportFlow
from vox_verbatim.ui.main_window import MainWindow

from tests.conftest import wait_until, write_fake_audio

ALL_KINDS = list(ExportKind)


def _transcript(name: str, *words: str) -> Transcript:
    return Transcript(recording_name=name, tokens=[FinalToken(text=word) for word in words])


@pytest.fixture
def recordings(tmp_path):
    folder = tmp_path / "recordings"
    folder.mkdir()
    paths = []
    for name in ("alpha.m4a", "beta.m4a"):
        write_fake_audio(folder / name)
        paths.append(folder / name)
    return paths


@pytest.fixture
def out(tmp_path):
    folder = tmp_path / "out"
    folder.mkdir()
    return folder


def _transcribe(path, *words, smooth: bool = True) -> TranscriptStore:
    store = TranscriptStore(path)
    store.save(_transcript(path.name, *words))
    if smooth:
        store.write_export(SMOOTH_EXPORT_NAME, f"Smooth {path.name}\n")
    return store


def test_the_files_are_named_after_the_recording_without_its_extension(tmp_path):
    recording = tmp_path / "Interview 3.m4a"

    assert export_file_name(recording, ExportKind.TRANSCRIPT) == "Interview 3 - transcript.txt"
    assert (
        export_file_name(recording, ExportKind.SMOOTH_TRANSCRIPT)
        == "Interview 3 - smooth transcript.txt"
    )
    assert export_file_name(recording, ExportKind.REVIEW_REPORT) == "Interview 3 - review report.md"


def test_two_recordings_with_all_three_kinds_give_six_files(recordings, out):
    for path in recordings:
        _transcribe(path, "hello", "there")

    plan = plan_export(recordings, out, ALL_KINDS, TranscriptStore)
    result = run_export(plan, TranscriptStore, replace_existing=True)

    assert sorted(path.name for path in out.iterdir()) == [
        "alpha - review report.md",
        "alpha - smooth transcript.txt",
        "alpha - transcript.txt",
        "beta - review report.md",
        "beta - smooth transcript.txt",
        "beta - transcript.txt",
    ]
    assert len(result.written) == 6
    assert result.failed == [] and result.skipped == []


def test_a_corrected_word_reaches_the_export_and_the_recordings_own_copy(recordings, out):
    store = _transcribe(recordings[0], "Mister", "Smyth")
    # The review window saves a correction to the transcript, and the old
    # transcript.txt still has the word as it was heard.
    store.write_export(TEXT_EXPORT_NAME, "Mister Smyth\n")
    corrected = store.load()
    corrected.tokens[1].text = "Smith"
    store.save(corrected)

    plan = plan_export(recordings[:1], out, [ExportKind.TRANSCRIPT], TranscriptStore)
    run_export(plan, TranscriptStore, replace_existing=True)

    assert "Mister Smith" in (out / "alpha - transcript.txt").read_text(encoding="utf-8")
    own = (store.exports_folder / TEXT_EXPORT_NAME).read_text(encoding="utf-8")
    assert "Mister Smith" in own


def test_existing_files_are_listed_and_replace_or_skip_do_what_they_say(recordings, out):
    _transcribe(recordings[0], "new", "words")
    target = out / "alpha - transcript.txt"
    target.write_text("old\n", encoding="utf-8")
    plan = plan_export(recordings[:1], out, [ExportKind.TRANSCRIPT], TranscriptStore)

    assert plan.existing() == [target]

    skipped = run_export(plan, TranscriptStore, replace_existing=False)
    assert target.read_text(encoding="utf-8") == "old\n"
    assert skipped.kept == [target] and skipped.written == []

    replaced = run_export(plan, TranscriptStore, replace_existing=True)
    assert "new words" in target.read_text(encoding="utf-8")
    assert replaced.written == [target]


def test_a_recording_with_no_smooth_transcript_exports_the_rest_and_is_named(recordings, out):
    _transcribe(recordings[0], "one")
    _transcribe(recordings[1], "two", smooth=False)

    plan = plan_export(recordings, out, ALL_KINDS, TranscriptStore)
    result = run_export(plan, TranscriptStore, replace_existing=True)

    assert len(result.written) == 5
    assert (out / "beta - transcript.txt").is_file()
    message = summary_text(result)
    assert message.startswith(f"Exported 5 files to {out}.")
    assert "Skipped: beta.m4a: it has no smooth transcript." in message


def test_two_recordings_that_share_a_name_are_both_skipped_rather_than_overwritten(
    recordings, out
):
    _transcribe(recordings[0], "original")
    copy = recordings[0].with_suffix(".wav")
    write_fake_audio(copy)
    _transcribe(copy, "enhanced")

    plan = plan_export([recordings[0], copy, recordings[1]], out, ALL_KINDS, TranscriptStore)
    result = run_export(plan, TranscriptStore, replace_existing=True)

    assert not (out / "alpha - transcript.txt").exists()
    message = summary_text(result)
    assert "alpha.m4a: it would export to the same file names as alpha.wav" in message
    assert "alpha.wav: it would export to the same file names as alpha.m4a" in message


def test_a_recording_never_transcribed_is_skipped_with_its_reason(recordings, out):
    plan = plan_export(recordings[:1], out, [ExportKind.TRANSCRIPT], TranscriptStore)
    result = run_export(plan, TranscriptStore, replace_existing=True)

    assert result.written == []
    assert (
        "alpha.m4a: it has not been transcribed, so it has no transcript."
        in summary_text(result)
    )


def test_the_export_settings_have_their_defaults_and_survive_a_save(tmp_path):
    assert ExportSettings() == ExportSettings(
        folder="", transcript=True, smooth_transcript=True, review_report=False
    )
    store = SettingsStore(tmp_path / "settings.json")
    settings = Settings()
    settings.export = ExportSettings(folder="C:/Out", review_report=True, transcript=False)
    store.save(settings)

    assert store.load().export == settings.export


# -- The dialog --------------------------------------------------------------


def test_the_dialog_will_not_export_with_no_kind_or_no_folder(qapp, out):
    dialog = ExportDialog(2, ExportSettings(folder=""))
    try:
        assert "2 recordings will be exported" in dialog._count_label.text()
        dialog.accept()
        assert dialog.result() != QDialog.DialogCode.Accepted
        assert dialog._problem_label.text() == NO_FOLDER_CHOSEN

        for box in (dialog._transcript_box, dialog._smooth_box, dialog._report_box):
            box.setChecked(False)
        dialog._folder_edit.setText(str(out))
        dialog.accept()
        assert dialog._problem_label.text() == NO_KIND_CHOSEN

        dialog._report_box.setChecked(True)
        dialog.accept()
        assert dialog.result() == QDialog.DialogCode.Accepted
        assert dialog.chosen_kinds() == [ExportKind.REVIEW_REPORT]
    finally:
        dialog.deleteLater()


def _next_tab_stop(widget):
    """The control Tab moves to from ``widget``, skipping labels and boxes."""
    candidate = widget.nextInFocusChain()
    while not candidate.focusPolicy() & Qt.FocusPolicy.TabFocus:
        candidate = candidate.nextInFocusChain()
    return candidate


def test_every_dialog_control_has_a_name_and_the_tab_order_is_logical(qapp):
    dialog = ExportDialog(1, ExportSettings())
    try:
        controls = [
            dialog._transcript_box,
            dialog._smooth_box,
            dialog._report_box,
            dialog._folder_edit,
            dialog._browse_button,
        ]
        assert all(control.accessibleName() for control in controls)
        for before, after in zip(controls, controls[1:]):
            assert _next_tab_stop(before) is after
    finally:
        dialog.deleteLater()


def test_browse_fills_in_the_folder(qapp, out, monkeypatch):
    dialog = ExportDialog(1, ExportSettings())
    try:
        monkeypatch.setattr(dialog, "_ask_for_folder", lambda _start: str(out))
        dialog._browse_button.click()
        assert dialog.chosen_folder() == out
    finally:
        dialog.deleteLater()


# -- The main window ---------------------------------------------------------


@pytest.fixture
def window(qapp, tmp_path, recordings):
    main = MainWindow(SessionStore(tmp_path / "session.json"))
    main._folder_panel.folderChosen.emit(str(recordings[0].parent))
    assert wait_until(qapp, lambda: main._model.rowCount() == 2)
    yield main
    main.close()
    main.deleteLater()


class _AcceptingDialog:
    """Stands in for the dialog, accepted with the given choices."""

    def __init__(self, settings: ExportSettings, accepted: bool = True) -> None:
        self.settings = settings
        self.accepted = accepted

    def __call__(self, count, settings, parent, **_options):
        self.count = count
        return self

    def exec(self):
        return QDialog.DialogCode.Accepted if self.accepted else QDialog.DialogCode.Rejected

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
        return False

    def deleteLater(self):
        pass


def test_export_is_in_the_file_menu_with_its_key(window):
    assert window._export_action.text() == "Ex&port Transcripts..."
    assert window._export_action.shortcut() == QKeySequence("Ctrl+Shift+E")
    assert window._export_action in window.actions()


def test_exporting_two_checked_recordings_writes_six_files(window, recordings, out, monkeypatch):
    for path in recordings:
        _transcribe(path, "hello")
    window._model.set_checked(0, True)
    window._model.set_checked(1, True)
    fake = _AcceptingDialog(ExportSettings(folder=str(out), review_report=True))
    monkeypatch.setattr(export_flow_module, "ExportDialog", fake)
    summaries = []
    monkeypatch.setattr(
        ExportFlow, "_show_export_summary", lambda _flow, m: summaries.append(m) or False
    )

    window.show_export()

    assert fake.count == 2
    assert len(list(out.iterdir())) == 6
    assert summaries and summaries[0].startswith("Exported 6 files")
    # The choices are remembered for next time.
    assert window._settings.export == fake.settings


def test_files_already_there_are_asked_about_once_and_cancel_writes_nothing(
    window, recordings, out, monkeypatch
):
    for path in recordings:
        _transcribe(path, "hello")
    window._model.set_checked(0, True)
    window._model.set_checked(1, True)
    (out / "alpha - transcript.txt").write_text("old\n", encoding="utf-8")
    (out / "beta - transcript.txt").write_text("old\n", encoding="utf-8")
    monkeypatch.setattr(
        export_flow_module, "ExportDialog", _AcceptingDialog(ExportSettings(folder=str(out)))
    )
    asked = []
    monkeypatch.setattr(
        ExportFlow,
        "_ask_about_existing_exports",
        lambda _flow, paths, _f: asked.append(paths) or "cancel",
    )

    window.show_export()

    assert len(asked) == 1 and len(asked[0]) == 2
    assert sorted(path.name for path in out.iterdir()) == [
        "alpha - transcript.txt",
        "beta - transcript.txt",
    ]
    assert "cancelled" in window._status_label.text()


def test_the_dialog_is_a_modal_child_of_the_main_window(qapp, window):
    """Qt gives the focus back to the parent window when a modal child closes.

    The offscreen test platform has no window activation, so the focus
    itself cannot be watched here; what makes Qt return it can.
    """
    window._table.select_row(0)
    seen = []

    def close_the_dialog():
        dialog = QApplication.activeModalWidget()
        seen.append((dialog, dialog.parent(), dialog.isModal()))
        dialog.reject()

    QTimer.singleShot(50, close_the_dialog)
    window.show_export()

    dialog, parent, modal = seen[0]
    assert isinstance(dialog, ExportDialog)
    assert parent is window and modal
    assert "closed without exporting" in window._status_label.text()


def test_the_summarys_button_opens_the_folder(window, recordings, out, monkeypatch):
    _transcribe(recordings[0], "hello")
    window._table.select_row(0)
    monkeypatch.setattr(
        export_flow_module,
        "ExportDialog",
        _AcceptingDialog(ExportSettings(folder=str(out), smooth_transcript=False)),
    )
    monkeypatch.setattr(ExportFlow, "_show_export_summary", lambda _flow, _m: True)
    opened = []
    monkeypatch.setattr(ExportFlow, "_open_folder", lambda _flow, path: opened.append(path))

    window.show_export()

    assert opened == [out]
