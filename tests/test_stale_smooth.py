"""Tests for saying when a smooth transcript is older than the corrections.

A smooth transcript keeps a fingerprint of the literal text it was made
from. A correction changes that text, and the smooth copy is then out of
date: the export dialog names it and offers to make it again first, and
Open Smooth Transcript says so. A fake engine stands in for the language
model, so no request leaves the machine.
"""

from __future__ import annotations

import threading

import pytest
from PySide6.QtCore import Qt

from vox_verbatim.session import SessionStore
from vox_verbatim.settings import ExportSettings
from vox_verbatim.transcription import smoothing
from vox_verbatim.transcription.exports import SMOOTH_EXPORT_NAME
from vox_verbatim.transcription.folder_export import (
    ExportKind,
    out_of_date_smooth,
    plan_export,
    run_export,
    summary_text,
)
from vox_verbatim.transcription.model import FinalToken, Transcript
from vox_verbatim.transcription.smoothing import (
    SMOOTH_SOURCE_NAME,
    smooth_is_out_of_date,
    source_fingerprint,
    write_smooth_transcript,
)
from vox_verbatim.transcription.store import TranscriptStore
from vox_verbatim.ui import export_flow as export_flow_module
from vox_verbatim.ui import smooth_commands
from vox_verbatim.ui.export_dialog import ExportDialog
from vox_verbatim.ui.export_flow import ExportFlow
from vox_verbatim.ui.main_window import MainWindow
from vox_verbatim.ui.smooth_commands import open_smooth_transcript, smooth_path

from tests.conftest import wait_until, write_fake_audio
from tests.test_review_export import FakeDialog, open_review
from tests.test_review_window import select_word
from tests.test_smooth_commands import FakeSmoother


def _transcript(name: str, *words: str) -> Transcript:
    return Transcript(recording_name=name, tokens=[FinalToken(text=word) for word in words])


def _recording(folder, name: str = "interview.m4a"):
    folder.mkdir(exist_ok=True)
    path = folder / name
    write_fake_audio(path)
    return path


def _smoothed(path, *words) -> TranscriptStore:
    """A transcribed recording whose smooth transcript matches it."""
    store = TranscriptStore(path)
    transcript = _transcript(path.name, *words)
    store.save(transcript)
    assert write_smooth_transcript(transcript, store, FakeSmoother()).succeeded
    return store


def _correct(store: TranscriptStore, index: int, text: str) -> None:
    """Change one word and save, as the review window does."""
    transcript = store.load()
    transcript.tokens[index].text = text
    store.save(transcript)


# -- The fingerprint ---------------------------------------------------------


def test_making_the_smooth_transcript_records_what_it_was_made_from(tmp_path):
    store = _smoothed(_recording(tmp_path / "audio"), "Mister", "Smyth")

    recorded = (store.folder / SMOOTH_SOURCE_NAME).read_text(encoding="utf-8").strip()
    assert recorded == source_fingerprint(store.load())
    # Beside the transcript, not among the files a person opens and shares.
    assert not (store.exports_folder / SMOOTH_SOURCE_NAME).exists()
    assert not smooth_is_out_of_date(store, store.load())


def test_a_correction_makes_the_smooth_transcript_out_of_date(tmp_path):
    store = _smoothed(_recording(tmp_path / "audio"), "Mister", "Smyth")

    _correct(store, 1, "Smith")

    assert smooth_is_out_of_date(store, store.load())


def test_a_smooth_transcript_made_after_the_correction_is_not_out_of_date(tmp_path):
    store = _smoothed(_recording(tmp_path / "audio"), "Mister", "Smyth")
    _correct(store, 1, "Smith")

    write_smooth_transcript(store.load(), store, FakeSmoother())

    assert not smooth_is_out_of_date(store, store.load())


def test_a_smooth_transcript_from_before_fingerprints_is_out_of_date(tmp_path):
    store = TranscriptStore(_recording(tmp_path / "audio"))
    store.save(_transcript("interview.m4a", "hello"))
    store.write_export(SMOOTH_EXPORT_NAME, "Hello.\n")

    assert smooth_is_out_of_date(store, store.load())


def test_no_smooth_transcript_is_not_out_of_date(tmp_path):
    store = TranscriptStore(_recording(tmp_path / "audio"))
    store.save(_transcript("interview.m4a", "hello"))

    assert not smooth_is_out_of_date(store, store.load())


def test_the_fingerprint_is_of_the_words_sent_while_the_request_was_out(tmp_path):
    """A correction saved while the model works is not in that smooth copy."""
    store = TranscriptStore(_recording(tmp_path / "audio"))
    sent = _transcript("interview.m4a", "Mister", "Smyth")
    store.save(sent)

    class CorrectsMeanwhile(FakeSmoother):
        def smooth(self, transcript):
            _correct(store, 1, "Smith")
            return super().smooth(transcript)

    write_smooth_transcript(sent, store, CorrectsMeanwhile())

    assert smooth_is_out_of_date(store, store.load())


# -- Planning the export -----------------------------------------------------


def test_the_out_of_date_recordings_are_listed_and_named_in_the_summary(tmp_path):
    audio = tmp_path / "audio"
    stale = _recording(audio, "alpha.m4a")
    current = _recording(audio, "beta.m4a")
    _correct(_smoothed(stale, "one", "two"), 0, "won")
    _smoothed(current, "three")
    out = tmp_path / "out"
    out.mkdir()

    assert out_of_date_smooth([stale, current], TranscriptStore) == [stale]

    plan = plan_export([stale, current], out, list(ExportKind), TranscriptStore)
    assert plan.out_of_date == [stale]
    message = summary_text(run_export(plan, TranscriptStore, replace_existing=True))
    assert (
        "The smooth transcript of alpha.m4a was exported as it was, older than the "
        "latest corrections." in message
    )

    without_smooth = plan_export([stale], out, [ExportKind.TRANSCRIPT], TranscriptStore)
    assert without_smooth.out_of_date == []


# -- The dialog --------------------------------------------------------------


def _next_tab_stop(widget):
    candidate = widget.nextInFocusChain()
    while not candidate.focusPolicy() & Qt.FocusPolicy.TabFocus:
        candidate = candidate.nextInFocusChain()
    return candidate


def test_the_dialog_names_the_out_of_date_ones_and_offers_both_choices(qapp):
    dialog = ExportDialog(
        2, ExportSettings(), out_of_date_names=["alpha.m4a", "beta.m4a"]
    )
    try:
        group = dialog._out_of_date_group
        assert group is not None and "out of date" in group.title()
        sentence = (
            "The smooth transcripts of alpha.m4a and beta.m4a were made before the "
            "latest corrections."
        )
        assert sentence in dialog._remake_radio.accessibleDescription()
        assert sentence in dialog._as_is_radio.accessibleDescription()
        assert sentence in dialog._smooth_box.accessibleDescription()
        assert dialog._remake_radio.accessibleName() == "Make them again first"
        assert dialog._as_is_radio.accessibleName() == "Export them as they are"
        # Making them again is the default, and each choice has its own key.
        assert dialog._remake_radio.isChecked() and dialog.chosen_remake()
        assert "&" in dialog._remake_radio.text() and "&" in dialog._as_is_radio.text()
        # In Tab order after the report box and before the folder.
        assert _next_tab_stop(dialog._report_box) is dialog._remake_radio
        assert _next_tab_stop(dialog._as_is_radio) is dialog._folder_edit

        dialog._as_is_radio.setChecked(True)
        assert not dialog.chosen_remake()

        dialog._remake_radio.setChecked(True)
        dialog._smooth_box.setChecked(False)
        assert not group.isEnabled()
        assert not dialog.chosen_remake()
    finally:
        dialog.deleteLater()


def test_a_dialog_with_nothing_out_of_date_asks_nothing_about_it(qapp):
    dialog = ExportDialog(1, ExportSettings())
    try:
        assert dialog._out_of_date_group is None
        assert not dialog.chosen_remake()
        assert _next_tab_stop(dialog._report_box) is dialog._folder_edit
    finally:
        dialog.deleteLater()


# -- Exporting ---------------------------------------------------------------


@pytest.fixture
def main(qapp, tmp_path):
    window = MainWindow(SessionStore(tmp_path / "session.json"))
    yield window
    window._smooth_runner.wait(5)
    window.close()
    window.deleteLater()


@pytest.fixture
def out(tmp_path):
    folder = tmp_path / "out"
    folder.mkdir()
    return folder


def _catch_summary(monkeypatch) -> list[str]:
    summaries: list[str] = []
    monkeypatch.setattr(
        ExportFlow, "_show_export_summary", lambda _flow, m: summaries.append(m) or False
    )
    return summaries


def test_a_word_corrected_in_review_is_named_and_made_again_before_exporting(
    qapp, tmp_path, main, out, monkeypatch
):
    fake = FakeSmoother()
    monkeypatch.setattr(smoothing, "smoother_for", lambda *_args, **_kw: fake)
    review = open_review(tmp_path, main)
    try:
        select_word(review, "Bosch")
        name = review.current_occurrence().recording_name
        store = TranscriptStore(review._recording_paths[name])
        # Smoothed before the correction below.
        write_smooth_transcript(store.load(), store, FakeSmoother())
        review._replacement_edit.setText("Bausch")
        assert review.apply_replacement_to_word() is True

        dialog = FakeDialog(
            ExportSettings(folder=str(out), transcript=False), remake=True
        )
        monkeypatch.setattr(export_flow_module, "ExportDialog", dialog)
        summaries = _catch_summary(monkeypatch)

        review.export_transcripts()

        assert dialog.calls[0]["out_of_date_names"] == [name]
        assert "again before exporting, 1 of 1" in review._status_label.text()
        assert wait_until(qapp, lambda: bool(summaries))
        stem = name.rsplit(".", 1)[0]
        exported = (out / f"{stem} - smooth transcript.txt").read_text(encoding="utf-8")
        assert "Bausch" in exported
        assert "older than the latest corrections" not in summaries[0]
        assert not smooth_is_out_of_date(store, store.load())
        # The export said what happened; the main window said nothing of its own.
        assert "is ready" not in main._status_label.text()
    finally:
        main._smooth_runner.wait(5)
        review.close()
        review.deleteLater()


def test_exporting_as_they_are_copies_the_old_file_and_says_so(
    qapp, tmp_path, main, out, monkeypatch
):
    store = _smoothed(_recording(tmp_path / "audio"), "Mister", "Smyth")
    _correct(store, 1, "Smith")
    dialog = FakeDialog(ExportSettings(folder=str(out), transcript=False), remake=False)
    monkeypatch.setattr(export_flow_module, "ExportDialog", dialog)
    summaries = _catch_summary(monkeypatch)

    main.export_recordings([store.recording_path])

    assert dialog.calls[0]["out_of_date_names"] == ["interview.m4a"]
    assert "Smyth" in (out / "interview - smooth transcript.txt").read_text(encoding="utf-8")
    assert "exported as it was, older than the latest corrections" in summaries[0]


def test_a_remake_that_fails_is_in_the_summary_and_its_old_file_is_not_exported(
    qapp, tmp_path, main, out, monkeypatch
):
    error = f"{smoothing.NOT_MADE}: no OpenAI API key has been entered."
    monkeypatch.setattr(
        smoothing, "smoother_for", lambda *_args, **_kw: FakeSmoother(error=error)
    )
    store = _smoothed(_recording(tmp_path / "audio"), "Mister", "Smyth")
    _correct(store, 1, "Smith")
    monkeypatch.setattr(
        export_flow_module,
        "ExportDialog",
        FakeDialog(ExportSettings(folder=str(out)), remake=True),
    )
    summaries = _catch_summary(monkeypatch)

    main.export_recordings([store.recording_path])

    assert wait_until(qapp, lambda: bool(summaries))
    assert error in summaries[0]
    assert "interview.m4a: it has no smooth transcript." in summaries[0]
    assert not (out / "interview - smooth transcript.txt").exists()
    assert (out / "interview - transcript.txt").is_file()


def test_a_busy_runner_means_nothing_is_exported_and_it_says_why(
    qapp, tmp_path, main, out, monkeypatch
):
    store = _smoothed(_recording(tmp_path / "audio"), "Mister", "Smyth")
    _correct(store, 1, "Smith")
    release = threading.Event()

    class Slow(FakeSmoother):
        def smooth(self, transcript):
            release.wait(5)
            return super().smooth(transcript)

    other = TranscriptStore(_recording(tmp_path / "audio", "other.m4a"))
    main._smooth_runner.start("other.m4a", _transcript("other.m4a", "x"), other, Slow())
    monkeypatch.setattr(
        export_flow_module,
        "ExportDialog",
        FakeDialog(ExportSettings(folder=str(out)), remake=True),
    )
    summaries = _catch_summary(monkeypatch)
    try:
        main.export_recordings([store.recording_path])

        assert "already being made" in main._status_label.text()
        assert "Nothing was exported" in main._status_label.text()
        assert summaries == [] and list(out.iterdir()) == []
    finally:
        release.set()
        main._smooth_runner.wait(5)


# -- Opening it --------------------------------------------------------------


def test_opening_an_out_of_date_smooth_transcript_says_so_and_how_to_update_it(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        smooth_commands.QDesktopServices, "openUrl", staticmethod(lambda _url: True)
    )
    store = _smoothed(_recording(tmp_path / "audio"), "Mister", "Smyth")

    opened, message = open_smooth_transcript("interview.m4a", store)
    assert opened and message == "Opened the smooth transcript of interview.m4a."

    _correct(store, 1, "Smith")
    opened, message = open_smooth_transcript("interview.m4a", store)

    assert opened
    assert "It is out of date" in message
    assert "Make Smooth Transcript Again, Ctrl+Shift+M, updates it." in message
    assert smooth_path(store).is_file()
