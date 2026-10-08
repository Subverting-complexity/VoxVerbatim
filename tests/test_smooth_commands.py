"""Tests for making the smooth transcript again on request, and opening it.

A fake engine stands in for the language model, so no request leaves the
machine. It writes the words it was given, which is enough to show that a
correction reaches the smooth file.
"""

from __future__ import annotations

import threading

import pytest
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QMenu

from vox_verbatim.session import SessionStore
from vox_verbatim.transcription.project import ProjectStore
from vox_verbatim.transcription import smoothing
from vox_verbatim.transcription.exports import SMOOTH_EXPORT_NAME
from vox_verbatim.transcription.model import FinalToken, Transcript
from vox_verbatim.transcription.smoothing import SmoothOutcome
from vox_verbatim.transcription.store import TranscriptStore
from vox_verbatim.ui import smooth_commands
from vox_verbatim.ui.main_window import MainWindow
from vox_verbatim.ui.review_window import ReviewWindow
from vox_verbatim.ui.smooth_commands import (
    SmoothRunner,
    make_smooth_transcript,
    open_smooth_transcript,
    smooth_path,
)

from tests.conftest import wait_until, write_fake_audio
from tests.test_review_window import FakePlayer, select_word, two_file_folder


class FakeSmoother:
    """Answers with the words it was given, or with an error."""

    def __init__(self, error: str | None = None) -> None:
        self.error = error
        self.seen: list[str] = []

    def smooth(self, transcript: Transcript) -> SmoothOutcome:
        words = " ".join(token.text for token in transcript.tokens)
        self.seen.append(words)
        if self.error:
            return SmoothOutcome(error=self.error)
        return SmoothOutcome(text=f"Edited.\n\n{words}\n")


def _transcript(*words: str) -> Transcript:
    return Transcript(
        recording_name="interview.wav", tokens=[FinalToken(text=word) for word in words]
    )


def _store(tmp_path) -> TranscriptStore:
    recording = tmp_path / "interview.wav"
    write_fake_audio(recording)
    return TranscriptStore(recording)


def test_a_corrected_word_reaches_the_smooth_file(tmp_path):
    store = _store(tmp_path)
    transcript = _transcript("Mister", "Smyth", "said")
    transcript.tokens[1].text = "Smith"

    result = make_smooth_transcript("interview.wav", transcript, store, FakeSmoother())

    assert result.succeeded
    assert result.message == "The smooth transcript of interview.wav is ready."
    assert "Mister Smith said" in smooth_path(store).read_text(encoding="utf-8")


def test_a_failure_removes_the_older_smooth_file_and_says_why(tmp_path):
    store = _store(tmp_path)
    store.write_export(SMOOTH_EXPORT_NAME, "an older edit\n")
    error = f"{smoothing.NOT_MADE}: no OpenAI API key has been entered."

    result = make_smooth_transcript(
        "interview.wav", _transcript("hello"), store, FakeSmoother(error=error)
    )

    assert not result.succeeded
    assert result.message.startswith(f"interview.wav: {error}")
    assert "older smooth transcript was removed" in result.message
    assert not smooth_path(store).exists()


def test_an_engine_that_raises_is_reported_rather_than_raised(tmp_path):
    class Broken:
        def smooth(self, _transcript):
            raise ValueError("boom")

    result = make_smooth_transcript("interview.wav", _transcript("hi"), _store(tmp_path), Broken())

    assert not result.succeeded
    assert "boom" in result.message


def test_opening_a_recording_with_no_smooth_file_says_so(tmp_path):
    opened, message = open_smooth_transcript("interview.wav", _store(tmp_path))

    assert not opened
    assert message.startswith("interview.wav has no smooth transcript yet.")
    assert "Ctrl+Shift+M" in message


def test_opening_an_existing_smooth_file_hands_it_to_windows(tmp_path, monkeypatch):
    store = _store(tmp_path)
    store.write_export(SMOOTH_EXPORT_NAME, "text\n")
    opened_urls = []
    monkeypatch.setattr(
        smooth_commands.QDesktopServices,
        "openUrl",
        staticmethod(lambda url: opened_urls.append(url.toLocalFile()) or True),
    )

    opened, message = open_smooth_transcript("interview.wav", store)

    assert opened
    assert message == "Opened the smooth transcript of interview.wav."
    assert opened_urls == [str(smooth_path(store)).replace("\\", "/")]


def test_the_runner_works_in_the_background_on_a_copy(qapp, tmp_path):
    store = _store(tmp_path)
    transcript = _transcript("one", "two")
    runner = SmoothRunner()
    results = []
    runner.finished.connect(lambda result, requester: results.append((result, requester)))

    assert runner.start("interview.wav", transcript, store, FakeSmoother(), requester="me")
    assert not runner.start("interview.wav", transcript, store, FakeSmoother())
    # A correction made while the request is out does not reach this run.
    transcript.tokens[0].text = "changed"

    assert wait_until(qapp, lambda: bool(results))
    assert results[0][0].succeeded and results[0][1] == "me"
    assert "one two" in smooth_path(store).read_text(encoding="utf-8")


# -- The main window -------------------------------------------------------


@pytest.fixture
def window(qapp, tmp_path):
    folder = tmp_path / "recordings"
    folder.mkdir()
    for name in ("alpha.m4a", "beta.m4a"):
        write_fake_audio(folder / name)
    main = MainWindow(SessionStore(tmp_path / "session.json"))
    main._folder_panel.folderChosen.emit(str(folder))
    assert wait_until(qapp, lambda: main._model.rowCount() == 2)
    yield main
    main._smooth_runner.wait(5)
    main.close()
    main.deleteLater()


def test_the_main_window_offers_both_commands_with_their_keys(window):
    assert window._make_smooth_action.shortcut() == QKeySequence("Ctrl+Shift+M")
    assert window._open_smooth_action.shortcut() == QKeySequence("Ctrl+Shift+O")
    assert window._make_smooth_action in window.actions()
    assert window._open_smooth_action in window.actions()
    keys = [
        action.shortcut().toString()
        for action in window.actions()
        if not action.shortcut().isEmpty()
    ]
    assert len(keys) == len(set(keys)), "two commands share a key"


def test_the_main_window_makes_the_highlighted_recordings_smooth_file(
    qapp, window, monkeypatch
):
    fake = FakeSmoother()
    monkeypatch.setattr(smoothing, "smoother_for", lambda *_args, **_kw: fake)
    window._table.select_row(1)
    path = window._model.file_at(1).path
    TranscriptStore(path).save(_transcript("corrected", "words"))

    window.make_smooth_transcript_again()

    assert "Making the smooth transcript of beta.m4a" in window._status_label.text()
    assert wait_until(qapp, lambda: "is ready" in window._status_label.text())
    assert fake.seen == ["corrected words"]


def test_the_main_window_announces_a_failure_and_removes_the_older_file(
    qapp, window, monkeypatch
):
    error = f"{smoothing.NOT_MADE}: no OpenAI API key has been entered."
    monkeypatch.setattr(
        smoothing, "smoother_for", lambda *_args, **_kw: FakeSmoother(error=error)
    )
    window._table.select_row(0)
    store = TranscriptStore(window._model.file_at(0).path)
    store.save(_transcript("words"))
    store.write_export(SMOOTH_EXPORT_NAME, "older\n")

    window.make_smooth_transcript_again()

    assert wait_until(qapp, lambda: error in window._status_label.text())
    assert not smooth_path(store).exists()


def test_the_main_window_says_when_there_is_no_transcript_to_smooth(window):
    window._table.select_row(0)

    window.make_smooth_transcript_again()

    assert "alpha.m4a has no transcript yet" in window._status_label.text()


def test_the_main_window_says_when_there_is_no_smooth_file_to_open(window):
    window._table.select_row(0)

    window.open_smooth_transcript()

    assert "alpha.m4a has no smooth transcript yet" in window._status_label.text()


# -- The review window -----------------------------------------------------


def _review_window(
    tmp_path, transcript_store_for=None, build_smoother=None, smooth_runner=None
) -> ReviewWindow:
    folder = two_file_folder()
    window = ReviewWindow(
        tmp_path,
        folder.names,
        folder.load,
        {name: tmp_path / name for name in folder.names},
        ProjectStore(tmp_path),
        FakePlayer(),
        folder.save,
        transcript_store_for=transcript_store_for,
        build_smoother=build_smoother,
        smooth_runner=smooth_runner,
    )
    window.show()
    window.process_low_confidence_words()
    return window


def test_the_review_window_offers_both_commands_with_their_keys(qapp, tmp_path):
    window = _review_window(tmp_path)
    try:
        assert window._make_smooth_action.shortcut() == QKeySequence("Ctrl+Shift+M")
        assert window._open_smooth_action.shortcut() == QKeySequence("Ctrl+Shift+O")
        keys = [
            action.shortcut().toString()
            for menu in window.menuBar().findChildren(QMenu)
            for action in menu.actions()
            if not action.shortcut().isEmpty()
        ]
        assert len(keys) == len(set(keys)), "two commands share a key"
    finally:
        window.close()
        window.deleteLater()


def test_the_review_window_smooths_the_selected_occurrences_recording(qapp, tmp_path):
    fake = FakeSmoother()
    stores: dict[str, TranscriptStore] = {}

    def store_for(name: str) -> TranscriptStore:
        return stores.setdefault(name, TranscriptStore(tmp_path / name))

    window = _review_window(tmp_path, store_for, lambda: fake)
    try:
        select_word(window, "Bosch")
        window._replacement_edit.setText("Bausch")
        assert window.apply_replacement_to_word() is True
        occurrence = window.current_occurrence()
        assert occurrence is not None

        window.make_smooth_transcript_again()

        assert wait_until(qapp, lambda: "is ready" in window._status_label.text())
        text = smooth_path(stores[occurrence.recording_name]).read_text(encoding="utf-8")
        assert "Bausch" in text
    finally:
        window._smooth_runner.wait(5)
        window.close()
        window.deleteLater()


def test_a_shared_runner_refuses_a_second_run_and_only_the_asker_announces(
    qapp, tmp_path
):
    """The main window lends its runner, so the two windows never race on one file."""
    runner = SmoothRunner()
    release = threading.Event()

    class Slow(FakeSmoother):
        def smooth(self, transcript):
            release.wait(5)
            return super().smooth(transcript)

    first = _review_window(tmp_path, lambda name: TranscriptStore(tmp_path / name), Slow, runner)
    second = _review_window(
        tmp_path, lambda name: TranscriptStore(tmp_path / name), FakeSmoother, runner
    )
    try:
        first.make_smooth_transcript_again()
        second.make_smooth_transcript_again()
        assert "already being made" in second._status_label.text()

        release.set()
        assert wait_until(qapp, lambda: "is ready" in first._status_label.text())
        assert "is ready" not in second._status_label.text()
    finally:
        release.set()
        runner.wait(5)
        for window in (first, second):
            window.close()
            window.deleteLater()


def test_the_review_window_says_when_there_is_no_smooth_file_to_open(qapp, tmp_path):
    window = _review_window(tmp_path, lambda name: TranscriptStore(tmp_path / name))
    try:
        window.open_smooth_transcript()

        assert "has no smooth transcript yet" in window._status_label.text()
    finally:
        window.close()
        window.deleteLater()
