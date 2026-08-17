"""Tests that the window holds the session together across restarts.

The later tests here cover the commands the window offers on the files in the
list. The dialogs they open are replaced wherever what is being tested is the
wiring rather than the dialog, and the transcription pipeline is never run:
these tests must not call a paid service or open a socket.
"""

from __future__ import annotations

import pytest
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QDialog

from audio_transcriber.session import SessionStore
from audio_transcriber.transcription.calibration import ProviderStatistics
from audio_transcriber.transcription.model import FinalToken, ReviewStatus, Transcript
from audio_transcriber.transcription.runner import RecordingOutcome, RunSummary
from audio_transcriber.transcription.store import TranscriptStore
from audio_transcriber.transcription.vocabulary import (
    Vocabulary,
    VocabularyLevel,
    VocabularyProfile,
)
from audio_transcriber.ui import main_window as main_window_module
from audio_transcriber.ui.main_window import MainWindow
from audio_transcriber.ui.settings_dialog import SettingsDialog

from tests.conftest import wait_until, write_fake_audio


@pytest.fixture
def audio_folder(tmp_path):
    folder = tmp_path / "recordings"
    folder.mkdir()
    for name in ("alpha.m4a", "beta.m4a", "gamma.m4a"):
        write_fake_audio(folder / name)
    return folder


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "session.json")


def open_window(qapp, store, expected_files: int | None = None) -> MainWindow:
    window = MainWindow(store)
    if expected_files is not None:
        assert wait_until(qapp, lambda: window._model.rowCount() == expected_files)
    return window


def close_window(window: MainWindow) -> None:
    window.close()
    window.deleteLater()


def test_choosing_a_folder_lists_its_audio_files(qapp, store, audio_folder):
    window = open_window(qapp, store)
    try:
        window._folder_panel.folderChosen.emit(str(audio_folder))

        assert wait_until(qapp, lambda: window._model.rowCount() == 3)
        assert [f.name for f in window._model.files()] == [
            "alpha.m4a",
            "beta.m4a",
            "gamma.m4a",
        ]
        # The first file is highlighted so playback has something to work with.
        assert window._table.selected_row() == 0
    finally:
        close_window(window)


def test_the_folder_checked_files_and_highlight_survive_a_restart(qapp, store, audio_folder):
    window = open_window(qapp, store)
    try:
        window._folder_panel.folderChosen.emit(str(audio_folder))
        assert wait_until(qapp, lambda: window._model.rowCount() == 3)
        window._model.set_checked(0, True)
        window._model.set_checked(2, True)
        window._table.select_row(1)
    finally:
        close_window(window)

    restored = open_window(qapp, store, expected_files=3)
    try:
        assert restored._folder == audio_folder
        assert restored._model.checked_names() == ["alpha.m4a", "gamma.m4a"]
        assert restored._table.selected_row() == 1
    finally:
        close_window(restored)


def test_files_deleted_between_runs_are_forgotten(qapp, store, audio_folder):
    window = open_window(qapp, store)
    try:
        window._folder_panel.folderChosen.emit(str(audio_folder))
        assert wait_until(qapp, lambda: window._model.rowCount() == 3)
        window._model.set_checked_names(["alpha.m4a", "beta.m4a"])
        window._table.select_row(1)
    finally:
        close_window(window)

    (audio_folder / "beta.m4a").unlink()

    restored = open_window(qapp, store, expected_files=2)
    try:
        assert restored._model.checked_names() == ["alpha.m4a"]
        # The highlighted file is gone, so the first remaining file takes over.
        assert restored._table.selected_row() == 0
    finally:
        close_window(restored)


def test_a_folder_that_has_gone_away_is_reported_and_does_not_stop_the_window(
    qapp, store, audio_folder
):
    window = open_window(qapp, store)
    try:
        window._folder_panel.folderChosen.emit(str(audio_folder))
        assert wait_until(qapp, lambda: window._model.rowCount() == 3)
    finally:
        close_window(window)

    for child in audio_folder.iterdir():
        child.unlink()
    audio_folder.rmdir()

    restored = MainWindow(store)
    try:
        assert "not available" in restored._status_label.text()
        assert restored._model.rowCount() == 0
    finally:
        close_window(restored)


def test_an_empty_folder_says_so(qapp, store, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    window = open_window(qapp, store)
    try:
        window._folder_panel.folderChosen.emit(str(empty))

        assert wait_until(qapp, lambda: "No audio files were found" in window._status_label.text())
        assert window._model.rowCount() == 0
        assert window._table.selected_row() == -1
    finally:
        close_window(window)


def test_playing_straight_after_moving_the_highlight_uses_the_new_file(
    qapp, store, audio_folder
):
    """A command given before the new file has loaded must still act on it.

    Loading is held back for a moment after the highlight moves, so that
    arrowing through a long list does not open every file on the way. A
    command given inside that moment has to bring the load forward, or it
    would act on the file the user has just moved away from.
    """
    window = open_window(qapp, store)
    try:
        window._folder_panel.folderChosen.emit(str(audio_folder))
        assert wait_until(qapp, lambda: window._model.rowCount() == 3)

        # No events are processed between these two lines, so the delay
        # before loading has not run out when Play is pressed.
        window._table.select_row(2)
        window.play()

        assert window._player.source_path == audio_folder / "gamma.m4a"
    finally:
        close_window(window)


def test_moving_the_highlight_stops_what_was_playing(qapp, store, audio_folder):
    window = open_window(qapp, store)
    try:
        window._folder_panel.folderChosen.emit(str(audio_folder))
        assert wait_until(qapp, lambda: window._model.rowCount() == 3)
        window.play()

        window._table.select_row(1)

        assert not window._player.is_playing
    finally:
        close_window(window)


def test_the_summary_counts_files_and_checked_files(qapp, store, audio_folder):
    window = open_window(qapp, store)
    try:
        window._folder_panel.folderChosen.emit(str(audio_folder))
        assert wait_until(qapp, lambda: window._model.rowCount() == 3)

        assert window._summary_label.text() == "3 files, 0 checked"

        window._model.set_checked(0, True)

        assert window._summary_label.text() == "3 files, 1 checked"
    finally:
        close_window(window)


# -- Choosing what a command works on ------------------------------------


def loaded_window(qapp, store, audio_folder) -> MainWindow:
    """A window with the sample folder already read."""
    window = open_window(qapp, store)
    window._folder_panel.folderChosen.emit(str(audio_folder))
    assert wait_until(qapp, lambda: window._model.rowCount() == 3)
    return window


def test_a_command_works_on_the_checked_files_when_there_are_any(
    qapp, store, audio_folder
):
    window = loaded_window(qapp, store, audio_folder)
    try:
        window._table.select_row(0)
        window._model.set_checked(1, True)
        window._model.set_checked(2, True)

        assert [f.name for f in window.files_to_transcribe()] == ["beta.m4a", "gamma.m4a"]
        # Every command answers this the same way, which is the point of it.
        assert window.files_to_transcribe() == window.files_to_enhance()
    finally:
        close_window(window)


def test_a_command_works_on_the_highlighted_file_when_nothing_is_checked(
    qapp, store, audio_folder
):
    window = loaded_window(qapp, store, audio_folder)
    try:
        window._table.select_row(2)

        assert [f.name for f in window.files_to_transcribe()] == ["gamma.m4a"]
    finally:
        close_window(window)


# -- Transcribing --------------------------------------------------------


def fake_transcribe_dialog(monkeypatch, run_summary=None, review=None) -> list:
    """Put a stand-in in the Transcribe dialog's place, and collect what it got."""
    opened: list = []

    class Fake:
        def __init__(self, recordings, settings, vocabulary=None, parent=None):
            self.recordings = list(recordings)
            self.settings = settings
            self.vocabulary = vocabulary
            self.summary = run_summary
            self.review_request = review
            opened.append(self)

        def exec(self) -> int:
            return QDialog.DialogCode.Rejected

        def deleteLater(self) -> None:
            pass

    monkeypatch.setattr(main_window_module, "TranscribeDialog", Fake)
    return opened


def make_transcript(name: str, needing_review: int = 1) -> Transcript:
    transcript = Transcript(recording_name=name)
    for index in range(3):
        token = FinalToken(text=f"word{index}")
        if index < needing_review:
            token.review_status = ReviewStatus.PENDING
        transcript.tokens.append(token)
    return transcript


def test_transcribe_is_offered_as_a_button_a_menu_entry_and_a_shortcut(qapp, store):
    window = open_window(qapp, store)
    try:
        assert window._transcribe_button.text() == "&Transcribe..."
        assert window._transcribe_button.accessibleName() == "Transcribe"
        assert window._transcribe_action.shortcut() == QKeySequence("Ctrl+T")
        assert window._review_action.shortcut() == QKeySequence("Ctrl+R")
        # The shortcuts belong to the window, so they work wherever the focus
        # happens to be, including inside the file list.
        assert window._transcribe_action in window.actions()
    finally:
        close_window(window)


def test_transcribing_opens_the_dialog_on_the_chosen_recordings(
    qapp, monkeypatch, store, audio_folder
):
    opened = fake_transcribe_dialog(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        window._model.set_checked(0, True)
        window.show_transcribe()

        assert len(opened) == 1
        assert [f.name for f in opened[0].recordings] == ["alpha.m4a"]
        assert opened[0].settings is window.settings.transcription
    finally:
        close_window(window)


def test_transcribing_hands_the_saved_vocabulary_to_the_dialog(
    qapp, monkeypatch, store, audio_folder
):
    vocabulary = Vocabulary(
        profiles=[VocabularyProfile(id="acme", level=VocabularyLevel.CLIENT, name="Acme")]
    )
    opened = fake_transcribe_dialog(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        window._vocabulary_store.save(vocabulary)

        window.show_transcribe()

        assert [profile.id for profile in opened[0].vocabulary.profiles] == ["acme"]
    finally:
        close_window(window)


def test_transcribing_nothing_says_so_rather_than_opening_an_empty_dialog(
    qapp, monkeypatch, store
):
    opened = fake_transcribe_dialog(monkeypatch)
    window = open_window(qapp, store)
    try:
        window.show_transcribe()

        assert opened == []
        assert "nothing to transcribe" in window._status_label.text()
    finally:
        close_window(window)


def test_how_the_run_went_is_reported_in_the_status_bar(
    qapp, monkeypatch, store, audio_folder
):
    summary = RunSummary(
        results=[
            RecordingOutcome(
                path=audio_folder / "alpha.m4a",
                transcript=make_transcript("alpha.m4a", needing_review=2),
            )
        ]
    )
    fake_transcribe_dialog(monkeypatch, run_summary=summary)
    window = loaded_window(qapp, store, audio_folder)
    try:
        window.show_transcribe()

        assert "1 of 1 recordings transcribed" in window._status_label.text()
        assert "2 words to review" in window._status_label.text()
    finally:
        close_window(window)


def test_asking_to_review_from_the_dialog_opens_the_review_window(
    qapp, monkeypatch, store, audio_folder
):
    """The dialog is modal, so the review window is opened once it has gone."""
    transcript = make_transcript("alpha.m4a")
    review = RecordingOutcome(path=audio_folder / "alpha.m4a", transcript=transcript)
    fake_transcribe_dialog(
        monkeypatch, run_summary=RunSummary(results=[review]), review=review
    )
    window = loaded_window(qapp, store, audio_folder)
    try:
        window.show_transcribe()

        assert window._review_window is not None
        assert window._review_window._transcript is transcript
    finally:
        close_window(window)


# -- Reviewing a transcript that is already there ------------------------


def save_transcript(window: MainWindow, recording, transcript: Transcript) -> TranscriptStore:
    store = window._transcript_store(recording)
    assert store.save(transcript)
    return store


def test_reviewing_a_recording_that_has_a_transcript_opens_it(qapp, store, audio_folder):
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "beta.m4a", make_transcript("beta.m4a"))
        window._table.select_row(1)

        window.show_review()

        assert window._review_window is not None
        assert window._review_window._transcript.recording_name == "beta.m4a"
        assert "Reviewing beta.m4a" in window._status_label.text()
    finally:
        close_window(window)


def test_reviewing_a_recording_with_no_transcript_says_so(qapp, store, audio_folder):
    window = loaded_window(qapp, store, audio_folder)
    try:
        window._table.select_row(0)

        window.show_review()

        assert window._review_window is None
        assert "has not been transcribed yet" in window._status_label.text()
    finally:
        close_window(window)


def test_only_one_review_window_is_kept_open(qapp, store, audio_folder):
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", make_transcript("alpha.m4a"))
        save_transcript(window, audio_folder / "beta.m4a", make_transcript("beta.m4a"))

        window._table.select_row(0)
        window.show_review()
        first = window._review_window
        window._table.select_row(1)
        window.show_review()

        assert window._review_window is not first
        assert window._review_window._transcript.recording_name == "beta.m4a"
    finally:
        close_window(window)


def test_a_correction_that_could_not_be_saved_is_reported(qapp, store, audio_folder):
    """The one failure here that must never pass quietly."""
    window = loaded_window(qapp, store, audio_folder)
    try:
        transcript = make_transcript("alpha.m4a")
        transcript_store = save_transcript(window, audio_folder / "alpha.m4a", transcript)

        class RefusingStore:
            transcript_path = transcript_store.transcript_path

            def save(self, _transcript) -> bool:
                return False

        window._save_correction(RefusingStore(), transcript)

        assert "could not be saved" in window._status_label.text()
    finally:
        close_window(window)


# -- The vocabulary and the settings dialog ------------------------------


def fake_settings_dialog(monkeypatch, accepted=True, returning=None) -> list:
    """Put a stand-in in the Settings dialog's place, and collect what it got."""
    opened: list = []

    class Fake:
        def __init__(self, settings, parent=None, vocabulary=None, statistics=None):
            self.settings = settings
            self.vocabulary = vocabulary
            self.statistics = statistics
            opened.append(self)

        def exec(self) -> int:
            if accepted:
                return QDialog.DialogCode.Accepted
            return QDialog.DialogCode.Rejected

        def chosen_settings(self):
            return self.settings

        def chosen_vocabulary(self) -> Vocabulary:
            return returning if returning is not None else self.vocabulary

    monkeypatch.setattr(main_window_module, "SettingsDialog", Fake)
    return opened


def test_the_settings_dialog_is_given_the_saved_vocabulary_and_statistics(
    qapp, monkeypatch, store
):
    opened = fake_settings_dialog(monkeypatch)
    window = open_window(qapp, store)
    try:
        window._vocabulary_store.save(
            Vocabulary(profiles=[VocabularyProfile(id="acme", name="Acme")])
        )

        window.show_settings()

        assert [profile.id for profile in opened[0].vocabulary.profiles] == ["acme"]
        assert isinstance(opened[0].statistics, ProviderStatistics)
    finally:
        close_window(window)


def test_what_the_settings_dialog_returns_is_written_to_the_vocabulary_file(
    qapp, monkeypatch, store
):
    """The loose end this fixes: the page used to edit a copy nobody kept."""
    edited = Vocabulary(
        profiles=[VocabularyProfile(id="rollout", level=VocabularyLevel.PROJECT, name="Rollout")]
    )
    fake_settings_dialog(monkeypatch, returning=edited)
    window = open_window(qapp, store)
    try:
        window.show_settings()

        saved = window._vocabulary_store.load()
        assert [profile.id for profile in saved.profiles] == ["rollout"]
    finally:
        close_window(window)


def test_cancelling_the_settings_dialog_leaves_the_vocabulary_alone(
    qapp, monkeypatch, store
):
    fake_settings_dialog(monkeypatch, accepted=False, returning=Vocabulary())
    window = open_window(qapp, store)
    try:
        window._vocabulary_store.save(
            Vocabulary(profiles=[VocabularyProfile(id="acme", name="Acme")])
        )

        window.show_settings()

        assert [profile.id for profile in window._vocabulary_store.load().profiles] == ["acme"]
    finally:
        close_window(window)


def test_the_vocabulary_survives_a_trip_through_the_real_settings_dialog(
    qapp, monkeypatch, store
):
    """End to end, with the real dialog, because both halves have to agree.

    The window loads the vocabulary, hands it in, takes back what the dialog
    says it now is, and writes that out. If the dialog were still editing a
    blank copy of its own, this would write an empty file over a good one.
    """
    monkeypatch.setattr(SettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window = open_window(qapp, store)
    try:
        window._vocabulary_store.save(
            Vocabulary(
                profiles=[
                    VocabularyProfile(id="acme", level=VocabularyLevel.CLIENT, name="Acme")
                ]
            )
        )

        window.show_settings()

        saved = window._vocabulary_store.load()
        assert [profile.id for profile in saved.profiles] == ["acme"]
        assert saved.profiles[0].name == "Acme"
    finally:
        close_window(window)
