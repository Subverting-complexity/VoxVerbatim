"""Tests that the window holds the session together across restarts.

The later tests here cover the commands the window offers on the files in the
list. The dialogs they open are replaced wherever what is being tested is the
wiring rather than the dialog, and the transcription pipeline is never run:
these tests must not call a paid service or open a socket.
"""

from __future__ import annotations

import inspect
import os
import re
import shutil

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QAbstractButton, QApplication, QDialog, QLabel, QWidget

from vox_verbatim.audio.player import AudioPlayer
from vox_verbatim.session import SessionStore
from vox_verbatim.transcription import grouping
from vox_verbatim.transcription.calibration import CalibrationStore, ProviderStatistics
from vox_verbatim.transcription.learning import StatisticsChange
from vox_verbatim.transcription.model import (
    FinalToken,
    Provider,
    ReviewReason,
    ReviewStatus,
    StatisticsNote,
    Transcript,
)
from vox_verbatim.transcription.normalise import normalise
from vox_verbatim.transcription.project import FlaggedItem, ProjectStore, ReplacementRule
from vox_verbatim.transcription.runner import RecordingOutcome, RunSummary
from vox_verbatim.transcription.store import TranscriptStore
from vox_verbatim.transcription.vocabulary import (
    Vocabulary,
    VocabularyLevel,
    VocabularyProfile,
)
from vox_verbatim.ui import main_window as main_window_module
from vox_verbatim.ui import review_lists
from vox_verbatim.ui.main_window import MainWindow
from vox_verbatim.ui.settings_dialog import SettingsDialog

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


def mnemonic_of(text: str) -> str | None:
    """The letter after a single ``&`` in a label, or None; ``&&`` is a literal."""
    match = re.search(r"(?<!&)&([^&])", text.replace("&&", ""))
    return match.group(1).lower() if match else None


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


def test_f5_after_the_folder_comes_back_restores_the_remembered_files(
    qapp, store, audio_folder
):
    """The remembered checks wait for the drive, so F5 must not drop them."""
    window = open_window(qapp, store)
    try:
        window._folder_panel.folderChosen.emit(str(audio_folder))
        assert wait_until(qapp, lambda: window._model.rowCount() == 3)
        window._model.set_checked(0, True)
        window._model.set_checked(2, True)
        window._table.select_row(1)
    finally:
        close_window(window)

    away = audio_folder.with_name("disconnected")
    audio_folder.rename(away)
    restored = MainWindow(store)
    try:
        assert "not available" in restored._status_label.text()
        away.rename(audio_folder)

        restored.refresh()

        assert wait_until(qapp, lambda: restored._model.rowCount() == 3)
        assert restored._model.checked_names() == ["alpha.m4a", "gamma.m4a"]
        assert restored._selected_file_name() == "beta.m4a"
    finally:
        close_window(restored)


def test_f5_on_a_loaded_folder_keeps_the_current_checks(qapp, store, audio_folder):
    window = open_window(qapp, store)
    try:
        window._folder_panel.folderChosen.emit(str(audio_folder))
        assert wait_until(qapp, lambda: window._model.rowCount() == 3)
        window._model.set_checked(1, True)
        window._table.select_row(2)

        window.refresh()

        assert wait_until(qapp, lambda: window._model.rowCount() == 3)
        assert wait_until(qapp, lambda: window._file_list_is_current)
        assert window._model.checked_names() == ["beta.m4a"]
        assert window._selected_file_name() == "gamma.m4a"
    finally:
        close_window(window)


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


def fake_transcribe_dialog(
    monkeypatch, run_summary=None, review=None, still_stopping: bool = False
) -> list:
    """Put a stand-in in the Transcribe dialog's place, and collect what it got."""
    opened: list = []

    class Fake(QObject):
        detachedRunFinished = Signal(object)

        def __init__(self, recordings, settings, vocabulary=None, parent=None):
            super().__init__()
            self.recordings = list(recordings)
            self.settings = settings
            self.vocabulary = vocabulary
            self.summary = run_summary
            self.review_request = review
            self.is_stopping_in_background = still_stopping
            self.deleted = False
            opened.append(self)

        def exec(self) -> int:
            return QDialog.DialogCode.Rejected

        def deleteLater(self) -> None:
            self.deleted = True

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


def test_settings_opens_on_ctrl_comma_where_the_platform_names_no_preferences_key(qapp, store):
    """Qt gives StandardKey.Preferences no key on Windows.

    The Settings action used it, so on Windows it had no shortcut at all
    while the README and the F1 list both said to press Ctrl+comma.
    """
    window = open_window(qapp, store)
    try:
        assert window._settings_action.shortcut() == QKeySequence("Ctrl+,")
        assert window._settings_action in window.actions()
    finally:
        close_window(window)


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
            RecordingOutcome.of_transcript(
                audio_folder / "alpha.m4a",
                make_transcript("alpha.m4a", needing_review=2),
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


def test_a_dialog_closed_on_a_stopping_run_is_kept_and_the_person_is_told(
    qapp, monkeypatch, store, audio_folder
):
    """The runner inside the dialog is what the run still reports to.

    Deleting it would take the run's receiver away before the request it is
    waiting on had come back, and nobody would withdraw the request that
    keeps the machine awake. So the dialog is left to get rid of itself, and
    the status bar says plainly what is happening and that the recordings
    already done are safe.
    """
    opened = fake_transcribe_dialog(monkeypatch, still_stopping=True)
    window = loaded_window(qapp, store, audio_folder)
    try:
        window.show_transcribe()

        assert opened[0].deleted is False
        status = window._status_label.text()
        assert "still stopping" in status
        assert "already transcribed are saved" in status
    finally:
        close_window(window)


def test_a_second_run_is_refused_while_the_first_is_still_stopping(
    qapp, monkeypatch, store, audio_folder
):
    """Two runs at once would write the same transcript folders, and the first
    to finish would withdraw the request keeping the machine awake from under
    the second. So Transcribe says no, out loud, until the first has stopped.
    """
    opened = fake_transcribe_dialog(monkeypatch, still_stopping=True)
    said = announcements(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        window.show_transcribe()
        assert len(opened) == 1

        window.show_transcribe()
        assert len(opened) == 1
        assert "previous run is still stopping" in window._status_label.text()
        assert any("previous run is still stopping" in words for _w, words, _u in said)

        # The first run stops. The person is told how it went, because the
        # dialog that would have shown them is hidden, and a run may start.
        outcome = RecordingOutcome.of_transcript(
            audio_folder / "alpha.m4a", make_transcript("alpha.m4a", needing_review=0)
        )
        opened[0].is_stopping_in_background = False
        opened[0].detachedRunFinished.emit(RunSummary(results=[outcome], cancelled=True))
        status = window._status_label.text()
        assert "run that was stopping has finished" in status
        assert "1 of 1 recordings transcribed" in status
        assert said[-1][1] == status
        assert window._stopping_dialog is None

        window.show_transcribe()
        assert len(opened) == 2
    finally:
        close_window(window)


def test_asking_to_review_from_the_dialog_opens_the_project_review_there(
    qapp, monkeypatch, store, audio_folder
):
    """The dialog is modal, so the review window is opened once it has gone.

    What opens is the review of the whole folder, and the recording just
    transcribed is where the person is put down in it. A word in the new file
    is very often the same word as one already settled in an older file, so
    opening on the new file alone would hide the decision already taken.
    """
    opened = fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))
        save_transcript(window, audio_folder / "beta.m4a", weak_transcript("beta.m4a"))
        outcome = RecordingOutcome.of_transcript(
            audio_folder / "alpha.m4a", make_transcript("alpha.m4a")
        )
        fake_transcribe_dialog(
            monkeypatch, run_summary=RunSummary(results=[outcome]), review=outcome
        )

        window.show_transcribe()

        assert len(opened) == 1
        assert sorted(opened[0].recording_names) == ["alpha.m4a", "beta.m4a"]
        # Put down on the recording that was just transcribed, and it found
        # something there, so nothing is said about landing elsewhere.
        assert opened[0].selected == ["alpha.m4a"]
        assert "so the review opens where you last left it" not in window._status_label.text()
    finally:
        close_window(window)


def test_a_hand_off_to_a_recording_with_nothing_waiting_says_where_it_landed(
    qapp, monkeypatch, store, audio_folder
):
    """A recording with no words waiting is not a failure, but it is worth saying.

    Somebody who asked to review what they had just transcribed and was put
    somewhere else entirely would reasonably think the window had opened on
    the wrong thing.
    """
    opened = fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))
        # gamma has a transcript, but every word in it is settled, so the
        # project holds nothing at all for that recording.
        save_transcript(window, audio_folder / "gamma.m4a", strong_transcript("gamma.m4a"))
        outcome = RecordingOutcome.of_transcript(
            audio_folder / "gamma.m4a", make_transcript("gamma.m4a")
        )
        fake_transcribe_dialog(
            monkeypatch, run_summary=RunSummary(results=[outcome]), review=outcome
        )

        window.show_transcribe()

        assert opened[0].selected == ["gamma.m4a"]
        assert "There is nothing waiting in gamma.m4a" in window._status_label.text()
    finally:
        close_window(window)


def announcements(monkeypatch) -> list[tuple[object, str, bool]]:
    """Everything the main window asks a screen reader to read out.

    The widget is collected as well as the words, because which window an
    announcement is raised on decides whether it is heard at all: one raised
    from a window that is not in front is unreliable across screen readers.
    """
    said: list[tuple[object, str, bool]] = []

    def record(widget, message: str, urgent: bool = False) -> None:
        said.append((widget, message, urgent))

    monkeypatch.setattr(main_window_module, "announce", record)
    return said


def test_the_hand_off_says_one_thing_once_and_says_it_where_it_can_be_heard(
    qapp, monkeypatch, store, audio_folder
):
    """Two announcements used to race here, and the wrong one won.

    Landing on the recording read the row out on the review window; the
    sentence that followed was raised on the main window, which was behind it
    by then, and arrived assertively whenever a transcript could not be read,
    so it cut the row reading off. That sentence is the only place the person
    is ever told how many words were corrected on their behalf, so it may not
    be the one that loses, and it may not be the one that talks over anything
    either.
    """
    opened = fake_review_window(monkeypatch)
    said = announcements(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))
        outcome = RecordingOutcome.of_transcript(
            audio_folder / "alpha.m4a", make_transcript("alpha.m4a")
        )
        fake_transcribe_dialog(
            monkeypatch, run_summary=RunSummary(results=[outcome]), review=outcome
        )
        said.clear()

        window.show_transcribe()

        review = opened[0]
        # The window landed silently, so the sentence below is the whole of
        # what is said about the hand-over.
        assert review.announced_arrival == [False]
        opening = [item for item in said if "Reviewing " in item[1]]
        assert len(opening) == 1
        widget, message, urgent = opening[0]
        assert widget is review
        assert urgent is False
        assert "You are on the first word waiting in alpha.m4a." in message
        # And it is on the screen as well, for anybody who can look.
        assert message == window._status_label.text()
    finally:
        close_window(window)


def test_the_focus_is_placed_in_the_review_rather_than_left_to_luck(
    qapp, monkeypatch, store, audio_folder
):
    """Where a window opens must be a decision, not the widget order.

    The focus is asked for after the window is shown and before anything is
    announced, so the screen reader reads the word the person has been put on
    and the sentence follows it rather than across it.
    """
    opened = fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))

        window.open_project_review(land_on="alpha.m4a")

        assert opened[0].focused == ["Word Groups"]
    finally:
        close_window(window)


# -- Reviewing the folder ------------------------------------------------


def save_transcript(window: MainWindow, recording, transcript: Transcript) -> TranscriptStore:
    store = window._transcript_store(recording)
    assert store.save(transcript)
    return store


def load_transcript(window: MainWindow, recording) -> Transcript:
    loaded = window._transcript_store(recording).load()
    assert loaded is not None
    return loaded


def words_of(transcript: Transcript) -> list[str]:
    return [token.text for token in transcript.tokens]


def spoken_transcript(name: str, words) -> Transcript:
    """A transcript of the given words, each with a saved composite confidence.

    The strength is what the review sweep thresholds on: a word below the
    project's minimum ends up in front of a person, and a word above it is one
    the application treats as settled.
    """
    transcript = Transcript(recording_name=name)
    for index, (text, strength) in enumerate(words):
        token = FinalToken(text=text)
        token.confidence_strength = strength
        token.start = float(index)
        token.end = float(index) + 0.5
        transcript.tokens.append(token)
    return transcript


def weak_transcript(name: str, word: str = "Bosch") -> Transcript:
    """A transcript with one shaky word in it, which the review will pick up."""
    return spoken_transcript(name, [("the", 0.99), (word, 0.30), ("account", 0.99)])


def strong_transcript(name: str, word: str = "Bosch") -> Transcript:
    """A transcript the application is sure of, so nothing in it wants a person."""
    return spoken_transcript(name, [("the", 0.99), (word, 0.95), ("account", 0.99)])


def add_replacement_rule(
    folder, matched: str = "Bosch", replacement: str = "Bosche"
) -> ReplacementRule:
    """Put into a folder's project what accepting a replacement there would leave.

    The review window is what normally creates these. Writing one straight
    into the project file is how a test can say "the person already decided
    this last week" without driving the whole window to do it.
    """
    project_store = ProjectStore(folder)
    state = project_store.load()
    rule = ReplacementRule(
        id=f"rule-{matched.lower()}",
        matched_text=matched,
        normalised_text=normalise(matched),
        replacement=replacement,
        language="unknown",
        created_at="2026-01-01T09:00:00+02:00",
    )
    state.rules.append(rule)
    assert project_store.save(state)
    return rule


def fake_review_window(monkeypatch) -> list:
    """Put a stand-in in the review window's place, and collect what it was handed.

    The review window is a large window of its own with its own tests. What
    matters here is the hand-over: which folder, which recordings, which
    project and which player the main window gives it, and whether it was
    asked to land on a particular recording.

    It also records how many transcripts were actually parsed while the window
    was opened, which is the figure the whole lazy arrangement exists to keep
    small, and which no other assertion would notice going wrong.
    """
    opened: list = []

    class Fake(QWidget):
        def __init__(
            self,
            folder,
            recording_names,
            load_transcript,
            recording_paths,
            project_store,
            player,
            save_correction=None,
            parent=None,
            record_statistics=None,
            vocabulary=None,
        ):
            super().__init__(parent)
            self.record_statistics = record_statistics
            self.vocabulary = vocabulary
            self.folder = folder
            self.recording_names = list(recording_names)
            self.load_transcript = load_transcript
            self.recording_paths = recording_paths
            self.project_store = project_store
            self.player = player
            self.save_correction = save_correction
            self.selected: list[str] = []
            self.announced_arrival: list[bool] = []
            self.focused: list[str] = []
            opened.append(self)

        def select_recording(
            self, recording_name: str, announce_arrival: bool = True
        ) -> bool:
            """Answer as the real window does: is there anything here for it?"""
            self.selected.append(recording_name)
            self.announced_arrival.append(announce_arrival)
            state = self.project_store.load()
            return any(item.recording_name == recording_name for item in state.occurrences)

        def focus_word_list(self) -> None:
            self.focused.append("Word Groups")

    monkeypatch.setattr(main_window_module, "ReviewWindow", Fake)
    return opened


def beta_occurrences(folder) -> int:
    """How many words of beta.m4a the project still holds a decision about."""
    state = ProjectStore(folder).load()
    return len([item for item in state.occurrences if item.recording_name == "beta.m4a"])


def count_transcript_reads(monkeypatch) -> list[str]:
    """Record every transcript actually parsed, in the order they were parsed.

    Reading a transcript is the expensive thing this whole arrangement exists
    to avoid, and it is invisible to every other assertion: a review that
    quietly read the entire folder would look exactly like one that read
    nothing. So it is counted.
    """
    read: list[str] = []
    real = TranscriptStore.load

    def counted(self):
        read.append(self.recording_path.name)
        return real(self)

    monkeypatch.setattr(TranscriptStore, "load", counted)
    return read


def test_opening_a_folder_again_reads_no_transcripts_at_all(
    qapp, monkeypatch, store, audio_folder
):
    """The whole point of naming recordings rather than handing them over.

    A folder of fifty hour-long interviews is more than a gigabyte of
    transcripts, and pressing Ctrl+R must not mean reading it. Everything the
    two lists in the review window need is already in the project file, so a
    second opening of an unchanged folder reads nothing.
    """
    fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        for name in ("alpha.m4a", "beta.m4a", "gamma.m4a"):
            save_transcript(window, audio_folder / name, weak_transcript(name))
        window.show_review()

        read = count_transcript_reads(monkeypatch)
        window.show_review()

        assert read == []
    finally:
        close_window(window)


def flagged_transcript(name: str, flagged: str = "15,000") -> Transcript:
    """A transcript with one shaky word and one the pipeline flagged by rule.

    The two reach the review by different roads and both have to survive the
    folder being closed. The shaky word is found by the low-confidence sweep,
    which saves an occurrence for it. The flagged word is strong and the sweep
    never looks at it; a rule objected to it, and the only place that objection
    is written down is the transcript.
    """
    transcript = spoken_transcript(name, [("the", 0.99), ("Bosch", 0.30), (flagged, 0.99)])
    transcript.tokens[2].flag(ReviewReason.NUMERIC_DISAGREEMENT)
    return transcript


def flagged_rows(review) -> list[str]:
    """The words in the second block of the review window's first list."""
    return [
        row.word
        for row in review._group_model.rows()
        if row.key.startswith(review_lists.UNCERTAINTY_KEY_PREFIX)
    ]


def test_reopening_a_folder_keeps_its_flagged_words_and_reads_nothing(
    qapp, monkeypatch, store, audio_folder
):
    """The gap this closed: they used to be found only while a transcript was read.

    A folder that has not changed is not read again, which is what makes
    opening it quick. Before the flagged words were saved, that meant the first
    review of a folder showed them and every later one showed an empty second
    block, which reads as a folder with nothing wrong in it. They are now in
    the project file for the same reason the shaky words are, so a reopening
    has them all without opening a single transcript.
    """
    fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        for name in ("alpha.m4a", "beta.m4a", "gamma.m4a"):
            save_transcript(window, audio_folder / name, flagged_transcript(name))
        window.show_review()

        read = count_transcript_reads(monkeypatch)
        window.show_review()

        assert read == []
        saved = ProjectStore(audio_folder).load()
        assert sorted(item.recording_name for item in saved.flagged) == [
            "alpha.m4a",
            "beta.m4a",
            "gamma.m4a",
        ]
        assert {item.text for item in saved.flagged} == {"15,000"}
        assert saved.flagged[0].reasons == ["numeric_disagreement"]
    finally:
        close_window(window)


def test_a_recording_deleted_from_the_folder_leaves_the_review_altogether(
    qapp, monkeypatch, store, audio_folder
):
    """Reproduces the defect exactly, from the outside.

    Two recordings, both with the same shaky word, so the analysis puts them
    in one group. One of the recordings is then deleted and the folder read
    again. The occurrence of the deleted recording used to stay in the
    project for ever, because nothing pruned it and the analysis deliberately
    keeps the occurrences of a recording it could not read. The visible harm
    was the sentence the review reads out to somebody who cannot see the list:
    "applies to 2 occurrences across 2 files", one of which no longer exists.
    """
    fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        for name in ("alpha.m4a", "beta.m4a"):
            save_transcript(window, audio_folder / name, weak_transcript(name))
        window.show_review()
        first = ProjectStore(audio_folder).load()
        assert sorted({item.recording_name for item in first.occurrences}) == [
            "alpha.m4a",
            "beta.m4a",
        ]
        group = first.groups[0]
        assert "across 2 files" in grouping.affected_summary(first, group.id)

        transcripts = window._transcript_store(audio_folder / "beta.m4a").transcript_path
        shutil.rmtree(transcripts.parent)
        (audio_folder / "beta.m4a").unlink()
        window.refresh()
        assert wait_until(qapp, lambda: window._model.rowCount() == 2)
        window.show_review()

        second = ProjectStore(audio_folder).load()
        assert {item.recording_name for item in second.occurrences} == {"alpha.m4a"}
        assert second.flagged == []
        remaining = second.groups[0]
        assert grouping.affected_summary(second, remaining.id) == (
            "This replacement applies to 1 occurrence in 1 file."
        )
    finally:
        close_window(window)


def test_a_transcript_that_cannot_be_read_keeps_its_place_in_the_review(
    qapp, monkeypatch, store, audio_folder
):
    """The distinction the pruning turns on, from the other side.

    A recording whose transcript is damaged, or busy, or on a drive that is
    not connected, is still in the folder. Its words and everything decided
    about them must survive, which is the behaviour the analysis already had
    and which pruning must not undo.
    """
    fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        for name in ("alpha.m4a", "beta.m4a"):
            save_transcript(window, audio_folder / name, weak_transcript(name))
        window.show_review()

        damaged = window._transcript_store(audio_folder / "beta.m4a").transcript_path
        damaged.write_text("this is not a transcript", encoding="utf-8")
        window.show_review()

        saved = ProjectStore(audio_folder).load()
        assert sorted({item.recording_name for item in saved.occurrences}) == [
            "alpha.m4a",
            "beta.m4a",
        ]
    finally:
        close_window(window)


def test_the_second_block_is_drawn_from_the_project_rather_than_the_transcripts(
    qapp, monkeypatch, store, audio_folder
):
    """The same thing again, but looking at the list the person actually sees.

    The one transcript that is read is the one holding the word the window
    lands on, whose detail panel genuinely needs it. The other two are never
    opened, and two of the three flagged words on the list come out of them.
    """
    window = loaded_window(qapp, store, audio_folder)
    try:
        for name in ("alpha.m4a", "beta.m4a", "gamma.m4a"):
            save_transcript(window, audio_folder / name, flagged_transcript(name))
        first = window.open_project_review()
        assert first is not None
        assert flagged_rows(first) == ["15,000", "15,000", "15,000"]

        read = count_transcript_reads(monkeypatch)
        second = window.open_project_review()

        assert second is not None
        assert flagged_rows(second) == ["15,000", "15,000", "15,000"]
        assert read == ["alpha.m4a"]
        # And the block does not claim to be waiting for anything.
        assert "found when the folder is processed" not in second._count_label.text()
    finally:
        close_window(window)


def test_a_flagged_word_that_has_gone_from_a_new_transcript_is_not_still_listed(
    qapp, monkeypatch, store, audio_folder
):
    """A saved item must never outlive the word it points at.

    Transcribing a recording again gives every word a fresh identifier, so a
    saved flagged item then points at nothing. Nothing re-matches it: the
    recording is read because its transcript is newer than the analysis, and
    everything the project remembered about its flagged words is replaced by
    what it now says. A word the new transcript does not flag is therefore
    simply gone, rather than sitting in the list as a row that can be selected
    but not played, corrected or confirmed.
    """
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", flagged_transcript("alpha.m4a"))
        first = window.open_project_review()
        assert first is not None
        assert flagged_rows(first) == ["15,000"]
        gone = ProjectStore(audio_folder).load().flagged[0].token_id

        # Transcribed again. The words are the same words, but they are new
        # objects with new identifiers, and this time nothing objects to the
        # number.
        again = spoken_transcript(
            "alpha.m4a", [("the", 0.99), ("Bosch", 0.30), ("15,000", 0.99)]
        )
        save_transcript(window, audio_folder / "alpha.m4a", again)
        second = window.open_project_review()

        assert second is not None
        assert flagged_rows(second) == []
        saved = ProjectStore(audio_folder).load()
        assert [item.token_id for item in saved.flagged] == []
        assert gone not in [token.id for token in again.tokens]
    finally:
        close_window(window)


def speaker_doubt_transcript(name: str) -> Transcript:
    """Three agreed words whose speaker a second opinion doubted, after one plain word."""
    transcript = spoken_transcript(
        name, [("so", 0.99), ("that", 0.99), ("is", 0.99), ("fine", 0.99)]
    )
    transcript.tokens[0].speaker = "speaker_0"
    for token in transcript.tokens[1:]:
        token.speaker = "speaker_1"
        token.speaker_alternative = "speaker_0"
        token.flag(ReviewReason.SPEAKER_UNCERTAIN)
    return transcript


def save_in_the_old_format(folder, transcript: Transcript) -> None:
    """Make the project look as one analysed before speaker doubts had a list.

    Each doubted word was then a flagged word of its own, and no stretch was
    saved at all.
    """
    project_store = ProjectStore(folder)
    state = project_store.load()
    state.speaker_doubts = []
    state.flagged = [
        FlaggedItem(
            recording_name=transcript.recording_name,
            token_id=token.id,
            text=token.text,
            start=token.start,
            reasons=[ReviewReason.SPEAKER_UNCERTAIN.value],
        )
        for token in transcript.tokens[1:]
    ]
    assert project_store.save(state)


def test_a_project_saved_before_speaker_doubts_had_a_list_still_shows_them(
    qapp, monkeypatch, store, audio_folder
):
    """The doubted words leave the word list, so their stretch must appear.

    The transcript has not changed, so it would not be read again, and
    without a reading nothing would ever build the stretch. Beta holds the
    doubts and alpha a flagged word, so the window lands on alpha and
    reading beta for its detail panel cannot hide the fault.
    """
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", flagged_transcript("alpha.m4a"))
        doubted = speaker_doubt_transcript("beta.m4a")
        save_transcript(window, audio_folder / "beta.m4a", doubted)
        first = window.open_project_review()
        assert first is not None
        first.close()
        save_in_the_old_format(audio_folder, doubted)

        read = count_transcript_reads(monkeypatch)
        review = window.open_project_review()

        assert review is not None
        assert "beta.m4a" in read
        model = review._speaker_doubt_model
        assert model.rowCount() == 1
        assert model.items()[0].recording_name == "beta.m4a"
        assert model.items()[0].token_ids == [token.id for token in doubted.tokens[1:]]
        saved = ProjectStore(audio_folder).load()
        assert len(saved.speaker_doubts) == 1
        assert all(item.recording_name != "beta.m4a" for item in saved.flagged)

        # Once the stretch is saved, the recording is spared again.
        review.close()
        read.clear()
        window.open_project_review()
        assert "beta.m4a" not in read
    finally:
        close_window(window)


def test_a_recording_not_read_again_keeps_its_speaker_doubts(
    qapp, monkeypatch, store, audio_folder
):
    """Alpha changes and is read again; beta does not, and keeps its stretch."""
    fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        for name in ("alpha.m4a", "beta.m4a"):
            save_transcript(window, audio_folder / name, speaker_doubt_transcript(name))
        window.show_review()
        before = {
            item.recording_name: item.token_ids
            for item in ProjectStore(audio_folder).load().speaker_doubts
        }
        assert sorted(before) == ["alpha.m4a", "beta.m4a"]

        renewed = speaker_doubt_transcript("alpha.m4a")
        save_transcript(window, audio_folder / "alpha.m4a", renewed)
        read = count_transcript_reads(monkeypatch)
        window.show_review()

        assert read == ["alpha.m4a"]
        after = {
            item.recording_name: item.token_ids
            for item in ProjectStore(audio_folder).load().speaker_doubts
        }
        assert after["beta.m4a"] == before["beta.m4a"]
        assert after["alpha.m4a"] == [token.id for token in renewed.tokens[1:]]
    finally:
        close_window(window)


def test_a_recording_deleted_from_the_folder_takes_its_speaker_doubts_with_it(
    qapp, monkeypatch, store, audio_folder
):
    fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        for name in ("alpha.m4a", "beta.m4a"):
            save_transcript(window, audio_folder / name, speaker_doubt_transcript(name))
        window.show_review()

        transcripts = window._transcript_store(audio_folder / "beta.m4a").transcript_path
        shutil.rmtree(transcripts.parent)
        (audio_folder / "beta.m4a").unlink()
        window.refresh()
        assert wait_until(qapp, lambda: window._model.rowCount() == 2)
        window.show_review()

        saved = ProjectStore(audio_folder).load()
        assert [item.recording_name for item in saved.speaker_doubts] == ["alpha.m4a"]
    finally:
        close_window(window)


def test_a_transcript_written_since_the_last_review_is_read_again(
    qapp, monkeypatch, store, audio_folder
):
    """Skipping this one would be the failure the whole feature exists to prevent."""
    fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))
        window.show_review()

        read = count_transcript_reads(monkeypatch)
        # A recording transcribed after the last review, which the project has
        # never seen a single word of.
        save_transcript(window, audio_folder / "beta.m4a", weak_transcript("beta.m4a"))
        window.show_review()

        assert "beta.m4a" in read
        assert "alpha.m4a" not in read
        state = ProjectStore(audio_folder).load()
        assert sorted({item.recording_name for item in state.occurrences}) == [
            "alpha.m4a",
            "beta.m4a",
        ]
    finally:
        close_window(window)


def test_a_transcript_replaced_by_a_newer_one_is_read_again(
    qapp, monkeypatch, store, audio_folder
):
    """A recording already in the project can still have been transcribed again."""
    fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))
        window.show_review()

        read = count_transcript_reads(monkeypatch)
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a", "Bosh"))
        window.show_review()

        assert read.count("alpha.m4a") >= 1
        state = ProjectStore(audio_folder).load()
        assert [item.detected_text for item in state.occurrences] == ["Bosh"]
    finally:
        close_window(window)


def test_a_transcript_rewritten_with_the_same_time_and_size_is_read_again(
    qapp, monkeypatch, store, audio_folder
):
    """Two saves inside one tick of the Windows file clock carry the same time.

    The test above meets this case only when its two saves happen to land in
    one tick, which is what made it fail now and then. Here the time is put
    back by hand and the new word is as long as the old one, so nothing but
    the file's identity tells the two transcripts apart, on every run.
    """
    fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        transcript_path = save_transcript(
            window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a")
        ).transcript_path
        window.show_review()
        before = transcript_path.stat()

        read = count_transcript_reads(monkeypatch)
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a", "Busch"))
        os.utime(transcript_path, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert transcript_path.stat().st_size == before.st_size
        window.show_review()

        assert "alpha.m4a" in read
        state = ProjectStore(audio_folder).load()
        assert [item.detected_text for item in state.occurrences] == ["Busch"]
    finally:
        close_window(window)


def test_a_transcript_restored_from_a_backup_is_read_again_though_it_is_older(
    qapp, monkeypatch, store, audio_folder
):
    """A file put back from yesterday is a different file, not an old one.

    Whether a recording has to be read again is decided by asking whether its
    transcript is still the one that was analysed, and never by asking whether
    it is newer than the analysis. This is the case that tells the two
    questions apart in the open. A transcript copied back from a backup, or
    from another machine, carries the age it had there, which is older than
    the review that has already happened here. Asked which is newer, the
    answer says the file cannot have changed and the restored words are never
    looked at. Asked whether it is the same file, the answer is plainly no.
    """
    fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))
        window.show_review()

        read = count_transcript_reads(monkeypatch)
        transcript_path = window._transcript_store(audio_folder / "alpha.m4a").transcript_path
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a", "Bosh"))
        # Restored with the age it had wherever it was kept, which is a day
        # before this folder was ever reviewed.
        yesterday = transcript_path.stat().st_mtime - 24 * 60 * 60
        os.utime(transcript_path, (yesterday, yesterday))
        window.show_review()

        assert "alpha.m4a" in read
        state = ProjectStore(audio_folder).load()
        assert [item.detected_text for item in state.occurrences] == ["Bosh"]
    finally:
        close_window(window)


def test_a_transcript_that_could_not_be_read_is_tried_again_next_time(
    qapp, monkeypatch, store, audio_folder
):
    """Nothing has looked at it yet, so nothing may say it has been looked at.

    A recording is spared only because the project noted what its transcript
    was when it was analysed. One that could not be read was never analysed,
    so no note is kept about it, and the next opening tries it again without
    needing anything about the file to change in the meantime. Keeping the
    note from before it went bad would pass the recording over from then on,
    even once somebody had put a good transcript back in its place.
    """
    fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        for name in ("alpha.m4a", "beta.m4a"):
            save_transcript(window, audio_folder / name, weak_transcript(name))
        window.show_review()
        damaged = window._transcript_store(audio_folder / "beta.m4a").transcript_path
        damaged.write_text("this is not a transcript", encoding="utf-8")
        window.show_review()
        assert ProjectStore(audio_folder).load().transcript_times.get("beta.m4a") is None

        read = count_transcript_reads(monkeypatch)
        # Nothing about the file has changed since the opening that failed to
        # read it. It is tried again all the same.
        window.show_review()

        assert "beta.m4a" in read
        assert "alpha.m4a" not in read
    finally:
        close_window(window)


def test_the_review_window_constructor_matches_the_shared_contract(qapp):
    """The one place the two halves of this feature have to agree exactly.

    The main window and the review window are built separately against a
    written contract, so the signature is worth asserting rather than
    discovering at run time in front of a user.
    """
    from vox_verbatim.ui.review_window import ReviewWindow

    parameters = list(inspect.signature(ReviewWindow.__init__).parameters)
    assert parameters[:7] == [
        "self",
        "folder",
        "recording_names",
        "load_transcript",
        "recording_paths",
        "project_store",
        "player",
    ]
    assert "save_correction" in parameters
    assert "parent" in parameters
    assert hasattr(ReviewWindow, "select_recording")


def test_reviewing_covers_the_folder_with_no_recording_highlighted(
    qapp, monkeypatch, store, audio_folder
):
    """Which file the highlight is on decides nothing about what is reviewed."""
    opened = fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))
        save_transcript(window, audio_folder / "gamma.m4a", weak_transcript("gamma.m4a"))
        window._clear_selection()

        window.show_review()

        assert window._review_window is not None
        assert sorted(opened[0].recording_names) == ["alpha.m4a", "gamma.m4a"]
        assert opened[0].folder == audio_folder
        assert opened[0].recording_paths["alpha.m4a"] == audio_folder / "alpha.m4a"
        assert "Reviewing 2 recordings" in window._status_label.text()
    finally:
        close_window(window)


def test_opening_a_review_says_it_is_reading_before_it_starts_and_lets_the_window_breathe(
    qapp, monkeypatch, store, audio_folder
):
    """Reading a folder of long recordings is minutes on the window's thread.

    The person is told before the first transcript is parsed, not after,
    because after is when the window has been frozen for the whole wait. And
    the event loop is given a turn between recordings so that the window
    repaints and the screen reader gets to say the sentence.
    """
    fake_review_window(monkeypatch)
    said = announcements(monkeypatch)
    breaths: list[str] = []
    monkeypatch.setattr(
        main_window_module, "_let_the_window_breathe", lambda: breaths.append("breath")
    )
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))
        save_transcript(window, audio_folder / "gamma.m4a", weak_transcript("gamma.m4a"))
        said.clear()

        window.show_review()

        reading = [item for item in said if item[1].startswith("Reading ")]
        assert len(reading) == 1
        widget, message, urgent = reading[0]
        assert message == "Reading 2 transcripts in this folder. This can take a while."
        assert widget is window._status_label
        assert urgent is True
        # Said before the review's own opening sentence, which comes last.
        assert said.index(reading[0]) < len(said) - 1
        assert said[-1][1].startswith("Reviewing ")
        # One breath per transcript actually parsed.
        assert len(breaths) >= 2
        # The cursor is put back whatever happened.
        assert QApplication.overrideCursor() is None
    finally:
        close_window(window)


def test_a_second_review_asked_for_while_the_first_is_opening_is_refused(
    qapp, monkeypatch, store, audio_folder
):
    """The turn the event loop gets between transcripts is enough for a second Ctrl+R."""
    opened = fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    inner: list[object] = []

    def press_again() -> None:
        inner.append(window.open_project_review())

    monkeypatch.setattr(main_window_module, "_let_the_window_breathe", press_again)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))

        first = window.open_project_review()

        assert first is not None
        assert inner and all(item is None for item in inner)
        assert len(opened) == 1
    finally:
        close_window(window)


def test_a_folder_change_while_the_review_is_opening_is_refused(
    qapp, monkeypatch, store, audio_folder, tmp_path
):
    """A new folder arriving between two transcripts would change the folder
    under the opening, and the review would be built from one folder's
    transcripts and shown under the other's name. So the change is refused,
    said out loud, and the window opens on the folder whose transcripts it read.
    """
    opened = fake_review_window(monkeypatch)
    said = announcements(monkeypatch)
    other = tmp_path / "other"
    other.mkdir()
    write_fake_audio(other / "delta.m4a")
    window = loaded_window(qapp, store, audio_folder)
    refused: list[str] = []

    def change_folder() -> None:
        window._folder_panel.folderChosen.emit(str(other))
        refused.append(window._status_label.text())

    monkeypatch.setattr(main_window_module, "_let_the_window_breathe", change_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))

        review = window.open_project_review()

        assert review is not None
        assert refused and all("cannot be changed yet" in text for text in refused)
        assert any("cannot be changed yet" in words for _w, words, _u in said)
        assert window._folder == audio_folder
        assert opened[0].folder == audio_folder
        # Once the review is open the folder may change again.
        window._folder_panel.folderChosen.emit(str(other))
        assert window._folder == other
    finally:
        close_window(window)


def test_closing_the_window_while_a_review_is_opening_abandons_the_opening(
    qapp, monkeypatch, store, audio_folder
):
    """A review window must not appear after the main window has gone."""
    opened = fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    monkeypatch.setattr(main_window_module, "_let_the_window_breathe", window.close)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))

        review = window.open_project_review()

        assert review is None
        assert opened == []
        assert window._review_window is None
        assert window._opening_review is False
        assert QApplication.overrideCursor() is None
    finally:
        close_window(window)


def test_the_review_window_is_handed_a_reader_that_does_not_run_the_event_loop(
    qapp, monkeypatch, store, audio_folder
):
    """The turn the reader gives the event loop while the review is opening
    would, in the review window's hands, let a key press ask for a second
    recording part way through reading the first. So it is taken away before
    the window gets the reader.
    """
    opened = fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))

        window.show_review()

        reader = opened[0].load_transcript.__self__
        assert isinstance(reader, main_window_module._TranscriptReader)
        assert reader.between_recordings is None
    finally:
        close_window(window)


def test_the_reader_holds_one_transcript_and_pauses_only_before_a_real_read(
    qapp, store, audio_folder
):
    window = loaded_window(qapp, store, audio_folder)
    try:
        alpha = save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))
        gamma = save_transcript(window, audio_folder / "gamma.m4a", weak_transcript("gamma.m4a"))
        pauses: list[int] = []
        reader = main_window_module._TranscriptReader(
            {"alpha.m4a": alpha, "gamma.m4a": gamma}, between_recordings=lambda: pauses.append(1)
        )

        first = reader.load("alpha.m4a")
        assert reader.load("alpha.m4a") is first
        assert len(pauses) == 1
        reader.load("gamma.m4a")
        assert len(pauses) == 2
        # Going back is a real read again: only one is ever held.
        assert reader.load("alpha.m4a") is not first
        assert len(pauses) == 3
    finally:
        close_window(window)


def test_an_untranscribed_highlight_does_not_stop_the_review(
    qapp, monkeypatch, store, audio_folder
):
    """A file with no transcript is not reviewable content, so it is left out.

    It is emphatically not a reason to refuse. The person may well be standing
    on the one recording nobody has transcribed while the other two are full of
    words waiting.
    """
    opened = fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "gamma.m4a", weak_transcript("gamma.m4a"))
        window._table.select_row(0)  # alpha.m4a, which has never been transcribed

        window.show_review()

        assert window._review_window is not None
        assert opened[0].recording_names == ["gamma.m4a"]
        assert "Reviewing 1 recording" in window._status_label.text()
    finally:
        close_window(window)


def test_a_folder_with_nothing_transcribed_refuses_and_says_so(
    qapp, monkeypatch, store, audio_folder
):
    """The wording leads with the project, because that is what was asked about."""
    opened = fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        window._table.select_row(1)

        window.show_review()

        assert opened == []
        assert window._review_window is None
        message = window._status_label.text()
        assert "There is nothing in this project to review." in message
        assert "has been transcribed yet" in message
        # It is announced, as the other refusals in this window are.
        assert "nothing in this project" in message
    finally:
        close_window(window)


def test_a_folder_whose_only_transcript_cannot_be_read_says_it_may_be_damaged(
    qapp, monkeypatch, store, audio_folder
):
    opened = fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        damaged = window._transcript_store(audio_folder / "beta.m4a")
        damaged.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        damaged.transcript_path.write_text("{ this is not json", encoding="utf-8")

        window.show_review()

        assert opened == []
        message = window._status_label.text()
        assert "There is nothing in this project to review." in message
        assert "may be damaged" in message
        assert "beta.m4a" in message
    finally:
        close_window(window)


def test_a_damaged_transcript_beside_a_good_one_is_named_rather_than_fatal(
    qapp, monkeypatch, store, audio_folder
):
    opened = fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))
        damaged = window._transcript_store(audio_folder / "beta.m4a")
        damaged.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        damaged.transcript_path.write_text("{ this is not json", encoding="utf-8")

        window.show_review()

        assert opened[0].recording_names == ["alpha.m4a"]
        assert "1 transcript could not be read" in window._status_label.text()
        assert "beta.m4a" in window._status_label.text()
    finally:
        close_window(window)


def test_a_transcript_that_goes_bad_later_is_announced_when_it_is_reached(
    qapp, monkeypatch, store, audio_folder
):
    """The price of not reading the folder in advance, paid honestly.

    Nothing knows a transcript is damaged until something tries to read it, so
    a file that cannot be read is reported at the moment that is discovered
    rather than being swallowed because the window has already opened.

    What it must not say is that the recording has left the review. Its words
    are still there and still carry every decision made about them, and the
    review window says as much in its own panels, so the status bar has to
    agree with it rather than frighten somebody into doing the work again.
    """
    opened = fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))
        save_transcript(window, audio_folder / "beta.m4a", weak_transcript("beta.m4a"))
        window.show_review()
        assert "could not be read" not in window._status_label.text()

        broken = window._transcript_store(audio_folder / "beta.m4a")
        broken.transcript_path.write_text("{ this is not json", encoding="utf-8")
        # Read the other recording first, so that the one transcript the
        # reader holds is no longer the one about to be broken.
        assert opened[0].load_transcript("alpha.m4a") is not None

        assert opened[0].load_transcript("beta.m4a") is None

        message = window._status_label.text()
        assert "The transcript for beta.m4a could not be read" in message
        assert "Nothing you have decided about them has been lost" in message
        # The recording is still in the review, so nothing may suggest it has
        # been dropped out of it.
        assert "not in this review" not in message
        assert beta_occurrences(audio_folder) == 1
    finally:
        close_window(window)


def test_the_review_window_gets_a_player_of_its_own(
    qapp, monkeypatch, store, audio_folder
):
    """Two windows driving one player would fight over what is loaded."""
    opened = fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))

        window.show_review()

        player = opened[0].player
        assert isinstance(player, AudioPlayer)
        assert player is not window._player
        # It belongs to the review window, so Qt disposes of it with the window.
        assert player.parent() is opened[0]
    finally:
        close_window(window)


def test_only_one_review_window_is_kept_open(qapp, monkeypatch, store, audio_folder):
    opened = fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))

        window.show_review()
        first = window._review_window
        window.show_review()

        assert len(opened) == 2
        assert window._review_window is not first
    finally:
        close_window(window)


def test_changing_folder_closes_the_review_window(
    qapp, monkeypatch, store, audio_folder, tmp_path
):
    """A review belongs to one folder, and its player is loading files from it."""
    fake_review_window(monkeypatch)
    other = tmp_path / "other"
    other.mkdir()
    write_fake_audio(other / "one.m4a")
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))
        window.show_review()
        assert window._review_window is not None

        window._folder_panel.folderChosen.emit(str(other))

        assert window._review_window is None
        assert wait_until(qapp, lambda: window._model.rowCount() == 1)
    finally:
        close_window(window)


def test_refreshing_the_file_list_leaves_the_review_window_alone(
    qapp, monkeypatch, store, audio_folder
):
    """Reading the same folder again is not a change of folder."""
    fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))
        window.show_review()
        review = window._review_window

        window.refresh()
        assert wait_until(qapp, lambda: window._model.rowCount() == 3)

        assert window._review_window is review
    finally:
        close_window(window)


def test_a_correction_that_could_not_be_saved_is_reported(qapp, store, audio_folder):
    """The one failure here that must never pass quietly."""
    window = loaded_window(qapp, store, audio_folder)
    try:
        transcript = weak_transcript("alpha.m4a")
        transcript_store = save_transcript(window, audio_folder / "alpha.m4a", transcript)

        class RefusingStore:
            transcript_path = transcript_store.transcript_path

            def save(self, _transcript) -> bool:
                return False

        reader = main_window_module._TranscriptReader({"alpha.m4a": RefusingStore()})

        # The answer is what lets the review window, which is the one in
        # front, say so and leave the word as it was.
        assert window._save_correction(reader, "alpha.m4a", transcript) is False

        assert "could not be saved" in window._status_label.text()
    finally:
        close_window(window)


def test_a_correction_that_could_not_be_saved_is_announced_by_the_review_window_only(
    qapp, monkeypatch, store, audio_folder
):
    """Two urgent announcements one after the other cut across each other.

    The review window is in front and says it; this window only keeps the
    reason, with the path, on its status bar.
    """
    said: list[str] = []
    monkeypatch.setattr(
        main_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    window = loaded_window(qapp, store, audio_folder)
    try:
        transcript = weak_transcript("alpha.m4a")
        transcript_store = save_transcript(window, audio_folder / "alpha.m4a", transcript)

        class RefusingStore:
            transcript_path = transcript_store.transcript_path

            def save(self, _transcript) -> bool:
                return False

        reader = main_window_module._TranscriptReader({"alpha.m4a": RefusingStore()})
        said.clear()

        window._save_correction(reader, "alpha.m4a", transcript)

        assert said == []
    finally:
        close_window(window)


def test_a_saved_correction_is_what_the_next_read_of_that_recording_sees(
    qapp, store, audio_folder
):
    """Otherwise the person is shown the word they have just changed.

    Only one transcript is held at a time, so a correction written straight to
    the file would leave the reader holding the version from before it.
    """
    window = loaded_window(qapp, store, audio_folder)
    try:
        transcript_store = save_transcript(
            window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a")
        )
        reader = main_window_module._TranscriptReader({"alpha.m4a": transcript_store})
        assert words_of(reader.load("alpha.m4a")) == ["the", "Bosch", "account"]

        assert (
            window._save_correction(
                reader, "alpha.m4a", strong_transcript("alpha.m4a", "Bosche")
            )
            is True
        )

        assert words_of(reader.load("alpha.m4a")) == ["the", "Bosche", "account"]
    finally:
        close_window(window)


def test_a_correction_to_a_recording_this_review_does_not_know_is_refused(
    qapp, store, audio_folder
):
    """Guessing a path from a name is how a correction lands in the wrong folder."""
    window = loaded_window(qapp, store, audio_folder)
    try:
        reader = main_window_module._TranscriptReader({})
        assert (
            window._save_correction(reader, "stranger.m4a", weak_transcript("stranger.m4a"))
            is False
        )

        assert "not one of the ones being reviewed" in window._status_label.text()
    finally:
        close_window(window)


# -- The project a folder remembers --------------------------------------


def test_opening_a_folder_builds_and_saves_its_own_project(
    qapp, monkeypatch, store, audio_folder
):
    opened = fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))

        window.show_review()

        project_store = ProjectStore(audio_folder)
        assert project_store.path.is_file()
        assert project_store.path.parent == audio_folder
        state = project_store.load()
        assert [item.detected_text for item in state.occurrences] == ["Bosch"]
        # Every weak word gets a group, even a group of one, or it would be
        # unreachable in the window's first list.
        assert len(state.groups) == 1
        assert opened[0].project_store.path == project_store.path
    finally:
        close_window(window)


def test_two_folders_keep_their_projects_entirely_apart(
    qapp, monkeypatch, store, tmp_path
):
    """The property most likely to break silently, so it is tested directly.

    A replacement accepted while reviewing one client's interviews has no
    business rewriting another client's, even though the two folders hold the
    very same shaky word.
    """
    fake_review_window(monkeypatch)
    first = tmp_path / "client one"
    second = tmp_path / "client two"
    first.mkdir()
    second.mkdir()
    write_fake_audio(first / "one.m4a")
    write_fake_audio(first / "later.m4a")
    write_fake_audio(second / "two.m4a")

    window = open_window(qapp, store)
    try:
        window._folder_panel.folderChosen.emit(str(first))
        assert wait_until(qapp, lambda: window._model.rowCount() == 2)
        save_transcript(window, first / "one.m4a", weak_transcript("one.m4a"))
        window.show_review()
        # The person accepts "Bosche" while reviewing the first folder, and a
        # recording transcribed afterwards there is answered by it.
        add_replacement_rule(first)
        save_transcript(window, first / "later.m4a", weak_transcript("later.m4a"))
        window.show_review()
        assert words_of(load_transcript(window, first / "later.m4a")) == [
            "the",
            "Bosche",
            "account",
        ]

        window._folder_panel.folderChosen.emit(str(second))
        assert wait_until(qapp, lambda: window._model.rowCount() == 1)
        save_transcript(window, second / "two.m4a", weak_transcript("two.m4a"))
        window.show_review()

        # Nothing at all crossed over: not the rule, not the correction, and
        # not the occurrences.
        second_state = ProjectStore(second).load()
        assert second_state.rules == []
        assert words_of(load_transcript(window, second / "two.m4a")) == [
            "the",
            "Bosch",
            "account",
        ]
        assert ProjectStore(second).path.parent == second
        assert [item.recording_name for item in second_state.occurrences] == ["two.m4a"]

        # And the first folder was left exactly as it was found.
        first_state = ProjectStore(first).load()
        assert [rule.replacement for rule in first_state.rules] == ["Bosche"]
        assert sorted({item.recording_name for item in first_state.occurrences}) == [
            "later.m4a",
            "one.m4a",
        ]
    finally:
        close_window(window)


def test_a_file_transcribed_later_is_answered_by_what_the_project_knows(
    qapp, monkeypatch, store, audio_folder
):
    """The point of remembering a replacement at all.

    The new file's "Bosch" is one the services were confident about, which is
    the case this exists for: the words a service gets confidently wrong are
    the proper names, so a correction that only reached words already in doubt
    would miss it.
    """
    fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))
        window.show_review()
        rule = add_replacement_rule(audio_folder)

        # A recording transcribed afterwards, whose "Bosch" nothing doubts.
        save_transcript(window, audio_folder / "beta.m4a", strong_transcript("beta.m4a"))
        window.show_review()

        beta = load_transcript(window, audio_folder / "beta.m4a")
        assert words_of(beta) == ["the", "Bosche", "account"]
        corrected = beta.tokens[1]
        # What the services really said is kept, so the change can be explained
        # and undone.
        assert corrected.original_text == "Bosch"

        state = ProjectStore(audio_folder).load()
        answered = [item for item in state.occurrences if item.recording_name == "beta.m4a"]
        assert len(answered) == 1
        assert answered[0].auto_applied is True
        assert answered[0].reviewed is True
        assert answered[0].applied_rule_id == rule.id
        assert answered[0].detected_text == "Bosch"
        assert [item.occurrence_count for item in state.rules] == [1]

        assert "1 word was corrected automatically" in window._status_label.text()
    finally:
        close_window(window)


def test_a_word_a_rule_has_already_answered_is_not_answered_twice(
    qapp, monkeypatch, store, audio_folder
):
    """Correcting it again would record the correction as what was really said."""
    fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))
        window.show_review()
        add_replacement_rule(audio_folder)
        save_transcript(window, audio_folder / "beta.m4a", strong_transcript("beta.m4a"))
        window.show_review()

        window.show_review()

        beta = load_transcript(window, audio_folder / "beta.m4a")
        assert words_of(beta) == ["the", "Bosche", "account"]
        assert beta.tokens[1].original_text == "Bosch"
        state = ProjectStore(audio_folder).load()
        assert [item.occurrence_count for item in state.rules] == [1]
        assert "corrected automatically" not in window._status_label.text()
    finally:
        close_window(window)


def test_a_correction_a_rule_could_not_be_saved_is_reported(
    qapp, monkeypatch, store, audio_folder
):
    """A rule's correction that never reached the disk must not pass quietly.

    The project file records it as having happened, so silence here would
    leave the person with a project and a transcript telling different stories
    and nothing to say which one to believe.
    """
    fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))
        window.show_review()
        add_replacement_rule(audio_folder)
        save_transcript(window, audio_folder / "beta.m4a", strong_transcript("beta.m4a"))

        monkeypatch.setattr(TranscriptStore, "save", lambda self, transcript: False)
        window.show_review()

        assert "could not be saved" in window._status_label.text()
        assert "beta.m4a" in window._status_label.text()
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


def test_every_mnemonic_in_the_main_window_is_its_own(qapp, store):
    """No two controls in the main window answer to the same Alt key.

    Enhance Audio and the details panel's File name label both took Alt+N,
    so the key moved between them and never reliably started Enhance.
    """
    window = open_window(qapp, store)
    try:
        texts = [action.text() for action in window.menuBar().actions()]
        texts += [label.text() for label in window.findChildren(QLabel)]
        texts += [button.text() for button in window.findChildren(QAbstractButton)]
        keys = [key for key in map(mnemonic_of, texts) if key]

        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        assert duplicates == []
        assert "n" in keys and "m" in keys
    finally:
        close_window(window)


def test_the_review_window_records_its_statistics_in_the_shared_file(
    qapp, monkeypatch, store, audio_folder, tmp_path
):
    opened = fake_review_window(monkeypatch)
    window = loaded_window(qapp, store, audio_folder)
    try:
        window._calibration_store = CalibrationStore(tmp_path / "calibration.json")
        save_transcript(window, audio_folder / "alpha.m4a", weak_transcript("alpha.m4a"))
        window.show_review()
        note = StatisticsNote(provider=Provider.OPENAI, counted=True, text_corrected=True)

        assert opened[-1].record_statistics(StatisticsChange(added=(note,))) is True

        counts = window._calibration_store.load().counts_for(Provider.OPENAI)
        assert (counts.chosen, counts.corrected) == (1, 1)
    finally:
        close_window(window)
