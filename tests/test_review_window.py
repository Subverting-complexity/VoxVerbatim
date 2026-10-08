"""Tests for the review window: the two lists, the details, playback and corrections.

Every transcript is built by hand, the audio player is a stand-in that
records what it was asked to do, and the folder of transcripts is a small
object that answers reads and remembers writes. So none of this needs a
network, an API key or an audio file, which is the whole point of the window
taking a list of names, a way to read one, and a callback rather than
reaching for the pipeline itself.

The folder stand-in also counts its reads, because the window is expected to
read a transcript when it wants one and not before: a folder of fifty
hour-long recordings is well over a gigabyte, and opening a window must not
cost that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAccessible, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableView,
)

from vox_verbatim.transcription.calibration import CALIBRATION_FILE_NAME, CalibrationStore
from vox_verbatim.transcription.grouping import affected_summary
from vox_verbatim.transcription.model import (
    AudioSpan,
    Candidate,
    CanonicalAudio,
    Confidence,
    FinalToken,
    Language,
    LanguageEvidence,
    Provider,
    ProviderResult,
    ProviderToken,
    ReviewReason,
    ReviewStatus,
    RiskCategory,
    Speaker,
    TimingStatus,
    TokenReference,
    Transcript,
)
from vox_verbatim.transcription.project import ProjectStore, remove_learned_name
from vox_verbatim.transcription.vocabulary import VocabularyStore
from vox_verbatim.ui import review_lists
from vox_verbatim.ui import review_window as review_window_module
from vox_verbatim.ui import review_queue
from vox_verbatim.ui.review_lists import (
    GROUP_COLUMN_CONFIDENCE,
    GROUP_COLUMN_COUNT,
    GROUP_COLUMN_REVIEWED,
    GROUP_COLUMN_TITLES,
    GROUP_COLUMN_WHEN,
    GROUP_COLUMN_WHY,
    GROUP_COLUMN_WORD,
    NOT_REVIEWED,
    OCCURRENCE_COLUMN_CONFIDENCE,
    OCCURRENCE_COLUMN_FILE,
    OCCURRENCE_COLUMN_LANGUAGE,
    OCCURRENCE_COLUMN_REPLACEMENT,
    OCCURRENCE_COLUMN_REVIEWED,
    OCCURRENCE_COLUMN_WHEN,
    REVIEWED_AS_DETECTED,
    REVIEWED_AUTOMATICALLY,
    loose_key,
)
from vox_verbatim.ui.review_window import (
    AUDIO_EVENT,
    CANDIDATE_COLUMN_CHOICE,
    CANDIDATE_COLUMN_CONFIDENCE,
    CANDIDATE_COLUMN_KIND,
    CANDIDATE_COLUMN_SERVICE,
    CANDIDATE_COLUMN_TEXT,
    CANDIDATE_COLUMN_VOCABULARY,
    CURRENT_CHOICE,
    IN_VOCABULARY,
    NOT_IN_VOCABULARY,
    SHORT_CONTEXT_SECONDS,
    WIDE_CONTEXT_SECONDS,
    WORD_MARGIN_SECONDS,
    ReviewWindow,
    candidate_rows,
)

RECORDING = "board meeting.m4a"
OTHER_RECORDING = "site visit.m4a"


class FakePlayer(QObject):
    """Stands in for the audio player, recording what it was asked to do.

    It carries the same signals as the real one, because the window watches
    the position to know when to stop at the end of a region, and watches the
    duration to know when a newly loaded file is ready to be seeked in.

    Nothing here becomes ready by itself. A test that wants the media to
    arrive says so, with :meth:`becomes_ready`, which is what lets the delay
    between asking for audio and being able to play it be tested at all.
    """

    positionChanged = Signal(int)
    durationChanged = Signal(int)
    playingChanged = Signal(bool)
    playbackFinished = Signal()
    errorOccurred = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.loaded: str | None = None
        self.loads: list[str] = []
        self.seeks: list[int] = []
        self.plays = 0
        self.pauses = 0
        self.stops = 0
        self.cleared = 0

    def load(self, path) -> None:
        """Open a file, or let go of the one that is open when given nothing.

        Clearing is counted separately from loading, because the two are asked
        for at different moments for different reasons and the window is
        expected to do the second exactly once, as it closes. Letting go is
        what actually releases the operating system's handle on the recording,
        which is what decides whether the person can rename the file
        afterwards.
        """
        if path is None:
            self.loaded = None
            self.cleared += 1
            return
        self.loaded = str(path)
        self.loads.append(str(path))

    def becomes_ready(self, milliseconds: int = 600_000) -> None:
        self.durationChanged.emit(milliseconds)

    def seek_to(self, milliseconds: int) -> None:
        self.seeks.append(int(milliseconds))

    def play(self) -> None:
        self.plays += 1

    def pause(self) -> None:
        self.pauses += 1

    def stop(self) -> None:
        self.stops += 1


class Folder:
    """The transcripts of a folder, standing in for whoever stores them.

    Reads and writes are both counted, so that a test can say not only that
    the window produced the right transcript but that it went and got the
    right one, and that it did not go and get all of them.
    """

    def __init__(self, transcripts: dict[str, Transcript]) -> None:
        self.transcripts = dict(transcripts)
        self.reads: list[str] = []
        self.saved: list[tuple[str, Transcript]] = []
        # The recordings whose transcripts cannot be written, standing in for
        # a full disk or a file a sync client is holding open.
        self.refusing: set[str] = set()

    @property
    def names(self) -> list[str]:
        return list(self.transcripts)

    def load(self, recording_name: str) -> Transcript | None:
        self.reads.append(recording_name)
        return self.transcripts.get(recording_name)

    def save(self, recording_name: str, transcript: Transcript) -> bool:
        if recording_name in self.refusing:
            return False
        self.transcripts[recording_name] = transcript
        self.saved.append((recording_name, transcript))
        return True

    def token(self, recording_name: str, token_id: str) -> FinalToken | None:
        return self.transcripts[recording_name].token_by_id(token_id)

    def texts(self, recording_name: str) -> list[str]:
        return [token.text for token in self.transcripts[recording_name].tokens]


def make_token(
    text: str = "contract",
    start: float | None = 30.0,
    end: float | None = 30.5,
    reasons: tuple[ReviewReason, ...] = (ReviewReason.PROVIDER_DISAGREEMENT,),
    confidence: Confidence = Confidence.REVIEW_REQUIRED,
    speaker: str | None = "speaker_0",
    span: AudioSpan | None = None,
    candidates: tuple[Candidate, ...] = (),
    strength: float | None = None,
    **extra,
) -> FinalToken:
    token = FinalToken(
        text=text,
        start=start,
        end=end,
        text_source=Provider.ELEVENLABS,
        text_confidence=confidence,
        confidence_strength=strength,
        timing_source=Provider.ELEVENLABS,
        timing_status=TimingStatus.EXACT_PROVIDER_TIME,
        timing_confidence=Confidence.HIGH,
        speaker=speaker,
        speaker_source=Provider.ASSEMBLYAI,
        speaker_confidence=Confidence.HIGH,
        language=Language.ENGLISH,
        source_audio_span=span,
        candidates=list(candidates),
        **extra,
    )
    for reason in reasons:
        token.flag(reason)
    return token


def weak_token(text: str, start: float, strength: float = 0.42) -> FinalToken:
    """A word that is quietly weak: low confidence, and flagged for nothing.

    This is what the low-confidence sweep is for. Nothing in the pipeline
    noticed it, so it never reaches the review queue by the other road.
    """
    return make_token(
        text,
        start=start,
        end=start + 0.4,
        reasons=(),
        confidence=Confidence.HIGH,
        strength=strength,
    )


def make_transcript(
    tokens: list[FinalToken] | None = None,
    name: str = RECORDING,
    duration: float = 600.0,
    with_audio: bool = True,
) -> Transcript:
    transcript = Transcript(recording_name=name)
    transcript.tokens = list(tokens or [make_token()])
    transcript.speakers = [Speaker(id="speaker_0", name="Jacques"), Speaker(id="speaker_1")]
    if with_audio:
        transcript.canonical_audio = CanonicalAudio(
            path=f"C:/Audio/{name}.wav",
            original_path=f"C:/Audio/{name}",
            duration=duration,
            sample_rate=48000,
            channels=1,
            size_bytes=1024,
            container="wav",
        )
    return transcript


def open_window(
    tmp_path,
    folder: Folder | None = None,
    player: FakePlayer | None = None,
    store: ProjectStore | None = None,
    process: bool = False,
    show_details: bool = False,
) -> ReviewWindow:
    """Open the window on a folder, optionally having processed it first.

    ``show_details`` opens it with the details on, as a folder whose person
    turned them on last time would. Without it the window opens simple, as
    every new folder does.
    """
    folder = folder or Folder({RECORDING: make_transcript()})
    store = store or ProjectStore(tmp_path)
    if show_details:
        state = store.load()
        state.settings.show_details = True
        store.save(state)
    window = ReviewWindow(
        tmp_path,
        folder.names,
        folder.load,
        {name: Path(f"C:/Audio/{name}") for name in folder.names},
        store,
        player or FakePlayer(),
        folder.save,
    )
    window.show()
    if process:
        window.process_low_confidence_words()
    return window


def two_file_folder() -> Folder:
    """A surname spelled three ways across two recordings, plus a flagged number.

    This is the case the whole feature exists for: one decision that should
    land in two files, beside an item that reached the list by the other road.
    """
    return Folder(
        {
            RECORDING: make_transcript(
                [
                    weak_token("Bosch", 10.0, 0.42),
                    weak_token("Bosh", 40.0, 0.45),
                    make_token(
                        "15,000",
                        start=70.0,
                        end=70.5,
                        reasons=(ReviewReason.NUMERIC_DISAGREEMENT,),
                    ),
                ],
                name=RECORDING,
            ),
            OTHER_RECORDING: make_transcript(
                [weak_token("Bosche", 12.0, 0.44), weak_token("settled", 20.0, 0.95)],
                name=OTHER_RECORDING,
            ),
        }
    )


def interactive_widgets(window: ReviewWindow) -> list:
    """Every control a person can land on, whatever it happens to be.

    Gathered by widget type rather than by name, so that a control added
    later is caught by these tests without anybody remembering to list it.
    """
    kinds = (
        QPushButton,
        QLineEdit,
        QCheckBox,
        QComboBox,
        QPlainTextEdit,
        QTableView,
        QSpinBox,
    )
    return [widget for kind in kinds for widget in window.findChildren(kind)]


def accessible_name(widget) -> str:
    interface = QAccessible.queryAccessibleInterface(widget)
    assert interface is not None
    return interface.text(QAccessible.Text.Name)


def group_row(window: ReviewWindow, word: str):
    for row in window._group_model.rows():
        if row.word == word:
            return row
    raise AssertionError(f"{word} is not in the word list")


def select_word(window: ReviewWindow, word: str) -> None:
    row = group_row(window, word)
    window._groups.select_row(window._group_model.row_for_key(row.key))


# -- What the window reads, and when -------------------------------------


def test_opening_the_window_reads_no_transcripts_at_all(qapp, tmp_path):
    """Fifty hour-long recordings are over a gigabyte; opening must not cost that."""
    folder = two_file_folder()
    window = open_window(tmp_path, folder)
    try:
        assert folder.reads == []
        assert window._count_label.text().startswith("There is nothing to review.")
    finally:
        window.close()


def test_the_list_says_the_flagged_words_have_not_been_looked_for_yet(qapp, tmp_path):
    """An empty block must not be read as a folder with nothing wrong in it."""
    window = open_window(tmp_path, two_file_folder())
    try:
        assert "found when the folder is processed" in window._count_label.text()
    finally:
        window.close()


def flagged_only_folder() -> Folder:
    """Two recordings whose every word is strong, each with one flagged by a rule.

    Nothing here reaches the list by the low-confidence sweep, so the second
    block is all there is, which is what makes it possible to say plainly
    whether that block survived the window closing.
    """
    return Folder(
        {
            RECORDING: make_transcript(
                [
                    make_token(
                        "15,000",
                        start=70.0,
                        end=70.5,
                        reasons=(ReviewReason.NUMERIC_DISAGREEMENT,),
                    )
                ],
                name=RECORDING,
            ),
            OTHER_RECORDING: make_transcript(
                [
                    make_token(
                        "Jurgen",
                        start=20.0,
                        end=20.5,
                        reasons=(ReviewReason.HIGH_RISK_ENTITY,),
                    )
                ],
                name=OTHER_RECORDING,
            ),
        }
    )


def test_the_flagged_words_come_back_from_the_project_without_a_read(qapp, tmp_path):
    """They used to be found only while a transcript was being read.

    That made them appear on the review that processed the folder and vanish
    from every later one, because an unchanged folder is deliberately not read
    again. They are saved in the project now, so the block is whole from the
    moment the window opens. The one read here is the transcript behind the
    row the window lands on, whose detail panel genuinely needs it; the other
    recording is never opened and still has its word on the list.
    """
    folder = flagged_only_folder()
    store = ProjectStore(tmp_path)
    first = open_window(tmp_path, folder, store=store, process=True)
    first.close()

    folder.reads.clear()
    second = open_window(tmp_path, folder, store=store)
    try:
        assert [row.word for row in second._group_model.rows()] == ["15,000", "Jurgen"]
        assert folder.reads == [RECORDING]
        assert group_row(second, "Jurgen").why == "A value that the services heard differently"
        assert group_row(second, "Jurgen").reviewed == NOT_REVIEWED
    finally:
        second.close()


def test_a_processed_folder_no_longer_says_the_flagged_words_are_still_to_come(
    qapp, tmp_path
):
    """The sentence must not claim an incompleteness the block no longer has."""
    folder = flagged_only_folder()
    store = ProjectStore(tmp_path)
    first = open_window(tmp_path, folder, store=store, process=True)
    first.close()

    second = open_window(tmp_path, folder, store=store)
    try:
        assert second._count_label.text() == "Showing 2 other uncertainties."
    finally:
        second.close()


def test_a_saved_flagged_word_that_has_gone_says_so_and_refuses_to_be_corrected(
    qapp, tmp_path
):
    """The second line of defence against a saved item outliving its word.

    The first is that a regenerated transcript is read again by the next
    analysis, which replaces everything remembered about that recording. This
    is what happens in between: the person lands on the row, the window loads
    the transcript for the detail panel as it would for any row, and the word
    is not there. It says so and every correction refuses, which costs nothing
    because that read was going to happen anyway.
    """
    folder = flagged_only_folder()
    store = ProjectStore(tmp_path)
    first = open_window(tmp_path, folder, store=store, process=True)
    first.close()

    # Transcribed again behind the window's back: the same word, a new
    # identifier, and nothing left for the saved item to point at.
    folder.transcripts[RECORDING] = make_transcript(
        [make_token("15,000", start=70.0, end=70.5, reasons=(ReviewReason.NUMERIC_DISAGREEMENT,))],
        name=RECORDING,
    )
    second = open_window(tmp_path, folder, store=store)
    try:
        select_word(second, "15,000")

        assert second._timing_edit.text() == review_window_module.WORD_NOT_IN_TRANSCRIPT
        assert second.confirm_item() is False
    finally:
        second.close()


def test_a_flagged_word_settled_earlier_says_it_was_reviewed(qapp, tmp_path):
    """What it does not say is how, because the project does not know how.

    Which of the two ways somebody settled a word is written into the
    transcript, and the transcript is not open. Guessing between "confirmed"
    and "corrected" would be a claim about what a person did, made on nothing,
    in the one column that exists to tell their decisions apart.
    """
    folder = flagged_only_folder()
    store = ProjectStore(tmp_path)
    first = open_window(tmp_path, folder, store=store, process=True)
    select_word(first, "15,000")
    first.confirm_item()
    first.close()

    second = open_window(tmp_path, folder, store=store)
    try:
        assert [row.word for row in second._group_model.rows()] == ["Jurgen"]

        second.set_show_reviewed(True)

        assert group_row(second, "15,000").reviewed == review_lists.REVIEWED_PLAIN
    finally:
        second.close()


def test_only_the_recording_being_worked_in_is_read(qapp, tmp_path):
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "15,000")
        # Forget what the analysis left behind, so that the read this
        # selection needs is the only one there is to see.
        window._cached_name = None
        window._cached_transcript = None
        folder.reads.clear()
        select_word(window, "Bosch")

        # The selected occurrence's recording, and no other.
        assert folder.reads == [RECORDING]
    finally:
        window.close()


def test_moving_through_one_recording_reads_it_once(qapp, tmp_path):
    """Re-reading tens of megabytes on every arrow key would be its own kind of slow."""
    folder = Folder(
        {
            RECORDING: make_transcript(
                [weak_token("Bosch", 10.0), weak_token("Bosch", 20.0), weak_token("Bosch", 30.0)]
            )
        }
    )
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        folder.reads.clear()
        window.go_to_next_item()
        window.go_to_next_item()

        assert folder.reads == []
    finally:
        window.close()


def test_a_read_overtaken_by_another_does_not_file_its_transcript_under_the_wrong_name(
    qapp, tmp_path
):
    """A loader may let the event loop run while it parses, and a key press
    handled in that time can ask the window for another recording. The outer
    read must not come back and store the first recording's words under the
    second recording's name, which is what writing the name before the read
    used to do.
    """
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    re_entered: list[Transcript | None] = []

    def load_and_re_enter_once(recording_name: str) -> Transcript | None:
        if recording_name == RECORDING and not re_entered:
            # Part way through reading the first recording, the window is
            # asked for the second, as a key press would.
            re_entered.append(window._transcript(OTHER_RECORDING))
        return folder.load(recording_name)

    try:
        window._load_transcript = load_and_re_enter_once
        window._cached_name = None
        window._cached_transcript = None

        outer = window._transcript(RECORDING)

        # Each caller got the recording it asked for.
        assert outer is folder.transcripts[RECORDING]
        assert re_entered == [folder.transcripts[OTHER_RECORDING]]
        # And what is held is a matching pair, not one's words under the
        # other's name.
        assert window._cached_name == OTHER_RECORDING
        assert window._cached_transcript is folder.transcripts[OTHER_RECORDING]
        assert window._transcript(OTHER_RECORDING) is folder.transcripts[OTHER_RECORDING]
    finally:
        window.close()


def test_a_transcript_that_cannot_be_read_says_so_without_claiming_the_word_is_gone(
    qapp, tmp_path
):
    """Two different pieces of news, and only one of them is about the word."""
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        del folder.transcripts[RECORDING]
        window._cached_name = None

        window.refresh()

        assert "could not be read just now" in window._timing_edit.text()
        assert "has been lost" in window._timing_edit.text()
        assert window.apply_replacement_to_occurrence() is False
    finally:
        window.close()


# -- The two lists --------------------------------------------------------


def test_the_words_are_gathered_across_recordings_into_one_row(qapp, tmp_path):
    """The whole point: one decision rather than the same decision three times."""
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        row = group_row(window, "Bosch")

        assert row.count == 3
        assert row.file_count == 2
        assert {occurrence.recording_name for occurrence in row.occurrences} == {
            RECORDING,
            OTHER_RECORDING,
        }
    finally:
        window.close()


def test_the_second_block_holds_the_words_the_pipeline_flagged(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        words = [row.word for row in window._group_model.rows()]

        # The low confidence word first, the flagged one after it, never mixed.
        assert words == ["Bosch", "15,000"]
        assert group_row(window, "15,000").why == "A number differs between services"
        assert group_row(window, "Bosch").why == review_lists.WHY_LOW_CONFIDENCE
    finally:
        window.close()


def test_a_settled_word_is_not_in_either_block(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        assert "settled" not in [row.word for row in window._group_model.rows()]
    finally:
        window.close()


def test_the_weakest_word_comes_first(qapp, tmp_path):
    folder = Folder(
        {
            RECORDING: make_transcript(
                [weak_token("stronger", 10.0, 0.51), weak_token("weaker", 20.0, 0.12)]
            )
        }
    )
    window = open_window(tmp_path, folder, process=True)
    try:
        assert [row.word for row in window._group_model.rows()] == ["weaker", "stronger"]
    finally:
        window.close()


def test_the_second_list_holds_every_occurrence_of_the_selected_word(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")

        model = window._occurrence_model
        assert model.rowCount() == 3
        assert model.data(model.index(0, OCCURRENCE_COLUMN_FILE)) in (
            RECORDING,
            OTHER_RECORDING,
        )
        assert model.data(model.index(0, OCCURRENCE_COLUMN_LANGUAGE)) == "English"
    finally:
        window.close()


def test_every_column_of_both_lists_is_a_word_or_a_number(qapp, tmp_path):
    """Nothing is a colour, an icon, or a blank cell whose meaning is its column."""
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        groups = window._group_model
        assert groups.data(groups.index(0, GROUP_COLUMN_WORD)) == "Bosch"
        assert groups.data(groups.index(0, GROUP_COLUMN_WHEN)) == "0:10 and 2 more"
        assert groups.data(groups.index(0, GROUP_COLUMN_COUNT)) == "3"
        assert groups.data(groups.index(0, GROUP_COLUMN_CONFIDENCE)) == "42% to 45%"
        assert groups.data(groups.index(0, GROUP_COLUMN_WHY)) == "Low confidence"
        assert groups.data(groups.index(0, GROUP_COLUMN_REVIEWED)) == NOT_REVIEWED

        # A count read out as a bare "3" leaves a listener to remember which
        # column they are in to know three of what.
        assert groups.data(
            groups.index(0, GROUP_COLUMN_COUNT), Qt.ItemDataRole.AccessibleTextRole
        ) == "3 occurrences across 2 files"
        assert groups.data(
            groups.index(0, GROUP_COLUMN_CONFIDENCE), Qt.ItemDataRole.AccessibleTextRole
        ) == "42 percent to 45 percent"
        # The time is said in words, and so is how many more there are.
        assert groups.data(
            groups.index(0, GROUP_COLUMN_WHEN), Qt.ItemDataRole.AccessibleTextRole
        ) == "10 seconds, and 2 more"

        items = window._occurrence_model
        assert items.data(items.index(0, OCCURRENCE_COLUMN_REPLACEMENT)) == "None set"
        assert items.data(items.index(0, OCCURRENCE_COLUMN_REVIEWED)) == NOT_REVIEWED
        assert items.data(
            items.index(0, OCCURRENCE_COLUMN_WHEN), Qt.ItemDataRole.AccessibleTextRole
        ) == "10 seconds"
        assert items.data(
            items.index(0, OCCURRENCE_COLUMN_CONFIDENCE), Qt.ItemDataRole.AccessibleTextRole
        ) == "42 percent"
    finally:
        window.close()


def test_a_word_nobody_measured_says_so_rather_than_showing_nought(qapp, tmp_path):
    """"Nobody measured this" and "this is as weak as it gets" are opposites."""
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "15,000")

        model = window._occurrence_model
        assert model.data(model.index(0, OCCURRENCE_COLUMN_CONFIDENCE)) == "Not measured"
        # The row itself still says how bad it is, using the category the
        # pipeline did record.
        assert group_row(window, "15,000").confidence == "Review required"
    finally:
        window.close()


def test_the_window_opens_on_the_first_word_rather_than_on_nothing(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        assert window._groups.selected_row() == 0
        assert window.current_row().word == "Bosch"
        assert window.current_occurrence() is not None
    finally:
        window.close()


def test_a_folder_with_nothing_to_review_says_so_and_switches_the_actions_off(qapp, tmp_path):
    folder = Folder({RECORDING: make_transcript([weak_token("fine", 1.0, 0.99)])})
    window = open_window(tmp_path, folder, process=True)
    try:
        assert window._group_model.rowCount() == 0
        assert window._count_label.text().startswith("There is nothing to review.")
        assert not window._confirm_button.isEnabled()
        assert not window._play_button.isEnabled()
        assert not window._apply_word_button.isEnabled()
    finally:
        window.close()


def test_moving_between_occurrences_and_between_words_are_different_keys(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        first = window.current_occurrence().id

        window.go_to_next_item()
        assert window.current_occurrence().id != first
        assert window.current_row().word == "Bosch"

        window.go_to_next_group()
        assert window.current_row().word == "15,000"
    finally:
        window.close()


def test_the_ends_of_a_word_point_the_way_out_rather_than_wrapping_round(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "15,000")
        window.go_to_previous_item()
        assert "first occurrence" in window._status_label.text()
        assert "Ctrl+Shift+F3" in window._status_label.text()

        window.go_to_next_item()
        assert "last occurrence" in window._status_label.text()

        window.go_to_next_group()
        assert "last word" in window._status_label.text()
    finally:
        window.close()


def test_moving_to_a_word_reads_the_whole_row_out(qapp, tmp_path, monkeypatch):
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        said.clear()
        window.go_to_next_group()

        assert said == [
            (
                "15,000. 1 occurrence in 1 file. Review required. "
                "A number differs between services. Not reviewed."
            )
        ]
    finally:
        window.close()


def test_moving_to_an_occurrence_reads_the_whole_row_out(qapp, tmp_path, monkeypatch):
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        said.clear()
        window.go_to_next_item()

        assert len(said) == 1
        assert said[0].endswith("Replacement None set. Not reviewed.")
        assert "percent" in said[0]
        assert RECORDING in said[0] or OTHER_RECORDING in said[0]
    finally:
        window.close()


# -- The filters, which narrow the second block ---------------------------


def test_every_reason_has_a_check_box_of_its_own_named_for_what_it_means(qapp, tmp_path):
    """Not a combo box of magic strings that shows one choice at a time."""
    window = open_window(tmp_path)
    try:
        assert set(window._reason_boxes) == set(ReviewReason)
        for reason, box in window._reason_boxes.items():
            assert box.text() == reason.display_name
            assert box.accessibleName() == reason.display_name
            assert box.isChecked()
    finally:
        window.close()


def test_every_confidence_category_has_a_check_box_of_its_own(qapp, tmp_path):
    window = open_window(tmp_path)
    try:
        assert set(window._confidence_boxes) == set(Confidence)
        for confidence, box in window._confidence_boxes.items():
            assert box.text() == confidence.display_name
    finally:
        window.close()


def test_clearing_a_reason_hides_the_words_flagged_for_it(qapp, tmp_path):
    folder = Folder(
        {
            RECORDING: make_transcript(
                [
                    make_token("one", reasons=(ReviewReason.PROVIDER_DISAGREEMENT,)),
                    make_token(
                        "two", start=40.0, reasons=(ReviewReason.LOW_ACOUSTIC_CONFIDENCE,)
                    ),
                ]
            )
        }
    )
    window = open_window(tmp_path, folder, process=True)
    try:
        window._reason_boxes[ReviewReason.LOW_ACOUSTIC_CONFIDENCE].setChecked(False)

        assert [row.word for row in window._group_model.rows()] == ["one"]
    finally:
        window.close()


def test_clearing_a_confidence_category_hides_the_words_rated_that_way(qapp, tmp_path):
    folder = Folder(
        {
            RECORDING: make_transcript(
                [
                    make_token("one", confidence=Confidence.REVIEW_REQUIRED),
                    make_token("two", start=40.0, confidence=Confidence.UNRESOLVED),
                ]
            )
        }
    )
    window = open_window(tmp_path, folder, process=True)
    try:
        window._confidence_boxes[Confidence.UNRESOLVED].setChecked(False)

        assert [row.word for row in window._group_model.rows()] == ["one"]
    finally:
        window.close()


def test_changing_a_filter_says_how_much_is_showing(qapp, tmp_path, monkeypatch):
    """A filter that silently empties the list looks exactly like a broken one."""
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        said.clear()
        window._reason_boxes[ReviewReason.NUMERIC_DISAGREEMENT].setChecked(False)

        assert said == ["Showing 1 word group and 0 of 1 other uncertainties."]
        assert window._count_label.text() == (
            "Showing 1 word group and 0 of 1 other uncertainties."
        )
    finally:
        window.close()


def test_the_count_is_also_on_the_list_itself_for_anyone_tabbing_into_it(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        assert window._groups.accessibleDescription().startswith(
            "Showing 1 word group and 1 other uncertainty."
        )
    finally:
        window.close()


def test_showing_everything_is_one_change_and_one_announcement(qapp, tmp_path, monkeypatch):
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        window._reason_boxes[ReviewReason.NUMERIC_DISAGREEMENT].setChecked(False)
        said.clear()

        window.show_everything()

        assert said.count("Showing 1 word group and 1 other uncertainty.") == 1
        assert all(box.isChecked() for box in window._reason_boxes.values())
        assert all(box.isChecked() for box in window._confidence_boxes.values())
    finally:
        window.close()


# -- The detail panel ----------------------------------------------------


def test_the_word_says_which_file_and_when_now_that_a_window_holds_a_folder(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")

        assert window._detected_edit.text() in ("Bosch", "Bosh", "Bosche")
        assert window._file_edit.text() in (RECORDING, OTHER_RECORDING)
        assert window._position_edit.text() in ("0:10", "0:12", "0:40")
        assert window._strength_edit.text().endswith("%")
        assert window._word_language_edit.text() == "English"
    finally:
        window.close()


def test_the_three_answers_are_shown_as_three_things_with_their_own_sources(qapp, tmp_path):
    """Merging them into one line would hide what is being decided."""
    token = weak_token("Jurgen", 92.0)
    token.timing_status = TimingStatus.SHARED_PHRASE_SPAN
    token.timing_confidence = Confidence.REVIEW_SUGGESTED
    token.speaker_confidence = Confidence.REVIEW_REQUIRED
    folder = Folder({RECORDING: make_transcript([token])})
    window = open_window(tmp_path, folder, process=True)
    try:
        # The start and the exact length, rather than two ends both rounded
        # to the nearest second, which would say the word ran from 1:32 to
        # 1:33 and lasted 0.4 seconds in the same breath.
        assert window._timing_edit.text() == "From 1 minute 32 seconds, 0.4 seconds long"
        assert "share one span" in window._timing_status_edit.text()
        assert window._timing_source_edit.text() == Provider.ELEVENLABS.display_name
        assert window._timing_confidence_edit.text() == "Review suggested"

        assert window._speaker_edit.text() == "Jacques"
        assert window._speaker_source_edit.text() == Provider.ASSEMBLYAI.display_name
        assert window._speaker_confidence_edit.text() == "Review required"
    finally:
        window.close()


def test_a_word_with_no_boundaries_is_described_by_the_region_it_sits_in(qapp, tmp_path):
    token = make_token(start=None, end=None, span=AudioSpan(30.0, 33.0))
    token.timing_status = TimingStatus.UNALIGNED
    folder = Folder({RECORDING: make_transcript([token])})
    window = open_window(tmp_path, folder, process=True)
    try:
        assert window._timing_edit.text() == "Somewhere in the 3.0 seconds from 30 seconds"
    finally:
        window.close()


def test_the_language_evidence_is_shown_as_several_numbers_not_one_answer(qapp, tmp_path):
    token = make_token()
    token.language = Language.GERMAN
    token.language_evidence = LanguageEvidence(
        scores={Language.GERMAN: 0.62, Language.ENGLISH: 0.38}
    )
    folder = Folder({RECORDING: make_transcript([token])})
    window = open_window(tmp_path, folder, process=True)
    try:
        assert window._language_edit.text() == "German"
        assert window._language_evidence_edit.text() == "German 62 percent, English 38 percent"
    finally:
        window.close()


def test_risk_categories_are_spelled_out_because_they_must_never_be_guessed(qapp, tmp_path):
    token = make_token("15,000", reasons=(ReviewReason.HIGH_RISK_ENTITY,))
    token.risk_categories = [RiskCategory.MONEY, RiskCategory.ACCOUNT_NUMBER]
    folder = Folder({RECORDING: make_transcript([token])})
    window = open_window(tmp_path, folder, process=True)
    try:
        assert window._risk_edit.text() == "Money, Account number"
    finally:
        window.close()


def test_a_word_that_is_both_weak_and_flagged_appears_once_and_says_both(qapp, tmp_path):
    """It reaches the list by the low confidence road, and keeps its other flags."""
    token = make_token(
        "Bosch",
        start=10.0,
        end=10.4,
        reasons=(ReviewReason.HIGH_RISK_ENTITY,),
        strength=0.3,
    )
    folder = Folder({RECORDING: make_transcript([token])})
    window = open_window(tmp_path, folder, process=True)
    try:
        rows = window._group_model.rows()
        assert len(rows) == 1
        assert rows[0].why == review_lists.WHY_LOW_CONFIDENCE
        # And the reason it was flagged for is still on the screen.
        assert window._reasons_edit.text() == "A value that the services heard differently"
        # The count must not report it as a second item being kept back,
        # since it is in fact on the screen.
        assert window._count_label.text() == "Showing 1 word group."
    finally:
        window.close()


def test_what_the_language_model_decided_is_shown_where_it_was_consulted(qapp, tmp_path):
    decided = weak_token("15,000", 10.0)
    decided.llm_decision = "Chose 15,000 on the surrounding figures."
    silent = weak_token("contract", 20.0)
    folder = Folder({RECORDING: make_transcript([decided, silent])})
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "15,000")
        assert window._decision_text.toPlainText() == (
            "Chose 15,000 on the surrounding figures."
        )

        select_word(window, "contract")
        assert window._decision_text.toPlainText() == "The language model was not consulted."
    finally:
        window.close()


def test_moving_the_selection_fills_the_detail_panel_with_the_new_item(qapp, tmp_path):
    first = make_token("one", start=10.0, end=10.4, speaker="speaker_0", strength=0.3)
    second = make_token("two", start=20.0, end=20.4, speaker="speaker_1", strength=0.3)
    folder = Folder({RECORDING: make_transcript([first, second])})
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "one")
        assert window._speaker_edit.text() == "Jacques"
        assert window._replacement_edit.text() == "one"

        select_word(window, "two")
        assert window._speaker_edit.text() == "Speaker speaker_1"
        assert window._replacement_edit.text() == "two"
        assert "20 seconds" in window._timing_edit.text()
    finally:
        window.close()


# -- The candidates ------------------------------------------------------


def make_candidate_transcript() -> Transcript:
    """A disputed number, with two services on one answer and one on another."""
    transcript = make_transcript(
        [
            make_token(
                "50,000",
                reasons=(ReviewReason.NUMERIC_DISAGREEMENT,),
                candidates=(
                    Candidate(
                        text="15,000",
                        providers=(Provider.ELEVENLABS,),
                        source_tokens=(TokenReference(Provider.ELEVENLABS, 0),),
                        in_vocabulary=True,
                    ),
                    Candidate(
                        text="50,000",
                        providers=(Provider.OPENAI, Provider.MICROSOFT),
                        source_tokens=(
                            TokenReference(Provider.OPENAI, 4),
                            TokenReference(Provider.MICROSOFT, 7),
                        ),
                    ),
                ),
            )
        ]
    )
    transcript.provider_results[Provider.ELEVENLABS] = ProviderResult(
        provider=Provider.ELEVENLABS,
        tokens=[
            ProviderToken(
                provider=Provider.ELEVENLABS, index=0, text="15,000", log_probability=-0.11
            )
        ],
    )
    transcript.provider_results[Provider.OPENAI] = ProviderResult(
        provider=Provider.OPENAI,
        tokens=[ProviderToken(provider=Provider.OPENAI, index=4, text="50,000", confidence=0.92)],
    )
    transcript.provider_results[Provider.MICROSOFT] = ProviderResult(
        provider=Provider.MICROSOFT,
        tokens=[ProviderToken(provider=Provider.MICROSOFT, index=7, text="50,000")],
    )
    return transcript


def candidate_window(
    tmp_path, player: FakePlayer | None = None, show_details: bool = False
) -> ReviewWindow:
    return open_window(
        tmp_path,
        Folder({RECORDING: make_candidate_transcript()}),
        player,
        process=True,
        show_details=show_details,
    )


def test_every_service_gets_its_own_row_with_the_confidence_it_reported(qapp):
    transcript = make_candidate_transcript()

    rows = candidate_rows(transcript, transcript.tokens[0])

    assert [(row.service, row.text, row.confidence) for row in rows] == [
        (Provider.ELEVENLABS.display_name, "15,000", "Log probability -0.11"),
        (Provider.OPENAI.display_name, "50,000", "92 percent confident"),
        (Provider.MICROSOFT.display_name, "50,000", "Not reported"),
    ]


def test_the_current_choice_and_the_vocabulary_match_are_words_not_marks(qapp):
    transcript = make_candidate_transcript()

    rows = candidate_rows(transcript, transcript.tokens[0])

    assert [row.vocabulary for row in rows] == [
        IN_VOCABULARY,
        NOT_IN_VOCABULARY,
        NOT_IN_VOCABULARY,
    ]
    assert [row.choice for row in rows] == ["Not chosen", CURRENT_CHOICE, CURRENT_CHOICE]


def test_the_chosen_word_is_always_on_the_table_even_with_no_candidate_behind_it(qapp):
    transcript = make_transcript([make_token("database")])

    rows = candidate_rows(transcript, transcript.tokens[0])

    assert [(row.service, row.text, row.choice) for row in rows] == [
        ("The application's choice", "database", CURRENT_CHOICE)
    ]


def test_a_sound_is_marked_as_a_sound_rather_than_read_as_a_word(qapp):
    """Laughter is something that happened, not a transcription mistake."""
    transcript = make_transcript(
        [
            make_token(
                "laughter",
                candidates=(
                    Candidate(
                        text="laughter",
                        providers=(Provider.ELEVENLABS,),
                        source_tokens=(TokenReference(Provider.ELEVENLABS, 0),),
                    ),
                ),
            )
        ]
    )
    transcript.provider_results[Provider.ELEVENLABS] = ProviderResult(
        provider=Provider.ELEVENLABS,
        tokens=[
            ProviderToken(
                provider=Provider.ELEVENLABS,
                index=0,
                text="laughter",
                is_audio_event=True,
            )
        ],
    )

    rows = candidate_rows(transcript, transcript.tokens[0])

    assert rows[0].kind == AUDIO_EVENT


def test_the_candidates_reach_the_table_a_screen_reader_reads(qapp, tmp_path):
    window = candidate_window(tmp_path)
    try:
        model = window._candidates_model
        assert model.rowCount() == 3
        assert model.data(model.index(0, CANDIDATE_COLUMN_SERVICE)) == (
            Provider.ELEVENLABS.display_name
        )
        assert model.data(model.index(0, CANDIDATE_COLUMN_TEXT)) == "15,000"
        assert model.data(model.index(0, CANDIDATE_COLUMN_CONFIDENCE)) == "Log probability -0.11"
        assert model.data(model.index(0, CANDIDATE_COLUMN_VOCABULARY)) == IN_VOCABULARY
        assert model.data(model.index(0, CANDIDATE_COLUMN_KIND)) == "A spoken word"
        assert model.data(model.index(1, CANDIDATE_COLUMN_CHOICE)) == CURRENT_CHOICE
        # The same words are offered to a screen reader as are shown.
        assert model.data(
            model.index(0, CANDIDATE_COLUMN_TEXT), Qt.ItemDataRole.AccessibleTextRole
        ) == "15,000"
    finally:
        window.close()


def test_a_candidate_can_be_read_closely_in_a_field_with_a_real_caret(qapp, tmp_path):
    """Names and numbers have to be spelled out, not glanced at."""
    window = candidate_window(tmp_path)
    try:
        assert window._candidate_text.text() == "15,000"
        assert window._candidate_text.isReadOnly()

        window._candidates.setCurrentIndex(
            window._candidates_model.index(1, CANDIDATE_COLUMN_SERVICE)
        )

        assert window._candidate_text.text() == "50,000"
    finally:
        window.close()


def test_a_candidate_can_be_put_straight_into_the_replacement_box(qapp, tmp_path):
    folder = Folder({RECORDING: make_candidate_transcript()})
    window = open_window(tmp_path, folder, process=True)
    try:
        window.use_selected_candidate()

        assert window._replacement_edit.text() == "15,000"
        # Nothing has changed yet: the replacement still has to be applied.
        assert folder.texts(RECORDING) == ["50,000"]
    finally:
        window.close()


# -- Playback ------------------------------------------------------------


def test_playing_a_word_plays_it_alone_with_only_a_small_margin(qapp, tmp_path):
    """The margin is there because the services' timings are not exact."""
    folder = Folder({RECORDING: make_transcript([weak_token("word", 30.0)])})
    window = open_window(tmp_path, folder, process=True)
    try:
        assert window.span_to_play() == AudioSpan(
            30.0 - WORD_MARGIN_SECONDS, 30.4 + WORD_MARGIN_SECONDS
        )
    finally:
        window.close()


def test_asking_for_some_or_more_context_widens_the_same_span(qapp, tmp_path):
    folder = Folder({RECORDING: make_transcript([weak_token("word", 30.0)])})
    window = open_window(tmp_path, folder, process=True)
    try:
        assert window.span_to_play(SHORT_CONTEXT_SECONDS) == AudioSpan(
            30.0 - SHORT_CONTEXT_SECONDS, 30.4 + SHORT_CONTEXT_SECONDS
        )
        assert window.span_to_play(WIDE_CONTEXT_SECONDS) == AudioSpan(
            30.0 - WIDE_CONTEXT_SECONDS, 30.4 + WIDE_CONTEXT_SECONDS
        )
    finally:
        window.close()


def test_the_padding_never_asks_for_audio_outside_the_recording(qapp, tmp_path):
    folder = Folder(
        {
            RECORDING: make_transcript(
                [weak_token("early", 0.5), weak_token("late", 59.0)], duration=60.0
            )
        }
    )
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "early")
        assert window.span_to_play(WIDE_CONTEXT_SECONDS).start == 0.0

        select_word(window, "late")
        assert window.span_to_play(WIDE_CONTEXT_SECONDS).end == 60.0
    finally:
        window.close()


def test_a_word_with_no_boundaries_plays_the_wider_region_it_was_found_in(qapp, tmp_path):
    """Which is exactly when a person most needs to hear it."""
    token = make_token(start=None, end=None, span=AudioSpan(30.0, 33.0), strength=0.3)
    folder = Folder({RECORDING: make_transcript([token])})
    window = open_window(tmp_path, folder, process=True)
    try:
        assert window.span_to_play() == AudioSpan(
            30.0 - WORD_MARGIN_SECONDS, 33.0 + WORD_MARGIN_SECONDS
        )
    finally:
        window.close()


def test_playing_waits_for_the_media_before_seeking_into_it(qapp, tmp_path):
    """Qt loads a file in the background, so an immediate seek is dropped in silence."""
    player = FakePlayer()
    folder = Folder({RECORDING: make_transcript([weak_token("word", 30.0)])})
    window = open_window(tmp_path, folder, player, process=True)
    try:
        assert window.play_span() is True

        assert player.loaded == str(Path(f"C:/Audio/{RECORDING}.wav"))
        assert player.seeks == []
        assert player.plays == 0
        assert "Loading" in window._status_label.text()

        # Qt says nought first, meaning it does not know yet, and that must
        # not be mistaken for the file being ready.
        player.becomes_ready(0)
        assert player.seeks == []

        player.becomes_ready()
        assert player.seeks == [29_850]
        assert player.plays == 1
    finally:
        window.close()


def test_moving_to_another_recording_loads_that_file_before_playing_it(qapp, tmp_path):
    player = FakePlayer()
    window = open_window(tmp_path, two_file_folder(), player, process=True)
    try:
        select_word(window, "Bosch")
        window.play_span()
        player.becomes_ready()
        first = player.loaded

        for _ in range(2):
            window.go_to_next_item()
            if window.current_occurrence().recording_name != Path(first).stem:
                break
        window.play_span()

        assert len(player.loads) >= 2
        assert player.loads[-1] != player.loads[0]
    finally:
        window.close()


def test_a_second_play_in_the_same_file_does_not_load_it_again(qapp, tmp_path):
    player = FakePlayer()
    folder = Folder({RECORDING: make_transcript([weak_token("word", 30.0)])})
    window = open_window(tmp_path, folder, player, process=True)
    try:
        window.play_span()
        player.becomes_ready()
        window.play_span(WIDE_CONTEXT_SECONDS)

        assert player.loads == [str(Path(f"C:/Audio/{RECORDING}.wav"))]
        assert player.seeks == [29_850, 18_000]
    finally:
        window.close()


def _moved_folder_window(tmp_path, canonical_path: Path, player: FakePlayer) -> ReviewWindow:
    """A window on a folder whose transcript still names where it used to be.

    The recording itself is in ``tmp_path / "new"``, which is where the window
    is told it is, as it would be after the folder was opened from its new
    place.
    """
    transcript = make_transcript([weak_token("word", 30.0)])
    transcript.canonical_audio.path = str(canonical_path)
    folder = Folder({RECORDING: transcript})
    recording = tmp_path / "new" / RECORDING
    recording.parent.mkdir(parents=True, exist_ok=True)
    recording.write_bytes(b"original")
    window = ReviewWindow(
        recording.parent,
        folder.names,
        folder.load,
        {RECORDING: recording},
        ProjectStore(tmp_path),
        player,
        folder.save,
    )
    window.show()
    window.process_low_confidence_words()
    return window


def test_a_moved_folder_plays_the_recording_beside_the_transcript(qapp, tmp_path):
    """The saved path is absolute and names a place the folder has left."""
    player = FakePlayer()
    old = tmp_path / "old" / RECORDING
    window = _moved_folder_window(tmp_path, old, player)
    try:
        assert window.play_span() is True

        assert player.loaded == str(tmp_path / "new" / RECORDING)
    finally:
        window.close()


def test_a_moved_folder_still_plays_the_converted_copy_that_moved_with_it(qapp, tmp_path):
    """The copy's timestamps are the ones to trust, so it wins over the original."""
    player = FakePlayer()
    copy_name = "board meeting-canonical.wav"
    old = tmp_path / "old" / f"{RECORDING}.transcript" / copy_name
    moved_copy = tmp_path / "new" / f"{RECORDING}.transcript" / copy_name
    moved_copy.parent.mkdir(parents=True)
    moved_copy.write_bytes(b"copy")
    window = _moved_folder_window(tmp_path, old, player)
    try:
        assert window.play_span() is True

        assert player.loaded == str(moved_copy)
    finally:
        window.close()


def test_a_converted_copy_that_is_still_where_it_was_saved_is_played(qapp, tmp_path):
    player = FakePlayer()
    copy = tmp_path / "work" / "board meeting-canonical.wav"
    copy.parent.mkdir(parents=True)
    copy.write_bytes(b"copy")
    window = _moved_folder_window(tmp_path, copy, player)
    try:
        assert window.play_span() is True

        assert player.loaded == str(copy)
    finally:
        window.close()


def test_the_recording_is_opened_before_anybody_asks_to_hear_it(qapp, tmp_path):
    """Opening a file takes Qt a moment, which every first clip used to wait for."""
    player = FakePlayer()
    copy = tmp_path / "work" / "board meeting-canonical.wav"
    copy.parent.mkdir(parents=True)
    copy.write_bytes(b"copy")
    window = _moved_folder_window(tmp_path, copy, player)
    try:
        assert player.loaded == str(copy)
        assert player.plays == 0
        player.becomes_ready()

        assert window.play_span() is True

        # Played at once, with no second load and no wait for the file.
        assert player.loads == [str(copy)]
        assert player.seeks == [29_850]
        assert player.plays == 1
        assert window._status_label.text().startswith("Playing")
    finally:
        window.close()


def test_a_recording_that_is_not_there_is_not_opened_until_it_is_asked_for(qapp, tmp_path):
    """Opening a missing file early would raise an error nobody asked to hear."""
    player = FakePlayer()
    folder = Folder({RECORDING: make_transcript([weak_token("word", 30.0)])})
    window = open_window(tmp_path, folder, player, process=True)
    try:
        assert player.loads == []
    finally:
        window.close()


def _two_real_recordings(tmp_path) -> tuple[Folder, dict[str, Path]]:
    """The two-file folder, with an audio file that exists behind each recording."""
    folder = two_file_folder()
    files = {}
    for name, transcript in folder.transcripts.items():
        audio = tmp_path / "audio" / f"{name}.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"audio")
        transcript.canonical_audio.path = str(audio)
        files[name] = audio
    return folder, files


def test_moving_to_a_word_in_another_recording_opens_that_recording_at_once(qapp, tmp_path):
    """The file is ready before the clip is due, rather than loaded when it is."""
    player = FakePlayer()
    folder, files = _two_real_recordings(tmp_path)
    window = open_window(tmp_path, folder, player, process=True)
    try:
        select_word(window, "Bosch")
        first = window.current_occurrence().recording_name
        for _ in range(3):
            if window.current_occurrence().recording_name != first:
                break
            window.go_to_next_item()
        other = window.current_occurrence().recording_name
        assert other != first

        assert player.loaded == str(files[other])
        assert player.plays == 0
        loads = len(player.loads)
        player.becomes_ready()

        assert window.play_span() is True
        assert len(player.loads) == loads
        assert player.plays == 1
    finally:
        window.close()


def test_a_recording_that_fails_to_open_early_is_not_announced_or_opened_again(
    qapp, tmp_path
):
    """Only moving to a word asks for nothing, so its failure is not said yet."""
    player = FakePlayer()
    folder, files = _two_real_recordings(tmp_path)
    window = open_window(tmp_path, folder, player, process=True)
    try:
        select_word(window, "Bosch")
        failing = Path(player.loaded)
        status = window._status_label.text()

        player.errorOccurred.emit(f"{failing.name} could not be played.")
        assert window._status_label.text() == status

        # Moving about, and back onto the same recording, does not open the
        # failed file again.
        loads = player.loads.count(str(failing))
        start = window.current_occurrence().id
        for _ in range(3):
            window.go_to_next_item()
        while window.current_occurrence().id != start:
            before = window.current_occurrence().id
            window.go_to_previous_item()
            assert window.current_occurrence().id != before
        assert window._audio_path() == failing
        assert player.loads.count(str(failing)) == loads

        # Asking for the audio opens it again, and this time a failure is said.
        window.play_span()
        assert player.loads.count(str(failing)) == loads + 1
        player.errorOccurred.emit(f"{failing.name} could not be played.")
        assert window._status_label.text() == f"{failing.name} could not be played."
    finally:
        window.close()


def test_an_error_while_a_clip_plays_is_still_said_out_loud(qapp, tmp_path):
    """Only a failure while opening early is quiet; one during playback is not."""
    player = FakePlayer()
    folder, _files = _two_real_recordings(tmp_path)
    window = open_window(tmp_path, folder, player, process=True)
    try:
        select_word(window, "Bosch")
        player.becomes_ready()
        window.play_span()

        player.errorOccurred.emit("The audio device was disconnected.")
        assert window._status_label.text() == "The audio device was disconnected."
    finally:
        window.close()


def test_a_recording_that_opens_when_asked_is_opened_early_again(qapp, tmp_path):
    """One failure is not held against a file that has since opened."""
    player = FakePlayer()
    folder, _files = _two_real_recordings(tmp_path)
    window = open_window(tmp_path, folder, player, process=True)
    try:
        select_word(window, "Bosch")
        failing = Path(player.loaded)
        player.errorOccurred.emit(f"{failing.name} could not be played.")
        window.play_span()
        player.becomes_ready()

        start = window.current_occurrence().id
        for _ in range(3):
            window.go_to_next_item()
        assert window._audio_path() != failing
        loads = player.loads.count(str(failing))
        while window.current_occurrence().id != start:
            before = window.current_occurrence().id
            window.go_to_previous_item()
            assert window.current_occurrence().id != before

        assert player.loads.count(str(failing)) == loads + 1
    finally:
        window.close()


def test_the_clip_is_stopped_on_time_rather_than_at_the_next_position_report(qapp, tmp_path):
    """Qt reports the position only now and then, so the clip used to run on."""
    player = FakePlayer()
    folder = Folder({RECORDING: make_transcript([weak_token("word", 30.0)])})
    window = open_window(tmp_path, folder, player, process=True)
    try:
        window.play_span()
        player.becomes_ready()

        assert window._stop_timer.isActive()
        assert window._stop_timer.interval() == 30_550 - 29_850

        window._stop_timer.timeout.emit()
        assert player.pauses == 1

        # The position report that arrives afterwards does not pause again.
        player.positionChanged.emit(30_600)
        assert player.pauses == 1
    finally:
        window.close()


def test_playback_stops_at_the_end_of_the_region_rather_than_running_on(qapp, tmp_path):
    player = FakePlayer()
    folder = Folder({RECORDING: make_transcript([weak_token("word", 30.0)])})
    window = open_window(tmp_path, folder, player, process=True)
    try:
        window.play_span()
        player.becomes_ready()

        # The word ends at 30.4 seconds, and its last sound must not be cut
        # off, so it plays on through it into the margin.
        player.positionChanged.emit(30_400)
        player.positionChanged.emit(30_500)
        assert player.pauses == 0

        player.positionChanged.emit(30_600)
        assert player.pauses == 1

        # And having stopped, it stays stopped rather than pausing again on
        # every position the player reports afterwards.
        player.positionChanged.emit(31_000)
        assert player.pauses == 1
    finally:
        window.close()


def test_automatic_playback_plays_the_word_alone(qapp, tmp_path):
    """Landing on an occurrence plays the word, not the sentence around it.

    The sentence is a keypress away on Shift+F5. Playing it every time buried
    the word the person had come to hear among the ones either side of it.
    """
    player = FakePlayer()
    window = open_window(tmp_path, two_file_folder(), player, process=True)
    try:
        select_word(window, "Bosch")
        window.go_to_next_item()
        window._play_after_waiting()
        player.becomes_ready()

        span = window.current_token().audible_span
        assert player.seeks == [round((span.start - WORD_MARGIN_SECONDS) * 1000)]
        assert window._stop_at_ms == round((span.end + WORD_MARGIN_SECONDS) * 1000)
    finally:
        window.close()


def test_each_play_key_has_a_menu_item_that_names_it(qapp, tmp_path):
    """The menu is where a person finds out which key plays how much."""
    window = open_window(tmp_path)
    try:
        playback = [
            (action.text(), action.shortcut())
            for action in (
                window._play_action,
                window._play_short_action,
                window._play_wide_action,
            )
        ]
        assert playback == [
            ("Play the &Word", QKeySequence(Qt.Key.Key_F5)),
            ("Play with &Some Context", QKeySequence("Shift+F5")),
            ("Play with &More Context", QKeySequence("Ctrl+F5")),
        ]
    finally:
        window.close()


def test_each_play_key_plays_the_length_it_names(qapp, tmp_path):
    player = FakePlayer()
    folder = Folder({RECORDING: make_transcript([weak_token("word", 30.0)])})
    window = open_window(tmp_path, folder, player, process=True)
    try:
        window._play_action.trigger()
        player.becomes_ready()
        window._play_short_action.trigger()
        window._play_wide_action.trigger()

        assert player.seeks == [29_850, 28_500, 18_000]
        assert window._stop_at_ms == 42_400
    finally:
        window.close()


def test_a_word_with_no_audio_is_not_played_by_itself_either(qapp, tmp_path):
    """Nothing is lined up, and nothing new is said on arriving at it."""
    player = FakePlayer()
    token = make_token(start=None, end=None, span=None, strength=0.3)
    folder = Folder({RECORDING: make_transcript([token])})
    window = open_window(tmp_path, folder, player, process=True)
    try:
        window._start_automatic_playback()
        assert not window._play_timer.isActive()

        assert window.play_span() is False
        assert window._status_label.text() == review_window_module.NO_AUDIO
        assert player.loads == []
    finally:
        window.close()


def test_a_word_with_no_audio_at_all_says_so_instead_of_playing_silence(qapp, tmp_path):
    player = FakePlayer()
    token = make_token(start=None, end=None, span=None, strength=0.3)
    folder = Folder({RECORDING: make_transcript([token])})
    window = open_window(tmp_path, folder, player, process=True)
    try:
        assert window.play_span() is False

        assert player.plays == 0
        assert not window._play_button.isEnabled()
        assert "no audio behind it" in window._span_edit.text()
    finally:
        window.close()


def test_the_field_says_what_pressing_play_would_actually_play(qapp, tmp_path):
    folder = Folder({RECORDING: make_transcript([weak_token("word", 30.0)])})
    window = open_window(tmp_path, folder, process=True)
    try:
        assert window._span_edit.text() == "From 30 seconds, 0.7 seconds in all"
    finally:
        window.close()


def test_a_recording_with_no_audio_file_says_so_rather_than_failing(qapp, tmp_path):
    player = FakePlayer()
    folder = Folder(
        {RECORDING: make_transcript([weak_token("word", 30.0)], with_audio=False)}
    )
    window = ReviewWindow(
        tmp_path,
        folder.names,
        folder.load,
        {},
        ProjectStore(tmp_path),
        player,
        folder.save,
    )
    window.show()
    window.process_low_confidence_words()
    try:
        assert player.loaded is None
        assert window.play_span() is False
        assert "no audio file for this recording" in window._span_edit.text()
    finally:
        window.close()


def test_a_failure_from_the_player_is_said_out_loud(qapp, tmp_path):
    player = FakePlayer()
    folder = Folder({RECORDING: make_transcript([weak_token("word", 30.0)])})
    window = open_window(tmp_path, folder, player, process=True)
    try:
        window.play_span()
        player.errorOccurred.emit("board meeting.wav could not be opened.")

        assert window._status_label.text() == "board meeting.wav could not be opened."
        # And the file is forgotten, so asking again reloads rather than
        # waiting for a readiness that will never arrive.
        assert window._loaded_path is None
    finally:
        window.close()


# -- Automatic playback ---------------------------------------------------


def test_landing_on_an_occurrence_lines_the_audio_up_without_being_asked(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        window.go_to_next_item()

        assert window._play_timer.isActive()
        assert window._play_timer.isSingleShot()
    finally:
        window.close()


def test_automatic_playback_never_moves_the_focus(qapp, tmp_path):
    """Being taken out of the list by the sound arriving is worse than no sound."""
    player = FakePlayer()
    window = open_window(tmp_path, two_file_folder(), player, process=True)
    try:
        select_word(window, "Bosch")
        window._occurrences.setFocus(Qt.FocusReason.TabFocusReason)
        window.go_to_next_item()
        window._play_after_waiting()
        player.becomes_ready()

        assert window.focusWidget() is window._occurrences
        assert player.plays == 1
    finally:
        window.close()


def test_holding_the_arrow_down_does_not_queue_a_clip_for_every_occurrence(qapp, tmp_path):
    """Each move restarts the wait, so only the one stopped on is ever played."""
    player = FakePlayer()
    window = open_window(tmp_path, two_file_folder(), player, process=True)
    try:
        select_word(window, "Bosch")
        window.go_to_next_item()
        window.go_to_next_item()

        # The timer is still waiting, and nothing has been asked of the player.
        # This holds with no wait set, which is the default.
        assert window._state.settings.auto_play_delay_seconds == 0
        assert window._play_timer.isActive()
        assert player.loads == []

        window._play_after_waiting()
        assert len(player.loads) == 1
    finally:
        window.close()


def test_switching_automatic_playback_off_is_remembered_with_the_folder(qapp, tmp_path):
    store = ProjectStore(tmp_path)
    window = open_window(tmp_path, two_file_folder(), store=store, process=True)
    try:
        window.set_play_automatically(False)

        assert store.load().settings.play_automatically is False
        assert window._auto_play_box.isChecked() is False
        assert window._auto_play_action.isChecked() is False

        select_word(window, "Bosch")
        window.go_to_next_item()
        assert not window._play_timer.isActive()
    finally:
        window.close()


# -- Correcting a word, which reaches every file it is in -----------------


def test_a_group_replacement_changes_every_occurrence_in_every_file(qapp, tmp_path):
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosch")

        assert window.apply_replacement_to_word() is True

        assert folder.texts(RECORDING) == ["Bosch", "Bosch", "15,000"]
        assert folder.texts(OTHER_RECORDING) == ["Bosch", "settled"]
    finally:
        window.close()


def test_a_group_replacement_touches_no_speaker_and_no_timing(qapp, tmp_path):
    """The spine of the design: three answers, and this one changes one of them."""
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosch")
        window.apply_replacement_to_word()

        for name in (RECORDING, OTHER_RECORDING):
            for token in folder.transcripts[name].tokens:
                assert token.speaker == "speaker_0"
                assert token.speaker_source is Provider.ASSEMBLYAI
                assert token.timing_status is TimingStatus.EXACT_PROVIDER_TIME
                assert token.timing_source is Provider.ELEVENLABS
                assert token.start is not None
    finally:
        window.close()


def test_the_change_is_announced_with_its_blast_radius_counted(qapp, tmp_path, monkeypatch):
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        group_id = window.current_row().group_id
        expected = affected_summary(window.state, group_id)
        said.clear()
        window._replacement_edit.setText("Bosch")
        window.apply_replacement_to_word()

        assert len(said) == 1
        # The wording on the screen and the wording read out come from the
        # same sentence, so they cannot drift apart.
        assert expected == "This replacement applies to 3 occurrences across 2 files."
        assert expected in said[0]
        assert "The speaker and the timing of every one of them are unchanged." in said[0]
    finally:
        window.close()


def test_the_field_says_the_blast_radius_before_anything_is_changed(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")

        assert window._affected_edit.text() == (
            "This replacement applies to 3 occurrences across 2 files."
        )
    finally:
        window.close()


def test_editing_a_replacement_twice_gives_what_typing_it_once_would_have(qapp, tmp_path):
    """Each occurrence is rewritten from what it originally said, never from now."""
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosch")
        window.apply_replacement_to_word()
        # The word has been settled, so it leaves the list until asked for.
        window.set_show_reviewed(True)
        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosche")
        window.apply_replacement_to_word()

        assert folder.texts(RECORDING) == ["Bosche", "Bosche", "15,000"]
        # What each word originally said is still what the services produced,
        # not the first attempt at correcting it.
        originals = [token.original_text for token in folder.transcripts[RECORDING].tokens]
        assert originals[:2] == ["Bosch", "Bosh"]
        for occurrence in window.state.occurrences:
            assert occurrence.detected_text in ("Bosch", "Bosh", "Bosche")
    finally:
        window.close()


def test_a_word_replacement_makes_one_rule_for_each_detected_form(qapp, tmp_path):
    """So a file transcribed next week saying Bosh is answered by a Bosh decision."""
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        group_id = window.current_row().group_id
        window._replacement_edit.setText("Bosch")
        window.apply_replacement_to_word()

        rules = window.state.rules
        assert sorted(rule.matched_text for rule in rules) == ["Bosch", "Bosche", "Bosh"]
        assert {rule.replacement for rule in rules} == {"Bosch"}
        # Each rule names the word it came from, so that changing your mind
        # about that word can find them again.
        assert {rule.group_id for rule in rules} == {group_id}
    finally:
        window.close()


def test_changing_your_mind_updates_the_rules_rather_than_adding_more(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosch")
        window.apply_replacement_to_word()
        # The word has been settled, so it leaves the list until asked for.
        window.set_show_reviewed(True)
        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosche")
        window.apply_replacement_to_word()

        assert len(window.state.rules) == 3
        assert {rule.replacement for rule in window.state.rules} == {"Bosche"}
    finally:
        window.close()


def test_correcting_a_word_as_detected_puts_the_words_back_and_drops_its_rules(
    qapp, tmp_path
):
    """A withdrawn decision must not go on rewriting files nobody has opened."""
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosch")
        window.apply_replacement_to_word()
        # The word has been settled, so it leaves the list until asked for.
        window.set_show_reviewed(True)
        select_word(window, "Bosch")

        assert window.correct_word_as_detected() is True

        assert folder.texts(RECORDING) == ["Bosch", "Bosh", "15,000"]
        assert folder.texts(OTHER_RECORDING) == ["Bosche", "settled"]
        assert window.state.rules == []
    finally:
        window.close()


# -- The names a folder's reviews teach -----------------------------------


def test_a_replacement_teaches_the_folder_the_name(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosch")

        assert window.apply_replacement_to_word() is True

        [name] = ProjectStore(tmp_path).load().learned_names
        assert name.text == "Bosch"
        assert name.language == Language.ENGLISH.value
        assert "Bosh" in name.wrong_forms
        assert "Bosch" not in name.wrong_forms
    finally:
        window.close()


def test_a_replacement_of_one_occurrence_teaches_the_name_too(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        for row in range(window._occurrence_model.rowCount()):
            window._occurrences.select_row(row)
            if window.current_occurrence().detected_text == "Bosh":
                break
        assert window.current_occurrence().detected_text == "Bosh"
        window._replacement_edit.setText("Bosch")

        assert window.apply_replacement_to_occurrence() is True

        names = ProjectStore(tmp_path).load().learned_names
        assert [(name.text, name.wrong_forms) for name in names] == [("Bosch", ["Bosh"])]
    finally:
        window.close()


def test_a_correction_that_is_not_a_name_teaches_nothing(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "15,000")
        window._replacement_edit.setText("50,000")

        assert window.apply_replacement_to_word() is True

        assert ProjectStore(tmp_path).load().learned_names == []
    finally:
        window.close()


def test_undoing_a_correction_keeps_the_name_it_taught(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosch")
        window.apply_replacement_to_word()
        window.set_show_reviewed(True)
        select_word(window, "Bosch")

        assert window.correct_word_as_detected() is True

        names = ProjectStore(tmp_path).load().learned_names
        assert [name.text for name in names] == ["Bosch"]
    finally:
        window.close()


def test_a_name_removed_while_the_window_is_open_stays_removed(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosch")
        window.apply_replacement_to_word()
        assert remove_learned_name(tmp_path, "Bosch", Language.ENGLISH.value) is True

        # Another change saves the project again from the window's copy.
        select_word(window, "15,000")
        window._replacement_edit.setText("50,000")
        assert window.apply_replacement_to_word() is True

        assert ProjectStore(tmp_path).load().learned_names == []
    finally:
        window.close()


def test_a_review_never_writes_the_shared_vocabulary(qapp, tmp_path, monkeypatch):
    written: list[object] = []
    monkeypatch.setattr(
        VocabularyStore, "save", lambda store, vocabulary: written.append(vocabulary) or True
    )
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosch")
        window.apply_replacement_to_word()
    finally:
        window.close()

    assert written == []


def test_a_settled_word_leaves_the_list_and_the_work_carries_on(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        window.correct_word_as_detected()

        assert [row.word for row in window._group_model.rows()] == ["15,000"]
        assert window.current_row().word == "15,000"
    finally:
        window.close()


def test_an_empty_replacement_is_refused_rather_than_deleting_the_word(qapp, tmp_path):
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        window._replacement_edit.setText("   ")

        assert window.apply_replacement_to_word() is False

        assert folder.texts(RECORDING) == ["Bosch", "Bosh", "15,000"]
        assert "cannot be empty" in window._status_label.text()
    finally:
        window.close()


def test_a_replacement_that_changes_nothing_is_refused_and_said_out_loud(qapp, tmp_path):
    folder = Folder({RECORDING: make_transcript([weak_token("contract", 10.0)])})
    window = open_window(tmp_path, folder, process=True)
    try:
        window._replacement_edit.setText("contract")

        assert window.apply_replacement_to_word() is False

        assert "unchanged" in window._status_label.text()
        assert folder.saved == []
    finally:
        window.close()


# -- Correcting one occurrence --------------------------------------------


def test_one_occurrence_can_be_replaced_without_touching_the_others(qapp, tmp_path):
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        occurrence = window.current_occurrence()
        window._replacement_edit.setText("Bosche")

        assert window.apply_replacement_to_occurrence() is True

        changed = folder.token(occurrence.recording_name, occurrence.token_id)
        assert changed.text == "Bosche"
        assert len([token for token in folder.transcripts[RECORDING].tokens
                    if token.text == "Bosche"]) == 1
    finally:
        window.close()


def test_an_occurrence_with_its_own_answer_keeps_it_when_the_word_changes(qapp, tmp_path):
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        occurrence = window.current_occurrence()
        window._replacement_edit.setText("Bosche")
        window.apply_replacement_to_occurrence()

        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosch")
        window.apply_replacement_to_word()

        assert folder.token(occurrence.recording_name, occurrence.token_id).text == "Bosche"
    finally:
        window.close()


def test_correcting_the_speaker_leaves_the_text_and_the_timing_alone(qapp, tmp_path):
    folder = Folder({RECORDING: make_transcript([weak_token("contract", 30.0)])})
    window = open_window(tmp_path, folder, process=True)
    try:
        token_id = window.current_occurrence().token_id
        window._speaker_box.setCurrentText("Speaker speaker_1")

        assert window.apply_speaker_correction() is True

        corrected = folder.token(RECORDING, token_id)
        assert corrected.speaker == "speaker_1"
        assert corrected.speaker_confidence is Confidence.HIGH
        assert corrected.text == "contract"
        assert corrected.text_source is Provider.ELEVENLABS
        assert corrected.start == 30.0
        assert corrected.timing_status is TimingStatus.EXACT_PROVIDER_TIME
    finally:
        window.close()


def test_correcting_the_speaker_says_it_reached_no_other_occurrence(qapp, tmp_path):
    """Two mentions of one surname are as likely to be two people as one."""
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        window._speaker_box.setCurrentText("the chairman")
        window.apply_speaker_correction()

        assert "no other occurrence of this word is affected" in window._status_label.text()
    finally:
        window.close()


def test_a_speaker_chosen_by_name_is_recorded_by_the_label_the_services_used(qapp, tmp_path):
    """Saving the friendly name would key the word on something nothing else knows."""
    folder = Folder(
        {RECORDING: make_transcript([weak_token("contract", 30.0)])}
    )
    folder.transcripts[RECORDING].tokens[0].speaker = "speaker_1"
    window = open_window(tmp_path, folder, process=True)
    try:
        token_id = window.current_occurrence().token_id
        window._speaker_box.setCurrentText("Jacques")

        assert window.chosen_speaker() == "speaker_0"

        window.apply_speaker_correction()

        assert folder.token(RECORDING, token_id).speaker == "speaker_0"
    finally:
        window.close()


def test_a_speaker_the_services_never_separated_out_can_still_be_typed(qapp, tmp_path):
    folder = Folder({RECORDING: make_transcript([weak_token("contract", 30.0)])})
    window = open_window(tmp_path, folder, process=True)
    try:
        token_id = window.current_occurrence().token_id
        window._speaker_box.setCurrentText("the chairman")

        assert window.apply_speaker_correction() is True

        assert folder.token(RECORDING, token_id).speaker == "the chairman"
    finally:
        window.close()


def test_confirming_the_timing_leaves_the_text_and_the_speaker_alone(qapp, tmp_path):
    token = make_token(
        "contract",
        reasons=(ReviewReason.WEAK_ALIGNMENT,),
        confidence=Confidence.REVIEW_SUGGESTED,
    )
    token.timing_confidence = Confidence.REVIEW_REQUIRED
    folder = Folder({RECORDING: make_transcript([token])})
    window = open_window(tmp_path, folder, process=True)
    try:
        assert window.decide_timing(True) is True

        decided = folder.token(RECORDING, token.id)
        assert decided.timing_confidence is Confidence.HIGH
        assert decided.start == 30.0
        assert decided.text == "contract"
        assert decided.text_confidence is Confidence.REVIEW_SUGGESTED
        assert decided.speaker == "speaker_0"
        assert decided.review_reasons == []
        assert decided.needs_review is False
    finally:
        window.close()


def test_confirming_the_timing_leaves_the_word_in_the_list_for_its_other_reasons(
    qapp, tmp_path
):
    """A word also disputed on spelling is not settled by settling its clock."""
    token = make_token(
        "15,000",
        reasons=(ReviewReason.WEAK_ALIGNMENT, ReviewReason.NUMERIC_DISAGREEMENT),
    )
    folder = Folder({RECORDING: make_transcript([token])})
    window = open_window(tmp_path, folder, process=True)
    try:
        window.decide_timing(True)

        decided = folder.token(RECORDING, token.id)
        assert decided.review_reasons == [ReviewReason.NUMERIC_DISAGREEMENT]
        assert decided.needs_review is True
        assert window._group_model.rowCount() == 1
    finally:
        window.close()


def test_rejecting_the_timing_keeps_the_numbers_so_the_word_can_still_be_played(
    qapp, tmp_path
):
    token = make_token("contract", start=30.0, end=30.5)
    folder = Folder({RECORDING: make_transcript([token])})
    window = open_window(tmp_path, folder, process=True)
    try:
        assert window.decide_timing(False) is True

        decided = folder.token(RECORDING, token.id)
        assert decided.start == 30.0
        assert decided.end == 30.5
        assert decided.timing_status is TimingStatus.UNCERTAIN
        assert decided.timing_confidence is Confidence.UNRESOLVED
        assert ReviewReason.WEAK_ALIGNMENT in decided.review_reasons
        assert decided.needs_review is True
        assert decided.text == "contract"
        assert decided.speaker == "speaker_0"
        assert window.span_to_play() is not None
    finally:
        window.close()


def test_confirming_an_occurrence_settles_it_without_changing_a_thing(qapp, tmp_path):
    """The common case, and the fastest thing on the screen."""
    token = make_token("contract", confidence=Confidence.UNRESOLVED)
    folder = Folder({RECORDING: make_transcript([token])})
    window = open_window(tmp_path, folder, process=True)
    try:
        assert window.confirm_item() is True

        confirmed = folder.token(RECORDING, token.id)
        assert confirmed.text == "contract"
        assert confirmed.start == 30.0
        assert confirmed.speaker == "speaker_0"
        assert confirmed.review_status is ReviewStatus.CONFIRMED
        assert confirmed.review_reasons == []
        assert confirmed.needs_review is False
        assert confirmed.confidence is Confidence.HIGH
        # Nothing changed, so there is nothing for the vocabulary to learn.
        assert confirmed.human_corrected is False
        assert confirmed.original_text is None
    finally:
        window.close()


def test_f4_moves_to_the_next_occurrence_of_the_word(qapp, tmp_path):
    """Confirming is done one occurrence after another, without F3 in between."""
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        assert window._occurrence_model.rowCount() == 3
        window._occurrences.select_row(0)
        second = window._occurrence_model.occurrence_at(1)

        assert window.confirm_item() is True

        assert window.current_row().word == "Bosch"
        assert window.current_occurrence().id == second.id
    finally:
        window.close()


def test_f4_on_the_last_occurrence_moves_to_the_next_word(qapp, tmp_path, monkeypatch):
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        window.set_show_reviewed(True)
        select_word(window, "Bosch")
        window._occurrences.select_row(window._occurrence_model.rowCount() - 1)
        said.clear()

        window.confirm_item()

        assert window.current_row().word == "15,000"
        assert window._occurrences.selected_row() == 0
        assert len(said) == 1
        assert "Now on 15,000." in said[0]
    finally:
        window.close()


def test_confirming_every_occurrence_marks_the_word_reviewed(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        group = window.state.group(window.current_row().group_id)
        window._occurrences.select_row(0)
        for _ in range(3):
            assert window.current_row().word == "Bosch"
            window.confirm_item()

        assert group.reviewed is True
        assert group.correct_as_detected is True
        assert "Bosch" not in [row.word for row in window._group_model.rows()]
    finally:
        window.close()


def test_confirming_a_replaced_occurrence_does_not_call_it_correct_as_detected(
    qapp, tmp_path
):
    """A group replacement changed the word, so it was not right as detected."""
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosch")
        assert window.apply_replacement_to_word() is True
        window.set_show_reviewed(True)
        select_word(window, "Bosch")
        window._occurrences.select_row(1)
        occurrence = window.current_occurrence()

        assert window.confirm_item() is True

        assert occurrence.reviewed is True
        assert occurrence.correct_as_detected is False
        assert review_lists.occurrence_reviewed_text(occurrence) != REVIEWED_AS_DETECTED
        assert folder.token(occurrence.recording_name, occurrence.token_id).text == "Bosch"
        assert window.state.group(group_row(window, "Bosch").group_id).replacement == "Bosch"
    finally:
        window.close()


def test_f4_on_the_very_last_item_stays_and_says_so(qapp, tmp_path):
    folder = Folder({RECORDING: make_transcript([weak_token("contract", 30.0)])})
    window = open_window(tmp_path, folder, process=True)
    try:
        window.set_show_reviewed(True)
        select_word(window, "contract")

        assert window.confirm_item() is True

        assert window.current_row().word == "contract"
        assert "That was the last occurrence of the last word in the list." in (
            window._status_label.text()
        )
    finally:
        window.close()


def test_f4_keeps_the_focus_in_the_occurrence_list(qapp, tmp_path, monkeypatch):
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    folder = Folder(
        {
            RECORDING: make_transcript(
                [weak_token("Bosch", 10.0, 0.42), weak_token("Bosh", 40.0, 0.45)]
            )
        }
    )
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        assert window._occurrence_model.rowCount() == 2
        window._occurrences.select_row(0)
        window._occurrences.setFocus(Qt.FocusReason.TabFocusReason)
        said.clear()

        window.confirm_item()

        assert window.focusWidget() is window._occurrences
        assert window._occurrences.selected_row() == 1
        assert len(said) == 1
    finally:
        window.close()


def test_a_correction_is_announced_along_with_where_the_person_now_is(qapp, tmp_path, monkeypatch):
    """The change happens away from where the focus lands next."""
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        said.clear()
        window.confirm_item()

        assert len(said) == 1
        assert said[0].startswith("Bosch confirmed as correct in ")
        assert "Now on" in said[0]
    finally:
        window.close()


def test_the_last_correction_says_the_list_is_empty_rather_than_going_quiet(
    qapp, tmp_path, monkeypatch
):
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    folder = Folder({RECORDING: make_transcript([weak_token("one", 10.0)])})
    window = open_window(tmp_path, folder, process=True)
    try:
        said.clear()
        window.correct_word_as_detected()

        assert len(said) == 1
        assert said[0].startswith("one confirmed as correct as detected.")
        assert "There is nothing to review." in said[0]
    finally:
        window.close()


def test_every_correction_is_handed_to_the_callback_and_the_signal(qapp, tmp_path):
    """The window knows nothing about storage, so somebody else must be told."""
    folder = Folder({RECORDING: make_transcript([weak_token("one", 10.0)])})
    emitted: list[tuple[str, Transcript]] = []
    window = open_window(tmp_path, folder, process=True)
    window.transcriptChanged.connect(lambda name, transcript: emitted.append((name, transcript)))
    try:
        folder.saved.clear()
        window.confirm_item()

        assert len(folder.saved) == 1
        assert folder.saved[0][0] == RECORDING
        assert emitted == folder.saved
    finally:
        window.close()


def test_correcting_does_not_reach_back_into_the_transcript_it_replaced(qapp, tmp_path):
    """The copies share their lists, so a correction must rebind rather than edit."""
    token = make_token("one", reasons=(ReviewReason.HIGH_RISK_ENTITY,))
    original = make_transcript([token])
    folder = Folder({RECORDING: original})
    before = list(token.review_reasons)
    window = open_window(tmp_path, folder, process=True)
    try:
        window.confirm_item()

        assert original.tokens[0].review_reasons == before
        assert original.tokens[0].review_status is ReviewStatus.PENDING
    finally:
        window.close()


# -- Taking an occurrence out of its word ---------------------------------


def test_isolating_takes_an_occurrence_out_of_its_word_and_leaves_it_reachable(
    qapp, tmp_path
):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        occurrence = window.current_occurrence()

        assert window.isolate_occurrence() is True

        assert occurrence.isolated is True
        assert window.state.group_of(occurrence.id) is None
        # And it is still a row of its own, so it can be played and decided.
        words = [row.word for row in window._group_model.rows()]
        assert occurrence.detected_text in words
    finally:
        window.close()


def test_isolating_says_what_is_left_of_the_word_it_came_out_of(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        window.isolate_occurrence()

        assert "This replacement applies to 2 occurrences" in window._status_label.text()
        assert "no longer part of" in window._status_label.text()
    finally:
        window.close()


def test_an_isolated_occurrence_is_not_gathered_up_again_by_regrouping(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        occurrence = window.current_occurrence()
        window.isolate_occurrence()

        window.regroup_words()

        assert window.state.group_of(occurrence.id) is None
    finally:
        window.close()


def test_isolating_is_offered_only_where_there_is_a_group_to_leave(qapp, tmp_path):
    folder = Folder({RECORDING: make_transcript([weak_token("alone", 10.0)])})
    window = open_window(tmp_path, folder, process=True)
    try:
        assert not window._isolate_button.isEnabled()
        assert not window._isolate_action.isEnabled()
    finally:
        window.close()


# -- Reviewed words, and how they were reviewed ---------------------------


def test_a_reviewed_word_is_hidden_until_it_is_asked_for(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        window.correct_word_as_detected()
        assert "Bosch" not in [row.word for row in window._group_model.rows()]

        window.set_show_reviewed(True)

        assert "Bosch" in [row.word for row in window._group_model.rows()]
    finally:
        window.close()


def test_the_reviewed_state_is_a_word_that_says_which_kind_it_was(qapp, tmp_path):
    """A word you judged and a word a rule judged for you are different facts."""
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        window.correct_word_as_detected()
        window.set_show_reviewed(True)

        assert group_row(window, "Bosch").reviewed == REVIEWED_AS_DETECTED
    finally:
        window.close()


def test_a_word_a_rule_settled_says_so_rather_than_looking_like_your_own_work(
    qapp, tmp_path
):
    folder = two_file_folder()
    store = ProjectStore(tmp_path)
    window = open_window(tmp_path, folder, store=store, process=True)
    try:
        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosch")
        window.apply_replacement_to_word()
    finally:
        window.close()

    # A recording arrives later holding a spelling a rule already answers.
    folder.transcripts["late.m4a"] = make_transcript(
        [weak_token("Bosh", 5.0)], name="late.m4a"
    )
    second = open_window(tmp_path, folder, store=store)
    try:
        second.process_low_confidence_words()
        second.set_show_reviewed(True)

        late = [
            occurrence
            for occurrence in second.state.occurrences
            if occurrence.recording_name == "late.m4a"
        ]
        assert len(late) == 1
        assert late[0].auto_applied is True
        assert folder.texts("late.m4a") == ["Bosch"]
        row = next(
            row
            for row in second._group_model.rows()
            if any(item.id == late[0].id for item in row.occurrences)
        )
        assert row.reviewed == REVIEWED_AUTOMATICALLY
    finally:
        second.close()


def test_a_group_replacement_reaches_a_word_a_rule_answered(qapp, tmp_path):
    """Correcting a rule's answer changes the file and leaves one rule saying so."""
    folder = two_file_folder()
    store = ProjectStore(tmp_path)
    window = open_window(tmp_path, folder, store=store, process=True)
    try:
        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosch")
        window.apply_replacement_to_word()
    finally:
        window.close()

    folder.transcripts["late.m4a"] = make_transcript(
        [weak_token("Bosh", 5.0)], name="late.m4a"
    )
    second = open_window(tmp_path, folder, store=store)
    try:
        second.process_low_confidence_words()
        second.set_show_reviewed(True)
        late = next(
            occurrence
            for occurrence in second.state.occurrences
            if occurrence.recording_name == "late.m4a"
        )
        assert late.auto_applied is True
        row = next(
            row
            for row in second._group_model.rows()
            if any(item.id == late.id for item in row.occurrences)
        )
        second._groups.select_row(second._group_model.row_for_key(row.key))
        second._replacement_edit.setText("Bosche")

        assert second.apply_replacement_to_word() is True

        assert folder.texts("late.m4a") == ["Bosche"]
        assert late.auto_applied is False
        assert late.applied_rule_id is None
        assert late.replacement is None
        rules = [rule for rule in second.state.rules if rule.normalised_text == "bosh"]
        assert len(rules) == 1
        assert rules[0].replacement == "Bosche"
    finally:
        second.close()


def test_a_group_replacement_reaches_a_rule_answer_but_not_a_persons_own(
    qapp, tmp_path
):
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        members = list(window.current_row().occurrences)
        assert len(members) >= 2
        own, answered = members[0], members[1]
        answered.replacement = "Bosch"
        answered.auto_applied = True
        answered.applied_rule_id = "an-earlier-rule"
        window._occurrences.select_row(window._occurrence_model.row_for_id(own.id))
        assert window.current_occurrence().id == own.id
        window._replacement_edit.setText("Bosche")
        assert window.apply_replacement_to_occurrence() is True

        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosk")
        assert window.apply_replacement_to_word() is True

        assert folder.token(own.recording_name, own.token_id).text == "Bosche"
        assert own.replacement == "Bosche"
        assert folder.token(answered.recording_name, answered.token_id).text == "Bosk"
        assert answered.auto_applied is False
        assert answered.replacement is None
    finally:
        window.close()


def test_the_count_says_how_many_reviewed_words_are_being_kept_back(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        window.correct_word_as_detected()

        assert "1 reviewed word is hidden." in window._count_label.text()
    finally:
        window.close()


def test_the_second_block_can_be_switched_off_and_the_count_says_so(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        window.set_show_other_uncertainties(False)

        assert [row.word for row in window._group_model.rows()] == ["Bosch"]
        assert "1 other uncertainty, hidden" in window._count_label.text()
    finally:
        window.close()


# -- The analysis and its settings ----------------------------------------


def test_processing_finds_words_nothing_in_the_pipeline_had_flagged(qapp, tmp_path):
    """A word can be quietly weak with nothing but a low number to say so."""
    window = open_window(tmp_path, two_file_folder())
    try:
        # Nothing at all before the folder is processed, not even the
        # flagged words, because finding those means reading transcripts.
        assert window._group_model.rowCount() == 0

        window.process_low_confidence_words()

        assert [row.word for row in window._group_model.rows()] == ["Bosch", "15,000"]
        assert "Processed 2 recordings" in window._status_label.text()
        assert "3 words to look at, in 1 group" in window._status_label.text()
    finally:
        window.close()


def settle(spin) -> None:
    """Finish with a spin box, the way leaving it or pressing Enter does.

    Setting the value is only half of what a person does to a spin box. The
    window deliberately keeps the number and the decision apart: the number is
    taken on every intermediate value so nothing can read a stale threshold,
    and the saving and the announcement wait for this.
    """
    spin.editingFinished.emit()


def test_the_two_thresholds_are_remembered_and_take_effect_on_regrouping(qapp, tmp_path):
    store = ProjectStore(tmp_path)
    folder = Folder(
        {RECORDING: make_transcript([weak_token("borderline", 10.0, 0.6)])}
    )
    window = open_window(tmp_path, folder, store=store, process=True)
    try:
        assert window._group_model.rowCount() == 0

        window._minimum_spin.setValue(70)
        settle(window._minimum_spin)

        assert store.load().settings.minimum_confidence == 0.7
        assert "Regroup the words" in window._status_label.text()
        # Nothing has moved yet, on purpose: the list must not rearrange
        # itself while somebody is still typing a number into a spin box.
        assert window._group_model.rowCount() == 0

        window.regroup_words()

        assert [row.word for row in window._group_model.rows()] == ["borderline"]
    finally:
        window.close()


def test_the_grouping_tolerance_is_remembered_too(qapp, tmp_path):
    store = ProjectStore(tmp_path)
    window = open_window(tmp_path, two_file_folder(), store=store, process=True)
    try:
        window._tolerance_spin.setValue(12)
        settle(window._tolerance_spin)

        assert store.load().settings.grouping_tolerance == 0.12
    finally:
        window.close()


def test_a_threshold_moved_by_the_keyboard_is_used_before_it_has_been_finished_with(
    qapp, tmp_path
):
    """Regrouping is a shortcut, and a shortcut moves no focus.

    So Ctrl+G can be pressed while the spin box still has the focus and its
    editing has therefore not finished. The number the person can see must be
    the number the analysis uses, which is why the value is taken on every
    change even though nothing is said or saved until they have finished.
    """
    folder = Folder({RECORDING: make_transcript([weak_token("borderline", 10.0, 0.6)])})
    window = open_window(tmp_path, folder, process=True)
    try:
        window._minimum_spin.setValue(70)

        window.regroup_words()

        assert [row.word for row in window._group_model.rows()] == ["borderline"]
    finally:
        window.close()


def test_passing_through_a_value_says_nothing_and_writes_nothing(qapp, tmp_path, monkeypatch):
    """Typing "55" is two keystrokes, and holding an arrow key is sixty.

    Every one of them used to be a whole spoken sentence that could not be
    interrupted or typed through, and a write of the project file underneath
    it. Nothing at all happens until the person has finished with the box.
    """
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    store = ProjectStore(tmp_path)
    window = open_window(tmp_path, two_file_folder(), store=store, process=True)
    try:
        writes = 0
        real_save = store.save

        def counted(state):
            nonlocal writes
            writes += 1
            return real_save(state)

        monkeypatch.setattr(store, "save", counted)
        said.clear()

        for percentage in range(50, 61):
            window._minimum_spin.setValue(percentage)

        assert said == []
        assert writes == 0

        settle(window._minimum_spin)

        assert said == [
            (
                "The minimum confidence is now 60 percent. Regroup the words, with "
                "Ctrl+G, to apply it."
            )
        ]
        assert writes == 1
    finally:
        window.close()


def test_a_spin_box_that_was_only_visited_announces_nothing(qapp, tmp_path, monkeypatch):
    """Qt raises editingFinished for a box somebody merely tabbed through."""
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        said.clear()
        settle(window._minimum_spin)
        settle(window._tolerance_spin)
        settle(window._delay_spin)

        assert said == []
    finally:
        window.close()


def test_regrouping_keeps_every_decision_that_was_already_made(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosch")
        window.apply_replacement_to_word()

        window.regroup_words()

        rules = {rule.matched_text for rule in window.state.rules}
        assert rules == {"Bosch", "Bosh", "Bosche"}
        assert all(
            occurrence.reviewed
            for occurrence in window.state.occurrences
            if occurrence.detected_text in ("Bosch", "Bosh", "Bosche")
        )
    finally:
        window.close()


def test_a_folder_with_no_transcripts_says_so_rather_than_appearing_to_work(qapp, tmp_path):
    window = open_window(tmp_path, Folder({}))
    try:
        assert window.process_low_confidence_words() is False
        assert "nothing to process" in window._status_label.text()
    finally:
        window.close()


# -- Saving, which happens by itself and is never quiet about failing -----


def test_every_decision_is_written_to_the_project_at_once(qapp, tmp_path):
    """There is no Save button on this screen and there is not going to be one."""
    store = ProjectStore(tmp_path)
    window = open_window(tmp_path, two_file_folder(), store=store, process=True)
    try:
        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosch")
        window.apply_replacement_to_word()

        saved = ProjectStore(tmp_path).load()
        assert [group.replacement for group in saved.groups] == ["Bosch"]
        assert len(saved.rules) == 3
    finally:
        window.close()


def test_a_project_that_could_not_be_saved_is_said_urgently(qapp, tmp_path, monkeypatch):
    """The person believes their work is kept, and will find out otherwise too late."""
    urgent: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent_=False, **kw: urgent.append(
            f"{'urgent' if kw.get('urgent') else 'polite'}: {message}"
        ),
    )
    store = ProjectStore(tmp_path)
    window = open_window(tmp_path, two_file_folder(), store=store, process=True)
    try:
        select_word(window, "Bosch")
        monkeypatch.setattr(store, "save", lambda state: False)
        urgent.clear()
        window.confirm_item()

        assert len(urgent) == 1
        assert urgent[0].startswith("urgent: ")
        assert "could not be saved" in urgent[0]
        assert "only in this window" in urgent[0]
    finally:
        window.close()


def test_where_the_person_had_got_to_is_remembered_for_next_time(qapp, tmp_path):
    store = ProjectStore(tmp_path)
    folder = two_file_folder()
    window = open_window(tmp_path, folder, store=store, process=True)
    select_word(window, "Bosch")
    window.go_to_next_item()
    wanted = window.current_occurrence().id
    window.close()

    again = open_window(tmp_path, folder, store=store)
    try:
        assert again.current_occurrence() is not None
        assert again.current_occurrence().id == wanted
    finally:
        again.close()


def test_an_isolated_occurrence_is_remembered_for_next_time(qapp, tmp_path):
    """It has no group, so the place is found by the occurrence itself."""
    store = ProjectStore(tmp_path)
    folder = two_file_folder()
    window = open_window(tmp_path, folder, store=store, process=True)
    select_word(window, "Bosch")
    window.go_to_next_item()
    wanted = window.current_occurrence().id
    assert window.isolate_occurrence() is True
    assert window.current_occurrence().id == wanted
    window.close()

    again = open_window(tmp_path, folder, store=store)
    try:
        assert again.current_row().key == loose_key(wanted)
        assert again.current_occurrence().id == wanted
    finally:
        again.close()


def test_the_window_can_be_put_on_the_work_from_one_recording(qapp, tmp_path):
    """What the main window calls after transcribing a file."""
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        assert window.select_recording(OTHER_RECORDING) is True
        assert window.current_occurrence().recording_name == OTHER_RECORDING

        # Nothing waiting is an ordinary answer rather than a failure.
        assert window.select_recording("never seen.m4a") is False
    finally:
        window.close()


# -- Accessibility -------------------------------------------------------


def test_the_two_lists_are_named_as_the_specification_names_them(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        assert window._groups.accessibleName() == "Word Groups"
        assert window._occurrences.accessibleName() == "Occurrences"
    finally:
        window.close()


def test_every_control_in_the_window_has_an_accessible_name(qapp, tmp_path):
    """Gathered by type, so a control added later cannot slip through unnamed."""
    window = candidate_window(tmp_path)
    try:
        unnamed = [
            f"{widget.__class__.__name__} in {widget.parentWidget().__class__.__name__}"
            for widget in interactive_widgets(window)
            if not widget.accessibleName()
        ]

        assert unnamed == []
    finally:
        window.close()


def test_every_control_in_the_window_can_be_reached_from_the_keyboard(qapp, tmp_path):
    window = candidate_window(tmp_path)
    try:
        unreachable = [
            widget.accessibleName()
            for widget in interactive_widgets(window)
            if not (widget.focusPolicy() & Qt.FocusPolicy.TabFocus)
        ]

        assert unreachable == []
    finally:
        window.close()


def alt_letters(window: ReviewWindow) -> list[str]:
    """Every Alt letter the window claims, from its controls and its menu bar.

    The menu bar is in here, and it was not before, which is exactly how three
    collisions got in unnoticed. A menu title and a control on the same window
    are competing for the same keystroke: Qt refuses neither and cycles
    between them instead, so the same Alt press does one thing and then the
    other on the next press, and neither is reliably reachable. Entries
    *inside* a menu are left out, because those are only live while that menu
    is open and are scoped to it.
    """
    texts = [
        widget.text()
        for kind in (QLabel, QPushButton, QCheckBox)
        for widget in window.findChildren(kind)
    ]
    texts += [action.text() for action in window.menuBar().actions()]
    return [
        text[text.index("&") + 1].casefold()
        for text in texts
        if "&" in text and not text.endswith("&")
    ]


def test_no_two_controls_answer_the_same_alt_key(qapp, tmp_path):
    """Two claimants on one Alt letter means neither is reliably reachable.

    Three letters were claimed twice, and one of them was dangerous. Alt+P was
    the Project menu and also "Apply the speaker", which attributes the
    highlighted word to whoever is in the speaker box and writes the transcript
    at once, there being no Save button on this screen to catch it.
    """
    window = open_window(tmp_path)
    try:
        letters = alt_letters(window)

        assert sorted(letters) == sorted(set(letters)), f"repeated Alt letters in {letters}"
    finally:
        window.close()


def test_the_menu_bar_letters_are_not_claimed_by_any_control(qapp, tmp_path):
    """Said separately from the count above, so a failure says which is which."""
    window = open_window(tmp_path)
    try:
        menus = {
            action.text()[action.text().index("&") + 1].casefold()
            for action in window.menuBar().actions()
            if "&" in action.text()
        }
        controls = [
            text[text.index("&") + 1].casefold()
            for text in (
                widget.text()
                for kind in (QLabel, QPushButton, QCheckBox)
                for widget in window.findChildren(kind)
            )
            if "&" in text and not text.endswith("&")
        ]

        assert menus & set(controls) == set()
    finally:
        window.close()


def test_the_visible_label_and_the_accessible_name_call_the_list_one_thing(qapp, tmp_path):
    """A colleague reading the screen and a reader reading it aloud must agree."""
    window = open_window(tmp_path)
    try:
        assert window._groups_label.text().replace("&", "") == window._groups.accessibleName()
        assert (
            window._occurrences_label.text().replace("&", "")
            == window._occurrences.accessibleName()
        )
    finally:
        window.close()


def test_each_field_is_reached_by_its_own_label(qapp, tmp_path):
    """A label with no buddy names nothing, and the field is read as blank."""
    window = open_window(tmp_path)
    try:
        labelled = {label.buddy() for label in window.findChildren(QLabel) if label.buddy()}

        for field in (
            window._groups,
            window._occurrences,
            window._candidate_text,
            window._detected_edit,
            window._strength_edit,
            window._file_edit,
            window._timing_edit,
            window._speaker_edit,
            window._language_edit,
            window._risk_edit,
            window._decision_text,
            window._span_edit,
            window._replacement_edit,
            window._affected_edit,
            window._speaker_box,
            window._minimum_spin,
            window._tolerance_spin,
        ):
            assert field in labelled, f"{field.accessibleName()} has no label pointing at it"
    finally:
        window.close()


def test_the_panel_key_lands_on_a_control_rather_than_a_container(qapp, tmp_path):
    """F6 must not strand the focus on a container that answers no keys."""
    window = candidate_window(tmp_path, show_details=True)
    try:
        landed = []
        for _ in range(7):
            window.focus_next_panel()
            landed.append(window.focusWidget())

        assert all(widget is not None for widget in landed)
        for widget in landed:
            assert widget.focusPolicy() != Qt.FocusPolicy.NoFocus
        # Seven panels, and F6 visits a different control in each of them.
        assert len(set(landed)) == 7
    finally:
        window.close()


def test_the_panel_key_says_which_panel_it_landed_on(qapp, tmp_path):
    window = open_window(tmp_path)
    try:
        window.focus_next_panel()

        assert window._status_label.text() in (
            "Word Groups",
            "Occurrences",
            "Review settings",
            "Item details",
            "Playback",
            "Corrections",
            "About this control",
        )
    finally:
        window.close()


def test_every_action_has_a_shortcut_a_person_can_actually_press(qapp, tmp_path):
    window = open_window(tmp_path)
    try:
        # The occurrence is the inner loop, so it gets the plain keys.
        assert window._next_action.shortcut() == QKeySequence(Qt.Key.Key_F3)
        assert window._previous_action.shortcut() == QKeySequence("Shift+F3")
        assert window._next_group_action.shortcut() == QKeySequence("Ctrl+F3")
        assert window._previous_group_action.shortcut() == QKeySequence("Ctrl+Shift+F3")
        assert window._correct_text_action.shortcut() == QKeySequence(Qt.Key.Key_F2)
        # Confirming is the common case, so it is one keystroke with no
        # modifier and nothing to hold down.
        assert window._confirm_action.shortcut() == QKeySequence(Qt.Key.Key_F4)
        assert window._play_action.shortcut() == QKeySequence(Qt.Key.Key_F5)
        assert window._play_short_action.shortcut() == QKeySequence("Shift+F5")
        assert window._play_wide_action.shortcut() == QKeySequence("Ctrl+F5")
        assert window._isolate_action.shortcut() == QKeySequence("Ctrl+I")
        assert window._process_action.shortcut() == QKeySequence("Ctrl+L")
        assert window._regroup_action.shortcut() == QKeySequence("Ctrl+G")
    finally:
        window.close()


def test_every_action_is_on_a_menu_as_well_as_on_a_control(qapp, tmp_path):
    """A menu is where a screen reader user looks for what a window can do."""
    window = open_window(tmp_path)
    try:
        titles = {
            action.text().replace("&", "")
            for menu in window.menuBar().findChildren(type(window.menuBar().addMenu("x")))
            for action in menu.actions()
        }

        for wanted in (
            "Process Low Confidence Words",
            "Regroup Words",
            "Set the Minimum Confidence",
            "Set the Grouping Tolerance",
            "Apply to This Word",
            "Correct as Detected",
            "Isolate This Occurrence",
            "Apply to This Occurrence Only",
            "Apply the Speaker",
            "Show Reviewed Words",
            "Show Other Uncertainties",
            "Play Automatically",
        ):
            assert wanted in titles, f"{wanted} is not on any menu"
    finally:
        window.close()


def test_the_toggles_say_whether_they_are_on_rather_than_flipping_a_verb(qapp, tmp_path):
    window = open_window(tmp_path)
    try:
        for action in (
            window._show_reviewed_action,
            window._show_uncertainties_action,
            window._auto_play_action,
        ):
            assert action.isCheckable()

        window.set_show_reviewed(True)
        assert window._show_reviewed_action.isChecked()
        assert window._show_reviewed_box.isChecked()

        window._show_reviewed_box.setChecked(False)
        assert not window._show_reviewed_action.isChecked()
    finally:
        window.close()


def test_the_note_panel_follows_the_focus_onto_every_control_that_has_one(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        # On a word with several occurrences, so that every control below is
        # enabled: a disabled control cannot take the focus, and the note
        # would silently stay on the one before it.
        select_word(window, "Bosch")
        for widget, key in (
            (window._groups, review_window_module.GROUPS),
            (window._occurrences, review_window_module.OCCURRENCES),
            (window._detected_edit, review_window_module.WORD_DETAILS),
            (window._candidates, review_window_module.CANDIDATES),
            (window._timing_edit, review_window_module.TIMING),
            (window._speaker_edit, review_window_module.SPEAKER),
            (window._risk_edit, review_window_module.RISK),
            (window._play_button, review_window_module.PLAYBACK),
            (window._replacement_edit, review_window_module.REPLACEMENT),
            (window._speaker_box, review_window_module.CORRECT_SPEAKER),
            (window._confirm_button, review_window_module.CONFIRM),
            (window._isolate_button, review_window_module.ISOLATE),
            (window._process_button, review_window_module.PROCESSING),
            (window._minimum_spin, review_window_module.THRESHOLDS),
            (window._show_reviewed_box, review_window_module.VISIBILITY),
            (window._auto_play_box, review_window_module.AUTO_PLAY),
        ):
            widget.setFocus(Qt.FocusReason.TabFocusReason)
            qapp.processEvents()

            assert window._showing_note == key, widget.accessibleName()
            assert window._notes_text.toPlainText() == review_window_module.note_for(key).note
    finally:
        window.close()


def test_reading_a_note_does_not_change_the_note(qapp, tmp_path):
    """The panel takes focus so it can be read, and must hold still when it does."""
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        window._confirm_button.setFocus(Qt.FocusReason.TabFocusReason)
        qapp.processEvents()

        window._notes_text.setFocus(Qt.FocusReason.TabFocusReason)
        qapp.processEvents()

        assert window._showing_note == review_window_module.CONFIRM
    finally:
        window.close()


def test_a_closed_window_stops_watching_the_focus(qapp, tmp_path):
    """Otherwise every review window ever opened watches for the rest of the run."""
    windows = [open_window(tmp_path) for _ in range(3)]
    try:
        qapp.processEvents()
        assert [window._watching_focus for window in windows] == [True, True, True]

        for window in windows:
            window.close()
            qapp.processEvents()

        assert [window._watching_focus for window in windows] == [False, False, False]
    finally:
        for window in windows:
            window.close()


def test_the_status_label_reports_its_message_rather_than_a_name(qapp, tmp_path):
    """A label has no accessible value: naming it would hide what it says."""
    window = open_window(tmp_path)
    try:
        window._set_status("Showing 5 of 12 items.")

        assert accessible_name(window._status_label) == "Showing 5 of 12 items."
        assert not window._count_label.accessibleName()
    finally:
        window.close()


def test_the_focus_is_caught_when_the_item_controls_switch_off(qapp, tmp_path):
    """Disabling a control that has focus must not drop the user somewhere random.

    The move is said as part of the sentence about what happened, rather
    than as a second announcement arriving while the user is still
    listening to the first.
    """
    folder = Folder({RECORDING: make_transcript([weak_token("one", 10.0)])})
    window = open_window(tmp_path, folder, process=True)
    try:
        window._confirm_button.setFocus(Qt.FocusReason.TabFocusReason)
        assert window.focusWidget() is window._confirm_button

        window.correct_word_as_detected()

        assert window.focusWidget() is window._groups
        assert window._status_label.text().endswith(
            "The focus has moved to the word groups."
        )
    finally:
        window.close()


# -- The audio and the screen reader, which used to talk over each other ---


def test_the_wait_before_the_audio_starts_is_the_persons_own(qapp, tmp_path):
    """Only the person knows how fast their own screen reader speaks.

    Landing on an occurrence hands the reader a sentence of six facts, which
    takes it between three and eight seconds. The wait used to be 400
    milliseconds, fixed, so the clip started while the reader was still on the
    file name and the person never heard the confidence, the language or the
    replacement -- the three things they were about to decide on.
    """
    store = ProjectStore(tmp_path)
    window = open_window(tmp_path, two_file_folder(), store=store, process=True)
    try:
        # No wait by default, so a person reviewing hundreds of words is not
        # held up on every one; somebody whose reader needs time sets it.
        assert window._delay_spin.value() == 0
        assert window._delay_spin.suffix() == " seconds"

        window._delay_spin.setValue(5)
        settle(window._delay_spin)

        assert store.load().settings.auto_play_delay_seconds == 5
        assert "5 seconds after you arrive" in window._status_label.text()
    finally:
        window.close()


def test_the_wait_is_what_the_timer_actually_waits(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        window._delay_spin.setValue(4)
        settle(window._delay_spin)
        select_word(window, "Bosch")
        window.go_to_next_item()

        assert window._play_timer.isActive()
        assert window._play_timer.interval() == 4000
    finally:
        window.close()


def test_a_wait_of_nothing_is_a_real_choice_and_says_so(qapp, tmp_path):
    """Somebody working by eye has no reading for the clip to talk over."""
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        window._delay_spin.setValue(3)
        settle(window._delay_spin)
        window._delay_spin.setValue(0)
        settle(window._delay_spin)

        assert window._status_label.text() == review_window_module.NO_AUTO_PLAY_DELAY
        select_word(window, "Bosch")
        window.go_to_next_item()
        # Not exactly nothing: a moment short enough to sound instant, which
        # lets a held arrow key move on before any clip starts.
        assert window._play_timer.interval() == review_window_module.SETTLE_MILLISECONDS
        assert review_window_module.SETTLE_MILLISECONDS < 200
    finally:
        window.close()


def test_the_saved_wait_comes_back_with_the_folder(qapp, tmp_path):
    store = ProjectStore(tmp_path)
    first = open_window(tmp_path, two_file_folder(), store=store, process=True)
    first._delay_spin.setValue(7)
    settle(first._delay_spin)
    first.close()

    second = open_window(tmp_path, two_file_folder(), store=store)
    try:
        assert second._delay_spin.value() == 7
    finally:
        second.close()


def test_the_wait_is_measured_from_when_the_word_was_handed_over(qapp, tmp_path, monkeypatch):
    """Arrowing the first list used to start the clock before the sentence existed.

    Filling the second list starts the wait, and that happens while the word's
    own sentence has not yet been given to the reader at all. So the whole of
    the wait was spent in silence and the clip and the sentence then set off
    together.
    """
    started: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: started.append("said"),
    )
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        real_start = window._play_timer.start

        def noted(interval):
            started.append("timer")
            real_start(interval)

        monkeypatch.setattr(window._play_timer, "start", noted)
        started.clear()
        window.go_to_next_group()

        assert started[-1] == "timer"
        assert "said" in started
    finally:
        window.close()


def test_moving_between_neighbouring_occurrences_says_only_what_changed(qapp, tmp_path):
    """Six facts a row is most of a sentence the person heard a second ago.

    The occurrences of one word usually share a language, share the word's
    replacement and are usually all unreviewed, so three of the six are the
    same on every row. They stay in their columns and are left out of the
    reading, which is what makes the sentence short enough for the audio to
    follow it rather than land on top of it.
    """
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        first = window.current_occurrence()
        window.go_to_next_item()
        second = window.current_occurrence()

        arriving = review_lists.spoken_occurrence_summary(first, None)
        moving_on = review_lists.spoken_occurrence_summary(second, None, first)

        # Arriving at the word gives the whole row, because nothing was said
        # a moment ago to leave anything out of.
        assert arriving.endswith("Replacement None set. Not reviewed.")
        assert "English" in arriving
        # Moving to its neighbour repeats neither the language nor the
        # replacement nor the reviewed state, all three being unchanged.
        assert "English" not in moving_on
        assert "Replacement" not in moving_on
        assert "Not reviewed" not in moving_on
        # What did change is still said: the file, the position and the
        # confidence, which is what the person is navigating by.
        assert moving_on.startswith(second.recording_name)
        assert "percent" in moving_on
    finally:
        window.close()


def test_a_neighbour_that_differs_says_the_thing_that_differs(qapp, tmp_path):
    """Leaving out what has not changed must never leave out what has."""
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        first, second = window.current_row().occurrences[:2]
        second.replacement = "Bosche"

        moving_on = review_lists.spoken_occurrence_summary(second, None, first)

        assert "Replacement Bosche." in moving_on
        assert "Replaced with Bosche." in moving_on
    finally:
        window.close()


# -- Saying what a long job is, before it stops answering ------------------


def test_processing_says_what_it_is_about_to_do_before_it_starts(qapp, tmp_path, monkeypatch):
    """The analysis runs on the graphical thread and the window then goes deaf.

    Fifty hour-long recordings take about 37 seconds, during which nothing
    repaints, no key is answered, and Windows eventually writes "Not
    Responding" into the title bar for a screen reader to read out. Somebody
    who was told nothing beforehand cannot tell that from a crash, and their
    only move is to kill the application in the middle of a run that writes
    transcripts.
    """
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    window = open_window(tmp_path, two_file_folder())
    try:
        said.clear()
        window.process_low_confidence_words()

        assert said[0] == (
            "Looking at every word of 2 recordings. The window will not answer until "
            "it has finished."
        )
        assert len(said) == 2
        assert said[1].startswith("Processed 2 recordings.")
    finally:
        window.close()


def test_the_cursor_says_the_window_is_busy_and_gives_it_back_afterwards(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder())
    try:
        seen: list[object] = []
        real = window._analyse

        def watching(verb):
            seen.append(qapp.overrideCursor())
            return real(verb)

        window._analyse = watching
        window.process_low_confidence_words()

        assert seen and seen[0] is not None
        assert seen[0].shape() == Qt.CursorShape.WaitCursor
        assert qapp.overrideCursor() is None
    finally:
        window.close()


# -- The focus-rescue note, which used to go missing and reappear ----------


def test_hiding_the_second_block_carries_the_note_about_the_focus(qapp, tmp_path):
    """Emptying the list takes the focus off the controls, silently until now."""
    window = open_window(tmp_path, flagged_only_folder(), process=True)
    try:
        window._confirm_button.setFocus(Qt.FocusReason.TabFocusReason)
        assert window.focusWidget() is window._confirm_button

        window.set_show_other_uncertainties(False)

        assert window.focusWidget() is window._groups
        assert window._status_label.text().endswith(
            "The focus has moved to the word groups."
        )
    finally:
        window.close()


def one_flagged_word() -> Folder:
    """A single recording holding a single flagged word and nothing else.

    The smallest folder in which settling one thing empties the whole window,
    which is what the focus rescue is for and what a folder with a second word
    in it can never demonstrate: the highlight simply moves to the neighbour
    and every control stays live.
    """
    return Folder(
        {
            RECORDING: make_transcript(
                [
                    make_token(
                        "15,000",
                        start=70.0,
                        end=70.5,
                        reasons=(ReviewReason.NUMERIC_DISAGREEMENT,),
                    )
                ],
                name=RECORDING,
            )
        }
    )


def test_hiding_the_reviewed_words_carries_the_note_about_the_focus(qapp, tmp_path):
    window = open_window(tmp_path, one_flagged_word(), process=True)
    try:
        window.set_show_reviewed(True)
        select_word(window, "15,000")
        window.confirm_item()
        select_word(window, "15,000")
        window._confirm_button.setFocus(Qt.FocusReason.TabFocusReason)

        window.set_show_reviewed(False)

        assert window._group_model.rowCount() == 0
        assert window.focusWidget() is window._groups
        assert window._status_label.text().endswith(
            "The focus has moved to the word groups."
        )
    finally:
        window.close()


def test_a_note_nobody_carried_does_not_turn_up_later(qapp, tmp_path):
    """The flag is taken, not left standing for some unrelated sentence.

    This is the other half of the same fault. A toggle that emptied the list
    said nothing about the focus having moved, and the flag saying it had
    stayed set, so the note surfaced at the end of whatever was announced
    next, about something else entirely.
    """
    window = open_window(tmp_path, flagged_only_folder(), process=True)
    try:
        window._confirm_button.setFocus(Qt.FocusReason.TabFocusReason)
        window.set_show_other_uncertainties(False)
        assert window._focus_caught is False

        window.set_show_other_uncertainties(True)

        assert "focus has moved" not in window._status_label.text()
    finally:
        window.close()


def test_landing_on_a_recording_carries_the_note_too(qapp, tmp_path, monkeypatch):
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        window._focus_caught = True
        said.clear()

        assert window.select_recording(OTHER_RECORDING) is True

        assert said[0].endswith("The focus has moved to the word groups.")
    finally:
        window.close()


def test_landing_silently_leaves_the_announcement_to_the_caller(qapp, tmp_path, monkeypatch):
    """Opening a review after a transcription has more to say than the row does."""
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        said.clear()

        assert window.select_recording(OTHER_RECORDING, announce_arrival=False) is True

        assert said == []
        assert window.current_occurrence().recording_name == OTHER_RECORDING
    finally:
        window.close()


def test_landing_silently_on_another_word_says_nothing_and_plays_nothing(
    qapp, tmp_path, monkeypatch
):
    """Moving word is what announced the row and started its audio before."""
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "15,000")
        window._play_timer.stop()
        said.clear()

        assert window.select_recording(OTHER_RECORDING, announce_arrival=False) is True

        assert window.current_row().key == group_row(window, "Bosch").key
        assert said == []
        assert not window._play_timer.isActive()
    finally:
        window.close()


def test_a_correction_that_moves_to_another_word_names_that_word(
    qapp, tmp_path, monkeypatch
):
    """Somebody still in the Replacement box must hear which word they are on."""
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        corrected = window.current_row().key
        said.clear()

        window.correct_word_as_detected()

        row = window.current_row()
        assert row is not None and row.key != corrected
        assert len(said) == 1
        assert f"Now on {review_window_module.spoken_group_summary(row)}" in said[0]
    finally:
        window.close()


# -- Being left in an empty table -----------------------------------------


def test_emptying_the_occurrence_list_rescues_the_focus_out_of_it(qapp, tmp_path):
    """A table with no rows answers no arrow key and reads nothing out.

    Confirming the last occurrence of a word empties the second list. The
    controls that act on an item are all switched off at that moment and their
    focus is caught, but the list itself was not among them, so somebody
    working in it was left standing in an empty table with no way out but Tab
    and no hint that Tab was the answer.
    """
    window = open_window(tmp_path, one_flagged_word(), process=True)
    try:
        window._occurrences.setFocus(Qt.FocusReason.TabFocusReason)
        assert window.focusWidget() is window._occurrences
        assert window._occurrence_model.rowCount() == 1

        window.confirm_item()

        assert window._occurrence_model.rowCount() == 0
        assert window._occurrences.isEnabled() is False
        assert window.focusWidget() is window._groups
        assert window._status_label.text().endswith(
            "The focus has moved to the word groups."
        )
    finally:
        window.close()


def test_the_occurrence_list_is_live_again_as_soon_as_it_has_rows(qapp, tmp_path):
    """Switching it off must not be a one-way trip."""
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        assert window._occurrences.isEnabled() is True
    finally:
        window.close()


def test_the_occurrence_list_stays_live_for_a_word_that_cannot_be_corrected(qapp, tmp_path):
    """Being unable to correct a word is not a reason to take its rows away.

    A transcript that cannot be read switches every correction off, and the
    list is deliberately not switched off with them: its rows are still worth
    moving through and still describe what the person decided.
    """
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        folder.transcripts.clear()
        window._cached_name = None
        window._show_occurrence(window.current_occurrence())

        assert window._occurrence_model.rowCount() > 0
        assert window._occurrences.isEnabled() is True
        assert window._confirm_button.isEnabled() is False
    finally:
        window.close()


# -- A group somebody made themselves --------------------------------------


def test_a_group_you_put_together_yourself_says_so(qapp, tmp_path):
    """It was stored and preserved from the beginning and never shown.

    An isolated occurrence already says "kept on its own", so the pattern was
    understood; a hand-made group was simply not given the same treatment, and
    read exactly like one the analysis had suggested. The difference matters
    because a suggested group is rebuilt every time a threshold moves and a
    hand-made one is not.
    """
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        assert group_row(window, "Bosch").why == review_lists.WHY_LOW_CONFIDENCE

        window.state.groups[0].user_created = True
        window.refresh()

        assert group_row(window, "Bosch").why == review_lists.WHY_USER_CREATED
        assert review_lists.WHY_USER_CREATED != review_lists.WHY_LOW_CONFIDENCE
    finally:
        window.close()


# -- Fitting on the screen, and reaching what is off the edge ---------------


def test_the_window_never_opens_wider_than_the_screen_it_is_on(qapp, tmp_path):
    """1440 raw pixels is off the edge of a 1366 laptop, and of any screen at 150 percent."""
    window = open_window(tmp_path)
    try:
        available = window.screen().availableGeometry()

        assert window.width() <= available.width()
        assert window.height() <= available.height()
    finally:
        window.close()


def test_both_scrolling_panels_can_be_reached_sideways(qapp, tmp_path):
    """Resizable and no horizontal bar means anything too wide is simply cut off.

    At a large Windows text size that is the row of buttons at the top of the
    settings panel; under a magnifier, where the viewport is a fraction of its
    nominal width, it is most of the panel.
    """
    window = open_window(tmp_path)
    try:
        areas = window.findChildren(QScrollArea)

        assert areas
        for area in areas:
            assert (
                area.horizontalScrollBarPolicy()
                != Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
    finally:
        window.close()


def test_the_text_panels_grow_when_the_text_size_does(qapp, tmp_path):
    """Both were measured once, while the window was being built, and pinned there."""
    window = open_window(tmp_path)
    try:
        before = window._notes_text.minimumHeight()
        font = window.font()
        font.setPointSize(font.pointSize() + 12)
        window.setFont(font)
        qapp.processEvents()

        assert window._notes_text.minimumHeight() > before
        assert window._decision_text.minimumHeight() > 0
    finally:
        window.close()


# -- Moving between panels, and moving the divider -------------------------


def test_the_panel_key_goes_both_ways(qapp, tmp_path):
    """Shift+F6 is what Windows does everywhere else, and it was missing."""
    window = candidate_window(tmp_path)
    try:
        assert window._previous_panel_action.shortcut() == QKeySequence("Shift+F6")

        window.focus_next_panel()
        window.focus_next_panel()
        forward = window._status_label.text()
        window.focus_previous_panel()
        window.focus_next_panel()

        assert window._status_label.text() == forward
    finally:
        window.close()


def test_the_panel_key_does_not_talk_over_the_control_it_landed_on(qapp, tmp_path, monkeypatch):
    """Moving the focus already makes the reader name where it went.

    An announcement raised at the same moment arrives on top of the very
    sentence that answers the question it was asking. The name stays on the
    screen for anyone who wants to look, which is what a magnifier user needs
    and a screen reader user does not.
    """
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    window = open_window(tmp_path)
    try:
        said.clear()
        window.focus_next_panel()

        assert said == []
        assert window._status_label.text()
    finally:
        window.close()


def test_the_divider_can_be_moved_from_the_keyboard(qapp, tmp_path):
    """The adjustment a magnifier user most wants, and the only one needing a mouse.

    Qt gives a splitter handle no keyboard behaviour of its own, so widening
    either side could not be done at all without pointing at it. At four times
    magnification a panel a third of the window wide holds about two words.
    """
    window = open_window(tmp_path)
    try:
        assert window._widen_lists_action.shortcut() == QKeySequence("Ctrl+Shift+Left")
        assert window._widen_details_action.shortcut() == QKeySequence("Ctrl+Shift+Right")
        before = window._splitter.sizes()

        window.adjust_divider(1)

        assert window._splitter.sizes()[0] > before[0]
        assert "percent of the window" in window._status_label.text()

        window.adjust_divider(-1)
        assert window._splitter.sizes()[0] <= before[0] + 1
    finally:
        window.close()


def test_each_widen_action_widens_the_side_it_names(qapp, tmp_path):
    """Widen the Lists gives the lists more room, and Widen the Details less.

    The two actions were wired the wrong way round, so the menu item a
    magnifier user reached for did the opposite of what it said, and the
    spoken share then went down. Triggering the actions themselves, not only
    adjust_divider, is what catches the wiring.
    """
    window = open_window(tmp_path)
    try:
        before = window._splitter.sizes()[0]

        window._widen_lists_action.trigger()
        widened = window._splitter.sizes()[0]
        assert widened > before

        window._widen_details_action.trigger()
        assert window._splitter.sizes()[0] < widened
    finally:
        window.close()


# -- Asking about occurrences and being answered about occurrences ----------


def test_asking_for_an_occurrence_with_none_selected_answers_about_occurrences(
    qapp, tmp_path
):
    """F3 asked about occurrences and used to be answered about words.

    The count of the *other* list was read out, in a sentence mentioning
    neither the key that was pressed nor the list it was pressed in.
    """
    window = open_window(tmp_path, two_file_folder())
    try:
        window.go_to_next_item()

        assert window._status_label.text().startswith(
            "No word is selected, so there are no occurrences to move through."
        )
    finally:
        window.close()


# -- A failed save that is heard rather than talked over -------------------


def failing_store(tmp_path, monkeypatch) -> ProjectStore:
    """A project store whose every write fails, quietly, the way a full disk does."""
    store = ProjectStore(tmp_path)
    monkeypatch.setattr(store, "save", lambda state: False)
    return store


def capture(monkeypatch) -> list[tuple[str, bool]]:
    """Every announcement, with whether it interrupted or waited its turn."""
    said: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append((message, urgent)),
    )
    return said


def test_a_failed_save_is_not_wiped_out_by_the_next_sentence(qapp, tmp_path, monkeypatch):
    """Six paths saved loudly and then said what they had done, in that order.

    Setting the status writes the label and announces it in one go, so a
    second call does not give the person a second message: it wipes the first
    off the screen and cuts across it being read. Saving loudly and then
    reporting therefore destroyed the more urgent of the two every time. There
    is now one sentence, with the failure as its tail, and it is urgent.
    """
    store = ProjectStore(tmp_path)
    window = open_window(tmp_path, two_file_folder(), store=store, process=True)
    try:
        said = capture(monkeypatch)
        monkeypatch.setattr(store, "save", lambda state: False)

        for act, wanted in (
            (lambda: window.set_show_reviewed(True), "Reviewed words are now shown."),
            (
                lambda: window.set_show_other_uncertainties(False),
                "Other uncertainties are now hidden.",
            ),
            (
                lambda: window.set_play_automatically(False),
                "Occurrences no longer play by themselves.",
            ),
        ):
            said.clear()
            act()

            assert len(said) == 1, said
            message, urgent = said[0]
            assert wanted in message
            assert "could not be saved" in message
            assert urgent is True
    finally:
        window.close()


def test_a_threshold_whose_save_failed_says_both_things_at_once(qapp, tmp_path, monkeypatch):
    store = ProjectStore(tmp_path)
    window = open_window(tmp_path, two_file_folder(), store=store, process=True)
    try:
        said = capture(monkeypatch)
        monkeypatch.setattr(store, "save", lambda state: False)

        for spin, wanted in (
            (window._minimum_spin, "The minimum confidence is now 71 percent."),
            (window._tolerance_spin, "The grouping tolerance is now 71 percent."),
            (window._delay_spin, "The audio now starts 7 seconds after you arrive"),
        ):
            said.clear()
            spin.setValue(71 if spin is not window._delay_spin else 7)
            settle(spin)

            assert len(said) == 1, said
            message, urgent = said[0]
            assert wanted in message
            assert "could not be saved" in message
            assert urgent is True
    finally:
        window.close()


def test_processing_whose_save_failed_is_the_damaging_one_and_says_so(
    qapp, tmp_path, monkeypatch
):
    """Rule answers are written into the transcripts before the project is saved.

    So a project save that fails leaves the transcripts on disk changed with
    no record of it in the project. On the next opening those words carry a
    strength of 1.0, the low-confidence sweep passes over them, and the review
    never mentions them again. The sentence saying the save failed used to be
    the one that was overwritten.
    """
    store = ProjectStore(tmp_path)
    window = open_window(tmp_path, two_file_folder(), store=store)
    try:
        said = capture(monkeypatch)
        monkeypatch.setattr(store, "save", lambda state: False)

        window.process_low_confidence_words()

        # The warning that a wait is coming, then one sentence covering both
        # what happened and that it could not be kept.
        assert len(said) == 2
        message, urgent = said[1]
        assert message.startswith("Processed 2 recordings.")
        assert "could not be saved" in message
        assert urgent is True
    finally:
        window.close()


# -- A timing decision is not a text replacement ---------------------------


def test_confirming_the_timing_is_not_reported_as_a_replacement(qapp, tmp_path):
    """The person confirmed a clock and was told they had rewritten a word.

    ``human_corrected`` means "a person decided something about this word",
    which a timing decision sets as readily as a correction does. Reading the
    replacement off it filled the Replacement column with the word's own
    unchanged text, so the Occurrences table said "Replacement: 15,000" and
    "Reviewed: Replaced with 15,000" about somebody who had changed no letter
    of anything -- in the one column that exists to keep the three answers
    apart.
    """
    folder = one_flagged_word()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "15,000")
        occurrence = window.current_occurrence()

        assert window.decide_timing(True) is True

        token = folder.token(RECORDING, occurrence.token_id)
        # The decision really was recorded, and really was only about timing.
        assert token.human_corrected is True
        assert token.text_corrected is False

        model = window._occurrence_model
        assert (
            model.data(model.index(0, OCCURRENCE_COLUMN_REPLACEMENT))
            == review_lists.NO_REPLACEMENT
        )
        assert model.data(model.index(0, OCCURRENCE_COLUMN_REVIEWED)) == NOT_REVIEWED
    finally:
        window.close()


def test_a_word_whose_text_was_replaced_still_says_what_it_was_replaced_with(
    qapp, tmp_path
):
    """Leaving the timing case alone must not silence the case it was copied from."""
    folder = one_flagged_word()
    window = open_window(tmp_path, folder, process=True)
    try:
        # Replacing the text settles the word, so it leaves the queue. Asking
        # to see the settled words is what keeps the row on screen to be read.
        window.set_show_reviewed(True)
        select_word(window, "15,000")
        occurrence = window.current_occurrence()
        window._replacement_edit.setText("15,500")

        assert window.apply_replacement_to_word() is True

        token = folder.token(RECORDING, occurrence.token_id)
        assert token.text_corrected is True
        model = window._occurrence_model
        assert model.data(model.index(0, OCCURRENCE_COLUMN_REPLACEMENT)) == "15,500"
    finally:
        window.close()


# -- Rules from rows that have no group behind them ------------------------


def test_a_correction_in_the_second_block_teaches_the_project_a_rule(qapp, tmp_path):
    """This is where the confidently-wrong proper names and amounts live.

    A word flagged for provider disagreement has no measured strength, so the
    low-confidence sweep never sees it and it never gets a group. Rules were
    only ever made from groups, so this whole class of word -- precisely the
    class the tenth decision exists for -- taught the project nothing at all.
    """
    window = open_window(tmp_path, one_flagged_word(), process=True)
    try:
        select_word(window, "15,000")
        assert window.current_row().group_id is None
        window._replacement_edit.setText("15,500")

        assert window.apply_replacement_to_word() is True

        assert len(window.state.rules) == 1
        rule = window.state.rules[0]
        assert rule.matched_text == "15,000"
        assert rule.replacement == "15,500"
        assert rule.group_id is None
    finally:
        window.close()


def test_a_correction_on_an_isolated_occurrence_teaches_the_project_a_rule(qapp, tmp_path):
    """Taking a word out of a group is a statement about it, not a withdrawal of it."""
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        window.go_to_next_item()
        detected = window.current_occurrence().detected_text

        assert window.isolate_occurrence() is True
        assert window.current_row().group_id is None

        window._replacement_edit.setText("Bosworth")
        assert window.apply_replacement_to_word() is True

        loose = [rule for rule in window.state.rules if rule.group_id is None]
        assert [rule.matched_text for rule in loose] == [detected]
        assert loose[0].replacement == "Bosworth"
    finally:
        window.close()


def test_reversing_such_a_correction_drops_the_rule_it_made(qapp, tmp_path):
    """A withdrawn decision must not go on rewriting files nobody has opened."""
    window = open_window(tmp_path, one_flagged_word(), process=True)
    try:
        window.set_show_reviewed(True)
        select_word(window, "15,000")
        window._replacement_edit.setText("15,500")
        window.apply_replacement_to_word()
        assert len(window.state.rules) == 1

        select_word(window, "15,000")
        assert window.correct_word_as_detected() is True

        assert window.state.rules == []
    finally:
        window.close()


def test_changing_your_mind_about_a_loose_word_leaves_one_rule_not_two(qapp, tmp_path):
    """Two rules answering one detected form would be settled by whichever came first."""
    window = open_window(tmp_path, one_flagged_word(), process=True)
    try:
        window.set_show_reviewed(True)
        select_word(window, "15,000")
        window._replacement_edit.setText("15,500")
        window.apply_replacement_to_word()

        select_word(window, "15,000")
        window._replacement_edit.setText("15,050")
        window.apply_replacement_to_word()

        assert len(window.state.rules) == 1
        assert window.state.rules[0].replacement == "15,050"
    finally:
        window.close()


# -- The rule tally, kept on both roads into a transcript -------------------


def test_a_rule_the_window_applies_is_counted(qapp, tmp_path):
    """Two places write a rule's correction into a transcript, and one kept the count.

    Whether a rule's tally was right therefore depended on which of the two
    roads a particular correction happened to take, which is not a difference
    anybody looking at the number could have known about.
    """
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosch")
        window.apply_replacement_to_word()
        rule = next(rule for rule in window.state.rules if rule.matched_text == "Bosh")
        assert rule.occurrence_count == 0

        # A word the analysis handed over as answered by a rule, which this
        # window is what actually writes into the transcript and saves.
        occurrence = next(
            item for item in window.state.occurrences if item.detected_text == "Bosh"
        )
        folder.transcripts[occurrence.recording_name] = (
            folder.transcripts[occurrence.recording_name].with_correction(
                occurrence.token_id, text="Bosh"
            )
        )
        window._cached_name = None
        occurrence.auto_applied = True
        occurrence.applied_rule_id = rule.id
        occurrence.replacement = "Bosch"

        assert window._write_rule_answers() == 1

        assert rule.occurrence_count == 1
    finally:
        window.close()


# -- Counting the work that is left ----------------------------------------


def test_the_analysis_counts_only_the_words_still_wanting_a_person(qapp, tmp_path):
    """It counted every occurrence in the project, settled ones included.

    So it promised four words to look at above a list holding none of them,
    which reads as a broken window rather than as a finished folder.
    """
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosch")
        window.apply_replacement_to_word()
        assert len(window.state.occurrences) == 3

        window.regroup_words()

        assert "0 words to look at" in window._status_label.text()
    finally:
        window.close()


def test_the_count_of_automatic_answers_takes_the_verb_with_it(qapp, tmp_path):
    """The number was interpolated into a sentence whose verb stayed put."""
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        window._write_rule_answers = lambda: 1
        window.regroup_words()
        assert "A further 1 word was answered automatically" in window._status_label.text()

        window._write_rule_answers = lambda: 2
        window.regroup_words()
        assert "A further 2 words were answered automatically" in window._status_label.text()
    finally:
        window.close()


# -- Closing the window ----------------------------------------------------


def test_closing_the_window_stops_the_audio_and_lets_the_recording_go(qapp, tmp_path):
    """Stopping was only ever done when the main window was the one closing this one.

    Pressing this window's own close button part-way through a twelve-second
    clip left the sound playing with no window on screen to stop it, and left
    the player holding an open handle on the recording, so the person could not
    then move, rename or delete the file they had just been listening to and
    nothing anywhere would tell them why.
    """
    player = FakePlayer()
    window = open_window(tmp_path, two_file_folder(), player, process=True)
    select_word(window, "Bosch")
    assert window.play_span() is True
    player.becomes_ready()
    assert player.plays == 1

    window.close()

    assert player.stops == 1
    # Clearing the source, not merely stopping, is what releases the handle.
    assert player.cleared == 1
    assert player.loaded is None


def test_a_save_that_fails_as_the_window_closes_keeps_it_there_to_say_so(
    qapp, tmp_path, monkeypatch
):
    """Announcing it to a window on its way off the screen is announcing it to nobody.

    Only the resume markers are at stake, every decision having been written
    through when it was made, so this is the mildest failure in the window.
    It is still a failure, and it was still silent.
    """
    store = ProjectStore(tmp_path)
    window = open_window(tmp_path, two_file_folder(), store=store, process=True)
    said = capture(monkeypatch)
    monkeypatch.setattr(store, "save", lambda state: False)

    window.close()

    assert window.isVisible() is True
    message, urgent = said[-1]
    assert "could not be saved" in message
    assert "Close the window again to close it anyway." in message
    assert urgent is True

    # And asking a second time closes anyway. A window somebody cannot get out
    # of would be a worse fault than the one being reported.
    window.close()

    assert window.isVisible() is False


# -- A transcript that cannot be written ------------------------------------


def test_a_correction_that_could_not_be_saved_is_said_from_this_window(
    qapp, tmp_path, monkeypatch
):
    """The failure used to be said only by the main window, which is behind this one.

    Meanwhile this window kept the change, marked the word replaced and said
    it was done. The person moved away and back, the file gave the old text
    back, and the project no longer flagged the word.
    """
    folder = two_file_folder()
    emitted: list[str] = []
    window = open_window(tmp_path, folder, process=True)
    window.transcriptChanged.connect(lambda name, transcript: emitted.append(name))
    try:
        select_word(window, "Bosch")
        occurrence = window.current_occurrence()
        before = folder.token(occurrence.recording_name, occurrence.token_id).text
        folder.refusing.add(occurrence.recording_name)
        said = capture(monkeypatch)
        window._replacement_edit.setText("Bosche")

        assert window.apply_replacement_to_occurrence() is False

        assert len(said) == 1, said
        message, urgent = said[0]
        assert "could not be saved" in message
        assert occurrence.recording_name in message
        assert "has not been made" in message
        assert urgent is True
        # Nothing in the project, the file, the window's copy or the signal
        # says the change happened.
        assert occurrence.reviewed is False
        assert occurrence.replacement is None
        assert folder.token(occurrence.recording_name, occurrence.token_id).text == before
        transcript = window._transcript(occurrence.recording_name)
        assert transcript.token_by_id(occurrence.token_id).text == before
        assert emitted == []
    finally:
        window.close()


def test_every_one_occurrence_action_stops_when_its_save_fails(
    qapp, tmp_path, monkeypatch
):
    """Confirming, the speaker and the timing all went on as if it had worked."""

    def change_speaker(window: ReviewWindow) -> bool:
        window._speaker_box.setCurrentText("Speaker speaker_1")
        return window.apply_speaker_correction()

    for act in (
        lambda window: window.confirm_item(),
        lambda window: window.decide_timing(True),
        change_speaker,
    ):
        folder = Folder({RECORDING: make_transcript([weak_token("contract", 30.0)])})
        window = open_window(tmp_path, folder, process=True)
        try:
            occurrence = window.current_occurrence()
            folder.refusing.add(RECORDING)
            folder.saved.clear()
            said = capture(monkeypatch)

            assert act(window) is False

            assert folder.saved == []
            assert occurrence.reviewed is False
            assert occurrence.correct_as_detected is False
            assert len(said) == 1, said
            assert "could not be saved" in said[0][0]
            assert said[0][1] is True
        finally:
            window.close()


def test_a_word_none_of_whose_files_could_be_saved_is_left_exactly_as_it_was(
    qapp, tmp_path, monkeypatch
):
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        group = window.state.group(window.current_row().group_id)
        folder.refusing.update({RECORDING, OTHER_RECORDING})
        said = capture(monkeypatch)
        window._replacement_edit.setText("Bosch")

        assert window.apply_replacement_to_word() is False

        # Said once for the whole word, not once for each file.
        assert len(said) == 1, said
        message, urgent = said[0]
        assert "2 recordings" in message
        assert urgent is True
        assert group.reviewed is False
        assert group.replacement is None
        assert window.state.rules == []
        assert not any(item.reviewed for item in window.state.occurrences_of(group.id))
        assert folder.texts(RECORDING) == ["Bosch", "Bosh", "15,000"]
        assert folder.texts(OTHER_RECORDING) == ["Bosche", "settled"]
    finally:
        window.close()


def test_a_word_saved_in_some_files_only_stays_open_and_says_which_failed(
    qapp, tmp_path, monkeypatch
):
    """Settling it would take it off the list while one file still says the old word."""
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        group = window.state.group(window.current_row().group_id)
        folder.refusing.add(OTHER_RECORDING)
        said = capture(monkeypatch)
        window._replacement_edit.setText("Bosch")

        assert window.apply_replacement_to_word() is True

        assert len(said) == 1, said
        message, urgent = said[0]
        assert "not to all of them" in message
        assert OTHER_RECORDING in message
        assert urgent is True
        assert folder.texts(RECORDING) == ["Bosch", "Bosch", "15,000"]
        assert folder.texts(OTHER_RECORDING) == ["Bosche", "settled"]
        # The word and its rules wait until every file has taken the change.
        assert group.reviewed is False
        assert group.replacement is None
        assert window.state.rules == []
        for item in window.state.occurrences_of(group.id):
            assert item.reviewed is (item.recording_name == RECORDING)

        # And applying it again, once the file can be written, finishes it.
        folder.refusing.clear()
        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosch")

        assert window.apply_replacement_to_word() is True

        assert folder.texts(OTHER_RECORDING) == ["Bosch", "settled"]
        assert group.reviewed is True
        assert group.replacement == "Bosch"
        assert len(window.state.rules) == 3
    finally:
        window.close()


def test_correcting_a_word_as_detected_keeps_the_replacement_when_no_file_was_saved(
    qapp, tmp_path, monkeypatch
):
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        group = window.state.group(window.current_row().group_id)
        window._replacement_edit.setText("Bosch")
        window.apply_replacement_to_word()
        window.set_show_reviewed(True)
        select_word(window, "Bosch")
        folder.refusing.update({RECORDING, OTHER_RECORDING})
        said = capture(monkeypatch)

        assert window.correct_word_as_detected() is False

        assert len(said) == 1, said
        assert "could not be saved" in said[0][0]
        assert said[0][1] is True
        assert group.replacement == "Bosch"
        assert group.correct_as_detected is False
        assert len(window.state.rules) == 3
        assert not any(
            item.correct_as_detected for item in window.state.occurrences_of(group.id)
        )
        assert folder.texts(RECORDING) == ["Bosch", "Bosch", "15,000"]
    finally:
        window.close()


def test_correcting_a_word_as_detected_in_some_files_only_keeps_the_word_open(
    qapp, tmp_path, monkeypatch
):
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        group = window.state.group(window.current_row().group_id)
        window._replacement_edit.setText("Bosch")
        window.apply_replacement_to_word()
        window.set_show_reviewed(True)
        select_word(window, "Bosch")
        folder.refusing.add(OTHER_RECORDING)
        said = capture(monkeypatch)

        assert window.correct_word_as_detected() is True

        message, urgent = said[-1]
        assert "but not in all of them" in message
        assert OTHER_RECORDING in message
        assert urgent is True
        assert folder.texts(RECORDING) == ["Bosch", "Bosh", "15,000"]
        assert folder.texts(OTHER_RECORDING) == ["Bosch", "settled"]
        # The group keeps its replacement and its rules until every file agrees.
        assert group.correct_as_detected is False
        assert len(window.state.rules) == 3
        for item in window.state.occurrences_of(group.id):
            assert item.correct_as_detected is (item.recording_name == RECORDING)
    finally:
        window.close()


def test_a_rule_answer_that_could_not_be_saved_is_not_counted(qapp, tmp_path):
    """Nothing was changed, so neither the run nor the rule's tally may count it."""
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosch")
        window.apply_replacement_to_word()
        rule = next(rule for rule in window.state.rules if rule.matched_text == "Bosh")
        occurrence = next(
            item for item in window.state.occurrences if item.detected_text == "Bosh"
        )
        folder.transcripts[occurrence.recording_name] = (
            folder.transcripts[occurrence.recording_name].with_correction(
                occurrence.token_id, text="Bosh"
            )
        )
        window._cached_name = None
        occurrence.auto_applied = True
        occurrence.applied_rule_id = rule.id
        occurrence.replacement = "Bosch"
        folder.refusing.add(occurrence.recording_name)

        assert window._write_rule_answers() == 0

        assert rule.occurrence_count == 0
        assert window._unsaved_rule_answers == [occurrence.recording_name]
        assert folder.token(occurrence.recording_name, occurrence.token_id).text == "Bosh"
    finally:
        window.close()


def test_processing_says_which_automatic_answers_could_not_be_saved(
    qapp, tmp_path, monkeypatch
):
    """In the one sentence about the run, because a second would wipe it out."""
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:

        def write_none() -> int:
            window._unsaved_rule_answers = [OTHER_RECORDING]
            return 0

        window._write_rule_answers = write_none
        said = capture(monkeypatch)

        window.regroup_words()

        message, urgent = said[-1]
        assert "could not be saved" in message
        assert OTHER_RECORDING in message
        assert "written again the next time the folder is processed" in message
        assert urgent is True
    finally:
        window.close()


def test_a_window_with_nowhere_to_save_still_makes_the_change(qapp, tmp_path):
    """No callback at all means every save counts as having worked."""
    folder = two_file_folder()
    window = ReviewWindow(
        tmp_path,
        folder.names,
        folder.load,
        {name: Path(f"C:/Audio/{name}") for name in folder.names},
        ProjectStore(tmp_path),
        FakePlayer(),
    )
    window.show()
    window.process_low_confidence_words()
    try:
        select_word(window, "Bosch")
        occurrence = window.current_occurrence()
        window._replacement_edit.setText("Bosche")

        assert window.apply_replacement_to_occurrence() is True

        assert occurrence.reviewed is True
        assert occurrence.replacement == "Bosche"
    finally:
        window.close()


def test_every_mnemonic_in_the_item_menu_is_its_own(qapp, tmp_path):
    """No two items in the Item menu answer to the same letter.

    Isolate This Occurrence and Confirm the Timing both took I, so the letter
    moved between them instead of choosing either.
    """
    window = open_window(tmp_path)
    try:
        menu = window._isolate_action.associatedObjects()[0]
        keys = []
        for action in menu.actions():
            text = action.text().replace("&&", "")
            if "&" in text:
                keys.append(text[text.index("&") + 1].lower())

        assert len(keys) == len(set(keys)), keys
        assert window._confirm_timing_action.text() == "Confirm the &Timing"
    finally:
        window.close()


# -- What a review teaches the service statistics ------------------------


def open_counting_window(tmp_path, folder: Folder, record_statistics) -> ReviewWindow:
    window = ReviewWindow(
        tmp_path,
        folder.names,
        folder.load,
        {name: Path(f"C:/Audio/{name}") for name in folder.names},
        ProjectStore(tmp_path),
        FakePlayer(),
        folder.save,
        record_statistics=record_statistics,
    )
    window.show()
    window.process_low_confidence_words()
    return window


def test_a_correction_in_the_review_window_updates_the_statistics_on_disk(qapp, tmp_path):
    folder = two_file_folder()
    store = CalibrationStore(tmp_path / CALIBRATION_FILE_NAME)
    window = open_counting_window(tmp_path, folder, lambda change: store.apply(change.apply))
    try:
        select_word(window, "Bosch")
        occurrence = window.current_occurrence()
        window._replacement_edit.setText("Bosche")

        assert window.apply_replacement_to_occurrence() is True

        counts = store.load().counts_for(Provider.ELEVENLABS)
        assert (counts.chosen, counts.corrected) == (1, 1)
        note = folder.token(occurrence.recording_name, occurrence.token_id).statistics_note
        assert note is not None and note.counted and note.text_corrected
    finally:
        window.close()


def test_a_confirmed_word_is_counted_as_right(qapp, tmp_path):
    folder = two_file_folder()
    store = CalibrationStore(tmp_path / CALIBRATION_FILE_NAME)
    window = open_counting_window(tmp_path, folder, lambda change: store.apply(change.apply))
    try:
        select_word(window, "Bosch")

        assert window.confirm_item() is True

        counts = store.load().counts_for(Provider.ELEVENLABS)
        assert (counts.chosen, counts.corrected) == (1, 0)
    finally:
        window.close()


def test_a_word_a_folder_rule_answered_is_not_counted(qapp, tmp_path):
    folder = two_file_folder()
    changes = []
    window = open_counting_window(tmp_path, folder, lambda change: changes.append(change) or True)
    try:
        select_word(window, "Bosch")
        occurrence = window.current_occurrence()
        occurrence.auto_applied = True
        transcript = window._transcript(occurrence.recording_name)

        corrected = transcript.with_correction(occurrence.token_id, text="Bosche")
        assert window._hand_on(occurrence.recording_name, corrected) is True

        assert changes == []
        assert folder.token(occurrence.recording_name, occurrence.token_id).statistics_note is None
    finally:
        window.close()


def test_a_statistics_failure_keeps_the_correction_and_says_so(qapp, tmp_path, monkeypatch):
    folder = two_file_folder()
    window = open_counting_window(tmp_path, folder, lambda change: False)
    try:
        select_word(window, "Bosch")
        occurrence = window.current_occurrence()
        said = capture(monkeypatch)
        window._replacement_edit.setText("Bosche")

        assert window.apply_replacement_to_occurrence() is True

        token = folder.token(occurrence.recording_name, occurrence.token_id)
        assert token.text == "Bosche"
        # The words stay uncounted, so the next save that works counts them.
        assert token.statistics_note is not None
        assert token.statistics_note.counted is False
        assert any("service statistics could not be saved" in message for message, _ in said)
    finally:
        window.close()


def test_the_next_save_that_works_counts_the_words_a_failure_left(qapp, tmp_path):
    folder = two_file_folder()
    store = CalibrationStore(tmp_path / CALIBRATION_FILE_NAME)
    working = [False]

    def record(change):
        return working[0] and store.apply(change.apply)

    window = open_counting_window(tmp_path, folder, record)
    try:
        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosche")
        assert window.apply_replacement_to_occurrence() is True
        assert store.load().total_words == 0

        working[0] = True
        # Any later save of the same file is the next one that works.
        assert window.current_occurrence().recording_name == RECORDING
        assert window.confirm_item() is True

        counts = store.load().counts_for(Provider.ELEVENLABS)
        assert (counts.chosen, counts.corrected) == (1, 1)
    finally:
        window.close()


def test_a_word_replacement_counts_the_words_in_each_file_it_saves(qapp, tmp_path):
    folder = two_file_folder()
    store = CalibrationStore(tmp_path / CALIBRATION_FILE_NAME)
    window = open_counting_window(tmp_path, folder, lambda change: store.apply(change.apply))
    try:
        select_word(window, "Bosch")
        window._replacement_edit.setText("Bosk")

        assert window.apply_replacement_to_word() is True

        counts = store.load().counts_for(Provider.ELEVENLABS)
        assert (counts.chosen, counts.corrected) == (3, 3)
        changed = [
            token
            for name in folder.names
            for token in folder.transcripts[name].tokens
            if token.text == "Bosk"
        ]
        assert len(changed) == 3
        assert all(token.statistics_note.counted for token in changed)
    finally:
        window.close()


def test_a_correction_after_accepting_the_timing_is_counted(qapp, tmp_path):
    folder = two_file_folder()
    store = CalibrationStore(tmp_path / CALIBRATION_FILE_NAME)
    window = open_counting_window(tmp_path, folder, lambda change: store.apply(change.apply))
    try:
        select_word(window, "Bosch")
        assert window.decide_timing(True) is True
        window._replacement_edit.setText("Bosche")

        assert window.apply_replacement_to_occurrence() is True

        counts = store.load().counts_for(Provider.ELEVENLABS)
        assert (counts.chosen, counts.corrected) == (1, 1)
    finally:
        window.close()


# -- Doubt about the speaker alone ---------------------------------------


def speaker_doubt(text: str, start: float) -> FinalToken:
    """A word every service agreed on, given to another person by the second opinion."""
    return make_token(
        text,
        start=start,
        end=start + 0.4,
        reasons=(ReviewReason.SPEAKER_UNCERTAIN,),
        confidence=Confidence.HIGH,
        speaker="speaker_1",
        speaker_alternative="speaker_0",
    )


def speaker_doubt_folder() -> Folder:
    """One stretch of three words in doubt for their speaker, and one real text doubt."""
    return Folder(
        {
            RECORDING: make_transcript(
                [
                    make_token("15,000", start=10.0, end=10.5),
                    speaker_doubt("that", 40.0),
                    speaker_doubt("is", 40.5),
                    speaker_doubt("fine", 41.0),
                ]
            )
        }
    )


def next_tab_stop(widget):
    """The control Tab moves to from this one."""
    candidate = widget.nextInFocusChain()
    while candidate is not widget:
        if (
            candidate.focusPolicy() & Qt.FocusPolicy.TabFocus
            and candidate.isVisible()
            and candidate.isEnabled()
        ):
            return candidate
        candidate = candidate.nextInFocusChain()
    return None


def test_flagged_in_leaves_out_a_word_in_doubt_only_for_its_speaker():
    transcript = speaker_doubt_folder().transcripts[RECORDING]

    items = review_lists.flagged_in(RECORDING, transcript)

    assert [item.text for item in items] == ["15,000"]


def test_a_saved_flagged_word_still_says_what_kind_of_value_it_is():
    """The project file keeps the kind, so the list can name it without a read."""
    amount = make_token("15,000", reasons=(ReviewReason.HIGH_RISK_ENTITY,))
    amount.risk_categories = [RiskCategory.MONEY]

    [item] = review_lists.flagged_in(RECORDING, make_transcript([amount]))
    item.risk_categories.append("a kind this version has never heard of")
    restored = review_lists.restored_token(item)

    assert restored.risk_categories == [RiskCategory.MONEY]
    assert review_queue.reason_text(restored) == "An amount of money that the services heard differently"


def test_a_word_in_doubt_for_its_text_and_its_speaker_is_still_flagged():
    both = make_token(
        "fifteen",
        reasons=(ReviewReason.PROVIDER_DISAGREEMENT, ReviewReason.SPEAKER_UNCERTAIN),
    )

    items = review_lists.flagged_in(RECORDING, make_transcript([both]))

    assert [item.text for item in items] == ["fifteen"]


def test_speaker_only_words_are_not_rows_of_the_word_list(qapp, tmp_path):
    window = open_window(tmp_path, speaker_doubt_folder(), process=True)
    try:
        assert [row.word for row in window._group_model.rows()] == ["15,000"]
    finally:
        window.close()


def test_a_stretch_whose_speaker_is_in_doubt_is_one_row(qapp, tmp_path):
    window = open_window(tmp_path, speaker_doubt_folder(), process=True)
    try:
        model = window._speaker_doubt_model
        assert model.rowCount() == 1
        cells = [model.data(model.index(0, column)) for column in range(model.columnCount())]
        assert cells == [RECORDING, "0:40", "0:41", "Speaker speaker_1 or Jacques"]
        assert window._speaker_doubt_count_label.text() == (
            "1 stretch of speech where the speaker is in doubt."
        )
    finally:
        window.close()


def test_the_speaker_doubts_are_saved_in_the_project(qapp, tmp_path):
    store = ProjectStore(tmp_path)
    window = open_window(tmp_path, speaker_doubt_folder(), store=store, process=True)
    try:
        window._save_project(quiet=True)
    finally:
        window.close()

    saved = store.load().speaker_doubts
    assert [(item.start, item.end, item.speakers) for item in saved] == [
        (40.0, 41.4, ["Speaker speaker_1", "Jacques"])
    ]


def test_regrouping_keeps_the_speaker_doubts(qapp, tmp_path):
    folder = speaker_doubt_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        before = list(window.state.speaker_doubts)
        assert len(before) == 1

        assert window.regroup_words() is True

        assert window.state.speaker_doubts == before
        assert window._speaker_doubt_model.items() == before
    finally:
        window.close()


def test_regrouping_keeps_the_speaker_doubts_of_a_recording_it_could_not_read(
    qapp, tmp_path
):
    """A recording that cannot be read now has not lost its doubts."""
    folder = speaker_doubt_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        before = list(window.state.speaker_doubts)
        del folder.transcripts[RECORDING]

        window.regroup_words()

        assert window.state.speaker_doubts == before
        assert window._speaker_doubt_model.rowCount() == 1
    finally:
        window.close()


def test_the_speaker_doubt_list_says_so_when_there_are_none(qapp, tmp_path):
    window = open_window(tmp_path, process=True)
    try:
        assert window._speaker_doubt_model.rowCount() == 0
        assert window._speaker_doubt_count_label.text().startswith("No speaker doubts.")
    finally:
        window.close()


def test_the_speaker_doubt_list_is_named_as_its_label_reads(qapp, tmp_path):
    window = open_window(tmp_path)
    try:
        label = window._speaker_doubts_label
        assert label.buddy() is window._speaker_doubts
        assert label.text().replace("&", "") == window._speaker_doubts.accessibleName()
        assert window._speaker_doubts.accessibleName() == "Speaker doubts"
        assert window._speaker_doubts.accessibleDescription()
    finally:
        window.close()


def test_tab_reaches_the_speaker_doubts_between_the_occurrences_and_the_settings(
    qapp, tmp_path
):
    window = open_window(tmp_path, speaker_doubt_folder(), process=True, show_details=True)
    try:
        assert next_tab_stop(window._occurrences) is window._speaker_doubts
        assert next_tab_stop(window._speaker_doubts) is window._process_button
    finally:
        window.close()


def test_the_panel_key_reaches_the_speaker_doubts(qapp, tmp_path):
    window = open_window(tmp_path, speaker_doubt_folder(), process=True, show_details=True)
    try:
        window._occurrences.setFocus()
        window.focus_next_panel()

        assert window._status_label.text() == "Speaker doubts"
    finally:
        window.close()


def test_the_simple_window_shows_the_speaker_doubts(qapp, tmp_path):
    """Hiding the details must not hide a stretch the person has to decide."""
    window = open_window(tmp_path, speaker_doubt_folder(), process=True)
    try:
        assert window.state.settings.show_details is False
        assert window._speaker_doubts.isVisibleTo(window)
        assert window._speaker_doubt_model.rowCount() == 1
    finally:
        window.close()


def test_tab_and_the_panel_key_reach_the_speaker_doubts_in_the_simple_window(qapp, tmp_path):
    window = open_window(tmp_path, speaker_doubt_folder(), process=True)
    try:
        assert next_tab_stop(window._groups) is window._speaker_doubts

        window.focus_word_list()
        window.focus_next_panel()

        assert window._status_label.text() == "Speaker doubts"
        assert window._holds_focus(window._speaker_doubts)
    finally:
        window.close()


def test_a_screen_reader_hears_the_times_and_both_speakers(qapp, tmp_path):
    window = open_window(tmp_path, speaker_doubt_folder(), process=True)
    try:
        model = window._speaker_doubt_model
        spoken = [
            model.data(model.index(0, column), Qt.ItemDataRole.AccessibleTextRole)
            for column in range(model.columnCount())
        ]
        assert spoken[0] == RECORDING
        assert spoken[1] == review_lists.time_spoken(40.0)
        assert spoken[2] == review_lists.time_spoken(41.4)
        assert spoken[3] == "Speaker speaker_1 or Jacques"
        assert "Speaker speaker_1 or Jacques" in model.data(
            model.index(0, 2), Qt.ItemDataRole.ToolTipRole
        )
    finally:
        window.close()


def test_moving_to_a_stretch_says_what_it_is(qapp, tmp_path, monkeypatch):
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    window = open_window(tmp_path, speaker_doubt_folder(), process=True)
    try:
        said.clear()
        window._speaker_doubts.select_row(0)

        assert any("Speaker speaker_1 or Jacques" in message for message in said)
    finally:
        window.close()


def test_enter_on_a_stretch_plays_the_whole_stretch(qapp, tmp_path):
    player = FakePlayer()
    window = open_window(tmp_path, speaker_doubt_folder(), player=player, process=True)
    try:
        window._speaker_doubts.setFocus()
        window._speaker_doubts.select_row(0)
        QTest.keyClick(window._speaker_doubts, Qt.Key.Key_Return)
        player.becomes_ready()

        assert player.seeks[-1] == round((40.0 - SHORT_CONTEXT_SECONDS) * 1000)
        assert window._stop_at_ms is not None
        assert window._stop_at_ms == round((41.4 + SHORT_CONTEXT_SECONDS) * 1000)
        assert player.plays >= 1
    finally:
        window.close()


# -- The simple window, and the switch that shows the details ------------


def detail_panels(window: ReviewWindow) -> list:
    return [
        window._occurrences_holder,
        window._settings_group,
        window._details_group,
        window._corrections_group,
        window._notes_group,
    ]


def shown_choices(window: ReviewWindow) -> list[str]:
    return [
        button.accessibleName()
        for button in window._choice_buttons
        if button.isVisibleTo(window)
    ]


def capture_announcements(monkeypatch) -> list[str]:
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    return said


def test_a_new_window_opens_simple(qapp, tmp_path):
    """The words, the choices for one of them, and playback. Nothing else."""
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        assert window.state.settings.show_details is False
        for panel in detail_panels(window):
            assert not panel.isVisibleTo(window)
        for panel in (window._groups, window._choices_group, window._playback_group):
            assert panel.isVisibleTo(window)
        assert window._show_details_action.isCheckable()
        assert not window._show_details_action.isChecked()
        assert window._show_details_action.shortcut() == QKeySequence("Ctrl+D")
    finally:
        window.close()


def test_the_simple_word_list_shows_the_word_its_time_and_why(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        def visible_titles() -> list[str]:
            return [
                GROUP_COLUMN_TITLES[column]
                for column in range(window._group_model.columnCount())
                if not window._groups.isColumnHidden(column)
            ]

        assert visible_titles() == ["Word", "Time", "Why it needs review"]

        window.set_show_details(True)

        assert visible_titles() == list(GROUP_COLUMN_TITLES)
    finally:
        window.close()


def test_a_word_said_once_gives_its_time_alone(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        groups = window._group_model
        position = groups.row_for_key(group_row(window, "15,000").key)

        assert groups.data(groups.index(position, GROUP_COLUMN_WHEN)) == "1:10"
        assert groups.data(
            groups.index(position, GROUP_COLUMN_WHEN), Qt.ItemDataRole.AccessibleTextRole
        ) == "1 minute 10 seconds"
    finally:
        window.close()


def test_there_is_one_choice_for_each_different_word_the_services_heard(qapp, tmp_path):
    """Two services heard 50,000, so that is one button naming both."""
    window = candidate_window(tmp_path)
    try:
        assert shown_choices(window) == [
            f"Keep 50,000, said by {Provider.OPENAI.display_name} and "
            f"{Provider.MICROSOFT.display_name}",
            f"Use 15,000, said by {Provider.ELEVENLABS.display_name}",
        ]
        # The words are on the buttons as well, not only in their names.
        assert window._choice_buttons[1].text().startswith("Use 15,000")
        for button in window._choice_buttons:
            assert "&" not in button.text().replace("&&", "")
    finally:
        window.close()


def test_a_word_is_kept_as_detected_without_opening_the_details(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        select_word(window, "Bosch")
        group = window.state.group(window.current_row().group_id)
        assert shown_choices(window)[0] == "Keep Bosch"

        window._choice_buttons[0].click()

        assert group.reviewed is True
        assert group.correct_as_detected is True
    finally:
        window.close()


def test_another_services_word_replaces_it_without_opening_the_details(qapp, tmp_path):
    folder = Folder({RECORDING: make_candidate_transcript()})
    window = open_window(tmp_path, folder, process=True)
    try:
        window._choice_buttons[1].click()

        assert folder.texts(RECORDING) == ["15,000"]
    finally:
        window.close()


def test_a_typed_word_is_applied_with_enter(qapp, tmp_path):
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        window._typed_edit.setText("Bausch")

        QTest.keyClick(window._typed_edit, Qt.Key.Key_Return)

        assert "Bausch" in folder.texts(RECORDING)
        assert "Bausch" in folder.texts(OTHER_RECORDING)
        assert window._typed_edit.text() == ""
    finally:
        window.close()


def test_an_empty_typed_word_is_refused_out_loud(qapp, tmp_path, monkeypatch):
    said = capture_announcements(monkeypatch)
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        said.clear()
        window._typed_edit.setText("   ")

        assert window.apply_typed_word() is False

        assert said == ["Type the word that was said first. The box is empty."]
        assert folder.saved == []
    finally:
        window.close()


def test_typing_the_word_as_it_stands_names_the_keep_button(qapp, tmp_path, monkeypatch):
    said = capture_announcements(monkeypatch)
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        window.set_show_reviewed(True)
        select_word(window, "Bosch")
        window._typed_edit.setText("Bausch")
        assert window.apply_typed_word() is True
        select_word(window, "Bosch")
        said.clear()
        window._typed_edit.setText("Bausch")

        assert window.apply_typed_word() is False

        assert said == [
            "The replacement is unchanged. Use the Keep button under Decide this "
            "word to settle this word as it stands."
        ]
    finally:
        window.close()


def test_the_switch_shows_and_hides_the_details_and_says_so(qapp, tmp_path, monkeypatch):
    said = capture_announcements(monkeypatch)
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        window._show_details_action.trigger()

        assert window._show_details_action.isChecked()
        for panel in detail_panels(window):
            assert panel.isVisibleTo(window)
        # The Corrections panel does the job of the choices when it is on show.
        assert not window._choices_group.isVisibleTo(window)
        assert said[-1] == "The details are now shown."

        window._show_details_action.trigger()

        assert not window._show_details_action.isChecked()
        for panel in detail_panels(window):
            assert not panel.isVisibleTo(window)
        assert said[-1] == "The details are now hidden."
    finally:
        window.close()


def test_the_switch_is_remembered_with_the_folder(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        window.set_show_details(True)
    finally:
        window.close()

    assert ProjectStore(tmp_path).load().settings.show_details is True
    window = open_window(tmp_path, two_file_folder())
    try:
        assert window._show_details_action.isChecked()
        assert window._occurrences_holder.isVisibleTo(window)
    finally:
        window.close()


def test_hiding_the_details_takes_the_focus_somewhere_visible(qapp, tmp_path, monkeypatch):
    said = capture_announcements(monkeypatch)
    window = open_window(tmp_path, two_file_folder(), process=True, show_details=True)
    try:
        window.focus_occurrence_list()
        assert window.focusWidget() is window._occurrences

        window.set_show_details(False)

        assert window.focusWidget() is window._groups
        assert said[-1] == (
            "The details are now hidden. The focus has moved to the word groups."
        )
    finally:
        window.close()


def test_showing_the_details_from_a_choice_moves_to_the_replacement(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        window._choice_buttons[0].setFocus()

        window.set_show_details(True)

        assert window.focusWidget() is window._replacement_edit
    finally:
        window.close()


def test_tab_moves_through_the_simple_window_in_reading_order(qapp, tmp_path):
    window = candidate_window(tmp_path)
    try:
        def reachable(widget) -> bool:
            return bool(
                widget.isVisibleTo(window)
                and widget.isEnabled()
                and widget.focusPolicy() & Qt.FocusPolicy.TabFocus
            )

        chain = [window._groups]
        widget = window._groups.nextInFocusChain()
        while widget is not window._groups:
            if reachable(widget):
                chain.append(widget)
            widget = widget.nextInFocusChain()

        expected = [
            window._groups,
            window._speaker_doubts,
            window._choice_buttons[0],
            window._choice_buttons[1],
            window._typed_edit,
            window._apply_typed_button,
            window._span_edit,
            window._play_button,
            window._play_short_button,
            window._play_wide_button,
        ]
        assert [widget for widget in chain if widget in expected] == expected
        # Only the word list, the speaker doubts, the choices and playback
        # are on the way.
        assert set(chain) <= set(expected)
    finally:
        window.close()


def test_the_panel_key_skips_the_hidden_panels(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        window.focus_word_list()
        names = []
        for _ in range(4):
            window.focus_next_panel()
            names.append(window._status_label.text())

        assert names == ["Speaker doubts", "Decide this word", "Playback", "Word Groups"]
    finally:
        window.close()


def test_f2_in_the_simple_window_goes_to_the_typed_word(qapp, tmp_path):
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        window.focus_replacement()

        assert window.focusWidget() is window._typed_edit
    finally:
        window.close()


def test_the_apply_to_word_menu_item_uses_the_typed_word_in_the_simple_window(qapp, tmp_path):
    """F2 sends the typing to the typed-word box, so the menu must read that box."""
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        window.focus_replacement()
        window._typed_edit.setText("Bausch")

        window._apply_word_action.trigger()

        assert "Bausch" in folder.texts(RECORDING)
        assert "Bausch" in folder.texts(OTHER_RECORDING)
        assert window._typed_edit.text() == ""
    finally:
        window.close()


def test_the_apply_to_occurrence_menu_item_uses_the_typed_word_in_the_simple_window(
    qapp, tmp_path
):
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True)
    try:
        select_word(window, "Bosch")
        occurrence = window.current_occurrence()
        window.focus_replacement()
        window._typed_edit.setText("Bausch")

        window._apply_occurrence_action.trigger()

        assert occurrence.replacement == "Bausch"
        assert window._typed_edit.text() == ""
    finally:
        window.close()


def test_the_apply_menu_items_use_the_replacement_box_with_the_details_shown(qapp, tmp_path):
    folder = two_file_folder()
    window = open_window(tmp_path, folder, process=True, show_details=True)
    try:
        select_word(window, "Bosch")
        window._typed_edit.setText("Ignored")
        window._replacement_edit.setText("Bausch")

        window._apply_word_action.trigger()

        assert "Bausch" in folder.texts(RECORDING)
        assert "Ignored" not in folder.texts(RECORDING)
    finally:
        window.close()


@pytest.mark.parametrize("show_details", [False, True])
def test_the_occurrence_summary_is_spoken_from_a_visible_list(
    qapp, tmp_path, monkeypatch, show_details
):
    """A screen reader may not read an announcement raised from a hidden widget."""
    speakers: list[object] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: speakers.append(widget),
    )
    window = open_window(tmp_path, two_file_folder(), process=True, show_details=show_details)
    try:
        window.show()
        select_word(window, "Bosch")
        speakers.clear()

        window.go_to_next_item()

        expected = window._occurrences if show_details else window._groups
        assert speakers and all(widget is expected for widget in speakers)
    finally:
        window.close()


def test_setting_a_threshold_from_the_menu_shows_the_details_first(qapp, tmp_path):
    """The menu entry must never send the focus to a control nobody can see."""
    window = open_window(tmp_path, two_file_folder(), process=True)
    try:
        window._minimum_action.trigger()

        assert window.state.settings.show_details is True
        assert window._settings_group.isVisibleTo(window)
        assert window._holds_focus(window._minimum_spin)
    finally:
        window.close()
