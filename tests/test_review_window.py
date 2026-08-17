"""Tests for the review window: the queue, the detail panel, playback and corrections.

Every transcript is built by hand and the audio player is a stand-in that
records what it was asked to do, so none of this needs a network, an API
key or an audio file. That is the whole point of the window taking a
transcript and a callback rather than reaching for the pipeline itself.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAccessible, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableView,
)

from audio_transcriber.transcription.model import (
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
from audio_transcriber.ui import review_window as review_window_module
from audio_transcriber.ui.review_window import (
    AUDIO_EVENT,
    CANDIDATE_COLUMN_CHOICE,
    CANDIDATE_COLUMN_CONFIDENCE,
    CANDIDATE_COLUMN_KIND,
    CANDIDATE_COLUMN_SERVICE,
    CANDIDATE_COLUMN_TEXT,
    CANDIDATE_COLUMN_VOCABULARY,
    CONTEXT_SECONDS,
    CURRENT_CHOICE,
    IN_VOCABULARY,
    NOT_IN_VOCABULARY,
    WIDE_CONTEXT_SECONDS,
    ReviewWindow,
    candidate_rows,
)


class FakePlayer(QObject):
    """Stands in for the audio player, recording what it was asked to do.

    It carries the same signals as the real one, because the window
    watches the position to know when to stop at the end of a region.
    """

    positionChanged = Signal(int)
    durationChanged = Signal(int)
    playingChanged = Signal(bool)
    playbackFinished = Signal()
    errorOccurred = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.loaded: str | None = None
        self.seeks: list[int] = []
        self.plays = 0
        self.pauses = 0

    def load(self, path) -> None:
        self.loaded = str(path)

    def seek_to(self, milliseconds: int) -> None:
        self.seeks.append(int(milliseconds))

    def play(self) -> None:
        self.plays += 1

    def pause(self) -> None:
        self.pauses += 1


def make_token(
    text: str = "contract",
    start: float | None = 30.0,
    end: float | None = 30.5,
    reasons: tuple[ReviewReason, ...] = (ReviewReason.PROVIDER_DISAGREEMENT,),
    confidence: Confidence = Confidence.REVIEW_REQUIRED,
    speaker: str | None = "speaker_0",
    span: AudioSpan | None = None,
    candidates: tuple[Candidate, ...] = (),
    **extra,
) -> FinalToken:
    token = FinalToken(
        text=text,
        start=start,
        end=end,
        text_source=Provider.ELEVENLABS,
        text_confidence=confidence,
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


def make_transcript(
    tokens: list[FinalToken] | None = None,
    duration: float = 60.0,
    with_audio: bool = True,
) -> Transcript:
    transcript = Transcript(recording_name="board meeting")
    transcript.tokens = list(tokens or [make_token()])
    transcript.speakers = [Speaker(id="speaker_0", name="Jacques"), Speaker(id="speaker_1")]
    if with_audio:
        transcript.canonical_audio = CanonicalAudio(
            path="C:/Audio/board.wav",
            original_path="C:/Audio/board.m4a",
            duration=duration,
            sample_rate=48000,
            channels=1,
            size_bytes=1024,
            container="wav",
        )
    return transcript


def open_window(transcript=None, player=None, save=None) -> ReviewWindow:
    window = ReviewWindow(transcript or make_transcript(), player or FakePlayer(), save)
    window.show()
    return window


def interactive_widgets(window: ReviewWindow) -> list:
    """Every control a person can land on, whatever it happens to be.

    Gathered by widget type rather than by name, so that a control added
    later is caught by these tests without anybody remembering to list it.
    """
    kinds = (QPushButton, QLineEdit, QCheckBox, QComboBox, QPlainTextEdit, QTableView)
    return [widget for kind in kinds for widget in window.findChildren(kind)]


def accessible_name(widget) -> str:
    interface = QAccessible.queryAccessibleInterface(widget)
    assert interface is not None
    return interface.text(QAccessible.Text.Name)


# -- The queue -----------------------------------------------------------


def test_the_queue_holds_only_the_words_needing_a_person(qapp):
    settled = FinalToken(text="the", start=0.1, end=0.2, text_confidence=Confidence.HIGH)
    window = open_window(make_transcript([settled, make_token("contract")]))
    try:
        assert window._model.rowCount() == 1
        assert window.current_token().text == "contract"
    finally:
        window.close()


def test_the_window_opens_on_the_first_item_rather_than_on_nothing(qapp):
    window = open_window(make_transcript([make_token("one"), make_token("two")]))
    try:
        assert window._queue.selected_row() == 0
        assert window.current_token().text == "one"
    finally:
        window.close()


def test_a_transcript_with_nothing_to_review_says_so_and_switches_the_actions_off(qapp):
    settled = FinalToken(text="the", start=0.1, end=0.2, text_confidence=Confidence.HIGH)
    window = open_window(make_transcript([settled]))
    try:
        assert window._model.rowCount() == 0
        assert window._count_label.text() == "There is nothing to review."
        assert not window._confirm_button.isEnabled()
        assert not window._play_button.isEnabled()
    finally:
        window.close()


def test_moving_through_the_queue_works_without_leaving_the_keyboard(qapp):
    window = open_window(make_transcript([make_token("one"), make_token("two")]))
    try:
        window.go_to_next_item()
        assert window.current_token().text == "two"

        window.go_to_previous_item()
        assert window.current_token().text == "one"
    finally:
        window.close()


def test_the_ends_of_the_queue_are_stated_rather_than_wrapping_round(qapp):
    """Silently starting again at the top would have items reviewed twice."""
    window = open_window(make_transcript([make_token("one")]))
    try:
        window.go_to_previous_item()
        assert "first item" in window._status_label.text()

        window.go_to_next_item()
        assert "last item" in window._status_label.text()
    finally:
        window.close()


def test_moving_the_selection_reads_the_whole_item_out(qapp, monkeypatch):
    """A person works down the queue without hunting for the detail panel."""
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    window = open_window(
        make_transcript(
            [
                make_token("one"),
                make_token(
                    "15,000", start=92.0, end=92.4, reasons=(ReviewReason.HIGH_RISK_ENTITY,)
                ),
            ]
        )
    )
    try:
        said.clear()
        window.go_to_next_item()

        assert said == [
            "1 minute 32 seconds. 15,000. A value that must not be guessed. Review required."
        ]
    finally:
        window.close()


# -- The filters ---------------------------------------------------------


def test_every_reason_has_a_check_box_of_its_own_named_for_what_it_means(qapp):
    """Not a combo box of magic strings that shows one choice at a time."""
    window = open_window()
    try:
        assert set(window._reason_boxes) == set(ReviewReason)
        for reason, box in window._reason_boxes.items():
            assert box.text() == reason.display_name
            assert box.accessibleName() == reason.display_name
            assert box.isChecked()
    finally:
        window.close()


def test_every_confidence_category_has_a_check_box_of_its_own(qapp):
    window = open_window()
    try:
        assert set(window._confidence_boxes) == set(Confidence)
        for confidence, box in window._confidence_boxes.items():
            assert box.text() == confidence.display_name
    finally:
        window.close()


def test_clearing_a_reason_hides_the_words_flagged_for_it(qapp):
    window = open_window(
        make_transcript(
            [
                make_token("one", reasons=(ReviewReason.PROVIDER_DISAGREEMENT,)),
                make_token("two", reasons=(ReviewReason.SPEAKER_UNCERTAIN,)),
            ]
        )
    )
    try:
        window._reason_boxes[ReviewReason.SPEAKER_UNCERTAIN].setChecked(False)

        assert [token.text for token in window._model.visible_tokens()] == ["one"]
    finally:
        window.close()


def test_clearing_a_confidence_category_hides_the_words_rated_that_way(qapp):
    window = open_window(
        make_transcript(
            [
                make_token("one", confidence=Confidence.REVIEW_REQUIRED),
                make_token("two", confidence=Confidence.UNRESOLVED),
            ]
        )
    )
    try:
        window._confidence_boxes[Confidence.UNRESOLVED].setChecked(False)

        assert [token.text for token in window._model.visible_tokens()] == ["one"]
    finally:
        window.close()


def test_changing_a_filter_says_how_many_items_are_showing(qapp, monkeypatch):
    """A filter that silently empties the list looks exactly like a broken one."""
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    window = open_window(make_transcript([make_token("one"), make_token("two")]))
    try:
        said.clear()
        window._reason_boxes[ReviewReason.PROVIDER_DISAGREEMENT].setChecked(False)

        assert "No items match the filters. All 2 items are hidden." in said
        assert window._count_label.text() == (
            "No items match the filters. All 2 items are hidden."
        )
    finally:
        window.close()


def test_the_count_is_also_on_the_queue_itself_for_anyone_tabbing_into_it(qapp):
    window = open_window(make_transcript([make_token("one"), make_token("two")]))
    try:
        assert window._queue.accessibleDescription().startswith("Showing all 2 items.")

        window._confidence_boxes[Confidence.REVIEW_REQUIRED].setChecked(False)

        assert window._queue.accessibleDescription().startswith("No items match the filters.")
    finally:
        window.close()


def test_showing_everything_is_one_change_and_one_announcement(qapp, monkeypatch):
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    window = open_window(make_transcript([make_token("one"), make_token("two")]))
    try:
        window._reason_boxes[ReviewReason.PROVIDER_DISAGREEMENT].setChecked(False)
        said.clear()

        window.show_everything()

        assert said.count("Showing all 2 items.") == 1
        assert all(box.isChecked() for box in window._reason_boxes.values())
        assert all(box.isChecked() for box in window._confidence_boxes.values())
    finally:
        window.close()


# -- The detail panel ----------------------------------------------------


def test_the_three_answers_are_shown_as_three_things_with_their_own_sources(qapp):
    """Merging them into one line would hide what is being decided."""
    token = make_token("Jurgen", start=92.0, end=92.6)
    token.timing_status = TimingStatus.SHARED_PHRASE_SPAN
    token.timing_confidence = Confidence.REVIEW_SUGGESTED
    token.speaker_confidence = Confidence.REVIEW_REQUIRED
    window = open_window(make_transcript([token]))
    try:
        # The start and the exact length, rather than two ends both rounded
        # to the nearest second, which would say the word ran from 1:32 to
        # 1:33 and lasted 0.6 seconds in the same breath.
        assert window._timing_edit.text() == "From 1 minute 32 seconds, 0.6 seconds long"
        assert "share one span" in window._timing_status_edit.text()
        assert window._timing_source_edit.text() == Provider.ELEVENLABS.display_name
        assert window._timing_confidence_edit.text() == "Review suggested"

        assert window._speaker_edit.text() == "Jacques"
        assert window._speaker_source_edit.text() == Provider.ASSEMBLYAI.display_name
        assert window._speaker_confidence_edit.text() == "Review required"
    finally:
        window.close()


def test_a_word_with_no_boundaries_is_described_by_the_region_it_sits_in(qapp):
    token = make_token(start=None, end=None, span=AudioSpan(30.0, 33.0))
    token.timing_status = TimingStatus.UNALIGNED
    window = open_window(make_transcript([token]))
    try:
        assert window._timing_edit.text() == "Somewhere in the 3.0 seconds from 30 seconds"
    finally:
        window.close()


def test_the_language_evidence_is_shown_as_several_numbers_not_one_answer(qapp):
    token = make_token()
    token.language = Language.GERMAN
    token.language_evidence = LanguageEvidence(
        scores={Language.GERMAN: 0.62, Language.ENGLISH: 0.38}
    )
    window = open_window(make_transcript([token]))
    try:
        assert window._language_edit.text() == "German"
        assert window._language_evidence_edit.text() == "German 62 percent, English 38 percent"
    finally:
        window.close()


def test_risk_categories_are_spelled_out_because_they_must_never_be_guessed(qapp):
    token = make_token("15,000", reasons=(ReviewReason.HIGH_RISK_ENTITY,))
    token.risk_categories = [RiskCategory.MONEY, RiskCategory.ACCOUNT_NUMBER]
    window = open_window(make_transcript([token]))
    try:
        assert window._risk_edit.text() == "Money, Account number"
    finally:
        window.close()


def test_what_the_language_model_decided_is_shown_where_it_was_consulted(qapp):
    decided = make_token("15,000", llm_decision="Chose 15,000 on the surrounding figures.")
    silent = make_token("contract")
    window = open_window(make_transcript([decided, silent]))
    try:
        assert window._decision_text.toPlainText() == (
            "Chose 15,000 on the surrounding figures."
        )

        window.go_to_next_item()

        assert window._decision_text.toPlainText() == "The language model was not consulted."
    finally:
        window.close()


def test_moving_the_selection_fills_the_detail_panel_with_the_new_item(qapp):
    first = make_token("one", start=10.0, end=10.4, speaker="speaker_0")
    second = make_token("two", start=20.0, end=20.4, speaker="speaker_1")
    window = open_window(make_transcript([first, second]))
    try:
        assert window._speaker_edit.text() == "Jacques"
        assert window._text_edit.text() == "one"

        window.go_to_next_item()

        assert window._speaker_edit.text() == "Speaker speaker_1"
        assert window._text_edit.text() == "two"
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


def test_the_candidates_reach_the_table_a_screen_reader_reads(qapp):
    window = open_window(make_candidate_transcript())
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


def test_a_candidate_can_be_read_closely_in_a_field_with_a_real_caret(qapp):
    """Names and numbers have to be spelled out, not glanced at."""
    window = open_window(make_candidate_transcript())
    try:
        assert window._candidate_text.text() == "15,000"
        assert window._candidate_text.isReadOnly()

        window._candidates.setCurrentIndex(
            window._candidates_model.index(1, CANDIDATE_COLUMN_SERVICE)
        )

        assert window._candidate_text.text() == "50,000"
    finally:
        window.close()


def test_a_candidate_can_be_put_straight_into_the_correction_box(qapp):
    window = open_window(make_candidate_transcript())
    try:
        window.use_selected_candidate()

        assert window._text_edit.text() == "15,000"
        # Nothing has changed yet: the correction still has to be applied.
        assert window.transcript.tokens[0].text == "50,000"
    finally:
        window.close()


# -- Playback ------------------------------------------------------------


def test_playing_a_word_includes_several_seconds_either_side_of_it(qapp):
    """A word without its sentence cannot be judged."""
    window = open_window(make_transcript([make_token(start=30.0, end=30.5)]))
    try:
        span = window.span_to_play()

        assert span == AudioSpan(30.0 - CONTEXT_SECONDS, 30.5 + CONTEXT_SECONDS)
    finally:
        window.close()


def test_asking_for_more_context_widens_the_same_span(qapp):
    window = open_window(make_transcript([make_token(start=30.0, end=30.5)]))
    try:
        span = window.span_to_play(wide=True)

        assert span == AudioSpan(30.0 - WIDE_CONTEXT_SECONDS, 30.5 + WIDE_CONTEXT_SECONDS)
    finally:
        window.close()


def test_the_padding_never_asks_for_audio_outside_the_recording(qapp):
    window = open_window(
        make_transcript([make_token(start=0.5, end=1.0), make_token(start=59.0, end=59.5)],
                        duration=60.0)
    )
    try:
        assert window.span_to_play().start == 0.0

        window.go_to_next_item()

        assert window.span_to_play().end == 60.0
    finally:
        window.close()


def test_a_word_with_no_boundaries_plays_the_wider_region_it_was_found_in(qapp):
    """Which is exactly when a person most needs to hear it."""
    window = open_window(
        make_transcript([make_token(start=None, end=None, span=AudioSpan(30.0, 33.0))])
    )
    try:
        span = window.span_to_play()

        assert span == AudioSpan(30.0 - CONTEXT_SECONDS, 33.0 + CONTEXT_SECONDS)
    finally:
        window.close()


def test_playing_seeks_to_the_start_of_the_padded_span_and_starts(qapp):
    player = FakePlayer()
    window = open_window(make_transcript([make_token(start=30.0, end=30.5)]), player=player)
    try:
        assert player.loaded == "C:/Audio/board.wav"

        assert window.play_span() is True

        assert player.seeks[-1] == 27_000
        assert player.plays == 1
    finally:
        window.close()


def test_playback_stops_at_the_end_of_the_region_rather_than_running_on(qapp):
    player = FakePlayer()
    window = open_window(make_transcript([make_token(start=30.0, end=30.5)]), player=player)
    try:
        window.play_span()

        # Still inside the region, so it plays on.
        player.positionChanged.emit(33_000)
        assert player.pauses == 0

        player.positionChanged.emit(33_500)
        assert player.pauses == 1

        # And having stopped, it stays stopped rather than pausing again on
        # every position the player reports afterwards.
        player.positionChanged.emit(34_000)
        assert player.pauses == 1
    finally:
        window.close()


def test_a_word_with_no_audio_at_all_says_so_instead_of_playing_silence(qapp):
    player = FakePlayer()
    window = open_window(
        make_transcript([make_token(start=None, end=None, span=None)]), player=player
    )
    try:
        assert window.play_span() is False

        assert player.plays == 0
        assert not window._play_button.isEnabled()
        assert "no audio behind it" in window._span_edit.text()
    finally:
        window.close()


def test_the_field_says_what_pressing_play_would_actually_play(qapp):
    window = open_window(make_transcript([make_token(start=30.0, end=30.5)]))
    try:
        assert window._span_edit.text() == "From 27 seconds, 6.5 seconds in all"
    finally:
        window.close()


def test_a_transcript_with_no_audio_file_says_so_rather_than_failing(qapp):
    player = FakePlayer()
    window = open_window(
        make_transcript([make_token()], with_audio=False), player=player
    )
    try:
        assert player.loaded is None
        assert window.play_span() is False
        assert "no audio file for this transcript" in window._span_edit.text()
    finally:
        window.close()


# -- Corrections ---------------------------------------------------------


def test_correcting_the_text_leaves_the_timing_and_the_speaker_alone(qapp):
    token = make_token("Jurgen", start=30.0, end=30.5, speaker="speaker_0")
    window = open_window(make_transcript([token]))
    try:
        window._text_edit.setText("Jürgen")

        assert window.apply_text_correction() is True

        corrected = window.transcript.token_by_id(token.id)
        assert corrected.text == "Jürgen"
        assert corrected.original_text == "Jurgen"
        assert corrected.text_confidence is Confidence.HIGH
        assert corrected.start == 30.0
        assert corrected.end == 30.5
        assert corrected.timing_source is Provider.ELEVENLABS
        assert corrected.timing_status is TimingStatus.EXACT_PROVIDER_TIME
        assert corrected.speaker == "speaker_0"
        assert corrected.speaker_source is Provider.ASSEMBLYAI
    finally:
        window.close()


def test_correcting_the_speaker_leaves_the_text_and_the_timing_alone(qapp):
    token = make_token("contract", speaker="speaker_0")
    window = open_window(make_transcript([token]))
    try:
        window._speaker_box.setCurrentText("Speaker speaker_1")

        assert window.apply_speaker_correction() is True

        corrected = window.transcript.token_by_id(token.id)
        assert corrected.speaker == "speaker_1"
        assert corrected.speaker_confidence is Confidence.HIGH
        assert corrected.text == "contract"
        assert corrected.text_source is Provider.ELEVENLABS
        assert corrected.start == 30.0
        assert corrected.timing_status is TimingStatus.EXACT_PROVIDER_TIME
    finally:
        window.close()


def test_a_speaker_chosen_by_name_is_recorded_by_the_label_the_services_used(qapp):
    """Saving the friendly name would key the word on something nothing else knows."""
    token = make_token(speaker="speaker_1")
    window = open_window(make_transcript([token]))
    try:
        window._speaker_box.setCurrentText("Jacques")

        assert window.chosen_speaker() == "speaker_0"

        window.apply_speaker_correction()

        assert window.transcript.token_by_id(token.id).speaker == "speaker_0"
    finally:
        window.close()


def test_a_speaker_the_services_never_separated_out_can_still_be_typed(qapp):
    token = make_token(speaker="speaker_0")
    window = open_window(make_transcript([token]))
    try:
        window._speaker_box.setCurrentText("the chairman")

        assert window.apply_speaker_correction() is True

        assert window.transcript.token_by_id(token.id).speaker == "the chairman"
    finally:
        window.close()


def test_confirming_the_timing_leaves_the_text_and_the_speaker_alone(qapp):
    token = make_token(
        "contract",
        reasons=(ReviewReason.WEAK_ALIGNMENT,),
        confidence=Confidence.REVIEW_SUGGESTED,
    )
    token.timing_confidence = Confidence.REVIEW_REQUIRED
    window = open_window(make_transcript([token]))
    try:
        assert window.decide_timing(True) is True

        decided = window.transcript.token_by_id(token.id)
        assert decided.timing_confidence is Confidence.HIGH
        assert decided.start == 30.0
        assert decided.text == "contract"
        assert decided.text_confidence is Confidence.REVIEW_SUGGESTED
        assert decided.speaker == "speaker_0"
        assert decided.review_reasons == []
        assert decided.needs_review is False
    finally:
        window.close()


def test_confirming_the_timing_leaves_the_word_in_the_queue_for_its_other_reasons(qapp):
    """A word also disputed on spelling is not settled by settling its clock."""
    token = make_token(
        "15,000",
        reasons=(ReviewReason.WEAK_ALIGNMENT, ReviewReason.NUMERIC_DISAGREEMENT),
    )
    window = open_window(make_transcript([token]))
    try:
        window.decide_timing(True)

        decided = window.transcript.token_by_id(token.id)
        assert decided.review_reasons == [ReviewReason.NUMERIC_DISAGREEMENT]
        assert decided.needs_review is True
        assert window._model.rowCount() == 1
    finally:
        window.close()


def test_rejecting_the_timing_keeps_the_numbers_so_the_word_can_still_be_played(qapp):
    token = make_token("contract", start=30.0, end=30.5)
    window = open_window(make_transcript([token]))
    try:
        assert window.decide_timing(False) is True

        decided = window.transcript.token_by_id(token.id)
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


def test_confirming_an_item_settles_it_without_changing_a_thing(qapp):
    """The common case, and the fastest thing on the screen."""
    token = make_token("contract", confidence=Confidence.UNRESOLVED)
    window = open_window(make_transcript([token]))
    try:
        assert window.confirm_item() is True

        confirmed = window.transcript.token_by_id(token.id)
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


def test_a_settled_item_leaves_the_queue_and_the_work_carries_on(qapp):
    window = open_window(make_transcript([make_token("one"), make_token("two")]))
    try:
        window.confirm_item()

        assert window._model.rowCount() == 1
        assert window.current_token().text == "two"
    finally:
        window.close()


def test_a_correction_is_announced_along_with_where_the_person_now_is(qapp, monkeypatch):
    """The change happens away from where the focus lands next."""
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    window = open_window(make_transcript([make_token("one"), make_token("two")]))
    try:
        said.clear()
        window._text_edit.setText("won")
        window.apply_text_correction()

        assert len(said) == 1
        assert said[0].startswith(
            "Text corrected to won. The timing and the speaker are unchanged."
        )
        assert "Now on" in said[0]
        assert "two" in said[0]
    finally:
        window.close()


def test_the_last_correction_says_the_queue_is_empty_rather_than_going_quiet(qapp, monkeypatch):
    said: list[str] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    window = open_window(make_transcript([make_token("one")]))
    try:
        said.clear()
        window.confirm_item()

        assert said == ["one confirmed as correct. There is nothing to review."]
    finally:
        window.close()


def test_a_correction_that_changes_nothing_is_refused_and_said_out_loud(qapp):
    token = make_token("contract")
    window = open_window(make_transcript([token]))
    try:
        window._text_edit.setText("contract")

        assert window.apply_text_correction() is False

        assert "unchanged" in window._status_label.text()
        assert window.transcript.token_by_id(token.id).review_status is ReviewStatus.PENDING
    finally:
        window.close()


def test_an_empty_correction_is_refused_rather_than_deleting_the_word(qapp):
    token = make_token("contract")
    window = open_window(make_transcript([token]))
    try:
        window._text_edit.setText("   ")

        assert window.apply_text_correction() is False

        assert window.transcript.token_by_id(token.id).text == "contract"
    finally:
        window.close()


def test_every_correction_is_handed_to_the_callback_and_the_signal(qapp):
    """The window knows nothing about storage, so somebody else must be told."""
    saved: list[Transcript] = []
    emitted: list[Transcript] = []
    window = open_window(make_transcript([make_token("one")]), save=saved.append)
    window.transcriptChanged.connect(emitted.append)
    try:
        window.confirm_item()

        assert len(saved) == 1
        assert saved[0] is window.transcript
        assert emitted == saved
    finally:
        window.close()


def test_correcting_does_not_reach_back_into_the_transcript_it_replaced(qapp):
    """The copies share their lists, so a correction must rebind rather than edit."""
    original = make_transcript([make_token("one", reasons=(ReviewReason.HIGH_RISK_ENTITY,))])
    before = list(original.tokens[0].review_reasons)
    window = open_window(original)
    try:
        window.confirm_item()

        assert original.tokens[0].review_reasons == before
        assert original.tokens[0].review_status is ReviewStatus.PENDING
    finally:
        window.close()


# -- Accessibility -------------------------------------------------------


def test_every_control_in_the_window_has_an_accessible_name(qapp):
    """Gathered by type, so a control added later cannot slip through unnamed."""
    window = open_window(make_candidate_transcript())
    try:
        unnamed = [
            f"{widget.__class__.__name__} in {widget.parentWidget().__class__.__name__}"
            for widget in interactive_widgets(window)
            if not widget.accessibleName()
        ]

        assert unnamed == []
    finally:
        window.close()


def test_every_control_in_the_window_can_be_reached_from_the_keyboard(qapp):
    window = open_window(make_candidate_transcript())
    try:
        unreachable = [
            widget.accessibleName()
            for widget in interactive_widgets(window)
            if not (widget.focusPolicy() & Qt.FocusPolicy.TabFocus)
        ]

        assert unreachable == []
    finally:
        window.close()


def test_no_two_controls_answer_the_same_alt_key(qapp):
    """Two controls on one Alt letter means neither is reliably reachable."""
    window = open_window()
    try:
        letters = [
            text[text.index("&") + 1].casefold()
            for text in (
                widget.text()
                for kind in (QLabel, QPushButton, QCheckBox)
                for widget in window.findChildren(kind)
            )
            if "&" in text and not text.endswith("&")
        ]

        assert sorted(letters) == sorted(set(letters)), f"repeated Alt letters in {letters}"
    finally:
        window.close()


def test_each_field_is_reached_by_its_own_label(qapp):
    """A label with no buddy names nothing, and the field is read as blank."""
    window = open_window()
    try:
        labelled = {label.buddy() for label in window.findChildren(QLabel) if label.buddy()}

        for field in (
            window._queue,
            window._candidate_text,
            window._timing_edit,
            window._speaker_edit,
            window._language_edit,
            window._risk_edit,
            window._decision_text,
            window._span_edit,
            window._text_edit,
            window._speaker_box,
        ):
            assert field in labelled, f"{field.accessibleName()} has no label pointing at it"
    finally:
        window.close()


def test_the_panel_key_lands_on_a_control_rather_than_a_container(qapp):
    """F6 must not strand the focus on a container that answers no keys."""
    window = open_window(make_candidate_transcript())
    try:
        landed = []
        for _ in range(6):
            window.focus_next_panel()
            landed.append(window.focusWidget())

        assert all(widget is not None for widget in landed)
        for widget in landed:
            assert widget.focusPolicy() != Qt.FocusPolicy.NoFocus
        # Six panels, and F6 visits a different control in each of them.
        assert len(set(landed)) == 6
    finally:
        window.close()


def test_the_panel_key_says_which_panel_it_landed_on(qapp):
    window = open_window()
    try:
        window.focus_next_panel()

        assert window._status_label.text() in (
            "Filters",
            "Review queue",
            "Item details",
            "Playback",
            "Corrections",
            "About this control",
        )
    finally:
        window.close()


def test_every_action_has_a_shortcut_a_person_can_actually_press(qapp):
    window = open_window()
    try:
        assert window._next_action.shortcut() == QKeySequence(Qt.Key.Key_F3)
        assert window._previous_action.shortcut() == QKeySequence("Shift+F3")
        assert window._correct_text_action.shortcut() == QKeySequence(Qt.Key.Key_F2)
        # Confirming is the common case, so it is one keystroke with no
        # modifier and nothing to hold down.
        assert window._confirm_action.shortcut() == QKeySequence(Qt.Key.Key_F4)
        assert window._play_action.shortcut() == QKeySequence(Qt.Key.Key_F5)
        assert window._play_wide_action.shortcut() == QKeySequence("Shift+F5")
    finally:
        window.close()


def test_the_note_panel_follows_the_focus_onto_every_control_that_has_one(qapp):
    window = open_window(make_candidate_transcript())
    try:
        for widget, key in (
            (window._queue, review_window_module.QUEUE),
            (window._candidates, review_window_module.CANDIDATES),
            (window._timing_edit, review_window_module.TIMING),
            (window._speaker_edit, review_window_module.SPEAKER),
            (window._risk_edit, review_window_module.RISK),
            (window._play_button, review_window_module.PLAYBACK),
            (window._text_edit, review_window_module.CORRECT_TEXT),
            (window._speaker_box, review_window_module.CORRECT_SPEAKER),
            (window._confirm_button, review_window_module.CONFIRM),
        ):
            widget.setFocus(Qt.FocusReason.TabFocusReason)
            qapp.processEvents()

            assert window._showing_note == key
            assert window._notes_text.toPlainText() == review_window_module.note_for(key).note
    finally:
        window.close()


def test_reading_a_note_does_not_change_the_note(qapp):
    """The panel takes focus so it can be read, and must hold still when it does."""
    window = open_window()
    try:
        window._confirm_button.setFocus(Qt.FocusReason.TabFocusReason)
        qapp.processEvents()

        window._notes_text.setFocus(Qt.FocusReason.TabFocusReason)
        qapp.processEvents()

        assert window._showing_note == review_window_module.CONFIRM
    finally:
        window.close()


def test_a_closed_window_stops_watching_the_focus(qapp):
    """Otherwise every review window ever opened watches for the rest of the run."""
    windows = [open_window() for _ in range(3)]
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


def test_the_status_label_reports_its_message_rather_than_a_name(qapp):
    """A label has no accessible value: naming it would hide what it says."""
    window = open_window()
    try:
        window._set_status("Showing 5 of 12 items.")

        assert accessible_name(window._status_label) == "Showing 5 of 12 items."
        assert not window._count_label.accessibleName()
    finally:
        window.close()


def test_the_focus_is_caught_when_the_item_controls_switch_off(qapp):
    """Disabling a control that has focus must not drop the user somewhere random.

    The move is said as part of the sentence about what happened, rather
    than as a second announcement arriving while the user is still
    listening to the first.
    """
    window = open_window(make_transcript([make_token("one")]))
    try:
        window._confirm_button.setFocus(Qt.FocusReason.TabFocusReason)
        assert window.focusWidget() is window._confirm_button

        window.confirm_item()

        assert window.focusWidget() is window._queue
        assert window._status_label.text() == (
            "one confirmed as correct. There is nothing to review. "
            "The focus has moved to the review queue."
        )
    finally:
        window.close()
