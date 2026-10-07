"""Tests for the review queue: what it holds, what it says, and what it hides.

Every transcript here is built by hand. The queue is deliberately ignorant
of providers, reconciliation and storage, so none of these tests needs a
network, an API key or an audio file.
"""

from __future__ import annotations

from PySide6.QtCore import Qt

from vox_verbatim.transcription.model import (
    AudioSpan,
    Confidence,
    FinalToken,
    SPEAKER_ONLY_REASONS,
    ReviewReason,
    ReviewStatus,
)
from vox_verbatim.ui.review_queue import (
    COLUMN_CONFIDENCE,
    COLUMN_REASON,
    COLUMN_TEXT,
    COLUMN_TIME,
    NO_REASON_TEXT,
    NO_TEXT_DISPLAY,
    ReviewQueueModel,
    ReviewQueueView,
    count_text,
    spoken_summary,
)


#: The reasons that put a word in the word list. A reason about the speaker
#: alone does not, because the text of such a word is not in doubt.
WORD_REASONS = [reason for reason in ReviewReason if reason not in SPEAKER_ONLY_REASONS]


def make_token(
    text: str = "contract",
    start: float | None = 1.0,
    end: float | None = 1.5,
    reasons: tuple[ReviewReason, ...] = (ReviewReason.PROVIDER_DISAGREEMENT,),
    confidence: Confidence = Confidence.REVIEW_REQUIRED,
    span: AudioSpan | None = None,
) -> FinalToken:
    """One word needing review, with everything else left at its strongest.

    The text confidence carries the category under test and the other two
    are high, so that the weakest of the three is the one the test asked
    for rather than an accident of the defaults.
    """
    token = FinalToken(
        text=text,
        start=start,
        end=end,
        text_confidence=confidence,
        timing_confidence=Confidence.HIGH,
        speaker_confidence=Confidence.HIGH,
        source_audio_span=span,
    )
    for reason in reasons:
        token.flag(reason)
    return token


def settled_token(text: str = "the") -> FinalToken:
    return FinalToken(
        text=text,
        start=0.1,
        end=0.2,
        text_confidence=Confidence.HIGH,
        timing_confidence=Confidence.HIGH,
        speaker_confidence=Confidence.HIGH,
        review_status=ReviewStatus.SETTLED,
    )


def test_the_queue_has_the_four_required_columns_in_order(qapp):
    model = ReviewQueueModel()
    model.set_tokens([make_token()])

    assert model.columnCount() == 4
    headers = [
        model.headerData(column, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
        for column in range(4)
    ]
    assert headers == ["When", "Chosen text", "Why it needs review", "Confidence"]


def test_only_the_words_needing_a_person_reach_the_queue(qapp):
    """A transcript is mostly settled words, and none of them belong here."""
    model = ReviewQueueModel()

    model.set_tokens([settled_token("the"), make_token("contract"), settled_token("was")])

    assert model.rowCount() == 1
    assert model.total_count == 1
    assert model.token_at(0).text == "contract"


def test_a_word_pending_review_counts_even_with_no_reason_recorded(qapp):
    model = ReviewQueueModel()
    waiting = make_token(reasons=())
    waiting.review_status = ReviewStatus.PENDING

    model.set_tokens([waiting])

    assert model.rowCount() == 1
    assert model.data(model.index(0, COLUMN_REASON)) == NO_REASON_TEXT


def test_the_time_is_compact_on_screen_and_written_out_for_a_screen_reader(qapp):
    model = ReviewQueueModel()
    model.set_tokens([make_token(start=92.0, end=92.4)])
    index = model.index(0, COLUMN_TIME)

    assert model.data(index, Qt.ItemDataRole.DisplayRole) == "1:32"
    assert model.data(index, Qt.ItemDataRole.AccessibleTextRole) == "1 minute 32 seconds"


def test_a_word_with_no_boundaries_is_placed_by_its_containing_span_and_says_so(qapp):
    """Claiming a start it does not have would be a lie the rest would believe."""
    model = ReviewQueueModel()
    model.set_tokens(
        [make_token(start=None, end=None, span=AudioSpan(92.0, 95.0))]
    )
    index = model.index(0, COLUMN_TIME)

    assert model.data(index, Qt.ItemDataRole.DisplayRole) == "About 1:32"
    assert model.data(index, Qt.ItemDataRole.AccessibleTextRole) == "about 1 minute 32 seconds"


def test_a_word_with_no_audio_at_all_says_the_time_is_not_known(qapp):
    model = ReviewQueueModel()
    model.set_tokens([make_token(start=None, end=None, span=None)])
    index = model.index(0, COLUMN_TIME)

    assert model.data(index, Qt.ItemDataRole.DisplayRole) == "Unknown"
    assert model.data(index, Qt.ItemDataRole.AccessibleTextRole) == "the time is not known"


def test_a_word_the_services_could_not_settle_reads_as_nothing_chosen(qapp):
    model = ReviewQueueModel()
    model.set_tokens([make_token(text="", confidence=Confidence.UNRESOLVED)])

    assert model.data(model.index(0, COLUMN_TEXT)) == NO_TEXT_DISPLAY
    assert model.data(model.index(0, COLUMN_TEXT), Qt.ItemDataRole.AccessibleTextRole) == (
        "nothing chosen"
    )


def test_the_reason_column_uses_the_same_wording_as_the_filters(qapp):
    """Two wordings for one reason would leave the filter looking unrelated."""
    model = ReviewQueueModel()
    model.set_tokens(
        [
            make_token(
                reasons=(ReviewReason.HIGH_RISK_ENTITY, ReviewReason.NUMERIC_DISAGREEMENT)
            )
        ]
    )

    assert model.data(model.index(0, COLUMN_REASON)) == (
        f"{ReviewReason.HIGH_RISK_ENTITY.display_name}; "
        f"{ReviewReason.NUMERIC_DISAGREEMENT.display_name}"
    )


def test_the_confidence_is_a_word_in_a_column_and_never_a_colour(qapp):
    """A shade of orange is nothing at all to a magnifier or a screen reader."""
    model = ReviewQueueModel()
    model.set_tokens([make_token(confidence=Confidence.REVIEW_SUGGESTED)])
    index = model.index(0, COLUMN_CONFIDENCE)

    assert model.data(index, Qt.ItemDataRole.DisplayRole) == "Review suggested"
    assert model.data(index, Qt.ItemDataRole.AccessibleTextRole) == "Review suggested"
    assert model.data(index, Qt.ItemDataRole.BackgroundRole) is None
    assert model.data(index, Qt.ItemDataRole.ForegroundRole) is None
    assert model.data(index, Qt.ItemDataRole.DecorationRole) is None


def test_the_confidence_is_the_weakest_of_the_three_answers(qapp):
    """Text we are sure of, at a time we are not, is not a settled word."""
    model = ReviewQueueModel()
    token = make_token()
    token.timing_confidence = Confidence.REVIEW_REQUIRED
    model.set_tokens([token])

    assert model.data(model.index(0, COLUMN_CONFIDENCE)) == "Review required"


# -- The filters ---------------------------------------------------------


def test_clearing_any_one_reason_hides_exactly_the_words_flagged_for_it(qapp):
    """Every reason in the model is a filter, so every one of them is tried."""
    model = ReviewQueueModel()
    model.set_tokens([make_token(text=reason.value, reasons=(reason,)) for reason in WORD_REASONS])
    total = len(WORD_REASONS)
    assert model.rowCount() == total

    for reason in WORD_REASONS:
        model.set_reason_shown(reason, False)

        assert model.rowCount() == total - 1
        assert reason.value not in [token.text for token in model.visible_tokens()]

        model.set_reason_shown(reason, True)
        assert model.rowCount() == total


def test_a_word_flagged_for_two_reasons_survives_while_either_is_shown(qapp):
    model = ReviewQueueModel()
    model.set_tokens(
        [
            make_token(
                reasons=(ReviewReason.HIGH_RISK_ENTITY, ReviewReason.NUMERIC_DISAGREEMENT)
            )
        ]
    )

    model.set_reason_shown(ReviewReason.HIGH_RISK_ENTITY, False)
    assert model.rowCount() == 1

    model.set_reason_shown(ReviewReason.NUMERIC_DISAGREEMENT, False)
    assert model.rowCount() == 0


def test_clearing_any_one_confidence_category_hides_the_words_rated_that_way(qapp):
    model = ReviewQueueModel()
    model.set_tokens(
        [make_token(text=value.value, confidence=value) for value in Confidence]
    )
    total = len(list(Confidence))
    assert model.rowCount() == total

    for confidence in Confidence:
        model.set_confidence_shown(confidence, False)

        assert model.rowCount() == total - 1
        assert confidence.value not in [token.text for token in model.visible_tokens()]

        model.set_confidence_shown(confidence, True)
        assert model.rowCount() == total


def test_the_two_filters_narrow_together_rather_than_separately(qapp):
    model = ReviewQueueModel()
    model.set_tokens(
        [
            make_token(text="one", reasons=(ReviewReason.HIGH_RISK_ENTITY,)),
            make_token(
                text="two",
                reasons=(ReviewReason.HIGH_RISK_ENTITY,),
                confidence=Confidence.UNRESOLVED,
            ),
        ]
    )

    model.set_confidence_shown(Confidence.UNRESOLVED, False)

    assert [token.text for token in model.visible_tokens()] == ["one"]


def test_a_word_with_no_reason_recorded_is_hidden_once_the_reasons_are_narrowed(qapp):
    """Asking for particular reasons is not asking for words that carry none."""
    model = ReviewQueueModel()
    waiting = make_token(reasons=())
    waiting.review_status = ReviewStatus.PENDING
    model.set_tokens([waiting])
    assert model.rowCount() == 1

    model.set_reason_shown(ReviewReason.SPEAKER_UNCERTAIN, False)

    assert model.rowCount() == 0


def test_showing_everything_puts_every_filter_back_on(qapp):
    model = ReviewQueueModel()
    model.set_tokens([make_token(text=reason.value, reasons=(reason,)) for reason in WORD_REASONS])
    model.set_reason_shown(ReviewReason.PROVIDER_DISAGREEMENT, False)
    model.set_confidence_shown(Confidence.REVIEW_REQUIRED, False)
    assert model.rowCount() == 0

    model.show_everything()

    assert model.rowCount() == len(WORD_REASONS)


def test_a_word_in_doubt_only_for_its_speaker_is_not_in_the_word_list(qapp):
    """Its text is agreed, so it is shown once per stretch elsewhere instead."""
    model = ReviewQueueModel()
    speaker_only = make_token(
        text="yes", reasons=(ReviewReason.SPEAKER_UNCERTAIN,), confidence=Confidence.HIGH
    )
    both = make_token(
        text="fifteen",
        reasons=(ReviewReason.PROVIDER_DISAGREEMENT, ReviewReason.SPEAKER_UNCERTAIN),
    )

    model.set_tokens([speaker_only, both])

    assert [token.text for token in model.visible_tokens()] == ["fifteen"]


# -- The count -----------------------------------------------------------


def test_the_count_is_reported_whenever_the_contents_or_the_filters_change(qapp):
    reported: list[tuple[int, int]] = []
    model = ReviewQueueModel()
    model.visibleCountChanged.connect(lambda shown, total: reported.append((shown, total)))

    model.set_tokens([make_token(text="one"), make_token(text="two")])
    model.set_reason_shown(ReviewReason.PROVIDER_DISAGREEMENT, False)
    model.show_everything()

    assert reported == [(2, 2), (0, 2), (2, 2)]


def test_the_count_says_plainly_that_items_are_being_hidden(qapp):
    """An empty table on its own is indistinguishable from a broken queue."""
    assert count_text(0, 0) == "There is nothing to review."
    assert count_text(0, 12) == "No items match the filters. All 12 items are hidden."
    assert count_text(5, 12) == "Showing 5 of 12 items."
    assert count_text(12, 12) == "Showing all 12 items."
    assert count_text(1, 1) == "Showing the only item."


def test_the_model_reports_its_own_count_in_the_same_words(qapp):
    model = ReviewQueueModel()
    model.set_tokens([make_token(text="one"), make_token(text="two")])

    model.set_confidence_shown(Confidence.REVIEW_REQUIRED, False)

    assert model.count_text() == "No items match the filters. All 2 items are hidden."


# -- Following a word through a correction -------------------------------


def test_a_word_is_found_again_by_its_identifier_rather_than_by_position(qapp):
    """A correction replaces the word, so holding on to the object would fail."""
    model = ReviewQueueModel()
    first = make_token(text="one")
    second = make_token(text="two")
    model.set_tokens([first, second])

    assert model.row_for_token_id(second.id) == 1
    assert model.row_for_token_id("not a word in this transcript") == -1
    assert model.row_for_token_id(None) == -1


def test_a_hidden_word_reports_no_row_at_all(qapp):
    model = ReviewQueueModel()
    token = make_token()
    model.set_tokens([token])

    model.set_confidence_shown(Confidence.REVIEW_REQUIRED, False)

    assert model.row_for_token_id(token.id) == -1


# -- What gets read out --------------------------------------------------


def test_the_whole_row_is_said_in_one_sentence_when_the_highlight_moves(qapp):
    """Otherwise the detail panel has to be visited for every single item."""
    token = make_token(text="15,000", start=92.0, reasons=(ReviewReason.HIGH_RISK_ENTITY,))

    assert spoken_summary(token) == (
        "1 minute 32 seconds. 15,000. A value that must not be guessed. Review required."
    )


# -- The view ------------------------------------------------------------


def test_tab_moves_out_of_the_queue_rather_than_across_its_cells(qapp):
    view = ReviewQueueView()

    assert not view.tabKeyNavigation()


def test_selecting_a_row_reports_where_the_highlight_now_is(qapp):
    model = ReviewQueueModel()
    model.set_tokens([make_token(text="one"), make_token(text="two")])
    view = ReviewQueueView()
    view.setModel(model)

    view.select_row(1)

    assert view.selected_row() == 1
    assert model.token_at(view.selected_row()).text == "two"


def test_selecting_a_row_that_is_not_there_leaves_the_highlight_alone(qapp):
    model = ReviewQueueModel()
    model.set_tokens([make_token()])
    view = ReviewQueueView()
    view.setModel(model)
    view.select_row(0)

    view.select_row(7)

    assert view.selected_row() == 0
