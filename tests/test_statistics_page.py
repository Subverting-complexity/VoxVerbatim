"""Tests for the read-only statistics page that goes into the Settings dialog."""

from __future__ import annotations

from PySide6.QtCore import Qt

from audio_transcriber.transcription.calibration import (
    DEFAULT_WEIGHT,
    Observation,
    ProviderStatistics,
)
from audio_transcriber.transcription.model import Confidence, Language, Provider
from audio_transcriber.ui.statistics_page import (
    COLUMN_CHOSEN,
    COLUMN_COUNT,
    COLUMN_EVIDENCE,
    COLUMN_RATE,
    COLUMN_SERVICE,
    COLUMN_TITLES,
    COLUMN_WEIGHT,
    NO_EVIDENCE_TEXT,
    StatisticsPage,
)


def statistics_with(chosen: int, corrected: int, provider=Provider.OPENAI) -> ProviderStatistics:
    statistics = ProviderStatistics()
    observation = Observation(provider=provider, language=Language.ENGLISH)
    for index in range(chosen):
        statistics.record_choice(observation, corrected=index < corrected)
    return statistics


def cell(page: StatisticsPage, provider: Provider, column: int, role=None) -> str:
    model = page._table.model()
    row = list(Provider).index(provider)
    index = model.index(row, column)
    return model.data(index, role or Qt.ItemDataRole.DisplayRole)


def test_the_page_lists_every_service_including_the_ones_never_used(qapp):
    page = StatisticsPage(statistics_with(chosen=40, corrected=4))

    model = page._table.model()
    assert model.rowCount() == len(list(Provider))
    assert model.columnCount() == COLUMN_COUNT
    assert cell(page, Provider.DEEPGRAM, COLUMN_SERVICE) == Provider.DEEPGRAM.display_name
    assert cell(page, Provider.DEEPGRAM, COLUMN_RATE) == NO_EVIDENCE_TEXT


def test_a_rate_is_never_shown_without_the_number_of_words_behind_it(qapp):
    """Three words out of three is not a perfect service, and must not read as one."""
    page = StatisticsPage(statistics_with(chosen=3, corrected=0))

    rate = cell(page, Provider.OPENAI, COLUMN_RATE)
    assert "3 words" in rate
    assert "very little evidence" in cell(page, Provider.OPENAI, COLUMN_EVIDENCE).casefold()


def test_the_spoken_form_of_a_rate_also_carries_its_sample_size(qapp):
    page = StatisticsPage(statistics_with(chosen=200, corrected=20))

    spoken = cell(page, Provider.OPENAI, COLUMN_RATE, Qt.ItemDataRole.AccessibleTextRole)
    assert "10 per cent" in spoken
    assert "200 words" in spoken


def test_a_service_with_no_record_shows_the_starting_weight_and_says_so(qapp):
    page = StatisticsPage(ProviderStatistics())

    weight = cell(page, Provider.MICROSOFT, COLUMN_WEIGHT)
    assert f"{DEFAULT_WEIGHT:.2f}" in weight
    assert "starting weight" in weight
    assert cell(page, Provider.MICROSOFT, COLUMN_CHOSEN) == "0"


def test_the_summary_says_in_words_what_the_numbers_mean(qapp):
    statistics = statistics_with(chosen=400, corrected=8)
    for index in range(100):
        statistics.record_review(Confidence.HIGH, corrected=index < 3)
    page = StatisticsPage(statistics)

    summary = page.summary_text()
    assert Provider.OPENAI.display_name in summary
    assert "400 words" in summary
    assert "100 words" in summary


def test_an_empty_page_says_nothing_has_been_learned_yet(qapp):
    page = StatisticsPage()

    assert "Nothing has been learned yet" in page.summary_text()


def test_the_page_can_be_given_fresh_statistics_without_being_rebuilt(qapp):
    page = StatisticsPage()

    page.set_statistics(statistics_with(chosen=60, corrected=6))

    assert cell(page, Provider.OPENAI, COLUMN_CHOSEN) == "60"
    assert "60 words" in page.summary_text()


def test_every_column_is_named_for_a_screen_reader(qapp):
    page = StatisticsPage(statistics_with(chosen=10, corrected=1))
    model = page._table.model()

    for column, title in enumerate(COLUMN_TITLES):
        for role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.AccessibleTextRole):
            assert model.headerData(column, Qt.Orientation.Horizontal, role) == title


def test_the_controls_are_named_and_can_be_reached_from_the_keyboard(qapp):
    page = StatisticsPage(statistics_with(chosen=10, corrected=1))

    assert page._table.accessibleName() == "Service accuracy"
    assert page._summary.accessibleName() == "What these numbers mean"
    for control in (page._table, page._summary):
        assert control.focusPolicy() != Qt.FocusPolicy.NoFocus


def test_the_summary_is_read_only_but_still_takes_focus_and_shows_a_caret(qapp):
    page = StatisticsPage(statistics_with(chosen=10, corrected=1))

    assert page._summary.isReadOnly() is True
    flags = page._summary.textInteractionFlags()
    assert flags & Qt.TextInteractionFlag.TextSelectableByKeyboard


def test_the_page_can_hand_the_focus_to_its_first_control(qapp):
    page = StatisticsPage(statistics_with(chosen=10, corrected=1))
    page.show()

    page.focus_first_control()

    assert page.focusWidget() is page._table
