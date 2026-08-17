"""The page in Settings that shows what the application has learned so far.

The provider statistics are the one part of the application that quietly
changes what it does on the strength of the user's own corrections. That
makes showing them a matter of trust rather than of decoration: a person who
cannot see why one service is now believed over another has no way to tell
learning from drift, and no way to notice when the numbers say something
they know to be wrong.

The page is read-only and it does two things. A standard Qt table gives each
service its record, one row each, including the services that have no record
at all, because a service missing from a table reads as a service with
nothing wrong with it. Beside the table sits a paragraph that says in plain
words what the numbers mean.

The rule that shapes all of the wording is that a rate never appears without
the number of words it was worked out from. Three words out of three is a
hundred per cent, and a reader given only the percentage will believe it. So
every rate here is written as a share *and* a count, and every row also says
in words how much evidence sits behind it. Nothing is carried by colour, by
an icon or by where a row happens to sit.

The values a person reads closely sit in controls that take focus. The
summary is a read-only text box rather than a label, so it can be reached
with the Tab key, read a line at a time with a screen reader, selected,
copied, and followed by ZoomText through a real caret.
"""

from __future__ import annotations

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from audio_transcriber.transcription.calibration import (
    DEFAULT_WEIGHT,
    ProviderStatistics,
    evidence_phrase,
)
from audio_transcriber.transcription.model import Provider
from audio_transcriber.ui.accessibility import describe

COLUMN_SERVICE = 0
COLUMN_CHOSEN = 1
COLUMN_CORRECTED = 2
COLUMN_RATE = 3
COLUMN_WEIGHT = 4
COLUMN_EVIDENCE = 5
COLUMN_COUNT = 6

COLUMN_TITLES = (
    "Service",
    "Words used",
    "Words corrected",
    "Correction rate",
    "Reliability weight",
    "How much evidence",
)

NO_EVIDENCE_TEXT = "No evidence yet"


class ProviderStatisticsModel(QAbstractTableModel):
    """One row per service, whether or not it has been used yet.

    A service with no record keeps its row and says so. Leaving it out would
    make an untried service indistinguishable from a faultless one, which is
    the misreading this whole page is written to prevent.
    """

    def __init__(
        self,
        statistics: ProviderStatistics | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._statistics = statistics or ProviderStatistics()
        self._providers = list(Provider)

    def set_statistics(self, statistics: ProviderStatistics) -> None:
        self.beginResetModel()
        self._statistics = statistics
        self.endResetModel()

    def statistics(self) -> ProviderStatistics:
        return self._statistics

    # -- QAbstractTableModel --------------------------------------------

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._providers)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return COLUMN_COUNT

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ):
        if orientation != Qt.Orientation.Horizontal:
            return None
        if not 0 <= section < COLUMN_COUNT:
            return None
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.AccessibleTextRole):
            return COLUMN_TITLES[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        provider = self._provider_at(index.row())
        if provider is None:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self._display_text(provider, index.column())
        # Screen readers are given the same facts in whole sentences, so a
        # rate is never read out without the count it came from.
        if role == Qt.ItemDataRole.AccessibleTextRole:
            return self._spoken_text(provider, index.column())
        if role == Qt.ItemDataRole.ToolTipRole:
            return self._spoken_text(provider, index.column())
        if role == Qt.ItemDataRole.TextAlignmentRole and index.column() in (
            COLUMN_CHOSEN,
            COLUMN_CORRECTED,
            COLUMN_WEIGHT,
        ):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    # -- Cell text ------------------------------------------------------

    def _provider_at(self, row: int) -> Provider | None:
        if 0 <= row < len(self._providers):
            return self._providers[row]
        return None

    def _display_text(self, provider: Provider, column: int) -> str:
        counts = self._statistics.counts_for(provider)
        if column == COLUMN_SERVICE:
            return provider.display_name
        if column == COLUMN_CHOSEN:
            return f"{counts.chosen:,}"
        if column == COLUMN_CORRECTED:
            return f"{counts.corrected:,}"
        if column == COLUMN_RATE:
            if counts.chosen <= 0:
                return NO_EVIDENCE_TEXT
            return f"{_share(counts.corrected, counts.chosen)} of {counts.chosen:,} words"
        if column == COLUMN_WEIGHT:
            if counts.chosen <= 0:
                return f"{DEFAULT_WEIGHT:.2f} (starting weight)"
            return f"{counts.weight:.2f}"
        if column == COLUMN_EVIDENCE:
            return evidence_phrase(counts.chosen).capitalize()
        return ""

    def _spoken_text(self, provider: Provider, column: int) -> str:
        counts = self._statistics.counts_for(provider)
        if column == COLUMN_SERVICE:
            return provider.display_name
        if column == COLUMN_CHOSEN:
            return f"used for {counts.chosen:,} words"
        if column == COLUMN_CORRECTED:
            return f"{counts.corrected:,} words corrected"
        if column == COLUMN_RATE:
            if counts.chosen <= 0:
                return "no words yet, so there is no correction rate"
            return (
                f"{_share(counts.corrected, counts.chosen)} of the {counts.chosen:,} words "
                "it was used for needed correcting"
            )
        if column == COLUMN_WEIGHT:
            if counts.chosen <= 0:
                return f"the starting weight of {DEFAULT_WEIGHT:.2f}, nothing learned yet"
            return f"{counts.weight:.2f} out of 1"
        if column == COLUMN_EVIDENCE:
            return evidence_phrase(counts.chosen)
        return ""


class StatisticsPage(QWidget):
    """A read-only page of provider statistics, to be dropped into Settings.

    The dialog owns the tab it sits in and its title; this is only the
    contents. It takes the statistics it should show, and can be given a
    fresh set later without being rebuilt.
    """

    def __init__(
        self,
        statistics: ProviderStatistics | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._model = ProviderStatisticsModel(statistics, self)

        layout = QVBoxLayout(self)

        table_label = QLabel("&Service accuracy", self)
        self._table = QTableView(self)
        self._table.setModel(self._model)
        self._prepare_table()
        table_label.setBuddy(self._table)
        describe(
            self._table,
            "Service accuracy",
            "How often each service was used, and how often you had to correct it.",
        )

        summary_label = QLabel("&What these numbers mean", self)
        self._summary = QPlainTextEdit(self)
        self._summary.setReadOnly(True)
        # Read-only, but still selectable from the keyboard, which is what
        # gives the box a real caret for ZoomText to follow and lets a
        # screen reader walk the paragraph a line at a time.
        self._summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self._summary.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        summary_label.setBuddy(self._summary)
        describe(self._summary, "What these numbers mean")

        layout.addWidget(table_label)
        layout.addWidget(self._table, 1)
        layout.addWidget(summary_label)
        layout.addWidget(self._summary)

        self.set_statistics(statistics or ProviderStatistics())

    def _prepare_table(self) -> None:
        table = self._table
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.verticalHeader().setVisible(False)
        # Tab moves on to the summary rather than walking across the cells,
        # so the order through the page stays predictable. The arrow keys
        # still reach every cell.
        table.setTabKeyNavigation(False)
        header = table.horizontalHeader()
        header.setSectionsClickable(False)
        header.setHighlightSections(False)
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COLUMN_EVIDENCE, QHeaderView.ResizeMode.Stretch)

    def set_statistics(self, statistics: ProviderStatistics) -> None:
        """Show a set of statistics, replacing whatever was shown before."""
        self._model.set_statistics(statistics)
        self._summary.setPlainText("\n\n".join(statistics.summary_sentences()))
        self._summary.moveCursor(self._summary.textCursor().MoveOperation.Start)

    def summary_text(self) -> str:
        """The paragraph as it currently reads, which is what the tests check."""
        return self._summary.toPlainText()

    def focus_first_control(self) -> None:
        """Put the focus on the table, for a dialog opening this page."""
        self._table.setFocus(Qt.FocusReason.TabFocusReason)


def _share(part: int, whole: int) -> str:
    """A percentage, for use only where the counts are shown beside it."""
    if whole <= 0:
        return NO_EVIDENCE_TEXT
    value = 100.0 * part / whole
    if 0 < value < 1:
        return "under 1 per cent"
    return f"{value:.0f} per cent"
