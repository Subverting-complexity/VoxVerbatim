"""The list of words the transcription could not settle, as a Qt table.

This is the spine of the review window. Everything the reconciliation
rules could not decide on its own arrives here, and the person works down
the list until it is empty.

It is a table model and a ``QTableView`` for the same reason the file list
is: Qt already exposes a table to Windows accessibility with real rows,
columns, headers and a highlighted cell, and a hand-drawn control would
have to reproduce all of that and would get some of it wrong.

Two decisions in here are worth explaining.

The first is that the confidence of a word is a word in a column, never a
colour. A category such as "Review required" is meaningless as a shade of
orange to somebody using a magnifier with inverted colours, or to anybody
hearing the row read out.

The second is that filtering happens inside the model rather than in a
proxy on top of it. The window has to say how many items it is showing out
of how many there are, and announce that when it changes, so the count is
part of what the model reports rather than something a caller has to work
out for itself.
"""

from __future__ import annotations

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, Signal
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableView, QWidget

from vox_verbatim.formatting import format_duration, spoken_duration
from vox_verbatim.transcription.model import Confidence, FinalToken, ReviewReason

COLUMN_TIME = 0
COLUMN_TEXT = 1
COLUMN_REASON = 2
COLUMN_CONFIDENCE = 3
COLUMN_COUNT = 4

COLUMN_TITLES = ("When", "Chosen text", "Why it needs review", "Confidence")

#: What to show where a word has no text at all, which happens when the
#: services disagreed so completely that nothing was chosen.
NO_TEXT_DISPLAY = "(nothing chosen)"
NO_TEXT_SPOKEN = "nothing chosen"

#: What to show where a word is waiting for review without a reason having
#: been recorded against it.
NO_REASON_TEXT = "Waiting for review"

UNKNOWN_TIME_DISPLAY = "Unknown"
UNKNOWN_TIME_SPOKEN = "the time is not known"

#: Said in front of a time that comes from the wider containing span rather
#: than from the word's own boundaries. Without it the table would claim to
#: know where a word starts when the whole reason it is in this list may be
#: that nobody does.
APPROXIMATE_DISPLAY = "About "
APPROXIMATE_SPOKEN = "about "


def start_of(token: FinalToken) -> tuple[float | None, bool]:
    """The time to show for a word, and whether it is only approximate.

    A word with no timing of its own still sits somewhere, and the wider
    span it was found in is the only honest answer available. Saying so is
    better than leaving the column blank, because a person working down the
    queue navigates by these numbers.
    """
    if token.start is not None:
        return token.start, False
    if token.source_audio_span is not None:
        return token.source_audio_span.start, True
    return None, False


def time_display(token: FinalToken) -> str:
    """The compact time for the column, such as ``1:32`` or ``About 1:32``."""
    seconds, approximate = start_of(token)
    if seconds is None:
        return UNKNOWN_TIME_DISPLAY
    prefix = APPROXIMATE_DISPLAY if approximate else ""
    return f"{prefix}{format_duration(seconds)}"


def time_spoken(token: FinalToken) -> str:
    """The same time written out in words, for a screen reader to read."""
    seconds, approximate = start_of(token)
    if seconds is None:
        return UNKNOWN_TIME_SPOKEN
    prefix = APPROXIMATE_SPOKEN if approximate else ""
    return f"{prefix}{spoken_duration(seconds)}"


def chosen_text(token: FinalToken) -> str:
    """The word as the application currently has it, or a plain stand-in."""
    return token.text.strip() or NO_TEXT_DISPLAY


def chosen_text_spoken(token: FinalToken) -> str:
    return token.text.strip() or NO_TEXT_SPOKEN


def reason_text(token: FinalToken) -> str:
    """Why this word is in the queue, in the wording the filters use.

    The reasons come from :class:`ReviewReason` rather than being written
    out again here, so that the sentence in the table and the label on the
    filter check box can never drift apart.
    """
    if not token.review_reasons:
        return NO_REASON_TEXT
    return "; ".join(reason.display_name for reason in token.review_reasons)


def spoken_summary(token: FinalToken) -> str:
    """The whole row in one sentence, for announcing the selected item.

    A person working down the queue needs the time, the word, the reason
    and the confidence without going to look for the detail panel, so all
    four are said together when the highlight moves.
    """
    return (
        f"{time_spoken(token)}. {chosen_text_spoken(token)}. "
        f"{reason_text(token)}. {token.confidence.display_name}."
    )


def count_text(shown: int, total: int) -> str:
    """How many items the filters are letting through, in a full sentence.

    A filter that quietly empties the list looks exactly like a broken one,
    so the wording says plainly that items are being hidden rather than
    leaving an empty table to speak for itself.
    """
    if total == 0:
        return "There is nothing to review."
    if shown == 0:
        return f"No items match the filters. All {total} items are hidden."
    if shown == total:
        return f"Showing all {total} items." if total != 1 else "Showing the only item."
    return f"Showing {shown} of {total} items."


class ReviewQueueModel(QAbstractTableModel):
    """Holds the words waiting for review, and which of them get through the filters."""

    visibleCountChanged = Signal(int, int)
    """How many rows are on show, and how many there are altogether."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._all: list[FinalToken] = []
        self._rows: list[FinalToken] = []
        # Everything is on show to begin with. A review queue that opened
        # already filtered would hide work from somebody who never asked
        # for it to be hidden.
        self._reasons: set[ReviewReason] = set(ReviewReason)
        self._confidences: set[Confidence] = set(Confidence)

    # -- Contents -------------------------------------------------------

    def set_tokens(self, tokens: list[FinalToken]) -> None:
        """Take the words of a transcript, keeping the ones needing a person.

        The whole transcript can be handed over, because deciding what
        belongs in the queue is this model's job and nowhere else's. A
        corrected word stops needing review, so it leaves the queue the
        next time the transcript is handed back.
        """
        self.beginResetModel()
        self._all = [token for token in tokens if token.needs_review]
        self._rows = self._filtered()
        self.endResetModel()
        self.visibleCountChanged.emit(len(self._rows), len(self._all))

    def clear(self) -> None:
        self.set_tokens([])

    def token_at(self, row: int) -> FinalToken | None:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None

    def row_for_token_id(self, token_id: str | None) -> int:
        """Where a word sits now, or ``-1`` if the filters are hiding it.

        Correcting a word replaces it with a new object, so the window
        follows the selection by identifier rather than by holding on to
        the word itself.
        """
        if not token_id:
            return -1
        for row, token in enumerate(self._rows):
            if token.id == token_id:
                return row
        return -1

    def visible_tokens(self) -> list[FinalToken]:
        return list(self._rows)

    @property
    def total_count(self) -> int:
        """How many words need review, before the filters have their say."""
        return len(self._all)

    @property
    def visible_count(self) -> int:
        return len(self._rows)

    def count_text(self) -> str:
        return count_text(len(self._rows), len(self._all))

    # -- Filters --------------------------------------------------------

    def is_reason_shown(self, reason: ReviewReason) -> bool:
        return reason in self._reasons

    def set_reason_shown(self, reason: ReviewReason, shown: bool) -> None:
        self._set_membership(self._reasons, reason, shown)

    def is_confidence_shown(self, confidence: Confidence) -> bool:
        return confidence in self._confidences

    def set_confidence_shown(self, confidence: Confidence, shown: bool) -> None:
        self._set_membership(self._confidences, confidence, shown)

    def show_everything(self) -> None:
        """Put every filter back on, which is how the queue starts out."""
        if self._reasons == set(ReviewReason) and self._confidences == set(Confidence):
            return
        self._reasons = set(ReviewReason)
        self._confidences = set(Confidence)
        self._reapply()

    def _set_membership(self, values: set, value, shown: bool) -> None:
        if (value in values) == shown:
            return
        if shown:
            values.add(value)
        else:
            values.discard(value)
        self._reapply()

    def _reapply(self) -> None:
        self.beginResetModel()
        self._rows = self._filtered()
        self.endResetModel()
        self.visibleCountChanged.emit(len(self._rows), len(self._all))

    def _filtered(self) -> list[FinalToken]:
        return [token for token in self._all if self._matches(token)]

    def _matches(self, token: FinalToken) -> bool:
        """Whether a word gets past both filters.

        A word with no reason recorded against it is shown only while the
        reasons are left alone. Narrowing to particular reasons is asking
        for the words flagged for those reasons, and a word that carries
        none of them is not one of the answers to that question.
        """
        if token.confidence not in self._confidences:
            return False
        if token.review_reasons:
            return any(reason in self._reasons for reason in token.review_reasons)
        return self._reasons == set(ReviewReason)

    # -- QAbstractTableModel --------------------------------------------

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._rows)

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
        token = self.token_at(index.row())
        if token is None:
            return None
        column = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            return self._display_text(token, column)

        # Screen readers prefer whole words to abbreviations, so the spoken
        # form of a time is offered separately from the compact one shown.
        if role == Qt.ItemDataRole.AccessibleTextRole:
            return self._spoken_text(token, column)

        if role == Qt.ItemDataRole.ToolTipRole:
            return spoken_summary(token)

        if role == Qt.ItemDataRole.UserRole:
            return token

        return None

    @staticmethod
    def _display_text(token: FinalToken, column: int) -> str:
        if column == COLUMN_TIME:
            return time_display(token)
        if column == COLUMN_TEXT:
            return chosen_text(token)
        if column == COLUMN_REASON:
            return reason_text(token)
        if column == COLUMN_CONFIDENCE:
            return token.confidence.display_name
        return ""

    @staticmethod
    def _spoken_text(token: FinalToken, column: int) -> str:
        if column == COLUMN_TIME:
            return time_spoken(token)
        if column == COLUMN_TEXT:
            return chosen_text_spoken(token)
        return ReviewQueueModel._display_text(token, column)


class ReviewQueueView(QTableView):
    """The queue table, worked entirely from the keyboard."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setWordWrap(False)
        self.setSortingEnabled(False)
        self.verticalHeader().setVisible(False)

        # Tab moves on to the next control rather than walking across the
        # cells, so the way through the window stays predictable. The arrow
        # keys still reach every cell for anyone who wants them.
        self.setTabKeyNavigation(False)

        header = self.horizontalHeader()
        header.setSectionsClickable(False)
        header.setHighlightSections(False)

    def setModel(self, model) -> None:
        """Attach a model, then size the columns.

        Column widths can only be set once the header knows how many
        sections it has, which is only true after a model is attached.
        """
        super().setModel(model)
        if model is None:
            return
        header = self.horizontalHeader()
        # The reason is the column that varies most in length and carries
        # the most meaning, so it takes whatever room is left over.
        header.setSectionResizeMode(COLUMN_TIME, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COLUMN_TEXT, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COLUMN_REASON, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(COLUMN_CONFIDENCE, QHeaderView.ResizeMode.ResizeToContents)

    def selected_row(self) -> int:
        index = self.currentIndex()
        return index.row() if index.isValid() else -1

    def select_row(self, row: int) -> None:
        """Highlight a row and bring it into view."""
        model = self.model()
        if model is None or not 0 <= row < model.rowCount():
            return
        index = model.index(row, COLUMN_TIME)
        self.setCurrentIndex(index)
        self.scrollTo(index, QAbstractItemView.ScrollHint.EnsureVisible)
