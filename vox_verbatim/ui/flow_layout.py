"""A layout that puts widgets in a row and wraps onto the next line when it runs out of room.

Qt has no such layout of its own. Without one, the eight transport buttons
have to sit in a single row, and that row sets a floor on how narrow the
window can ever be. At the text sizes a low-vision user runs, that floor
grows past the width of the screen and part of the window ends up somewhere
the user cannot reach.

The widgets themselves are ordinary buttons in ordinary layout items, so
the keyboard order and everything a screen reader sees are unchanged.
"""

from __future__ import annotations

from PySide6.QtCore import QMargins, QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QLayoutItem, QSizePolicy, QWidget


class FlowLayout(QLayout):
    """Lays widgets out left to right, starting a new line when needed."""

    def __init__(self, parent: QWidget | None = None, spacing: int = 6) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setSpacing(spacing)

    def addItem(self, item: QLayoutItem) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> QLayoutItem | None:
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> Qt.Orientation:
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._arrange(QRect(0, 0, width, 0), apply_geometry=False)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._arrange(rect, apply_geometry=True)

    def sizeHint(self) -> QSize:
        """The size of a single row, so one row is used whenever it fits."""
        margins = self.contentsMargins()
        width = margins.left() + margins.right()
        height = 0
        for index, item in enumerate(self._items):
            hint = item.sizeHint()
            width += hint.width() + (self.spacing() if index else 0)
            height = max(height, hint.height())
        return QSize(width, height + margins.top() + margins.bottom())

    def minimumSize(self) -> QSize:
        """Room for the widest single widget, so everything else can wrap."""
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(
            margins.left() + margins.right(), margins.top() + margins.bottom()
        )

    def _arrange(self, rect: QRect, apply_geometry: bool) -> int:
        """Place the widgets, and return the height the result needs."""
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x = area.x()
        y = area.y()
        line_height = 0

        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self.spacing()
            if line_height and next_x - self.spacing() > area.right() + 1:
                x = area.x()
                y = y + line_height + self.spacing()
                next_x = x + hint.width() + self.spacing()
                line_height = 0
            if apply_geometry:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())

        return y + line_height - rect.y() + margins.bottom()


class FlowWidget(QWidget):
    """A widget whose children flow onto more than one line when space is short."""

    def __init__(self, parent: QWidget | None = None, spacing: int = 6) -> None:
        super().__init__(parent)
        self._flow = FlowLayout(self, spacing=spacing)
        policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        # Without this the widget is given a fixed height worked out from a
        # single row, and the wrapped rows are cut off.
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def add(self, widget: QWidget) -> None:
        self._flow.addWidget(widget)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._flow.heightForWidth(width)
