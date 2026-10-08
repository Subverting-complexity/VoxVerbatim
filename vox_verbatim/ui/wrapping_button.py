"""A push button whose text wraps onto more lines instead of widening it.

A ``QPushButton`` never wraps. Its smallest width is the width of its whole
text on one line, and a layout holding it can never be narrower than that.
The review window's choice buttons carry text such as "Keep Smith, said by
ElevenLabs, Deepgram and AssemblyAI", and one such button was enough to push
the panel holding it wider than the screen.

This button keeps the full text and shows it broken into lines that fit the
width it is given. It is still a standard ``QPushButton`` drawn by the style,
so it keeps its role, its focus rectangle and its keyboard behaviour. The
line breaks are only on screen: the caller names the button with the full
text through :func:`~vox_verbatim.ui.accessibility.describe`, and a screen
reader reads that name, never the broken text.
"""

from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyleOptionButton,
    QWidget,
)

# The narrowest the button asks to be, in average characters. Narrow enough
# that a choice never sets the width of the window, wide enough that a word
# of ordinary length still fits on a line of its own.
_NARROWEST_CHARACTERS = 12


class WrappingButton(QPushButton):
    """A push button whose text is broken into lines to fit its width."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full_text = ""
        # Preferred across rather than the push button's own Minimum, which
        # would make the width of the text as it stands the smallest the
        # button may be. Minimum down, so a button with more lines is given
        # the height to show them.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

    def full_text(self) -> str:
        """The text as given, before it was broken into lines."""
        return self._full_text

    def set_full_text(self, text: str) -> None:
        """Show ``text``, broken into lines that fit the button's width.

        ``text`` is the text as ``setText`` would take it, so an ampersand
        that is meant to be seen is written twice.
        """
        self._full_text = text
        self._wrap()
        self.updateGeometry()

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        narrowest = self._chrome_width() + (
            self.fontMetrics().averageCharWidth() * _NARROWEST_CHARACTERS
        )
        return QSize(min(hint.width(), narrowest), hint.height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._wrap():
            # More or fewer lines change the height the button needs.
            self.updateGeometry()

    def _chrome_width(self) -> int:
        """The width the style adds around the text: the bevel and padding."""
        option = QStyleOptionButton()
        self.initStyleOption(option)
        option.text = ""
        empty = self.style().sizeFromContents(
            QStyle.ContentsType.CT_PushButton, option, QSize(0, 0), self
        )
        return empty.width()

    def _wrap(self) -> bool:
        """Break the full text to fit the width. Answers whether it changed."""
        room = self.width() - self._chrome_width()
        wrapped = "\n".join(_lines(self._full_text, room, self.fontMetrics()))
        if wrapped == self.text():
            return False
        self.setText(wrapped)
        return True


def _lines(text: str, room: int, metrics) -> list[str]:
    """Break ``text`` at spaces into lines no wider than ``room`` pixels.

    A word wider than the room has a line of its own and is not cut. The
    button is then wider than the room, which is still far narrower than the
    whole text on one line.
    """
    # The doubled ampersand draws as one, so it is measured as one.
    def width(line: str) -> int:
        return metrics.horizontalAdvance(line.replace("&&", "&"))

    lines: list[str] = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}" if line else word
        if line and width(candidate) > room:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines
