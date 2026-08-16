"""Small helpers for talking to screen readers.

Two things are needed throughout the user interface. The first is naming a
control, so that JAWS, NVDA or ZoomText announce something meaningful when
it takes focus. The second is announcing a change that happened somewhere
other than where the focus is, such as a folder finishing loading, which a
screen reader would otherwise never mention.

An announcement is sent as an accessibility alert on the widget that
displays the message. The message is always visible on screen as well, so a
sighted user and a screen reader user get the same information.
"""

from __future__ import annotations

import logging

from PySide6.QtGui import QAccessible, QAccessibleEvent
from PySide6.QtWidgets import QWidget

_log = logging.getLogger(__name__)


def describe(
    widget: QWidget,
    name: str,
    description: str | None = None,
) -> QWidget:
    """Give a widget its accessible name, and optionally a longer description.

    The name is the short label a screen reader reads on focus. The
    description is the extra sentence that explains how to work the control,
    which most screen readers read after the name.
    """
    widget.setAccessibleName(name)
    if description is not None:
        widget.setAccessibleDescription(description)
        if not widget.toolTip():
            widget.setToolTip(description)
    return widget


def announce(widget: QWidget, message: str) -> None:
    """Ask any running screen reader to read ``message`` out now.

    The widget passed in should be the one showing the message, because
    that is where the screen reader looks for the text of the alert.
    """
    if not message:
        return
    try:
        if not QAccessible.isActive():
            return
        QAccessible.updateAccessibility(QAccessibleEvent(widget, QAccessible.Event.Alert))
    except Exception:  # accessibility must never take the application down
        _log.debug("Could not send an accessibility alert.", exc_info=True)
