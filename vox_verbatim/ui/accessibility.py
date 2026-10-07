"""Small helpers for talking to screen readers.

Two things are needed throughout the user interface. The first is naming a
control, so that JAWS, NVDA or ZoomText announce something meaningful when
it takes focus. The second is announcing a change that happened somewhere
other than where the focus is, such as a folder finishing loading, which a
screen reader would otherwise never mention.

Announcements carry their own text. Qt turns them into the Windows
notification that NVDA and JAWS listen for. An earlier version raised a bare
alert on the label showing the message and left the screen reader to go and
read it, which does not work: the message never travelled with the event.
"""

from __future__ import annotations

import logging

from PySide6.QtGui import QAccessible, QAccessibleAnnouncementEvent
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

    The description also becomes the tooltip, and follows it when it is
    described again, so a mouse or magnifier user is never shown an old
    sentence. A tooltip other code set on purpose is left alone: only an
    empty tooltip, or one that is still the previous description, is
    replaced.

    Never call this on a label that carries a message. A label has no value
    of its own: its accessible name *is* its text, so naming it hides
    whatever it says behind the name instead.
    """
    widget.setAccessibleName(name)
    if description is not None:
        previous = widget.accessibleDescription()
        widget.setAccessibleDescription(description)
        tooltip = widget.toolTip()
        if not tooltip or tooltip == previous:
            widget.setToolTip(description)
    return widget


def announce(widget: QWidget, message: str, urgent: bool = False) -> None:
    """Ask any running screen reader to read ``message`` out.

    An urgent message interrupts whatever is being said, which suits errors
    and things the user is waiting for. Anything else waits its turn rather
    than talking over the user.

    The message is always on screen as well, so a sighted user and a screen
    reader user are told the same thing.
    """
    if not message:
        return
    try:
        if not QAccessible.isActive():
            return
        event = QAccessibleAnnouncementEvent(widget, message)
        event.setPoliteness(
            QAccessible.AnnouncementPoliteness.Assertive
            if urgent
            else QAccessible.AnnouncementPoliteness.Polite
        )
        QAccessible.updateAccessibility(event)
    except Exception:  # accessibility must never take the application down
        _log.debug("Could not announce %r.", message, exc_info=True)
