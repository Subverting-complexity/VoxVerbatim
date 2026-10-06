"""Formatting helpers for durations and file sizes.

Every value has two forms. The compact form is what appears on screen, and
the spoken form is what a screen reader reads out. A screen reader reading
"4.2 MB" or "1:05:03" out loud is hard to follow, so the spoken form writes
the units out in full instead.
"""

from __future__ import annotations

UNKNOWN_TEXT = "Unknown"

_SIZE_UNITS: tuple[tuple[str, str, int], ...] = (
    ("GB", "gigabytes", 1024**3),
    ("MB", "megabytes", 1024**2),
    ("KB", "kilobytes", 1024),
)


def _plural(count: int, unit: str) -> str:
    """Return ``count`` and ``unit``, adding an "s" unless the count is one."""
    return f"{count} {unit}" if count == 1 else f"{count} {unit}s"


def _whole_seconds(seconds: float | None) -> int | None:
    if seconds is None:
        return None
    return int(round(max(0.0, float(seconds))))


def format_duration(seconds: float | None) -> str:
    """Return a compact duration such as ``5:32`` or ``1:05:03``.

    The hours part is left out when the recording is shorter than an hour.
    ``None`` means the duration could not be read.
    """
    total = _whole_seconds(seconds)
    if total is None:
        return UNKNOWN_TEXT
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def spoken_duration(seconds: float | None) -> str:
    """Return a duration written out in words, such as ``5 minutes 32 seconds``."""
    total = _whole_seconds(seconds)
    if total is None:
        return UNKNOWN_TEXT
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    parts: list[str] = []
    if hours:
        parts.append(_plural(hours, "hour"))
    if minutes:
        parts.append(_plural(minutes, "minute"))
    if secs or not parts:
        parts.append(_plural(secs, "second"))
    return " ".join(parts)


def compact_interval(seconds: int) -> str:
    """Return a short label for a skip interval, such as ``15 sec`` or ``2 min``.

    This is what fits on a button. The spoken form of the same interval
    comes from :func:`spoken_duration`, which writes the units out in full.
    """
    seconds = max(0, int(seconds))
    minutes, remainder = divmod(seconds, 60)
    if not minutes:
        return f"{remainder} sec"
    if not remainder:
        return f"{minutes} min"
    return f"{minutes} min {remainder} sec"


def format_position(milliseconds: int | None) -> str:
    """Return a compact playback position for a value in milliseconds."""
    if milliseconds is None:
        return UNKNOWN_TEXT
    return format_duration(milliseconds / 1000.0)


def spoken_position(milliseconds: int | None) -> str:
    """Return a playback position written out in words."""
    if milliseconds is None:
        return UNKNOWN_TEXT
    return spoken_duration(milliseconds / 1000.0)


def _size_unit(num_bytes: int) -> tuple[str, str, int] | None:
    """Return the unit a size is shown in, or ``None`` for plain bytes.

    The unit is chosen from the value as it will be shown, rounded to one
    decimal place, not from the raw count. Otherwise 1,048,575 bytes, which
    is just under a megabyte, would round up to "1024.0 KB".
    """
    for index, unit in enumerate(_SIZE_UNITS):
        factor = unit[2]
        if num_bytes >= factor:
            if index > 0 and round(num_bytes / factor, 1) >= 1024:
                return _SIZE_UNITS[index - 1]
            return unit
    return None


def format_size(num_bytes: int | None) -> str:
    """Return a compact file size such as ``4.2 MB``.

    Sizes use the 1024-based units that Windows Explorer shows, so the
    numbers match what the user sees elsewhere on their machine.
    """
    if num_bytes is None:
        return UNKNOWN_TEXT
    num_bytes = max(0, int(num_bytes))
    unit = _size_unit(num_bytes)
    if unit is None:
        return _plural(num_bytes, "byte")
    short_unit, _spoken_unit, factor = unit
    return f"{num_bytes / factor:.1f} {short_unit}"


def spoken_size(num_bytes: int | None) -> str:
    """Return a file size written out in words, such as ``4.2 megabytes``."""
    if num_bytes is None:
        return UNKNOWN_TEXT
    num_bytes = max(0, int(num_bytes))
    unit = _size_unit(num_bytes)
    if unit is None:
        return _plural(num_bytes, "byte")
    _short_unit, spoken_unit, factor = unit
    return f"{num_bytes / factor:.1f} {spoken_unit}"
