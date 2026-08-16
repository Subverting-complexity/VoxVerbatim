"""Reading and writing the small JSON files the application keeps for itself.

Both the working session and the user's settings are held this way, and
both need the same care. Reading must never stop the application from
starting, whatever state the file is in, and writing must never leave a
half-written file behind where a good one used to be.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


def read_json_object(path: Path) -> dict[str, Any] | None:
    """Return the object held in ``path``, or ``None`` if there is nothing usable.

    A missing file, an unreadable one, one that is not valid JSON, one that
    is not even valid text, and one holding something other than an object
    all come back as ``None``. The caller falls back to its defaults, which
    is always better than refusing to start.
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError:
        _log.warning("Could not read %s", path, exc_info=True)
        return None

    # The bytes are decoded here rather than on the way in, so that a file
    # holding something other than UTF-8 text is treated as damaged like any
    # other unreadable file instead of raising.
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        _log.warning("%s could not be read as JSON; ignoring it.", path)
        return None

    if not isinstance(data, dict):
        _log.warning("%s does not contain an object; ignoring it.", path)
        return None
    return data


def write_json_object(path: Path, data: dict[str, Any]) -> bool:
    """Write ``data`` to ``path``, returning whether it worked.

    The file is written under a temporary name and then moved into place.
    An interrupted save therefore leaves the previous file intact rather
    than a truncated one that cannot be parsed.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name,
            suffix=".tmp",
            delete=False,
        )
        temp_name = handle.name
        try:
            with handle:
                json.dump(data, handle, indent=2)
            os.replace(temp_name, path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
    except OSError:
        _log.warning("Could not save %s", path, exc_info=True)
        return False
    return True
