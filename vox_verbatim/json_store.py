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
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable

_log = logging.getLogger(__name__)


class JsonReadStatus(Enum):
    """What reading a JSON file found, for a caller that must tell them apart.

    Most callers only want the object or nothing, and use
    :func:`read_json_object`. A caller that is about to write the file back
    needs more than that. A file that is not there yet is safe to create, but
    a file that is there and could not be read -- locked by another program,
    or damaged -- must not be overwritten with what this caller happens to
    hold, because what is on the disk may be newer or may be all there is.
    """

    READ = "read"
    """The file was read and held a JSON object."""

    MISSING = "missing"
    """There is no file."""

    DAMAGED = "damaged"
    """The file is there but could not be read as a JSON object."""


def read_json_object_status(path: Path) -> tuple[dict[str, Any] | None, JsonReadStatus]:
    """Return the object held in ``path`` and what reading it found.

    The object is ``None`` unless the status is :attr:`JsonReadStatus.READ`.
    A missing file is :attr:`~JsonReadStatus.MISSING`. A file that could not
    be opened (another program holding it, say), one that is not valid text,
    one that is not valid JSON and one holding something other than an object
    are all :attr:`~JsonReadStatus.DAMAGED`. Nothing here raises.
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None, JsonReadStatus.MISSING
    except OSError:
        _log.warning("Could not read %s", path, exc_info=True)
        return None, JsonReadStatus.DAMAGED

    # The bytes are decoded here rather than on the way in, so that a file
    # holding something other than UTF-8 text is treated as damaged like any
    # other unreadable file instead of raising. ``utf-8-sig`` drops a leading
    # byte order mark, which Notepad and PowerShell add when a person edits
    # the file by hand as the README invites; plain ``utf-8`` keeps the mark
    # and ``json.loads`` then refuses the file, so every setting is lost.
    # Otherwise it decodes exactly like ``utf-8``, so bad bytes still raise.
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError):
        _log.warning("%s could not be read as JSON; ignoring it.", path)
        return None, JsonReadStatus.DAMAGED

    if not isinstance(data, dict):
        _log.warning("%s does not contain an object; ignoring it.", path)
        return None, JsonReadStatus.DAMAGED
    return data, JsonReadStatus.READ


def read_json_object(path: Path) -> dict[str, Any] | None:
    """Return the object held in ``path``, or ``None`` if there is nothing usable.

    A missing file, an unreadable one, one that is not valid JSON, one that
    is not even valid text, and one holding something other than an object
    all come back as ``None``. The caller falls back to its defaults, which
    is always better than refusing to start. :func:`read_json_object_status`
    says which of those it was.
    """
    data, _status = read_json_object_status(path)
    return data


#: How a move into place is retried when the destination is held open.
#: Five tries, waiting a little longer each time, so that a file a
#: synchronisation client or a virus scanner has open for a moment is given
#: a few seconds to be let go of before the save is given up on.
REPLACE_ATTEMPTS = 5
REPLACE_FIRST_WAIT_SECONDS = 0.5
REPLACE_LONGEST_WAIT_SECONDS = 2.0

#: What is added to a file's name when the finished data could not be moved
#: over it, so that the data is kept beside the file rather than thrown away.
UNSAVED_SUFFIX = ".unsaved"


def unsaved_copy_path(path: Path) -> Path:
    """Where a save that could not be moved into place leaves its data."""
    return path.with_name(path.name + UNSAVED_SUFFIX)


def write_json_object(
    path: Path,
    data: dict[str, Any],
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Write ``data`` to ``path``, returning whether it worked.

    The file is written under a temporary name and then moved into place.
    An interrupted save therefore leaves the previous file intact rather
    than a truncated one that cannot be parsed.

    The move is the part that fails in practice, and it fails on Windows in
    a way that has nothing to do with the data. A file that something else
    has open -- OneDrive uploading the previous version, a virus scanner
    reading it, an editor somebody left it in -- cannot be replaced while it
    is held, and the attempt raises PermissionError. Most of those holds last
    a moment, so the move is tried again a few times with a short wait
    between. If it still cannot be done the finished data is not deleted: it
    is left beside the target under the same name with ``.unsaved`` on the
    end, and the log says where, because a transcript that took an hour of
    paid services to make must not be thrown away over a file lock.

    ``sleep`` is how the waits are skipped in a test.
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
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
        try:
            moved = replace_with_retries(temp_name, path, sleep)
        except BaseException:
            # Not a held file but something wrong with the place itself, so
            # there is nothing to keep the data for.
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
        if not moved:
            keep_unsaved_copy(temp_name, path)
            return False
    except OSError:
        _log.warning("Could not save %s", path, exc_info=True)
        return False
    return True


def replace_with_retries(temp_name: str, path: Path, sleep: Callable[[float], None]) -> bool:
    """Move the finished file over ``path``, trying again while it is held open.

    Only a PermissionError is retried. It is the error Windows raises for a
    file another program has open, and it is the one that goes away by
    itself. Anything else -- a folder that has gone, a name that is too long
    -- is as wrong on the fifth try as on the first, and is raised for the
    caller to report.
    """
    wait = REPLACE_FIRST_WAIT_SECONDS
    for attempt in range(1, REPLACE_ATTEMPTS + 1):
        try:
            os.replace(temp_name, path)
            return True
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS:
                _log.warning(
                    "Could not move the new %s into place after %d attempts; something "
                    "else is holding the file open.",
                    path,
                    attempt,
                    exc_info=True,
                )
                return False
            _log.info(
                "%s is held open by another program; trying again in %.1f seconds.",
                path,
                wait,
            )
            sleep(wait)
            wait = min(REPLACE_LONGEST_WAIT_SECONDS, wait * 2)
    return False


def keep_unsaved_copy(temp_name: str, path: Path) -> None:
    """Leave the data that could not be moved into place where it can be found.

    The temporary name means nothing to anybody, so the file is renamed to
    the target's name with ``.unsaved`` on the end. A copy already there
    from an earlier failure is replaced, because the newer data is the one
    worth keeping. If even that rename fails the temporary file is left as it
    is rather than deleted, and the log says so.
    """
    kept = unsaved_copy_path(path)
    try:
        os.replace(temp_name, kept)
    except OSError:
        _log.warning(
            "The unsaved data for %s was left at %s because it could not be renamed.",
            path,
            temp_name,
            exc_info=True,
        )
        return
    _log.warning("The unsaved data for %s was kept at %s.", path, kept)
