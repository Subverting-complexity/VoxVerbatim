"""Saving and restoring the user's working session.

The session is kept in a small JSON file so that it can be read, edited or
deleted by hand if something ever goes wrong with it. A damaged or missing
file is never fatal: the application falls back to an empty session and
carries on, because losing the last folder is a far smaller problem than
refusing to start.

This module deliberately avoids Qt so it can be tested on its own. The
window geometry is stored as an opaque string, which the user interface
layer encodes and decodes.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

SESSION_FILE_NAME = "session.json"

#: Bumped only if the on-disk shape changes in a way that needs migrating.
SESSION_FORMAT_VERSION = 1


@dataclass
class SessionState:
    """Everything the application remembers between runs."""

    folder: str | None = None
    """The audio folder that was open, as an absolute path."""

    checked_files: list[str] = field(default_factory=list)
    """File names that were ticked for later transcription."""

    selected_file: str | None = None
    """The file name that was highlighted in the file list."""

    window_geometry: str | None = None
    """Base64 encoded Qt window geometry."""

    splitter_state: str | None = None
    """Base64 encoded position of the divider between the two panels."""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["version"] = SESSION_FORMAT_VERSION
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionState":
        """Build a session from loaded JSON, ignoring anything unrecognised.

        Every field is checked rather than trusted. A hand-edited file with
        the wrong type in it should degrade to the default for that field,
        not crash the application on start.
        """
        state = cls()
        folder = data.get("folder")
        if isinstance(folder, str) and folder.strip():
            state.folder = folder
        checked = data.get("checked_files")
        if isinstance(checked, list):
            state.checked_files = [item for item in checked if isinstance(item, str)]
        selected = data.get("selected_file")
        if isinstance(selected, str) and selected.strip():
            state.selected_file = selected
        geometry = data.get("window_geometry")
        if isinstance(geometry, str):
            state.window_geometry = geometry
        splitter = data.get("splitter_state")
        if isinstance(splitter, str):
            state.splitter_state = splitter
        return state


class SessionStore:
    """Reads and writes the session file at a fixed location."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> SessionState:
        """Return the saved session, or an empty one if there is nothing usable."""
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return SessionState()
        except OSError:
            _log.warning("Could not read the session file %s", self._path, exc_info=True)
            return SessionState()

        try:
            data = json.loads(raw)
        except ValueError:
            _log.warning("The session file %s is not valid JSON; ignoring it.", self._path)
            return SessionState()

        if not isinstance(data, dict):
            _log.warning("The session file %s does not contain an object; ignoring it.", self._path)
            return SessionState()

        return SessionState.from_dict(data)

    def save(self, state: SessionState) -> bool:
        """Write the session to disk, returning whether it worked.

        The file is written to a temporary name first and then moved into
        place. That way an interrupted save leaves the previous session
        intact instead of a half-written file that cannot be parsed.
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=self._path.name,
                suffix=".tmp",
                delete=False,
            )
            temp_name = handle.name
            try:
                with handle:
                    json.dump(state.to_dict(), handle, indent=2)
                os.replace(temp_name, self._path)
            except BaseException:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
                raise
        except OSError:
            _log.warning("Could not save the session to %s", self._path, exc_info=True)
            return False
        return True
