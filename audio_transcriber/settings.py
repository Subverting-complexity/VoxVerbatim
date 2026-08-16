"""The settings the user chooses, and how they are kept.

These are preferences: things the user decides once and expects to hold
until they decide otherwise. They are deliberately separate from the
working session, which is the state of what they happened to be doing at
the time. Deleting the session loses your place; deleting the settings
loses your preferences.

Nothing here depends on Qt, so it can be tested on its own.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from audio_transcriber.json_store import read_json_object, write_json_object

SETTINGS_FILE_NAME = "settings.json"

#: Bumped only if the shape on disk changes in a way that needs migrating.
SETTINGS_FORMAT_VERSION = 1

#: A skip of less than a second is no use, and more than an hour is beyond
#: what these buttons are for.
MINIMUM_SKIP_SECONDS = 1
MAXIMUM_SKIP_SECONDS = 3600


@dataclass
class Settings:
    """Everything the user can choose in the Settings dialog."""

    reopen_last_folder: bool = True
    """Whether the folder from the last run is opened again on start-up."""

    short_skip_seconds: int = 15
    medium_skip_seconds: int = 120
    long_skip_seconds: int = 300
    """How far the three pairs of skip buttons move."""

    @property
    def skip_seconds(self) -> tuple[int, int, int]:
        """The three skip intervals, shortest first."""
        return (self.short_skip_seconds, self.medium_skip_seconds, self.long_skip_seconds)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["version"] = SETTINGS_FORMAT_VERSION
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        """Build settings from loaded JSON, ignoring anything unusable.

        Every value is checked rather than trusted. A hand-edited file with
        nonsense in it should fall back to the default for that one setting,
        not stop the application or leave a skip button that moves zero
        seconds.
        """
        settings = cls()
        reopen = data.get("reopen_last_folder")
        if isinstance(reopen, bool):
            settings.reopen_last_folder = reopen
        settings.short_skip_seconds = _clean_skip(
            data.get("short_skip_seconds"), settings.short_skip_seconds
        )
        settings.medium_skip_seconds = _clean_skip(
            data.get("medium_skip_seconds"), settings.medium_skip_seconds
        )
        settings.long_skip_seconds = _clean_skip(
            data.get("long_skip_seconds"), settings.long_skip_seconds
        )
        return settings


def _clean_skip(value: Any, default: int) -> int:
    """Return a usable number of seconds, falling back to ``default``."""
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    if not MINIMUM_SKIP_SECONDS <= value <= MAXIMUM_SKIP_SECONDS:
        return default
    return value


class SettingsStore:
    """Reads and writes the settings file at a fixed location."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> Settings:
        """Return the saved settings, or the defaults if there are none usable."""
        data = read_json_object(self._path)
        if data is None:
            return Settings()
        return Settings.from_dict(data)

    def save(self, settings: Settings) -> bool:
        return write_json_object(self._path, settings.to_dict())
