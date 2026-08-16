"""The settings the user chooses, and how they are kept.

These are preferences: things the user decides once and expects to hold
until they decide otherwise. They are deliberately separate from the
working session, which is the state of what they happened to be doing at
the time. Deleting the session loses your place; deleting the settings
loses your preferences.

Nothing here depends on Qt, so it can be tested on its own.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from audio_transcriber.audio.enhance import (
    DEFAULT_CEILING_DBTP,
    DEFAULT_MAXIMUM_GAIN_DB,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_TARGET_LUFS,
    MAXIMUM_CEILING_DBTP,
    MAXIMUM_MAXIMUM_GAIN_DB,
    MAXIMUM_TARGET_LUFS,
    MINIMUM_CEILING_DBTP,
    MINIMUM_MAXIMUM_GAIN_DB,
    MINIMUM_TARGET_LUFS,
    output_format_for,
)
from audio_transcriber.json_store import read_json_object, write_json_object

SETTINGS_FILE_NAME = "settings.json"

#: Bumped only if the shape on disk changes in a way that needs migrating.
SETTINGS_FORMAT_VERSION = 1

#: A skip of less than a second is no use, and more than an hour is beyond
#: what these buttons are for.
MINIMUM_SKIP_SECONDS = 1
MAXIMUM_SKIP_SECONDS = 3600


@dataclass
class EnhanceSettings:
    """What the Enhance Audio dialog opens with.

    These are kept separately from the rest because they are set in their
    own dialog rather than in Settings, and because there are enough of
    them that mixing them in would bury the handful of settings that are
    about the player.
    """

    output_folder: str | None = None
    """Where enhanced copies were last written. Empty until the first run."""

    target_lufs: float = DEFAULT_TARGET_LUFS
    ceiling_dbtp: float = DEFAULT_CEILING_DBTP
    maximum_gain_db: float = DEFAULT_MAXIMUM_GAIN_DB
    use_limiter: bool = False
    output_format: str = DEFAULT_OUTPUT_FORMAT.key
    replace_existing: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnhanceSettings":
        settings = cls()
        folder = data.get("output_folder")
        if isinstance(folder, str) and folder.strip():
            settings.output_folder = folder
        settings.target_lufs = _clean_number(
            data.get("target_lufs"), settings.target_lufs, MINIMUM_TARGET_LUFS, MAXIMUM_TARGET_LUFS
        )
        settings.ceiling_dbtp = _clean_number(
            data.get("ceiling_dbtp"),
            settings.ceiling_dbtp,
            MINIMUM_CEILING_DBTP,
            MAXIMUM_CEILING_DBTP,
        )
        settings.maximum_gain_db = _clean_number(
            data.get("maximum_gain_db"),
            settings.maximum_gain_db,
            MINIMUM_MAXIMUM_GAIN_DB,
            MAXIMUM_MAXIMUM_GAIN_DB,
        )
        limiter = data.get("use_limiter")
        if isinstance(limiter, bool):
            settings.use_limiter = limiter
        # An unknown format name falls back to the default rather than
        # being kept, so a hand-edited file cannot ask for an encoder that
        # does not exist.
        settings.output_format = output_format_for(data.get("output_format")).key
        replace = data.get("replace_existing")
        if isinstance(replace, bool):
            settings.replace_existing = replace
        return settings


@dataclass
class Settings:
    """Everything the user can choose in the Settings dialog."""

    reopen_last_folder: bool = True
    """Whether the folder from the last run is opened again on start-up."""

    short_skip_seconds: int = 15
    medium_skip_seconds: int = 120
    long_skip_seconds: int = 300
    """How far the three pairs of skip buttons move."""

    enhance: EnhanceSettings = field(default_factory=EnhanceSettings)
    """What the Enhance Audio dialog opens with next time."""

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
        enhance = data.get("enhance")
        if isinstance(enhance, dict):
            settings.enhance = EnhanceSettings.from_dict(enhance)
        return settings


def _clean_skip(value: Any, default: int) -> int:
    """Return a usable number of seconds, falling back to ``default``."""
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    if not MINIMUM_SKIP_SECONDS <= value <= MAXIMUM_SKIP_SECONDS:
        return default
    return value


def _clean_number(value: Any, default: float, lowest: float, highest: float) -> float:
    """Return a usable measurement, falling back to ``default``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    if not lowest <= value <= highest:
        return default
    return float(value)


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
