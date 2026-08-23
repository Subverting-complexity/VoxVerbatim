"""Finding audio files in a folder and reading their basic metadata.

Nothing in this module depends on Qt, so it can be tested and reused
without starting a user interface.

Reading a file size is fast, but reading a duration means opening the file
and parsing its header. The two jobs are therefore kept apart: the caller
lists the files first and shows them straight away, then fills the
durations in afterwards, usually from a background thread.
"""

from __future__ import annotations

import enum
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

_log = logging.getLogger(__name__)

#: Extensions the application will list. The specification requires ``.m4a``;
#: the others are handled by the same metadata reader and the same player, so
#: they cost nothing to support.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".m4a", ".mp3", ".wav", ".flac"})


class DurationState(enum.Enum):
    """How far along we are in working out how long a recording is."""

    PENDING = "pending"
    """The duration has not been read yet."""

    KNOWN = "known"
    """The duration was read successfully."""

    UNAVAILABLE = "unavailable"
    """The duration could not be read, for example because the file is damaged."""


@dataclass
class AudioFile:
    """One audio file in the selected folder."""

    path: Path
    size_bytes: int
    duration_seconds: float | None = None
    duration_state: DurationState = field(default=DurationState.PENDING)

    @property
    def name(self) -> str:
        """The file name, without the folder that contains it."""
        return self.path.name

    @property
    def key(self) -> str:
        """The identifier used to remember this file between sessions.

        The file name is used rather than the full path, so that a whole
        folder can be moved or renamed without losing the checked state of
        the files inside it.
        """
        return self.path.name

    def set_duration(self, seconds: float | None) -> None:
        """Record the duration that was read, or that it could not be read."""
        if seconds is None:
            self.duration_seconds = None
            self.duration_state = DurationState.UNAVAILABLE
        else:
            self.duration_seconds = float(seconds)
            self.duration_state = DurationState.KNOWN


def is_supported_audio_file(path: Path | str) -> bool:
    """Return whether the file name ends in an extension the application lists."""
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def list_audio_files(folder: Path | str) -> list[AudioFile]:
    """Return the supported audio files directly inside ``folder``.

    Sub-folders are not searched. The result is sorted by file name,
    ignoring case, so the order does not change between runs. Files that
    vanish between being listed and being measured are skipped rather than
    raising, because another program can always delete a file underneath us.

    Raises:
        OSError: if the folder itself cannot be read.
    """
    folder_path = Path(folder)
    files: list[AudioFile] = []
    with os.scandir(folder_path) as entries:
        for entry in entries:
            if not is_supported_audio_file(entry.name):
                continue
            try:
                if not entry.is_file():
                    continue
                size = entry.stat().st_size
            except OSError:
                _log.debug("Skipping unreadable entry %s", entry.name, exc_info=True)
                continue
            files.append(AudioFile(path=folder_path / entry.name, size_bytes=size))
    files.sort(key=lambda item: item.name.casefold())
    return files


def read_duration(path: Path | str) -> float | None:
    """Return the length of an audio file in seconds, or ``None`` if unknown.

    The duration comes from the file's own header via the ``mutagen``
    library. If ``mutagen`` is not installed, or the file is damaged or not
    really audio, ``None`` comes back and the caller shows the duration as
    unknown rather than failing.
    """
    try:
        from mutagen import File as MutagenFile  # imported lazily; it is optional
    except ImportError:
        _log.warning(
            "The mutagen library is not installed, so durations cannot be read "
            "from file headers."
        )
        return None

    try:
        media = MutagenFile(os.fspath(path))
    except Exception:  # mutagen raises a wide range of parsing errors
        _log.debug("Could not read metadata from %s", path, exc_info=True)
        return None

    info = getattr(media, "info", None)
    length = getattr(info, "length", None)
    if length is None:
        return None
    try:
        length = float(length)
    except (TypeError, ValueError):
        return None
    if length <= 0:
        return None
    return length
