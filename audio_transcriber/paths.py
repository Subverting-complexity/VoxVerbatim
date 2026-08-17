"""Where the application keeps its own files.

Everything the application writes for itself lives in one per-user folder:
the settings, the working session, and the log. Windows puts that folder
under the user's local application data, in a folder named after the
organisation and then the application.

Both the start-up code and the window need these paths, so they live here
rather than in either of them.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QStandardPaths

from audio_transcriber import APPLICATION_NAME
from audio_transcriber.session import SESSION_FILE_NAME
from audio_transcriber.settings import SETTINGS_FILE_NAME
from audio_transcriber.transcription.calibration import CALIBRATION_FILE_NAME
from audio_transcriber.transcription.vocabulary import VOCABULARY_FILE_NAME

LOG_FILE_NAME = "audio-transcriber.log"


def config_directory(create: bool = False) -> Path:
    """Return the folder holding the settings, the session and the log.

    Qt is asked where per-user configuration belongs rather than the path
    being written down here, so the application follows whatever the
    platform expects. Pass ``create`` when the folder is about to be written
    to; simply asking where it is should not bring it into being.
    """
    location = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)
    if not location:
        location = str(Path.home() / f".{APPLICATION_NAME.lower().replace(' ', '-')}")
    directory = Path(location)
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def log_file_path() -> Path:
    return config_directory() / LOG_FILE_NAME


def session_file_path() -> Path:
    return config_directory() / SESSION_FILE_NAME


def settings_file_path() -> Path:
    return config_directory() / SETTINGS_FILE_NAME


def vocabulary_file_path() -> Path:
    """Where the words the user has taught the application are kept.

    It sits beside the settings rather than beside the recordings, because
    it is knowledge about the person's own vocabulary rather than about any
    one recording. The same client's surname is worth knowing in every
    folder they will ever open.
    """
    return config_directory() / VOCABULARY_FILE_NAME


def calibration_file_path() -> Path:
    """Where the record of how each service has performed is kept.

    Also per user rather than per folder, and for the same reason. The
    statistics are built up one correction at a time over months, so
    scattering them across recording folders would mean starting again
    every time the user worked somewhere new.
    """
    return config_directory() / CALIBRATION_FILE_NAME
