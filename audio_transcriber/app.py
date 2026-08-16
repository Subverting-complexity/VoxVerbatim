"""Starting the application: settings location, logging, and the main window."""

from __future__ import annotations

import logging
import logging.handlers
import sys
import traceback
from pathlib import Path
from types import TracebackType

from PySide6.QtCore import QCoreApplication, QStandardPaths
from PySide6.QtWidgets import QApplication, QMessageBox

from audio_transcriber import APPLICATION_NAME, ORGANISATION_NAME, __version__
from audio_transcriber.session import SESSION_FILE_NAME, SessionStore
from audio_transcriber.ui.main_window import MainWindow

_log = logging.getLogger(__name__)

LOG_FILE_NAME = "audio-transcriber.log"
_LOG_MAX_BYTES = 1024 * 1024
_LOG_BACKUP_COUNT = 2


def config_directory() -> Path:
    """Return the per-user folder holding the session file and the log.

    On Windows this sits under the user's AppData folder, which is where a
    desktop application is expected to keep its settings.
    """
    location = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)
    if not location:
        location = str(Path.home() / f".{APPLICATION_NAME.lower().replace(' ', '-')}")
    directory = Path(location)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def configure_logging(directory: Path) -> None:
    """Send log messages to a file next to the session file.

    The application normally runs without a console window, so a log file
    is the only place a problem report can go. Failing to open it is not
    worth stopping for.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    try:
        handler = logging.handlers.RotatingFileHandler(
            directory / LOG_FILE_NAME,
            maxBytes=_LOG_MAX_BYTES,
            backupCount=_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
    except OSError:
        return
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    )
    root.addHandler(handler)


def _install_exception_hook() -> None:
    """Report an unexpected failure instead of disappearing without a word.

    A crash with no console window and no message leaves the user with
    nothing to go on. The message box is a standard dialog, so a screen
    reader reads it out.
    """

    def hook(
        exc_type: type[BaseException],
        value: BaseException,
        tb: TracebackType | None,
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, value, tb)
            return
        details = "".join(traceback.format_exception(exc_type, value, tb))
        _log.error("Unhandled exception:\n%s", details)
        message = QMessageBox(
            QMessageBox.Icon.Critical,
            f"{APPLICATION_NAME} problem",
            f"Something went wrong: {value}",
            QMessageBox.StandardButton.Close,
        )
        message.setInformativeText(
            f"The details were written to the log file in {config_directory()}."
        )
        message.setDetailedText(details)
        message.exec()

    sys.excepthook = hook


def main(argv: list[str] | None = None) -> int:
    """Run the application and return the exit code."""
    QCoreApplication.setOrganizationName(ORGANISATION_NAME)
    QCoreApplication.setApplicationName(APPLICATION_NAME)
    QCoreApplication.setApplicationVersion(__version__)

    app = QApplication(list(sys.argv if argv is None else argv))
    app.setApplicationDisplayName(APPLICATION_NAME)

    directory = config_directory()
    configure_logging(directory)
    _install_exception_hook()
    _log.info("Starting %s version %s", APPLICATION_NAME, __version__)

    window = MainWindow(SessionStore(directory / SESSION_FILE_NAME))
    window.show()
    return app.exec()
