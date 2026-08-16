"""Shared test setup.

The tests run without a visible desktop, so Qt is told to use its offscreen
platform before any widget is created.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402  (the environment must be set first)
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """One application object for the whole test run, as Qt requires."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def wait_until(app: QApplication, predicate, timeout_seconds: float = 10.0) -> bool:
    """Run the Qt event loop until ``predicate`` is true or the time runs out.

    Background work reports itself through queued signals, which only arrive
    while the event loop is running, so a test that just sleeps would wait
    for ever.
    """
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        app.processEvents()
        time.sleep(0.01)
    return predicate()


def write_fake_audio(path, size_bytes: int = 2048) -> None:
    """Create a file that looks like audio to the folder scanner.

    The contents are not real audio, so the duration cannot be read. That is
    deliberate: it exercises the path where a duration is unavailable.
    """
    path.write_bytes(b"\0" * size_bytes)
