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


def write_real_audio(
    path,
    level_db: float = -30.0,
    seconds: float = 2.0,
    codec: str = "aac",
    layout: str = "stereo",
    rate: int = 48000,
    tone_hz: float = 440.0,
) -> None:
    """Write a real recording of a steady tone at a known level.

    The enhancement tests need audio that can actually be decoded and
    measured, which the empty files above cannot be. A steady tone is used
    because its loudness and its peak are both predictable, so a test can
    say what the answer should be rather than only that there was one.

    ``level_db`` is the amplitude of the tone in decibels below full scale.
    A tone at this level measures close to the same figure in LUFS, because
    the loudness weighting is flat near this frequency.
    """
    import array
    import math

    import av

    count = int(rate * seconds)
    channels = 2 if layout == "stereo" else 1
    amplitude = (10 ** (level_db / 20.0)) * 32767.0
    tone = array.array("h", (0,)) * (count * channels)
    for index in range(count):
        value = int(amplitude * math.sin(2.0 * math.pi * tone_hz * index / rate))
        for channel in range(channels):
            tone[index * channels + channel] = value

    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=rate, layout=layout)
        frame = av.AudioFrame(format="s16", layout=layout, samples=count)
        frame.planes[0].update(tone.tobytes())
        frame.sample_rate = rate
        frame.pts = 0
        queue = av.audio.fifo.AudioFifo()
        queue.write(frame)
        # Encoders want blocks of a size they choose, so the samples go
        # through a queue rather than in one lump.
        for block in queue.read_many(1024):
            for packet in stream.encode(block):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
