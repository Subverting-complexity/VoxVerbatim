"""Entry point so the application can be started with ``python -m audio_transcriber``."""

from __future__ import annotations

import sys

from audio_transcriber.app import main

if __name__ == "__main__":
    sys.exit(main())
