"""Entry point so the application can be started with ``python -m vox_verbatim``."""

from __future__ import annotations

import sys

from vox_verbatim.app import main

if __name__ == "__main__":
    sys.exit(main())
