"""The script the packaged application starts from.

The built application could have been pointed at ``audio_transcriber/__main__.py``
instead, and it would have worked. The reason for this separate file is that
PyInstaller adds the folder holding the starting script to the import path.
Had the starting script lived inside the package, that folder would have been
``audio_transcriber`` itself, and its own sub-packages would have become
importable a second time under bare names such as ``ui`` and ``audio``. Two
copies of a module in one program is a subtle and unpleasant class of bug,
because each copy gets its own module-level state. Starting from outside the
package keeps that from ever arising.
"""

from __future__ import annotations

import sys

from audio_transcriber.app import main

if __name__ == "__main__":
    sys.exit(main())
