"""Audio Transcriber.

An accessible Windows desktop application for reviewing and playing audio
recordings. This first phase provides the user interface, folder and file
handling, session persistence, and audio playback. Transcription services
are added in a later phase.
"""

from __future__ import annotations

__version__ = "0.1.0"

APPLICATION_NAME = "Audio Transcriber"

#: The organisation the application belongs to. Windows works out where
#: per-user files are kept from this and the application name, so changing
#: it moves the settings and the session to a different folder.
ORGANISATION_NAME = "JB Org"
