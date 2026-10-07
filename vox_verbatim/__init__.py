"""VoxVerbatim.

An accessible Windows desktop application that turns folders of audio
recordings into transcripts. It plays and enhances recordings, transcribes
them with several speech-to-text services at once, combines the answers,
and offers a review of the words the services were unsure of.
"""

from __future__ import annotations

__version__ = "0.1.0"

APPLICATION_NAME = "VoxVerbatim"

#: The organisation the application belongs to. Windows works out where
#: per-user files are kept from this and the application name, so changing
#: it moves the settings and the session to a different folder.
ORGANISATION_NAME = "JB Org"

#: The same name again, in the form used for file names. Lower case and
#: hyphenated, because that is what a file called by hand at a command
#: prompt, or typed into a search box, is least awkward to write.
#:
#: It is written out rather than being worked out from the application name,
#: so that a name with a space, an apostrophe or a capital in the middle of a
#: word cannot quietly produce a file name nobody expected. Every file the
#: application names after itself is built from this one string, so the log,
#: the folder state file and the fallback settings folder cannot drift apart.
DISTRIBUTION_NAME = "vox-verbatim"
