# Audio Transcriber

An accessible Windows desktop application for working through audio
recordings. This version gives you the player and the file handling. Later
versions will transcribe the recordings with several transcription services
at once and compare their results against each other for accuracy.

## Running it

Double-click **Audio Transcriber.cmd**.

The first run creates a private Python environment in a `.venv` folder and
downloads the libraries the application needs, which takes about a minute
and needs an internet connection. Every run after that starts straight away.

You need Python 3.11 or newer installed, from
<https://www.python.org/downloads/windows/>. Tick "Add python.exe to PATH"
during the installation.

To put the application on your desktop or taskbar, right-click
**Audio Transcriber.cmd**, choose "Show more options" and then "Send to",
and pick "Desktop (create shortcut)".

If you would rather start it yourself from a command prompt:

```bash
python -m audio_transcriber
```

## What it does

Choose a folder of recordings and the application lists every audio file in
it with its length and its size. Pick one and play it, moving backwards and
forwards through it in steps of 15 seconds, 2 minutes or 5 minutes, or by
dragging the seek bar.

Each file also has a check box. Checking files does nothing yet: it is how
you will choose which recordings to transcribe once transcription arrives.
Checking a file and playing a file are separate. The player always follows
the highlighted file, never the checked ones.

Your folder, your checked files, the file you were on and the size and
position of the window are all remembered. Close the application and open it
again and you carry on where you left off. Files that were deleted or
renamed in the meantime are quietly dropped.

`.m4a`, `.mp3`, `.wav` and `.flac` files are listed.

## Accessibility

The application is built for JAWS, NVDA and ZoomText Magnifier/Reader, and
everything in it can be done from the keyboard.

* Every control has a name a screen reader reads out. The file list offers a
  spoken form of each value as well as the compact one on screen, so "4.2 MB"
  is read as "4.2 megabytes" and "1:32" as "1 minute 32 seconds". The
  Selected file panel writes its values out in words, because that is where
  a value is read closely rather than scanned down a column.
* The file list is a standard Qt table with real rows, columns and check
  boxes, so a screen reader can navigate it as a table. The space bar checks
  or clears a file from any column of the highlighted row.
* Values you might want to read closely, such as the folder path and the
  details of the selected file, sit in read-only text boxes. They take
  focus, they can be read a word at a time or copied, and they show a real
  caret for ZoomText to follow. The caret stays where you left it even while
  the playback position rewrites itself.
* Status changes that happen away from the focus, such as a folder finishing
  loading or playback failing, are shown on screen and announced. Errors
  interrupt; everything else waits its turn.
* The skip buttons say how far they move in words, so nothing depends on
  counting arrow brackets or noticing which way they point.
* The transport buttons wrap onto a second line rather than forcing the
  window wider than the screen, so the application still fits at large
  Windows text sizes and high scaling.
* Nothing is signalled by colour, icon or position alone.

Press **F1** in the application for the full list of keyboard shortcuts.
The main ones:

| Key | What it does |
| --- | --- |
| `Ctrl+O` | Choose the audio folder |
| `F5` | Read the folder again |
| `Up` / `Down` | Move through the file list |
| `Space` | Check or clear the highlighted file |
| `Ctrl+Space` | Play, or pause if already playing |
| `Alt+Left` / `Alt+Right` | Back or forward 15 seconds |
| `Alt+Shift+Left` / `Alt+Shift+Right` | Back or forward 2 minutes |
| `Alt+Ctrl+Left` / `Alt+Ctrl+Right` | Back or forward 5 minutes |
| `F6` | Move to the next panel |

## Where your settings are kept

The session file and the log file live in
`%LOCALAPPDATA%\Audio Transcriber\Audio Transcriber`. Deleting `session.json`
resets the application to its first-run state. If something goes wrong,
`audio-transcriber.log` in the same folder is the place to look.

## How the code is arranged

```
audio_transcriber/
    app.py           Starting up: settings location, logging, the main window
    session.py       Reading and writing the saved session
    formatting.py    Durations and sizes, in compact and spoken forms
    audio/
        library.py   Finding audio files and reading their metadata
        scanner.py   Doing that on a background thread
        player.py    Playback, wrapping Qt Multimedia
    ui/
        main_window.py     Ties everything together
        folder_panel.py    Choosing the folder
        file_table.py      The file list, its model and its keyboard handling
        player_panel.py    Transport buttons and the seek bar
        file_info_panel.py Details of the selected file
        help_dialogs.py    Keyboard shortcuts and About
        accessibility.py   Naming controls and announcing changes
        flow_layout.py     A row of buttons that wraps when space is short
```

Nothing in `audio_transcriber/session.py`, `formatting.py` or
`audio/library.py` depends on the user interface, which is what will let the
transcription services be added underneath the same window without
disturbing it.

## Working on it

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
```

The tests run without a visible desktop by asking Qt for its offscreen
platform, so they can run anywhere.
