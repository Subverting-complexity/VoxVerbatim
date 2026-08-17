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

## Enhance Audio

Recordings made at a conservative level are often perfectly clean and
simply too quiet, both to listen to and for a transcription service to work
with. **Enhance Audio** writes louder copies of them into a folder you
choose, leaving the originals untouched.

Press `Ctrl+E`, or use the **Enhance Audio** button under the file list. It
works on the checked files, or on the highlighted file if none are checked.

What it does to each recording is measure it, multiply the whole waveform by
one number, and write the result out losslessly. Nothing else. It does not
compress the dynamic range, it does not attempt noise reduction, and it
never encodes back into a lossy format, which would add a second generation
of coding damage for no benefit.

Two measurements decide the number:

* **Integrated loudness**, in LUFS, is how loud the recording sounds over
  its whole length, measured the way broadcasters measure it.
* **True peak**, in dBTP, is the highest value the waveform reaches
  *between* the stored samples. A recording can sit below zero at every
  stored sample and still overshoot in between, which is what distorts on
  playback.

The gain is whatever brings the loudness to your target, held back so that
the true peak stays under your ceiling. So a recording measuring -29.4 LUFS
with a true peak of -11.7 dBTP is raised by 10.7 dB rather than the 11.4 dB
it asked for, landing at -18.7 LUFS with its peak exactly on -1 dBTP.

You can change:

| Setting | What it means |
| --- | --- |
| Target loudness | How loud each copy is made. Around -18 LUFS suits speech. A recording already louder than this is turned down to it. |
| True-peak ceiling | How close to full scale the loudest moment may come. -1 dBTP leaves room so that nothing distorts on playback. |
| Maximum gain | The furthest a quiet recording is raised, even if that leaves it short of the target. |
| Limiter | Off by default. Holds down the few loud moments that would otherwise keep a whole quiet recording down. It is the only setting that changes the shape of the sound rather than only its volume. |
| Output format | 16-bit WAV, 24-bit WAV or FLAC. All three are lossless; they differ in file size and in how widely they are accepted. |
| Replace existing files | Off by default, so a copy that is already there is left alone and reported as skipped. |

Raising the volume cannot improve the ratio of speech to background noise:
it lifts both by the same amount. That is what the maximum gain is for. A
recording so quiet that it needs 40 dB is better left short of the target
than dragged up to it.

You do not have to remember any of this while you are using it. The dialog
has a panel at the bottom that explains whichever setting you are on: tab
onto the limiter and the panel tells you what ticking it does, what it
costs, and when to bother. Tab into the panel and it holds still so you can
read it. `F1`, or the **Guide to these settings** button, shows all of the
explanations together in one window you can read straight through and copy
from.

The run happens in the background with a progress bar, and can be cancelled
at any point. A cancelled run keeps the files it had already written and
throws away the one it was part way through. When it finishes you get a
message saying how it went, and a report giving, for each recording, what it
measured, how far it was moved, where it was written, and whether anything
held the gain back.

The measuring, the gain and the limiter all come from FFmpeg, which arrives
with the PyAV library. There is nothing to install separately.

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
  Windows text sizes and high scaling. The Enhance Audio dialog scrolls
  rather than growing past the bottom of the screen, so Start and Cancel
  never go out of reach.
* Explanations that are too long for a tooltip are not hidden behind one.
  Each Enhance Audio setting carries a short summary that a screen reader
  reads when the focus lands on it, and a full explanation in a panel that
  follows the focus. The panel takes focus itself so it can be read, and
  holds still while you read it.
* Nothing is signalled by colour, icon or position alone.

Press **F1** in the application for the full list of keyboard shortcuts.
The main ones:

| Key | What it does |
| --- | --- |
| `Ctrl+O` | Choose the audio folder |
| `F5` | Read the folder again |
| `Up` / `Down` | Move through the file list |
| `Space` | Check or clear the highlighted file |
| `Ctrl+E` | Enhance the checked files, or the highlighted one |
| `Ctrl+Space` | Play, or pause if already playing |
| `Alt+Left` / `Alt+Right` | Back or forward by the short skip |
| `Alt+Shift+Left` / `Alt+Shift+Right` | Back or forward by the medium skip |
| `Alt+Ctrl+Left` / `Alt+Ctrl+Right` | Back or forward by the long skip |
| `Ctrl+,` | Open the Settings dialog |
| `F6` | Move to the next panel |
| `Ctrl+Q` | Close the application |

## Settings

Press `Ctrl+,`, or choose **File** then **Settings**. You can change:

* Whether the folder from your last run is opened again when you start.
  Switching this off does not forget the folder; it simply waits for you to
  choose one.
* The three skip intervals, which start at 15 seconds, 2 minutes and 5
  minutes. Changing one changes the buttons, the Playback menu, the keyboard
  shortcut list and what a screen reader reads out, all together.

The **File** menu also opens the log file, and the folder that holds your
settings, so you never have to go looking for either.

## Where your settings are kept

Everything the application writes for itself lives in one folder:

```
%LOCALAPPDATA%\JB Org\Audio Transcriber
```

That is `C:\Users\<you>\AppData\Local\JB Org\Audio Transcriber`. The path is
not written into the code. It comes from asking Windows where per-user
configuration belongs, using the organisation name and the application name,
so renaming either moves the folder.

Three files live there:

| File | What it holds |
| --- | --- |
| `settings.json` | What you chose in the Settings and Enhance Audio dialogs |
| `session.json` | Where you were: folder, checked files, highlighted file, window layout |
| `audio-transcriber.log` | What went wrong, if anything did |

They are separate on purpose. Your settings are decisions you made and
expect to keep; the session is only where you happened to be. Deleting
`session.json` loses your place, and deleting `settings.json` loses your
preferences, and neither stops the application from starting. Both files are
plain JSON you can read or edit by hand, and a damaged one is ignored rather
than being fatal.

## How the code is arranged

```
audio_transcriber/
    app.py           Starting up: logging, the stores, the main window
    paths.py         Where the settings, the session and the log are kept
    json_store.py    Reading and writing those files safely
    settings.py      The settings the user chooses
    session.py       Reading and writing the saved session
    formatting.py    Durations and sizes, in compact and spoken forms
    audio/
        library.py       Finding audio files and reading their metadata
        scanner.py       Doing that on a background thread
        player.py        Playback, wrapping Qt Multimedia
        enhance.py       Measuring loudness and writing louder copies
        enhance_runner.py  Doing that on a background thread
    ui/
        main_window.py     Ties everything together
        folder_panel.py    Choosing the folder
        file_table.py      The file list, its model and its keyboard handling
        player_panel.py    Transport buttons and the seek bar
        file_info_panel.py Details of the selected file
        settings_dialog.py Changing the settings
        enhance_dialog.py  Setting up and running an enhancement
        enhance_notes.py   What each enhancement setting means, in words
        help_dialogs.py    Keyboard shortcuts and About
        accessibility.py   Naming controls and announcing changes
        flow_layout.py     A row of buttons that wraps when space is short
```

Nothing in `audio_transcriber/session.py`, `settings.py`, `formatting.py`,
`audio/library.py` or `audio/enhance.py` depends on the user interface,
which is what will let the transcription services be added underneath the
same window without disturbing it.

Adding a setting means adding a field to `Settings`, a row to the dialog,
and using it wherever it belongs. Nothing else: reading, checking, saving
and surviving a damaged file are already handled for every field.

## Working on it

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
```

The tests run without a visible desktop by asking Qt for its offscreen
platform, so they can run anywhere.
