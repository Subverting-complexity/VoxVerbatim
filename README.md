# Audio Transcriber

An accessible Windows desktop application for working through audio
recordings. It plays them, it makes louder copies of the quiet ones, and it
transcribes them by sending each recording to several speech services at
once and working out one answer from what they all said.

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

## Sending it to somebody else

Everything above asks the person running the application to have Python,
which is a great deal to ask of somebody who was simply sent a program to
try. Double-click **publish.cmd** and you get a copy that does not.

It builds the application into `publish\Audio Transcriber`, a folder holding
the program and everything it needs, Python included. Copy that whole folder
to another Windows computer and the person you sent it to opens
**Audio Transcriber.exe** inside it. Nothing is installed on their machine;
the folder is the application. Their settings still go to their own user
folder rather than into it, so you can send them a newer folder later
without disturbing anything they have chosen or taught it.

The folder is about 260 MB, most of which is Qt. The first build takes
several minutes, because it downloads the libraries in `requirements.txt`
and the build tool before it starts, which is a few hundred megabytes
between them. Later builds take about a minute and a half. The `publish` folder is not kept in the
repository; it is rebuilt whenever you run the script, which begins by
deleting the previous build, so close any copy of the application you have
running from it.

A short note travels inside the folder for whoever receives it, because
there is one thing they will meet that needs explaining. The program is not
signed with a certificate, so the first run brings up SmartScreen's "Windows
protected your PC". That is a statement about how many people have run this
particular program, not about what is in it, and the way past it is "More
info" and then "Run anyway".

Keep the repository somewhere with a reasonably short path. Some of the file
names inside the ElevenLabs library are long enough that a deep folder
carries them past the 260-character limit Windows still applies by default,
and installing the libraries fails partway through with an error about a
file it cannot find.

## What it does

Choose a folder of recordings and the application lists every audio file in
it with its length and its size. Pick one and play it, moving backwards and
forwards through it in steps of 15 seconds, 2 minutes or 5 minutes, or by
dragging the seek bar.

Each file also has a check box. Checking is how you choose which recordings
to work on: **Enhance Audio** and **Transcribe** both act on the checked
files, or on the highlighted one if you have checked nothing. Checking a
file and playing a file are separate. The player always follows the
highlighted file, never the checked ones.

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

## Transcription

Press `Ctrl+T`, or choose **File** then **Transcribe**. It works on the
checked files, or on the highlighted file if none are checked.

Several speech services transcribe the same recording and the application
works out one answer from what they all said. Three of them, ElevenLabs
Scribe, OpenAI and Microsoft MAI, are sent every recording in full and at
the same time, because they do not depend on each other and running them in
sequence would take three times as long for nothing. A fourth, AssemblyAI,
is deliberately held back: it is asked only about the passages the first
three could not agree on, since a second opinion is worth paying for where
there is a dispute and worth nothing where there is not. A fifth, Deepgram,
has a settings page and an adapter but takes no part in a run: nothing in
this version depends on it, and it is there so that comparing it against
the others later costs nobody a rewrite.

Where the services disagree, the disagreement is the useful part. Two
services hearing the same word two ways is exactly the signal that says
"listen to this bit yourself", and it is a signal a single service cannot
give you, because a single service has nothing to disagree with.

### One word, three answers

This is the idea worth understanding, because everything else follows from
it. A transcript answers three separate questions about every word: what
was said, when it was said, and who said it. Those are three questions, not
one, and the service best placed to answer one of them is often not the
service best placed to answer another. Only ElevenLabs times individual
words. Only ElevenLabs and AssemblyAI tell speakers apart. Microsoft is the
only one that says which language each phrase was in.

So the three answers are kept apart, and a word may take each of them from
a different place. Suppose the word is a surname, Nkosi. OpenAI writes it
down as "Nkosi" and Microsoft agrees, so the spelling is theirs. Neither of
them times a word at all, so the fact that it runs from 4.2 seconds to 4.7
comes from ElevenLabs, which does.
ElevenLabs put this word and the one before it in the same person's mouth,
but AssemblyAI, asked afterwards about that passage, heard the speaker
change; the speaker label is AssemblyAI's. One word, three answers, three
sources, and the transcript records which service each of the three came
from.

That is an ordinary word, not a broken one. The alternative is worse in
both directions. Trusting one service for everything means accepting its
spelling of every surname, however good another service is at names; and
letting whichever service "won" the spelling supply the timing as well
means taking a time from a service that never measured one. A correction
you make later works the same way: changing a spelling does not touch the
clock, and changing who was speaking does not touch the words.

### Being unsure is a real answer

Where the evidence does not settle something, the transcript says so and
waits for you. It does not quietly pick whichever candidate scored highest.
In the plain-text export an unsettled word is written out as the choice it
still is:

```
the invoice came to [UNCERTAIN: 15,000 / 50,000] before VAT
```

This matters most for amounts, dates, account numbers and the like, and it
matters there for a particular reason. Get a common word wrong and the
sentence reads slightly oddly, which is how you notice. Get an amount wrong
and the sentence reads perfectly. There is nothing on the page to catch
your eye, nothing that looks like a mistake, and the error survives every
reading until somebody happens to check it against something else. A number
is exactly the kind of thing plausibility cannot decide, because both
candidates are equally plausible sitting in a sentence.

So values of that kind are never settled by scoring, and they are never
handed to the language model to settle either. They go to you.

### Afrikaans, chosen per recording

The Transcribe dialog has a box for whether Afrikaans may be spoken in
these recordings. It is off by default, and it is a decision about the
recordings in front of you rather than a preference you set once, which is
why it lives in that dialog and not in Settings.

Left off, Afrikaans is not one of the available answers. There is no
Afrikaans detection, no Afrikaans model anywhere in the run, and no service
is told that Afrikaans is possible. A German passage therefore cannot be
mistaken for Afrikaans, because there is nothing to mistake it for.

Switching it on narrows the evidence rather than widening it, and that is
worth knowing before you do it. Of the five services only ElevenLabs and
OpenAI handle Afrikaans well. Microsoft and Deepgram do not support it at
all, so they contribute nothing to an Afrikaans passage. AssemblyAI can
only hear it on its older model, at a published word error rate of between
a quarter and a half, so a second opinion on Afrikaans is a far weaker
second opinion than on English. Switch it on where Afrikaans may genuinely
occur, and leave it off where it may not. There is nothing to be gained by
switching it on just in case.

### What you do

Check the recordings, press `Ctrl+T`, and the dialog opens with them
already listed. You then say the few things no service can work out for
itself: whether Afrikaans may occur, how many people you expect to hear,
their names if you know them, a sentence about what the recording is, and
which of your vocabulary profiles apply to it.

Below that is what the run is expected to cost, worked out from the length
of the audio and the rates you keep in Settings. It is an estimate and says
so: the rates are typed in by hand and prices change, and the parts that
depend on how much the services disagree cannot be known until they have
answered. Unless you switch it off, you are asked to agree to the figure
before anything is sent. Everything that would stop a run, a missing key
most often, is checked first, because finding that out half way through
means having paid for a transcript nobody can use.

Then it runs in the background, one recording after another, with a
progress bar and a running sentence saying what is happening. It can be
cancelled at any point, and a cancelled run keeps what it had and says the
transcript is incomplete.

When it finishes you are told how many places need a person, and offered
the two useful things: the transcript folder, and the review window.

The review window, on `Ctrl+R`, is where you work through them. It holds
only the words that need you, and it can be narrowed by why each one was
flagged and by how confident the application is. For each one you can hear
the word with several seconds either side, see what every service heard
there, and then correct the text, the speaker or the timing separately.
`F3` moves to the next item, `F5` plays it again, and `F4` confirms it as
correct and moves on. Every correction is also remembered: the next
transcription is sent the words you have already put right.

### What ends up beside the recording

A transcript is not one file, so it gets a folder of its own next to the
recording, named after it:

```
meeting.m4a
meeting.m4a.transcript/
    transcript.json      The words, what every service said, and every decision
    raw-responses/       What each service actually sent back, untouched
    exports/
        transcript.txt   The transcript itself, to read or paste
        review-report.md Whether it can be trusted, and what still needs you
    chunks/              The pieces a service was sent, where one had to be cut up
```

Beside the recording rather than in one central place, so that moving a
recording to another drive takes its transcript, its evidence and its
provenance with it. Two more things appear in there when they are needed: a
lossless copy of the recording, where the original was in a format the
services will not take, and an `escalation` folder holding the few seconds
of audio around each passage that was sent out for a second opinion.

The two exports answer two different questions. `transcript.txt` answers
"what was said": the words, grouped into turns, labelled with who was
speaking, wrapped so they can be read anywhere. `review-report.md` answers
"can I trust this": which services answered and which did not, how much of
the transcript is solid, and then every place that still needs a person,
with the time it happens, why it was flagged, what each service heard there
and what the application settled on.

The raw answers are kept because a transcription cannot otherwise be
explained afterwards. Services change what sits behind a fixed model name,
without telling anyone, so the day the same recording comes back different
is the day you need to know whether the application changed or the service
did. The only way to tell is to still have what the service said the first
time. The files are numbered upwards and never rewritten, so a second run
adds to the folder and cannot overwrite what the first one found.

No API key is ever written into any of it. The settings a run used are
recorded in the transcript so that the run can be explained later, and they
are stripped of credentials on their way out of Settings and checked a
second time as the file is written. A transcript is a document people send
to each other, and a key that has reached a disk cannot be called back.

### Where the settings live

Everything about the services is in the Settings dialog, on `Ctrl+,`, which
has a page for each: ElevenLabs, OpenAI for transcription, OpenAI for
settling disputes, Microsoft, AssemblyAI and Deepgram. Each page holds that
service's API key, its model name, and a box of free-form parameters that
are passed through exactly as you write them. That last box is there
because services add parameters between our releases, and a user who knows
about a new one should be able to use it by typing it in rather than by
waiting for us. The text is checked when you press OK, so a mistake stops
the dialog closing and says exactly what is wrong rather than failing in
the middle of a paid run.

Beside them are pages for how a run behaves, for the rates each service
charges, for your vocabulary profiles, and for what the application has
learned about each service from your own corrections.

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
* The review queue is a table in the same way, and how confident the
  application is about a word is a word in a column rather than a shade of
  orange. It also says how many items it is showing out of how many there
  are, and announces that when a filter changes, because a filter that
  silently empties a list cannot be told from a broken one.
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
  Every setting in the Enhance Audio, Transcribe and Settings dialogs
  carries a short summary that a screen reader reads when the focus lands on
  it, and a full explanation in a panel that follows the focus. The panel
  takes focus itself so it can be read, and holds still while you read it.
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
| `Ctrl+T` | Transcribe the checked files, or the highlighted one |
| `Ctrl+R` | Open the review window on the highlighted recording |
| `Ctrl+Space` | Play, or pause if already playing |
| `Alt+Left` / `Alt+Right` | Back or forward by the short skip |
| `Alt+Shift+Left` / `Alt+Shift+Right` | Back or forward by the medium skip |
| `Alt+Ctrl+Left` / `Alt+Ctrl+Right` | Back or forward by the long skip |
| `Ctrl+,` | Open the Settings dialog |
| `F6` | Move to the next panel |
| `Ctrl+Q` | Close the application |

## Settings

Press `Ctrl+,`, or choose **File** then **Settings**. There are a great many
settings now, most of them belonging to one speech service or another, so
the dialog is a list of categories on the left and one page at a time on the
right. Whichever setting has the focus is explained in a panel at the
bottom, in the same way the Enhance Audio dialog explains its own.

Under **General**:

* Whether the folder from your last run is opened again when you start.
  Switching this off does not forget the folder; it simply waits for you to
  choose one.
* The three skip intervals, which start at 15 seconds, 2 minutes and 5
  minutes. Changing one changes the buttons, the Playback menu, the keyboard
  shortcut list and what a screen reader reads out, all together.

Every other category belongs to transcription, and the Transcription section
above describes them.

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

Five files live there:

| File | What it holds |
| --- | --- |
| `settings.json` | What you chose in the Settings and Enhance Audio dialogs |
| `session.json` | Where you were: folder, checked files, highlighted file, window layout |
| `vocabulary.json` | Your vocabulary profiles, and the corrections you have made |
| `calibration.json` | How each service has done on your own recordings |
| `audio-transcriber.log` | What went wrong, if anything did |

They are separate on purpose. Your settings are decisions you made and
expect to keep; the session is only where you happened to be. The vocabulary
and the statistics are per user rather than per folder, because the same
client's surname is worth knowing in every folder you will ever open, and
because statistics built up one correction at a time over months should not
start again when you work somewhere new. Deleting any of them costs you what
that file held and nothing else; none of it stops the application from
starting. They are all plain JSON you can read or edit by hand, and a
damaged one is ignored rather than being fatal.

Nothing in the transcript folders beside your recordings is written here,
and nothing here holds a copy of a transcript. The two are kept apart so
that a recording and its transcript travel together.

## How the code is arranged

```
audio_transcriber/
    app.py           Starting up: logging, the stores, the main window
    paths.py         Where the settings, the session, the vocabulary and the log are kept
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
    transcription/
        model.py         What a transcript is made of: evidence and decisions
        pipeline.py      The order it all happens in, from recording to transcript
        passes.py        Sending the recording to every service at once
        canonical.py     The one audio file every timestamp is measured against
        chunking.py      Cutting a recording up for a service that cannot take it whole
        context.py       What each service is told about the recording
        alignment.py     Laying every service's words side by side on one timeline
        normalise.py     Recognising two spellings of the same spoken thing
        reconcile.py     Deciding what was said, word by word
        language.py      Which language each word is in
        confidence.py    Turning many kinds of evidence into one honest category
        risk.py          Amounts and account numbers, which are never guessed at
        diarisation.py   Who said it
        timing.py        When it was said, and measuring again where the words changed
        escalation.py    A second opinion on the places still in doubt
        adjudication.py  Letting a reasoning model settle what is left
        vocabulary.py    Names and specialist words the services should expect
        learning.py      What your corrections teach the next transcription
        calibration.py   How well each service has actually done
        cost.py          What a run will cost, before it starts
        exports.py       The two documents a person actually reads
        store.py         The folder of files that sits beside each recording
        runner.py        Working through several recordings on a background thread
        providers/
            base.py      What every service adapter must offer
            registry.py  Turning your settings into the services that get called
            elevenlabs.py, openai.py, microsoft.py, assemblyai.py, deepgram.py
    ui/
        main_window.py     Ties everything together
        folder_panel.py    Choosing the folder
        file_table.py      The file list, its model and its keyboard handling
        player_panel.py    Transport buttons and the seek bar
        file_info_panel.py Details of the selected file
        settings_dialog.py The category list, the pages and the note panel
        settings_pages.py  One page per category of setting
        settings_notes.py  What each setting means, in words
        statistics_page.py How each service has done, as a table
        enhance_dialog.py  Setting up and running an enhancement
        enhance_notes.py   What each enhancement setting means, in words
        transcribe_dialog.py Setting up a transcription run, pricing it, running it
        review_window.py   Working through what could not be settled
        review_queue.py    The list of words needing a person, as a table
        help_dialogs.py    Keyboard shortcuts and About
        accessibility.py   Naming controls and announcing changes
        flow_layout.py     A row of buttons that wraps when space is short

packaging/
    audio-transcriber.spec  How the shareable build is put together
    launch.py               What the built application starts from
    Read me first.txt       The note that travels inside the built folder
```

Nothing in `audio_transcriber/session.py`, `settings.py`, `formatting.py`,
`audio/` or `transcription/` depends on the user interface, apart from the
two runners, whose whole job is to put slow work on a background thread and
report back. The whole of transcription can therefore be run, and is tested,
without a window existing at all; and the review window, in the other
direction, takes a finished transcript and something that can play audio and
knows nothing about services or storage.

Within `transcription/`, each module does one job and knows nothing about
the others. `pipeline.py` is the only one that knows the order they go in,
deliberately, so that somebody who wants to understand what the application
does to a recording can read one function rather than trace calls through
nine files.

Adding a setting means adding a field to `Settings`, a row to the dialog,
and using it wherever it belongs. Nothing else: reading, checking, saving
and surviving a damaged file are already handled for every field. Adding a
speech service means writing an adapter against the contract in
`providers/base.py`, a builder in `providers/registry.py`, and a settings
page. Nothing above the adapter learns its name.

## Working on it

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
```

The tests run without a visible desktop by asking Qt for its offscreen
platform, so they can run anywhere.
