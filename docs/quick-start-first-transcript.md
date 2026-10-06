# Quick start: transcribe your first file

This guide tells you how to make a transcript of one recording.

Before you start, set up your API keys. Read
[Quick start: set up your API keys](quick-start-api-keys.md).

## What you need

- VoxVerbatim, with your API keys set up.
- A recording in one of these formats: `.m4a`, `.mp3`, `.wav` or `.flac`.
- A connection to the internet.

**Tip:** For your first try, use a short recording of 1 to 5 minutes.
It is fast and it costs very little.

## Step 1: Start VoxVerbatim

Double-click **VoxVerbatim.cmd**.

The first time, this takes about 1 minute. VoxVerbatim downloads the
parts it needs. After that, it starts immediately.

## Step 2: Open the folder with your recording

1. Press `Ctrl+O`. Or, select the **Browse...** button.
2. Find the folder that contains your recording.
3. Select the folder.

The file list shows each recording in the folder, with its length and
size.

## Step 3: Select the recording

1. Go to the file list.
2. Use the `Up` and `Down` arrow keys to highlight your recording.

VoxVerbatim transcribes the highlighted recording. To transcribe more
than one recording, press `Space` on each one to tick it. VoxVerbatim
then transcribes only the ticked recordings.

To listen to the highlighted recording, press `Ctrl+Space`.

## Step 4: Open the Transcribe dialog

Press `Ctrl+T`. Or, select the **Transcribe...** button under the file
list.

The Transcribe dialog opens. It shows the recordings it will transcribe.

## Step 5: Tell VoxVerbatim about the recording

All of these are optional. Good answers give a better transcript.

| Box | What to type |
| --- | --- |
| **Afrikaans may be spoken in these recordings** | Tick this only if a person in the recording can speak Afrikaans. Otherwise, leave it clear. |
| **Expected number of speakers** | The number of people who speak in the recording. |
| **Known speaker names** | The names of the people, if you know them. |
| **Recording context** | One sentence about the recording. For example: "A meeting about the new office lease." |
| **Vocabulary profiles to use** | Tick the lists of special words that apply. On your first run, you will not have any. |

Each box has an explanation. Move the focus to a box, then read the
panel that shows the explanation. To read all the explanations together,
select **Guide to these settings...**.

## Step 6: Check the cost

The dialog shows the approximate cost of the run. This is an estimate.
The real cost can be a little different.

## Step 7: Start the transcription

1. Select **Start**.
2. A message asks "Start transcribing these recordings?" It shows the
   cost again.
3. Select **Yes** to start. Select **Cancel** to stop.

**Note:** In this message, `Enter` selects **Cancel**. This prevents an
accidental start. Use `Tab` to move to **Yes**, then press `Space`.

If something is missing, such as a key, VoxVerbatim tells you before it
sends anything. You do not pay for a run that cannot finish. Do what the
message tells you, then start again.

## Step 8: Wait for the transcription to finish

A progress bar and a sentence show what VoxVerbatim is doing. A short
recording takes a few minutes. A long recording takes longer.

To stop, select **Cancel**. VoxVerbatim keeps the work it has done. It
marks the transcript as incomplete.

## Step 9: Read your transcript

When the run is complete, a message tells you how many places need a
person to check them. Select **OK**.

Then select one of these buttons:

- **Open transcript folder** opens the folder with your transcript.
- **Open review window** opens the window where you check the uncertain
  words.

Your transcript is in a new folder next to the recording. For a
recording named `meeting.m4a`, the files are here:

```
meeting.m4a.transcript/
    exports/
        transcript.txt     The transcript. Open it to read it.
        review-report.md   How much you can trust the transcript.
```

## Step 10: Check the uncertain words

Sometimes the services do not agree on a word. Then the transcript shows
the possible words, like this:

```
the invoice came to [UNCERTAIN: 15,000 / 50,000] before VAT
```

VoxVerbatim never guesses numbers, amounts or dates. You must decide.

To decide, use the review window:

1. Press `Ctrl+R`, or select **Open review window**.
2. Select an item. VoxVerbatim plays that part of the recording.
3. Do one of these:
   - If the word is correct, press `F4`.
   - If the word is wrong, type the correct word.
4. Go to the next item. Press `F3` for the next one, or `Shift+F3` for
   the one before. Press `F5` to hear it again.

You do not need to save. VoxVerbatim saves each change immediately.

When you correct a word, VoxVerbatim remembers it for this folder. Next
time, it corrects the same word in new recordings in this folder
automatically.

## If you need help

- Press `F1` in the main window to see all the keyboard shortcuts.
- If a run stops with an error, select **File**, then **Open Log File**.
  The log tells you what went wrong.
