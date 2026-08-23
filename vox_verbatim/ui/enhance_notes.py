"""What each Enhance Audio setting means, written out for the user to read.

Every setting carries two pieces of writing. The **summary** is one or two
sentences: it becomes the control's accessible description and its tooltip,
so a screen reader reads it on focus and a mouse user sees it on hover. It
has to be short, because hearing five paragraphs every time the focus lands
on a control is unusable.

The **note** is the full explanation, and it is what this module is really
for. It says what the setting changes, gives a worked example where a
number alone would not land, states the trade-off honestly, and ends with
what to actually do. It is too long for a tooltip, so the dialog shows it
in a panel that follows the focus, and offers the whole set together in a
guide window.

Keeping the words here, apart from the widgets, means the same sentence is
never written twice, and that the wording can be read and checked without
starting a user interface.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

#: Every note in the dialog is filed under one of these.
FILES = "files"
OUTPUT_FOLDER = "output_folder"
TARGET_LOUDNESS = "target_loudness"
CEILING = "ceiling"
MAXIMUM_GAIN = "maximum_gain"
LIMITER = "limiter"
OUTPUT_FORMAT = "output_format"
REPLACE_EXISTING = "replace_existing"


@dataclass(frozen=True)
class SettingNote:
    """What one setting is called, and what there is to say about it."""

    key: str
    title: str
    """The heading it appears under in the guide."""

    summary: str
    """One or two sentences, for the accessible description and the tooltip."""

    note: str
    """The full explanation, for the panel and the guide."""


def _reflowable(text: str) -> str:
    """Put each paragraph on one line, so it can be wrapped to fit.

    The notes are written wrapped in the source below, because that is
    where they are read and reviewed. On screen the wrapping has to follow
    the width of the panel and the size of the user's font instead, and a
    screen reader reads a text box a line at a time, so a paragraph broken
    across eight lines is read as eight lines with a pause in each gap.
    Both want one long line per paragraph and let the widget do the rest.
    """
    paragraphs = text.strip().split("\n\n")
    return "\n\n".join(
        " ".join(line.strip() for line in paragraph.splitlines()).strip()
        for paragraph in paragraphs
    )


_WRITTEN_NOTES: tuple[SettingNote, ...] = (
    SettingNote(
        key=FILES,
        title="Files to enhance",
        summary=(
            "The recordings this run will work through. To change them, close this "
            "dialog, check the files you want in the file list, and open it again."
        ),
        note="""\
These are the recordings this run will work through.

They are the files you checked in the main window, or the highlighted file
if you checked nothing. To change the list, close this dialog, check the
files you want, and open it again.

Every recording is measured and adjusted on its own terms. A whispered
phone recording and a clear desk recording in the same run both come out at
the target, which is the whole point of doing it this way rather than
applying one setting to everything.""",
    ),
    SettingNote(
        key=OUTPUT_FOLDER,
        title="Output folder",
        summary=(
            "Where the enhanced copies are written. It is created if it does not "
            "exist yet, and your original recordings are never changed."
        ),
        note="""\
This is where the enhanced copies are written. The folder is created if it
is not there yet.

The suggestion is a folder called Enhanced beside your recordings. That
keeps the copies with the originals without them turning up in the file
list, because the file list reads one folder and does not look inside
sub-folders. So enhancing the same folder again does not enhance the
enhanced copies.

Your original recordings are never written to, moved or renamed. If the
folder and format you choose would put a copy exactly where an original
already sits, that recording is skipped and reported rather than
overwritten.

Whatever you choose is remembered, so the next run opens on the same
folder.""",
    ),
    SettingNote(
        key=TARGET_LOUDNESS,
        title="Target loudness",
        summary=(
            "How loud each finished copy is made overall. Around -18 LUFS suits "
            "speech. A recording already louder than this is turned down to it."
        ),
        note="""\
This sets how loud each finished copy is, measured across the whole
recording rather than at any one moment. LUFS is the scale broadcasters
use, and it follows how loud something actually sounds rather than how big
the numbers in the file are.

A recording quieter than the target is raised to it. A recording already
louder is turned down to it. That is deliberate: after a run, every file
sits at the same level, and you stop reaching for the volume control
between them.

Around -18 LUFS suits speech for transcription. Most services are happy
anywhere between about -23 and -16, so the exact figure matters far less
than every recording arriving at the same one.

Asking for a higher target does not add any detail. It only asks for more
gain, which the true-peak ceiling and your maximum gain may refuse to give.

In practice, leave it at -18 unless a transcription service asks you for
something else.""",
    ),
    SettingNote(
        key=CEILING,
        title="True-peak ceiling",
        summary=(
            "How close to full scale the loudest moment may come. Zero is full "
            "scale; -1 dBTP leaves room so that nothing distorts on playback."
        ),
        note="""\
This sets how close to full scale the loudest moment is allowed to come.
Zero is full scale, and going above it distorts.

True peak means the highest value the waveform actually reaches, including
between the stored samples. A recording can sit below zero at every stored
sample and still overshoot in between. That overshoot does no harm sitting
in the file, but it turns into a click or a crackle when the file is
played, converted, or compressed for upload, which is exactly what happens
to a recording on its way to a transcription service.

-1 dBTP leaves a decibel of room for it. That costs you one decibel of
loudness and buys protection against a fault that cannot be repaired
afterwards, because once a peak has been squared off the original shape is
gone.

Lower it to -2 or -3 if the enhanced files will be encoded again by
something else, such as a service that converts uploads to MP3. Move it
towards zero only if you know the file will be played exactly as it is.

The ceiling is never exceeded, whatever else is set.""",
    ),
    SettingNote(
        key=MAXIMUM_GAIN,
        title="Maximum gain",
        summary=(
            "The furthest a quiet recording is raised, even if that leaves it short "
            "of the target. It never stops a loud recording being turned down."
        ),
        note="""\
This limits how far a quiet recording is raised, even when that leaves it
short of the target.

Turning the volume up cannot improve a recording. It lifts the speech and
the background hiss by exactly the same amount, so a recording that needs
40 dB reaches the target with its noise floor 40 dB higher as well. What
was a faint hiss becomes the loudest thing in the room, and a transcription
service has a harder time with it, not an easier one.

30 dB is a generous allowance. It catches genuinely quiet recordings
without turning near-silence into noise. A recording that hits the limit is
usually one where something went wrong at the time: the wrong microphone,
or a phone left in a pocket.

The report tells you when a recording was held back by this, so you can see
which ones hit it and judge whether they are worth recording again.

Lower it to around 20 dB if you would rather have files that are quiet than
files that are noisy. It has no effect on turning a loud recording down,
which is always allowed however far it has to go.""",
    ),
    SettingNote(
        key=LIMITER,
        title="Use a limiter to reach the target loudness",
        summary=(
            "Changes how the ceiling is enforced, not the ceiling itself. It lets "
            "quiet speech reach the target by briefly holding down loud peaks."
        ),
        note="""\
This changes how the ceiling is enforced, not the ceiling itself.

With the box clear, the loudest peak limits how much the entire recording
can be boosted. For example, if a cough is already close to the ceiling,
the software may only be able to raise everything by 5 dB, even though the
speech needs 22 dB. The result may still be too quiet.

With the box ticked, the software boosts the recording enough to reach the
target, then uses a limiter to stop individual loud peaks from exceeding
the ceiling. The speech can therefore get the full 22 dB boost while the
cough is briefly reduced.

The trade-off is that a limiter slightly changes the shape of the audio
around loud peaks. Without it, the entire waveform is simply made louder by
the same amount. With it, very loud moments are temporarily reduced.

For transcription, using the limiter is usually worthwhile, because louder
speech is more useful than preserving the exact shape of a cough or other
sudden noise. For critical listening, leave it off if you want the audio
changed only by a fixed gain.

Your maximum gain still applies in both cases, and the ceiling is never
exceeded.

In practice, try it without the limiter first. If the report says the
true-peak ceiling prevented the recording from getting loud enough, switch
the limiter on and run it again.""",
    ),
    SettingNote(
        key=OUTPUT_FORMAT,
        title="Output format",
        summary=(
            "What the enhanced copies are written as. All three are lossless, so "
            "they differ only in file size and in what will accept them."
        ),
        note="""\
This chooses what the enhanced copies are written as. All three are
lossless, so none of them throws anything away. They differ in size and in
what will accept them.

WAV, 16-bit holds the same detail as a compact disc and is accepted by
every transcription service there is. Start here.

WAV, 24-bit keeps more detail in the very quietest parts of a recording.
Once speech has been raised to a sensible level there is little down there
to keep, so in practice this mostly buys you a file half again as large.

FLAC holds exactly the same samples as 16-bit WAV in roughly half the
space, and unpacks back to them without any difference at all. It is worth
choosing when the files have to be uploaded or stored, and worth avoiding
if something further down the line will not read it.

Nothing is ever encoded into MP3, AAC or any other lossy format. Doing so
would add a second round of coding damage on top of whatever the original
recording already carries, for no benefit.""",
    ),
    SettingNote(
        key=REPLACE_EXISTING,
        title="Replace files that are already in the folder",
        summary=(
            "What happens when the output folder already holds a copy. With this "
            "clear, that recording is left alone and reported as skipped."
        ),
        note="""\
This decides what happens when the output folder already holds a copy of a
recording you are enhancing.

With the box clear, that recording is left exactly as it is and reported as
skipped. Running the same folder a second time therefore costs nothing and
changes nothing, which makes it safe to run again after adding a few new
recordings: only the new ones are done.

With the box ticked, the copy is written again and the previous one is
gone. Tick it when you have changed a setting and want the whole set redone
at the new one.

Your original recordings are not at stake either way. This is only ever
about the enhanced copies in the output folder.""",
    ),
)

#: The notes as the rest of the application sees them, with each paragraph
#: on a single line ready to be wrapped to whatever width it is shown at.
_NOTES: tuple[SettingNote, ...] = tuple(
    replace(note, note=_reflowable(note.note)) for note in _WRITTEN_NOTES
)

NOTES: dict[str, SettingNote] = {note.key: note for note in _NOTES}


def note_for(key: str) -> SettingNote:
    """Return the note filed under ``key``.

    Raises:
        KeyError: if there is no such note, which is a mistake in the code
            rather than anything the user can cause.
    """
    return NOTES[key]


def summary_of(key: str) -> str:
    """The short form, for an accessible description and a tooltip."""
    return NOTES[key].summary


def note_text(key: str) -> str:
    """The full explanation, for the panel and the guide."""
    return NOTES[key].note


def guide_text() -> str:
    """Every note, one after another, for reading straight through.

    The panel in the dialog shows one note at a time, which suits somebody
    working through the settings. This is the same words laid out for
    somebody who would rather read the lot before touching anything.
    """
    parts = ["Enhance Audio: what each setting does", ""]
    for note in _NOTES:
        parts.append(note.title)
        parts.append("-" * len(note.title))
        parts.append("")
        parts.append(note.note)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"
