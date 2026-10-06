"""Making quiet recordings louder without damaging them.

Most of the recordings this application is built for are clean but were
made at a conservative level. They are perfectly good; they are simply
quiet. The fix for that is to multiply the waveform by a constant, which
changes nothing except how loud it is.

That is all this module does. It does not compress the dynamic range, it
does not attempt noise reduction, and it does not re-encode into a lossy
format. A recording that arrives as ``.m4a`` is decoded once into
uncompressed samples, multiplied, and written out losslessly. Encoding it
back into a lossy format would add a second generation of coding damage
for no benefit, so it is never done.

Working out how far a recording can be raised takes two numbers:

* The **integrated loudness**, in LUFS. This is how loud the recording
  sounds over its whole length, measured the way broadcasters measure it
  (EBU R128, which is the ITU-R BS.1770 method).
* The **true peak**, in dBTP. This is the highest value the waveform
  actually reaches between the stored samples, not merely at them. A
  signal can sit below zero at every stored sample and still overshoot in
  between, which is what clips on playback. True peak is measured on an
  oversampled copy of the signal and so catches that.

The gain is then whatever brings the loudness to the target, held back so
that the true peak stays under the ceiling. A ceiling one decibel below
full scale leaves room for the overshoot that lossy playback adds.

Each file is read twice. The first pass measures it, and the second pass
applies the gain and writes the result. Reading twice avoids holding a
whole recording in memory: an hour of stereo audio is well over a
gigabyte once it is uncompressed, and decoding is fast.

The measuring, the gain and the optional limiter all come from FFmpeg,
through the PyAV binding. Nothing here re-implements the loudness
standard; it drives the reference implementation and reads the numbers
back.

Nothing in this module depends on Qt, so it can be tested on its own.
"""

from __future__ import annotations

import enum
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from vox_verbatim.audio.library import is_supported_audio_file

_log = logging.getLogger(__name__)

#: Loudness targets outside this range are not useful for speech. The
#: quiet end is below anything a transcription service wants; the loud end
#: leaves no room for peaks.
MINIMUM_TARGET_LUFS = -40.0
MAXIMUM_TARGET_LUFS = -6.0
DEFAULT_TARGET_LUFS = -18.0

#: How close to full scale the loudest moment may come. Zero is full
#: scale, and going above it clips.
MINIMUM_CEILING_DBTP = -9.0
MAXIMUM_CEILING_DBTP = 0.0
DEFAULT_CEILING_DBTP = -1.0

#: A ceiling on the boost, so that a nearly silent recording is not raised
#: forty decibels into a wall of hiss. Raising the volume cannot improve
#: the ratio of speech to noise: it lifts both equally.
MINIMUM_MAXIMUM_GAIN_DB = 0.0
MAXIMUM_MAXIMUM_GAIN_DB = 60.0
DEFAULT_MAXIMUM_GAIN_DB = 30.0

#: The loudness meter reports this when it hears nothing at all. It is the
#: floor of the measurement, not a real reading, so a file that measures
#: it is left alone rather than being amplified by an absurd amount.
_SILENCE_LUFS = -70.0


@dataclass(frozen=True)
class OutputFormat:
    """One of the lossless formats an enhanced recording can be written in."""

    key: str
    """The name stored in the settings file."""

    label: str
    """What the user sees in the list of formats."""

    codec: str
    """The FFmpeg encoder that writes it."""

    extension: str
    """The file extension, including the dot."""


#: 16-bit WAV comes first because every transcription service accepts it.
#: FLAC holds exactly the same samples in roughly half the space, which
#: matters when the file has to be uploaded.
OUTPUT_FORMATS: tuple[OutputFormat, ...] = (
    OutputFormat("wav16", "WAV, 16-bit (accepted everywhere)", "pcm_s16le", ".wav"),
    OutputFormat("wav24", "WAV, 24-bit (larger files)", "pcm_s24le", ".wav"),
    OutputFormat("flac", "FLAC (lossless, about half the size)", "flac", ".flac"),
)

DEFAULT_OUTPUT_FORMAT = OUTPUT_FORMATS[0]


def output_format_for(key: str | None) -> OutputFormat:
    """Return the format saved under ``key``, or the default if it is unknown."""
    for output_format in OUTPUT_FORMATS:
        if output_format.key == key:
            return output_format
    return DEFAULT_OUTPUT_FORMAT


@dataclass(frozen=True)
class EnhanceOptions:
    """Everything the user can decide before a run starts."""

    output_folder: Path
    target_lufs: float = DEFAULT_TARGET_LUFS
    ceiling_dbtp: float = DEFAULT_CEILING_DBTP
    maximum_gain_db: float = DEFAULT_MAXIMUM_GAIN_DB
    use_limiter: bool = False
    """Whether peaks may be held down so that the target can still be reached.

    Without this, a recording with a few loud transients cannot be raised
    far, because the whole waveform has to stay under the ceiling. With it,
    a look-ahead limiter pulls those moments down and the rest of the
    recording reaches the target. That does change the shape of the
    waveform, which is why it is a choice and not the default.
    """

    output_format: OutputFormat = DEFAULT_OUTPUT_FORMAT
    replace_existing: bool = False

    def output_path_for(self, source: Path) -> Path:
        """Where the enhanced copy of ``source`` will be written."""
        return self.output_folder / (source.stem + self.output_format.extension)


class Outcome(enum.Enum):
    """How one file ended up."""

    ENHANCED = "enhanced"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class FileResult:
    """What happened to one file, in numbers and in words."""

    source: Path
    outcome: Outcome
    message: str
    """One sentence, written to be read aloud as well as seen."""

    output: Path | None = None
    input_lufs: float | None = None
    input_dbtp: float | None = None
    gain_db: float | None = None
    output_lufs: float | None = None
    output_dbtp: float | None = None

    @property
    def name(self) -> str:
        return self.source.name


class GainLimit(enum.Enum):
    """What stopped the gain short of the loudness target, if anything."""

    NONE = "none"
    CEILING = "the true-peak ceiling"
    MAXIMUM = "the maximum gain"


@dataclass(frozen=True)
class GainPlan:
    """How far a recording will be moved, and what held it back."""

    gain_db: float
    limited_by: GainLimit = GainLimit.NONE


def plan_gain(measured_lufs: float, measured_dbtp: float, options: EnhanceOptions) -> GainPlan:
    """Work out the gain to apply to a recording that measured these numbers.

    The wanted gain is simply the distance from where the recording is to
    the target. Two things can hold it back. The peak ceiling is a hard
    limit, because exceeding it clips; it is ignored only when the limiter
    is switched on, because the limiter then keeps the peaks in check. The
    maximum gain is the user's own limit on how far a quiet recording is
    dragged up.

    Neither limit applies to turning a recording *down*. A file already
    louder than the target is reduced to it, and one whose peaks are
    already over the ceiling is reduced until they are not, however far
    that is.
    """
    gain_db = options.target_lufs - measured_lufs
    limited_by = GainLimit.NONE

    if gain_db > options.maximum_gain_db:
        gain_db = options.maximum_gain_db
        limited_by = GainLimit.MAXIMUM

    if not options.use_limiter:
        headroom_db = options.ceiling_dbtp - measured_dbtp
        if gain_db > headroom_db:
            gain_db = headroom_db
            limited_by = GainLimit.CEILING

    return GainPlan(gain_db=gain_db, limited_by=limited_by)


#: Called with a fraction between 0 and 1 as one file is worked through.
ProgressCallback = Callable[[float], None]

#: Asked between decoded chunks whether the user has pressed Cancel.
CancelledCallback = Callable[[], bool]


class _Cancelled(Exception):
    """Raised inside a pass when the user asks to stop."""


def enhance_file(
    source: Path,
    options: EnhanceOptions,
    progress: ProgressCallback | None = None,
    cancelled: CancelledCallback | None = None,
) -> FileResult:
    """Measure one recording, raise it to the target, and write it out.

    Every foreseeable problem comes back as a :class:`FileResult` saying
    what went wrong, rather than as an exception. One unreadable file in a
    folder of twenty must not stop the other nineteen.
    """
    report = progress or (lambda _fraction: None)
    stop_requested = cancelled or (lambda: False)

    try:
        import av  # imported here so a missing PyAV is reported, not crashed on
    except ImportError:
        return FileResult(
            source=source,
            outcome=Outcome.FAILED,
            message=(
                f"{source.name} was not enhanced. The PyAV library is not installed, "
                "so audio cannot be decoded."
            ),
        )

    output = options.output_path_for(source)
    if output == source:
        return FileResult(
            source=source,
            outcome=Outcome.SKIPPED,
            message=(
                f"{source.name} was skipped. Writing it would replace the recording "
                "itself. Choose a different output folder or format."
            ),
        )
    # With the output folder set to the recordings folder, the enhanced
    # copy of meeting.m4a is meeting.wav, which may be another recording
    # rather than an earlier enhanced copy. Nothing on disk tells the two
    # apart, so a recording is never written over, even when replacing
    # existing files is switched on.
    if (
        output.exists()
        and _same_folder(output.parent, source.parent)
        and is_supported_audio_file(output)
    ):
        return FileResult(
            source=source,
            outcome=Outcome.SKIPPED,
            output=output,
            message=(
                f"{source.name} was skipped, because {output.name} is a recording in "
                "the same folder. Choose a different output folder."
            ),
        )
    if output.exists() and not options.replace_existing:
        return FileResult(
            source=source,
            outcome=Outcome.SKIPPED,
            output=output,
            message=(
                f"{source.name} was skipped, because {output.name} is already in the "
                "output folder. Switch on replacing existing files to write it again."
            ),
        )

    try:
        # The first pass covers the first half of this file's progress and
        # the second pass the other half, so the bar moves steadily rather
        # than jumping from nothing to everything.
        measured = _run_pass(
            source,
            filters=(),
            sink=None,
            progress=lambda fraction: report(fraction * 0.5),
            cancelled=stop_requested,
        )
    except _Cancelled:
        return _cancelled_result(source)
    except (OSError, IndexError, av.FFmpegError) as error:
        return _failed_result(source, "could not be read", error)

    if measured.lufs is None or measured.dbtp is None:
        return FileResult(
            source=source,
            outcome=Outcome.FAILED,
            message=(
                f"{source.name} was not enhanced, because its loudness could not be "
                "measured. The recording may be damaged or too short."
            ),
        )
    # A recording of pure silence has no peak at all, and the loudness
    # meter reports its own floor rather than a real reading. Raising
    # either of those by any amount still gives silence.
    if measured.lufs <= _SILENCE_LUFS or not math.isfinite(measured.dbtp):
        return FileResult(
            source=source,
            outcome=Outcome.SKIPPED,
            input_lufs=measured.lufs,
            input_dbtp=measured.dbtp,
            message=(
                f"{source.name} was skipped, because it is silent. There is nothing "
                "to raise."
            ),
        )

    plan = plan_gain(measured.lufs, measured.dbtp, options)
    filters = [("volume", f"volume={plan.gain_db:.4f}dB")]
    if options.use_limiter:
        # level=disabled stops the limiter making up the level it took
        # away, which would undo the gain that was just calculated.
        ceiling = 10 ** (options.ceiling_dbtp / 20.0)
        filters.append(("alimiter", f"limit={ceiling:.6f}:level=disabled"))

    # The file is written under a temporary name and only takes the real
    # name once it is complete. Writing straight to the real name would
    # empty an earlier enhanced copy the moment the file was opened, so a
    # run that was cancelled or failed part-way through would lose it.
    unfinished = _unfinished_path_for(output)
    try:
        options.output_folder.mkdir(parents=True, exist_ok=True)
        with av.open(str(unfinished), "w") as destination:
            written = _run_pass(
                source,
                filters=tuple(filters),
                sink=_Encoder(destination, options.output_format.codec),
                progress=lambda fraction: report(0.5 + fraction * 0.5),
                cancelled=stop_requested,
            )
        os.replace(unfinished, output)
    except _Cancelled:
        _remove_partial(unfinished)
        return _cancelled_result(source)
    except (OSError, IndexError, av.FFmpegError) as error:
        _remove_partial(unfinished)
        return _failed_result(source, "could not be written", error)

    report(1.0)
    return FileResult(
        source=source,
        outcome=Outcome.ENHANCED,
        output=output,
        input_lufs=measured.lufs,
        input_dbtp=measured.dbtp,
        gain_db=plan.gain_db,
        output_lufs=written.lufs,
        output_dbtp=written.dbtp,
        message=_describe(source, output, measured, plan, written),
    )


def enhance_files(
    sources: Iterable[Path],
    options: EnhanceOptions,
    progress: ProgressCallback | None = None,
    cancelled: CancelledCallback | None = None,
    on_start: Callable[[Path, int, int], None] | None = None,
    on_result: Callable[[FileResult], None] | None = None,
) -> list[FileResult]:
    """Work through several recordings, reporting progress across all of them.

    Each file takes an equal share of the progress, whatever its length.
    That is not strictly proportional to the work, but it needs only the
    number of files rather than the length of every one of them, and it
    still moves steadily rather than in jumps.

    ``on_start`` is called with the file, its number and how many there
    are, before each one begins. ``on_result`` is called with each result
    as it arrives, so a caller can report on the way rather than at the
    end.

    Two sources whose names differ only in extension, such as ``talk.m4a``
    and ``talk.mp3``, would both be written as ``talk.wav``, the second
    over the first. Those are found before anything is written, and every
    file in such a group is skipped and reported first. Skipping them all,
    rather than inventing a new name for one, keeps each enhanced copy
    named after its recording.
    """
    files = list(sources)
    results: list[FileResult] = []
    report = progress or (lambda _fraction: None)
    stop_requested = cancelled or (lambda: False)
    total = len(files) or 1

    clashes = _find_output_clashes(files, options)
    for source in files:
        if source in clashes:
            result = clashes[source]
            results.append(result)
            if on_result is not None:
                on_result(result)
    remaining = [source for source in files if source not in clashes]
    if clashes:
        report(len(clashes) / total)

    for index, source in enumerate(remaining, start=len(clashes)):
        if stop_requested():
            break
        if on_start is not None:
            on_start(source, index + 1, len(files))
        result = enhance_file(
            source,
            options,
            progress=lambda fraction, i=index: report((i + fraction) / total),
            cancelled=stop_requested,
        )
        results.append(result)
        if on_result is not None:
            on_result(result)
        if result.outcome == Outcome.CANCELLED:
            break
        report((index + 1) / total)
    return results


def _find_output_clashes(files: list[Path], options: EnhanceOptions) -> dict[Path, FileResult]:
    """Find the sources that would be written to the same output file.

    Returns a skipped result for every source that shares its output with
    at least one other source in the run, each naming the other files.
    """
    by_output: dict[str, list[Path]] = {}
    for source in files:
        key = _path_key(options.output_path_for(source))
        sharing = by_output.setdefault(key, [])
        if source not in sharing:
            sharing.append(source)

    clashes: dict[Path, FileResult] = {}
    for sharing in by_output.values():
        if len(sharing) < 2:
            continue
        output_name = options.output_path_for(sharing[0]).name
        for source in sharing:
            others = _join_names([other.name for other in sharing if other != source])
            clashes[source] = FileResult(
                source=source,
                outcome=Outcome.SKIPPED,
                message=(
                    f"{source.name} was skipped, because {others} in this run "
                    f"would also be written as {output_name}. Enhance them in "
                    "separate runs or choose a different format."
                ),
            )
    return clashes


def _join_names(names: list[str]) -> str:
    """Join file names into a phrase that reads naturally aloud."""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _path_key(path: Path) -> str:
    """A form of ``path`` that is equal for every spelling of the same file."""
    return os.path.normcase(os.path.abspath(path))


def _same_folder(first: Path, second: Path) -> bool:
    return _path_key(first) == _path_key(second)


# -- Reporting -----------------------------------------------------------


def _describe(source: Path, output: Path, measured, plan: GainPlan, written) -> str:
    """Write one sentence saying what was done to a recording."""
    direction = "raised" if plan.gain_db >= 0 else "reduced"
    sentence = (
        f"{source.name} measured {measured.lufs:.1f} LUFS and {measured.dbtp:.1f} dBTP, "
        f"and was {direction} by {abs(plan.gain_db):.1f} dB to "
        f"{_reading(written.lufs, 'LUFS')} and {_reading(written.dbtp, 'dBTP')}. "
        f"It was written as {output.name}."
    )
    if plan.limited_by != GainLimit.NONE:
        sentence += (
            f" The target loudness was not reached, because {plan.limited_by.value} "
            "held the gain back."
        )
    return sentence


def _reading(value: float | None, unit: str) -> str:
    return "an unknown level" if value is None else f"{value:.1f} {unit}"


def _failed_result(source: Path, what_happened: str, error: BaseException) -> FileResult:
    _log.warning("Enhancing %s failed.", source, exc_info=error)
    reason = str(error).strip() or error.__class__.__name__
    return FileResult(
        source=source,
        outcome=Outcome.FAILED,
        message=f"{source.name} {what_happened}. {reason}.",
    )


def _cancelled_result(source: Path) -> FileResult:
    return FileResult(
        source=source,
        outcome=Outcome.CANCELLED,
        message=f"{source.name} was stopped before it finished, and was not written.",
    )


def _unfinished_path_for(output: Path) -> Path:
    """Where ``output`` is written until it is complete.

    It sits in the same folder, so that moving it into place is a rename
    rather than a copy, and it keeps the extension, because FFmpeg picks
    the container from the extension.
    """
    return output.with_name(f"{output.stem}.unfinished{output.suffix}")


def _remove_partial(output: Path) -> None:
    """Throw away a half-written file, so nothing is left that looks finished."""
    try:
        output.unlink(missing_ok=True)
    except OSError:
        _log.debug("The unfinished file %s could not be removed.", output, exc_info=True)


# -- Driving FFmpeg ------------------------------------------------------


@dataclass
class _Readings:
    """The loudness meter's last word on a recording."""

    lufs: float | None = None
    dbtp: float | None = None


class _Encoder:
    """Writes filtered audio into an output file.

    The filter chain hands back audio in whatever form it finds convenient,
    in blocks of whatever size fall out of decoding. The encoder is given a
    queue in between so that it always receives whole, evenly sized blocks,
    and it converts the sample format itself.
    """

    def __init__(self, destination, codec: str) -> None:
        self._destination = destination
        self._codec = codec
        self._stream = None
        self._queue = None

    def write(self, frame) -> None:
        import av

        if self._stream is None:
            self._stream = self._destination.add_stream(
                self._codec, rate=frame.sample_rate, layout=frame.layout.name
            )
            self._queue = av.audio.fifo.AudioFifo()
        # The queue insists on timestamps that follow one another, and the
        # filter chain does not always give them, so it works them out.
        frame.pts = None
        self._queue.write(frame)
        block = self._queue.read()
        if block is not None:
            self._encode(block)

    def close(self) -> None:
        if self._stream is None:
            return
        remainder = self._queue.read()
        if remainder is not None:
            self._encode(remainder)
        self._encode(None)

    def _encode(self, block) -> None:
        for packet in self._stream.encode(block):
            self._destination.mux(packet)


def _run_pass(
    source: Path,
    filters: tuple[tuple[str, str], ...],
    sink: _Encoder | None,
    progress: ProgressCallback,
    cancelled: CancelledCallback,
) -> _Readings:
    """Read a recording through a filter chain, measuring it on the way out.

    The loudness meter is always the last filter, so what it reports is
    what the file will actually contain. That makes the second pass a
    verification of the first rather than a repetition of it: the numbers
    in the report are measured from the finished audio, not predicted.
    """
    import av
    from av.filter import Graph

    readings = _Readings()
    with av.open(str(source)) as container:
        stream = container.streams.audio[0]
        graph = Graph()
        chain = [graph.add_abuffer(template=stream)]
        for name, arguments in filters:
            chain.append(graph.add(name, arguments))
        chain.append(graph.add("ebur128", "peak=true:metadata=1:framelog=quiet"))
        chain.append(graph.add("abuffersink"))
        for upstream, downstream in zip(chain, chain[1:]):
            upstream.link_to(downstream)
        graph.configure()

        seconds = _duration_seconds(container, stream)

        def drain() -> None:
            while True:
                try:
                    frame = graph.pull()
                except (av.BlockingIOError, av.EOFError):
                    return
                _read_meter(frame, readings)
                if sink is not None:
                    sink.write(frame)

        for frame in container.decode(stream):
            if cancelled():
                raise _Cancelled
            graph.push(frame)
            drain()
            if seconds and frame.time is not None:
                progress(min(1.0, frame.time / seconds))
        graph.push(None)
        drain()

    if sink is not None:
        sink.close()
    progress(1.0)
    return readings


def _duration_seconds(container, stream) -> float | None:
    """How long the recording is, for reporting progress against."""
    import av

    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / av.time_base
    return None


def _read_meter(frame, readings: _Readings) -> None:
    """Take the latest loudness reading off a frame, if it carries one.

    The meter attaches its running figures to a frame every hundred
    milliseconds. Both are cumulative, so the last one seen covers the
    whole recording.
    """
    metadata = frame.metadata
    if not metadata:
        return
    loudness = _as_float(metadata.get("lavfi.r128.I"))
    if loudness is not None and math.isfinite(loudness):
        readings.lufs = loudness
    peak = _as_float(metadata.get("lavfi.r128.true_peak"))
    if peak is not None:
        # The meter reports the peak as a plain amplitude, where one is
        # full scale. Decibels are what the ceiling is expressed in.
        readings.dbtp = 20.0 * math.log10(peak) if peak > 0 else -math.inf


def _as_float(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None
