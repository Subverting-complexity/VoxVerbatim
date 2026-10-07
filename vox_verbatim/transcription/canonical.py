"""The one recording, and the one clock, that every timestamp is measured against.

A transcript is a list of words with times attached, and a time on its own
means nothing at all. It means something only against a particular audio
file. Change the file underneath it, by trimming a second of silence off the
front or by re-encoding it with a codec that pads the start, and every time
in the transcript quietly becomes wrong. Nothing complains, nothing looks
broken, and the words simply no longer sit where they are said to sit.

This module exists to make that impossible. It settles one file as the
canonical recording, measures it once, and hands everything else a
description of it. Chunks sent to one service, windows cut for a second
opinion and clips played back in the review window are all cut from that one
file and all carry an exact offset into it, so a word can always be traced
back to the moment it was spoken.

Three rules follow from that, and they are worth stating plainly.

The original recording is never written to. It is the user's file and it is
evidence. Where a conversion is needed, a separate copy is written into the
recording's transcript folder and the original is left exactly as it was.

Where a copy is made, the copy becomes the canonical file, and every
measurement is taken from the copy rather than from the original. This is
the part that is easy to get wrong and expensive to discover later: a
conversion can change the duration by a few milliseconds, and if the
duration on record came from the original then every time in the transcript
is being interpreted against a file that is a little longer or shorter than
the one the services actually heard.

Everything this module writes, whether a canonical copy or a window cut for
escalation, is written as 16-bit WAV or as FLAC. It is never written back
into the codec it came from. There are two reasons, and both matter.

The first is quality: re-encoding speech into a lossy format would add a
generation of coding damage before the services ever hear it, which is a
strange thing to do to audio whose whole purpose is to be listened to
carefully.

The second is that it would not reliably work. Which codecs FFmpeg can
*decode* and which it can *encode* are different lists, and the copy of
FFmpeg that arrives inside PyAV is not built the same way everywhere. Some
builds read AAC perfectly and cannot write it. Since the recordings this
application is for are ``.m4a``, which is AAC, a window that tried to stay
in the format of its source would fail on exactly the files the user has
most of. The output format is therefore chosen for what can certainly be
written rather than for what the input happened to be.

Cutting a window out of the canonical file is the single easiest place in
the whole system to introduce a timestamp bug, so :func:`cut_window` reports
what it actually cut rather than what it was asked for. Ask for five seconds
that begin one second before the recording does, and you get four seconds
and an offset of zero, said out loud, instead of four seconds labelled as
though they began a second early.

Nothing here depends on Qt, so it can be tested on its own.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from vox_verbatim.audio.enhance import OUTPUT_FORMATS, OutputFormat
from vox_verbatim.transcription.model import AudioSpan, CanonicalAudio
from vox_verbatim.transcription.providers.base import ProviderCapabilities

_log = logging.getLogger(__name__)

#: 16-bit WAV is the default form for a canonical copy because every service
#: accepts it and because its header states the length in samples, so the
#: duration measured from it is exact rather than estimated.
DEFAULT_COPY_FORMAT: OutputFormat = OUTPUT_FORMATS[0]

#: FLAC holds the identical samples in roughly half the space. It is used
#: where the size of the file is the problem being solved.
COMPACT_COPY_FORMAT: OutputFormat = OUTPUT_FORMATS[2]

#: The containers to assume a service will take when the caller has not said.
#: It matches the default in :class:`ProviderCapabilities`, and the real set
#: should come from the services actually being used.
DEFAULT_ACCEPTED_CONTAINERS: frozenset[str] = frozenset({"wav", "mp3", "flac", "m4a"})

#: How far before a wanted window the reader seeks before it starts decoding.
#: Seeking lands on a packet boundary at or before the requested point, and a
#: couple of seconds of margin covers the largest packets in ordinary use.
_SEEK_MARGIN_SECONDS = 2.0

#: Speech in FLAC usually comes out near half the size of the same audio as
#: PCM. Planning is done with a deliberately pessimistic figure, because a
#: chunk that turns out smaller than planned costs nothing and one that turns
#: out larger than a service will accept costs a failed request.
_FLAC_PESSIMISTIC_RATIO = 0.75

#: Roughly what a container and its headers add. A few kilobytes is nothing
#: against a real recording, and counting it keeps a very short clip from
#: being estimated at less than it takes on disk.
_CONTAINER_OVERHEAD_BYTES = 8 * 1024


class CanonicalAudioError(Exception):
    """A recording could not be read, or a canonical copy could not be written.

    This is raised rather than returned, because a recording with no
    canonical form has no transcript either and there is nothing sensible to
    carry on with. The message is a plain sentence, so the pipeline can put
    it in front of the user as it stands.
    """


@dataclass(frozen=True)
class AudioProbe:
    """What a file turned out to contain when it was opened and measured."""

    path: Path
    duration: float
    sample_rate: int
    channels: int
    container: str
    """The short name a service would know the format by, such as ``wav``."""

    codec: str
    size_bytes: int

    @property
    def sample_count(self) -> int:
        """How many samples per channel the recording holds.

        Times and sample positions are converted through this rather than
        through the container's own timestamps wherever a cut is made, so
        that the two can never drift apart by a sample.
        """
        return round(self.duration * self.sample_rate)


@dataclass(frozen=True)
class CopyDecision:
    """Whether a canonical copy is needed, in what form, and why."""

    needed: bool
    output_format: OutputFormat
    reason: str
    """One sentence, written to be read aloud as well as logged."""


@dataclass(frozen=True)
class AudioClip:
    """A piece of the canonical recording, written out as its own file.

    ``span`` is where the clip really came from, after any clamping, and
    ``canonical_offset`` is the number that converts a time inside the clip
    back to canonical time. They are the same number by definition; both are
    here because the request record and the adapters name them separately,
    and an adapter should never have to work out which field it wants.
    """

    path: Path
    span: AudioSpan
    canonical_offset: float
    sample_rate: int
    channels: int
    size_bytes: int
    clamped: bool
    """Whether the recording ran out before the window that was asked for did."""

    @property
    def duration(self) -> float:
        return self.span.duration


# -- Measuring a recording -----------------------------------------------


def probe_audio(path: Path) -> AudioProbe:
    """Open a recording and read exactly what it is.

    The duration is taken from the stream header where there is one, because
    that is both exact and instant for the lossless formats a canonical copy
    is written in. A file whose header does not state a length is decoded
    from beginning to end and its samples counted, which is slow but is the
    only honest answer available; guessing from the file size would be wrong
    by whatever the bitrate varied by.
    """
    import av

    try:
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            sample_rate = int(stream.sample_rate or 0)
            if sample_rate <= 0:
                raise CanonicalAudioError(
                    f"{path.name} does not say what sample rate it was recorded at, "
                    "so it cannot be used as canonical audio."
                )
            duration = _header_duration(container, stream)
            if duration is None:
                duration = _decoded_sample_count(container, stream) / sample_rate
            return AudioProbe(
                path=path,
                duration=float(duration),
                sample_rate=sample_rate,
                channels=int(stream.channels or 1),
                container=_container_name(path, container),
                codec=str(stream.codec_context.name),
                size_bytes=path.stat().st_size,
            )
    except CanonicalAudioError:
        raise
    except IndexError as error:
        raise CanonicalAudioError(f"{path.name} holds no audio at all.") from error
    except (OSError, av.FFmpegError) as error:
        reason = str(error).strip() or error.__class__.__name__
        raise CanonicalAudioError(f"{path.name} could not be read. {reason}.") from error


def _header_duration(container, stream) -> float | None:
    """The length the file claims, or None where it does not say."""
    import av

    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / av.time_base
    return None


def _decoded_sample_count(container, stream) -> int:
    """Count the samples in a file whose header does not state its length."""
    container.seek(0)
    return sum(frame.samples for frame in container.decode(stream))


def _container_name(path: Path, container) -> str:
    """The short name for a file's format, as a service would recognise it.

    Services identify audio by its file extension and media type rather than
    by which demuxer FFmpeg chose, and one demuxer covers several extensions:
    an ``.m4a`` file is handled by the same code as ``.mp4``. So the
    extension is the answer whenever the demuxer agrees that it could be one
    of the formats it handles, and the demuxer's own first name is the answer
    when it does not, because then the extension is a lie.
    """
    names = [name.strip().lower() for name in (container.format.name or "").split(",")]
    names = [name for name in names if name]
    suffix = path.suffix.lower().lstrip(".")
    if suffix and suffix in names:
        return suffix
    return names[0] if names else suffix


def lossless_bytes_per_second(
    sample_rate: int, channels: int, output_format: OutputFormat = DEFAULT_COPY_FORMAT
) -> float:
    """How much one second of this audio takes in the given lossless format."""
    bytes_per_sample = _BYTES_PER_SAMPLE.get(output_format.key, 2.0)
    return sample_rate * channels * bytes_per_sample


#: PCM is exact. FLAC is not, because how far speech compresses depends on
#: the speech, so the pessimistic ratio stands in for it during planning.
_BYTES_PER_SAMPLE: dict[str, float] = {
    "wav16": 2.0,
    "wav24": 3.0,
    "flac": 2.0 * _FLAC_PESSIMISTIC_RATIO,
}


def estimate_lossless_bytes(
    duration: float,
    sample_rate: int,
    channels: int,
    output_format: OutputFormat = DEFAULT_COPY_FORMAT,
) -> int:
    """Estimate the size of a piece of audio written out losslessly.

    This is what lets a whole chunking plan be made before a single chunk is
    written. For PCM the answer is arithmetic and exact; for FLAC it is
    deliberately an overestimate, so that a plan made from it stays inside a
    service's limit even when the audio compresses badly.
    """
    seconds = max(0.0, duration)
    per_second = lossless_bytes_per_second(sample_rate, channels, output_format)
    return int(_CONTAINER_OVERHEAD_BYTES + seconds * per_second)


def containers_accepted_by_all(capabilities: Iterable[ProviderCapabilities]) -> frozenset[str]:
    """The formats every one of these services will take.

    A canonical file has to suit all of them at once, because there is only
    one of it. Where the services have nothing in common, the caller is told
    so by getting an empty set back rather than by being handed a format one
    of them will reject.
    """
    accepted: frozenset[str] | None = None
    for capability in capabilities:
        names = frozenset(name.lower() for name in capability.accepted_containers)
        accepted = names if accepted is None else (accepted & names)
    return accepted if accepted is not None else frozenset()


# -- Deciding whether a copy is needed ------------------------------------


def decide_canonical_copy(
    probe: AudioProbe,
    accepted_containers: Iterable[str] = DEFAULT_ACCEPTED_CONTAINERS,
    maximum_bytes: int | None = None,
    preferred_format: OutputFormat = DEFAULT_COPY_FORMAT,
) -> CopyDecision:
    """Work out whether this recording can be used as it is.

    Two things force a copy. The first is a container no service will take,
    which is simply a wall. The second is size, and it is subtler than it
    looks: converting a lossy recording into a lossless one always makes it
    bigger, so being over a limit is almost never a reason to convert. It is
    a reason only when the recording is already uncompressed, where writing
    it as FLAC genuinely halves it. Everything else that is too big for a
    service is dealt with by chunking, which is per service and does not
    touch the canonical file.
    """
    accepted = frozenset(name.lower() for name in accepted_containers)
    if probe.container not in accepted:
        listed = ", ".join(sorted(accepted)) or "nothing"
        return CopyDecision(
            needed=True,
            output_format=preferred_format,
            reason=(
                f"{probe.path.name} is a {probe.container} file, and the services accept "
                f"{listed}. A {preferred_format.label} copy will be made for them to work from."
            ),
        )

    if maximum_bytes is not None and probe.size_bytes > maximum_bytes:
        compact = estimate_lossless_bytes(
            probe.duration, probe.sample_rate, probe.channels, COMPACT_COPY_FORMAT
        )
        if compact < probe.size_bytes:
            return CopyDecision(
                needed=True,
                output_format=COMPACT_COPY_FORMAT,
                reason=(
                    f"{probe.path.name} is larger than the services will accept, and it is "
                    "stored uncompressed. A lossless FLAC copy holds the same audio in "
                    "about half the space, so that is what will be used."
                ),
            )
        # Making this file smaller would mean throwing audio away, which is
        # not something to do quietly behind the user's back. Per-service
        # chunking is the answer, and it leaves this file alone.
        return CopyDecision(
            needed=False,
            output_format=preferred_format,
            reason=(
                f"{probe.path.name} will be used as it is. It is too large for some services "
                "to take whole, so those services will be sent it in pieces."
            ),
        )

    return CopyDecision(
        needed=False,
        output_format=preferred_format,
        reason=(
            f"{probe.path.name} will be used as it is, because every service accepts "
            f"{probe.container} files."
        ),
    )


def canonical_copy_path(
    source: Path, folder: Path, output_format: OutputFormat = DEFAULT_COPY_FORMAT
) -> Path:
    """Where the canonical copy of a recording is written.

    Named after the recording rather than simply ``canonical``, so that a
    folder holding the work for more than one recording stays readable and so
    that the file is recognisable on its own if it is ever found loose.
    """
    return folder / f"{source.stem}-canonical{output_format.extension}"


def prepare_canonical_audio(
    source: Path,
    folder: Path,
    accepted_containers: Iterable[str] = DEFAULT_ACCEPTED_CONTAINERS,
    maximum_bytes: int | None = None,
    preferred_format: OutputFormat = DEFAULT_COPY_FORMAT,
) -> CanonicalAudio:
    """Settle the canonical recording for one file, converting it if it must.

    The original is opened and measured, and where it can be used as it is,
    it is: no copy, no conversion, and the canonical path is the original
    path.

    Where a copy has to be made, the copy is written and then **measured
    again from the copy**. That second measurement is not a formality. Every
    time in the finished transcript is interpreted against the canonical
    file, so the duration, sample rate and channel count on record must be
    the copy's own. Carrying the original's numbers over would mean
    describing one file while the services listen to another, and a
    difference of a few milliseconds between them would put every word
    slightly out of place with nothing to show why.
    """
    probe = probe_audio(source)
    decision = decide_canonical_copy(probe, accepted_containers, maximum_bytes, preferred_format)
    if not decision.needed:
        _log.info("%s", decision.reason)
        return CanonicalAudio(
            path=str(source),
            original_path=str(source),
            duration=probe.duration,
            sample_rate=probe.sample_rate,
            channels=probe.channels,
            size_bytes=probe.size_bytes,
            container=probe.container,
            is_copy=False,
        )

    _log.info("%s", decision.reason)
    destination = canonical_copy_path(source, folder, decision.output_format)
    folder.mkdir(parents=True, exist_ok=True)
    # The copy is derived from the original and holds no decisions of its
    # own, so writing it again over an older one is always safe and always
    # gives the same file.
    _copy_samples(source, destination, decision.output_format.codec, 0, None)

    copy = probe_audio(destination)
    return CanonicalAudio(
        path=str(destination),
        original_path=str(source),
        duration=copy.duration,
        sample_rate=copy.sample_rate,
        channels=copy.channels,
        size_bytes=copy.size_bytes,
        container=copy.container,
        is_copy=True,
    )


# -- Cutting a window out -------------------------------------------------


def cut_window(
    canonical: CanonicalAudio,
    span: AudioSpan,
    destination: Path | None = None,
    output_format: OutputFormat = DEFAULT_COPY_FORMAT,
    sample_rate: int | None = None,
    channels: int | None = None,
) -> AudioClip:
    """Cut one window out of the canonical recording into its own file.

    This is what escalation runs on: a few seconds around a disputed word,
    with enough speech either side for a service to make sense of it. What
    comes back is a file and, more importantly, the offset that converts the
    times the service reports inside that file back to canonical time.

    The window is cut on sample boundaries rather than by asking a decoder to
    approximate a seek, and the returned span is measured from the samples
    that were actually written. That matters at both ends of a recording. A
    window that begins before the recording does begins at zero, and says so;
    a window that runs past the end stops at the end, and says so. In both
    cases the offset describes the audio in the file rather than the audio
    that was requested, because the offset is what every returned time is
    added to, and an offset describing a window that was never cut would move
    every word in the clip by the difference.

    The clip is written as 16-bit WAV unless another lossless format is
    asked for, whatever the canonical recording is. It is never written back
    into the canonical file's own codec, both because that would add coding
    damage and because the FFmpeg inside PyAV cannot necessarily encode it:
    the ``.m4a`` recordings this application is chiefly for are AAC, and
    several builds read AAC without being able to write it.

    Pass ``destination`` to choose where the clip goes. Without one a
    temporary file is made, and the caller is responsible for deleting it
    once the service has answered.

    Pass ``sample_rate`` or ``channels`` to have the clip written at a lower
    rate or with fewer channels than the canonical recording. The window is
    still chosen in the canonical recording's own samples, so the offset and
    the span are exactly what they would have been without the conversion;
    only the audio inside the file is resampled or folded to mono on its way
    out. Neither value is ever raised above the recording's own, because a
    clip carrying more samples than the recording it came from is bigger for
    nothing. The clip reports the rate and channel count it was actually
    written with.
    """
    source = Path(canonical.path)
    rate = canonical.sample_rate
    total_samples = round(canonical.duration * rate)
    output_rate = rate if sample_rate is None else max(1, min(rate, int(sample_rate)))
    output_channels = (
        canonical.channels if channels is None else max(1, min(canonical.channels, int(channels)))
    )

    requested_start = round(span.start * rate)
    requested_end = round(span.end * rate)
    start_sample = min(max(0, requested_start), total_samples)
    end_sample = min(max(start_sample, requested_end), total_samples)
    clamped = start_sample != requested_start or end_sample != requested_end

    if destination is None:
        handle, name = tempfile.mkstemp(
            prefix=f"{source.stem}-window-", suffix=output_format.extension
        )
        os.close(handle)
        destination = Path(name)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)

    written = _copy_samples(
        source,
        destination,
        output_format.codec,
        start_sample,
        end_sample - start_sample,
        output_rate=output_rate,
        output_channels=output_channels,
    )
    if written < end_sample - start_sample:
        # The recording ran out earlier than its header said it would. What
        # is in the file is the truth, so the span shrinks to match rather
        # than the clip claiming audio it does not contain.
        end_sample = start_sample + written
        clamped = True

    return AudioClip(
        path=destination,
        span=AudioSpan(start_sample / rate, end_sample / rate),
        canonical_offset=start_sample / rate,
        sample_rate=output_rate,
        channels=output_channels,
        size_bytes=destination.stat().st_size if destination.exists() else 0,
        clamped=clamped,
    )


class _SeekOvershot(Exception):
    """The decoder landed after the wanted audio, so the read must start over."""


def _copy_samples(
    source: Path,
    destination: Path,
    codec: str,
    start_sample: int,
    wanted_samples: int | None,
    output_rate: int | None = None,
    output_channels: int | None = None,
) -> int:
    """Write ``wanted_samples`` samples of ``source``, starting at ``start_sample``.

    Sample counting, rather than the container's timestamps, is what makes
    this exact. Frames come out of the decoder in whatever sizes the format
    happens to use, so they are poured into a queue and taken out again in
    exactly the counts wanted: the unwanted head is read off and thrown away,
    and then precisely the wanted number of samples is read and encoded.

    Reading a window three hours into a recording from the very beginning
    would be needlessly slow, so the reader seeks to shortly before the
    window first. Seeking lands at or before the requested point, never
    after, but a damaged index can break that promise, and a seek that
    overshot would silently cut the wrong audio. So the position of the first
    frame decoded is checked, and a read that started too late is thrown away
    and done again from the beginning of the file.

    ``start_sample`` and ``wanted_samples`` are always counted in the
    source's own samples, whatever rate the output is written at. Where an
    ``output_rate`` or ``output_channels`` is given and differs from the
    source, the samples are converted on their way into the encoder, after
    the counting has been done, so a conversion can never move the cut.

    Returns the number of source samples actually written, which is less
    than asked for only when the recording ended first.
    """
    try:
        return _copy_samples_once(
            source,
            destination,
            codec,
            start_sample,
            wanted_samples,
            seek=start_sample > 0,
            output_rate=output_rate,
            output_channels=output_channels,
        )
    except _SeekOvershot:
        _log.debug("Seeking in %s overshot, so it is being read from the start.", source)
        return _copy_samples_once(
            source,
            destination,
            codec,
            start_sample,
            wanted_samples,
            seek=False,
            output_rate=output_rate,
            output_channels=output_channels,
        )


def _copy_samples_once(
    source: Path,
    destination: Path,
    codec: str,
    start_sample: int,
    wanted_samples: int | None,
    seek: bool,
    output_rate: int | None = None,
    output_channels: int | None = None,
) -> int:
    import av

    try:
        with av.open(str(source)) as container, av.open(str(destination), "w") as output:
            stream = container.streams.audio[0]
            rate = int(stream.sample_rate)
            if seek and stream.time_base is not None:
                target = max(0.0, start_sample / rate - _SEEK_MARGIN_SECONDS)
                container.seek(int(target / stream.time_base), stream=stream, backward=True)

            queue = av.audio.fifo.AudioFifo()
            # The output stream is created before anything is decoded, so
            # that a window holding no samples at all still comes out as a
            # valid, empty audio file rather than as a file with no header.
            layout = _layout_name(stream)
            encoder = _Encoder(
                output,
                codec,
                rate,
                layout,
                output_rate=output_rate,
                output_layout=_layout_for_channels(output_channels, layout),
            )
            position: int | None = None if seek else 0
            written = 0

            for frame in container.decode(stream):
                if position is None:
                    # Where the seek actually landed, in samples. Everything
                    # after this is counted rather than timed, so this is the
                    # only container timestamp the cut depends on.
                    position = round((frame.time or 0.0) * rate)
                    if position > start_sample:
                        raise _SeekOvershot
                frame.pts = None
                queue.write(frame)

                while True:
                    if position < start_sample:
                        discard = min(start_sample - position, queue.samples)
                        if discard <= 0:
                            break
                        block = queue.read(discard)
                        if block is None:
                            break
                        position += block.samples
                        continue
                    if wanted_samples is not None and written >= wanted_samples:
                        break
                    take = queue.samples
                    if wanted_samples is not None:
                        take = min(wanted_samples - written, take)
                    if take <= 0:
                        break
                    block = queue.read(take)
                    if block is None:
                        break
                    written += block.samples
                    encoder.write(block)

                if wanted_samples is not None and written >= wanted_samples:
                    break

            encoder.close()
            return written
    except _SeekOvershot:
        raise
    except IndexError as error:
        raise CanonicalAudioError(f"{source.name} holds no audio at all.") from error
    except (OSError, av.FFmpegError) as error:
        reason = str(error).strip() or error.__class__.__name__
        raise CanonicalAudioError(
            f"A piece of {source.name} could not be written to {destination.name}. {reason}."
        ) from error


#: What to call a channel arrangement that the file did not name. Only the
#: two that occur in speech recordings are listed, which is all this
#: application has ever been handed.
_LAYOUT_BY_CHANNEL_COUNT = {1: "mono", 2: "stereo"}


def _layout_name(stream) -> str:
    """The name of the channel arrangement to write an output stream with.

    A container does not have to record which channel is which, and a plain
    WAV file does not: read one back and its layout is reported as
    "1 channels", which is a description of the file rather than the name of
    a layout. Handing that description to an encoder makes it refuse to open
    at all, with nothing more helpful than an invalid-argument error, and the
    failure looks like a broken codec rather than a missing label. So where
    the file did not say, the standard arrangement for that many channels is
    used instead.
    """
    name = stream.layout.name if stream.layout is not None else ""
    if name and "channel" not in name:
        return name
    return _LAYOUT_BY_CHANNEL_COUNT.get(int(stream.channels or 1), name or "mono")


def _layout_for_channels(channels: int | None, source_layout: str) -> str | None:
    """The layout name to write with, or None to keep the source's own.

    Only a reduction to mono is ever asked for, because the one reason to
    change the channel count is to make a chunk smaller for a service that
    will fold it to mono anyway.
    """
    if channels is None:
        return None
    name = _LAYOUT_BY_CHANNEL_COUNT.get(int(channels))
    if name is None or name == source_layout:
        return None
    return name


class _Encoder:
    """Writes blocks of samples into an output file.

    By default the sample rate and the channel arrangement are taken from the
    recording being read and are never changed. A canonical copy that quietly
    became mono, or that was resampled, would no longer be the recording the
    user handed over.

    A chunk for a service is different. Every speech service folds what it is
    sent to 16 kHz mono before it listens, so a chunk written at 48 kHz
    stereo carries six times the bytes for nothing, and the bytes are the
    reason the recording had to be chunked at all. So the encoder can be
    asked to write at a lower rate or with fewer channels, and then it
    converts each block on its way in. The conversion happens after the
    samples have been counted out upstream, which is what keeps a resampled
    chunk starting at exactly the canonical time it claims.
    """

    def __init__(
        self,
        destination,
        codec: str,
        rate: int,
        layout: str,
        output_rate: int | None = None,
        output_layout: str | None = None,
    ) -> None:
        import av

        self._destination = destination
        wanted_rate = rate if output_rate is None or output_rate >= rate else int(output_rate)
        wanted_layout = output_layout or layout
        self._stream = destination.add_stream(codec, rate=wanted_rate, layout=wanted_layout)
        self._resampler = None
        if wanted_rate != rate or wanted_layout != layout:
            # The resampler is also asked for the encoder's own sample
            # format, so the frames it hands out can go straight in.
            self._resampler = av.AudioResampler(
                format=self._stream.format.name, layout=wanted_layout, rate=wanted_rate
            )

    def write(self, block) -> None:
        # The queue upstream hands out blocks of the size that was asked for
        # rather than the size the encoder wants, and timestamps that do not
        # follow the encoder's own clock, so the encoder is left to work both
        # out for itself.
        block.pts = None
        if self._resampler is None:
            self._encode(block)
            return
        for converted in self._resampler.resample(block):
            converted.pts = None
            self._encode(converted)

    def close(self) -> None:
        if self._resampler is not None:
            # A resampler holds back a few samples until it is told the
            # audio has ended; without the flush the last fraction of a
            # second of every chunk would be missing.
            for converted in self._resampler.resample(None):
                converted.pts = None
                self._encode(converted)
        for packet in self._stream.encode(None):
            self._destination.mux(packet)

    def _encode(self, frame) -> None:
        for packet in self._stream.encode(frame):
            self._destination.mux(packet)
