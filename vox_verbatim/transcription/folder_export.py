"""Exporting transcripts to a folder the person chooses.

Each recording keeps its readable files in its own ``.transcript`` folder,
where nobody goes looking to share or file them. The export copies them to
one folder, named after the recording, so ``interview.m4a`` gives
``interview - transcript.txt``, ``interview - smooth transcript.txt`` and
``interview - review report.md``.

The transcript and the review report are written fresh from the saved
transcript, so every correction made in the review window is in them, and
the copies in the recording's own exports folder are brought up to date at
the same time. The smooth transcript is made by the language model and
cannot be rebuilt here, so it is copied as it is. A smooth transcript made
before the latest corrections is out of date; :func:`out_of_date_smooth`
lists those, so the person can have them made again before exporting, and
the summary names any that went out as they were.

The work is in two steps. :func:`plan_export` works out which files would be
written and which recordings have nothing to give, without writing anything,
so the caller can ask the person once about files already in the folder.
:func:`run_export` then writes them. Nothing here depends on Qt.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from vox_verbatim.transcription.exports import (
    REPORT_EXPORT_NAME,
    SMOOTH_EXPORT_NAME,
    TEXT_EXPORT_NAME,
    render_plain_text,
    render_review_report,
    write_exports,
)
from vox_verbatim.transcription.smoothing import smooth_is_out_of_date
from vox_verbatim.transcription.store import TranscriptStore

_log = logging.getLogger(__name__)


class ExportKind(Enum):
    """One of the three files a recording can export."""

    TRANSCRIPT = "transcript"
    SMOOTH_TRANSCRIPT = "smooth transcript"
    REVIEW_REPORT = "review report"

    @property
    def extension(self) -> str:
        return ".md" if self is ExportKind.REVIEW_REPORT else ".txt"

    @property
    def own_export_name(self) -> str:
        """The name of this file in the recording's own exports folder."""
        return {
            ExportKind.TRANSCRIPT: TEXT_EXPORT_NAME,
            ExportKind.SMOOTH_TRANSCRIPT: SMOOTH_EXPORT_NAME,
            ExportKind.REVIEW_REPORT: REPORT_EXPORT_NAME,
        }[self]


def export_file_name(recording: Path, kind: ExportKind) -> str:
    """The exported file's name: the recording's name without its audio extension."""
    return f"{recording.stem} - {kind.value}{kind.extension}"


@dataclass(frozen=True)
class ExportFile:
    """One file to write: which recording, which kind, and where."""

    recording: Path
    kind: ExportKind
    target: Path


@dataclass(frozen=True)
class SkippedFile:
    """A file a recording could not give, and why."""

    recording_name: str
    reason: str


@dataclass
class ExportPlan:
    folder: Path
    files: list[ExportFile] = field(default_factory=list)
    skipped: list[SkippedFile] = field(default_factory=list)
    out_of_date: list[Path] = field(default_factory=list)
    """Recordings whose smooth transcript is exported although it is older
    than the corrections. Filled only when the smooth transcript is chosen."""

    def existing(self) -> list[Path]:
        """The files that would replace one already in the folder."""
        return [item.target for item in self.files if item.target.exists()]


@dataclass
class ExportResult:
    folder: Path
    written: list[Path] = field(default_factory=list)
    kept: list[Path] = field(default_factory=list)
    """Files already in the folder that the person chose not to replace."""
    skipped: list[SkippedFile] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    """Sentences saying what could not be written, and why."""
    out_of_date: list[str] = field(default_factory=list)
    """The names of recordings whose smooth transcript was written although
    it is older than the corrections."""


StoreFor = Callable[[Path], TranscriptStore]


def is_out_of_date(store: TranscriptStore) -> bool:
    """Whether this recording's smooth transcript is older than its corrections.

    A transcript that cannot be read cannot be compared, and is not called
    out of date here; its transcript and report are reported as unreadable
    when the export reaches them.
    """
    if not (store.exports_folder / SMOOTH_EXPORT_NAME).is_file():
        return False
    transcript = store.load()
    return transcript is not None and smooth_is_out_of_date(store, transcript)


def out_of_date_smooth(recordings: Sequence[Path], store_for: StoreFor) -> list[Path]:
    """The recordings whose smooth transcript is older than their corrections."""
    return [recording for recording in recordings if is_out_of_date(store_for(recording))]


def plan_export(
    recordings: Sequence[Path],
    folder: Path,
    kinds: Iterable[ExportKind],
    store_for: StoreFor,
) -> ExportPlan:
    """Work out what an export would write. Writes nothing.

    A recording with no transcription gives no transcript and no report, and
    one with no smooth transcript gives no smooth transcript. Each is listed
    as skipped with the reason, and its other files are still exported.
    """
    chosen = set(kinds)
    wanted = [kind for kind in ExportKind if kind in chosen]
    plan = ExportPlan(folder=folder)
    # Two recordings that differ only in their audio extension, such as an
    # interview and its enhanced copy, would export to the same names, and
    # the second would overwrite the first without anybody being asked.
    # Neither is exported; both are named as skipped, so the person can
    # export them one at a time.
    stems: dict[str, list[Path]] = {}
    for recording in recordings:
        stems.setdefault(recording.stem.casefold(), []).append(recording)
    for recording in recordings:
        sharing = [other for other in stems[recording.stem.casefold()] if other != recording]
        if sharing:
            plan.skipped.append(
                SkippedFile(
                    recording.name,
                    f"it would export to the same file names as {sharing[0].name}, so neither "
                    "was exported. Export them one at a time",
                )
            )
            continue
        store = store_for(recording)
        for kind in wanted:
            if kind is ExportKind.SMOOTH_TRANSCRIPT:
                available = (store.exports_folder / SMOOTH_EXPORT_NAME).is_file()
                reason = "it has no smooth transcript"
            else:
                available = store.has_transcript
                reason = f"it has not been transcribed, so it has no {kind.value}"
            if available:
                plan.files.append(
                    ExportFile(recording, kind, folder / export_file_name(recording, kind))
                )
                if kind is ExportKind.SMOOTH_TRANSCRIPT and is_out_of_date(store):
                    plan.out_of_date.append(recording)
            else:
                plan.skipped.append(SkippedFile(recording.name, reason))
    return plan


def run_export(plan: ExportPlan, store_for: StoreFor, replace_existing: bool) -> ExportResult:
    """Write the planned files. Never raises.

    ``replace_existing`` False keeps a file already in the folder and lists
    it as kept. A file that cannot be written is named in ``failed``, and the
    rest are still written.
    """
    result = ExportResult(folder=plan.folder, skipped=list(plan.skipped))
    by_recording: dict[Path, list[ExportFile]] = {}
    for item in plan.files:
        by_recording.setdefault(item.recording, []).append(item)
    for recording, items in by_recording.items():
        _export_recording(
            recording, items, store_for(recording), replace_existing, result, plan.out_of_date
        )
    return result


def _export_recording(
    recording: Path,
    items: list[ExportFile],
    store: TranscriptStore,
    replace_existing: bool,
    result: ExportResult,
    out_of_date: Sequence[Path] = (),
) -> None:
    to_write = []
    for item in items:
        if item.target.exists() and not replace_existing:
            result.kept.append(item.target)
        else:
            to_write.append(item)
    if not to_write:
        return
    rendered = {item.kind for item in to_write if item.kind is not ExportKind.SMOOTH_TRANSCRIPT}
    transcript = store.load() if rendered else None
    if rendered:
        if transcript is None:
            result.failed.append(
                f"The transcript of {recording.name} could not be read, so its "
                f"{_kinds_text(rendered)} {'was' if len(rendered) == 1 else 'were'} not exported."
            )
            to_write = [item for item in to_write if item.kind not in rendered]
        else:
            failed = write_exports(
                transcript, store, names=[kind.own_export_name for kind in rendered]
            )
            if failed:
                result.failed.append(
                    f"The copy of {' and '.join(failed)} beside {recording.name} could not "
                    "be brought up to date. The exported copy is current."
                )
    for item in to_write:
        try:
            item.target.parent.mkdir(parents=True, exist_ok=True)
            if item.kind is ExportKind.SMOOTH_TRANSCRIPT:
                shutil.copyfile(store.exports_folder / SMOOTH_EXPORT_NAME, item.target)
                if recording in out_of_date:
                    result.out_of_date.append(recording.name)
            elif transcript is not None:
                render = (
                    render_plain_text
                    if item.kind is ExportKind.TRANSCRIPT
                    else render_review_report
                )
                item.target.write_text(render(transcript), encoding="utf-8")
        except OSError as error:
            _log.warning("Could not export %s.", item.target, exc_info=True)
            reason = error.strerror or type(error).__name__
            result.failed.append(f"{item.target.name} could not be written: {reason}.")
            continue
        result.written.append(item.target)


def summary_text(result: ExportResult) -> str:
    """What the export came to, in sentences a screen reader reads well."""
    count = len(result.written)
    noun = "file" if count == 1 else "files"
    sentences = [f"Exported {count} {noun} to {result.folder}."]
    if result.kept:
        kept = len(result.kept)
        sentences.append(
            f"Kept {kept} {'file' if kept == 1 else 'files'} that {'was' if kept == 1 else 'were'} "
            "already there."
        )
    if result.out_of_date:
        names = result.out_of_date
        one = len(names) == 1
        sentences.append(
            f"The smooth {'transcript' if one else 'transcripts'} of {names_text(names)} "
            f"{'was' if one else 'were'} exported as {'it was' if one else 'they were'}, "
            "older than the latest corrections."
        )
    if result.skipped:
        sentences.append("Skipped:")
        sentences.extend(f"{item.recording_name}: {item.reason}." for item in result.skipped)
    if result.failed:
        sentences.append("Problems:")
        sentences.extend(result.failed)
    return " ".join(sentences)


def names_text(names: Sequence[str]) -> str:
    """Names joined as a sentence reads them: "a", "a and b", "a, b and c"."""
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _kinds_text(kinds: Iterable[ExportKind]) -> str:
    chosen = set(kinds)
    return " and ".join(kind.value for kind in ExportKind if kind in chosen)
