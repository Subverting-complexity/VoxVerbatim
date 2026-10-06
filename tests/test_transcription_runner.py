"""Tests for the background runner that drives the transcription pipeline.

The pipeline itself is replaced throughout. These tests are about what the
runner reports and when it stops, not about transcription, and a real run
would call four paid services over the network. Nothing here imports a
provider or opens a socket.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from vox_verbatim.transcription import pipeline
from vox_verbatim.transcription.model import (
    FinalToken,
    ReviewStatus,
    Transcript,
)
from vox_verbatim.transcription.runner import (
    RecordingOutcome,
    RunSummary,
    TranscriptionRunner,
    summarise,
)

from tests.conftest import wait_until


def make_transcript(
    name: str,
    words: int = 3,
    needing_review: int = 0,
    warnings: tuple[str, ...] = (),
) -> Transcript:
    """A transcript with a known number of words, some of them still in doubt."""
    transcript = Transcript(recording_name=name, warnings=list(warnings))
    for index in range(words):
        token = FinalToken(text=f"word{index}")
        if index < needing_review:
            token.review_status = ReviewStatus.PENDING
        transcript.tokens.append(token)
    return transcript


@pytest.fixture
def recordings(tmp_path) -> list[Path]:
    """Two file names. Nothing reads them, because the pipeline is replaced."""
    return [tmp_path / "alpha.m4a", tmp_path / "beta.m4a"]


def fake_pipeline(monkeypatch, transcribe) -> None:
    """Put ``transcribe`` in the pipeline's place for the length of one test."""
    monkeypatch.setattr(pipeline, "transcribe_recording", transcribe)


# -- Reporting a run -----------------------------------------------------


def test_the_runner_reports_each_recording_and_finishes(qapp, monkeypatch, recordings):
    def transcribe(recording, options, progress=None, cancelled=None):
        if progress is not None:
            progress(
                pipeline.PipelineProgress(
                    fraction=0.5, stage=f"Comparing what each service heard for {recording.name}."
                )
            )
        return make_transcript(recording.name, words=4, needing_review=1)

    fake_pipeline(monkeypatch, transcribe)
    runner = TranscriptionRunner()
    started: list[tuple[str, int, int]] = []
    finished: list[RecordingOutcome] = []
    summaries: list[RunSummary] = []
    runner.recordingStarted.connect(
        lambda name, number, total: started.append((name, number, total))
    )
    runner.recordingFinished.connect(finished.append)
    runner.runFinished.connect(summaries.append)

    assert runner.start(recordings, pipeline.PipelineOptions()) is True
    assert wait_until(qapp, lambda: bool(summaries))

    assert started == [("alpha.m4a", 1, 2), ("beta.m4a", 2, 2)]
    assert [outcome.name for outcome in finished] == ["alpha.m4a", "beta.m4a"]
    assert all(outcome.succeeded for outcome in finished)
    assert summaries[0].transcribed == 2
    assert summaries[0].failed == 0
    assert summaries[0].review_count == 2
    assert summaries[0].cancelled is False


def test_progress_carries_the_stage_sentence_and_covers_the_whole_run(
    qapp, monkeypatch, recordings
):
    def transcribe(recording, options, progress=None, cancelled=None):
        progress(pipeline.PipelineProgress(fraction=0.0, stage=f"Preparing {recording.name}."))
        progress(pipeline.PipelineProgress(fraction=1.0, stage=f"Finished {recording.name}."))
        return make_transcript(recording.name)

    fake_pipeline(monkeypatch, transcribe)
    runner = TranscriptionRunner()
    reports: list[tuple[int, str]] = []
    summaries: list[RunSummary] = []
    runner.progressChanged.connect(lambda percentage, stage: reports.append((percentage, stage)))
    runner.runFinished.connect(summaries.append)

    runner.start(recordings, pipeline.PipelineOptions())
    assert wait_until(qapp, lambda: bool(summaries))

    # The first recording covers the first half of the bar and the second the
    # rest, so a two-recording run does not run to the end twice.
    assert reports == [
        (0, "Preparing alpha.m4a."),
        (50, "Finished alpha.m4a."),
        (50, "Preparing beta.m4a."),
        (100, "Finished beta.m4a."),
    ]


def test_the_warnings_from_a_transcript_are_carried_into_its_report(
    qapp, monkeypatch, recordings
):
    def transcribe(recording, options, progress=None, cancelled=None):
        return make_transcript(
            recording.name,
            warnings=("Microsoft MAI did not answer, so its opinion is missing.",),
        )

    fake_pipeline(monkeypatch, transcribe)
    runner = TranscriptionRunner()
    finished: list[RecordingOutcome] = []
    summaries: list[RunSummary] = []
    runner.recordingFinished.connect(finished.append)
    runner.runFinished.connect(summaries.append)

    runner.start(recordings[:1], pipeline.PipelineOptions())
    assert wait_until(qapp, lambda: bool(summaries))

    assert "Microsoft MAI did not answer" in finished[0].message


# -- Recordings go one at a time -----------------------------------------


def test_recordings_are_never_transcribed_at_the_same_time(qapp, monkeypatch, tmp_path):
    """Two at once would collide on rate limits that are counted per account."""
    running = 0
    most_at_once = 0
    lock = threading.Lock()

    def transcribe(recording, options, progress=None, cancelled=None):
        nonlocal running, most_at_once
        with lock:
            running += 1
            most_at_once = max(most_at_once, running)
        try:
            return make_transcript(recording.name)
        finally:
            with lock:
                running -= 1

    fake_pipeline(monkeypatch, transcribe)
    runner = TranscriptionRunner()
    summaries: list[RunSummary] = []
    runner.runFinished.connect(summaries.append)

    runner.start([tmp_path / f"take-{index}.m4a" for index in range(5)], pipeline.PipelineOptions())
    assert wait_until(qapp, lambda: bool(summaries))

    assert most_at_once == 1
    assert summaries[0].transcribed == 5


def test_a_second_run_cannot_start_on_top_of_the_first(qapp, monkeypatch, recordings):
    release = threading.Event()

    def transcribe(recording, options, progress=None, cancelled=None):
        release.wait(10.0)
        return make_transcript(recording.name)

    fake_pipeline(monkeypatch, transcribe)
    runner = TranscriptionRunner()
    summaries: list[RunSummary] = []
    runner.runFinished.connect(summaries.append)

    assert runner.start(recordings, pipeline.PipelineOptions()) is True
    assert runner.start(recordings, pipeline.PipelineOptions()) is False

    release.set()
    assert wait_until(qapp, lambda: bool(summaries))


# -- Stopping ------------------------------------------------------------


def test_cancelling_stops_the_queue_and_says_the_run_was_stopped(
    qapp, monkeypatch, tmp_path
):
    seen: list[str] = []
    release = threading.Event()

    def transcribe(recording, options, progress=None, cancelled=None):
        seen.append(recording.name)
        release.wait(10.0)
        return make_transcript(recording.name)

    fake_pipeline(monkeypatch, transcribe)
    runner = TranscriptionRunner()
    summaries: list[RunSummary] = []
    runner.runFinished.connect(summaries.append)

    runner.start([tmp_path / f"take-{index}.m4a" for index in range(4)], pipeline.PipelineOptions())
    assert wait_until(qapp, lambda: bool(seen))
    runner.cancel()
    release.set()
    assert wait_until(qapp, lambda: bool(summaries))

    assert summaries[0].cancelled is True
    # The recording in hand finishes, and nothing after it is started.
    assert seen == ["take-0.m4a"]
    assert len(summaries[0].results) == 1


def test_the_pipeline_is_given_a_way_to_ask_whether_it_should_stop(
    qapp, monkeypatch, recordings
):
    answers: list[bool] = []

    def transcribe(recording, options, progress=None, cancelled=None):
        answers.append(cancelled())
        return make_transcript(recording.name)

    fake_pipeline(monkeypatch, transcribe)
    runner = TranscriptionRunner()
    summaries: list[RunSummary] = []
    runner.runFinished.connect(summaries.append)

    runner.start(recordings[:1], pipeline.PipelineOptions())
    assert wait_until(qapp, lambda: bool(summaries))

    assert answers == [False]


# -- Failures ------------------------------------------------------------


def test_a_recording_that_cannot_be_read_becomes_a_result_and_the_run_goes_on(
    qapp, monkeypatch, recordings
):
    def transcribe(recording, options, progress=None, cancelled=None):
        if recording.name == "alpha.m4a":
            raise OSError("The file is not readable.")
        return make_transcript(recording.name)

    fake_pipeline(monkeypatch, transcribe)
    runner = TranscriptionRunner()
    summaries: list[RunSummary] = []
    runner.runFinished.connect(summaries.append)

    runner.start(recordings, pipeline.PipelineOptions())
    assert wait_until(qapp, lambda: bool(summaries))

    summary = summaries[0]
    assert summary.failed == 1
    assert summary.transcribed == 1
    assert "The file is not readable." in summary.results[0].message
    assert summary.results[1].succeeded is True


def test_a_failure_with_nothing_to_say_still_says_something(qapp, monkeypatch, recordings):
    def transcribe(recording, options, progress=None, cancelled=None):
        raise RuntimeError()

    fake_pipeline(monkeypatch, transcribe)
    runner = TranscriptionRunner()
    summaries: list[RunSummary] = []
    runner.runFinished.connect(summaries.append)

    runner.start(recordings[:1], pipeline.PipelineOptions())
    assert wait_until(qapp, lambda: bool(summaries))

    assert "RuntimeError" in summaries[0].results[0].message


# -- Saying how it went --------------------------------------------------


def test_a_clean_run_is_summarised_with_its_review_count():
    summary = RunSummary(
        results=[
            RecordingOutcome.of_transcript(
                Path("alpha.m4a"),
                make_transcript("alpha.m4a", words=10, needing_review=3),
            )
        ]
    )

    assert summarise(summary) == "Finished. 1 of 1 recordings transcribed. 3 words to review."


def test_a_run_with_nothing_to_review_says_so_rather_than_saying_nothing():
    summary = RunSummary(
        results=[RecordingOutcome.of_transcript(Path("alpha.m4a"), make_transcript("alpha.m4a"))]
    )

    assert "Nothing is waiting for review." in summarise(summary)


def test_a_stopped_run_is_not_reported_as_finished():
    summary = RunSummary(
        results=[RecordingOutcome.of_transcript(Path("alpha.m4a"), make_transcript("alpha.m4a"))],
        cancelled=True,
    )

    assert summarise(summary).startswith("The run was stopped.")


def test_a_failure_is_named_in_the_summary():
    summary = RunSummary(
        results=[RecordingOutcome(path=Path("alpha.m4a"), error="No key was set.")]
    )

    text = summarise(summary)
    assert "could not be transcribed" in text
    assert "0 of 1 recordings transcribed" in text


NO_SERVICE = "No transcription service produced a result, so there is nothing to compare."


def test_a_transcript_with_no_words_is_a_failure_not_a_success():
    outcome = RecordingOutcome.of_transcript(
        Path("a.m4a"), make_transcript("a.m4a", words=0, warnings=(NO_SERVICE,))
    )

    assert outcome.succeeded is False
    assert outcome.incomplete is False
    assert NO_SERVICE in outcome.error
    # The reason is said once, not again as a warning after it.
    assert outcome.message == f"a.m4a could not be transcribed. {NO_SERVICE}"


def test_a_transcript_with_no_words_and_no_warning_still_says_why():
    outcome = RecordingOutcome.of_transcript(Path("a.m4a"), make_transcript("a.m4a", words=0))

    assert outcome.error == "No service returned any words."


def test_a_run_where_no_service_answered_is_not_reported_as_finished():
    summary = RunSummary(
        results=[
            RecordingOutcome.of_transcript(
                Path("a.m4a"), make_transcript("a.m4a", words=0, warnings=(NO_SERVICE,))
            )
        ]
    )

    text = summarise(summary)
    assert summary.failed == 1
    assert summary.transcripts == []
    assert "0 of 1 recordings transcribed" in text
    assert "1 could not be transcribed" in text
    assert not text.startswith("Finished.")


def test_a_recording_stopped_part_way_is_reported_as_incomplete():
    transcript = make_transcript("a.m4a", words=5)
    transcript.stopped = True
    outcome = RecordingOutcome.of_transcript(Path("a.m4a"), transcript)
    summary = RunSummary(results=[outcome])

    assert outcome.incomplete is True
    # A partial transcript can still be opened and read.
    assert outcome.succeeded is True
    assert "incomplete" in outcome.message
    assert summary.transcribed == 0
    assert summary.failed == 0
    assert summary.incomplete == 1
    assert summary.transcripts == [outcome]
    text = summarise(summary)
    assert "1 stopped part way" in text
    assert not text.startswith("Finished.")


def test_a_recording_stopped_before_any_words_says_so():
    transcript = make_transcript("a.m4a", words=0)
    transcript.stopped = True
    outcome = RecordingOutcome.of_transcript(Path("a.m4a"), transcript)

    assert outcome.succeeded is False
    assert outcome.incomplete is True
    assert outcome.message == "a.m4a was stopped before it was transcribed."
    assert RunSummary(results=[outcome]).failed == 0


def test_the_runner_reports_an_empty_transcript_as_a_failure(qapp, monkeypatch, recordings):
    def transcribe(recording, options, progress=None, cancelled=None):
        return make_transcript(recording.name, words=0, warnings=(NO_SERVICE,))

    fake_pipeline(monkeypatch, transcribe)
    runner = TranscriptionRunner()
    summaries: list[RunSummary] = []
    runner.runFinished.connect(summaries.append)

    runner.start(recordings[:1], pipeline.PipelineOptions())
    assert wait_until(qapp, lambda: bool(summaries))

    summary = summaries[0]
    assert summary.failed == 1
    assert summary.transcribed == 0
    assert NO_SERVICE in summary.results[0].message


# -- What an outcome carries ---------------------------------------------


def test_an_outcome_keeps_the_numbers_and_the_folder_and_lets_the_transcript_go(
    qapp, monkeypatch, recordings
):
    """A run over a day of recordings must not hold every transcript in memory."""

    def transcribe(recording, options, progress=None, cancelled=None):
        return make_transcript(
            recording.name, words=7, needing_review=2, warnings=("One service was slow.",)
        )

    fake_pipeline(monkeypatch, transcribe)
    runner = TranscriptionRunner()
    finished: list[RecordingOutcome] = []
    summaries: list[RunSummary] = []
    runner.recordingFinished.connect(finished.append)
    runner.runFinished.connect(summaries.append)

    runner.start(recordings[:1], pipeline.PipelineOptions())
    assert wait_until(qapp, lambda: bool(summaries))

    outcome = finished[0]
    assert outcome.succeeded is True
    assert outcome.word_count == 7
    assert outcome.review_count == 2
    assert outcome.warnings == ("One service was slow.",)
    assert outcome.transcript_folder == recordings[0].parent / "alpha.m4a.transcript"
    assert not hasattr(outcome, "transcript")


# -- Keeping the machine awake -------------------------------------------


def asked_of_windows(monkeypatch) -> list[int]:
    """Stand in for the Windows call and keep the flags it was given."""
    from vox_verbatim.transcription import runner as runner_module

    asked: list[int] = []
    monkeypatch.setattr(runner_module, "set_thread_execution_state", asked.append)
    monkeypatch.setattr(runner_module.sys, "platform", "win32")
    # Requests are counted across runners. A run from an earlier test that
    # was never waited for may still hold one, so the count starts afresh.
    monkeypatch.setattr(runner_module, "_awake_requests", 0)
    return asked


def test_the_machine_is_held_awake_for_the_run_and_released_when_it_ends(
    qapp, monkeypatch, recordings
):
    from vox_verbatim.transcription.runner import ES_CONTINUOUS, ES_SYSTEM_REQUIRED

    asked = asked_of_windows(monkeypatch)
    fake_pipeline(monkeypatch, lambda recording, *args, **kwargs: make_transcript(recording.name))
    runner = TranscriptionRunner()
    summaries: list[RunSummary] = []
    runner.runFinished.connect(summaries.append)

    runner.start(recordings, pipeline.PipelineOptions())
    assert asked == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED]
    assert wait_until(qapp, lambda: bool(summaries))

    assert asked == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED, ES_CONTINUOUS]
    # The display is left alone: nobody needs to watch a run that lasts hours.
    assert all(flags & 0x00000002 == 0 for flags in asked)


def test_a_windows_call_that_fails_does_not_stop_the_run(qapp, monkeypatch, recordings):
    from vox_verbatim.transcription import runner as runner_module

    def refuse(flags: int) -> None:
        raise OSError("no kernel32 here")

    monkeypatch.setattr(runner_module, "set_thread_execution_state", refuse)
    monkeypatch.setattr(runner_module.sys, "platform", "win32")
    fake_pipeline(monkeypatch, lambda recording, *args, **kwargs: make_transcript(recording.name))
    runner = TranscriptionRunner()
    summaries: list[RunSummary] = []
    runner.runFinished.connect(summaries.append)

    assert runner.start(recordings, pipeline.PipelineOptions()) is True
    assert wait_until(qapp, lambda: bool(summaries))
    assert summaries[0].transcribed == 2


def test_nothing_is_asked_of_a_machine_that_is_not_windows(monkeypatch):
    from vox_verbatim.transcription import runner as runner_module

    asked: list[int] = []
    monkeypatch.setattr(runner_module, "set_thread_execution_state", asked.append)
    monkeypatch.setattr(runner_module.sys, "platform", "linux")

    runner_module.keep_system_awake(True)
    runner_module.keep_system_awake(False)

    assert asked == []


def test_the_machine_is_let_go_only_when_the_last_run_has_withdrawn(monkeypatch):
    """Windows keeps one state per thread, not a count.

    A second run started while the first is still stopping would otherwise
    have its request cleared the moment the first run withdrew, and the
    machine could sleep under a run that had hours to go.
    """
    from vox_verbatim.transcription import runner as runner_module
    from vox_verbatim.transcription.runner import ES_CONTINUOUS, ES_SYSTEM_REQUIRED

    asked = asked_of_windows(monkeypatch)
    awake = ES_CONTINUOUS | ES_SYSTEM_REQUIRED

    runner_module.keep_system_awake(True)
    runner_module.keep_system_awake(True)
    assert asked == [awake]
    runner_module.keep_system_awake(False)
    assert asked == [awake]
    runner_module.keep_system_awake(False)
    assert asked == [awake, ES_CONTINUOUS]
    # A withdrawal with nothing outstanding clears nobody's request.
    runner_module.keep_system_awake(False)
    assert asked == [awake, ES_CONTINUOUS]


def test_a_stopped_run_withdraws_its_request_once(qapp, monkeypatch, recordings):
    """A stopped run reaches the withdrawal twice, from stop and from its summary.

    Withdrawing twice would take another run's request with it now that the
    requests are counted, so the runner only withdraws what it holds.
    """
    from vox_verbatim.transcription import runner as runner_module
    from vox_verbatim.transcription.runner import ES_CONTINUOUS, ES_SYSTEM_REQUIRED

    asked = asked_of_windows(monkeypatch)
    release = threading.Event()

    def transcribe(recording, options, progress=None, cancelled=None):
        release.wait(10.0)
        return make_transcript(recording.name)

    fake_pipeline(monkeypatch, transcribe)
    first = TranscriptionRunner()
    second = TranscriptionRunner()
    summaries: list[RunSummary] = []
    first.runFinished.connect(summaries.append)
    second.runFinished.connect(summaries.append)

    first.start(recordings, pipeline.PipelineOptions())
    second.start(recordings, pipeline.PipelineOptions())
    first.stop(timeout_seconds=0.01)
    # The first run has withdrawn, but the second is still going.
    assert asked == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED]

    release.set()
    assert wait_until(qapp, lambda: len(summaries) == 2)
    # The first run's summary did not withdraw a second time; the second
    # run's did, and that is when the machine was let go.
    assert asked == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED, ES_CONTINUOUS]
    assert runner_module._awake_requests == 0
