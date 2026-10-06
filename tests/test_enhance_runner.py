"""Tests for what the enhancement runner does when the engine lets it down.

The engine itself is replaced here. The ordinary path, with real audio, is
covered alongside the Enhance Audio dialog; these tests are about the runner
still reporting back when the engine raises, and about the machine being
held awake for the length of a run.
"""

from __future__ import annotations

from pathlib import Path

from vox_verbatim.audio import enhance_runner as runner_module
from vox_verbatim.audio.enhance import EnhanceOptions, FileResult, Outcome
from vox_verbatim.audio.enhance_runner import EnhanceRunner
from vox_verbatim.transcription import runner as transcription_runner

from tests.conftest import wait_until


def test_an_engine_that_raises_still_ends_the_run_with_a_failed_summary(
    qapp, monkeypatch, tmp_path
):
    """Without this the dialog stays disabled for ever, saying it is working."""
    files = [tmp_path / "alpha.m4a", tmp_path / "beta.m4a", tmp_path / "gamma.m4a"]

    def explode(sources, options, progress, cancelled, on_start, on_result):
        on_start(sources[0], 1, len(sources))
        on_result(FileResult(source=sources[0], outcome=Outcome.ENHANCED, message="Done."))
        raise RuntimeError("The decoder fell over.")

    monkeypatch.setattr(runner_module, "enhance_files", explode)
    runner = EnhanceRunner()
    summaries: list = []
    runner.runFinished.connect(summaries.append)

    assert runner.start(files, EnhanceOptions(output_folder=tmp_path / "enhanced")) is True
    assert wait_until(qapp, lambda: bool(summaries))

    summary = summaries[0]
    assert summary.enhanced == 1
    assert summary.failed == 2
    assert [result.name for result in summary.results] == ["alpha.m4a", "beta.m4a", "gamma.m4a"]
    assert "The decoder fell over." in summary.results[1].message
    assert runner.is_running is False



def test_a_crash_after_clashes_were_reported_still_names_every_file_once(
    qapp, monkeypatch, tmp_path
):
    """Clashing files are reported before the others, so results are out of file order."""
    files = [tmp_path / "talk.m4a", tmp_path / "quiet.m4a", tmp_path / "talk.mp3"]

    def explode(sources, options, progress, cancelled, on_start, on_result):
        on_result(FileResult(source=sources[0], outcome=Outcome.SKIPPED, message="Clash."))
        on_result(FileResult(source=sources[2], outcome=Outcome.SKIPPED, message="Clash."))
        on_start(sources[1], 3, len(sources))
        raise RuntimeError("The decoder fell over.")

    monkeypatch.setattr(runner_module, "enhance_files", explode)
    runner = EnhanceRunner()
    summaries: list = []
    runner.runFinished.connect(summaries.append)

    assert runner.start(files, EnhanceOptions(output_folder=tmp_path / "enhanced")) is True
    assert wait_until(qapp, lambda: bool(summaries))

    summary = summaries[0]
    assert sorted(result.name for result in summary.results) == sorted(f.name for f in files)
    assert summary.failed == 1
    failed = [result for result in summary.results if result.outcome is Outcome.FAILED]
    assert failed[0].name == "quiet.m4a"

def test_the_machine_is_held_awake_for_the_run_and_released_after(qapp, monkeypatch, tmp_path):
    asked: list[int] = []
    monkeypatch.setattr(transcription_runner, "set_thread_execution_state", asked.append)
    monkeypatch.setattr(transcription_runner.sys, "platform", "win32")
    monkeypatch.setattr(
        runner_module, "enhance_files", lambda *args, **kwargs: []
    )
    runner = EnhanceRunner()
    summaries: list = []
    runner.runFinished.connect(summaries.append)

    runner.start([Path(tmp_path / "alpha.m4a")], EnhanceOptions(output_folder=tmp_path / "out"))
    assert wait_until(qapp, lambda: bool(summaries))

    awake = transcription_runner.ES_CONTINUOUS | transcription_runner.ES_SYSTEM_REQUIRED
    assert asked == [awake, transcription_runner.ES_CONTINUOUS]
