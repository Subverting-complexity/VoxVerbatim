"""What the retry loop shared by every adapter must get right.

The adapters used to lean on their client libraries to retry, and two of
the libraries turned out to be unsafe to lean on: one resends a request
without its form body, and one retries nothing at all. So the retrying now
happens once, in the base class, and these tests hold it to its promises:
try again only where the adapter said that may help, stop when the user
stops the run, wait as long as the service asked, say how many attempts
were made, and keep the adapter's record of the last attempt.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vox_verbatim.settings import MAXIMUM_PROVIDER_RETRY_BACKOFF_SECONDS

from vox_verbatim.transcription.model import (
    Provider,
    ProviderRequestRecord,
    ProviderResult,
)
from vox_verbatim.transcription.providers import base
from vox_verbatim.transcription.providers.base import (
    ProviderCapabilities,
    ProviderError,
    TranscriptionProvider,
    TranscriptionRequest,
    retry_after_seconds,
    timeout_for_duration,
)


class Flaky(TranscriptionProvider):
    """An adapter that fails a scripted number of times before answering."""

    provider = Provider.MICROSOFT
    capabilities = ProviderCapabilities()

    def __init__(self, failures: list[ProviderError], retries: int = 2) -> None:
        self.failures = list(failures)
        self.attempts = 0
        self.maximum_retries = retries
        self.retry_backoff_seconds = 1.0

    @property
    def model_identifier(self) -> str:
        return "fake"

    def is_configured(self) -> bool:
        return True

    def describe_configuration_problem(self) -> str | None:
        return None

    def _transcribe(self, request, cancelled=None) -> ProviderResult:
        self.attempts += 1
        if self.failures:
            raise self.failures.pop(0)
        return ProviderResult(provider=self.provider, tokens=[])


def request(tmp_path: Path) -> TranscriptionRequest:
    return TranscriptionRequest(audio_path=tmp_path / "a.wav", duration=10.0)


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    """Record the waits instead of taking them."""
    taken: list[float] = []
    monkeypatch.setattr(base, "_sleep", taken.append)
    return taken


def retryable(message: str = "timed out", **extra) -> ProviderError:
    return ProviderError(message, retryable=True, **extra)


def test_a_retryable_failure_is_tried_again_and_can_then_succeed(tmp_path, sleeps):
    adapter = Flaky([retryable(), retryable()], retries=2)

    result = adapter.transcribe(request(tmp_path))

    assert result.succeeded
    assert adapter.attempts == 3


def test_the_waits_double_and_the_final_sentence_counts_the_attempts(tmp_path, sleeps):
    adapter = Flaky([retryable("no route"), retryable("no route"), retryable("no route")], retries=2)

    result = adapter.transcribe(request(tmp_path))

    assert not result.succeeded
    assert adapter.attempts == 3
    assert result.error.startswith("Microsoft MAI failed after 3 attempts: no route")
    # One second, then two. The waits are slept in pieces of at most a
    # second, so the pieces are summed per wait.
    assert sum(sleeps) == pytest.approx(3.0)


def test_a_first_wait_longer_than_the_cap_is_honoured(tmp_path, sleeps):
    adapter = Flaky([retryable(), retryable()], retries=2)
    adapter.retry_backoff_seconds = 120.0

    adapter.transcribe(request(tmp_path))

    # Two minutes, then two minutes again: the doubling stops at the
    # longer of the cap and the wait the person asked for.
    assert sum(sleeps) == pytest.approx(240.0)


def test_every_wait_the_setting_allows_is_honoured():
    adapter = Flaky([])
    adapter.retry_backoff_seconds = MAXIMUM_PROVIDER_RETRY_BACKOFF_SECONDS

    assert adapter._wait_before_attempt(1, None) == MAXIMUM_PROVIDER_RETRY_BACKOFF_SECONDS


def test_short_waits_still_stop_doubling_at_the_cap():
    adapter = Flaky([])
    adapter.retry_backoff_seconds = 2.0

    assert adapter._wait_before_attempt(10, None) == adapter.maximum_backoff_seconds


def test_a_failure_the_adapter_called_final_is_not_tried_again(tmp_path, sleeps):
    adapter = Flaky([ProviderError("the key was refused", retryable=False)], retries=5)

    result = adapter.transcribe(request(tmp_path))

    assert adapter.attempts == 1
    assert result.error == "the key was refused"
    assert sleeps == []


def test_zero_retries_means_one_attempt(tmp_path, sleeps):
    adapter = Flaky([retryable()], retries=0)

    result = adapter.transcribe(request(tmp_path))

    assert adapter.attempts == 1
    assert "attempts" not in result.error


def test_a_stopped_run_is_not_tried_again(tmp_path, sleeps):
    adapter = Flaky([retryable(), retryable()], retries=3)
    stopped = False

    def cancelled() -> bool:
        return stopped

    # The first failure is allowed one retry; the user stops during the wait.
    def stop_during_sleep(seconds: float) -> None:
        nonlocal stopped
        stopped = True

    base._sleep = stop_during_sleep
    result = adapter.transcribe(request(tmp_path), cancelled=cancelled)

    assert not result.succeeded
    assert adapter.attempts == 1


def test_the_wait_the_service_asked_for_is_honoured(tmp_path, sleeps):
    adapter = Flaky([retryable(retry_after=7.0)], retries=1)

    adapter.transcribe(request(tmp_path))

    assert sum(sleeps) == pytest.approx(7.0)


def test_an_absurd_retry_after_is_capped(tmp_path, sleeps):
    adapter = Flaky([retryable(retry_after=3600.0)], retries=1)

    adapter.transcribe(request(tmp_path))

    assert sum(sleeps) == pytest.approx(base.MAXIMUM_RETRY_AFTER_SECONDS)


def test_the_adapters_record_of_the_last_attempt_is_kept(tmp_path, sleeps):
    """Provenance rides on the error so a retry does not throw it away."""
    record = ProviderRequestRecord(
        provider=Provider.MICROSOFT,
        model_identifier="fake",
        request_parameters={"locale": "en"},
        language_configuration="en",
        vocabulary_terms=(),
        started_at="now",
        succeeded=False,
        error="timed out",
    )
    carried = ProviderResult(
        provider=Provider.MICROSOFT, request=record, error="timed out", raw_response={"why": 1}
    )
    adapter = Flaky([retryable(result=carried), retryable(result=carried)], retries=1)

    result = adapter.transcribe(request(tmp_path))

    assert result.raw_response == {"why": 1}
    assert result.request is not None
    assert result.request.request_parameters == {"locale": "en"}
    assert result.request.error == result.error
    assert "failed after 2 attempts" in result.error


def test_something_that_is_not_a_provider_error_is_caught_but_never_retried(tmp_path, sleeps):
    class Broken(Flaky):
        def _transcribe(self, request, cancelled=None) -> ProviderResult:
            self.attempts += 1
            raise RuntimeError("a bug")

    adapter = Broken([], retries=3)
    result = adapter.transcribe(request(tmp_path))

    assert not result.succeeded
    assert "a bug" in result.error
    assert adapter.attempts == 1


# -- Helpers the adapters share -------------------------------------------


def test_the_timeout_grows_with_the_audio_and_stops_at_two_hours():
    assert timeout_for_duration(900.0, None) == 900.0
    assert timeout_for_duration(900.0, 60.0) == 900.0
    # Three hours of audio: 300 + 10800 is over the cap, so the cap.
    assert timeout_for_duration(900.0, 3 * 3600.0) == 7200.0
    # An hour: 300 + 3600.
    assert timeout_for_duration(900.0, 3600.0) == 3900.0
    # A configured timeout above the scaled one still wins.
    assert timeout_for_duration(5000.0, 3600.0) == 5000.0


def test_retry_after_is_read_in_seconds_or_milliseconds():
    assert retry_after_seconds({"retry-after": "12"}) == 12.0
    assert retry_after_seconds({"Retry-After": " 3 "}) == 3.0
    assert retry_after_seconds({"retry-after-ms": "1500"}) == 1.5
    assert retry_after_seconds({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}) is None
    assert retry_after_seconds({}) is None
    assert retry_after_seconds(None) is None
