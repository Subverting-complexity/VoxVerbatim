"""What every speech-to-text adapter must offer, and what it may assume.

An adapter has one job: take a piece of audio and a description of what is
wanted, and return words on the canonical timeline. Everything else about
the service it wraps stays behind this line.

Two ideas keep the adapters honest.

The first is that an adapter declares what it can do rather than being asked
to pretend. Only ElevenLabs times individual words; only ElevenLabs and
AssemblyAI tell speakers apart; only some accept a list of names to listen
out for. The pipeline reads :class:`ProviderCapabilities` and asks each
service for what it can actually give, instead of every adapter faking a
uniform feature set and quietly returning made-up timings.

The second is that a failure is a result, not an exception that escapes.
The whole architecture rests on carrying on when a service does not answer:
losing Microsoft should cost a little accuracy, and losing nothing else.
Adapters therefore catch what their client library throws and return a
failed :class:`~vox_verbatim.transcription.model.ProviderResult`, so
the pipeline never has to guard every call.

Nothing here imports a provider library. The adapters do that inside their
own modules, so a missing package disables one service rather than stopping
the application from starting.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from vox_verbatim.transcription.model import (
    AudioSpan,
    Language,
    Provider,
    ProviderResult,
)

_log = logging.getLogger(__name__)

#: Called between pieces of work so a run can be stopped promptly. It
#: returns True once the user has asked to cancel. Adapters check it around
#: anything slow, chiefly between chunks and between polls of a service that
#: makes you wait for a result.
CancelCheck = Callable[[], bool]

#: Called with a fraction from 0 to 1 as an adapter works through its
#: chunks, so the progress bar moves during a long recording instead of
#: sitting still until the whole service is done.
ProgressCallback = Callable[[float], None]


class ProviderError(Exception):
    """Something went wrong talking to a service.

    Adapters raise this internally and convert it to a failed result at
    their boundary. It carries whether trying again could help, because the
    retry policy has no other way to tell a rate limit from a rejected key.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        retry_after: float | None = None,
        result: ProviderResult | None = None,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after = retry_after
        """How long the service asked us to wait, in seconds, where it said.

        Read off a ``Retry-After`` header. The retry loop waits at least this
        long before the next attempt, because a service that names a wait and
        is called sooner answers with the same refusal and the attempt is
        wasted."""
        self.result = result
        """The failed result the adapter had already built, where it had.

        An adapter that got as far as sending a request has provenance worth
        keeping: what was asked, of which model, and the body the service
        sent back instead of a transcript. Carrying that result on the error
        lets the retry loop in :meth:`TranscriptionProvider.transcribe` try
        again where that may help and still hand back the full record when
        it gives up, instead of a bare sentence."""


class ProviderNotConfigured(ProviderError):
    """The service cannot be called because its settings are incomplete.

    Raised before any network call, so that a missing API key is reported as
    the plain fact it is rather than as an authentication failure from the
    far end.
    """

    def __init__(self, provider: Provider, what_is_missing: str):
        super().__init__(
            f"{provider.display_name} is not set up: {what_is_missing}. "
            "Open Settings and fill it in.",
            retryable=False,
        )
        self.provider = provider


class ProviderUnavailable(ProviderError):
    """The service's client library is not installed.

    Each library is optional. One that is missing switches its service off
    and says so, rather than stopping the application from starting.
    """

    def __init__(self, provider: Provider, package: str):
        super().__init__(
            f"{provider.display_name} needs the {package} package, which is not "
            f"installed. Run: pip install {package}",
            retryable=False,
        )
        self.provider = provider
        self.package = package


@dataclass(frozen=True)
class ProviderCapabilities:
    """What a service can actually do.

    Written down rather than assumed, so the pipeline can ask each service
    for what it is good at. This is also what stops the application from
    reaching for an older OpenAI model to get timestamps: OpenAI declares
    that it does not time words, and the timing comes from a service that
    does.
    """

    word_timings: bool = False
    """Returns a start and end time for each word."""

    diarisation: bool = False
    """Tells the speakers apart and labels them."""

    speaker_count_hint: bool = False
    """Accepts the expected or maximum number of speakers."""

    word_confidence: bool = False
    """Returns a per-word log probability or confidence."""

    language_detection: bool = False
    """Reports the language it heard, rather than only accepting one."""

    per_word_language: bool = False
    """Reports the language of individual words, not only the whole file."""

    vocabulary_biasing: bool = False
    """Accepts names and terms to listen out for, however it spells that."""

    context_prompt: bool = False
    """Accepts a sentence or two of context about the recording."""

    time_window: bool = False
    """Can be asked for part of a recording without the audio being cut up
    first, which makes escalation cheaper where it is supported."""

    forced_alignment: bool = False
    """Can measure where known words fall in known audio."""

    languages: frozenset[Language] = frozenset(
        {Language.ENGLISH, Language.GERMAN}
    )
    """The languages of ours this service is trusted with. Microsoft's
    omission of Afrikaans here is what keeps it out of Afrikaans spans."""

    maximum_file_bytes: int = 25 * 1024 * 1024
    maximum_duration_seconds: float | None = None
    accepted_containers: frozenset[str] = frozenset({"wav", "mp3", "flac", "m4a"})

    def supports(self, language: Language) -> bool:
        return language in self.languages or language is Language.UNKNOWN


@dataclass
class TranscriptionRequest:
    """Everything an adapter needs for one piece of audio.

    The same object serves a full pass over a recording and a few seconds
    cut out for a second opinion. What separates them is ``canonical_offset``
    and ``window``: an adapter adds the offset to every time the service
    reports, so a clip taken from three minutes in comes back on the same
    timeline as the rest of the transcript.

    An adapter takes from this only what its capabilities say it can use,
    and ignores the rest. Passing a speaker count to a service that cannot
    use one is not an error.
    """

    audio_path: Path
    """The file to send, which may be the canonical audio, a chunk of it or
    a window cut from it."""

    canonical_offset: float = 0.0
    """Seconds to add to this service's times to get canonical time."""

    duration: float | None = None
    window: AudioSpan | None = None
    """Where in the canonical recording this audio came from, for the record."""

    languages: tuple[Language, ...] = (Language.ENGLISH,)
    """Which languages are allowed. Afrikaans appears only when the user
    enabled it for this recording."""

    expected_speaker_count: int = 1
    diarise: bool = True
    vocabulary_terms: tuple[str, ...] = ()
    context_prompt: str = ""
    verbatim: bool = True
    """Keep fillers and false starts. The verbatim layer is the authoritative
    one, and a service that tidies its output silently breaks the link
    between the words and the audio."""

    model_override: str | None = None
    """Used where a service has more than one model and the choice depends on
    the span, as with AssemblyAI's Afrikaans model."""

    extra_parameters: dict[str, Any] = field(default_factory=dict)
    """Parameters from Settings that the adapter passes straight through.

    This is what lets a new parameter on a service be used without changing
    the application, which the specification requires. Adapters send these
    through their client library's pass-through mechanism, and fall back to
    a direct call where the library would reject an argument it does not
    know."""

    purpose: str = "transcription"
    chunk_index: int = 0


class TranscriptionProvider(ABC):
    """One speech-to-text service, seen from the pipeline.

    Subclasses do the work in :meth:`_transcribe` and let errors out.
    :meth:`transcribe` catches them and turns them into a failed result, so
    that every caller can treat "the service did not answer" as an ordinary
    outcome.
    """

    provider: Provider
    capabilities: ProviderCapabilities

    @property
    @abstractmethod
    def model_identifier(self) -> str:
        """The exact model this adapter will use, from settings.

        Recorded against every result. Services change what sits behind a
        model name, so a transcript that does not record the name cannot
        explain why reprocessing it later gave a different answer.
        """

    @abstractmethod
    def is_configured(self) -> bool:
        """Whether this service has everything it needs to be called."""

    @abstractmethod
    def describe_configuration_problem(self) -> str | None:
        """What is missing, in words a person can act on, or None if nothing is."""

    @abstractmethod
    def _transcribe(
        self,
        request: TranscriptionRequest,
        cancelled: CancelCheck | None = None,
    ) -> ProviderResult:
        """Do the work. May raise :class:`ProviderError`."""

    #: How many times a failed request is tried again. Zero means one attempt
    #: and no retry. Adapters set it from their constructor so the registry
    #: can pass the user's setting through.
    maximum_retries: int = 2

    #: The wait before the first retry, in seconds. Each later wait is twice
    #: the one before, up to :attr:`maximum_backoff_seconds` or the first wait,
    #: whichever is longer. The first wait is the user's setting, which allows
    #: more than the cap, and a person who asked for two minutes because a
    #: service is rate-limiting them must get two minutes, not one.
    retry_backoff_seconds: float = 2.0
    maximum_backoff_seconds: float = 60.0

    def transcribe(
        self,
        request: TranscriptionRequest,
        cancelled: CancelCheck | None = None,
    ) -> ProviderResult:
        """Transcribe one piece of audio, never raising, trying again where that may help.

        A service that fails comes back as a result carrying the reason. The
        pipeline reports it, notes it in the transcript's warnings, and
        carries on with the services that did answer.

        The retrying happens here, once, rather than inside each adapter or
        inside each client library. Two of the libraries were found to be
        unsafe to lean on: one drops the form body from a retried request, so
        every retry fails with a validation error, and another does not retry
        a connection failure at all. Every adapter already works out whether
        a failure is worth another attempt and says so on the error; this is
        the one place that reads the flag. Adapters open their file handle
        inside :meth:`_transcribe`, so every attempt re-reads the audio from
        the start rather than sending whatever was left of a half-read file.

        The wait between attempts doubles each time, and is never shorter
        than what the service asked for in a ``Retry-After`` header. Between
        attempts the cancel check is consulted, so a user who stops the run
        while a service is rate-limiting us is not kept waiting through the
        remaining backoff.
        """
        return self.call_with_retries(lambda: self._transcribe(request, cancelled), cancelled)

    def call_with_retries(
        self,
        call: Callable[[], ProviderResult],
        cancelled: CancelCheck | None = None,
    ) -> ProviderResult:
        """Run one request to the service, trying again where that may help.

        This is the loop :meth:`transcribe` runs, made available on its own
        so that a request which does not go through :meth:`transcribe`, such
        as the time-window questions escalation asks, is retried by the same
        rules rather than by none.
        """
        attempts_allowed = 1 + max(0, int(self.maximum_retries))
        name = self.provider.display_name
        attempt = 0
        while True:
            attempt += 1
            try:
                return call()
            except ProviderError as error:
                may_retry = error.retryable and attempt < attempts_allowed
                if may_retry and cancelled is not None and cancelled():
                    may_retry = False
                if not may_retry:
                    _log.warning("%s did not answer: %s", name, error)
                    return self._failed_result(error, attempt)
                wait = self._wait_before_attempt(attempt, error.retry_after)
                _log.warning(
                    "%s did not answer (attempt %d of %d): %s. Trying again in %.0f s.",
                    name,
                    attempt,
                    attempts_allowed,
                    error,
                    wait,
                )
                if not wait_unless_cancelled(wait, cancelled):
                    return self._failed_result(error, attempt)
            except Exception as error:  # a broken adapter must not take the run down
                _log.exception("%s raised an unexpected error.", name)
                return ProviderResult(
                    provider=self.provider,
                    error=f"{name} failed unexpectedly: {error}",
                )

    def _wait_before_attempt(self, attempts_so_far: int, retry_after: float | None) -> float:
        """How long to wait before the next attempt, in seconds."""
        first = max(0.0, float(self.retry_backoff_seconds))
        backoff = first * (2 ** (attempts_so_far - 1))
        backoff = min(backoff, max(first, float(self.maximum_backoff_seconds)))
        if retry_after is not None and retry_after > 0:
            # A service that names a wait is not bargained with: calling it
            # sooner gets the same refusal. The cap is a guard against a
            # header that asks for an hour, which is a reason to give up
            # rather than to sit still.
            backoff = max(backoff, min(float(retry_after), MAXIMUM_RETRY_AFTER_SECONDS))
        return backoff

    def _failed_result(self, error: ProviderError, attempts: int) -> ProviderResult:
        """The failed result to hand back, keeping the adapter's record where it made one.

        The sentence names the number of attempts only when there was more
        than one, so a failure that was never worth retrying reads as the
        plain fact it is.
        """
        message = str(error)
        if attempts > 1:
            message = f"{self.provider.display_name} failed after {attempts} attempts: {message}"
        result = error.result
        if result is None:
            return ProviderResult(provider=self.provider, error=message)
        result.error = message
        if result.request is not None:
            result.request = replace(result.request, error=message, succeeded=False)
        return result


#: The longest a ``Retry-After`` header is obeyed for. Anything longer is
#: treated as this, and the attempt after it is the last the service gets.
MAXIMUM_RETRY_AFTER_SECONDS = 120.0


def describe_api_key_characters(service_name: str, api_key: str) -> str | None:
    """Why this key cannot be sent, or None if every character in it can.

    A key travels in a request header, and a header carries only ASCII. A
    key with an en dash in place of a hyphen, which is what copying through
    Word, Outlook or a notes app does, therefore fails inside the client
    library before anything is sent, and the library's complaint about a
    codec reads as if the service could not be reached. Checking first lets
    the report point at the key instead of the network.

    The message gives the position of the first such character and never
    the character or any other part of the key, because this sentence is
    shown in the review report, written to the log and kept in the run
    folder.
    """
    for index, character in enumerate(api_key):
        if not character.isascii():
            return (
                f"the {service_name} API key has a character that is not allowed "
                f"at position {index + 1}. A key holds only plain letters, digits "
                "and punctuation, and copying it through Word, Outlook or a notes "
                "app can turn a hyphen into a dash. Copy the key again from the "
                "service's own website or portal"
            )
    return None


def wait_unless_cancelled(seconds: float, cancelled: CancelCheck | None) -> bool:
    """Sleep for the backoff, a little at a time, and return False if the run was stopped.

    Slept in short pieces rather than all at once so that a cancel during a
    long wait is noticed within a second rather than at the end of it.
    """
    remaining = max(0.0, seconds)
    while remaining > 0:
        if cancelled is not None and cancelled():
            return False
        piece = min(1.0, remaining)
        _sleep(piece)
        remaining -= piece
    return not (cancelled is not None and cancelled())


def _sleep(seconds: float) -> None:
    """Kept separate so a test can take the waiting out."""
    time.sleep(seconds)


#: How long a service is given to process audio before we stop waiting,
#: beyond the configured timeout. A fixed margin plus one second of waiting
#: per second of audio is several times slower than any of these services
#: has been seen to be, and the cap stops a wrong duration from hanging a
#: thread all night. See :func:`timeout_for_duration`.
TIMEOUT_MARGIN_SECONDS = 300.0
TIMEOUT_SECONDS_PER_AUDIO_SECOND = 1.0
MAXIMUM_TIMEOUT_SECONDS = 7200.0


def timeout_for_duration(configured_seconds: float, duration_seconds: float | None) -> float:
    """The time to allow a service for a piece of audio, grown to fit the audio.

    The configured timeout is a single number for every request. That fits
    a ten-minute chunk and not a three-hour file: a service that is silent
    until the transcript is ready takes longer than the timeout on a long
    file and is abandoned while it is still working, and then the retry
    sends the same file again to be abandoned again. So the allowance is
    whichever is larger, the configured timeout or a margin plus one second
    per second of audio, capped at two hours.
    """
    scaled = TIMEOUT_MARGIN_SECONDS + TIMEOUT_SECONDS_PER_AUDIO_SECOND * max(
        0.0, duration_seconds or 0.0
    )
    return max(float(configured_seconds), min(MAXIMUM_TIMEOUT_SECONDS, scaled))


def retry_after_seconds(headers: Any) -> float | None:
    """Read a ``Retry-After`` header, in seconds, from whatever holds the headers.

    Only the delay-seconds form is read. The HTTP-date form is rare from
    these services and getting the clock arithmetic wrong would wait for the
    wrong length of time, so it is treated as absent and the ordinary
    backoff applies.
    """
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if getter is None:
        return None
    for name in ("retry-after", "Retry-After", "retry-after-ms", "Retry-After-Ms"):
        try:
            value = getter(name)
        except Exception:  # noqa: BLE001 - a headers object of unknown shape
            return None
        if value is None:
            continue
        try:
            number = float(str(value).strip())
        except ValueError:
            return None
        return number / 1000.0 if name.lower().endswith("-ms") else number
    return None


class ForcedAligner(ABC):
    """Measuring where known words fall in known audio.

    Kept behind its own interface because the service that does this today
    is not the one that will do it forever, and because it cannot do
    Afrikaans. A span it cannot handle keeps its inherited timing and says
    plainly that the timing is approximate, which leaves room for an
    Afrikaans-capable aligner to be dropped in later without anything else
    changing.
    """

    provider: Provider

    @abstractmethod
    def is_configured(self) -> bool: ...

    @abstractmethod
    def supports(self, language: Language) -> bool: ...

    @abstractmethod
    def align(
        self,
        audio_path: Path,
        text: str,
        language: Language,
        canonical_offset: float = 0.0,
    ) -> list[tuple[str, float, float]]:
        """Return each word with its start and end in canonical time.

        Raises :class:`ProviderError` when it cannot. The caller must treat
        that as "keep the span you had and mark it approximate", never as a
        reason to invent boundaries.

        An aligner that also knows how well each word matched may return a
        fourth element carrying that figure, and it is worth returning where
        it exists: it is what lets the caller tell a measurement it can
        trust from one it should only call approximate. The convention is a
        **loss**, where a smaller number is better, which is the shape
        ElevenLabs uses and the opposite of the log probabilities on the
        transcription side. That inversion is easy to get backwards and
        would silently mark the worst alignments as the best, so anything
        reading it should say out loud which way it runs.

        The fourth element is optional so that this signature stays honest
        for an aligner that does not report one, and so that adding it later
        does not break an implementation already written. The timing rules
        accept three-tuples, four-tuples, dictionaries and objects, taking a
        loss wherever one is offered.
        """
