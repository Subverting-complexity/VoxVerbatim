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
failed :class:`~audio_transcriber.transcription.model.ProviderResult`, so
the pipeline never has to guard every call.

Nothing here imports a provider library. The adapters do that inside their
own modules, so a missing package disables one service rather than stopping
the application from starting.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from audio_transcriber.transcription.model import (
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

    def __init__(self, message: str, *, retryable: bool = False, status_code: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


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

    def transcribe(
        self,
        request: TranscriptionRequest,
        cancelled: CancelCheck | None = None,
    ) -> ProviderResult:
        """Transcribe one piece of audio, never raising.

        A service that fails comes back as a result carrying the reason. The
        pipeline reports it, notes it in the transcript's warnings, and
        carries on with the services that did answer.
        """
        try:
            return self._transcribe(request, cancelled)
        except ProviderError as error:
            _log.warning("%s did not answer: %s", self.provider.display_name, error)
            return ProviderResult(provider=self.provider, error=str(error))
        except Exception as error:  # a broken adapter must not take the run down
            _log.exception("%s raised an unexpected error.", self.provider.display_name)
            return ProviderResult(
                provider=self.provider,
                error=f"{self.provider.display_name} failed unexpectedly: {error}",
            )


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
