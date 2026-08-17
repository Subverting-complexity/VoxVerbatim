"""Talking to AssemblyAI, the only service here that will transcribe part of a file.

AssemblyAI is a strong all-rounder: it times its words, tells speakers apart,
reports a confidence for every word, takes a list of terms to listen out for
and takes a sentence or two of context. It is also the only one of the three
languages this application handles that will attempt Afrikaans at all.

The reason it matters more than that sounds is escalation. When two services
disagree about a word, the way to settle it is to ask somebody again about
just that moment of audio. Everywhere else that means cutting a clip out of
the recording, writing it to disk and uploading it, which is slow and adds a
second timeline to keep straight. AssemblyAI takes ``audio_start_from`` and
``audio_end_at`` instead, so the canonical recording can be uploaded once and
then asked about a hundred times, a few seconds at a time. That is what
:meth:`AssemblyAiProvider.transcribe_window` is for, and it is why this
adapter alone declares ``time_window``.

Three details of the service catch people out, and all three are handled here
rather than left to the caller.

The first is that ``speech_model``, singular, no longer exists. Sending a
current model name to it returns HTTP 400. Its replacement, ``speech_models``,
is an ordered fallback chain rather than a choice: the service tries the first
model and falls through to the next when the first cannot serve the request,
which is usually because of the language. That is exactly the Afrikaans path,
so the two models the user configures, a main one and an Afrikaans one, are
joined into one list here rather than turned into two separate calls. And
because the chain means the model that ran is not necessarily the model that
was asked for, provenance records ``speech_model_used``, which is the only
thing that tells you afterwards whether a recording fell through.

The second is that diarisation quietly needs punctuation. Setting
``speaker_labels`` without ``punctuate`` does not fail: it returns a transcript
with no speakers in it and no explanation. This adapter therefore turns
punctuation on itself whenever it asks for speakers, rather than trusting a
caller to remember a dependency that announces itself only by an absence.

The third is the units. Word times are milliseconds, and the file duration on
the same response is seconds. Treating them alike is a factor of a thousand
and is the classic mistake on this API, so the words are converted and the
duration is not.

One warning about Afrikaans, since this is the only service that offers it.
It runs only on Universal-2, and AssemblyAI itself files it as moderate
accuracy, meaning a word error rate somewhere between a quarter and a half.
Treat what comes back as evidence that informs a decision, never as an
authority that settles one.

The client library is imported inside the method that needs it, so a machine
without the ``assemblyai`` package loses this one service rather than failing
to start.
"""

from __future__ import annotations

import logging
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from audio_transcriber.transcription.model import (
    AudioSpan,
    ChunkRecord,
    Language,
    Provider,
    ProviderRequestRecord,
    ProviderResult,
    ProviderToken,
)
from audio_transcriber.transcription.providers.base import (
    CancelCheck,
    ProviderCapabilities,
    ProviderError,
    ProviderNotConfigured,
    ProviderUnavailable,
    TranscriptionProvider,
    TranscriptionRequest,
)

_log = logging.getLogger(__name__)

#: The package that must be installed for this service to work at all.
PACKAGE = "assemblyai"

#: The model tried first. ``universal-3-pro`` is deprecated and retires on
#: 2 September 2026, so nothing here should default to it.
DEFAULT_MODEL = "universal-3-5-pro"

#: The model the chain falls through to. It is the only model of theirs that
#: speaks Afrikaans at all, which is the whole reason there are two settings.
DEFAULT_AFRIKAANS_MODEL = "universal-2"

#: Key terms are limited to 1000 on the newest model and 200 on Universal-2.
#: The lower number is the one to design against, because the chain can fall
#: through to Universal-2 at any moment and a list that was legal for the
#: first model would then be rejected by the second.
DEFAULT_MAXIMUM_KEYTERMS = 200

#: Each key term may be at most six words. A longer one is dropped rather than
#: cut short, because half of a phrase is not a shorter phrase, it is a
#: different one.
MAXIMUM_KEYTERM_WORDS = 6

_GIGABYTE = 1024 * 1024 * 1024

#: Uploads are limited to 2.2 GB. Submitting an already-hosted URL allows 5 GB,
#: but this application uploads, so the smaller number is the real one.
MAXIMUM_FILE_BYTES = int(2.2 * _GIGABYTE)

#: Ten hours, far longer than anything this application expects.
MAXIMUM_DURATION_SECONDS = 10 * 60 * 60

ASSEMBLYAI_CAPABILITIES = ProviderCapabilities(
    word_timings=True,
    diarisation=True,
    speaker_count_hint=True,
    word_confidence=True,
    language_detection=True,
    # False: the language it reports belongs to the file, not to the word.
    per_word_language=False,
    vocabulary_biasing=True,
    context_prompt=True,
    # True, and uniquely so. This is what lets escalation ask about six
    # seconds of a two-hour recording without cutting a clip first.
    time_window=True,
    forced_alignment=False,
    languages=frozenset({Language.ENGLISH, Language.GERMAN, Language.AFRIKAANS}),
    maximum_file_bytes=MAXIMUM_FILE_BYTES,
    maximum_duration_seconds=MAXIMUM_DURATION_SECONDS,
    accepted_containers=frozenset(
        {"wav", "mp3", "flac", "m4a", "aac", "ogg", "opus", "webm", "mp4"}
    ),
)

#: Settings keys that could hold a credential. Anything named like one of
#: these is dropped before a request is written into provenance, because a
#: transcript folder is copied around and read by people.
_CREDENTIAL_NAMES = ("api_key", "apikey", "key", "token", "secret", "password", "authorization")

#: Errors worth trying again even though they carry no HTTP status, because
#: they happened before the far end answered at all.
_RETRYABLE_ERROR_NAMES = ("timeout", "connection")


class AssemblyAiProvider(TranscriptionProvider):
    """The AssemblyAI speech-to-text service.

    Everything it needs arrives through the constructor. It knows nothing
    about the application's settings file: a separate registry reads Settings
    and calls this, which keeps a settings change from rippling through the
    adapters and lets a test build a working adapter out of a fake client.

    The client this adapter talks to is a very small thing. It transcribes,
    given a piece of audio and a dictionary of parameters, and it uploads a
    file and returns a URL. The real one, built on first use, wraps the
    package's ``Transcriber`` and turns the dictionary into the library's
    configuration object; a test hands in anything with those two methods.
    That indirection exists because the parameters this adapter has to send
    include ones the installed library may be too old to know about, and the
    library's own escape hatch is the documented way to send them.
    """

    provider = Provider.ASSEMBLYAI
    capabilities = ASSEMBLYAI_CAPABILITIES

    def __init__(
        self,
        api_key: str = "",
        model: str = DEFAULT_MODEL,
        afrikaans_model: str = DEFAULT_AFRIKAANS_MODEL,
        parameters: dict[str, Any] | None = None,
        *,
        language_confidence_threshold: float | None = None,
        maximum_keyterms: int = DEFAULT_MAXIMUM_KEYTERMS,
        timeout_seconds: float = 900.0,
        client: Any | None = None,
    ) -> None:
        """Set up the adapter.

        ``model`` and ``afrikaans_model`` stay two separate settings because
        that is how a person thinks about them, and they are joined into one
        ordered chain when a request is built.

        ``language_confidence_threshold`` makes the service refuse rather than
        proceed when it is unsure which language it is hearing. That is useful
        as a guard, because a confident transcript in the wrong language is
        far more expensive to discover than a request that failed.

        ``client`` exists so a test, or a caller that wants to share one
        connection, can hand in a ready-made client. Left as ``None`` it means
        "build a real one when first needed", which is what keeps the library
        import out of application start-up.
        """
        self._api_key = api_key or ""
        self._model = model or DEFAULT_MODEL
        self._afrikaans_model = afrikaans_model or DEFAULT_AFRIKAANS_MODEL
        self._parameters = dict(parameters or {})
        self._language_confidence_threshold = language_confidence_threshold
        self._maximum_keyterms = max(0, maximum_keyterms)
        self._timeout_seconds = timeout_seconds
        self._client = client

    # -- What this service is ------------------------------------------

    @property
    def model_identifier(self) -> str:
        return self._model

    def is_configured(self) -> bool:
        return self.describe_configuration_problem() is None

    def describe_configuration_problem(self) -> str | None:
        if not self._api_key.strip():
            return "no API key has been entered"
        if not self._model.strip():
            return "no model has been chosen"
        return None

    def speech_models_for(self, request: TranscriptionRequest) -> tuple[str, ...]:
        """The ordered chain of models to offer for this request.

        The service tries these in turn and uses the first that can serve the
        request, so the Afrikaans model belongs on the end of every chain
        rather than in a request of its own: it engages only when the model in
        front of it cannot handle the language, which is precisely the rule we
        want and precisely what we would otherwise have to write ourselves.

        A ``model_override`` on the request takes the front of the chain, which
        is how a span already known to be Afrikaans can be sent straight to the
        model that speaks it without losing the fallback behind it.
        """
        chain: list[str] = []
        for name in (request.model_override, self._model, self._afrikaans_model):
            if name and name.strip() and name not in chain:
                chain.append(name)
        return tuple(chain)

    # -- Doing the work ------------------------------------------------

    def _transcribe(
        self,
        request: TranscriptionRequest,
        cancelled: CancelCheck | None = None,
    ) -> ProviderResult:
        return self.transcribe_window(request, cancelled=cancelled)

    def upload(self, audio_path: Path) -> str:
        """Upload a recording once and return the URL to ask about it again.

        This is half of the escalation path. The canonical recording goes up a
        single time, and every later question about a disputed moment names
        this URL and a window inside it, which costs nothing to prepare and
        leaves no temporary clips behind.
        """
        problem = self.describe_configuration_problem()
        if problem is not None:
            raise ProviderNotConfigured(self.provider, problem)

        size_bytes = _file_size(audio_path)
        if size_bytes > MAXIMUM_FILE_BYTES:
            raise ProviderError(
                f"{audio_path.name} is {size_bytes / _GIGABYTE:.1f} GB, and AssemblyAI "
                f"accepts uploads of at most {MAXIMUM_FILE_BYTES / _GIGABYTE:.1f} GB."
            )

        client = self._resolve_client()
        uploader = getattr(client, "upload_file", None)
        if uploader is None:
            # Losing this costs only the cheap windowed path, so it is worth
            # saying plainly rather than falling over: escalation can still cut
            # a clip and send it like every other service does.
            raise ProviderError(
                "The installed assemblyai package cannot upload a file on its own, so "
                "audio cannot be reused across windowed requests."
            )
        try:
            return str(uploader(str(audio_path)))
        except Exception as error:  # noqa: BLE001 - converted at this boundary
            message, retryable, status = _describe_failure(error)
            raise ProviderError(message, retryable=retryable, status_code=status) from error

    def transcribe_window(
        self,
        request: TranscriptionRequest,
        *,
        audio_url: str | None = None,
        window: AudioSpan | None = None,
        cancelled: CancelCheck | None = None,
    ) -> ProviderResult:
        """Transcribe a whole file, or one window of an already-uploaded one.

        With neither ``audio_url`` nor ``window``, this is an ordinary full
        pass over ``request.audio_path``. With both, it asks about a few
        seconds of audio that is already on the service, which is what makes
        escalation cheap enough to do many times over.

        ``window`` is in canonical seconds and is converted to the
        milliseconds the service wants. It defaults to ``request.window``, so a
        caller that has already described the region on the request does not
        have to say it twice.

        The times that come back are shifted by ``request.canonical_offset``,
        as everywhere else. Where the window is taken out of the canonical
        recording itself the service still counts from the start of that
        recording, so the caller leaves the offset at zero; the offset is there
        for the other case, where the audio sent was physically cut.
        """
        problem = self.describe_configuration_problem()
        if problem is not None:
            # Raised before anything is opened or sent, so that a missing key
            # is reported as the plain fact it is rather than coming back as
            # an authentication failure from the far end.
            raise ProviderNotConfigured(self.provider, problem)

        client = self._resolve_client()

        if cancelled is not None and cancelled():
            raise ProviderError("AssemblyAI was not called, because the run was stopped.")

        span = window if window is not None else request.window
        audio: Any = audio_url
        size_bytes = 0
        if audio is None:
            size_bytes = _file_size(request.audio_path)
            if size_bytes > MAXIMUM_FILE_BYTES:
                raise ProviderError(
                    f"{request.audio_path.name} is {size_bytes / _GIGABYTE:.1f} GB, and "
                    f"AssemblyAI accepts at most {MAXIMUM_FILE_BYTES / _GIGABYTE:.1f} GB. "
                    "It has to be cut into chunks first."
                )
            audio = str(request.audio_path)

        keyterms = self._keyterms_for(request.vocabulary_terms)
        parameters = self._parameters_for(request, keyterms=keyterms, window=span)

        started_at = datetime.now()
        try:
            transcript = client.transcribe(audio, parameters)
        except Exception as error:  # noqa: BLE001 - every failure becomes a result
            message, retryable, status = _describe_failure(error)
            return self._failed(
                request,
                parameters=parameters,
                keyterms=keyterms,
                window=span,
                size_bytes=size_bytes,
                started_at=started_at,
                transcript_id=None,
                message=message,
                retryable=retryable,
                status=status,
                # The library raised rather than returning, so there is no
                # body to keep.
                raw_response=None,
            )

        status_text = str(_field(transcript, "status") or "").lower()
        if status_text.endswith("error") or _field(transcript, "error"):
            # The library reports a rejected job through the returned object
            # rather than by raising, so this is a failure like any other and
            # has to be looked for deliberately.
            message = (
                f"AssemblyAI could not transcribe the audio: "
                f"{_field(transcript, 'error') or 'no reason was given'}"
            )
            return self._failed(
                request,
                parameters=parameters,
                keyterms=keyterms,
                window=span,
                size_bytes=size_bytes,
                started_at=started_at,
                transcript_id=_as_text(_field(transcript, "id")),
                message=message,
                retryable=False,
                status=None,
                # A rejected job did come back with a body, and it is the only
                # account of why the service would not do the work.
                raw_response=_raw_response_of(transcript),
            )

        tokens = self._tokens_from(transcript, request)
        record = self._request_record(
            request,
            parameters=parameters,
            keyterms=keyterms,
            window=span,
            size_bytes=size_bytes,
            started_at=started_at,
            transcript_id=_as_text(_field(transcript, "id")),
            model_used=_as_text(_field(transcript, "speech_model_used")),
            audio_duration_seconds=_as_float(_field(transcript, "audio_duration")),
            succeeded=True,
            error=None,
        )
        return ProviderResult(
            provider=self.provider,
            tokens=tokens,
            request=record,
            detected_language=Language.from_code(_as_text(_field(transcript, "language_code"))),
            speakers=_speakers_in(tokens),
            # The whole answer as the library received it, taken rather than
            # rebuilt. The pipeline hands it to the transcript store, which
            # writes it to a file of its own; this adapter never writes
            # anything and knows nothing about where transcripts are kept.
            raw_response=_raw_response_of(transcript),
        )

    # -- Turning the answer into evidence -------------------------------

    def _tokens_from(self, transcript: Any, request: TranscriptionRequest) -> list[ProviderToken]:
        """Convert every word the service returned onto the canonical timeline.

        Word times are milliseconds here, so they are divided by a thousand
        before the canonical offset is added. The ``audio_duration`` on the
        same response is already in seconds and is deliberately handled
        somewhere else, because converting it too would be wrong by exactly
        the factor that makes this bug hard to see.

        Speaker labels are kept exactly as the service gave them, which is
        single letters. Renaming them into anything a person recognises happens
        much later, against the whole transcript, and doing it here would throw
        away the one thing that connects a word to the service's own decision.
        """
        tokens: list[ProviderToken] = []
        for index, word in enumerate(_field(transcript, "words") or []):
            text = _field(word, "text") or ""
            tokens.append(
                ProviderToken(
                    provider=self.provider,
                    index=index,
                    text=text,
                    start=_canonical_seconds(
                        _as_float(_field(word, "start")), request.canonical_offset
                    ),
                    end=_canonical_seconds(
                        _as_float(_field(word, "end")), request.canonical_offset
                    ),
                    speaker=_as_text(_field(word, "speaker")),
                    confidence=_as_float(_field(word, "confidence")),
                    # Left unknown on purpose. The service reports one language
                    # for the file, and copying it onto every word would
                    # manufacture per-span evidence it never gave.
                    language=Language.UNKNOWN,
                    chunk_index=request.chunk_index,
                    is_punctuation=_is_punctuation_only(text),
                )
            )
        return tokens

    # -- Building the request -------------------------------------------

    def _keyterms_for(self, terms: tuple[str, ...]) -> tuple[str, ...]:
        """The terms to listen out for, cut to what the service will accept.

        Terms of more than six words are dropped, because that is the limit and
        a shortened phrase is a different phrase. The list is then cut to the
        limit of the smaller model in the chain. The vocabulary arrives ordered
        most specific first, which is what makes taking the head of it the
        right way to cut.
        """
        kept: list[str] = []
        for term in terms:
            cleaned = (term or "").strip()
            if not cleaned:
                continue
            if len(cleaned.split()) > MAXIMUM_KEYTERM_WORDS:
                _log.info(
                    "The term %r was not sent to AssemblyAI: it is longer than the "
                    "%d words a key term may have.",
                    cleaned,
                    MAXIMUM_KEYTERM_WORDS,
                )
                continue
            kept.append(cleaned)
        if len(kept) <= self._maximum_keyterms:
            return tuple(kept)
        _log.info(
            "Sending AssemblyAI the first %d of %d terms, which is all the smallest "
            "model in the chain accepts.",
            self._maximum_keyterms,
            len(kept),
        )
        return tuple(kept[: self._maximum_keyterms])

    def _parameters_for(
        self,
        request: TranscriptionRequest,
        *,
        keyterms: tuple[str, ...],
        window: AudioSpan | None,
    ) -> dict[str, Any]:
        """Everything that goes in the request, as one plain dictionary.

        Kept as a dictionary rather than as the library's configuration object
        so that the whole of it can be recorded in provenance and checked in a
        test without the package installed, and so that a parameter the
        installed library has never heard of still reaches the service.
        """
        parameters: dict[str, Any] = {
            # Plural. The singular field was removed and returns 400.
            "speech_models": list(self.speech_models_for(request)),
            # The verbatim switch. Without it the service removes the fillers
            # and false starts, and the layer this evidence feeds is defined
            # as the one that keeps them.
            "disfluencies": True,
            "punctuate": True,
        }

        if request.diarise:
            parameters["speaker_labels"] = True
            # Enforced here rather than trusted to the caller, because turning
            # punctuation off does not fail: it returns a transcript with no
            # speakers in it and says nothing at all about why.
            parameters["punctuate"] = True
            if request.expected_speaker_count > 1:
                parameters["speakers_expected"] = request.expected_speaker_count

        language_code = _single_language_code(request.languages)
        if language_code is not None:
            parameters["language_code"] = language_code
        else:
            # Detection here also chooses the model that runs, so it is not
            # merely a label on the answer.
            parameters["language_detection"] = True
            if self._language_confidence_threshold is not None:
                parameters["language_confidence_threshold"] = self._language_confidence_threshold

        if keyterms:
            parameters["keyterms_prompt"] = list(keyterms)
        if request.context_prompt.strip():
            parameters["prompt"] = request.context_prompt.strip()

        if window is not None:
            # Milliseconds, and integers. This is the whole of the cheap
            # escalation path: the audio stays where it is and only the
            # question changes.
            parameters["audio_start_from"] = int(round(window.start * 1000))
            parameters["audio_end_at"] = int(round(window.end * 1000))

        extras = {**self._parameters, **request.extra_parameters}
        for key, value in extras.items():
            if key == "speech_model":
                _log.warning(
                    "The speech_model parameter was not sent to AssemblyAI. It was "
                    "replaced by speech_models, and sending it returns an error."
                )
                continue
            if key == "punctuate" and not value and parameters.get("speaker_labels"):
                _log.warning(
                    "Punctuation was left on for AssemblyAI, because turning it off "
                    "silently switches the speaker labels off with it."
                )
                continue
            parameters[key] = value
        return parameters

    # -- Recording what happened ----------------------------------------

    def _failed(
        self,
        request: TranscriptionRequest,
        *,
        parameters: dict[str, Any],
        keyterms: tuple[str, ...],
        window: AudioSpan | None,
        size_bytes: int,
        started_at: datetime,
        transcript_id: str | None,
        message: str,
        retryable: bool,
        status: int | None,
        raw_response: Any = None,
    ) -> ProviderResult:
        """Report a failure as an ordinary result rather than an exception.

        Whatever the service said while refusing is carried out with the
        failure, because an explanation is worth keeping and this is the only
        route by which it can reach the folder beside the recording.
        """
        _log.warning("AssemblyAI did not answer: %s", message)
        _log.debug("AssemblyAI failure: retryable=%s status=%s", retryable, status)
        record = self._request_record(
            request,
            parameters=parameters,
            keyterms=keyterms,
            window=window,
            size_bytes=size_bytes,
            started_at=started_at,
            transcript_id=transcript_id,
            model_used=None,
            audio_duration_seconds=None,
            succeeded=False,
            error=message,
        )
        return ProviderResult(
            provider=self.provider,
            request=record,
            error=message,
            raw_response=raw_response,
        )

    def _request_record(
        self,
        request: TranscriptionRequest,
        *,
        parameters: dict[str, Any],
        keyterms: tuple[str, ...],
        window: AudioSpan | None,
        size_bytes: int,
        started_at: datetime,
        transcript_id: str | None,
        model_used: str | None,
        audio_duration_seconds: float | None,
        succeeded: bool,
        error: str | None,
    ) -> ProviderRequestRecord:
        """Note what was asked and what came back, without the credential.

        The model version recorded is the model that actually ran, not the
        chain that was offered. Those are different things whenever the first
        model could not serve the language, and the difference is the only
        warning that a passage was transcribed by the weaker model.
        """
        recorded = {
            key: value for key, value in parameters.items() if key != "keyterms_prompt"
        }
        # Named for what it counts rather than for the service's word for it,
        # because anything with "key" in its name is stripped out of a record
        # on the way to disk and a count of terms would vanish with it.
        recorded["vocabulary_term_count"] = len(keyterms)
        if transcript_id:
            recorded["response_transcript_id"] = transcript_id
        if model_used:
            recorded["response_speech_model_used"] = model_used
        if audio_duration_seconds is not None:
            # Already seconds on this response, unlike the word times beside
            # it, so it is written down exactly as it arrived.
            recorded["response_audio_duration_seconds"] = audio_duration_seconds

        chunk = ChunkRecord(
            provider=self.provider,
            chunk_index=request.chunk_index,
            canonical_start=window.start if window else request.canonical_offset,
            canonical_end=(
                window.end if window else request.canonical_offset + (request.duration or 0.0)
            ),
            canonical_offset=request.canonical_offset,
            encoded_size_bytes=size_bytes,
            path=str(request.audio_path),
            provider_request_id=transcript_id,
        )
        return ProviderRequestRecord(
            provider=self.provider,
            model_identifier=self._model,
            model_version=model_used,
            request_parameters=_without_credentials(recorded),
            language_configuration=str(parameters.get("language_code") or "auto"),
            vocabulary_terms=keyterms,
            chunks=(chunk,),
            processing_seconds=(datetime.now() - started_at).total_seconds(),
            started_at=started_at.isoformat(timespec="seconds"),
            succeeded=succeeded,
            error=error,
            purpose=request.purpose,
        )

    def _resolve_client(self) -> Any:
        """The client to call, built on first use unless one was handed in."""
        if self._client is None:
            self._client = _build_client(self._api_key, self._timeout_seconds)
        return self._client


# -- The real client -----------------------------------------------------


class _SdkClient:
    """The package's ``Transcriber``, behind the two methods this adapter uses.

    It exists for one reason. The parameters that have to be sent include
    ``speech_models``, which an installed copy of the library may predate, and
    passing an unknown argument to the library's configuration object raises
    rather than being sent. The library's documented escape hatch is its raw
    configuration, which is declared to accept extra fields, so everything goes
    through that and nothing depends on how new the installed package is.
    """

    def __init__(self, api_key: str, timeout_seconds: float) -> None:
        try:
            import assemblyai
        except ImportError as error:
            raise ProviderUnavailable(Provider.ASSEMBLYAI, PACKAGE) from error
        assemblyai.settings.api_key = api_key
        self._assemblyai = assemblyai
        self._timeout_seconds = timeout_seconds

    def transcribe(self, audio: Any, parameters: dict[str, Any]) -> Any:
        configuration = self._assemblyai.TranscriptionConfig()
        for key, value in parameters.items():
            setattr(configuration.raw, key, value)
        return self._assemblyai.Transcriber(config=configuration).transcribe(audio)

    def upload_file(self, audio_path: str) -> str:
        return str(self._assemblyai.Transcriber().upload_file(audio_path))


def _build_client(api_key: str, timeout_seconds: float) -> Any:
    """Build a real client, reporting a missing package as such.

    The import lives inside the client rather than at the top of the module so
    that the application starts on a machine where the package was never
    installed. That machine loses AssemblyAI and is told why.
    """
    return _SdkClient(api_key, timeout_seconds)


# -- Helpers -------------------------------------------------------------


def _canonical_seconds(milliseconds: float | None, offset: float) -> float | None:
    """Turn one of the service's word times into a canonical one.

    Milliseconds to seconds, then the offset of whatever piece of audio this
    was. Only the word times go through here. The file duration on the same
    response is already in seconds, and sending it through this function would
    be wrong by a factor of a thousand in a way nothing would notice.
    """
    if milliseconds is None:
        return None
    return (milliseconds / 1000.0) + offset


def _raw_response_of(transcript: Any) -> Any:
    """The service's own answer, as close to untouched as it can be had.

    The library keeps the complete JSON it was sent beside the object it built
    from it, so that is what is taken. Rebuilding the response out of the
    parsed object field by field would defeat the purpose of keeping it: what
    is wanted is evidence of what the service said, and a reconstruction is
    evidence of what this adapter understood instead. Older versions without
    that field fall back to the object, which the store can serialise.
    """
    raw = _field(transcript, "json_response")
    return transcript if raw is None else raw


def _single_language_code(languages: tuple[Language, ...]) -> str | None:
    """The one language code to send, or None to let the service detect it.

    Naming a language helps when there is only one it can be. Naming one of
    several would be worse than saying nothing, because it would push the
    service away from the others and, on this service, would also decide which
    model runs.
    """
    known = [language for language in languages if language is not Language.UNKNOWN]
    if len(known) == 1:
        return known[0].value
    return None


def _speakers_in(tokens: list[ProviderToken]) -> list[str]:
    """The speaker labels this service used, in the order they first spoke."""
    seen: list[str] = []
    for token in tokens:
        if token.speaker and token.speaker not in seen:
            seen.append(token.speaker)
    return seen


def _is_punctuation_only(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return all(unicodedata.category(character).startswith("P") for character in stripped)


def _field(source: Any, name: str, default: Any = None) -> Any:
    """Read a field from the answer, whichever shape the library gives it in."""
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _as_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError as error:
        raise ProviderError(f"{path} could not be read: {error}") from error


def _without_credentials(parameters: dict[str, Any]) -> dict[str, Any]:
    """Strip anything that looks like a secret out of a recorded parameter set.

    A request record is written into the recording's folder, which is copied
    around and read by people. An API key that reached it would be a leak
    nobody would ever notice.
    """
    cleaned: dict[str, Any] = {}
    for key, value in parameters.items():
        lowered = str(key).lower()
        if any(name in lowered for name in _CREDENTIAL_NAMES):
            continue
        cleaned[key] = _without_credentials(value) if isinstance(value, dict) else value
    return cleaned


def _describe_failure(error: Exception) -> tuple[str, bool, int | None]:
    """Turn whatever the library threw into a sentence, and say if it may be retried.

    The exception classes are not imported. Reading the status off the error
    covers every class the library has, and doing it this way means the module
    never imports the package merely to catch its errors, which would defeat
    the point of importing it lazily.
    """
    status = _as_int(getattr(error, "status_code", None))
    name = type(error).__name__
    retryable = False
    if status is not None:
        retryable = status in (408, 409, 429) or 500 <= status < 600
    elif any(hint in name.lower() for hint in _RETRYABLE_ERROR_NAMES):
        retryable = True

    detail = str(error).strip() or name
    if status is not None:
        return f"AssemblyAI answered with an error ({status}): {detail}", retryable, status
    return f"AssemblyAI could not be reached: {detail}", retryable, None
