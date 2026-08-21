"""Talking to Deepgram, which is fast, precise about time, and cannot do Afrikaans.

Deepgram times every word to the millisecond, gives each one a confidence,
tells speakers apart, and answers quickly enough that it is worth calling on
every recording. It also keeps the best record of itself of any service here:
its answer names the exact dated build of the model that produced it, so a
transcript made today can still be explained after the model behind the name
has moved on twice. That is written into provenance, and it is the reason this
adapter reads the metadata as carefully as it reads the words.

What it will not do is Afrikaans. Not on ``nova-3``, not on any other model of
theirs. Among the five services this application uses, only AssemblyAI's
Universal-2 attempts it at all, so an Afrikaans span never reaches this module.

Four things about the request are easy to get wrong, and all four are settled
here rather than left to configuration.

The first is the model. The default is an old one, so it is always named
explicitly.

The second is the language. Naming a single language makes the service
transcribe only that language and ignore anything else in the recording, which
in a bilingual meeting means quietly losing half of it. Where more than one
language is possible the request asks for ``multi`` instead, which covers ten
languages including German and lets the recording switch mid-sentence.

The third is diarisation, where two parameters do the same job and having both
is not a warning but an outright rejection. ``diarize`` is the deprecated one.
This adapter sends ``diarize_model`` and strips ``diarize`` out of whatever it
is given, so that a leftover setting cannot break every request.

The fourth is the key-term list, which is capped and where going over the cap
is an error rather than a truncation: send one term too many and the whole
request fails. The list is therefore cut down here, and the log says how much
was left out, because a term that was silently dropped is a name that will
come back misspelled with no explanation.

Times here are floating-point seconds, which makes Deepgram the odd one out
against Microsoft's and AssemblyAI's milliseconds. There is still a conversion
to do, because the canonical offset of the piece of audio has to be added.

One failure has a particular meaning. Deepgram publishes no maximum duration,
but a request that takes more than ten minutes to process comes back as a 504.
That is a budget on processing rather than a limit on length, so it says
nothing reliable about how long a recording may be, and it is reported as what
it is rather than as a generic failure.

The client library is imported inside the method that needs it, so a machine
without the ``deepgram-sdk`` package loses this one service rather than failing
to start.

How the request reaches that library needs a word, because the library changed
shape underneath it. Version 5 replaced the old accessor chain and the loose
options dictionary with one generated method whose parameters are fixed at
whatever version happens to be installed, and passing it a name it does not
declare raises rather than sending anything. Settings promises the opposite: a
parameter typed into the provider's parameter list is meant to reach the
service, so that a new option can be used before this application knows about
it. Everything therefore travels as query parameters through the library's own
request options, which is the one route that carries a name the generated
method has never heard of. Versions 3 and 4 cannot be reached that way at all,
which is why the required version is 5 or later.

That library also retries by itself, twice, on any 5xx, and its retries are
broken for a request shaped like this one. The first attempt reads the audio
handle to the end and nothing puts it back, so the repeats post an empty body.
Over a real connection the result is a protocol error carrying no status code
at all, which turns a transient 503 that was worth repeating into a nameless
failure that reads as permanent. That is worth knowing if you go looking: a
test that answers through a mock transport never exercises the wire and so
never shows it.

Its retries are therefore switched off and replaced with two of our own, which
rewind the audio before each attempt and use the same judgement about which
failures are worth repeating that the reported error already uses. Nothing
further out repeats a provider call, so removing these without replacing them
would have left Deepgram with a single attempt at a service that answers 503
under load. The processing-budget 504 is deliberately not among the failures
worth repeating, for the reason given above.
"""

from __future__ import annotations

import logging
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from audio_transcriber.transcription.model import (
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

#: The package that must be installed for this service to work at all. Note
#: that the distribution is named with a hyphen and imported without one.
PACKAGE = "deepgram-sdk"

#: The oldest package this adapter can call. Version 5 rebuilt the library
#: around a generated client, and nothing older exposes the entry point used
#: here. See the module docstring.
MINIMUM_PACKAGE_VERSION = "5.0"

#: The model. The service's own default is an older one, so this is always
#: sent rather than left out.
DEFAULT_MODEL = "nova-3"

#: The language code that lets a recording switch between languages. Ten are
#: covered, German among them.
MULTILINGUAL_CODE = "multi"

#: The diarisation parameter to send. The older ``diarize`` switch still
#: exists, and sending both is rejected outright rather than warned about.
DEFAULT_DIARISE_MODEL = "latest"

#: The service's cap on the key-term list. It counts model tokens rather than
#: words, and one term over the line fails the whole request instead of being
#: trimmed, so the list is cut here with room to spare.
MAXIMUM_KEYTERM_TOKENS = 500

#: How many tokens a word is assumed to cost. Deepgram publishes the limit but
#: not the arithmetic behind it, so this is an estimate rather than their
#: figure. Five is pessimistic on purpose: a vocabulary list is mostly unusual
#: names and part numbers, which cost far more tokens than ordinary words, and
#: overestimating costs a few terms at the tail while underestimating fails
#: the request outright.
TOKENS_PER_WORD = 5

#: The model families that can use a key-term list. Anything else has to fall
#: back on the older ``keywords`` feature, which this application does not use,
#: so a key term sent to one of them is refused or ignored. Matched as a
#: prefix, because the family has variants: ``nova-3-medical`` is a Nova-3.
#: Flux is listed for completeness rather than because it can be reached from
#: here; it is a streaming model and this adapter posts whole recordings.
KEYTERM_MODEL_PREFIXES = ("nova-3", "flux")

_GIGABYTE = 1024 * 1024 * 1024

#: 2 GB. There is no published duration limit, for the reason the module
#: docstring explains, so none is declared.
MAXIMUM_FILE_BYTES = 2 * _GIGABYTE

#: The status a request gets when it spends longer than its processing budget.
PROCESSING_TIMEOUT_STATUS = 504

DEEPGRAM_CAPABILITIES = ProviderCapabilities(
    word_timings=True,
    diarisation=True,
    speaker_count_hint=False,
    word_confidence=True,
    language_detection=True,
    # False. What it reports belongs to the channel rather than to the word,
    # and Microsoft's per-phrase locale is the only per-segment language
    # signal this application treats as real.
    per_word_language=False,
    vocabulary_biasing=True,
    context_prompt=False,
    time_window=False,
    forced_alignment=False,
    # No Afrikaans on any Deepgram model, which is why this set has two
    # languages in it rather than three.
    languages=frozenset({Language.ENGLISH, Language.GERMAN}),
    maximum_file_bytes=MAXIMUM_FILE_BYTES,
    maximum_duration_seconds=None,
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


class DeepgramProvider(TranscriptionProvider):
    """The Deepgram speech-to-text service.

    Everything it needs arrives through the constructor. It knows nothing
    about the application's settings file: a separate registry reads Settings
    and calls this, which keeps a settings change from rippling through the
    adapters and lets a test build a working adapter out of a fake client.
    """

    provider = Provider.DEEPGRAM
    capabilities = DEEPGRAM_CAPABILITIES

    def __init__(
        self,
        api_key: str = "",
        model: str = DEFAULT_MODEL,
        parameters: dict[str, Any] | None = None,
        *,
        language: str | None = None,
        diarise_model: str = DEFAULT_DIARISE_MODEL,
        smart_format: bool = True,
        punctuate: bool = True,
        utterances: bool = False,
        maximum_keyterm_tokens: int = MAXIMUM_KEYTERM_TOKENS,
        timeout_seconds: float = 900.0,
        client: Any | None = None,
    ) -> None:
        """Set up the adapter.

        ``language`` overrides what would otherwise be worked out from each
        request. Left as ``None``, a request that names one language asks for
        that language and a request that names several asks for ``multi``,
        which is nearly always what a real recording needs.

        ``parameters`` holds the free-form extras from Settings, merged with
        the extras on an individual request. The request wins where both name
        the same thing, because it is the more specific of the two.

        ``client`` exists so a test, or a caller that wants to share one
        connection, can hand in a ready-made client. Left as ``None`` it means
        "build a real one when first needed", which is what keeps the library
        import out of application start-up.
        """
        self._api_key = api_key or ""
        self._model = model or DEFAULT_MODEL
        self._parameters = dict(parameters or {})
        self._language = language
        self._diarise_model = diarise_model or DEFAULT_DIARISE_MODEL
        self._smart_format = smart_format
        self._punctuate = punctuate
        self._utterances = utterances
        self._maximum_keyterm_tokens = max(0, maximum_keyterm_tokens)
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

    # -- Doing the work ------------------------------------------------

    def _transcribe(
        self,
        request: TranscriptionRequest,
        cancelled: CancelCheck | None = None,
    ) -> ProviderResult:
        problem = self.describe_configuration_problem()
        if problem is not None:
            # Raised before anything is opened or sent, so that a missing key
            # is reported as the plain fact it is rather than coming back as
            # an authentication failure from the far end.
            raise ProviderNotConfigured(self.provider, problem)

        client = self._resolve_client()

        if cancelled is not None and cancelled():
            raise ProviderError("Deepgram was not called, because the run was stopped.")

        size_bytes = _file_size(request.audio_path)
        if size_bytes > MAXIMUM_FILE_BYTES:
            raise ProviderError(
                f"{request.audio_path.name} is {size_bytes / _GIGABYTE:.1f} GB, and "
                f"Deepgram accepts at most {MAXIMUM_FILE_BYTES // _GIGABYTE} GB. "
                "It has to be cut into chunks first."
            )

        # The model first, because whether key terms may be sent at all
        # depends on it and a setting can change it out from under the
        # constructor's answer.
        model = self._model_for(request)
        keyterms = self._keyterms_for(request.vocabulary_terms, model)
        options = self._options_for(request, model=model, keyterms=keyterms)

        started_at = datetime.now()
        try:
            with request.audio_path.open("rb") as handle:
                # The handle goes through rather than its contents. A recording
                # may be two gigabytes, and reading one into memory to post it
                # would cost more than the transcription does.
                response = client.transcribe(handle, options, timeout=self._timeout_seconds)
        except Exception as error:  # noqa: BLE001 - every failure becomes a result
            message, retryable, status = _describe_failure(error)
            _log.warning("Deepgram did not answer: %s", message)
            _log.debug("Deepgram failure: retryable=%s status=%s", retryable, status)
            record = self._request_record(
                request,
                options=options,
                keyterms=keyterms,
                size_bytes=size_bytes,
                started_at=started_at,
                answer=None,
                succeeded=False,
                error=message,
            )
            # The library raised rather than returning, so there is no body to
            # keep with the failure.
            return ProviderResult(provider=self.provider, request=record, error=message)

        # The whole answer as the library received it, taken rather than
        # rebuilt. The pipeline hands it to the transcript store, which writes
        # it to a file of its own; this adapter never writes anything and knows
        # nothing about where transcripts are kept.
        answer = _raw_response_of(response)
        alternative = _best_alternative(answer)

        if alternative is None:
            # Not a transcript. It may be the acknowledgement that a callback
            # request gets back, which carries a request id and nothing else,
            # or a body that is not what this adapter expects at all. Either
            # way the one thing it must not become is a successful result with
            # no words in it, because that is indistinguishable from a
            # recording of silence and would be reported as one.
            message = (
                "Deepgram answered without a transcript in it. The reply carried no "
                "channel this adapter could read, so there is nothing to say about "
                "what was said. The reply itself is kept beside the transcript."
            )
            _log.warning("%s", message)
            return ProviderResult(
                provider=self.provider,
                request=self._request_record(
                    request,
                    options=options,
                    keyterms=keyterms,
                    size_bytes=size_bytes,
                    started_at=started_at,
                    answer=answer,
                    succeeded=False,
                    error=message,
                ),
                error=message,
                raw_response=answer,
            )

        tokens = self._tokens_from(alternative, request)
        record = self._request_record(
            request,
            options=options,
            keyterms=keyterms,
            size_bytes=size_bytes,
            started_at=started_at,
            answer=answer,
            succeeded=True,
            error=None,
        )
        return ProviderResult(
            provider=self.provider,
            tokens=tokens,
            request=record,
            detected_language=Language.from_code(_detected_language_code(answer)),
            speakers=_speakers_in(tokens),
            raw_response=answer,
        )

    # -- Turning the answer into evidence -------------------------------

    def _tokens_from(self, alternative: Any, request: TranscriptionRequest) -> list[ProviderToken]:
        """Convert every word the service returned onto the canonical timeline.

        Times here are already seconds, so the only conversion is the canonical
        offset. It still has to happen: without it a word from a chunk that
        began ten minutes in points at the wrong sound, and nothing later in
        the application can tell that it does.

        The text taken is the punctuated form where the service supplies one,
        because that is what it actually decided the word was. The plain form
        beside it has had the punctuation and casing stripped, which is a
        comparison form rather than a transcript, and this application makes
        its own comparison forms.

        Speaker labels are integers here and letters at AssemblyAI. Each is
        kept exactly as its own service gave it, because a token records what a
        service said; putting the labels of different services into a common
        vocabulary is a decision, and decisions are made further on.
        """
        tokens: list[ProviderToken] = []
        for index, word in enumerate(_field(alternative, "words") or []):
            text = _field(word, "punctuated_word") or _field(word, "word") or ""
            speaker = _field(word, "speaker")
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
                    speaker=None if speaker is None else str(speaker),
                    confidence=_as_float(_field(word, "confidence")),
                    # Left unknown on purpose: what the service detects belongs
                    # to the channel, not to this word.
                    language=Language.UNKNOWN,
                    chunk_index=request.chunk_index,
                    is_punctuation=_is_punctuation_only(text),
                )
            )
        return tokens

    # -- Building the request -------------------------------------------

    def _keyterms_for(self, terms: tuple[str, ...], model: str) -> tuple[str, ...]:
        """Cut the vocabulary down to what one request is allowed to carry.

        The cap counts tokens rather than terms, and exceeding it fails the
        whole request rather than trimming it, so the budget is spent
        deliberately here with a pessimistic estimate of what each word costs.
        The vocabulary arrives ordered most specific first, which is what makes
        taking the head of it the right way to cut. What was dropped is logged,
        because a name that never reached the service comes back misspelled and
        nothing in the transcript explains why.

        Only some models can use a key-term list at all. A model that cannot
        gets none, which loses nothing it would have had and keeps a request
        the service may refuse from being built.
        """
        cleaned = [term.strip() for term in terms if term and term.strip()]
        if cleaned and not model.startswith(KEYTERM_MODEL_PREFIXES):
            _log.warning(
                "The %d vocabulary terms were not sent to Deepgram. Key terms need "
                "one of the %s models, and this request uses %r.",
                len(cleaned),
                " or ".join(KEYTERM_MODEL_PREFIXES),
                model,
            )
            return ()
        kept: list[str] = []
        spent = 0
        for term in cleaned:
            cost = max(1, len(term.split())) * TOKENS_PER_WORD
            if spent + cost > self._maximum_keyterm_tokens:
                break
            kept.append(term)
            spent += cost
        if len(kept) < len(cleaned):
            _log.info(
                "Sending Deepgram the first %d of %d terms. The rest were left out to "
                "stay under the %d-token limit, which rejects a request rather than "
                "shortening it.",
                len(kept),
                len(cleaned),
                self._maximum_keyterm_tokens,
            )
        return tuple(kept)

    def _model_for(self, request: TranscriptionRequest) -> str:
        """The model this request will actually be sent with.

        Not simply the one the constructor was given. The free-form settings
        are merged into the request last and may name a model of their own,
        which is deliberate: it is how a model this application has never heard
        of gets used at all. Two decisions depend on knowing which name really
        goes out. Key terms are refused by every model but a few, so a gate
        that consulted the constructor would either send them to a model that
        rejects the request or withhold them from one that would have taken
        them and then name the wrong model in the warning. And provenance
        records the model beside the parameters, which must not disagree.

        Trimmed and lowered, because that is the form that goes on the wire.
        Deepgram takes the name literally, so a stray space around it is a
        rejected request rather than a tolerated one.
        """
        extras = {**self._parameters, **request.extra_parameters}
        chosen = extras.get("model") or self._model
        return str(chosen).strip().lower()

    def _options_for(
        self,
        request: TranscriptionRequest,
        *,
        model: str,
        keyterms: tuple[str, ...],
    ) -> dict[str, Any]:
        """Everything that goes in the request, as one plain dictionary.

        Kept as a dictionary rather than as the library's typed options object,
        because a parameter from Settings that the installed library predates
        would be dropped by a typed object without a word, and being able to
        use a new parameter without changing the application is exactly what
        those settings are for.
        """
        options: dict[str, Any] = {
            "model": model,
            "language": self._language_for(request),
            "smart_format": self._smart_format,
            "punctuate": self._punctuate,
        }
        if self._utterances:
            options["utterances"] = True
        if request.verbatim:
            # The verbatim switch. It is honoured on English only, which is
            # where the fillers this application must keep mostly occur.
            options["filler_words"] = True
        if request.diarise:
            # Never ``diarize`` beside it. Sending both is rejected outright,
            # so this adapter uses only the newer parameter and drops the old
            # one wherever it appears.
            options["diarize_model"] = self._diarise_model
        if keyterms:
            options["keyterm"] = list(keyterms)

        extras = {**self._parameters, **request.extra_parameters}
        for key, value in extras.items():
            if key == "model":
                # Already taken, and already trimmed and lowered. Letting it
                # through here would put the untidied form back.
                continue
            if key == "diarize":
                _log.warning(
                    "The diarize parameter was not sent to Deepgram. It is the "
                    "deprecated switch, and sending it beside diarize_model has the "
                    "whole request rejected."
                )
                continue
            options[key] = value
        return options

    def _language_for(self, request: TranscriptionRequest) -> str:
        """The language code to ask for.

        A single code is restrictive on this service: it transcribes that
        language and ignores whatever else is in the recording. So a single
        code is sent only when there is genuinely only one language it can be,
        and anything else asks for the multilingual model.
        """
        if self._language:
            return self._language
        known = [
            language
            for language in request.languages
            if language is not Language.UNKNOWN and self.capabilities.supports(language)
        ]
        if len(known) == 1:
            return known[0].value
        return MULTILINGUAL_CODE

    # -- Recording what happened ----------------------------------------

    def _request_record(
        self,
        request: TranscriptionRequest,
        *,
        options: dict[str, Any],
        keyterms: tuple[str, ...],
        size_bytes: int,
        started_at: datetime,
        answer: Any,
        succeeded: bool,
        error: str | None,
    ) -> ProviderRequestRecord:
        """Note what was asked and what came back, without the credential.

        This is the best provenance of the five services. The model version is
        a real dated build string rather than a marketing name, so a transcript
        that reprocesses differently in a year can be shown to have met a
        different model rather than a changed application. The diarizer is
        named separately, because the speaker labels come from it rather than
        from the transcription model.
        """
        recorded = {key: value for key, value in options.items() if key != "keyterm"}
        # Named for what it counts rather than for the service's word for it,
        # because anything with "key" in its name is stripped out of a record
        # on the way to disk and a count of terms would vanish with it.
        recorded["vocabulary_term_count"] = len(keyterms)

        metadata = _field(answer, "metadata")
        request_id = _request_id_of(answer)
        if request_id:
            recorded["response_request_id"] = request_id
        model_version = _model_version_of(metadata)
        if model_version:
            recorded["response_model_version"] = model_version
        diarizer = _diarizer_of(metadata)
        if diarizer:
            recorded["response_diarizer"] = diarizer
        duration = _as_float(_field(metadata, "duration"))
        if duration is not None:
            # Seconds already, as everything on this service is.
            recorded["response_duration_seconds"] = duration

        window = request.window
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
            provider_request_id=request_id,
        )
        return ProviderRequestRecord(
            provider=self.provider,
            # The one that was sent, which a setting may have changed, rather
            # than the one this adapter was built with. A record whose named
            # model disagrees with its own recorded parameters explains
            # nothing.
            model_identifier=str(options.get("model") or self._model),
            model_version=model_version,
            request_parameters=_without_credentials(recorded),
            language_configuration=str(options.get("language") or "auto"),
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
            self._client = _build_client(self._api_key)
        return self._client


# -- The real client -----------------------------------------------------


class _SdkClient:
    """The package's client, behind the one method this adapter uses.

    The package reaches its pre-recorded transcription through a chain of
    accessors, and wrapping that chain here means there is exactly one place to
    change when it moves, rather than a call shape spread through the adapter
    and every test that drives it. It has moved once already: the chain this
    calls does not exist before version 5.

    The request itself is sent as query parameters rather than as arguments to
    the library's generated method, and that is deliberate. The method declares
    a fixed list of parameters, fixed at whatever version is installed, and it
    raises on a name it does not know instead of sending it. Query parameters
    are the route the library keeps open for exactly this, so a parameter typed
    into Settings reaches the service whether or not the installed package has
    heard of it. Deepgram takes every one of these on the query string anyway,
    so nothing is being smuggled: it is the same request by the same road.
    """

    def __init__(self, api_key: str) -> None:
        try:
            from deepgram import DeepgramClient
        except ImportError as error:
            raise ProviderUnavailable(Provider.DEEPGRAM, PACKAGE) from error
        # By keyword. Version 5 stopped taking the key positionally, and a
        # positional call raises before anything is sent.
        self._client = DeepgramClient(api_key=api_key)
        self._media = _media_client_of(self._client)

    #: How many further attempts a failure worth repeating is given. Two,
    #: which is what the library itself would have made, so this replaces its
    #: retries rather than adding to them.
    RETRIES = 2

    #: Seconds before the first retry, doubled for the second. Short, because
    #: what is being waited out is a moment of load at the far end and the
    #: request behind it may already have been running for minutes.
    FIRST_RETRY_SECONDS = 2.0

    def transcribe(self, audio: Any, options: dict[str, Any], timeout: float | None = None) -> Any:
        request_options: dict[str, Any] = {
            "additional_query_parameters": dict(options),
            # None from the library. See the module docstring, and the retry
            # below that takes its place.
            "max_retries": 0,
        }
        if timeout is not None:
            request_options["timeout"] = timeout

        delay = self.FIRST_RETRY_SECONDS
        for attempt in range(self.RETRIES + 1):
            remaining = self.RETRIES - attempt
            if not _rewound(audio):
                # Nothing can be repeated from a stream that will not go back
                # to its beginning, so this attempt is the only attempt.
                remaining = 0
            try:
                return self._media.transcribe_file(
                    request=audio, request_options=request_options
                )
            except Exception as error:
                # Caught only to decide whether to repeat it. Anything not
                # worth repeating goes straight back out, unchanged, for the
                # adapter to turn into a result.
                _, retryable, status = _describe_failure(error)
                if not remaining or not retryable:
                    raise
                _log.info(
                    "Deepgram answered with %s. Trying again in %.0f seconds, with %d "
                    "further attempt(s) after this one.",
                    status or type(error).__name__,
                    delay,
                    remaining - 1,
                )
                time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")  # pragma: no cover


def _rewound(audio: Any) -> bool:
    """Put the audio back to its beginning, and say whether that worked.

    This is the whole reason the retry lives here rather than being left to
    the library. An attempt reads the handle to the end, and a second attempt
    that does not first go back to the start posts nothing at all. Where the
    audio cannot be rewound there is no honest way to repeat the request, and
    saying so is better than sending an empty one.
    """
    try:
        if not audio.seekable():
            return False
        audio.seek(0)
    except (AttributeError, OSError, ValueError):
        return False
    return True


def _media_client_of(client: Any) -> Any:
    """The part of the library that transcribes a file, or a plain refusal.

    Reached one step at a time so that a package too old to have this can be
    reported as the version problem it is. Left to itself the chain would raise
    an attribute error naming whichever link is missing, which says nothing
    about what to do next.
    """
    media = getattr(getattr(getattr(client, "listen", None), "v1", None), "media", None)
    if media is None or not callable(getattr(media, "transcribe_file", None)):
        raise ProviderError(
            f"The installed {PACKAGE} package is too old for this application, which "
            f"needs version {MINIMUM_PACKAGE_VERSION} or later. "
            f"Run: pip install --upgrade '{PACKAGE}>={MINIMUM_PACKAGE_VERSION}'"
        )
    return media


def _build_client(api_key: str) -> Any:
    """Build a real client, reporting a missing package as such.

    The import lives inside the client rather than at the top of the module so
    that the application starts on a machine where the package was never
    installed. That machine loses Deepgram and is told why.
    """
    return _SdkClient(api_key)


# -- Helpers -------------------------------------------------------------


def _canonical_seconds(seconds: float | None, offset: float) -> float | None:
    """Move one of the service's times onto the canonical timeline.

    No unit conversion here, unlike the other two services in this family, but
    the offset still has to be added. It is easy to skip precisely because
    there is nothing else to do, and skipping it produces a transcript whose
    words all point at the wrong moment without a single number looking wrong.
    """
    if seconds is None:
        return None
    return seconds + offset


#: The ways a response object of one library generation or another will hand
#: back the plain structure it was built from, best first, with the arguments
#: each needs. ``to_dict`` was the old library's. ``model_dump`` is the current
#: one's, and it has to be asked for JSON: left to itself it hands back the
#: Python objects the library parsed the answer into, and one of those cannot
#: be written to a file. See :func:`_raw_response_of`.
_RESPONSE_CONVERTERS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("to_dict", {}),
    ("model_dump", {"mode": "json"}),
    ("model_dump", {}),
    ("dict", {}),
)


def _raw_response_of(response: Any) -> Any:
    """The service's own answer, as close to untouched as it can be had.

    The library's response objects convert themselves back to the plain
    structure they were built from, so that is what is taken. Rebuilding the
    response field by field would defeat the purpose of keeping it: what is
    wanted is evidence of what the service said, and a reconstruction is
    evidence of what this adapter understood instead. Anything that cannot
    convert itself is passed along as it is, for the store to serialise.

    Everything else in this module reads what comes back from here rather than
    the object the library returned, for two reasons. The store can only write
    a plain structure, and reading the typed object instead would tie this
    adapter to whatever shape the installed library happens to declare, which
    is a shape that has already changed once and will change again.

    Asking for JSON rather than taking the default dump is the part that is
    easy to get wrong and expensive to get wrong. The library parses the answer
    into Python objects, and the ``created`` timestamp in the metadata becomes
    a real ``datetime``. A dictionary with one of those in it is written
    nowhere: it reaches the transcript store, which serialises it as JSON, and
    fails there. That happens once every service has answered and been paid
    for, so it costs the whole run rather than this one service. Asked for
    JSON, the same dump gives a string.

    Untouched has one limit worth naming. That timestamp comes back in a
    canonical form, so a service that wrote ``.426Z`` is recorded as
    ``.426000Z``. Nothing else is changed, added or dropped.
    """
    for name, arguments in _RESPONSE_CONVERTERS:
        converter = getattr(response, name, None)
        if not callable(converter):
            continue
        try:
            converted = converter(**arguments)
        except Exception:  # noqa: BLE001 - trying the next way still beats nothing
            _log.debug("Deepgram's response would not convert through %s.", name)
            continue
        if isinstance(converted, dict):
            return converted
    _log.debug("Deepgram's response could not be converted; keeping it as it is.")
    return response


def _best_alternative(response: Any) -> Any:
    """The first alternative of the first channel, which is the transcript.

    The service can return several channels and several alternatives per
    channel. This application sends single-channel audio and asks for one
    reading of it, so anything beyond the first of each is not something to
    choose between: it is a sign the request was not what this adapter thinks
    it was, and taking the first is the only defensible reading.
    """
    results = _field(response, "results")
    channels = _field(results, "channels") or []
    if not channels:
        return None
    alternatives = _field(channels[0], "alternatives") or []
    return alternatives[0] if alternatives else None


def _request_id_of(response: Any) -> str | None:
    """The service's own identifier for this request, from either shape.

    A transcript keeps it in the metadata. An acknowledgement of a callback
    request has no metadata at all: it is a single ``request_id`` and nothing
    else. That is the shape most in need of the identifier, because the
    transcript for it arrives somewhere else entirely and this number is the
    only way to go and find it, so reading only the metadata lost it in
    exactly the case where it mattered most.
    """
    metadata = _field(response, "metadata")
    return _as_text(_field(metadata, "request_id")) or _as_text(_field(response, "request_id"))


def _detected_language_code(response: Any) -> str | None:
    """Which language the service says it heard, wherever it chose to say it.

    There are two places, and which one is filled depends on what was asked
    for. The request this adapter usually makes names ``multi``, and a
    multilingual answer lists every language it found on the alternative
    itself, ordered by how many words were in each, so the first entry is the
    dominant one. A ``detected_language`` on the channel appears only where
    language detection was asked for outright. Reading only the second, which
    is what this did before, meant the usual request never reported a language
    at all.
    """
    results = _field(response, "results")
    channels = _field(results, "channels") or []
    if not channels:
        return None
    alternatives = _field(channels[0], "alternatives") or []
    languages = _field(alternatives[0], "languages") if alternatives else None
    if isinstance(languages, (list, tuple)) and languages:
        return _as_text(languages[0])
    return _as_text(_field(channels[0], "detected_language"))


def _model_version_of(metadata: Any) -> str | None:
    """The dated build string of the model that actually ran.

    It sits under a dictionary keyed by the model's identifier, which is not
    known in advance, so the first entry is taken. Single-model requests, which
    is all this application makes, have exactly one.
    """
    model_info = _field(metadata, "model_info")
    if not isinstance(model_info, dict) or not model_info:
        return None
    first = next(iter(model_info.values()))
    version = _as_text(_field(first, "version"))
    name = _as_text(_field(first, "name"))
    if version and name:
        return f"{name} {version}"
    return version or name


def _diarizer_of(metadata: Any) -> str | None:
    """Which diarizer produced the speaker labels, where the service says.

    Separate from the model version on purpose: the speaker labels come from a
    different component, and a change in one explains a different answer that a
    change in the other would not.

    What the service puts here is an architecture, ``v1`` or ``v2``, and the
    identifier of the build that ran. Both are taken, in that order, because
    the architecture is the part a person reads and the identifier is the part
    that pins the answer down. A name and a version are accepted beside them so
    that a service that starts reporting those is not silently ignored.
    """
    info = _field(metadata, "diarize_info")
    if info is None:
        return None
    if isinstance(info, dict):
        parts = [
            str(value)
            for value in (
                info.get("arch"),
                info.get("name"),
                info.get("version"),
                info.get("model_uuid"),
            )
            if value
        ]
        return " ".join(parts) or None
    return str(info)


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


#: Where a refusal keeps its explanation, in the order worth trying. The
#: service writes ``err_msg``; the others cost nothing and cover a body from
#: somewhere between here and there, such as a proxy.
_FAILURE_MESSAGE_KEYS = ("err_msg", "message", "reason", "error", "detail")


def _detail_of(error: Exception) -> str:
    """What actually went wrong, in as few words as the failure allows.

    Printing the exception is not good enough any more. The library raises a
    single error type that prints itself as every response header, the status
    and the body all run together, and that whole block would otherwise be
    handed to the person reading the transcript's warnings. The sentence the
    service wrote is inside it, so that is taken where it can be found and the
    printed form is kept only as a last resort.
    """
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        for key in _FAILURE_MESSAGE_KEYS:
            text = _as_text(body.get(key))
            if text and text.strip():
                code = _as_text(body.get("err_code"))
                return f"{text.strip()} ({code})" if code else text.strip()
    if isinstance(body, str) and body.strip():
        return body.strip()
    return str(error).strip()


def _describe_failure(error: Exception) -> tuple[str, bool, int | None]:
    """Turn whatever the library threw into a sentence, and say if it may be retried.

    A 504 is singled out because it does not mean what it usually means. The
    service has no published limit on how long a recording may be; it has a
    ten-minute budget on how long it will spend processing one, and a request
    that overruns it comes back with this status however healthy the service
    is. Repeating the same request would spend the same budget again, so it is
    reported as the budget it is, and the way forward is the callback mode or
    a smaller piece of audio rather than another attempt.
    """
    status = _as_int(getattr(error, "status_code", None))
    if status is None:
        status = _as_int(getattr(error, "status", None))
    name = type(error).__name__
    detail = _detail_of(error) or name

    if status == PROCESSING_TIMEOUT_STATUS:
        return (
            "Deepgram spent longer than its ten-minute processing budget on this "
            "recording and returned 504. That is a limit on processing time rather "
            "than on how long the audio may be, so the recording has to go through in "
            "smaller pieces or through the callback mode instead.",
            False,
            status,
        )

    retryable = False
    if status is not None:
        retryable = status in (408, 409, 429) or 500 <= status < 600
    elif any(hint in name.lower() for hint in _RETRYABLE_ERROR_NAMES):
        retryable = True

    if status is not None:
        return f"Deepgram answered with an error ({status}): {detail}", retryable, status
    return f"Deepgram could not be reached: {detail}", retryable, None
