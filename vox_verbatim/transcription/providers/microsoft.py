"""Talking to Microsoft MAI-Transcribe, which hears well and cannot count speakers.

Microsoft is here as a second opinion on the words, and for one thing none of
the other four services can give us: it says what language it heard for every
phrase it returns. Everywhere else the language is a single answer for the
whole file, which is no use at all in a recording that switches between
English and German halfway through a sentence. Microsoft's per-phrase locale
is carried onto every word this module produces, and it is the only genuine
per-segment language evidence the application has.

Four things about this service are worth knowing before reading the code, and
each of them shaped something here.

The first is that MAI-Transcribe is not a model with an endpoint of its own.
There is no deployment to create and no per-model URL. It is a mode switch
inside the ordinary Azure Speech fast-transcription endpoint: you post to that
endpoint and name the model inside the request body. The endpoint this adapter
is given is therefore the Speech resource URL, not a deployment URL. Only a
few Azure regions serve MAI at all, and a resource in the wrong region fails in
a way that looks exactly like a misspelled model name, so when the model is
refused this module names the regions known to serve it rather than leaving
somebody to guess.

The second is why this one service is called over plain HTTP while the other
four go through a vendor library. The official ``azure-ai-transcription``
package exposes only ``task``, ``target_language``, ``prompt`` and ``enabled``
on its enhanced-mode options. It has no ``model`` field and no
``transcribeStyle`` field, and those two are precisely the ones we cannot do
without: one selects MAI at all, and the other keeps the fillers and false
starts that make the transcript verbatim. The generated objects happen to be
dict-like, so the keys could be forced in, but that is undocumented behaviour
to build a paid integration on. So the documented multipart request is made
directly with ``httpx``. Everything above this module is unaffected, and the
day the package catches up this file can change without anything noticing.

The third is that it cannot tell speakers apart. Not badly, not unreliably:
not at all. The capabilities below say so, which is what stops anything
upstream from asking.

The fourth is the confidence. Every phrase comes back with a confidence of
exactly zero, on every request, whatever was said. It is a constant, not a
measurement. Storing it would put a number in front of the reconciliation
rules that reads as "this service was certain it heard nothing of the sort",
which is a lie about evidence rather than an absence of it. So confidence is
left as ``None`` and Microsoft simply contributes no acoustic confidence.

One request-shaping decision is worth stating on its own, because the field
that carries it looks harmless. The service is multi-lingual unless it is told
otherwise, and naming a locale does not narrow the candidates it considers the
way the same field does on the plain fast-transcription endpoint: it forces
recognition into that one language. A request for an English and German meeting
that named English would come back as English from end to end, and the
per-phrase locale described above would be a constant rather than evidence. So
the locale is sent only where the recording is known to be in a single
language, and is left out entirely otherwise.

Times come back as an offset and a duration, both in milliseconds, rather than
as a start and an end. They are turned into canonical seconds here, once, and
never thought about again.

The client library is imported inside the method that needs it, so a machine
without ``httpx`` loses this one service rather than failing to start.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from vox_verbatim.transcription.model import (
    ChunkRecord,
    Language,
    Provider,
    ProviderRequestRecord,
    ProviderResult,
    ProviderToken,
)
from vox_verbatim.transcription.providers.base import (
    CancelCheck,
    ProviderCapabilities,
    ProviderError,
    ProviderNotConfigured,
    ProviderUnavailable,
    TranscriptionProvider,
    TranscriptionRequest,
    retry_after_seconds,
)

_log = logging.getLogger(__name__)

#: The package that must be installed for this service to work at all. It is
#: an HTTP client rather than a vendor library, for the reason the module
#: docstring gives.
PACKAGE = "httpx"

#: The current model. ``mai-transcribe-1`` is deprecated and retires on
#: 20 August 2026, so nothing here should ever default to it.
DEFAULT_MODEL = "mai-transcribe-1.5"

#: The dated version of the fast-transcription API this request shape belongs
#: to. It travels in the query string, and a different version can change the
#: response, so it is settable rather than buried.
DEFAULT_API_VERSION = "2025-10-15"

#: The path appended to the Speech resource endpoint. The model name does not
#: appear in it: it goes in the body.
TRANSCRIBE_PATH = "/speechtotext/transcriptions:transcribe"

#: The Azure regions known to serve MAI. A Speech resource anywhere else
#: answers ordinary fast-transcription requests perfectly well and refuses this
#: one, which is why a rejected model is reported together with this list.
#: Azure adds regions without telling anybody, so the list is offered as a
#: likely cause rather than as the whole truth.
MAI_REGIONS = ("eastus", "northeurope", "southeastasia", "westus")

#: Keeps the fillers, false starts and repetitions. Without it the service
#: returns a readability-optimised transcript, which would quietly break the
#: rule that the verbatim layer is the authoritative one: the words would no
#: longer be the words that are in the audio.
VERBATIM_STYLE = "verbatim"

#: The model that takes neither a verbatim style nor a phrase list. Both
#: arrived with ``mai-transcribe-1.5``; the model before it accepts neither,
#: and is deprecated besides. Any other name is assumed to accept both,
#: because a model newer than this file quietly losing its verbatim style
#: would be the worse of the two mistakes.
UNSTYLED_MODELS = frozenset({"mai-transcribe-1"})

#: Turns the profanity masking off. The default replaces letters with
#: asterisks, and a masked word is a corrupted transcript, not a polite one.
PROFANITY_FILTER_OFF = "None"

_MEGABYTE = 1024 * 1024

#: What one request may carry. The two documents describing this endpoint do
#: not agree: the MAI page asks for a file under 300 MB and says nothing about
#: length, while the API definition for this dated version says under 250 MB
#: and under two hours of audio. The smaller of the two is taken. A file over
#: the endpoint's own ceiling is refused outright, and planning to the smaller
#: number costs nothing but one more chunk boundary, which is already exact.
MAXIMUM_FILE_BYTES = 250 * _MEGABYTE
MAXIMUM_DURATION_SECONDS = 2 * 60 * 60

#: WAV, MP3 and FLAC only. The longer format lists in Azure's documentation
#: belong to plain fast transcription, not to the enhanced mode we use.
ACCEPTED_CONTAINERS = frozenset({"wav", "mp3", "flac"})

MICROSOFT_CAPABILITIES = ProviderCapabilities(
    word_timings=True,
    # Not switched off, and not weak. The service has no diarisation of any
    # kind, so nothing upstream should ever ask this adapter who spoke.
    diarisation=False,
    speaker_count_hint=False,
    # The service returns a confidence, and it is the constant zero. Declaring
    # this false is what stops a rule from reading that constant as evidence.
    word_confidence=False,
    language_detection=True,
    # True, and uniquely so among the five. Every phrase carries a locale, and
    # every word inherits the locale of the phrase it came from.
    per_word_language=True,
    vocabulary_biasing=True,
    # Prompt tuning is unsupported on MAI. Vocabulary goes through phraseList.
    context_prompt=False,
    time_window=False,
    forced_alignment=False,
    languages=frozenset({Language.ENGLISH, Language.GERMAN}),
    maximum_file_bytes=MAXIMUM_FILE_BYTES,
    maximum_duration_seconds=MAXIMUM_DURATION_SECONDS,
    accepted_containers=ACCEPTED_CONTAINERS,
)

#: Settings keys that could hold a credential. Anything named like one of
#: these is dropped before a request is written into provenance, because a
#: transcript folder is copied around and read by people.
_CREDENTIAL_NAMES = (
    "api_key",
    "apikey",
    "subscription-key",
    "subscription_key",
    "key",
    "token",
    "secret",
    "password",
    "authorization",
)

#: Errors worth trying again even though they carry no HTTP status, because
#: they happened before the far end answered at all. The service is known to
#: accept a request and then time out while generating the answer, which
#: arrives here as a network error rather than as an HTTP one.
#: Matched against the exception's class name. ``httpx.ConnectError`` is
#: spelt without the "ion", which is why "connect" and not "connection".
_RETRYABLE_ERROR_NAMES = ("timeout", "connect", "readerror", "protocol")

_CONTENT_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
}


class MicrosoftProvider(TranscriptionProvider):
    """The Microsoft MAI-Transcribe speech-to-text service.

    Everything it needs arrives through the constructor. It knows nothing
    about the application's settings file: a separate registry reads Settings
    and calls this, which keeps a settings change from rippling through the
    adapters and lets a test build a working adapter out of a fake client.
    """

    provider = Provider.MICROSOFT
    capabilities = MICROSOFT_CAPABILITIES

    def __init__(
        self,
        api_key: str = "",
        endpoint: str = "",
        model: str = DEFAULT_MODEL,
        parameters: dict[str, Any] | None = None,
        *,
        api_version: str = DEFAULT_API_VERSION,
        timeout_seconds: float = 900.0,
        maximum_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        """Set up the adapter.

        ``maximum_retries`` is how many times a failed request is tried
        again, by the retry loop in the base class. This adapter talks to
        the service over a bare HTTP client that retries nothing by itself,
        so without this a dropped connection on a long upload cost the chunk
        its transcript.

        ``endpoint`` is the Speech resource URL, such as
        ``https://my-resource.cognitiveservices.azure.com``. It is not a
        deployment URL, because MAI has no deployment of its own.

        ``parameters`` holds the free-form extras from Settings. They are
        merged into the JSON definition that travels beside the audio, with
        the extras on an individual request winning where both name the same
        thing, because the request is the more specific of the two.

        ``client`` is anything with an ``httpx``-shaped ``post``. Left as
        ``None`` it means "build a real one when first needed", which is what
        keeps the import out of application start-up and lets a test drive
        this adapter with no network and no package installed.
        """
        self._api_key = api_key or ""
        self._endpoint = (endpoint or "").strip()
        self._model = model or DEFAULT_MODEL
        self._parameters = dict(parameters or {})
        self._api_version = api_version or DEFAULT_API_VERSION
        self._timeout_seconds = timeout_seconds
        self.maximum_retries = max(0, maximum_retries)
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
        if not self._endpoint:
            return (
                "no endpoint has been entered. It is the Speech resource URL, "
                "such as https://my-resource.cognitiveservices.azure.com"
            )
        if not self._model.strip():
            return "no model has been chosen"
        return None

    @property
    def url(self) -> str:
        """The full address this adapter posts to.

        Built rather than configured, because the path and the API version are
        part of the request shape this module implements, and letting them be
        set separately would allow a combination that this code cannot read
        the answer to.
        """
        base = self._endpoint.rstrip("/")
        return f"{base}{TRANSCRIBE_PATH}?api-version={self._api_version}"

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
            raise ProviderError("Microsoft was not called, because the run was stopped.")

        container = request.audio_path.suffix.lower().lstrip(".")
        if container not in ACCEPTED_CONTAINERS:
            raise ProviderError(
                f"{request.audio_path.name} is a .{container} file, and enhanced mode "
                f"accepts only {', '.join(sorted(ACCEPTED_CONTAINERS))}. It has to be "
                "converted first."
            )

        size_bytes = _file_size(request.audio_path)
        if size_bytes > MAXIMUM_FILE_BYTES:
            raise ProviderError(
                f"{request.audio_path.name} is {size_bytes / _MEGABYTE:.0f} MB, and "
                f"Microsoft accepts at most {MAXIMUM_FILE_BYTES // _MEGABYTE} MB. "
                "It has to be cut into chunks first."
            )

        locales = self._locales_for(request)
        terms = tuple(term for term in request.vocabulary_terms if term and term.strip())
        definition = self._definition_for(request, locales=locales, terms=terms)
        # Read back out of the definition rather than reused from above. A
        # setting can add phrases, and a model that takes no phrase list drops
        # them altogether, so the two are not the same list. The record has to
        # say what the service was actually given, or it cannot answer the one
        # question it exists for: whether a later run made the same request.
        sent_locales = _locales_in(definition)
        sent_terms = _phrases_in(definition)

        started_at = datetime.now()
        try:
            # Opened inside the attempt, so that a retry sends the whole file
            # again rather than what was left of a half-read handle.
            with request.audio_path.open("rb") as handle:
                # Both parts go in the multipart body. The definition is a
                # JSON string inside a form part, not a JSON request body,
                # and posting it as a body is the single easiest thing to get
                # wrong here: the request looks entirely reasonable and the
                # service rejects it.
                response = client.post(
                    self.url,
                    headers={"Ocp-Apim-Subscription-Key": self._api_key},
                    files={
                        "audio": (
                            request.audio_path.name,
                            handle,
                            _content_type_for(request.audio_path),
                        ),
                        "definition": (None, json.dumps(definition), "application/json"),
                    },
                    timeout=self._timeout_seconds,
                )
                status = _as_int(_field(response, "status_code"))
                body = _body_of(response)
        except Exception as error:  # noqa: BLE001 - every failure becomes a result
            message, retryable, status_code = _describe_failure(error)
            raise self._failed(
                request,
                definition=definition,
                locales=sent_locales,
                terms=sent_terms,
                size_bytes=size_bytes,
                started_at=started_at,
                request_id=None,
                message=message,
                retryable=retryable,
                status=status_code,
                # Nothing came back at all here, so there is nothing to keep.
                raw_response=None,
            ) from error

        if status is not None and status >= 400:
            message, retryable = self._describe_rejection(status, body)
            raise self._failed(
                request,
                definition=definition,
                locales=sent_locales,
                terms=sent_terms,
                size_bytes=size_bytes,
                started_at=started_at,
                request_id=_header(response, "apim-request-id"),
                message=message,
                retryable=retryable,
                status=status,
                retry_after=retry_after_seconds(_field(response, "headers")),
                # Kept deliberately. The body of a refusal is often the most
                # useful thing in the whole folder: it is what says whether
                # the model name was wrong or the resource is in a region that
                # does not serve MAI, and those two look identical from here.
                raw_response=body,
            )

        unusable = _unusable_answer(body)
        if unusable is not None:
            reason, retryable = unusable
            raise self._failed(
                request,
                definition=definition,
                locales=sent_locales,
                terms=sent_terms,
                size_bytes=size_bytes,
                started_at=started_at,
                request_id=_header(response, "apim-request-id"),
                message=f"Microsoft answered, but {reason}",
                retryable=retryable,
                status=status,
                # Kept for the same reason a refusal is: whatever arrived is
                # the only account of why it could not be used.
                raw_response=body,
            )

        phrases = _field(body, "phrases") or []
        tokens = self._tokens_from(phrases, request)
        record = self._request_record(
            request,
            definition=definition,
            locales=sent_locales,
            terms=sent_terms,
            size_bytes=size_bytes,
            started_at=started_at,
            request_id=_header(response, "apim-request-id"),
            duration_milliseconds=_as_float(_field(body, "durationMilliseconds")),
            succeeded=True,
            error=None,
        )
        return ProviderResult(
            provider=self.provider,
            tokens=tokens,
            request=record,
            detected_language=_dominant_language(tokens),
            # Empty, and correctly so. The service returns no speaker of any
            # kind, and an empty list is the honest answer rather than a gap.
            speakers=[],
            # The parsed body, which on this service is exactly what arrived:
            # nothing sits between the wire and this dictionary, because the
            # request was made over plain HTTP rather than through a library
            # that would have turned it into objects of its own first. The
            # pipeline hands it to the transcript store, which writes it to a
            # file of its own. This adapter never writes anything, and knows
            # nothing about where transcripts are kept.
            raw_response=body,
        )

    # -- Turning the answer into evidence -------------------------------

    def _tokens_from(
        self,
        phrases: Any,
        request: TranscriptionRequest,
    ) -> list[ProviderToken]:
        """Convert every word the service returned onto the canonical timeline.

        Two conversions happen here and both matter. Times arrive as an offset
        and a duration in milliseconds, so the end of a word is its offset plus
        its duration, and the pair is divided by a thousand before
        ``canonical_offset`` is added. Get either half wrong and the words land
        somewhere plausible but false.

        The language of each phrase is copied down onto its words. That is not
        an assumption: the service genuinely reports a locale per phrase, and
        it is the only per-segment language evidence this application receives
        from anywhere.
        """
        tokens: list[ProviderToken] = []
        index = 0
        for phrase in phrases:
            language = Language.from_code(_field(phrase, "locale"))
            words = _field(phrase, "words") or []
            if not words:
                # A phrase with no word breakdown still carries text, a span
                # and a locale, so it is kept as one token covering the whole
                # phrase rather than thrown away.
                words = [phrase]
            for word in words:
                text = _field(word, "text") or ""
                start_ms = _as_float(_field(word, "offsetMilliseconds"))
                duration_ms = _as_float(_field(word, "durationMilliseconds"))
                # A negative duration would put the end before the start,
                # which nothing downstream can make sense of, so the length
                # is never allowed below zero.
                end_ms = None if start_ms is None else start_ms + max(0.0, duration_ms or 0.0)
                tokens.append(
                    ProviderToken(
                        provider=self.provider,
                        index=index,
                        text=text,
                        start=_canonical_seconds(start_ms, request.canonical_offset),
                        end=_canonical_seconds(end_ms, request.canonical_offset),
                        # No speaker, because the service does not have one,
                        # and no confidence, because the one it reports is the
                        # constant zero rather than a measurement.
                        speaker=None,
                        confidence=None,
                        language=language,
                        chunk_index=request.chunk_index,
                        is_punctuation=_is_punctuation_only(text),
                    )
                )
                index += 1
        return tokens

    # -- Building the request -------------------------------------------

    def _locales_for(self, request: TranscriptionRequest) -> tuple[str, ...]:
        """The locale to name, which for most recordings is none at all.

        Naming a locale here forces recognition into that one language rather
        than narrowing a field of candidates, so it is sent only where exactly
        one of the service's languages is in play. Where two are, the field is
        left out and the service stays in the multi-lingual mode it uses by
        default, which is both the documented behaviour and the only way its
        per-phrase locale can say anything.
        """
        codes: list[str] = []
        for language in request.languages:
            if language is Language.UNKNOWN:
                continue
            if not self.capabilities.supports(language):
                # Afrikaans is not among the service's languages at all, so
                # offering it would be asking for a wrong answer rather than
                # no answer.
                continue
            if language.value not in codes:
                codes.append(language.value)
        return tuple(codes) if len(codes) == 1 else ()

    def _definition_for(
        self,
        request: TranscriptionRequest,
        *,
        locales: tuple[str, ...],
        terms: tuple[str, ...],
    ) -> dict[str, Any]:
        """The JSON options that travel beside the audio.

        Two of these are sent on every request that can carry them, and
        neither is optional in practice. ``transcribeStyle`` is set to verbatim
        because the service otherwise returns a tidied, readable transcript,
        and a transcript that has quietly dropped the false starts no longer
        matches the audio it claims to describe. ``profanityFilterMode`` is
        turned off because its default masks words with asterisks, and a masked
        word is simply a wrong word that nothing downstream can recover.

        The style and the phrase list both arrived with ``mai-transcribe-1.5``.
        The model before it takes neither, so on that model they are left out
        and the loss is written to the log. That is better than a request the
        service refuses outright, and better than sending a field the service
        ignores while this code believes it was honoured.

        Phrases named in Settings are added to the recording's own terms rather
        than put in their place, since the setting applies to everything and
        the terms belong to this one recording. A phrase list that ends up
        holding no phrases at all is left out, because a biasing weight with
        nothing to weight is not a phrase list.
        """
        definition: dict[str, Any] = {
            "profanityFilterMode": PROFANITY_FILTER_OFF,
            "enhancedMode": {
                "enabled": True,
                "model": self._model,
            },
        }
        if locales:
            # One language, and only where the recording is known to be in it.
            definition["locales"] = list(locales)
        if self._supports_styling():
            # Always verbatim, even where a caller asks for otherwise. The
            # tidied style is not a formatting choice at this level: it
            # removes words that were spoken, and the layer this evidence
            # feeds is defined as the one that keeps them.
            definition["enhancedMode"]["transcribeStyle"] = VERBATIM_STYLE
            if terms:
                # The service's vocabulary biasing, documented as entity
                # biasing.
                definition["phraseList"] = {"phrases": list(terms)}
        elif terms:
            _log.warning(
                "The model %r accepts neither a verbatim style nor a phrase list, so "
                "Microsoft will return a tidied transcript and will not be given the "
                "%d vocabulary terms for this recording. Only mai-transcribe-1.5 "
                "accepts them.",
                self._model,
                len(terms),
            )
        else:
            _log.warning(
                "The model %r accepts no verbatim style, so Microsoft will return a "
                "tidied transcript rather than the words that were spoken. Only "
                "mai-transcribe-1.5 accepts one.",
                self._model,
            )

        extras = {**self._parameters, **request.extra_parameters}
        for key, value in extras.items():
            if key == "enhancedMode" and isinstance(value, dict):
                # Merged rather than replaced, so that a setting adding a new
                # enhanced-mode field cannot silently drop the model name or
                # the verbatim style along with it.
                merged = {**definition["enhancedMode"], **value}
                merged["model"] = self._model
                if self._supports_styling():
                    merged["transcribeStyle"] = VERBATIM_STYLE
                else:
                    merged.pop("transcribeStyle", None)
                definition["enhancedMode"] = merged
                continue
            if key == "phraseList" and isinstance(value, dict):
                if not self._supports_styling():
                    # Already reported above, along with the terms lost with it.
                    continue
                # Merged for the same reason as enhancedMode. A setting that
                # adds a biasing weight must not take the recording's own
                # vocabulary away with it, which a straight replacement does.
                current = definition.get("phraseList", {})
                merged = {**current, **value}
                supplied = value.get("phrases")
                if isinstance(supplied, (list, tuple)):
                    # Both lists are kept, the recording's first. A setting
                    # applies to everything transcribed on this machine, while
                    # the terms come from this recording, and the general one
                    # must not quietly displace the particular one.
                    ours = [str(phrase) for phrase in current.get("phrases") or ()]
                    merged["phrases"] = ours + [
                        phrase for phrase in supplied if phrase not in ours
                    ]
                definition["phraseList"] = merged
                continue
            if key == "profanityFilterMode" and value != PROFANITY_FILTER_OFF:
                _log.warning(
                    "The Microsoft setting asking for profanity filtering (%r) was "
                    "ignored and the filter left off. A masked word is a corrupted "
                    "transcript, not a polite one.",
                    value,
                )
                continue
            definition[key] = value

        phrase_list = definition.get("phraseList")
        if isinstance(phrase_list, dict) and not phrase_list.get("phrases"):
            # A biasing weight with no phrases to weight is at best ignored and
            # at worst refused, and it would go out on every recording that has
            # no vocabulary of its own, which is most of them.
            definition.pop("phraseList")
            _log.warning(
                "A Microsoft phrase list holding no phrases was left out of the "
                "request. A biasing weight on its own has nothing to bias."
            )
        return definition

    def _supports_styling(self) -> bool:
        """Whether this model accepts a verbatim style and a phrase list."""
        return self._model.strip().lower() not in UNSTYLED_MODELS

    def _describe_rejection(self, status: int, body: Any) -> tuple[str, bool]:
        """Turn an HTTP error into a sentence somebody can act on.

        A resource in a region that does not serve MAI refuses the model by
        name, which reads exactly like a misspelled model. Whenever the
        complaint touches the model or enhanced mode, the regions known to
        serve it are named, because that is the far more likely cause and
        nothing in the message from Azure hints at it.
        """
        detail = _detail_from(body) or "no detail was given"
        retryable = status in (408, 429) or 500 <= status < 600
        sentence = f"Microsoft answered with an error ({status}): {detail}"
        if status in (400, 403, 404) and _mentions_model(detail):
            sentence += (
                f" The model {self._model!r} is refused both when the name is wrong and "
                "when the Speech resource is in a region that does not serve MAI. The "
                f"regions known to serve it are {', '.join(MAI_REGIONS)}, and Azure's "
                "own list of Speech regions is the current one."
            )
        return sentence, retryable

    # -- Recording what happened ----------------------------------------

    def _failed(
        self,
        request: TranscriptionRequest,
        *,
        definition: dict[str, Any],
        locales: tuple[str, ...],
        terms: tuple[str, ...],
        size_bytes: int,
        started_at: datetime,
        request_id: str | None,
        message: str,
        retryable: bool,
        status: int | None,
        retry_after: float | None = None,
        raw_response: Any = None,
    ) -> ProviderError:
        """Build the error for a failure, with the failed result already on it.

        Losing Microsoft costs a little accuracy and a language signal. It must
        never cost the user their transcript, so the failure is written down as
        an ordinary result and the run carries on with the services that did
        answer. The result rides on the error rather than being returned, so
        that the retry loop in the base class can read the verdict, try again
        where that may help, and hand back the full record when it gives up.

        Whatever the service said while refusing is carried out with the
        failure, because an explanation is worth keeping and this is the only
        route by which it can reach the folder beside the recording.
        """
        _log.debug("Microsoft failure: retryable=%s status=%s", retryable, status)
        record = self._request_record(
            request,
            definition=definition,
            locales=locales,
            terms=terms,
            size_bytes=size_bytes,
            started_at=started_at,
            request_id=request_id,
            duration_milliseconds=None,
            succeeded=False,
            error=message,
        )
        return ProviderError(
            message,
            retryable=retryable,
            status_code=status,
            retry_after=retry_after,
            result=ProviderResult(
                provider=self.provider,
                request=record,
                error=message,
                raw_response=raw_response,
            ),
        )

    def _request_record(
        self,
        request: TranscriptionRequest,
        *,
        definition: dict[str, Any],
        locales: tuple[str, ...],
        terms: tuple[str, ...],
        size_bytes: int,
        started_at: datetime,
        request_id: str | None,
        duration_milliseconds: float | None,
        succeeded: bool,
        error: str | None,
    ) -> ProviderRequestRecord:
        """Note what was asked and what came back, without the credential.

        The key never enters this record: it travels in a header, and headers
        are not recorded. The definition is, because it is the whole of what
        made this request what it was, and a later run that produces different
        words needs to be able to show that the request was the same.
        """
        recorded: dict[str, Any] = {
            "url": self.url,
            "api_version": self._api_version,
            "definition": _without_credentials(
                {key: value for key, value in definition.items() if key != "phraseList"}
            ),
            "vocabulary_term_count": len(terms),
        }
        if request_id:
            recorded["response_request_id"] = request_id
        if duration_milliseconds is not None:
            # Converted here as well, so that nothing reading provenance has
            # to remember which of these services counts in milliseconds.
            recorded["response_duration_seconds"] = duration_milliseconds / 1000.0

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
            model_identifier=self._model,
            request_parameters=_without_credentials(recorded),
            language_configuration=",".join(locales) if locales else "auto",
            vocabulary_terms=terms,
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
            self._client = _build_client(self._timeout_seconds)
        return self._client


# -- Helpers -------------------------------------------------------------


def _build_client(timeout_seconds: float) -> Any:
    """Build a real HTTP client, reporting a missing package as such.

    The import lives here rather than at the top of the module so that the
    application starts on a machine where the package was never installed.
    That machine loses Microsoft and is told why.

    The timeout is deliberately generous. The service can accept a request and
    then take a long time producing the answer, and a short timeout turns a
    slow success into a failure.
    """
    try:
        import httpx
    except ImportError as error:
        raise ProviderUnavailable(Provider.MICROSOFT, PACKAGE) from error
    return httpx.Client(timeout=timeout_seconds)


def _canonical_seconds(milliseconds: float | None, offset: float) -> float | None:
    """Turn one of the service's times into a canonical one.

    Milliseconds to seconds, then the offset of whatever piece of audio this
    was, so a word ten seconds into a chunk that itself began three minutes in
    comes back where it really is. Both halves are done here, once, because
    doing either of them twice or not at all is invisible in the output.
    """
    if milliseconds is None:
        return None
    return (milliseconds / 1000.0) + offset


def _dominant_language(tokens: list[ProviderToken]) -> Language:
    """The language most of the words were in, as the file-level answer.

    Per-phrase locales are the valuable part and they stay on the tokens. This
    is only the summary that the rest of the result carries, so the commonest
    one is the honest answer for a recording that is mostly one language with
    a few phrases of another.
    """
    counts: dict[Language, int] = {}
    for token in tokens:
        if token.language is Language.UNKNOWN:
            continue
        counts[token.language] = counts.get(token.language, 0) + 1
    if not counts:
        return Language.UNKNOWN
    return max(counts.items(), key=lambda item: item[1])[0]


def _mentions_model(detail: str) -> bool:
    lowered = detail.lower()
    return "model" in lowered or "enhanced" in lowered


def _body_of(response: Any) -> Any:
    """The parsed answer, whatever the client hands back.

    A real client parses JSON on demand and an error page may not be JSON at
    all, so a body that cannot be parsed comes back as its text rather than
    raising, which would turn a readable error into an unreadable one.
    """
    reader = getattr(response, "json", None)
    if callable(reader):
        try:
            return reader()
        except Exception:  # noqa: BLE001 - an unparseable body is still a body
            pass
    return _field(response, "text")


def _unusable_answer(body: Any) -> tuple[str, bool] | None:
    """Why a successful answer cannot be used, and whether to ask again.

    This is the one failure on this service that would otherwise be silent, and
    it is worth spelling out. Every judgement the adapter makes about the words
    comes from ``phrases``. A body that has none, because it was truncated on
    the way here or rewritten by something in between, yields no words at all,
    and a result with no words and no error is indistinguishable from a
    recording in which nobody spoke. The run would be reported as a success and
    the recording left looking silent, which is worse than any error.

    An answer that is not an object at all, or one carrying no ``phrases`` field
    whatsoever, is recorded as a transport fault worth another attempt, because
    the status said the request itself succeeded. An answer that has the field
    and a transcript beside it, but nothing in the field, is a shape this
    adapter cannot place on the timeline, and asking again would produce the
    same shape at the same price.
    """
    phrases = _field(body, "phrases")
    if not isinstance(phrases, (list, tuple)):
        # A missing array, a null one, and a single object where a list belongs
        # are all the same thing from here: nothing that can be read as phrases.
        # The last of those is the dangerous one, because iterating it yields
        # words made of nothing and no error at all.
        return (
            "the answer was not a transcription result. The status said the request "
            "had succeeded, so this is a truncated or rewritten body rather than a "
            "refusal.",
            True,
        )
    if not phrases and _combined_text(body):
        return (
            "the answer carried a transcript with no phrases in it, so there is "
            "nothing that can be placed on the recording's timeline.",
            False,
        )
    return None


def _locales_in(definition: dict[str, Any]) -> tuple[str, ...]:
    """The locales the request actually named, which may be none."""
    value = definition.get("locales")
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)


def _phrases_in(definition: dict[str, Any]) -> tuple[str, ...]:
    """The phrases the request actually carried, which may be none."""
    phrase_list = definition.get("phraseList")
    phrases = phrase_list.get("phrases") if isinstance(phrase_list, dict) else None
    if not isinstance(phrases, (list, tuple)):
        return ()
    return tuple(str(item) for item in phrases)


def _combined_text(body: Any) -> str:
    """The whole-transcript text the service returns beside the phrases.

    Read only to tell an empty answer from a lost one. It carries no times and
    no locales, so it is never turned into words.
    """
    combined = _field(body, "combinedPhrases") or []
    if isinstance(combined, dict):
        combined = [combined]
    parts = [str(_field(entry, "text") or "").strip() for entry in combined]
    return " ".join(part for part in parts if part)


def _detail_from(body: Any) -> str | None:
    """Pull the human-readable message out of the service's error body."""
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("code")
            return str(message) if message else None
        return str(error) if error else None
    text = str(body).strip() if body else ""
    return text or None


def _header(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get(name)
    except AttributeError:
        return None
    return None if value is None else str(value)


def _is_punctuation_only(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return all(unicodedata.category(character).startswith("P") for character in stripped)


def _field(source: Any, name: str, default: Any = None) -> Any:
    """Read a field from the answer, whichever shape it arrives in."""
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


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


def _content_type_for(path: Path) -> str:
    return _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


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
    """Turn whatever went wrong on the wire into a sentence, and say if it may be retried.

    The client's exception classes are not imported. Reading the status off
    the error covers every one of them, and doing it this way means the module
    never imports the package merely to catch its errors, which would defeat
    the point of importing it lazily.
    """
    status = _as_int(getattr(error, "status_code", None))
    name = type(error).__name__
    retryable = False
    if status is not None:
        retryable = status in (408, 429) or 500 <= status < 600
    elif any(hint in name.lower() for hint in _RETRYABLE_ERROR_NAMES):
        # A request that timed out on the way back is worth repeating: the
        # service is known to accept work and then take longer than the client
        # will wait, and that arrives as a network error rather than an HTTP
        # one.
        retryable = True

    detail = str(error).strip() or name
    if status is not None:
        return f"Microsoft answered with an error ({status}): {detail}", retryable, status
    return f"Microsoft could not be reached: {detail}", retryable, None
