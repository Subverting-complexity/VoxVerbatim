"""Talking to OpenAI, which hears words well and cannot tell you when.

OpenAI is here for its ears, not for its clock. It is consistently strong on
what was actually said, particularly with names and unusual terms, and it
takes three separate kinds of context: a sentence or two about the
recording, a list of terms to listen out for, and the languages the audio
might be in. Those three map exactly onto what this application already
knows about a recording before it sends it.

What the model does not do is time anything. ``gpt-transcribe`` returns no
word times, no segment times, no log probabilities and no speakers. Those
are not switches somebody forgot to turn on; the model does not have them.
So every token this adapter returns has a start and an end of ``None``, and
that is correct rather than broken. It looks like a bug to anyone who has
not read the specification, and it is worth saying plainly: these words get
their positions later, when the alignment engine matches them against the
backbone service's words, which were measured against the audio. Text and
timing are separate throughout this application precisely so that a service
this good at one and incapable of the other is still useful.

That is also why this adapter refuses to use any other model. Word times and
log probabilities can be had from ``whisper-1`` and from the
``gpt-4o-transcribe`` family, and reaching for one of them to get them back
is exactly the trade the specification forbids. Those models are worse at
the words, and the words are the only thing this service is here to
contribute. A model name outside the ``gpt-transcribe`` family therefore
switches this service off and says why, and the run carries on without it.

Two further consequences of the missing times belong to other modules, but
are worth knowing here. Where a recording has to be cut up to fit inside the
25 MB upload limit, the overlap between two chunks cannot be resolved by
comparing times, because there are none; it has to be resolved on the text
itself. And the language the response reports covers the whole file, so it
is a second opinion to set beside ElevenLabs' rather than an answer for any
particular span.

The response carries no model name, no identifier and no fingerprint. The
only handle on a particular request is the HTTP request id in the response
headers, which is why the call is made through ``with_raw_response`` and
that id is written into provenance beside the model name that was sent.

All three kinds of context are named parameters of the library's own call,
which is why this module needs a version of the package new enough to have
them. An older one has no ``keywords`` and no ``languages`` argument and
raises before the request is built, so the floor in ``requirements.txt`` is
a working requirement rather than a preference.

The client library is imported inside the method that needs it, so a machine
without the ``openai`` package loses this one service rather than failing to
start.
"""

from __future__ import annotations

import logging
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
    retry_after_seconds,
)

# Somebody else's rule about what a keyword may contain, read from the one
# module that holds every such rule, so that a change at OpenAI's end is
# corrected in one place. It is renamed on the way in only because inside this
# module there is no other service to confuse it with.
#
# This direction is the only one that works. context.py reads
# ProviderCapabilities out of providers/base.py, so base.py must never import
# context.py and providers/__init__.py must never import an adapter eagerly.
# Either would close the loop. The ElevenLabs and AssemblyAI adapters read
# their own limits the same way.
from audio_transcriber.transcription.context import (
    OPENAI_UNSUPPORTED_KEYWORD_CHARACTERS as UNSUPPORTED_KEYWORD_CHARACTERS,
)

_log = logging.getLogger(__name__)

#: The package that must be installed for this service to work at all.
PACKAGE = "openai"

#: The only model family this adapter will ever call. The name is settable,
#: so a later member of the same family works without a code change, but
#: nothing outside the family is reachable through this adapter at all.
MODEL_FAMILY = "gpt-transcribe"

DEFAULT_MODEL = MODEL_FAMILY

#: Confirmed, and small enough that most recordings of any length have to be
#: cut up before they can be sent.
MAXIMUM_FILE_BYTES = 25 * 1024 * 1024

OPENAI_CAPABILITIES = ProviderCapabilities(
    # All four of these are false because the model cannot do them, not
    # because they were left switched off. Declaring them honestly is what
    # stops anything upstream from asking this service for timing and then
    # quietly accepting invented numbers.
    word_timings=False,
    diarisation=False,
    speaker_count_hint=False,
    word_confidence=False,
    language_detection=True,
    per_word_language=False,
    vocabulary_biasing=True,
    context_prompt=True,
    time_window=False,
    forced_alignment=False,
    languages=frozenset({Language.ENGLISH, Language.GERMAN, Language.AFRIKAANS}),
    maximum_file_bytes=MAXIMUM_FILE_BYTES,
    maximum_duration_seconds=None,
    accepted_containers=frozenset({"wav", "mp3", "m4a", "flac", "ogg", "webm", "mp4", "mpga"}),
)

#: Settings keys that could hold a credential. Anything named like one of
#: these is dropped before a request is written into provenance, because a
#: transcript folder is copied around and read by people.
_CREDENTIAL_NAMES = ("api_key", "apikey", "key", "token", "secret", "password", "authorization")

#: Errors worth trying again even though they carry no HTTP status, because
#: they happened before the far end answered at all.
#: Matched against the exception's class name. The library raises
#: ``APITimeoutError`` and ``APIConnectionError``; the underlying
#: ``httpx.ConnectError`` is spelt without the "ion", so "connect".
_RETRYABLE_ERROR_NAMES = ("timeout", "connect", "protocol")

#: How the extras are merged, which is what makes the two lists below
#: necessary. They go to the library as ``extra_body``, which it lays *over*
#: the named arguments rather than under them, so an extra sharing a name with
#: one of this adapter's arguments silently replaces it. The escape hatch is
#: worth having, but it cannot be allowed to reach inside the request and
#: change a decision that was made for a reason.

#: The three kinds of context this adapter works out from the recording.
#:
#: All three arrive as extras as a matter of course: the context module shapes
#: them for this service, and the pipeline passes that same shape along as
#: extras as well as through the request's own fields. Left in, the second copy
#: wins and every rule applied to the first copy is undone without a word.
_DECIDED_HERE = ("prompt", "keywords", "languages")

#: Names an extra may never set, and why each one is refused.
#:
#: ``model`` is the serious one. This adapter refuses every model outside the
#: ``gpt-transcribe`` family, and refuses it before the request is built, for
#: the reason set out at the top of this file. An extra named ``model`` walked
#: straight past that: the check passed on the settings model, the older model
#: in the extras replaced it on the wire, and the request record went on
#: naming the model that did not run. A record that names the wrong model is
#: worse than no record, because the whole purpose of keeping one is to
#: explain why a later run of the same recording said something different.
#:
#: The other three change the answer rather than the question, and leave this
#: adapter holding something it cannot read.
#:
#: The reasons are written out per name rather than shared, because they are
#: what the log says, and somebody who has just lost a parameter they wrote
#: deliberately needs to know which of these two very different things
#: happened.
_NEVER_FROM_EXTRAS: dict[str, str] = {
    "model": (
        "this application calls only the {family} family, and a model chosen "
        "here would run without the check that enforces that, leaving the "
        "request record naming a model that never ran"
    ),
    "stream": (
        "this adapter reads one whole answer rather than a stream of pieces, "
        "and would be left holding something it cannot read"
    ),
    "response_format": (
        "this adapter reads a JSON answer, and asking for text or subtitles "
        "instead would leave it holding something it cannot read"
    ),
    "file": "the audio to send is the recording itself, which is already attached",
}

_CONTENT_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
    ".mpga": "audio/mpeg",
}


def is_permitted_model(model: str) -> bool:
    """Whether this model name is one this adapter is allowed to call.

    The test is deliberately narrow. ``whisper-1`` and every member of the
    ``gpt-4o-transcribe`` family fail it, including the variants that would
    hand back the word timings and log probabilities this service otherwise
    lacks. That temptation is the entire reason the check exists.
    """
    name = (model or "").strip().lower()
    return name == MODEL_FAMILY or name.startswith(f"{MODEL_FAMILY}-")


class OpenAiTranscriptionProvider(TranscriptionProvider):
    """The OpenAI speech-to-text service.

    Everything it needs arrives through the constructor. It knows nothing
    about the application's settings file: a separate registry reads Settings
    and calls this, which keeps a settings change from rippling through the
    adapters and lets a test build a working one out of a fake client.
    """

    provider = Provider.OPENAI
    capabilities = OPENAI_CAPABILITIES

    def __init__(
        self,
        api_key: str = "",
        model: str = DEFAULT_MODEL,
        parameters: dict[str, Any] | None = None,
        *,
        temperature: float | None = None,
        chunking_strategy: str | None = None,
        timeout_seconds: float = 600.0,
        maximum_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        """Set up the adapter.

        ``parameters`` holds the free-form extras from Settings, merged with
        the extras on an individual request and sent through the library's
        ``extra_body``. The request wins where both name the same thing,
        because it is the more specific of the two.

        ``client`` exists so a test, or a caller that wants to share one
        connection pool, can hand in a ready-made client. Left as ``None`` it
        means "build a real one when first needed", which is what keeps the
        library import out of application start-up.
        """
        self._api_key = api_key or ""
        self._model = model or DEFAULT_MODEL
        self._parameters = dict(parameters or {})
        self._temperature = temperature
        self._chunking_strategy = chunking_strategy
        self._timeout_seconds = timeout_seconds
        # Read by the retry loop in the base class. The library's own retry
        # is switched off in ``_build_client`` so that nothing is retried
        # twice over: the library's retries multiplied by ours would turn
        # two configured retries into eight attempts on a dead service.
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
        if not self._model.strip():
            return "no model has been chosen"
        if not is_permitted_model(self._model):
            return (
                f"the model is set to {self._model!r}, and this application uses "
                f"only the {MODEL_FAMILY} family. Older models return word "
                "timings, but they are worse at the words themselves, which is "
                "the only thing OpenAI is asked for here"
            )
        return None

    # -- Doing the work ------------------------------------------------

    def _transcribe(
        self,
        request: TranscriptionRequest,
        cancelled: CancelCheck | None = None,
    ) -> ProviderResult:
        problem = self.describe_configuration_problem()
        if problem is not None:
            # Raised before anything is opened or sent. A wrong model name
            # and a missing key both stop the service here, so neither
            # reaches the network and neither takes the run down.
            raise ProviderNotConfigured(self.provider, problem)

        client = self._resolve_client()

        if cancelled is not None and cancelled():
            raise ProviderError("OpenAI was not called, because the run was stopped.")

        size_bytes = _file_size(request.audio_path)
        if size_bytes > MAXIMUM_FILE_BYTES:
            raise ProviderError(
                f"{request.audio_path.name} is {size_bytes / (1024 * 1024):.1f} MB, and "
                f"OpenAI accepts at most {MAXIMUM_FILE_BYTES // (1024 * 1024)} MB. "
                "It has to be cut into chunks first."
            )

        languages = _language_codes(request.languages)
        keywords = _keywords_for(request.vocabulary_terms)

        arguments: dict[str, Any] = {"model": self._model}
        if request.context_prompt.strip():
            arguments["prompt"] = request.context_prompt.strip()
        if keywords:
            arguments["keywords"] = list(keywords)
        if languages:
            arguments["languages"] = list(languages)
        if self._temperature is not None:
            arguments["temperature"] = self._temperature
        if self._chunking_strategy is not None:
            arguments["chunking_strategy"] = self._chunking_strategy

        # Built after the arguments, because what survives here depends on
        # what was decided above.
        extra_body = _extras_for({**self._parameters, **request.extra_parameters}, arguments)

        started_at = datetime.now()
        try:
            # Opened inside the attempt, so that a retry sends the whole file
            # again rather than what was left of a half-read handle.
            with request.audio_path.open("rb") as handle:
                # Through the raw response, because the transcription itself
                # carries no identifier of any kind and the request id in the
                # headers is the only handle provenance can record.
                raw = client.audio.transcriptions.with_raw_response.create(
                    file=(
                        request.audio_path.name,
                        handle,
                        _content_type_for(request.audio_path),
                    ),
                    extra_body=extra_body,
                    timeout=self._timeout_seconds,
                    **arguments,
                )
                request_id = _header(raw, "x-request-id")
                # Taken before the answer is parsed, because this is the
                # body the server actually sent, character for character.
                # Anything rebuilt from the parsed object afterwards would
                # already have lost whatever this adapter does not read.
                body = _raw_body(raw)
                response = raw.parse()
        except Exception as error:  # noqa: BLE001 - every failure becomes a result
            # Converted here rather than allowed out, because a failed
            # request still has provenance worth keeping: what was asked, of
            # which model, and what came back instead of a transcript.
            failure = _as_provider_error(error)
            _log.debug(
                "OpenAI did not answer: %s (worth retrying: %s)",
                failure,
                failure.retryable,
                exc_info=error,
            )
            record = self._request_record(
                request,
                arguments=arguments,
                extra_body=extra_body,
                keywords=keywords,
                languages=languages,
                size_bytes=size_bytes,
                started_at=started_at,
                request_id=_as_text(getattr(error, "request_id", None)),
                detected=(),
                succeeded=False,
                error=str(failure),
            )
            # The record rides on the error rather than being returned, so
            # the retry loop in the base class can try again where that may
            # help and still hand back the full record when it gives up.
            failure.result = ProviderResult(
                provider=self.provider,
                request=record,
                error=str(failure),
                # A body that explains a refusal is often the most useful
                # thing in the whole folder afterwards, so it is kept on the
                # same footing as a successful answer.
                raw_response=_raw_body_of_failure(error),
            )
            raise failure from error

        detected = _detected_languages(_field(response, "languages"))
        tokens = self._tokens_from(_field(response, "text") or "", request)
        record = self._request_record(
            request,
            arguments=arguments,
            extra_body=extra_body,
            keywords=keywords,
            languages=languages,
            size_bytes=size_bytes,
            started_at=started_at,
            request_id=request_id,
            detected=detected,
            succeeded=True,
            error=None,
        )
        return ProviderResult(
            provider=self.provider,
            tokens=tokens,
            request=record,
            detected_language=Language.from_code(detected[0] if detected else None),
            # The whole answer, not the part of it this adapter understood.
            # It goes no further than the pipeline, which hands it to the
            # store; an adapter that knew where transcripts are kept would
            # be far harder to test and would need a folder it has no other
            # use for.
            raw_response=body,
        )

    # -- Turning the answer into evidence -------------------------------

    def _tokens_from(self, text: str, request: TranscriptionRequest) -> list[ProviderToken]:
        """Cut the returned text into words, exactly as the service wrote them.

        The service returns one run of text with its own punctuation and
        casing, so the split is on whitespace and nothing else. Stripping
        commas off the ends, or splitting "don't" in two, would change what
        the service said, and the whole value of this evidence is that it is
        what the service said. The normalisation used for comparing words
        lives elsewhere and never touches the text stored here.

        Every token comes back without a start or an end. There is nothing
        missing: the model does not time words, and alignment against the
        backbone is what will place them.
        """
        tokens: list[ProviderToken] = []
        for index, word in enumerate(text.split()):
            tokens.append(
                ProviderToken(
                    provider=self.provider,
                    index=index,
                    text=word,
                    start=None,
                    end=None,
                    chunk_index=request.chunk_index,
                    is_punctuation=_is_punctuation_only(word),
                )
            )
        return tokens

    # -- Recording what happened ----------------------------------------

    def _request_record(
        self,
        request: TranscriptionRequest,
        *,
        arguments: dict[str, Any],
        extra_body: dict[str, Any],
        keywords: tuple[str, ...],
        languages: tuple[str, ...],
        size_bytes: int,
        started_at: datetime,
        request_id: str | None,
        detected: tuple[str, ...],
        succeeded: bool,
        error: str | None,
    ) -> ProviderRequestRecord:
        """Note what was asked and what came back, without the credential.

        The response has no model field, so the model name that was sent is
        recorded here with the HTTP request id that came back. Those two
        together are the only way to identify this request afterwards, and
        the model behind ``gpt-transcribe`` can change underneath the name.
        """
        recorded = {key: value for key, value in arguments.items() if key != "keywords"}
        recorded["extra_parameters"] = _without_credentials(extra_body)
        # Named for the terms rather than for OpenAI's word for them, because
        # the filter below drops anything whose name contains "key", and a
        # count recorded as "keyword_count" is thrown away before it is
        # written. The other adapters record theirs under this name too.
        recorded["vocabulary_term_count"] = len(keywords)
        if request_id:
            recorded["response_request_id"] = request_id
        if detected:
            # Whole-file rather than per-span, but a second, independent
            # language opinion is worth keeping when no service gives one
            # per word.
            recorded["response_languages"] = list(detected)

        window = request.window
        chunk = ChunkRecord(
            provider=self.provider,
            chunk_index=request.chunk_index,
            canonical_start=window.start if window else request.canonical_offset,
            canonical_end=(
                window.end
                if window
                else request.canonical_offset + (request.duration or 0.0)
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
            language_configuration=",".join(languages) if languages else "auto",
            vocabulary_terms=keywords,
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
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        """Build a real client, reporting a missing package as such.

        The import lives here rather than at the top of the module so that
        the application starts on a machine where the package was never
        installed. That machine loses OpenAI and is told why.
        """
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ProviderUnavailable(self.provider, PACKAGE) from error
        # Zero, so that the only retry is the one in the base class. The
        # library's retry is sound, unlike another library's here, but two
        # retry loops stacked multiply rather than add.
        return OpenAI(api_key=self._api_key, max_retries=0)


# -- Helpers -------------------------------------------------------------


def _language_codes(languages: tuple[Language, ...]) -> tuple[str, ...]:
    """The candidate language codes to offer, in the order they were given.

    All of them are sent, not one. The model takes a list of candidates, and
    narrowing it to a single code would be a guess where the recording may
    genuinely switch between two or three.
    """
    codes: list[str] = []
    for language in languages:
        if language is Language.UNKNOWN:
            continue
        if language.value not in codes:
            codes.append(language.value)
    return tuple(codes)


def _keywords_for(terms: tuple[str, ...]) -> tuple[str, ...]:
    """The vocabulary cut down to what the service will actually accept.

    A term holding one of the characters OpenAI refuses is dropped wherever it
    sits in the list, because sending it would have the whole request refused
    and cost the chunk its transcript rather than costing that one word. The
    shaping module applies the same rule before this ever runs; this is here
    for the callers that do not go through it, and because a rule worth
    obeying is worth obeying at the point the request is built.
    """
    kept: list[str] = []
    for term in terms:
        cleaned = (term or "").strip()
        if not cleaned:
            continue
        offending = sorted(
            {character for character in cleaned if character in UNSUPPORTED_KEYWORD_CHARACTERS}
        )
        if offending:
            listed = " and ".join(_named(character) for character in offending)
            _log.info(
                "The term %r was not sent to OpenAI: it contains %s, and a keyword "
                "holding one of those has the whole request refused.",
                cleaned,
                listed,
            )
            continue
        kept.append(cleaned)
    return tuple(kept)


def _named(character: str) -> str:
    """A character as something a person can read in a log line.

    Two of the four are invisible, and a log saying a term contains "" is no
    use to the person trying to work out which term to change.
    """
    if character == "\r":
        return "a carriage return"
    if character == "\n":
        return "a line feed"
    return f'"{character}"'


def _extras_for(extras: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """The free-form extras, with the ones this adapter decides removed.

    See :data:`_DECIDED_HERE` and :data:`_NEVER_FROM_EXTRAS` for why leaving
    them in would undo the request that was just built.

    A context parameter is removed quietly where the extra says the same thing
    as the argument, because that is the shaped context arriving by its second
    route and there is nothing anybody needs to do about it. Where the two
    differ, somebody has written that parameter into Settings by hand and is
    entitled to know it was not sent as written. The rest are always worth a
    word, because nothing sends them by accident.
    """
    kept = dict(extras)
    for name in _DECIDED_HERE:
        if name not in kept:
            continue
        supplied = kept.pop(name)
        sent = arguments.get(name)
        if _same_request_value(supplied, sent):
            continue
        _log.warning(
            "The %s parameter was not sent to OpenAI as written. This adapter "
            "works it out from the recording, and %s.",
            name,
            f"sent {sent!r} instead" if name in arguments else "sent nothing for it",
        )
    for name, reason in _NEVER_FROM_EXTRAS.items():
        if name not in kept:
            continue
        supplied = kept.pop(name)
        _log.warning(
            "The %s parameter was not sent to OpenAI, although it was set to %r: %s.",
            name,
            supplied,
            reason.format(family=MODEL_FAMILY),
        )
    return kept


def _same_request_value(supplied: Any, sent: Any) -> bool:
    """Whether an extra says the same thing as the argument it shares a name with.

    Lists and tuples of the same codes are the same instruction, and the two
    reach here in different shapes: the arguments hold lists, while what
    somebody types into Settings may be either. Empty and absent are the same
    thing too, because neither asks for anything.
    """
    if isinstance(supplied, (list, tuple)) and isinstance(sent, (list, tuple)):
        return list(supplied) == list(sent)
    if not supplied and sent is None:
        return True
    return bool(supplied == sent)


def _detected_languages(reported: Any) -> tuple[str, ...]:
    """The language codes out of what the service reported detecting.

    The field is a list of small objects, each carrying the code in a ``code``
    field, rather than a list of codes. Treating one of those objects as a
    string yields its repr, which nothing recognises as a language, so the
    service's only language opinion would be lost while appearing to be
    present.

    What the field is has to be established before it is walked, rather than
    left to the loop. The library builds the answer without validating it, so
    a field that arrives as something other than a list stays that way, and
    iterating whatever turns up is how a shape change becomes fabricated
    evidence instead of a missing one. A bare ``"en"`` walked one character at
    a time yields "e" and "n"; a bare mapping yields its keys, and "code"
    would be recorded as a language the service never named. Anything not
    recognised is treated as nothing reported, which is the honest answer and
    the one that cannot invent a fact.

    A single string is the one unrecognised shape worth reading, because it is
    the plausible way this field could be simplified later, and reading it
    whole is unambiguous.
    """
    if isinstance(reported, str):
        entries: list[Any] = [reported]
    elif isinstance(reported, (list, tuple)):
        entries = list(reported)
    else:
        if reported is not None:
            # Not an error: the transcript itself is fine and the run carries
            # on. But it is the one thing here nobody would otherwise notice.
            _log.warning(
                "OpenAI reported its detected languages as %s, which this "
                "adapter cannot read, so no language was taken from them.",
                type(reported).__name__,
            )
        return ()

    codes: list[str] = []
    for entry in entries:
        # An entry with no code at all reads as None and is passed over. That
        # covers a mapping without the field and an object without it alike.
        code = entry if isinstance(entry, str) else _field(entry, "code")
        text = (_as_text(code) or "").strip()
        if text:
            codes.append(text)
    return tuple(codes)


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


def _raw_body(raw: Any) -> Any:
    """The answer as the server sent it, taken off the raw HTTP response.

    The specification requires every response to be kept unchanged, so that
    a reprocessing run that disagrees with the original can be traced to a
    changed service rather than a changed application. The undecoded body is
    the most faithful form there is, and it is already to hand, because the
    call goes through ``with_raw_response`` for the request id anyway.

    Serialising the parsed object instead would be a rebuilt copy, and would
    silently drop any field the installed library does not model yet, which
    is precisely the sort of thing a later question turns out to be about.
    """
    text = getattr(raw, "text", None)
    if isinstance(text, str) and text:
        return text
    http_response = getattr(raw, "http_response", None)
    text = getattr(http_response, "text", None)
    if isinstance(text, str) and text:
        return text
    _log.debug("The OpenAI response carried no readable body to keep.")
    return None


def _raw_body_of_failure(error: Exception) -> Any:
    """Whatever the service said when it refused, in the most faithful form."""
    response_text = getattr(getattr(error, "response", None), "text", None)
    if isinstance(response_text, str) and response_text:
        return response_text
    return getattr(error, "body", None)


def _header(raw: Any, name: str) -> str | None:
    headers = getattr(raw, "headers", None)
    if headers is None:
        return None
    try:
        return _as_text(headers.get(name))
    except AttributeError:
        return None


def _as_text(value: Any) -> str | None:
    return None if value is None else str(value)


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


def _as_provider_error(error: Exception) -> ProviderError:
    """Turn whatever the library threw into one sentence and a retry verdict.

    The library's exception classes are not imported. Reading the status
    code off the error covers every class it has, and doing it this way
    means the module never imports the package merely to catch its errors,
    which would undo the point of importing it lazily.
    """
    status = _as_int(getattr(error, "status_code", None))
    name = type(error).__name__
    retryable = False
    if status is not None:
        retryable = status in (408, 409, 429) or 500 <= status < 600
    elif any(hint in name.lower() for hint in _RETRYABLE_ERROR_NAMES):
        # No status at all means the request never arrived, which is usually
        # worth one more attempt.
        retryable = True

    detail = str(error).strip() or name
    request_id = _as_text(getattr(error, "request_id", None))
    sentence = (
        f"OpenAI answered with an error ({status}): {detail}"
        if status is not None
        else f"OpenAI could not be reached: {detail}"
    )
    if request_id:
        sentence += f" (request {request_id})"
    response = getattr(error, "response", None)
    return ProviderError(
        sentence,
        retryable=retryable,
        status_code=status,
        # A rate limit says in its headers how long to wait, and the retry
        # loop honours it rather than calling back sooner to be refused again.
        retry_after=retry_after_seconds(getattr(response, "headers", None)),
    )


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
