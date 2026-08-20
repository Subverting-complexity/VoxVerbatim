"""Talking to ElevenLabs Scribe, the only service here that times words.

ElevenLabs earns its place as the backbone of a transcript for one reason:
it is the only one of these services that reports a start and an end for
every word it hears, and it tells the speakers apart while it is at it.
Everything further on that plays a word, seeks to it, or draws it against a
waveform is ultimately standing on numbers that came out of this module.

Because those numbers matter that much, the most important thing this
adapter does is convert them. A service is often sent a chunk cut out of the
middle of a recording, and it answers in its own local time, counting from
the start of whatever it was given. Every time that leaves this module has
``canonical_offset`` added to it first, so a word three minutes into a chunk
that itself began ten minutes in comes back at thirteen minutes, on the same
timeline as every other word in the transcript.

Three peculiarities of the service are worth knowing before reading the
code.

The first is that it does not return only words. The spacing between words
comes back as an entry of its own, punctuation often does too, and so do
non-speech sounds such as laughter, which arrive with a type of
``audio_event``. All of them are kept, because every one of them carries
timing that is real and measured, and none of them is matched against
another service's text, because matching "(laughter)" to a word OpenAI heard
would produce nonsense. They are marked apart from each other, though, and
that matters: spacing and punctuation are marks on the page, while an audio
event is something that happened in the room. A reader may well want to see
that somebody laughed, and would never want to see a stray comma, so the
service's own type is what decides, rather than the shape of the text.

The second is that it reports the language of the whole file and never of a
word or a segment. There is no per-word language field on the response, so
this adapter declares ``per_word_language`` as false and leaves each token's
language unknown. It would be easy, and wrong, to copy the file's language
onto every word to fill the gap in. Nothing downstream can tell a value that
was measured from one that was copied, so reconciliation would weigh that
invented per-word evidence exactly as though the service had really reported
it, and a German sentence inside an English recording would be argued
against by a service that had never looked at it. The whole-file answer and
its probability are recorded on the result instead, where they are honestly
one weak signal among several. Please leave it that way.

The third is money, and it is easy to get the wrong way round. Sending any
key terms at all adds a 20 per cent surcharge, which is simply the price of
vocabulary biasing and cannot be avoided by sending fewer of them. Sending
more than a hundred adds a second charge on top: the request is then billed
a minimum of 20 seconds of audio however short the recording is. That
second charge is the one a limit can avoid, so the list is cut to a hundred
before it goes. The vocabulary arrives ordered most-specific first, which is
what makes cutting the tail the right place to cut.

A key term can also be refused on its own account, for being 50 characters
or longer, for running to more than five words, or for containing one of the
characters the service does not accept. A refusal of that kind fails the
whole request rather than dropping the one term, so any term that would
cause it is left behind here instead.

Forced alignment lives here too, because it is the same service and the same
key, but it is a genuinely different job: it is given words that are already
known and asked only where they fall. It cannot do Afrikaans, which is why
the specification has an Afrikaans fallback at all.

The client library is imported inside the methods that need it. A machine
without the ``elevenlabs`` package loses this one service and says so,
rather than failing to start.
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
    ForcedAligner,
    ProviderCapabilities,
    ProviderError,
    ProviderNotConfigured,
    ProviderUnavailable,
    TranscriptionProvider,
    TranscriptionRequest,
)

# Every limit a key term is held to is declared at the top of context.py,
# beside every other service's, so that a number ElevenLabs changes is
# corrected in one place. They are renamed on the way in only because inside
# this module there is no other service to confuse them with.
#
# KEYTERM_MINIMUM_CHARGE_THRESHOLD is the count above which a request is
# billed a minimum of 20 seconds of audio. DEFAULT_MAXIMUM_KEYTERMS sits
# exactly on it rather than under it, because the threshold is "more than",
# and because the separate 20 per cent keyterm surcharge applies to any
# request carrying terms at all and so is not avoided by sending fewer.
#
# This direction is the only one that works. context.py reads
# ProviderCapabilities out of providers/base.py, so base.py must never import
# context.py and providers/__init__.py must never import an adapter eagerly.
# Either would close the loop. The AssemblyAI adapter reads its own limits
# the same way.
from audio_transcriber.transcription.context import (
    ELEVENLABS_KEYTERM_CHARACTER_LIMIT as KEYTERM_CHARACTER_LIMIT,
    ELEVENLABS_KEYTERM_MINIMUM_CHARGE_THRESHOLD as KEYTERM_MINIMUM_CHARGE_THRESHOLD,
    ELEVENLABS_MAXIMUM_KEYTERM_CHARACTERS as MAXIMUM_KEYTERM_CHARACTERS,
    ELEVENLABS_MAXIMUM_KEYTERM_WORDS as MAXIMUM_KEYTERM_WORDS,
    ELEVENLABS_MAXIMUM_KEYTERMS as DEFAULT_MAXIMUM_KEYTERMS,
    ELEVENLABS_UNSUPPORTED_KEYTERM_CHARACTERS as UNSUPPORTED_KEYTERM_CHARACTERS,
)

_log = logging.getLogger(__name__)

#: The package that must be installed for this service to work at all.
PACKAGE = "elevenlabs"

#: The current batch model. ``scribe_v2_realtime`` is for streaming and is
#: rejected by this endpoint, and ``scribe_v1`` is deprecated. The name is
#: settable, because the library types it loosely enough that a future model
#: works without an upgrade.
DEFAULT_MODEL = "scribe_v2"

_GIGABYTE = 1024 * 1024 * 1024

#: The API reference says under 5 GB and the capabilities page says 3 GB.
#: The smaller of two disagreeing numbers is the safe one to design against.
MAXIMUM_FILE_BYTES = 3 * _GIGABYTE

#: Ten hours, which is far longer than anything this application expects.
MAXIMUM_DURATION_SECONDS = 10 * 60 * 60

#: Forced alignment publishes a smaller limit than transcription does, and
#: here too the two vendor pages disagree, at 1 GB and 3 GB.
MAXIMUM_ALIGNMENT_FILE_BYTES = 1 * _GIGABYTE

#: The most speakers the service will predict. Asking for more is refused, so
#: the count is brought down to this before it goes rather than after. The
#: settings layer stops at the same number, for the same reason, but this
#: adapter is what talks to the service and cannot assume it was asked
#: politely.
MAXIMUM_SPEAKERS = 32

#: The word type the service uses for a non-speech sound.
AUDIO_EVENT_TYPE = "audio_event"

#: The word type the service uses for the gap between two words.
SPACING_TYPE = "spacing"

ELEVENLABS_CAPABILITIES = ProviderCapabilities(
    word_timings=True,
    diarisation=True,
    speaker_count_hint=True,
    word_confidence=True,
    language_detection=True,
    # Deliberately false. The service reports one language for the file and
    # has no per-word or per-segment language field at all.
    per_word_language=False,
    vocabulary_biasing=True,
    context_prompt=False,
    time_window=False,
    forced_alignment=True,
    languages=frozenset({Language.ENGLISH, Language.GERMAN, Language.AFRIKAANS}),
    maximum_file_bytes=MAXIMUM_FILE_BYTES,
    maximum_duration_seconds=MAXIMUM_DURATION_SECONDS,
    accepted_containers=frozenset(
        {"wav", "mp3", "flac", "m4a", "aac", "aiff", "ogg", "opus", "webm"}
    ),
)

#: The languages our own set that the forced aligner handles. It covers 29
#: languages and Afrikaans is not one of them, which is the whole reason the
#: specification asks for an Afrikaans phrase-span fallback.
ALIGNER_LANGUAGES = frozenset({Language.ENGLISH, Language.GERMAN})

#: Settings keys that could hold a credential. Anything named like one of
#: these is dropped before the request is written into provenance, because a
#: transcript folder is copied around and read by people.
_CREDENTIAL_NAMES = ("api_key", "apikey", "xi-api-key", "key", "token", "secret", "password")

#: Errors worth trying again even though they carry no HTTP status, because
#: they happened before the far end answered at all.
_RETRYABLE_ERROR_NAMES = ("timeout", "connection")

_CONTENT_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".webm": "audio/webm",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
}


class ElevenLabsProvider(TranscriptionProvider):
    """The ElevenLabs Scribe speech-to-text service.

    Everything it needs arrives through the constructor. It deliberately
    knows nothing about the application's settings file: a separate registry
    reads Settings and calls this, so that adding a setting changes one
    module rather than five, and so that a test can build a working adapter
    out of nothing but a fake client.
    """

    provider = Provider.ELEVENLABS
    capabilities = ELEVENLABS_CAPABILITIES

    def __init__(
        self,
        api_key: str = "",
        model: str = DEFAULT_MODEL,
        parameters: dict[str, Any] | None = None,
        *,
        tag_audio_events: bool = True,
        diarisation_threshold: float | None = None,
        timestamps_granularity: str = "word",
        maximum_keyterms: int = DEFAULT_MAXIMUM_KEYTERMS,
        timeout_seconds: float = 600.0,
        maximum_retries: int = 3,
        client: Any | None = None,
    ) -> None:
        """Set up the adapter.

        ``parameters`` holds the free-form extras from Settings, which are
        merged with the extras on an individual request and sent through the
        library's pass-through mechanism. The request wins where both name
        the same thing, because it is the more specific of the two.

        ``client`` exists so that a test, or a future caller that wants to
        share one connection pool, can hand in a ready-made client. Left as
        ``None`` it means "build a real one when first needed", which is
        also what keeps the library import out of application start-up.
        """
        self._api_key = api_key or ""
        self._model = model or DEFAULT_MODEL
        self._parameters = dict(parameters or {})
        self._tag_audio_events = tag_audio_events
        self._diarisation_threshold = diarisation_threshold
        self._timestamps_granularity = timestamps_granularity
        self._maximum_keyterms = max(0, maximum_keyterms)
        self._timeout_seconds = timeout_seconds
        self._maximum_retries = maximum_retries
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
            raise ProviderError("ElevenLabs was not called, because the run was stopped.")

        size_bytes = _file_size(request.audio_path)
        if size_bytes > MAXIMUM_FILE_BYTES:
            raise ProviderError(
                f"{request.audio_path.name} is {size_bytes / _GIGABYTE:.1f} GB, and "
                f"ElevenLabs accepts at most {MAXIMUM_FILE_BYTES / _GIGABYTE:.0f} GB. "
                "It has to be cut into chunks first."
            )

        extras = {**self._parameters, **request.extra_parameters}
        keyterms = self._keyterms_for(_terms_for(request, self._parameters))
        language_code = _single_language_code(request.languages)
        diarise, threshold = self._diarisation_for(request, extras)
        body_parameters = self._body_parameters_for(extras)

        arguments: dict[str, Any] = {
            "model_id": self._model,
            "timestamps_granularity": self._timestamps_granularity,
            "tag_audio_events": self._tag_audio_events,
            "diarize": diarise,
        }
        if language_code is not None:
            arguments["language_code"] = language_code
        if diarise and request.expected_speaker_count > 1:
            arguments["num_speakers"] = min(request.expected_speaker_count, MAXIMUM_SPEAKERS)
            if request.expected_speaker_count > MAXIMUM_SPEAKERS:
                _log.info(
                    "ElevenLabs was asked for %d speakers rather than the %d this "
                    "recording expects, because %d is the most it will predict.",
                    MAXIMUM_SPEAKERS,
                    request.expected_speaker_count,
                    MAXIMUM_SPEAKERS,
                )
        if threshold is not None:
            # The service accepts this only on a diarising request that has
            # not also been told how many speakers to expect, and refuses the
            # whole request otherwise. A speaker count is the stronger hint of
            # the two and comes from the person who listened to the recording,
            # so where both are present the count is what survives.
            if "num_speakers" in arguments:
                _log.info(
                    "The diarisation threshold was not sent to ElevenLabs, because "
                    "a speaker count of %d was sent instead and the service accepts "
                    "only one of the two.",
                    arguments["num_speakers"],
                )
            elif not diarise:
                _log.info(
                    "The diarisation threshold was not sent to ElevenLabs, because "
                    "this request does not ask for diarisation and the service "
                    "rejects the threshold without it."
                )
            else:
                arguments["diarization_threshold"] = threshold
        if keyterms:
            arguments["keyterms"] = list(keyterms)

        started_at = datetime.now()
        try:
            with request.audio_path.open("rb") as handle:
                response = client.speech_to_text.convert(
                    file=(
                        request.audio_path.name,
                        handle,
                        _content_type_for(request.audio_path),
                    ),
                    request_options={
                        "additional_body_parameters": body_parameters,
                        "timeout_in_seconds": int(self._timeout_seconds),
                        "max_retries": self._maximum_retries,
                    },
                    **arguments,
                )
        except Exception as error:  # noqa: BLE001 - every failure becomes a result
            # Converted here rather than allowed out, because a failed
            # request still has provenance worth keeping: what was asked, of
            # which model, and what came back instead of a transcript. An
            # exception would carry the message and lose all the rest.
            failure = _as_provider_error("ElevenLabs", error)
            _log.warning(
                "ElevenLabs did not answer: %s (worth retrying: %s)",
                failure,
                failure.retryable,
                exc_info=error,
            )
            record = self._request_record(
                request,
                arguments=arguments,
                body_parameters=body_parameters,
                keyterms=keyterms,
                language_code=language_code,
                size_bytes=size_bytes,
                started_at=started_at,
                transcription_id=None,
                language_probability=None,
                succeeded=False,
                error=str(failure),
            )
            return ProviderResult(
                provider=self.provider,
                request=record,
                error=str(failure),
                # A body that explains a refusal is often the most useful
                # thing in the whole folder afterwards, so it is kept on the
                # same footing as a successful answer.
                raw_response=getattr(error, "body", None),
            )

        tokens = self._tokens_from(response, request)
        detected = Language.from_code(_field(response, "language_code"))
        record = self._request_record(
            request,
            arguments=arguments,
            body_parameters=body_parameters,
            keyterms=keyterms,
            language_code=language_code,
            size_bytes=size_bytes,
            started_at=started_at,
            transcription_id=_field(response, "transcription_id"),
            language_probability=_field(response, "language_probability"),
            succeeded=True,
            error=None,
        )
        return ProviderResult(
            provider=self.provider,
            tokens=tokens,
            request=record,
            detected_language=detected,
            speakers=_speakers_in(tokens),
            # The whole answer, not the part of it this adapter understood.
            # It goes no further than the pipeline, which hands it to the
            # store; an adapter that knew where transcripts are kept would
            # be far harder to test and would need a folder it has no other
            # use for.
            raw_response=_raw_payload(response),
        )

    # -- Turning the answer into evidence -------------------------------

    def _tokens_from(
        self,
        response: Any,
        request: TranscriptionRequest,
    ) -> list[ProviderToken]:
        """Convert every word the service returned onto the canonical timeline."""
        tokens: list[ProviderToken] = []
        for index, word in enumerate(_field(response, "words") or []):
            text = _field(word, "text") or ""
            kind = _field(word, "type") or "word"
            start = _field(word, "start")
            end = _field(word, "end")
            tokens.append(
                ProviderToken(
                    provider=self.provider,
                    index=index,
                    text=text,
                    # This is the line the whole transcript rests on. The
                    # service answers in the local time of whatever audio it
                    # was given, and adding the offset here, once, is what
                    # puts every word on the canonical timeline. Miss it and
                    # a chunked recording produces a transcript where every
                    # word after the first chunk points at the wrong sound.
                    start=_shifted(start, request.canonical_offset),
                    end=_shifted(end, request.canonical_offset),
                    speaker=_field(word, "speaker_id"),
                    # Carried through as a log probability, not converted to
                    # a confidence. The model keeps the two apart because
                    # they are not the same thing and cannot be compared
                    # across services.
                    log_probability=_as_float(_field(word, "logprob")),
                    # Left unknown on purpose. The service knows only the
                    # language of the whole file, which the result carries.
                    language=Language.UNKNOWN,
                    chunk_index=request.chunk_index,
                    # Two flags rather than one. Alignment skips both, but a
                    # comma is a written mark and laughter is a thing that
                    # happened in the room, and only one of them is worth
                    # showing a reader.
                    is_punctuation=_is_written_mark(kind, text),
                    is_audio_event=kind == AUDIO_EVENT_TYPE,
                )
            )
        return tokens

    # -- Building the request -------------------------------------------

    def _keyterms_for(self, terms: tuple[str, ...]) -> tuple[str, ...]:
        """Cut the vocabulary down to what the service will actually accept.

        Two different cuts happen here, and they are not the same kind of
        thing. A term that breaks one of the service's own rules about a key
        term is dropped wherever it sits in the list, because sending it
        would have the whole request refused and cost the chunk its
        transcript rather than costing that one word. Running out of room is
        the other kind: the terms arrive ordered most specific and most
        confirmed first, so taking the head of the list keeps the ones most
        likely to matter and drops the ones least likely to.
        """
        kept: list[str] = []
        for term in terms:
            cleaned = (term or "").strip()
            if not cleaned:
                continue
            refusal = _why_the_service_would_refuse(cleaned)
            if refusal is not None:
                _log.info("The term %r was not sent to ElevenLabs: %s", cleaned, refusal)
                continue
            kept.append(cleaned)
        if len(kept) <= self._maximum_keyterms:
            return tuple(kept)
        _log.info(
            "Sending ElevenLabs the first %d of %d terms, to stay at or below the "
            "%d terms above which a request is billed a 20-second minimum.",
            self._maximum_keyterms,
            len(kept),
            KEYTERM_MINIMUM_CHARGE_THRESHOLD,
        )
        return tuple(kept[: self._maximum_keyterms])

    def _diarisation_for(
        self,
        request: TranscriptionRequest,
        extras: dict[str, Any],
    ) -> tuple[bool, float | None]:
        """Whether to diarise, and at what threshold, from wherever those were set.

        Both can only be set through the free-form extras at present. Nothing
        passes a threshold to this adapter's constructor, and the ``diarise``
        setting that the settings dialog writes is not read by the code that
        builds this adapter, so ``diarize`` in the extras is the only thing
        that has ever turned diarisation off for ElevenLabs. Reading them here
        keeps that working while still taking them out of the body, where the
        client library would otherwise spread them over what this adapter
        decided and produce a combination the service refuses.
        """
        diarise = request.diarise
        if "diarize" in extras:
            written = extras["diarize"]
            if isinstance(written, bool):
                diarise = written
            else:
                # Not turned into a boolean by guessing. "false" as a string
                # is somebody who meant False, but so is 0, and "no", and a
                # wrong guess here silently changes what the transcript is.
                # The parameter box is read as JSON, so a real false is
                # available to anybody who wants one.
                _log.warning(
                    "The diarize parameter was ignored, because %r is not true or "
                    "false. Diarisation stays %s.",
                    written,
                    "on" if diarise else "off",
                )

        threshold = self._diarisation_threshold
        if "diarization_threshold" in extras:
            supplied = _as_float(extras["diarization_threshold"])
            if supplied is not None:
                threshold = supplied
            else:
                _log.warning(
                    "The diarization_threshold parameter was ignored, because %r "
                    "is not a number.",
                    extras["diarization_threshold"],
                )
        return diarise, threshold

    def _body_parameters_for(self, extras: dict[str, Any]) -> dict[str, Any]:
        """The free-form extras, with the ones we never pass straight through removed.

        This matters more than it looks, because of how the client library
        merges them. The extras are handed over as
        ``additional_body_parameters`` and spread over the named arguments
        rather than under them, so an extra of the same name silently
        replaces what this adapter worked out. Anything decided here has to
        be taken out of the extras, or the decision does not hold.

        ``no_verbatim`` asks the service to tidy what it heard. Our own
        verbatim layer is the authoritative one, and a service that quietly
        removes fillers and false starts breaks the link between the words
        and the audio they came from, so it is dropped even when somebody
        has put it in the settings.

        ``num_speakers`` is dropped outright. The speaker count is worked out
        from the expected speaker count, which has a setting and a control of
        its own, so a second copy of it here could only conflict with the
        first.

        The other three are removed but not thrown away. ``diarize`` and
        ``diarization_threshold`` were read by ``_diarisation_for`` before
        this ran, and ``keyterms`` by ``_terms_for``, so that all of them go
        through this adapter's rules rather than round them. Each of those
        methods says in the log what it made of what it found, which is why
        nothing is said about them here.
        """
        merged = dict(extras)
        if "no_verbatim" in merged:
            merged.pop("no_verbatim")
            _log.warning(
                "The no_verbatim parameter was not sent to ElevenLabs. This "
                "application is verbatim by design."
            )
        if "num_speakers" in merged:
            merged.pop("num_speakers")
            _log.warning(
                "The num_speakers parameter was not sent to ElevenLabs as written. "
                "The speaker count comes from the expected speaker count, which "
                "has a setting of its own."
            )
        # Removed without a word, deliberately. What became of these three is
        # not known here: whether a list of terms was used or superseded, and
        # whether a diarisation value was read or was unusable, was settled
        # before this ran. Saying anything about them from here would mean
        # guessing, and a log that guesses wrongly is worse than one that is
        # quiet. The two methods that made those decisions report them.
        merged.pop("keyterms", None)
        merged.pop("diarize", None)
        merged.pop("diarization_threshold", None)
        return merged

    def _request_record(
        self,
        request: TranscriptionRequest,
        *,
        arguments: dict[str, Any],
        body_parameters: dict[str, Any],
        keyterms: tuple[str, ...],
        language_code: str | None,
        size_bytes: int,
        started_at: datetime,
        transcription_id: str | None,
        language_probability: Any,
        succeeded: bool,
        error: str | None,
    ) -> ProviderRequestRecord:
        """Note what was asked and what came back, without the credential.

        The response carries no model name or version of its own, so the
        model that was sent is recorded here alongside the transcription
        identifier that came back. Together they identify the request, which
        is the only way a later reprocessing run can tell a changed service
        from a changed application.
        """
        recorded = {key: value for key, value in arguments.items() if key != "keyterms"}
        recorded["extra_parameters"] = _without_credentials(body_parameters)
        recorded["keyterm_count"] = len(keyterms)
        if transcription_id:
            recorded["response_transcription_id"] = transcription_id
        probability = _as_float(language_probability)
        if probability is not None:
            # Weak, whole-file evidence, but it is one of the few language
            # signals that exists at all, so it is kept where it can be found.
            recorded["response_language_probability"] = probability

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
            provider_request_id=transcription_id,
        )
        return ProviderRequestRecord(
            provider=self.provider,
            model_identifier=self._model,
            request_parameters=_without_credentials(recorded),
            language_configuration=language_code or "auto",
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


class ElevenLabsForcedAligner(ForcedAligner):
    """Measuring where words we already know fall in audio we already have.

    This is used after the text of a span has changed, when the timing it
    inherited no longer describes what the words now say. The service is
    given the audio and the plain text and answers with a time for each
    word.

    Two things about it repeatedly catch people out.

    The first is that there is no language parameter. The text carries the
    language, so the text handed in must be plain running words, not a
    diarised transcript and not JSON.

    The second is the polarity of the number it returns per word. The
    transcription endpoint reports ``logprob``, where higher is better. This
    endpoint reports ``loss``, where **lower** is better. They look alike,
    they sit in the same position in the response, and getting them the
    wrong way round would silently mark the worst alignments as the best
    ones, which is the sort of mistake that never announces itself.
    """

    provider = Provider.ELEVENLABS

    def __init__(
        self,
        api_key: str = "",
        parameters: dict[str, Any] | None = None,
        *,
        timeout_seconds: float = 600.0,
        maximum_retries: int = 3,
        client: Any | None = None,
    ) -> None:
        self._api_key = api_key or ""
        self._parameters = dict(parameters or {})
        self._timeout_seconds = timeout_seconds
        self._maximum_retries = maximum_retries
        self._client = client

    def is_configured(self) -> bool:
        return bool(self._api_key.strip())

    def supports(self, language: Language) -> bool:
        """Whether this aligner can be trusted with a span in this language.

        Afrikaans is refused outright. It is not among the 29 languages the
        service aligns, and asking anyway would either fail or, worse,
        return times measured against a language model that has never heard
        the words. An unknown language is allowed through, because the text
        itself carries the language and the service covers far more of them
        than this application does.
        """
        if language is Language.AFRIKAANS:
            return False
        return language in ALIGNER_LANGUAGES or language is Language.UNKNOWN

    def align(
        self,
        audio_path: Path,
        text: str,
        language: Language,
        canonical_offset: float = 0.0,
    ) -> list[tuple[str, float, float]]:
        if not self.is_configured():
            raise ProviderNotConfigured(self.provider, "no API key has been entered")
        if not self.supports(language):
            raise ProviderError(
                f"ElevenLabs cannot align {language.display_name}. The span keeps "
                "the timing it had, and is marked approximate."
            )
        if not text.strip():
            raise ProviderError("There is no text to align against the audio.")

        size_bytes = _file_size(audio_path)
        if size_bytes > MAXIMUM_ALIGNMENT_FILE_BYTES:
            raise ProviderError(
                f"{audio_path.name} is too large for forced alignment, which "
                f"accepts at most {MAXIMUM_ALIGNMENT_FILE_BYTES / _GIGABYTE:.0f} GB."
            )

        if self._client is None:
            self._client = _build_client(self._api_key)

        try:
            with audio_path.open("rb") as handle:
                response = self._client.forced_alignment.create(
                    file=(audio_path.name, handle, _content_type_for(audio_path)),
                    text=text,
                    request_options={
                        "additional_body_parameters": dict(self._parameters),
                        "timeout_in_seconds": int(self._timeout_seconds),
                        "max_retries": self._maximum_retries,
                    },
                )
        except Exception as error:  # noqa: BLE001 - converted at this boundary
            raise _as_provider_error("Forced alignment", error) from error

        # Remember what this number is. It is a loss, so a small one means a
        # good alignment, which is the opposite way round to the logprob on a
        # transcription word. The caller uses it to decide between calling a
        # span forced_aligned and calling it approximate_span, and reading it
        # the wrong way round would mark the worst spans as the best ones
        # without ever failing visibly.
        overall_loss = _as_float(_field(response, "loss"))
        if overall_loss is not None:
            _log.debug(
                "Forced alignment finished with a loss of %.4f, where lower is better.",
                overall_loss,
            )

        aligned: list[tuple[str, float, float]] = []
        for word in _field(response, "words") or []:
            start = _shifted(_field(word, "start"), canonical_offset)
            end = _shifted(_field(word, "end"), canonical_offset)
            if start is None or end is None:
                # A word without a measured span is exactly what this call
                # was made to obtain, so an unmeasured one is dropped rather
                # than given a made-up position.
                continue
            aligned.append((_field(word, "text") or "", start, end))
        if not aligned:
            raise ProviderError("Forced alignment returned no timed words.")
        return aligned


# -- Shared helpers ------------------------------------------------------


def _build_client(api_key: str) -> Any:
    """Build a real client, reporting a missing package as such.

    The import lives here rather than at the top of the module so that the
    application starts on a machine where the package was never installed.
    That machine loses ElevenLabs and is told why.
    """
    try:
        from elevenlabs import ElevenLabs
    except ImportError as error:
        raise ProviderUnavailable(Provider.ELEVENLABS, PACKAGE) from error
    return ElevenLabs(api_key=api_key)


def _terms_for(
    request: TranscriptionRequest,
    settings_parameters: dict[str, Any],
) -> tuple[str, ...]:
    """The key terms to send, from the most specific place that names any.

    They can arrive by three doors, and the order between them matters. The
    context layer hands the same list in twice, once as the request's
    vocabulary and once as a ``keyterms`` parameter on the request, because
    every other service takes its terms as an ordinary parameter and this one
    takes them as an argument. Settings may also hold a list, written by
    somebody who wanted terms sent on every recording.

    The request is read first, then its vocabulary, and the settings list only
    if neither named anything. That last part is the ordering that is worth
    being deliberate about: the escalation path sends a live vocabulary
    gathered for one window and no request parameters at all, and a fixed list
    from the settings file should not be allowed to displace it. A general
    instruction is what you fall back on, not what overrules the specific work
    in front of you.

    Whichever list comes back is then held to the service's own rules about a
    key term, so a bracket in somebody's settings costs them that term rather
    than costing the recording its transcript.
    """
    candidates = (
        ("the request's parameters", request.extra_parameters.get("keyterms")),
        ("the request's vocabulary", request.vocabulary_terms),
        ("the settings", settings_parameters.get("keyterms")),
    )
    chosen: tuple[str, ...] | None = None
    chosen_from = ""
    for description, supplied in candidates:
        if supplied is not None and not isinstance(supplied, (list, tuple)):
            # Absent and empty are quiet, because neither names any terms and
            # neither is a mistake. This is a mistake: somebody meant to send
            # terms and wrote something that cannot hold any.
            _log.warning(
                "The keyterms value in %s was ignored, because %r is not a list "
                "of terms.",
                description,
                supplied,
            )
            continue
        if not supplied:
            continue
        if chosen is None:
            chosen = tuple(str(term) for term in supplied)
            chosen_from = description
            continue
        # Said out loud, because a standing list quietly losing to a live one
        # is right but surprising, and somebody who wrote that list deserves
        # to find out here rather than by wondering why it had no effect.
        _log.info(
            "The %d key terms in %s were not sent to ElevenLabs. The %d in %s "
            "are more specific to this recording.",
            len(supplied),
            description,
            len(chosen),
            chosen_from,
        )
    return chosen if chosen is not None else ()


def _why_the_service_would_refuse(term: str) -> str | None:
    """Why this key term cannot be sent, or None if it can.

    A sentence rather than a flag, because the only thing anybody does with
    the answer is write it into the log for somebody to read later, and
    "it is 63 characters long" tells them what to change about their
    vocabulary in a way that a bare rejection does not.
    """
    if len(term) > MAXIMUM_KEYTERM_CHARACTERS:
        return (
            f"it is {len(term)} characters long, and a key term must be shorter "
            f"than {KEYTERM_CHARACTER_LIMIT}."
        )
    if len(term.split()) > MAXIMUM_KEYTERM_WORDS:
        return (
            f"it has {len(term.split())} words, and a key term may have at most "
            f"{MAXIMUM_KEYTERM_WORDS}."
        )
    offending = sorted(
        {character for character in term if character in UNSUPPORTED_KEYTERM_CHARACTERS}
    )
    if offending:
        listed = " and ".join(f'"{character}"' for character in offending)
        return f"it contains {listed}, which a key term may not contain."
    return None


def _is_written_mark(kind: str, text: str) -> bool:
    """Whether this entry is punctuation or spacing rather than a spoken word.

    Both carry real timing and are worth keeping, and neither is anything a
    service could be said to have heard, so alignment steps over them.

    An audio event is deliberately not counted here. It is skipped by
    alignment too, for the same practical reason that lining laughter up
    against another service's text would match it to whatever word happened
    to be nearby, but it is skipped as a sound rather than as a mark on the
    page, and it is flagged as one. A reader may well want to see that
    somebody laughed; nobody wants to see a stray comma.
    """
    if kind == AUDIO_EVENT_TYPE:
        return False
    if kind == SPACING_TYPE:
        return True
    return _is_punctuation_only(text)


def _is_punctuation_only(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return all(unicodedata.category(character).startswith("P") for character in stripped)


def _raw_payload(response: Any) -> Any:
    """The service's whole answer, as close to untouched as it can be had.

    The specification requires every response to be kept unchanged, so that
    a reprocessing run that disagrees with the original can be traced to a
    changed service rather than a changed application. Only the original
    answer can settle that, and only if it is the original: a copy rebuilt
    field by field would quietly drop whatever this adapter does not read
    today, which is exactly the part a future question would be about.

    So the library's own conversion is used rather than any of ours. Every
    shape it might hand back is covered, and if none of them fits, the
    object goes on untouched and the store decides what to make of it.
    """
    if isinstance(response, (dict, list, str, bytes)):
        return response
    for method_name in ("model_dump", "dict"):
        method = getattr(response, method_name, None)
        if callable(method):
            try:
                return method()
            except TypeError:
                # Some versions insist on arguments we should not guess at.
                _log.debug("%s() would not convert the response.", method_name, exc_info=True)
    return response


def _shifted(value: Any, offset: float) -> float | None:
    """Move one of the service's times onto the canonical timeline."""
    number = _as_float(value)
    if number is None:
        return None
    return number + offset


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _field(source: Any, name: str, default: Any = None) -> Any:
    """Read a field from the answer, whichever shape the library gives it in.

    The library returns typed objects, the raw REST fallback returns plain
    dictionaries, and a test hands in something simpler again. All three are
    read the same way here so that none of them needs a separate code path.
    """
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _single_language_code(languages: tuple[Language, ...]) -> str | None:
    """The one language code to send, or None to let the service detect it.

    Naming a language helps when there is only one it can be. Naming one of
    several would be worse than saying nothing, because it would push the
    service away from the others.
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
    that nobody would ever notice.
    """
    cleaned: dict[str, Any] = {}
    for key, value in parameters.items():
        lowered = str(key).lower()
        if any(name in lowered for name in _CREDENTIAL_NAMES):
            continue
        cleaned[key] = _without_credentials(value) if isinstance(value, dict) else value
    return cleaned


def _as_provider_error(what: str, error: Exception) -> ProviderError:
    """Turn whatever the library threw into one sentence and a retry verdict.

    The library's exception classes are not imported, deliberately. A rate
    limit from this service has no class of its own and arrives as the base
    error class carrying status 429, so reading the status off the error is
    the only reliable test anyway, and doing it this way means the module
    never has to import the package merely to catch its errors, which would
    undo the point of importing it lazily.
    """
    status = _as_int(getattr(error, "status_code", None))
    name = type(error).__name__
    retryable = False
    if status is not None:
        retryable = status in (408, 409, 425, 429) or 500 <= status < 600
    elif any(hint in name.lower() for hint in _RETRYABLE_ERROR_NAMES):
        # No status at all means the request never arrived, which is usually
        # worth one more attempt.
        retryable = True

    detail = _detail_from(getattr(error, "body", None)) or str(error).strip() or name
    message = (
        f"{what} answered with an error ({status}): {detail}"
        if status is not None
        else f"{what} could not be reached: {detail}"
    )
    return ProviderError(message, retryable=retryable, status_code=status)


def _detail_from(body: Any) -> str | None:
    """Pull the human-readable message out of the service's error body."""
    if not isinstance(body, dict):
        return None
    detail = body.get("detail")
    if isinstance(detail, dict):
        message = detail.get("message")
        return str(message) if message else None
    return str(detail) if detail else None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
