"""The folder of files that sits beside a recording and explains its transcript.

A transcript is not one file. It is the finished words, the untouched
answer every service gave, the note of every request that was made, and the
exports a person actually reads. Keeping all of that in one folder next to
the recording means the whole thing can be copied, backed up or deleted by
moving one folder, and that a transcript never becomes separated from the
audio it only makes sense against.

The folder is named after the recording with a suffix on the end, so
``meeting.m4a`` gets ``meeting.m4a.transcript``. The full file name is used
rather than the stem because ``meeting.m4a`` and ``meeting.wav`` are
different recordings and must not share a folder.

Two rules shape everything below.

The first is that a save must never destroy a good file. Every write goes
through a temporary file that is moved into place, so an interrupted save
leaves the previous version intact, and a damaged or half-written file is
read as "there is nothing here" rather than stopping the application.

The second, and the more important one, is that **provenance is immutable**.
Once a service's answer has been written down it is never written again. The
file names carry a sequence number that only ever counts upwards, so a
second run over the same recording adds ``0043`` onwards and cannot touch
``0001`` to ``0042``, whatever it does. That is deliberate rather than
incidental: the specification requires that a later run can be explained
against an earlier one, and it can only be explained if the earlier evidence
is still there. A naming scheme that reused ``elevenlabs.json`` would lose
that evidence quietly, and nobody would notice until the day they needed it.

The names are also meant to be read. Somebody looking in the folder should
be able to tell which service answered and what was asked of it without
opening anything, which matters most for escalation, where one recording can
produce dozens of small windows.

Nothing here writes an API key. The settings recorded in a transcript arrive
already redacted from the settings layer, but they are checked again on the
way out, because a key that reaches the disk cannot be called back.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from vox_verbatim.json_store import (
    keep_unsaved_copy,
    read_json_object,
    replace_with_retries,
    unsaved_copy_path,
    write_json_object,
)
from vox_verbatim.transcription.model import (
    TRANSCRIPT_FORMAT_VERSION,
    AlignmentStatus,
    AudioSpan,
    Candidate,
    CanonicalAudio,
    ChunkRecord,
    CleanSegment,
    Confidence,
    FinalToken,
    Language,
    LanguageEvidence,
    Provider,
    ProviderRequestRecord,
    ProviderResult,
    ProviderToken,
    RecordingConfiguration,
    ReviewReason,
    ReviewStatus,
    RiskCategory,
    Speaker,
    StatisticsNote,
    TimingStatus,
    TokenReference,
    Transcript,
)

_log = logging.getLogger(__name__)

#: What is added to the recording's file name to make its folder. The running
#: application passes the user's own ``transcript_folder_suffix`` from
#: Settings; this is a parameter rather than an import so that the store can
#: be used and tested without a settings file behind it.
DEFAULT_FOLDER_SUFFIX = ".transcript"

TRANSCRIPT_FILE_NAME = "transcript.json"

#: Where the untouched provider answers live, and where the readable
#: exports go. Two folders rather than one, because the answers are evidence
#: nobody edits and the exports are documents people open, and mixing them
#: invites somebody to tidy up the wrong one.
RAW_RESPONSES_FOLDER = "raw-responses"
EXPORTS_FOLDER = "exports"

#: What replaces anything that looks like a credential.
REDACTED_TEXT = "[redacted]"

_SEQUENCE_PATTERN = re.compile(r"^(\d{4,})-")

#: The longest path Windows accepts unless long paths have been switched on
#: for the machine, which they usually have not. A recording that sits deep
#: in a folder tree with a long name can put its transcript folder past it.
WINDOWS_PATH_LIMIT = 260


# -- Keeping secrets off the disk ----------------------------------------

#: Words that make a setting's name a credential rather than a preference.
#: Whole words, so ``max_tokens`` is left alone while ``access_token`` is
#: not. Over-redacting a harmless setting costs a little explanation later;
#: under-redacting one costs the user their key, so the list leans towards
#: redacting.
_SECRET_WORDS = frozenset(
    {
        "key",
        "keys",
        "secret",
        "secrets",
        "token",
        "password",
        "passwd",
        "passphrase",
        "credential",
        "credentials",
        "authorization",
        "authorisation",
        "auth",
        "bearer",
        "signature",
    }
)

#: Settings whose names trip the rule above but which are plainly not
#: secrets. Vocabulary terms are the ones that matter: they are exactly what
#: a person wants to see when they ask why a name came out wrong.
_NOT_SECRET_NAMES = frozenset(
    {
        "key_terms",
        "keyterms",
        "key_term",
        "keyword",
        "keywords",
        "key_words",
        "key_phrases",
        "keyphrases",
    }
)

#: Shapes that are a credential whatever they are filed under. The first two
#: are how the services themselves write their keys; the last two are what an
#: opaque key looks like when it has no prefix at all. Model names, paths and
#: language codes all contain full stops or spaces, so they do not match.
_SECRET_VALUE_PREFIXES = ("sk-", "sk_", "xi-", "ghp_", "bearer ", "basic ")
_HEX_KEY = re.compile(r"^[0-9a-fA-F]{32,}$")
_OPAQUE_KEY = re.compile(r"^[A-Za-z0-9]{40,}$")


def _name_is_secret(name: str) -> bool:
    lowered = name.strip().lower()
    if lowered in _NOT_SECRET_NAMES:
        return False
    words = [word for word in re.split(r"[^a-z0-9]+", lowered) if word]
    return any(word in _SECRET_WORDS for word in words)


def _value_is_secret(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    lowered = text.lower()
    if any(lowered.startswith(prefix) for prefix in _SECRET_VALUE_PREFIXES):
        return True
    return bool(_HEX_KEY.match(text) or _OPAQUE_KEY.match(text))


def _redacted(name: str | None, value: Any) -> Any:
    """Return ``value`` with anything that looks like a credential removed."""
    if isinstance(value, dict):
        return {key: _redacted(key if isinstance(key, str) else None, item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redacted(name, item) for item in value]
    if name is not None and _name_is_secret(name):
        return REDACTED_TEXT
    if _value_is_secret(value):
        return REDACTED_TEXT
    return value


def redact_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``settings`` with every credential taken out.

    The settings layer is supposed to have done this already, and normally
    has. This is the second lock on the door. A key is caught either by the
    name it is filed under or by the shape of the value itself, so a key
    smuggled in under an innocent name is still stopped.
    """
    if not isinstance(settings, dict):
        return {}
    return {
        key: _redacted(key if isinstance(key, str) else None, value)
        for key, value in settings.items()
    }


# -- Reading loose JSON carefully ----------------------------------------
#
# Everything below treats the file as something a person may have edited by
# hand or a disk may have half-written. A field of the wrong type falls back
# to its default rather than raising, in the same way the session and the
# settings do, because a transcript that comes back slightly thin is far
# more use than one that will not open at all.


def _text_of(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value != "" else None


def _flag(value: Any, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _integer(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _enum_of(kind: Any, value: Any, default: Any) -> Any:
    """Turn a stored value back into one of ``kind``, or fall back.

    An unrecognised value is a transcript written by a newer version of the
    application, or a hand edit. Either way the sensible answer is the
    default for that field and a line in the log, not a refusal to open the
    file.
    """
    if isinstance(value, str):
        try:
            return kind(value)
        except ValueError:
            _log.warning("%r is not a known %s; using %r.", value, kind.__name__, default)
    return default


def _optional_enum(kind: Any, value: Any) -> Any:
    if value is None:
        return None
    return _enum_of(kind, value, None)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _enum_list(kind: Any, value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    found = [_enum_of(kind, item, None) for item in value]
    return [item for item in found if item is not None]


def _number_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        key: float(item)
        for key, item in value.items()
        if isinstance(key, str) and not isinstance(item, bool) and isinstance(item, (int, float))
    }


# -- Turning the model into JSON and back --------------------------------


def _span_to_dict(span: AudioSpan | None) -> dict[str, Any] | None:
    if span is None:
        return None
    return {"start": span.start, "end": span.end}


def _span_from_dict(data: Any) -> AudioSpan | None:
    if not isinstance(data, dict):
        return None
    start = _optional_number(data.get("start"))
    end = _optional_number(data.get("end"))
    if start is None or end is None:
        return None
    # A span that ends before it starts is refused by the model itself. Here
    # it means the file is damaged, and no span at all is the honest answer.
    if end < start:
        _log.warning("Ignoring an audio span that ends before it starts: %s to %s.", start, end)
        return None
    return AudioSpan(start, end)


def _reference_to_dict(reference: TokenReference) -> dict[str, Any]:
    return {"provider": reference.provider.value, "index": reference.index}


def _reference_from_dict(data: Any) -> TokenReference | None:
    if not isinstance(data, dict):
        return None
    provider = _optional_enum(Provider, data.get("provider"))
    if provider is None:
        return None
    return TokenReference(provider=provider, index=_integer(data.get("index")))


def _references_from(value: Any) -> list[TokenReference]:
    if not isinstance(value, list):
        return []
    found = [_reference_from_dict(item) for item in value]
    return [item for item in found if item is not None]


def _candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    return {
        "text": candidate.text,
        "providers": [provider.value for provider in candidate.providers],
        "score": candidate.score,
        "components": dict(candidate.components),
        "source_tokens": [_reference_to_dict(item) for item in candidate.source_tokens],
        "in_vocabulary": candidate.in_vocabulary,
    }


def _candidate_from_dict(data: Any) -> Candidate | None:
    if not isinstance(data, dict):
        return None
    return Candidate(
        text=_text_of(data.get("text")),
        providers=tuple(_enum_list(Provider, data.get("providers"))),
        score=_number(data.get("score")),
        components=_number_map(data.get("components")),
        source_tokens=tuple(_references_from(data.get("source_tokens"))),
        in_vocabulary=_flag(data.get("in_vocabulary")),
    )


def _language_evidence_to_dict(evidence: LanguageEvidence) -> dict[str, float]:
    return {language.value: score for language, score in evidence.scores.items()}


def _language_evidence_from_dict(data: Any) -> LanguageEvidence:
    scores: dict[Language, float] = {}
    for code, score in _number_map(data).items():
        language = _optional_enum(Language, code)
        if language is not None:
            scores[language] = score
    return LanguageEvidence(scores=scores)


def _strength_from(value: Any) -> float | None:
    """Read a saved composite confidence back, or say there is not one.

    Three different things all come back as ``None`` here, and the model
    reads that one answer as "this was never worked out": the key is absent,
    because the transcript predates the field; the value is not a number,
    because the file was damaged or edited by hand; or the value is a number
    outside nought to one, which is the same damage wearing a better
    disguise.

    The range check is the part worth explaining. Clamping a stray 42 to one
    would turn a broken file into a word that looks thoroughly settled and
    that nothing would ever put in front of a person again. Reading it as
    unknown leaves the word outside every threshold instead of hiding it
    inside one, which is the failure that can be noticed and corrected.
    """
    strength = _optional_number(value)
    if strength is None or not 0.0 <= strength <= 1.0:
        return None
    return strength


def _final_token_to_dict(token: FinalToken) -> dict[str, Any]:
    return {
        "id": token.id,
        "text": token.text,
        "normalised_text": token.normalised_text,
        "text_source": token.text_source.value if token.text_source else None,
        "text_confidence": token.text_confidence.value,
        "confidence_strength": token.confidence_strength,
        "start": token.start,
        "end": token.end,
        "timing_source": token.timing_source.value if token.timing_source else None,
        "timing_status": token.timing_status.value,
        "timing_confidence": token.timing_confidence.value,
        "speaker": token.speaker,
        "speaker_source": token.speaker_source.value if token.speaker_source else None,
        "speaker_confidence": token.speaker_confidence.value,
        "speaker_alternative": token.speaker_alternative,
        "language": token.language.value,
        "language_evidence": _language_evidence_to_dict(token.language_evidence),
        "source_tokens": [_reference_to_dict(item) for item in token.source_tokens],
        "source_audio_span": _span_to_dict(token.source_audio_span),
        "alignment_status": token.alignment_status.value,
        "candidates": [_candidate_to_dict(item) for item in token.candidates],
        "rejected_tokens": [_reference_to_dict(item) for item in token.rejected_tokens],
        "risk_categories": [item.value for item in token.risk_categories],
        "review_status": token.review_status.value,
        "review_reasons": [item.value for item in token.review_reasons],
        "llm_decision": token.llm_decision,
        "human_corrected": token.human_corrected,
        "text_corrected": token.text_corrected,
        "original_text": token.original_text,
        "statistics_note": _statistics_note_to_dict(token.statistics_note),
    }


def _final_token_from_dict(data: Any) -> FinalToken | None:
    if not isinstance(data, dict):
        return None
    token = FinalToken()
    stored_id = _optional_text(data.get("id"))
    if stored_id is not None:
        token.id = stored_id
    token.text = _text_of(data.get("text"))
    token.normalised_text = _text_of(data.get("normalised_text"))
    token.text_source = _optional_enum(Provider, data.get("text_source"))
    token.text_confidence = _enum_of(
        Confidence, data.get("text_confidence"), Confidence.UNRESOLVED
    )
    token.confidence_strength = _strength_from(data.get("confidence_strength"))
    token.start = _optional_number(data.get("start"))
    token.end = _optional_number(data.get("end"))
    token.timing_source = _optional_enum(Provider, data.get("timing_source"))
    token.timing_status = _enum_of(
        TimingStatus, data.get("timing_status"), TimingStatus.UNALIGNED
    )
    token.timing_confidence = _enum_of(
        Confidence, data.get("timing_confidence"), Confidence.UNRESOLVED
    )
    token.speaker = _optional_text(data.get("speaker"))
    token.speaker_source = _optional_enum(Provider, data.get("speaker_source"))
    token.speaker_confidence = _enum_of(
        Confidence, data.get("speaker_confidence"), Confidence.UNRESOLVED
    )
    # Absent from files written before it existed, which then load with no
    # alternative speaker; the review window falls back to the neighbours.
    token.speaker_alternative = _optional_text(data.get("speaker_alternative"))
    token.language = _enum_of(Language, data.get("language"), Language.UNKNOWN)
    token.language_evidence = _language_evidence_from_dict(data.get("language_evidence"))
    token.source_tokens = _references_from(data.get("source_tokens"))
    token.source_audio_span = _span_from_dict(data.get("source_audio_span"))
    token.alignment_status = _enum_of(
        AlignmentStatus, data.get("alignment_status"), AlignmentStatus.UNALIGNED
    )
    candidates = data.get("candidates")
    if isinstance(candidates, list):
        found = [_candidate_from_dict(item) for item in candidates]
        token.candidates = [item for item in found if item is not None]
    token.rejected_tokens = _references_from(data.get("rejected_tokens"))
    token.risk_categories = _enum_list(RiskCategory, data.get("risk_categories"))
    token.review_status = _enum_of(ReviewStatus, data.get("review_status"), ReviewStatus.SETTLED)
    token.review_reasons = _enum_list(ReviewReason, data.get("review_reasons"))
    token.llm_decision = _optional_text(data.get("llm_decision"))
    token.human_corrected = _flag(data.get("human_corrected"))
    token.original_text = _optional_text(data.get("original_text"))
    token.text_corrected = _text_corrected_from(data, token)
    token.statistics_note = _statistics_note_from_dict(data.get("statistics_note"))
    return token


def _statistics_note_to_dict(note: StatisticsNote | None) -> dict[str, Any] | None:
    if note is None:
        return None
    return {
        "provider": note.provider.value if note.provider else None,
        "language": note.language.value,
        "is_proper_noun": note.is_proper_noun,
        "is_numeric": note.is_numeric,
        "vocabulary_category": note.vocabulary_category,
        "confidence": note.confidence.value,
        "service_text": note.service_text,
        "rejected_providers": [provider.value for provider in note.rejected_providers],
        "speaker_provider": note.speaker_provider.value if note.speaker_provider else None,
        "service_speaker": note.service_speaker,
        "counted": note.counted,
        "text_corrected": note.text_corrected,
        "speaker_corrected": note.speaker_corrected,
    }


def _statistics_note_from_dict(data: Any) -> StatisticsNote | None:
    """Rebuild a word's statistics note, or return ``None`` if there is none.

    A damaged note is read as no note. The word is then left uncounted
    rather than counted twice, which is the cheaper of the two mistakes.
    """
    if not isinstance(data, dict):
        return None
    return StatisticsNote(
        provider=_optional_enum(Provider, data.get("provider")),
        language=_enum_of(Language, data.get("language"), Language.UNKNOWN),
        is_proper_noun=_flag(data.get("is_proper_noun")),
        is_numeric=_flag(data.get("is_numeric")),
        vocabulary_category=_optional_text(data.get("vocabulary_category")),
        confidence=_enum_of(Confidence, data.get("confidence"), Confidence.UNRESOLVED),
        service_text=_text_of(data.get("service_text")),
        rejected_providers=tuple(_enum_list(Provider, data.get("rejected_providers"))),
        speaker_provider=_optional_enum(Provider, data.get("speaker_provider")),
        service_speaker=_optional_text(data.get("service_speaker")),
        counted=_flag(data.get("counted")),
        text_corrected=_flag(data.get("text_corrected")),
        speaker_corrected=_flag(data.get("speaker_corrected")),
    )


def _text_corrected_from(data: dict[str, Any], token: FinalToken) -> bool:
    """Whether this word's text was replaced, including in a file written before
    the flag existed.

    A missing ``text_corrected`` must not simply read as ``False``. Every
    transcript written before this feature says nothing about the field, and
    reading the silence as "never rewritten" would strip a word somebody
    corrected by hand of the one thing that protects it: a replacement rule
    refuses to touch a corrected word, so an old transcript would have its
    corrections quietly overwritten the first time the folder was analysed.

    The answer for those files is not a guess. ``original_text`` is written
    only where the text was actually replaced, so a word that has one was
    rewritten and a word that has none was not. That is the same fact the flag
    records, read off the evidence that was already being saved.
    """
    stored = data.get("text_corrected")
    if isinstance(stored, bool):
        return stored
    return token.original_text is not None


def _provider_token_to_dict(token: ProviderToken) -> dict[str, Any]:
    return {
        "provider": token.provider.value,
        "index": token.index,
        "text": token.text,
        "start": token.start,
        "end": token.end,
        "speaker": token.speaker,
        "log_probability": token.log_probability,
        "confidence": token.confidence,
        "language": token.language.value,
        "chunk_index": token.chunk_index,
        "is_punctuation": token.is_punctuation,
        "is_audio_event": token.is_audio_event,
    }


def _provider_token_from_dict(data: Any, fallback: Provider) -> ProviderToken | None:
    if not isinstance(data, dict):
        return None
    return ProviderToken(
        provider=_optional_enum(Provider, data.get("provider")) or fallback,
        index=_integer(data.get("index")),
        text=_text_of(data.get("text")),
        start=_optional_number(data.get("start")),
        end=_optional_number(data.get("end")),
        speaker=_optional_text(data.get("speaker")),
        log_probability=_optional_number(data.get("log_probability")),
        confidence=_optional_number(data.get("confidence")),
        language=_enum_of(Language, data.get("language"), Language.UNKNOWN),
        chunk_index=_integer(data.get("chunk_index")),
        is_punctuation=_flag(data.get("is_punctuation")),
        is_audio_event=_flag(data.get("is_audio_event")),
    )


def _chunk_to_dict(chunk: ChunkRecord) -> dict[str, Any]:
    return {
        "provider": chunk.provider.value,
        "chunk_index": chunk.chunk_index,
        "canonical_start": chunk.canonical_start,
        "canonical_end": chunk.canonical_end,
        "canonical_offset": chunk.canonical_offset,
        "encoded_size_bytes": chunk.encoded_size_bytes,
        "path": chunk.path,
        "provider_request_id": chunk.provider_request_id,
        "overlap_before": chunk.overlap_before,
    }


def _chunk_from_dict(data: Any, fallback: Provider) -> ChunkRecord | None:
    if not isinstance(data, dict):
        return None
    return ChunkRecord(
        provider=_optional_enum(Provider, data.get("provider")) or fallback,
        chunk_index=_integer(data.get("chunk_index")),
        canonical_start=_number(data.get("canonical_start")),
        canonical_end=_number(data.get("canonical_end")),
        canonical_offset=_number(data.get("canonical_offset")),
        encoded_size_bytes=_integer(data.get("encoded_size_bytes")),
        path=_optional_text(data.get("path")),
        provider_request_id=_optional_text(data.get("provider_request_id")),
        overlap_before=_number(data.get("overlap_before")),
    )


def _request_to_dict(request: ProviderRequestRecord) -> dict[str, Any]:
    return {
        "provider": request.provider.value,
        "model_identifier": request.model_identifier,
        "model_version": request.model_version,
        # Adapters are meant to strip credentials out of these before the
        # record is built. Checking again here costs nothing and is the
        # difference between a mistake upstream being embarrassing and being
        # unrecoverable.
        "request_parameters": redact_settings(request.request_parameters),
        "language_configuration": request.language_configuration,
        "vocabulary_terms": list(request.vocabulary_terms),
        "chunks": [_chunk_to_dict(chunk) for chunk in request.chunks],
        "raw_response_file": request.raw_response_file,
        "processing_seconds": request.processing_seconds,
        "started_at": request.started_at,
        "succeeded": request.succeeded,
        "error": request.error,
        "purpose": request.purpose,
    }


def _request_from_dict(data: Any) -> ProviderRequestRecord | None:
    if not isinstance(data, dict):
        return None
    provider = _optional_enum(Provider, data.get("provider"))
    if provider is None:
        return None
    chunks = data.get("chunks")
    found = (
        [_chunk_from_dict(item, provider) for item in chunks] if isinstance(chunks, list) else []
    )
    parameters = data.get("request_parameters")
    return ProviderRequestRecord(
        provider=provider,
        model_identifier=_text_of(data.get("model_identifier")),
        model_version=_optional_text(data.get("model_version")),
        request_parameters=parameters if isinstance(parameters, dict) else {},
        language_configuration=_optional_text(data.get("language_configuration")),
        vocabulary_terms=tuple(_string_list(data.get("vocabulary_terms"))),
        chunks=tuple(chunk for chunk in found if chunk is not None),
        raw_response_file=_optional_text(data.get("raw_response_file")),
        processing_seconds=_number(data.get("processing_seconds")),
        started_at=_text_of(data.get("started_at")),
        succeeded=_flag(data.get("succeeded"), True),
        error=_optional_text(data.get("error")),
        purpose=_text_of(data.get("purpose"), "transcription"),
    )


def _provider_result_to_dict(result: ProviderResult) -> dict[str, Any]:
    return {
        "provider": result.provider.value,
        "tokens": [_provider_token_to_dict(token) for token in result.tokens],
        "request": _request_to_dict(result.request) if result.request else None,
        "detected_language": result.detected_language.value,
        "speakers": list(result.speakers),
        "error": result.error,
    }


def _provider_result_from_dict(data: Any, fallback: Provider) -> ProviderResult | None:
    if not isinstance(data, dict):
        return None
    provider = _optional_enum(Provider, data.get("provider")) or fallback
    tokens = data.get("tokens")
    found = (
        [_provider_token_from_dict(item, provider) for item in tokens]
        if isinstance(tokens, list)
        else []
    )
    return ProviderResult(
        provider=provider,
        tokens=[token for token in found if token is not None],
        request=_request_from_dict(data.get("request")),
        detected_language=_enum_of(Language, data.get("detected_language"), Language.UNKNOWN),
        speakers=_string_list(data.get("speakers")),
        error=_optional_text(data.get("error")),
    )


def _canonical_audio_to_dict(audio: CanonicalAudio) -> dict[str, Any]:
    return {
        "path": audio.path,
        "original_path": audio.original_path,
        "duration": audio.duration,
        "sample_rate": audio.sample_rate,
        "channels": audio.channels,
        "size_bytes": audio.size_bytes,
        "container": audio.container,
        "is_copy": audio.is_copy,
    }


def _canonical_audio_from_dict(data: Any) -> CanonicalAudio | None:
    if not isinstance(data, dict):
        return None
    return CanonicalAudio(
        path=_text_of(data.get("path")),
        original_path=_text_of(data.get("original_path")),
        duration=_number(data.get("duration")),
        sample_rate=_integer(data.get("sample_rate")),
        channels=_integer(data.get("channels")),
        size_bytes=_integer(data.get("size_bytes")),
        container=_text_of(data.get("container")),
        is_copy=_flag(data.get("is_copy")),
    )


def _configuration_to_dict(configuration: RecordingConfiguration) -> dict[str, Any]:
    return {
        "afrikaans_enabled": configuration.afrikaans_enabled,
        "expected_speaker_count": configuration.expected_speaker_count,
        "known_speakers": list(configuration.known_speakers),
        "recording_context": configuration.recording_context,
        "vocabulary_profile_ids": list(configuration.vocabulary_profile_ids),
    }


def _configuration_from_dict(data: Any) -> RecordingConfiguration:
    if not isinstance(data, dict):
        return RecordingConfiguration()
    return RecordingConfiguration(
        afrikaans_enabled=_flag(data.get("afrikaans_enabled")),
        expected_speaker_count=_integer(data.get("expected_speaker_count"), 1),
        known_speakers=_string_list(data.get("known_speakers")),
        recording_context=_text_of(data.get("recording_context")),
        vocabulary_profile_ids=_string_list(data.get("vocabulary_profile_ids")),
    )


def _clean_segment_to_dict(segment: CleanSegment) -> dict[str, Any]:
    return {
        "text": segment.text,
        "source_token_ids": list(segment.source_token_ids),
        "speaker": segment.speaker,
        "start": segment.start,
        "end": segment.end,
    }


def _clean_segment_from_dict(data: Any) -> CleanSegment | None:
    if not isinstance(data, dict):
        return None
    return CleanSegment(
        text=_text_of(data.get("text")),
        source_token_ids=_string_list(data.get("source_token_ids")),
        speaker=_optional_text(data.get("speaker")),
        start=_optional_number(data.get("start")),
        end=_optional_number(data.get("end")),
    )


def _speaker_to_dict(speaker: Speaker) -> dict[str, Any]:
    return {"id": speaker.id, "name": speaker.name}


def _speaker_from_dict(data: Any) -> Speaker | None:
    if not isinstance(data, dict):
        return None
    identifier = _optional_text(data.get("id"))
    if identifier is None:
        return None
    return Speaker(id=identifier, name=_optional_text(data.get("name")))


def transcript_to_dict(transcript: Transcript) -> dict[str, Any]:
    """Return the whole transcript as plain JSON-safe data.

    Enumerations are written by their value rather than their name, because
    the values are what settings, file names and provenance already use, and
    a name that is later spelled differently in the code should not change
    what is on the disk.
    """
    return {
        "version": TRANSCRIPT_FORMAT_VERSION,
        "recording_name": transcript.recording_name,
        "canonical_audio": (
            _canonical_audio_to_dict(transcript.canonical_audio)
            if transcript.canonical_audio
            else None
        ),
        "configuration": _configuration_to_dict(transcript.configuration),
        "tokens": [_final_token_to_dict(token) for token in transcript.tokens],
        "clean_segments": [_clean_segment_to_dict(item) for item in transcript.clean_segments],
        "speakers": [_speaker_to_dict(item) for item in transcript.speakers],
        "provider_results": {
            provider.value: _provider_result_to_dict(result)
            for provider, result in transcript.provider_results.items()
        },
        "requests": [_request_to_dict(item) for item in transcript.requests],
        "reconciliation_version": transcript.reconciliation_version,
        "created_at": transcript.created_at,
        "completed_at": transcript.completed_at,
        "effective_settings": redact_settings(transcript.effective_settings),
        "warnings": list(transcript.warnings),
    }


def transcript_from_dict(data: dict[str, Any]) -> Transcript | None:
    """Rebuild a transcript from stored data, or return ``None`` if it cannot.

    Only one thing is required: the name of the recording. Without it there
    is nothing a person could be shown, so the file is treated as damaged.
    Everything else degrades to its default, one field at a time.

    Every field is looked up by name and falls back to the model's own
    default when it is not there. That is what lets a transcript written by
    an older version of the application stay readable as the model grows a
    field, and it has to keep working: provenance is immutable, so an old
    file is evidence that will never be rewritten to match a new shape.
    """
    if not isinstance(data, dict):
        return None
    recording_name = _optional_text(data.get("recording_name"))
    if recording_name is None:
        _log.warning("A transcript file with no recording name was ignored.")
        return None

    version = _integer(data.get("version"), TRANSCRIPT_FORMAT_VERSION)
    if version > TRANSCRIPT_FORMAT_VERSION:
        # Reading it anyway is the right call: the fields this version knows
        # about are still there, and refusing would leave the user with a
        # transcript they cannot open at all.
        _log.warning(
            "%s was written by a newer version of the application (format %s); "
            "reading what can be understood.",
            recording_name,
            version,
        )

    transcript = Transcript(recording_name=recording_name)
    transcript.canonical_audio = _canonical_audio_from_dict(data.get("canonical_audio"))
    transcript.configuration = _configuration_from_dict(data.get("configuration"))

    tokens = data.get("tokens")
    if isinstance(tokens, list):
        found_tokens = [_final_token_from_dict(item) for item in tokens]
        transcript.tokens = [token for token in found_tokens if token is not None]

    segments = data.get("clean_segments")
    if isinstance(segments, list):
        found_segments = [_clean_segment_from_dict(item) for item in segments]
        transcript.clean_segments = [item for item in found_segments if item is not None]

    speakers = data.get("speakers")
    if isinstance(speakers, list):
        found_speakers = [_speaker_from_dict(item) for item in speakers]
        transcript.speakers = [item for item in found_speakers if item is not None]

    results = data.get("provider_results")
    if isinstance(results, dict):
        for code, stored in results.items():
            provider = _optional_enum(Provider, code)
            if provider is None:
                continue
            result = _provider_result_from_dict(stored, provider)
            if result is not None:
                transcript.provider_results[provider] = result

    requests = data.get("requests")
    if isinstance(requests, list):
        found_requests = [_request_from_dict(item) for item in requests]
        transcript.requests = [item for item in found_requests if item is not None]

    transcript.reconciliation_version = _integer(
        data.get("reconciliation_version"), transcript.reconciliation_version
    )
    transcript.created_at = _text_of(data.get("created_at"))
    transcript.completed_at = _text_of(data.get("completed_at"))
    settings = data.get("effective_settings")
    transcript.effective_settings = settings if isinstance(settings, dict) else {}
    transcript.warnings = _string_list(data.get("warnings"))
    return transcript


# -- Naming the raw responses --------------------------------------------


def _clock_label(seconds: float) -> str:
    """Return a time for a file name, such as ``3m03s`` or ``1h04m22s``.

    :func:`~vox_verbatim.formatting.format_duration` writes the same
    thing with colons in it, which Windows will not accept in a file name,
    so this is the only place in the application that spells a time out this
    way.
    """
    total = int(max(0.0, seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


def _safe_part(text: str) -> str:
    """Return ``text`` reduced to something safe in a file name."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower()).strip("-")
    return cleaned or "unknown"


class TranscriptStore:
    """The transcript folder for one recording, and everything in it.

    Create one with the path of the recording. Nothing is written until it
    is asked for, so making a store for a recording that has never been
    transcribed costs nothing and creates no folders.
    """

    def __init__(
        self,
        recording_path: Path | str,
        folder_suffix: str = DEFAULT_FOLDER_SUFFIX,
    ) -> None:
        self._recording_path = Path(recording_path)
        self._folder = transcript_folder_for(self._recording_path, folder_suffix)

    @property
    def recording_path(self) -> Path:
        return self._recording_path

    @property
    def folder(self) -> Path:
        return self._folder

    @property
    def transcript_path(self) -> Path:
        return self._folder / TRANSCRIPT_FILE_NAME

    @property
    def raw_responses_folder(self) -> Path:
        return self._folder / RAW_RESPONSES_FOLDER

    @property
    def exports_folder(self) -> Path:
        return self._folder / EXPORTS_FOLDER

    @property
    def unsaved_transcript_path(self) -> Path:
        """Where a transcript that could not be moved into place is left.

        A save fails on Windows when something else holds the old file open,
        and the finished transcript is then kept here rather than deleted,
        because it took an hour of paid services to make. Whoever reports the
        failure should name this path, so the person knows the words are not
        gone and where to find them.
        """
        return unsaved_copy_path(self.transcript_path)

    @property
    def has_transcript(self) -> bool:
        return self.transcript_path.is_file()

    def probe_writable(self) -> str | None:
        """Check that the transcript folder can be made and written to.

        Returns a sentence saying what is wrong, or ``None`` when a tiny file
        could be created in the folder and removed again. This is asked
        before a run starts, because the folder is where an hour of paid
        answers ends up, and the useful moment to learn that it is on a
        read-only drive, or that its path is longer than Windows allows, is
        before the money has been spent rather than after.

        Nothing here may raise. The folder is created if it does not exist,
        which the run would do anyway, and the probe file is removed whether
        or not the write worked.
        """
        folder = self._folder
        probe = folder / ".write-probe.tmp"
        too_long = len(str(probe)) > WINDOWS_PATH_LIMIT
        try:
            folder.mkdir(parents=True, exist_ok=True)
            with open(probe, "wb") as handle:
                handle.write(b"probe")
        except OSError as error:
            reason = str(error).strip() or type(error).__name__
            if too_long:
                return (
                    f"The transcript folder for {self._recording_path.name} cannot be "
                    f"written to, and its path is {len(str(probe))} characters long, "
                    f"which is more than the {WINDOWS_PATH_LIMIT} Windows allows. Move "
                    f"the recording to a shorter path, or shorten its name. ({reason})"
                )
            return (
                f"The transcript folder for {self._recording_path.name} cannot be "
                f"written to: {reason}. Check that the folder {folder.parent} is not "
                "read-only and that the drive is connected."
            )
        finally:
            try:
                probe.unlink()
            except OSError:
                pass
        return None

    # -- The transcript itself

    def load(self) -> Transcript | None:
        """Return the saved transcript, or ``None`` if there is nothing usable.

        A missing file, an unreadable one, one that is not JSON and one that
        is damaged all come back the same way, with a line in the log. The
        caller decides what to do about it; nothing here raises.
        """
        data = read_json_object(self.transcript_path)
        if data is None:
            return None
        try:
            return transcript_from_dict(data)
        except Exception:  # a hand-edited file must not take the application down
            _log.warning("%s could not be understood; ignoring it.", self.transcript_path,
                         exc_info=True)
            return None

    def save(self, transcript: Transcript) -> bool:
        """Write the transcript, returning whether it worked.

        The write goes through a temporary file, so a save interrupted by a
        power cut or a crash leaves the previous transcript where it was
        rather than a truncated file in its place.
        """
        return write_json_object(self.transcript_path, transcript_to_dict(transcript))

    # -- The untouched provider answers

    def write_raw_response(
        self,
        provider: Provider,
        payload: str | bytes | dict[str, Any],
        *,
        purpose: str = "transcription",
        chunk_index: int | None = None,
        window: AudioSpan | None = None,
        extension: str = "json",
    ) -> str | None:
        """Write one service's answer exactly as it arrived, and name it.

        Returns the file name to record in the request's provenance, or
        ``None`` if it could not be written.

        The name carries a sequence number, the service, what the request was
        for, and either the chunk number or the window of audio it covered,
        so that a person looking at the folder can see what happened without
        opening anything. The sequence number is what makes this safe to call
        again: it is one more than the highest already there, so a second run
        over the same recording writes new files beside the old ones and can
        never replace them.

        Nothing here may raise. This runs at the very end of a pass, after
        every service has answered and been paid for, and the caller treats a
        failure as a warning against a transcript that is otherwise good. An
        exception escaping instead would throw that transcript away, which is
        the opposite of what keeping a copy of the answer is for. A value the
        adapter left in a form that will not serialise is written through
        ``str`` rather than dropped, because a slightly flattened record still
        explains a transcript and no record at all does not.
        """
        folder = self.raw_responses_folder
        name = self._next_raw_name(provider, purpose, chunk_index, window, extension)
        body = _body_of(payload, provider)
        if body is None or not _write_bytes(folder / name, body):
            return None
        return name

    def raw_response_names(self) -> list[str]:
        """Every raw response written so far, oldest first."""
        folder = self.raw_responses_folder
        if not folder.is_dir():
            return []
        return sorted(item.name for item in folder.iterdir() if item.is_file())

    def raw_response_path(self, name: str) -> Path:
        return self.raw_responses_folder / name

    def read_raw_response(self, name: str) -> str | None:
        """Return the text of one stored answer, or ``None`` if it is not there."""
        try:
            return self.raw_response_path(name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            _log.warning("Could not read the stored response %s.", name, exc_info=True)
            return None

    def _next_raw_name(
        self,
        provider: Provider,
        purpose: str,
        chunk_index: int | None,
        window: AudioSpan | None,
        extension: str,
    ) -> str:
        parts = [f"{self._next_sequence():04d}", provider.value, _safe_part(purpose)]
        if chunk_index is not None:
            parts.append(f"chunk{chunk_index:02d}")
        if window is not None:
            parts.append(f"{_clock_label(window.start)}-{_clock_label(window.end)}")
        suffix = extension.lstrip(".")
        return "-".join(parts) + (f".{suffix}" if suffix else "")

    def _next_sequence(self) -> int:
        """One more than the highest sequence number already in the folder.

        Counting the files would be wrong, because a file removed by hand
        would make the count go backwards and the next write would land on a
        name that is still in use.
        """
        highest = 0
        for name in self.raw_response_names():
            match = _SEQUENCE_PATTERN.match(name)
            if match:
                highest = max(highest, int(match.group(1)))
        return highest + 1

    # -- The readable exports

    def write_export(
        self, name: str, text: str, patient: bool = True, rebuildable: bool = False
    ) -> Path | None:
        """Write one export, replacing any previous copy, and return its path.

        ``patient`` waits and tries again while another program holds the old
        file open. The review window passes False: it writes after every
        correction on the thread the person is working on, and several
        seconds of waiting there would freeze the window and the screen
        reader each time. Its next correction writes the file again anyway.

        Exports are the one thing here that is meant to be written again.
        They hold nothing that is not derived from the transcript, so a fresh
        run producing a fresh report loses nothing, which is the opposite of
        the rule that governs the responses next door.

        ``rebuildable`` is for an export made from the saved transcript alone.
        When the old file is held open, the new data is deleted rather than
        kept beside it as ``.unsaved``: the next write makes it again, and a
        stray copy in the folder a person hands on would only mislead. A
        successful write also removes any such copy an earlier version left.
        An export that cost a paid call to make, such as the smooth
        transcript, is not rebuildable and keeps its copy.
        """
        path = self.exports_folder / name
        sleep = time.sleep if patient else _no_wait
        if not _write_bytes(path, text.encode("utf-8"), sleep, keep_unsaved=not rebuildable):
            return None
        if rebuildable:
            stale = unsaved_copy_path(path)
            try:
                stale.unlink(missing_ok=True)
            except OSError:
                _log.warning("Could not remove the old unsaved copy %s.", stale, exc_info=True)
        return path

    def remove_export(self, name: str) -> bool:
        """Remove one export, if it is there. Returns whether it is now gone.

        Used for an export that no longer matches the transcript beside it,
        such as a smooth transcript made from an earlier run.
        """
        path = self.exports_folder / name
        try:
            path.unlink(missing_ok=True)
        except OSError:
            _log.warning("Could not remove the %s export.", name, exc_info=True)
            return False
        return True


def transcript_folder_for(
    recording_path: Path | str,
    folder_suffix: str = DEFAULT_FOLDER_SUFFIX,
) -> Path:
    """Return the transcript folder that belongs beside ``recording_path``.

    The recording's whole file name is used, extension included, so that two
    recordings of the same name in different formats keep their transcripts
    apart.
    """
    path = Path(recording_path)
    return path.parent / f"{path.name}{folder_suffix}"


def _body_of(payload: str | bytes | dict[str, Any], provider: Provider) -> bytes | None:
    """The bytes to write for one service's answer, or ``None`` if there are none.

    This cannot raise, and that is the whole point of it. It runs at the end
    of a pass, once every service has answered and been paid for, and its
    caller treats a failure as a warning against a transcript that is
    otherwise good. An exception escaping here would throw that transcript
    away instead, which is the opposite of what keeping the answer is for.

    A parsed answer is written as JSON, which is the closest thing to
    untouched there is for something that has already been parsed. Where a
    value in it will not serialise, it is written through ``str`` rather than
    the whole answer being dropped, because a record with one value flattened
    still explains a transcript and no record at all does not.
    """
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, str):
        return payload.encode("utf-8")
    try:
        return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError):
        _log.warning(
            "The answer from %s held something that is not JSON, so it has been "
            "written with those values as text.",
            provider.display_name,
            exc_info=True,
        )
    try:
        return json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        _log.warning(
            "The answer from %s could not be written at all.",
            provider.display_name,
            exc_info=True,
        )
        return None


def _no_wait(_seconds: float) -> None:
    """A sleep that does not wait, for a write that must not hold the window up."""


def _write_bytes(
    path: Path,
    body: bytes,
    sleep: Callable[[float], None] | None = None,
    keep_unsaved: bool = True,
) -> bool:
    """Write bytes to ``path`` atomically, returning whether it worked.

    :func:`~vox_verbatim.json_store.write_json_object` does this for
    objects the application builds itself, but a provider's answer must reach
    the disk byte for byte as it arrived. Parsing it and writing it out again
    would quietly change the spacing, the key order and the escaping, and the
    whole reason for keeping it is that it has not been changed.

    ``sleep`` is how to wait between attempts to move the file into place.
    None means the real clock, looked up when the call is made.

    ``keep_unsaved`` False deletes the data when the move fails, for a file
    that the next write rebuilds anyway.
    """
    if sleep is None:
        sleep = time.sleep
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=path.name,
            suffix=".tmp",
            delete=False,
        )
        temp_name = handle.name
        try:
            with handle:
                handle.write(body)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
        # The move is retried and, failing that, the data is kept beside the
        # target rather than deleted, for the same reason as the transcript
        # itself: the answer from a service is evidence, and the smooth
        # transcript cost a paid call. Only a file the caller can rebuild
        # from the saved transcript is let go.
        try:
            moved = replace_with_retries(temp_name, path, sleep)
        except BaseException:
            # Not a held file but something wrong with the place itself, so
            # there is nothing to keep the data for.
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
        if not moved:
            if keep_unsaved:
                keep_unsaved_copy(temp_name, path)
            else:
                try:
                    os.unlink(temp_name)
                except OSError:
                    _log.warning("Could not remove %s", temp_name, exc_info=True)
            return False
    except OSError:
        _log.warning("Could not write %s", path, exc_info=True)
        return False
    return True
