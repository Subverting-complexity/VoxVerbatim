"""Tests for the transcript folder that sits beside a recording.

Everything here is built by hand. No audio is read and no service is called,
because the point of these tests is what reaches the disk and what comes
back, not how the words were arrived at.
"""

from __future__ import annotations

import json
from pathlib import Path

from vox_verbatim.transcription.model import (
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
    TimingStatus,
    TokenReference,
    Transcript,
)
from vox_verbatim.transcription.store import (
    DEFAULT_FOLDER_SUFFIX,
    REDACTED_TEXT,
    TranscriptStore,
    redact_settings,
    transcript_folder_for,
    transcript_from_dict,
    transcript_to_dict,
)


def _full_transcript() -> Transcript:
    """A transcript using every enumeration and every nested type once.

    It is deliberately over-full. A round trip that only exercises the
    common fields would pass happily while quietly losing the provenance
    that only appears on the hard cases.
    """
    return Transcript(
        recording_name="board-meeting.m4a",
        canonical_audio=CanonicalAudio(
            path=r"C:\Audio\board-meeting.wav",
            original_path=r"C:\Audio\board-meeting.m4a",
            duration=754.5,
            sample_rate=16000,
            channels=1,
            size_bytes=24_137_216,
            container="wav",
            is_copy=True,
        ),
        configuration=RecordingConfiguration(
            afrikaans_enabled=True,
            expected_speaker_count=3,
            known_speakers=["Adrienne", "Michael"],
            recording_context="A quarterly board meeting about the new premises.",
            vocabulary_profile_ids=["global", "client-inversion"],
        ),
        tokens=[
            FinalToken(
                id="token-one",
                text="fifteen",
                normalised_text="fifteen",
                text_source=Provider.ELEVENLABS,
                text_confidence=Confidence.HIGH,
                confidence_strength=0.93,
                start=12.25,
                end=12.75,
                timing_source=Provider.ELEVENLABS,
                timing_status=TimingStatus.EXACT_PROVIDER_TIME,
                timing_confidence=Confidence.HIGH,
                speaker="speaker_0",
                speaker_source=Provider.ELEVENLABS,
                speaker_confidence=Confidence.HIGH,
                language=Language.ENGLISH,
                language_evidence=LanguageEvidence(
                    scores={Language.ENGLISH: 0.97, Language.AFRIKAANS: 0.03}
                ),
                source_tokens=[TokenReference(Provider.ELEVENLABS, 4)],
                source_audio_span=AudioSpan(12.0, 13.0),
                alignment_status=AlignmentStatus.EXACT,
                candidates=[
                    Candidate(
                        text="fifteen",
                        providers=(Provider.ELEVENLABS, Provider.OPENAI),
                        score=0.91,
                        components={"agreement": 2.0, "acoustic": 0.86},
                        source_tokens=(TokenReference(Provider.OPENAI, 4),),
                        in_vocabulary=True,
                    )
                ],
                rejected_tokens=[TokenReference(Provider.MICROSOFT, 5)],
                risk_categories=[RiskCategory.QUANTITY],
                review_status=ReviewStatus.SETTLED,
                review_reasons=[],
                llm_decision=None,
                human_corrected=False,
                original_text=None,
            ),
            FinalToken(
                id="token-two",
                text="50,000",
                normalised_text="50000",
                text_source=None,
                text_confidence=Confidence.UNRESOLVED,
                confidence_strength=0.21,
                start=None,
                end=None,
                timing_source=Provider.DEEPGRAM,
                timing_status=TimingStatus.SHARED_PHRASE_SPAN,
                timing_confidence=Confidence.REVIEW_REQUIRED,
                speaker="speaker_1",
                speaker_source=Provider.ASSEMBLYAI,
                speaker_confidence=Confidence.REVIEW_SUGGESTED,
                language=Language.GERMAN,
                language_evidence=LanguageEvidence(scores={Language.GERMAN: 0.61}),
                source_tokens=[
                    TokenReference(Provider.ELEVENLABS, 9),
                    TokenReference(Provider.OPENAI, 9),
                ],
                source_audio_span=AudioSpan(183.4, 188.1),
                alignment_status=AlignmentStatus.SUBSTITUTION,
                candidates=[
                    Candidate(text="15,000", providers=(Provider.ELEVENLABS,), score=0.4),
                    Candidate(text="50,000", providers=(Provider.OPENAI,), score=0.45),
                ],
                rejected_tokens=[],
                risk_categories=[RiskCategory.MONEY, RiskCategory.LEGAL_IDENTIFIER],
                review_status=ReviewStatus.PENDING,
                review_reasons=[
                    ReviewReason.NUMERIC_DISAGREEMENT,
                    ReviewReason.ESCALATION_UNRESOLVED,
                ],
                llm_decision="Declined: the services disagree on an amount.",
                human_corrected=True,
                original_text="fifty thousand",
            ),
        ],
        clean_segments=[
            CleanSegment(
                text="Fifteen of the fifty thousand.",
                source_token_ids=["token-one", "token-two"],
                speaker="speaker_0",
                start=12.25,
                end=188.1,
            )
        ],
        speakers=[Speaker(id="speaker_0", name="Adrienne"), Speaker(id="speaker_1")],
        provider_results={
            Provider.ELEVENLABS: ProviderResult(
                provider=Provider.ELEVENLABS,
                tokens=[
                    ProviderToken(
                        provider=Provider.ELEVENLABS,
                        index=4,
                        text="fifteen",
                        start=12.25,
                        end=12.75,
                        speaker="speaker_0",
                        log_probability=-0.02,
                        confidence=None,
                        language=Language.ENGLISH,
                        chunk_index=0,
                        is_punctuation=False,
                    ),
                    ProviderToken(
                        provider=Provider.ELEVENLABS,
                        index=5,
                        text=".",
                        start=12.75,
                        end=12.8,
                        is_punctuation=True,
                    ),
                    ProviderToken(
                        provider=Provider.ELEVENLABS,
                        index=6,
                        text="(laughter)",
                        start=12.8,
                        end=13.4,
                        is_audio_event=True,
                    ),
                ],
                request=ProviderRequestRecord(
                    provider=Provider.ELEVENLABS,
                    model_identifier="scribe-v2",
                    model_version="2026-05-01",
                    request_parameters={"diarize": True, "num_speakers": 3},
                    language_configuration="multi",
                    vocabulary_terms=("Inversion", "Adrienne"),
                    chunks=(
                        ChunkRecord(
                            provider=Provider.ELEVENLABS,
                            chunk_index=0,
                            canonical_start=0.0,
                            canonical_end=754.5,
                            canonical_offset=0.0,
                            encoded_size_bytes=24_137_216,
                            path=r"C:\Audio\chunk-0.wav",
                            provider_request_id="req-abc",
                            overlap_before=0.0,
                        ),
                    ),
                    raw_response_file="0001-elevenlabs-transcription-chunk00.json",
                    processing_seconds=42.5,
                    started_at="2026-08-17T10:00:00",
                    succeeded=True,
                    error=None,
                    purpose="transcription",
                ),
                detected_language=Language.ENGLISH,
                speakers=["speaker_0", "speaker_1"],
                error=None,
            ),
            Provider.MICROSOFT: ProviderResult(
                provider=Provider.MICROSOFT,
                tokens=[],
                request=None,
                detected_language=Language.UNKNOWN,
                speakers=[],
                error="The service did not answer within the timeout.",
            ),
        },
        requests=[
            ProviderRequestRecord(
                provider=Provider.ASSEMBLYAI,
                model_identifier="universal-2",
                request_parameters={"speaker_labels": True},
                chunks=(
                    ChunkRecord(
                        provider=Provider.ASSEMBLYAI,
                        chunk_index=0,
                        canonical_start=178.4,
                        canonical_end=193.1,
                        canonical_offset=178.4,
                        encoded_size_bytes=470_000,
                    ),
                ),
                processing_seconds=3.25,
                started_at="2026-08-17T10:03:00",
                purpose="escalation",
            )
        ],
        reconciliation_version=1,
        created_at="2026-08-17T10:00:00",
        completed_at="2026-08-17T10:04:00",
        effective_settings={"openai_model": "gpt-transcribe", "max_tokens": 4096},
        warnings=["Microsoft MAI did not answer, so this run had one fewer opinion."],
    )


# -- The round trip -------------------------------------------------------


def test_a_saved_transcript_comes_back_exactly_as_it_went_in(tmp_path):
    store = TranscriptStore(tmp_path / "board-meeting.m4a")
    original = _full_transcript()

    assert store.save(original)
    loaded = store.load()

    assert loaded == original


def test_every_enumeration_survives_the_round_trip(tmp_path):
    """Enumerations are written by value, and the values are what must return."""
    store = TranscriptStore(tmp_path / "board-meeting.m4a")
    store.save(_full_transcript())

    loaded = store.load()
    assert loaded is not None
    first, second = loaded.tokens

    assert first.text_source is Provider.ELEVENLABS
    assert first.text_confidence is Confidence.HIGH
    assert first.timing_status is TimingStatus.EXACT_PROVIDER_TIME
    assert first.alignment_status is AlignmentStatus.EXACT
    assert first.language is Language.ENGLISH
    assert first.risk_categories == [RiskCategory.QUANTITY]
    assert first.review_status is ReviewStatus.SETTLED
    assert second.review_reasons == [
        ReviewReason.NUMERIC_DISAGREEMENT,
        ReviewReason.ESCALATION_UNRESOLVED,
    ]
    assert second.language_evidence.scores == {Language.GERMAN: 0.61}
    assert loaded.provider_results[Provider.ELEVENLABS].detected_language is Language.ENGLISH


def test_the_shape_of_the_nested_types_is_kept(tmp_path):
    """Tuples must come back as tuples, or the frozen records stop comparing."""
    store = TranscriptStore(tmp_path / "board-meeting.m4a")
    store.save(_full_transcript())

    loaded = store.load()
    assert loaded is not None
    request = loaded.provider_results[Provider.ELEVENLABS].request
    assert request is not None
    assert isinstance(request.vocabulary_terms, tuple)
    assert isinstance(request.chunks, tuple)
    assert request.chunks[0].canonical_offset == 0.0
    assert isinstance(loaded.tokens[0].candidates[0].providers, tuple)
    assert isinstance(loaded.tokens[0].candidates[0].source_tokens, tuple)
    assert loaded.tokens[0].source_audio_span == AudioSpan(12.0, 13.0)
    assert loaded.tokens[1].start is None


def test_the_written_file_keys_enumerations_by_value(tmp_path):
    store = TranscriptStore(tmp_path / "board-meeting.m4a")
    store.save(_full_transcript())

    data = json.loads(store.transcript_path.read_text(encoding="utf-8"))

    assert set(data["provider_results"]) == {"elevenlabs", "microsoft"}
    assert data["tokens"][0]["text_confidence"] == "high"
    assert data["tokens"][0]["language_evidence"] == {"en": 0.97, "af": 0.03}


def test_a_transcript_with_nothing_in_it_still_round_trips(tmp_path):
    store = TranscriptStore(tmp_path / "empty.m4a")
    original = Transcript(recording_name="empty.m4a")

    assert store.save(original)
    assert store.load() == original


# -- Where the folder goes ------------------------------------------------


def test_the_folder_is_named_after_the_whole_recording(tmp_path):
    folder = transcript_folder_for(tmp_path / "talk.m4a")

    assert folder == tmp_path / f"talk.m4a{DEFAULT_FOLDER_SUFFIX}"


def test_two_recordings_of_the_same_name_do_not_share_a_folder(tmp_path):
    assert transcript_folder_for(tmp_path / "talk.m4a") != transcript_folder_for(
        tmp_path / "talk.wav"
    )


def test_the_suffix_can_be_changed(tmp_path):
    store = TranscriptStore(tmp_path / "talk.m4a", folder_suffix=".notes")

    assert store.folder == tmp_path / "talk.m4a.notes"


def test_making_a_store_writes_nothing(tmp_path):
    TranscriptStore(tmp_path / "talk.m4a")

    assert list(tmp_path.iterdir()) == []


# -- Checking the folder before the money is spent -----------------------


def test_a_writable_folder_passes_the_probe_and_is_left_clean(tmp_path):
    store = TranscriptStore(tmp_path / "talk.m4a")

    assert store.probe_writable() is None
    # The folder is made, which the run would do anyway, and the probe file
    # is not left in it.
    assert store.folder.is_dir()
    assert list(store.folder.iterdir()) == []


def test_a_folder_that_cannot_be_written_is_reported_in_a_sentence(tmp_path, monkeypatch):
    store = TranscriptStore(tmp_path / "talk.m4a")

    def refuse(self, *args, **kwargs):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(Path, "mkdir", refuse)
    problem = store.probe_writable()

    assert problem is not None
    assert "talk.m4a" in problem
    assert "Access is denied" in problem
    assert "read-only" in problem


def test_a_path_past_the_windows_limit_is_named_as_the_reason(tmp_path, monkeypatch):
    deep = tmp_path / ("a" * 200) / ("b" * 100)
    store = TranscriptStore(deep / "talk.m4a")

    def refuse(self, *args, **kwargs):
        raise FileNotFoundError(3, "The system cannot find the path specified")

    monkeypatch.setattr(Path, "mkdir", refuse)
    problem = store.probe_writable()

    assert problem is not None
    assert "260" in problem
    assert "shorter path" in problem


def test_a_transcript_held_open_is_kept_beside_the_file_rather_than_lost(tmp_path, monkeypatch):
    """The save says it failed, and the finished transcript is still on the disk."""
    from vox_verbatim import json_store

    store = TranscriptStore(tmp_path / "talk.m4a")
    assert store.save(Transcript(recording_name="talk.m4a")) is True

    real_replace = json_store.os.replace

    def held(source, destination):
        if Path(destination) == store.transcript_path:
            raise PermissionError(13, "The process cannot access the file")
        return real_replace(source, destination)

    monkeypatch.setattr(json_store.os, "replace", held)
    monkeypatch.setattr(json_store.time, "sleep", lambda _seconds: None)

    assert store.save(Transcript(recording_name="talk.m4a", warnings=["new"])) is False
    assert store.unsaved_transcript_path == store.folder / "transcript.json.unsaved"
    assert store.unsaved_transcript_path.is_file()
    assert "new" in store.unsaved_transcript_path.read_text(encoding="utf-8")
    # The previous transcript is untouched.
    assert store.load() is not None


# -- Damaged and partial files -------------------------------------------


def test_a_missing_transcript_reads_as_nothing(tmp_path):
    store = TranscriptStore(tmp_path / "never-transcribed.m4a")

    assert not store.has_transcript
    assert store.load() is None


def test_a_damaged_transcript_is_ignored_rather_than_raising(tmp_path):
    store = TranscriptStore(tmp_path / "talk.m4a")
    store.transcript_path.parent.mkdir(parents=True)
    store.transcript_path.write_text('{"recording_name": "talk.m4a", "tok', encoding="utf-8")

    assert store.load() is None


def test_a_transcript_that_is_not_text_at_all_is_ignored(tmp_path):
    store = TranscriptStore(tmp_path / "talk.m4a")
    store.transcript_path.parent.mkdir(parents=True)
    store.transcript_path.write_bytes(b'{"recording_name": "\xff\xfe"}')

    assert store.load() is None


def test_a_transcript_with_no_recording_name_is_ignored(tmp_path):
    store = TranscriptStore(tmp_path / "talk.m4a")
    store.transcript_path.parent.mkdir(parents=True)
    store.transcript_path.write_text(json.dumps({"tokens": []}), encoding="utf-8")

    assert store.load() is None


def test_a_partly_damaged_transcript_keeps_what_it_can(tmp_path):
    """One bad word must not cost the user the rest of the transcript."""
    store = TranscriptStore(tmp_path / "talk.m4a")
    store.transcript_path.parent.mkdir(parents=True)
    store.transcript_path.write_text(
        json.dumps(
            {
                "recording_name": "talk.m4a",
                "tokens": [
                    {"id": "good", "text": "hello", "text_confidence": "high"},
                    "this is not a word at all",
                    {"id": "odd", "text": "world", "text_confidence": "purple",
                     "start": "not a number", "review_reasons": ["invented_reason"]},
                ],
                "speakers": [{"name": "no identifier"}],
                "warnings": ["kept", 7],
                "provider_results": {"not-a-service": {}},
            }
        ),
        encoding="utf-8",
    )

    loaded = store.load()

    assert loaded is not None
    assert [token.text for token in loaded.tokens] == ["hello", "world"]
    assert loaded.tokens[1].text_confidence is Confidence.UNRESOLVED
    assert loaded.tokens[1].start is None
    assert loaded.tokens[1].review_reasons == []
    assert loaded.speakers == []
    assert loaded.warnings == ["kept"]
    assert loaded.provider_results == {}


# -- The composite confidence on each word --------------------------------


def test_the_strength_behind_a_word_survives_the_round_trip(tmp_path):
    """The number itself, not merely the category it happens to fall into.

    Two words are checked rather than one, because the interesting failure
    is a strength that is written but read back for the wrong word, and a
    single-word fixture cannot show that.
    """
    store = TranscriptStore(tmp_path / "board-meeting.m4a")
    store.save(_full_transcript())

    loaded = store.load()

    assert loaded is not None
    assert [token.confidence_strength for token in loaded.tokens] == [0.93, 0.21]


def test_the_strength_is_written_under_the_name_the_rest_of_the_feature_reads(tmp_path):
    """The key on disk is part of the contract, so it is pinned here."""
    store = TranscriptStore(tmp_path / "board-meeting.m4a")
    store.save(_full_transcript())

    data = json.loads(store.transcript_path.read_text(encoding="utf-8"))

    assert data["tokens"][0]["confidence_strength"] == 0.93


def test_whether_the_text_was_replaced_survives_the_round_trip(tmp_path):
    """Under the key the rest of the feature reads, which is part of the contract."""
    transcript = _full_transcript()
    transcript.tokens[1].text_corrected = True
    store = TranscriptStore(tmp_path / "board-meeting.m4a")
    store.save(transcript)

    data = json.loads(store.transcript_path.read_text(encoding="utf-8"))
    loaded = store.load()

    assert data["tokens"][0]["text_corrected"] is False
    assert data["tokens"][1]["text_corrected"] is True
    assert loaded is not None
    assert [token.text_corrected for token in loaded.tokens] == [False, True]


def test_a_word_corrected_before_the_flag_existed_is_still_known_to_be_corrected():
    """Every transcript already on disk looks exactly like this.

    Reading the missing flag as ``False`` would be a real loss rather than a
    tidy default: a replacement rule refuses to touch a word whose text was
    replaced, so the corrections in every older transcript would lose that
    protection and be overwritten the first time the folder was analysed.
    ``original_text`` is only ever written where the text was actually
    replaced, so it answers the question exactly for those files.
    """
    loaded = transcript_from_dict(
        {
            "recording_name": "talk.m4a",
            "tokens": [
                {
                    "id": "corrected",
                    "text": "Bosch",
                    "human_corrected": True,
                    "original_text": "Bosh",
                },
                {
                    "id": "timing-only",
                    "text": "Bosh",
                    "human_corrected": True,
                },
                {"id": "untouched", "text": "signed"},
            ],
        }
    )

    assert loaded is not None
    assert [token.text_corrected for token in loaded.tokens] == [True, False, False]


def test_a_damaged_text_corrected_flag_falls_back_rather_than_stopping_the_load():
    """A value of the wrong type is read as though the field were absent."""
    loaded = transcript_from_dict(
        {
            "recording_name": "talk.m4a",
            "tokens": [
                {"id": "nonsense", "text": "Bosch", "text_corrected": "yes",
                 "original_text": "Bosh"},
                {"id": "also", "text": "signed", "text_corrected": 1},
            ],
        }
    )

    assert loaded is not None
    assert [token.text_corrected for token in loaded.tokens] == [True, False]


def test_a_transcript_written_before_the_strength_existed_reads_as_unmeasured():
    """Every transcript already on disk looks exactly like this.

    The answer has to be "nobody worked this out", which is ``None``, and
    not a zero. A zero would say the word is as weak as a word can be, and
    every word of every transcript made before today would then arrive at
    the top of the review queue.
    """
    loaded = transcript_from_dict(
        {
            "recording_name": "talk.m4a",
            "tokens": [{"id": "old", "text": "hello", "text_confidence": "high"}],
        }
    )

    assert loaded is not None
    assert loaded.tokens[0].confidence_strength is None
    assert loaded.tokens[0].text_confidence is Confidence.HIGH


def test_a_damaged_strength_reads_as_unmeasured_rather_than_stopping_the_load():
    """Three kinds of damage, and one honest answer to all of them.

    A word instead of a number is a hand edit or a corrupted file. A number
    far outside nought to one is the same damage in a shape that would pass
    a type check, and clamping it to one would turn a broken file into a
    word that looks thoroughly settled and is never shown to anybody again.
    ``True`` is included because JSON booleans are integers in Python, and a
    check that only asked whether the value was a number would let it
    through as 1.0.
    """
    loaded = transcript_from_dict(
        {
            "recording_name": "talk.m4a",
            "tokens": [
                {"id": "text", "text": "one", "confidence_strength": "very good"},
                {"id": "high", "text": "two", "confidence_strength": 42},
                {"id": "negative", "text": "three", "confidence_strength": -0.5},
                {"id": "boolean", "text": "four", "confidence_strength": True},
            ],
        }
    )

    assert loaded is not None
    assert [token.text for token in loaded.tokens] == ["one", "two", "three", "four"]
    assert all(token.confidence_strength is None for token in loaded.tokens)


def test_the_ends_of_the_scale_are_kept_rather_than_treated_as_damage():
    """Nought and one are real answers, and both sit on the boundary."""
    loaded = transcript_from_dict(
        {
            "recording_name": "talk.m4a",
            "tokens": [
                {"id": "lowest", "text": "one", "confidence_strength": 0},
                {"id": "highest", "text": "two", "confidence_strength": 1},
            ],
        }
    )

    assert loaded is not None
    assert [token.confidence_strength for token in loaded.tokens] == [0.0, 1.0]


def test_a_span_that_ends_before_it_starts_is_dropped(tmp_path):
    """The model refuses such a span, so reading one must not raise."""
    loaded = transcript_from_dict(
        {
            "recording_name": "talk.m4a",
            "tokens": [{"id": "one", "source_audio_span": {"start": 9.0, "end": 2.0}}],
        }
    )

    assert loaded is not None
    assert loaded.tokens[0].source_audio_span is None


def test_what_a_service_returned_as_an_event_rather_than_a_word_survives(tmp_path):
    store = TranscriptStore(tmp_path / "board-meeting.m4a")
    store.save(_full_transcript())

    loaded = store.load()

    assert loaded is not None
    tokens = loaded.provider_results[Provider.ELEVENLABS].tokens
    assert [token.is_punctuation for token in tokens] == [False, True, False]
    assert [token.is_audio_event for token in tokens] == [False, False, True]
    assert [token.text for token in tokens if token.is_spoken_word] == ["fifteen"]


def test_a_transcript_written_before_a_field_existed_is_still_read():
    """Provenance is immutable, so old files must survive the model growing.

    The fixture is written out by hand with ``is_audio_event`` missing,
    which is exactly what every transcript on disk looked like before that
    field was added. It must read as False rather than raising.
    """
    loaded = transcript_from_dict(
        {
            "recording_name": "talk.m4a",
            "provider_results": {
                "elevenlabs": {
                    "provider": "elevenlabs",
                    "tokens": [
                        {"provider": "elevenlabs", "index": 0, "text": "hello",
                         "start": 0.5, "end": 0.9, "is_punctuation": False}
                    ],
                }
            },
        }
    )

    assert loaded is not None
    token = loaded.provider_results[Provider.ELEVENLABS].tokens[0]
    assert token.is_audio_event is False
    assert token.is_spoken_word is True


def test_a_transcript_missing_every_optional_field_falls_back_to_the_defaults():
    """The rule that keeps old files readable, pinned for the next new field.

    Each object is compared against a freshly built one, so a field added to
    the model without being read here shows up as a difference rather than
    as a silent default that happened to be right.
    """
    loaded = transcript_from_dict(
        {
            "recording_name": "talk.m4a",
            "tokens": [{}],
            "clean_segments": [{}],
            "speakers": [{"id": "0"}],
            "provider_results": {
                "openai": {"tokens": [{}], "request": {"provider": "openai"}}
            },
            "requests": [{"provider": "openai"}],
        }
    )

    assert loaded is not None
    assert loaded.tokens == [FinalToken(id=loaded.tokens[0].id)]
    assert loaded.clean_segments == [CleanSegment(text="")]
    assert loaded.speakers == [Speaker(id="0")]
    assert loaded.requests == [
        ProviderRequestRecord(provider=Provider.OPENAI, model_identifier="")
    ]
    assert loaded.provider_results == {
        Provider.OPENAI: ProviderResult(
            provider=Provider.OPENAI,
            tokens=[ProviderToken(provider=Provider.OPENAI, index=0, text="")],
            request=ProviderRequestRecord(provider=Provider.OPENAI, model_identifier=""),
        )
    }
    assert loaded.configuration == RecordingConfiguration()


def test_a_transcript_holding_only_its_name_is_an_empty_transcript():
    assert transcript_from_dict({"recording_name": "talk.m4a"}) == Transcript(
        recording_name="talk.m4a"
    )


def test_a_newer_format_version_is_still_read(tmp_path):
    loaded = transcript_from_dict(
        {"recording_name": "talk.m4a", "version": 99, "tokens": [{"text": "hello"}]}
    )

    assert loaded is not None
    assert loaded.tokens[0].text == "hello"


# -- Provenance is immutable ---------------------------------------------


def test_a_raw_response_is_stored_exactly_as_it_arrived(tmp_path):
    store = TranscriptStore(tmp_path / "talk.m4a")
    body = '{"words":[{"text":"hallo",   "start":0.5}]}\n'

    name = store.write_raw_response(Provider.ELEVENLABS, body)

    assert name is not None
    assert store.read_raw_response(name) == body



def test_a_value_that_will_not_serialise_never_costs_the_transcript(tmp_path):
    """The caller treats a failure here as a warning, so nothing may escape.

    This runs at the very end of a pass, once every service has answered and
    been paid for. An exception leaving this method throws away a transcript
    that is otherwise finished and good, which is the opposite of what keeping
    a copy of the answer is for.
    """
    import datetime

    store = TranscriptStore(tmp_path / "talk.m4a")
    moment = datetime.datetime(2024, 5, 12, 18, 57, 13, tzinfo=datetime.UTC)
    payload = {"metadata": {"created": moment}}

    name = store.write_raw_response(Provider.DEEPGRAM, payload)

    assert name is not None
    written = store.read_raw_response(name)
    assert written is not None and "2024-05-12 18:57:13" in written


def test_an_answer_that_cannot_be_written_at_all_is_reported_as_such(tmp_path):
    """Still no exception, and still an honest answer to the caller."""
    store = TranscriptStore(tmp_path / "talk.m4a")
    payload: dict = {}
    payload["itself"] = payload

    name = store.write_raw_response(Provider.DEEPGRAM, payload)

    assert name is None


def test_writing_the_same_response_twice_never_replaces_the_first(tmp_path):
    store = TranscriptStore(tmp_path / "talk.m4a")

    first = store.write_raw_response(Provider.OPENAI, '{"text": "the first answer"}')
    second = store.write_raw_response(Provider.OPENAI, '{"text": "the second answer"}')

    assert first != second
    assert store.read_raw_response(first) == '{"text": "the first answer"}'
    assert store.read_raw_response(second) == '{"text": "the second answer"}'
    assert len(store.raw_response_names()) == 2


def test_a_second_run_adds_to_the_provenance_and_changes_none_of_it(tmp_path):
    """Reprocessing a recording must leave the first run's evidence untouched."""
    recording = tmp_path / "talk.m4a"
    first_run = TranscriptStore(recording)
    for chunk in range(3):
        first_run.write_raw_response(
            Provider.ELEVENLABS, f'{{"chunk": {chunk}}}', chunk_index=chunk
        )
    before = {
        name: first_run.read_raw_response(name) for name in first_run.raw_response_names()
    }

    second_run = TranscriptStore(recording)
    for chunk in range(3):
        second_run.write_raw_response(
            Provider.ELEVENLABS, f'{{"chunk": {chunk}, "run": 2}}', chunk_index=chunk
        )

    after = {
        name: second_run.read_raw_response(name) for name in second_run.raw_response_names()
    }
    assert len(after) == 6
    # Every file the first run wrote is still there, saying what it said.
    for name, body in before.items():
        assert after[name] == body


def test_a_response_removed_by_hand_does_not_make_the_numbering_go_backwards(tmp_path):
    store = TranscriptStore(tmp_path / "talk.m4a")
    first = store.write_raw_response(Provider.OPENAI, "one")
    second = store.write_raw_response(Provider.OPENAI, "two")
    store.raw_response_path(first).unlink()

    third = store.write_raw_response(Provider.OPENAI, "three")

    assert third != second
    assert store.read_raw_response(second) == "two"


def test_the_names_say_which_service_and_which_request(tmp_path):
    store = TranscriptStore(tmp_path / "talk.m4a")

    store.write_raw_response(Provider.ELEVENLABS, "{}", chunk_index=0)
    store.write_raw_response(Provider.OPENAI, "{}", purpose="transcription", chunk_index=2)
    store.write_raw_response(
        Provider.ASSEMBLYAI,
        "{}",
        purpose="escalation",
        window=AudioSpan(183.4, 188.1),
    )
    store.write_raw_response(Provider.OPENAI, "{}", purpose="adjudication")

    assert store.raw_response_names() == [
        "0001-elevenlabs-transcription-chunk00.json",
        "0002-openai-transcription-chunk02.json",
        "0003-assemblyai-escalation-3m03s-3m08s.json",
        "0004-openai-adjudication.json",
    ]


def test_a_long_recording_names_its_windows_with_hours(tmp_path):
    store = TranscriptStore(tmp_path / "talk.m4a")

    name = store.write_raw_response(
        Provider.ASSEMBLYAI, "{}", purpose="escalation", window=AudioSpan(3863.0, 3868.0)
    )

    assert name == "0001-assemblyai-escalation-1h04m23s-1h04m28s.json"


def test_a_parsed_response_is_written_as_json(tmp_path):
    store = TranscriptStore(tmp_path / "talk.m4a")

    name = store.write_raw_response(Provider.DEEPGRAM, {"results": {"channels": []}})

    assert json.loads(store.read_raw_response(name)) == {"results": {"channels": []}}


def test_raw_responses_and_exports_live_in_their_own_folders(tmp_path):
    store = TranscriptStore(tmp_path / "talk.m4a")
    store.save(Transcript(recording_name="talk.m4a"))
    store.write_raw_response(Provider.OPENAI, "{}")
    store.write_export("transcript.txt", "Some words.\n")

    assert store.transcript_path.is_file()
    assert store.raw_responses_folder.is_dir()
    assert (store.exports_folder / "transcript.txt").read_text(encoding="utf-8") == "Some words.\n"


def test_an_export_may_be_written_again(tmp_path):
    """Exports are derived, so replacing one loses nothing."""
    store = TranscriptStore(tmp_path / "talk.m4a")

    store.write_export("review-report.md", "old")
    path = store.write_export("review-report.md", "new")

    assert path is not None
    assert path.read_text(encoding="utf-8") == "new"
    assert [item.name for item in store.exports_folder.iterdir()] == ["review-report.md"]


def test_saving_leaves_no_temporary_files_behind(tmp_path):
    store = TranscriptStore(tmp_path / "talk.m4a")

    store.save(Transcript(recording_name="talk.m4a"))
    store.save(Transcript(recording_name="talk.m4a", warnings=["one"]))
    store.write_raw_response(Provider.OPENAI, "{}")

    assert sorted(item.name for item in store.folder.iterdir()) == [
        "raw-responses",
        "transcript.json",
    ]
    assert len(list(store.raw_responses_folder.iterdir())) == 1


# -- No API key ever reaches the disk -------------------------------------


def test_a_key_smuggled_into_the_settings_never_reaches_the_disk(tmp_path):
    """The settings layer redacts these already. This is the second lock."""
    store = TranscriptStore(tmp_path / "talk.m4a")
    transcript = Transcript(
        recording_name="talk.m4a",
        effective_settings={
            "openai_api_key": "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789",
            "elevenlabs": {"api_key": "0123456789abcdef0123456789abcdef"},
            "assemblyai_token": "aa11bb22cc33dd44ee55ff66aa77bb88",
            "azure": {"subscription_key": "secretvalue"},
            "extra_headers": ["Bearer sk-live-1234567890"],
            "notes": "9f8e7d6c5b4a39281706f5e4d3c2b1a09f8e7d6c",
        },
    )

    assert store.save(transcript)

    written = store.transcript_path.read_text(encoding="utf-8")
    for secret in (
        "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789",
        "0123456789abcdef0123456789abcdef",
        "aa11bb22cc33dd44ee55ff66aa77bb88",
        "secretvalue",
        "Bearer sk-live-1234567890",
        "9f8e7d6c5b4a39281706f5e4d3c2b1a09f8e7d6c",
    ):
        assert secret not in written
    assert REDACTED_TEXT in written


def test_a_key_in_a_request_parameter_never_reaches_the_disk(tmp_path):
    store = TranscriptStore(tmp_path / "talk.m4a")
    transcript = Transcript(
        recording_name="talk.m4a",
        requests=[
            ProviderRequestRecord(
                provider=Provider.OPENAI,
                model_identifier="gpt-transcribe",
                request_parameters={
                    "temperature": 0.0,
                    "headers": {"Authorization": "Bearer sk-live-secret"},
                },
            )
        ],
    )

    assert store.save(transcript)

    written = store.transcript_path.read_text(encoding="utf-8")
    assert "sk-live-secret" not in written
    assert "temperature" in written


def test_redaction_leaves_the_settings_that_only_look_secret(tmp_path):
    """Over-redacting would hide the very settings that explain a result."""
    cleaned = redact_settings(
        {
            "max_tokens": 4096,
            "keyterms": ["Inversion", "Adrienne"],
            "key_terms": ["Inversion"],
            "keywords": ["premises"],
            "model": "gpt-transcribe-2026-05-01",
            "language": "en",
            "audio_path": r"C:\Audio\board-meeting.wav",
        }
    )

    assert cleaned == {
        "max_tokens": 4096,
        "keyterms": ["Inversion", "Adrienne"],
        "key_terms": ["Inversion"],
        "keywords": ["premises"],
        "model": "gpt-transcribe-2026-05-01",
        "language": "en",
        "audio_path": r"C:\Audio\board-meeting.wav",
    }


def test_redaction_catches_a_key_under_an_innocent_name():
    cleaned = redact_settings({"note_to_self": "sk-proj-abcdefghijklmnop"})

    assert cleaned == {"note_to_self": REDACTED_TEXT}


def test_redacting_does_not_change_the_transcript_in_memory(tmp_path):
    """A save must not quietly edit the object the application is still using."""
    store = TranscriptStore(tmp_path / "talk.m4a")
    settings = {"openai_api_key": "sk-proj-abcdefghijklmnop"}
    transcript = Transcript(recording_name="talk.m4a", effective_settings=settings)

    store.save(transcript)

    assert settings == {"openai_api_key": "sk-proj-abcdefghijklmnop"}


def test_the_dictionary_form_can_be_used_on_its_own():
    """The pipeline may want the data without a folder to put it in."""
    original = _full_transcript()

    rebuilt = transcript_from_dict(json.loads(json.dumps(transcript_to_dict(original))))

    assert rebuilt == original
