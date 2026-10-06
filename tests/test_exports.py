"""Tests for the two documents a person reads when the transcription is done.

Every transcript here is built by hand, so each test says exactly what state
it is about. Nothing reads audio and nothing calls a service.
"""

from __future__ import annotations

from vox_verbatim.transcription.cost import estimate_cost
from vox_verbatim.transcription.exports import (
    REPORT_EXPORT_NAME,
    TEXT_EXPORT_NAME,
    format_timestamp,
    render_plain_text,
    render_review_report,
)
from vox_verbatim.transcription.model import (
    AudioSpan,
    Candidate,
    CanonicalAudio,
    ChunkRecord,
    Confidence,
    FinalToken,
    Provider,
    ProviderRequestRecord,
    ProviderResult,
    ProviderToken,
    ReviewReason,
    ReviewStatus,
    RiskCategory,
    Speaker,
    TimingStatus,
    TokenReference,
    Transcript,
)


def _word(
    text: str,
    start: float | None = None,
    speaker: str | None = "0",
    confidence: Confidence = Confidence.HIGH,
    **extra,
) -> FinalToken:
    """A settled word, unless a test says otherwise."""
    return FinalToken(
        text=text,
        normalised_text=text.lower(),
        text_source=Provider.ELEVENLABS,
        text_confidence=confidence,
        start=start,
        end=None if start is None else start + 0.4,
        timing_source=Provider.ELEVENLABS,
        timing_status=TimingStatus.EXACT_PROVIDER_TIME,
        timing_confidence=confidence,
        speaker=speaker,
        speaker_source=Provider.ELEVENLABS,
        speaker_confidence=confidence,
        **extra,
    )


def _settled_transcript() -> Transcript:
    """Two speakers, nothing to review, and a full stop of its own."""
    return Transcript(
        recording_name="board-meeting.m4a",
        canonical_audio=CanonicalAudio(
            path=r"C:\Audio\board-meeting.wav",
            original_path=r"C:\Audio\board-meeting.m4a",
            duration=754.0,
            sample_rate=16000,
            channels=1,
            size_bytes=24_137_216,
            container="wav",
        ),
        tokens=[
            _word("We", 12.0),
            _word("agreed", 12.5),
            _word("on", 13.0),
            _word("the", 13.4),
            _word("figure", 13.8),
            _word(".", 14.2),
            _word("No", 20.0, speaker="1"),
            _word("we", 20.4, speaker="1"),
            _word("did", 20.8, speaker="1"),
            _word("not", 21.2, speaker="1"),
            _word("!", 21.6, speaker="1"),
        ],
        speakers=[Speaker(id="0", name="Adrienne"), Speaker(id="1")],
        provider_results={
            Provider.ELEVENLABS: ProviderResult(
                provider=Provider.ELEVENLABS,
                request=ProviderRequestRecord(
                    provider=Provider.ELEVENLABS,
                    model_identifier="scribe-v2",
                    model_version="2026-05-01",
                    processing_seconds=42.0,
                ),
            ),
            Provider.OPENAI: ProviderResult(
                provider=Provider.OPENAI,
                request=ProviderRequestRecord(
                    provider=Provider.OPENAI,
                    model_identifier="gpt-transcribe",
                    processing_seconds=31.0,
                ),
            ),
        },
        created_at="2026-08-17T10:00:00",
        completed_at="2026-08-17T10:04:00",
    )


def _contested_transcript() -> Transcript:
    """One recording where a good deal still needs a person."""
    disputed = FinalToken(
        id="amount",
        text="50,000",
        text_confidence=Confidence.UNRESOLVED,
        start=183.4,
        end=184.0,
        timing_status=TimingStatus.SHARED_PHRASE_SPAN,
        timing_confidence=Confidence.REVIEW_REQUIRED,
        speaker="0",
        speaker_confidence=Confidence.HIGH,
        source_audio_span=AudioSpan(183.0, 188.1),
        source_tokens=[
            TokenReference(Provider.ELEVENLABS, 0),
            TokenReference(Provider.OPENAI, 0),
        ],
        candidates=[
            Candidate(text="15,000", providers=(Provider.ELEVENLABS,), score=0.4),
            Candidate(text="50,000", providers=(Provider.OPENAI,), score=0.45),
        ],
        rejected_tokens=[TokenReference(Provider.MICROSOFT, 0)],
        risk_categories=[RiskCategory.MONEY],
        review_status=ReviewStatus.PENDING,
        review_reasons=[
            ReviewReason.NUMERIC_DISAGREEMENT,
            ReviewReason.ESCALATION_UNRESOLVED,
        ],
        llm_decision="Declined: an amount must not be settled on plausibility.",
    )
    unnamed = FinalToken(
        id="name",
        text="Bosch",
        text_confidence=Confidence.REVIEW_SUGGESTED,
        start=3863.0,
        end=3863.6,
        timing_confidence=Confidence.HIGH,
        speaker="1",
        speaker_confidence=Confidence.HIGH,
        source_tokens=[TokenReference(Provider.ELEVENLABS, 1)],
        review_status=ReviewStatus.PENDING,
        review_reasons=[ReviewReason.PROPER_NAME_DISAGREEMENT],
    )
    floating = FinalToken(
        id="floating",
        text="premises",
        text_confidence=Confidence.REVIEW_REQUIRED,
        speaker="1",
        speaker_confidence=Confidence.HIGH,
        timing_confidence=Confidence.REVIEW_REQUIRED,
        source_audio_span=AudioSpan(400.0, 404.0),
        review_status=ReviewStatus.PENDING,
        review_reasons=[ReviewReason.UNALIGNED_WORD],
    )
    return Transcript(
        recording_name="board-meeting.m4a",
        canonical_audio=CanonicalAudio(
            path=r"C:\Audio\board-meeting.wav",
            original_path=r"C:\Audio\board-meeting.m4a",
            duration=4000.0,
            sample_rate=16000,
            channels=1,
            size_bytes=64_000_000,
            container="wav",
        ),
        tokens=[_word("about", 183.0), disputed, unnamed, floating],
        speakers=[Speaker(id="0", name="Adrienne"), Speaker(id="1")],
        provider_results={
            Provider.ELEVENLABS: ProviderResult(
                provider=Provider.ELEVENLABS,
                tokens=[
                    ProviderToken(Provider.ELEVENLABS, 0, "15,000", 183.4, 184.0),
                    ProviderToken(Provider.ELEVENLABS, 1, "Boss"),
                ],
                request=ProviderRequestRecord(
                    provider=Provider.ELEVENLABS,
                    model_identifier="scribe-v2",
                    model_version="2026-05-01",
                    processing_seconds=42.0,
                    started_at="2026-08-17T10:00:00",
                ),
            ),
            Provider.OPENAI: ProviderResult(
                provider=Provider.OPENAI,
                tokens=[ProviderToken(Provider.OPENAI, 0, "50,000")],
                request=ProviderRequestRecord(
                    provider=Provider.OPENAI,
                    model_identifier="gpt-transcribe",
                    processing_seconds=31.0,
                ),
            ),
            Provider.MICROSOFT: ProviderResult(
                provider=Provider.MICROSOFT,
                tokens=[ProviderToken(Provider.MICROSOFT, 0, "fifty thousand")],
                error="Microsoft MAI did not answer within the timeout.",
            ),
        },
        requests=[
            ProviderRequestRecord(
                provider=Provider.ELEVENLABS,
                model_identifier="scribe-v2",
                processing_seconds=42.0,
            ),
            ProviderRequestRecord(
                provider=Provider.ASSEMBLYAI,
                model_identifier="universal-2",
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
                processing_seconds=3.0,
                purpose="escalation",
            ),
            ProviderRequestRecord(
                provider=Provider.OPENAI,
                model_identifier="gpt-5-adjudicate",
                processing_seconds=2.0,
                purpose="adjudication",
            ),
        ],
        created_at="2026-08-17T10:00:00",
        completed_at="2026-08-17T10:04:00",
        warnings=["Microsoft MAI did not answer, so this run had one fewer opinion."],
    )


# -- The plain text export ------------------------------------------------


def test_the_header_names_the_recording_the_time_and_the_services():
    text = render_plain_text(_settled_transcript())

    assert text.startswith("Recording: board-meeting.m4a\n")
    assert "Length: 12:34" in text
    assert "Transcribed: 17 August 2026 at 10:04" in text
    assert "Services: ElevenLabs Scribe and OpenAI" in text


def test_consecutive_words_are_grouped_into_one_turn_for_each_speaker():
    text = render_plain_text(_settled_transcript())

    assert "[0:12] Adrienne:" in text
    assert "[0:20] Speaker 1:" in text
    assert text.count("Adrienne:") == 1
    assert text.count("Speaker 1:") == 1


def test_a_turn_holds_the_words_that_belong_to_it():
    text = render_plain_text(_settled_transcript())

    assert "We agreed on the figure." in text
    assert "No we did not!" in text


def test_punctuation_returned_on_its_own_is_put_back_on_the_word_before_it():
    """Some services time a full stop as a word, which must not become " .\"."""
    text = render_plain_text(_settled_transcript())

    assert " ." not in text
    assert " !" not in text


def test_a_word_the_evidence_never_settled_is_written_as_the_choice_it_is():
    text = render_plain_text(_contested_transcript())

    assert "[UNCERTAIN: 15,000 / 50,000]" in text
    # The application must not quietly pick one of them.
    assert "about 50,000" not in text


def test_the_text_is_wrapped_to_the_width_asked_for():
    transcript = Transcript(
        recording_name="talk.m4a",
        tokens=[_word(word, float(index)) for index, word in enumerate(
            "the quick brown fox jumps over the lazy dog and then does it again "
            "twice more for good measure".split()
        )],
        speakers=[Speaker(id="0", name="Adrienne")],
    )

    text = render_plain_text(transcript, width=40)

    assert max(len(line) for line in text.splitlines()) <= 40
    assert "the quick brown fox" in text


def test_a_speaker_nobody_named_is_still_labelled():
    transcript = Transcript(
        recording_name="talk.m4a",
        tokens=[_word("hello", 1.0, speaker="7")],
    )

    assert "[0:01] Speaker 7:" in render_plain_text(transcript)


def test_a_word_with_no_speaker_at_all_is_labelled_honestly():
    transcript = Transcript(
        recording_name="talk.m4a",
        tokens=[_word("hello", 1.0, speaker=None)],
    )

    assert "Unknown speaker:" in render_plain_text(transcript)


def test_a_recording_that_produced_no_words_says_so():
    text = render_plain_text(Transcript(recording_name="silence.m4a"))

    assert "no words" in text
    assert "Services: none" in text


def test_a_time_is_written_as_a_clock_rather_than_as_seconds():
    assert format_timestamp(724.0) == "12:04"
    assert format_timestamp(3802.0) == "1:03:22"
    assert format_timestamp(None) == "--:--"


def test_a_word_with_no_timing_falls_back_to_the_span_it_sits_in():
    """An unaligned word is exactly the one a person needs to be able to find."""
    transcript = Transcript(
        recording_name="talk.m4a",
        tokens=[
            FinalToken(
                text="premises",
                speaker="0",
                source_audio_span=AudioSpan(400.0, 404.0),
            )
        ],
        speakers=[Speaker(id="0", name="Adrienne")],
    )

    assert "[6:40] Adrienne:" in render_plain_text(transcript)


# -- The review report ----------------------------------------------------


def test_the_report_is_named_after_the_recording():
    report = render_review_report(_settled_transcript())

    assert report.startswith("# Review report: board-meeting.m4a")


def test_the_report_says_plainly_when_nothing_needs_a_person():
    report = render_review_report(_settled_transcript())

    assert "## What still needs you" in report
    assert "Nothing." in report
    assert "no word in this transcript was flagged" in report.lower()


def test_the_report_still_explains_itself_when_there_is_nothing_to_review():
    """The parts that make the report trustworthy are not the exceptions."""
    report = render_review_report(_settled_transcript())

    assert "## Which services answered" in report
    assert "ElevenLabs Scribe" in report
    assert "scribe-v2 (2026-05-01)" in report
    assert "## How much of this can be trusted" in report
    assert "High confidence" in report


def test_the_report_explains_what_the_confidence_categories_mean():
    report = render_review_report(_settled_transcript())

    for category in Confidence:
        assert category.display_name in report
    assert "cannot be compared" in report


def test_the_report_counts_the_words_in_each_category():
    report = render_review_report(_contested_transcript())

    assert "| High confidence | 1 |" in report
    assert "| Unresolved | 1 |" in report
    assert "| Review required | 1 |" in report
    assert "| Review suggested | 1 |" in report


def test_the_report_says_which_service_did_not_answer_and_why():
    report = render_review_report(_contested_transcript())

    assert "Microsoft MAI" in report
    assert "did not answer within the timeout" in report
    assert "2 of the 3 services asked came back" in report


def test_the_report_says_how_long_the_run_took():
    report = render_review_report(_contested_transcript())

    assert "3 requests were made in all" in report
    assert "47 seconds" in report
    assert "4 minutes" in report


def test_the_report_shows_an_estimated_cost_when_the_caller_knows_one():
    report = render_review_report(_settled_transcript(), estimated_cost=0.4237)

    assert "The estimated cost of this run is 0.42 USD" in report


def test_the_report_takes_the_estimate_the_costing_module_worked_out():
    """Including which services it could not price, which a bare total hides."""
    estimate = estimate_cost(
        duration_seconds=600.0,
        providers=[Provider.ELEVENLABS, Provider.MICROSOFT],
        rates={Provider.ELEVENLABS: 0.02},
        currency="ZAR",
    )

    report = render_review_report(_settled_transcript(), estimated_cost=estimate)

    assert "0.20 ZAR" in report
    assert "It leaves out Microsoft MAI, which has no rate set in Settings" in report


def test_the_report_finds_a_cost_recorded_with_the_run():
    transcript = _settled_transcript()
    transcript.effective_settings = {"estimated_cost_usd": 1.5}

    assert "1.50 USD" in render_review_report(transcript)


def test_the_report_says_nothing_about_cost_when_there_is_none():
    assert "estimated cost" not in render_review_report(_settled_transcript())


def test_the_report_lists_every_span_that_still_needs_review():
    report = render_review_report(_contested_transcript())

    assert "3 places need a decision" in report
    # Each one, at a time a person can find in the recording.
    assert "| 3:03 |" in report
    assert "| 1:04:23 |" in report
    assert "| 6:40 |" in report


def test_each_span_gives_the_reason_in_plain_words():
    report = render_review_report(_contested_transcript())

    assert ReviewReason.NUMERIC_DISAGREEMENT.display_name in report
    assert ReviewReason.PROPER_NAME_DISAGREEMENT.display_name in report
    assert ReviewReason.UNALIGNED_WORD.display_name in report


def test_each_span_says_what_the_services_heard_and_what_was_chosen():
    report = render_review_report(_contested_transcript())

    assert "ElevenLabs Scribe: 15,000" in report
    assert "OpenAI: 50,000" in report
    assert "Microsoft MAI: fifty thousand (rejected)" in report
    assert "[UNCERTAIN: 15,000 / 50,000]" in report


def test_the_report_warns_about_values_that_must_not_be_guessed():
    report = render_review_report(_contested_transcript())

    assert "One of these is a value that must never be settled by what sounds plausible" in report
    assert "money" in report


def test_counts_of_one_are_written_as_english_rather_than_as_output():
    """A report that says "1 words" reads as something nobody checked."""
    transcript = _settled_transcript()
    del transcript.provider_results[Provider.OPENAI]
    transcript.requests = [
        ProviderRequestRecord(
            provider=Provider.ELEVENLABS,
            model_identifier="scribe-v2",
            processing_seconds=42.0,
        )
    ]

    report = render_review_report(transcript)

    assert "Every service asked came back with words: ElevenLabs Scribe." in report
    assert "1 request was made in all" in report
    assert " 1 words" not in report
    assert "1 places" not in report


def test_the_report_summarises_what_was_escalated():
    report = render_review_report(_contested_transcript())

    assert "## Second opinions and adjudication" in report
    assert "| Escalation | AssemblyAI | 1 |" in report
    # The window is the disputed span with several seconds of context either
    # side, which is the whole point of sending a window rather than a word.
    assert "15 seconds" in report


def test_the_report_says_what_the_language_model_decided():
    report = render_review_report(_contested_transcript())

    assert "asked about 1 word " in report
    assert "Declined: an amount must not be settled on plausibility." in report


def test_a_transcript_with_no_second_opinions_has_no_such_section():
    report = render_review_report(_settled_transcript())

    assert "## Second opinions and adjudication" not in report


def test_the_report_repeats_the_warnings_from_the_run():
    report = render_review_report(_contested_transcript())

    assert "## Things worth knowing" in report
    assert "- Microsoft MAI did not answer, so this run had one fewer opinion." in report


def test_a_vertical_bar_in_the_transcript_does_not_tear_the_table_apart():
    transcript = Transcript(
        recording_name="talk.m4a",
        tokens=[
            FinalToken(
                text="a | b",
                text_confidence=Confidence.REVIEW_REQUIRED,
                start=5.0,
                review_status=ReviewStatus.PENDING,
                review_reasons=[ReviewReason.PROVIDER_DISAGREEMENT],
            )
        ],
    )

    report = render_review_report(transcript)

    assert "a \\| b" in report
    for line in report.splitlines():
        if line.startswith("| 0:05 "):
            assert line.count("|") - line.count("\\|") == 5


def test_a_lot_to_review_is_all_of_it_rather_than_the_first_few():
    transcript = Transcript(
        recording_name="talk.m4a",
        tokens=[
            FinalToken(
                text=f"word{index}",
                text_confidence=Confidence.REVIEW_REQUIRED,
                start=float(index),
                review_status=ReviewStatus.PENDING,
                review_reasons=[ReviewReason.LOW_ACOUSTIC_CONFIDENCE],
            )
            for index in range(120)
        ],
    )

    report = render_review_report(transcript)

    assert "120 places need a decision" in report
    for index in range(120):
        assert f"word{index}" in report


def test_an_empty_transcript_still_produces_a_readable_report():
    report = render_review_report(Transcript(recording_name="silence.m4a"))

    assert "No speech-to-text service was called" in report
    assert "nothing to rate" in report
    assert "Nothing." in report


def test_the_exports_have_names_to_be_saved_under():
    assert TEXT_EXPORT_NAME.endswith(".txt")
    assert REPORT_EXPORT_NAME.endswith(".md")


def test_the_run_is_said_to_be_shorter_than_the_services_only_when_it_was():
    """Local work can outlast the services, and then the reason given is false."""
    transcript = _contested_transcript()  # 47 seconds of service time
    transcript.completed_at = "2026-08-17T10:00:30"
    shorter = render_review_report(transcript)
    transcript.completed_at = "2026-08-17T10:04:00"
    longer = render_review_report(transcript)

    assert "less than the total above" in shorter
    assert "From start to finish the run took 4 minutes." in longer
    assert "less than the total above" not in longer
