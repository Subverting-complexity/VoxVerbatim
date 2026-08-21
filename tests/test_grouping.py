"""Tests for turning a folder of transcripts into review work.

Four things are being held down here, and they are the four things that make
the difference between a feature that saves an afternoon and one that quietly
corrupts a transcript.

The sweep must threshold honestly, which means leaving out every word whose
strength was never worked out rather than reading the absence as a low
number. Grouping must be generous enough to gather three spellings of one
surname and not so generous that it gathers two different words. Re-matching
must refuse rather than guess, because a decision applied to the wrong word is
invisible. And reprocessing must be safe to run fifty times, because somebody
exploring a threshold will run it fifty times.
"""

from __future__ import annotations

import gc
import weakref
from dataclasses import replace

from audio_transcriber.transcription.grouping import (
    OCCURRENCE_TIME_TOLERANCE,
    SIMILARITY_THRESHOLD,
    affected_summary,
    apply_rules,
    group_occurrences,
    note_rule_applied,
    occurrences_in,
    rematch,
    reprocess,
)
from audio_transcriber.transcription.model import (
    FinalToken,
    Language,
    Transcript,
)
from audio_transcriber.transcription.project import (
    FlaggedItem,
    Occurrence,
    ProjectSettings,
    ProjectState,
    ReplacementRule,
    WordGroup,
)

RECORDING = "Interview 01.m4a"


def make_token(
    text: str,
    strength: float | None = 0.40,
    start: float | None = 1.0,
    language: Language = Language.ENGLISH,
    **changes,
) -> FinalToken:
    """One finished word, with only the fields this feature reads filled in."""
    end = None if start is None else start + 0.4
    return FinalToken(
        text=text,
        normalised_text=text.lower(),
        confidence_strength=strength,
        start=start,
        end=end if changes.pop("end", None) is None else None,
        language=language,
        **changes,
    )


def make_transcript(words: list[FinalToken], recording_name: str = RECORDING) -> Transcript:
    return Transcript(recording_name=recording_name, tokens=list(words))


def spoken(
    texts: list[str],
    strengths: list[float | None] | None = None,
    first_start: float = 1.0,
    step: float = 0.5,
) -> list[FinalToken]:
    """A little run of words, one after another, at half-second intervals."""
    if strengths is None:
        strengths = [0.40] * len(texts)
    return [
        make_token(text, strength=strength, start=first_start + position * step)
        for position, (text, strength) in enumerate(zip(texts, strengths))
    ]


def named(occurrences: list[Occurrence], text: str) -> Occurrence:
    """The one occurrence with this detected form, for a test that expects one."""
    return next(item for item in occurrences if item.detected_text == text)


def make_occurrence(**changes) -> Occurrence:
    """A saved occurrence with everything filled in, as the tests need it."""
    fields = {
        "id": "occurrence-1",
        "recording_name": RECORDING,
        "token_id": "token-1",
        "detected_text": "Bosch",
        "normalised_text": "bosch",
        "start": 10.0,
        "end": 10.4,
        "confidence_strength": 0.42,
        "language": "en",
        "context_before": "spoke to mister",
        "context_after": "about the report",
    }
    fields.update(changes)
    return Occurrence(**fields)


# -- The sweep ------------------------------------------------------------


def test_the_sweep_takes_the_words_below_the_threshold_and_leaves_the_rest():
    transcript = make_transcript(spoken(["Bosch", "signed", "the"], [0.42, 0.90, 0.54]))

    found = occurrences_in(transcript, RECORDING, 0.55)

    assert [item.detected_text for item in found] == ["Bosch", "the"]
    assert [item.normalised_text for item in found] == ["bosch", "the"]
    assert found[0].recording_name == RECORDING
    assert found[0].start == 1.0


def test_a_word_exactly_on_the_threshold_is_not_weak_enough():
    """The boundary belongs to the stronger side, so the count is predictable."""
    transcript = make_transcript(spoken(["Bosch"], [0.55]))

    assert occurrences_in(transcript, RECORDING, 0.55) == []


def test_a_word_with_no_strength_is_left_out_of_the_sweep_entirely():
    """It cannot be thresholded honestly, so it is not thresholded at all.

    Words like this reach the person by the other road, as existing
    uncertainty items with a stated reason. Reading the absence as a low
    number would put them in front of the person twice.
    """
    transcript = make_transcript(spoken(["Bosch", "ledger"], [None, 0.42]))

    found = occurrences_in(transcript, RECORDING, 0.55)

    assert [item.detected_text for item in found] == ["ledger"]


def test_a_word_with_no_strength_stays_out_however_high_the_threshold_goes():
    transcript = make_transcript(spoken(["Bosch"], [None]))

    assert occurrences_in(transcript, RECORDING, 1.0) == []


def test_punctuation_is_never_an_occurrence():
    transcript = make_transcript(spoken(["Bosch", ",", "signed"], [0.42, 0.10, 0.42]))

    found = occurrences_in(transcript, RECORDING, 0.55)

    assert [item.detected_text for item in found] == ["Bosch", "signed"]


def test_an_occurrence_remembers_the_words_around_it():
    transcript = make_transcript(spoken(["I", "spoke", "to", "Bosch", "about", "it"]))

    found = occurrences_in(transcript, RECORDING, 0.55)
    bosch = named(found, "Bosch")

    assert bosch.context_before == "I spoke to"
    assert bosch.context_after == "about it"


def test_the_detected_text_is_what_the_services_said_not_what_it_now_says():
    """Decision 7 rewrites an occurrence from what it originally said."""
    token = make_token("Bosch", strength=0.42)
    token.original_text = "Bosh"
    transcript = make_transcript([token])

    found = occurrences_in(transcript, RECORDING, 0.55)

    assert found[0].detected_text == "Bosh"
    assert found[0].normalised_text == "bosh"


# -- Gathering them into groups -------------------------------------------


def group_texts(groups, occurrences) -> list[set[str]]:
    """The detected forms in each group, as a set, for comparing easily."""
    by_id = {item.id: item for item in occurrences}
    return [{by_id[item].detected_text for item in group.occurrence_ids} for group in groups]


def occurrences_from(
    texts: list[str],
    strengths: list[float] | None = None,
    language: str = "en",
) -> list[Occurrence]:
    """Saved occurrences straight from a list of forms, for grouping tests."""
    if strengths is None:
        strengths = [0.42] * len(texts)
    return [
        make_occurrence(
            id=f"occurrence-{position}",
            token_id=f"token-{position}",
            detected_text=text,
            normalised_text=text.lower(),
            confidence_strength=strength,
            language=language,
            start=float(position),
        )
        for position, (text, strength) in enumerate(zip(texts, strengths))
    ]


def test_three_spellings_of_one_surname_are_gathered_together():
    """The case the whole feature was asked for.

    "Bosh" and "Bosche" are not similar enough to be joined to each other
    directly. They meet through "Bosch", which is why the clustering links
    rather than measuring everything against one centre.
    """
    occurrences = occurrences_from(["Bosch", "Bosh", "Bosche"])

    groups = group_occurrences(occurrences, 0.05)

    assert len(groups) == 1
    assert group_texts(groups, occurrences) == [{"Bosch", "Bosh", "Bosche"}]


def test_two_genuinely_different_words_stay_apart():
    occurrences = occurrences_from(["Bosch", "ledger"])

    groups = group_occurrences(occurrences, 0.05)

    assert len(groups) == 2
    assert group_texts(groups, occurrences) == [{"Bosch"}, {"ledger"}]


def test_a_word_on_its_own_still_gets_a_group():
    """Every weak word has to be reachable in the list of groups.

    A word the analysis found no company for is a group of one rather than
    something with no home, because a person who cannot select it cannot play
    it, correct it or mark it right.
    """
    occurrences = occurrences_from(["Bosch"])

    groups = group_occurrences(occurrences, 0.05)

    assert len(groups) == 1
    assert groups[0].occurrence_ids == ["occurrence-0"]
    assert groups[0].user_created is False


def test_case_and_punctuation_never_split_a_group():
    occurrences = occurrences_from(["Bosch", "bosch,", "BOSCH"])

    groups = group_occurrences(occurrences, 0.05)

    assert len(groups) == 1


def test_a_dropped_umlaut_is_recognised_by_the_existing_equivalence():
    occurrences = occurrences_from(["Jürgen", "Jurgen"])

    groups = group_occurrences(occurrences, 0.05)

    assert len(groups) == 1


def test_a_gap_in_confidence_splits_two_forms_that_are_only_similar():
    occurrences = occurrences_from(["Bosch", "Bosche"], [0.20, 0.50])

    groups = group_occurrences(occurrences, 0.05)

    assert len(groups) == 2


def test_a_gap_in_confidence_does_not_split_an_exact_match():
    """Two spellings of one surname at 42 and 61 percent are one surname."""
    occurrences = occurrences_from(["Bosch", "bosch"], [0.42, 0.61])

    groups = group_occurrences(occurrences, 0.05)

    assert len(groups) == 1
    assert group_texts(groups, occurrences) == [{"Bosch", "bosch"}]


def test_a_confidence_nobody_measured_never_splits_a_group():
    occurrences = occurrences_from(["Bosch", "Bosche"], [0.42, 0.42])
    occurrences[1].confidence_strength = None

    groups = group_occurrences(occurrences, 0.05)

    assert len(groups) == 1


def test_two_known_languages_never_group_however_alike_the_text():
    english = occurrences_from(["Bosch"], language="en")
    german = occurrences_from(["Bosche"], language="de")
    german[0].id = "occurrence-german"

    groups = group_occurrences(english + german, 0.05)

    assert len(groups) == 2


def test_an_unknown_language_joins_but_does_not_bridge():
    """An unknown language is missing information, not a bridge between two.

    Letting the middle word join both would put an English word and a German
    word in one group by way of a word nobody could place.
    """
    occurrences = (
        occurrences_from(["Bosch"], language="en")
        + [replace(occurrences_from(["Bosche"], language="unknown")[0], id="occurrence-unknown")]
        + [replace(occurrences_from(["Boschen"], language="de")[0], id="occurrence-german")]
    )

    groups = group_occurrences(occurrences, 0.05)

    assert len(groups) == 2
    assert {"Bosch", "Bosche"} in group_texts(groups, occurrences)
    assert {"Boschen"} in group_texts(groups, occurrences)


def test_the_representative_text_is_the_commonest_detected_form():
    occurrences = occurrences_from(["Bosh", "Bosch", "Bosch"])

    groups = group_occurrences(occurrences, 0.05)

    assert groups[0].representative_text == "Bosch"


def test_the_representative_text_breaks_a_tie_on_the_weakest_form():
    """Where two forms occur equally often, the shakier one is shown."""
    occurrences = occurrences_from(["Bosch", "Bosh"], [0.50, 0.47])

    groups = group_occurrences(occurrences, 0.05)

    assert groups[0].representative_text == "Bosh"


def test_a_form_nobody_measured_does_not_win_the_tie():
    occurrences = occurrences_from(["Bosch", "Bosh"], [0.50, 0.50])
    occurrences[1].confidence_strength = None

    groups = group_occurrences(occurrences, 0.05)

    assert groups[0].representative_text == "Bosch"


def test_the_similarity_threshold_is_where_the_module_says_it_is():
    """A guard on the constant, so moving it is a deliberate act.

    "Bosh" against "Bosche" scores 0.80 and is deliberately below the line;
    the pair reaches one group through "Bosch" instead.
    """
    from difflib import SequenceMatcher

    assert SequenceMatcher(None, "bosch", "bosh").ratio() > SIMILARITY_THRESHOLD
    assert SequenceMatcher(None, "bosch", "bosche").ratio() > SIMILARITY_THRESHOLD
    assert SequenceMatcher(None, "bosh", "bosche").ratio() < SIMILARITY_THRESHOLD


def test_short_words_are_not_guessed_at():
    """One letter out of three is a third of a word, so similarity is not used."""
    occurrences = occurrences_from(["the", "they"])

    groups = group_occurrences(occurrences, 0.05)

    assert len(groups) == 2


# -- Keeping a decision across a re-transcription -------------------------


def test_a_decision_survives_a_transcript_being_regenerated():
    saved = make_occurrence(
        token_id="old-token",
        start=10.0,
        context_before="spoke to mister",
        context_after="about the report",
        reviewed=True,
        replacement="Bosch",
    )
    words = spoken(
        ["spoke", "to", "mister", "Bosch", "about", "the", "report"],
        first_start=8.5,
    )
    transcript = make_transcript(words)

    matched, missing = rematch([saved], transcript, RECORDING)

    assert missing == []
    assert matched[0].token_id == words[3].id
    assert matched[0].stale is False
    # Everything the person decided comes through untouched.
    assert matched[0].reviewed is True
    assert matched[0].replacement == "Bosch"
    # What describes the word is refreshed from the new transcript.
    assert matched[0].start == words[3].start
    assert matched[0].context_before == "spoke to mister"


def test_a_word_that_is_no_longer_there_comes_back_stale():
    """Never attached to the nearest available word, whatever it costs."""
    saved = make_occurrence(reviewed=True, replacement="Bosch")
    transcript = make_transcript(spoken(["spoke", "to", "mister", "Smith"], first_start=9.0))

    matched, missing = rematch([saved], transcript, RECORDING)

    assert matched == []
    assert missing[0].stale is True
    assert missing[0].reviewed is True
    assert missing[0].replacement == "Bosch"


def test_a_word_that_has_moved_too_far_comes_back_stale():
    saved = make_occurrence(start=10.0)
    far = OCCURRENCE_TIME_TOLERANCE + 0.5
    transcript = make_transcript(spoken(["Bosch"], first_start=10.0 + far))

    matched, missing = rematch([saved], transcript, RECORDING)

    assert matched == []
    assert missing[0].stale is True


def test_the_surrounding_words_decide_between_two_equally_close_candidates():
    words = [
        make_token("mister", start=9.5),
        make_token("Bosch", start=9.9),
        make_token("and", start=10.0),
        make_token("Bosch", start=10.1),
        make_token("Smith", start=10.3),
    ]
    saved = make_occurrence(start=10.0, context_before="and", context_after="Smith")

    matched, missing = rematch([saved], make_transcript(words), RECORDING)

    assert missing == []
    assert matched[0].token_id == words[3].id


def test_two_candidates_the_context_cannot_separate_come_back_stale():
    """A tie is answered with a refusal, never with the first of them."""
    words = [
        make_token("the", start=9.8),
        make_token("Bosch", start=9.9),
        make_token("file", start=9.95),
        make_token("the", start=10.0),
        make_token("Bosch", start=10.1),
        make_token("file", start=10.2),
    ]
    saved = make_occurrence(start=10.0, context_before="the", context_after="file")

    matched, missing = rematch([saved], make_transcript(words), RECORDING)

    assert matched == []
    assert missing[0].stale is True


def test_two_occurrences_never_land_on_the_same_word():
    words = spoken(["Bosch", "and", "Bosch"], first_start=10.0, step=0.3)
    first = make_occurrence(id="occurrence-1", start=10.0, context_before="", context_after="and")
    second = make_occurrence(id="occurrence-2", start=10.6, context_before="and", context_after="")

    matched, missing = rematch([first, second], make_transcript(words), RECORDING)

    assert missing == []
    assert {item.token_id for item in matched} == {words[0].id, words[2].id}


def test_an_occurrence_from_another_recording_is_handed_straight_back():
    saved = make_occurrence(recording_name="Interview 02.m4a")
    transcript = make_transcript(spoken(["Bosch"], first_start=10.0))

    matched, missing = rematch([saved], transcript, RECORDING)

    assert matched == []
    # Not stale: this transcript has nothing to say about it either way.
    assert missing[0].stale is False


def test_an_already_corrected_word_is_still_found_by_what_it_used_to_say():
    """Reprocessing the same file must not mark every correction stale."""
    token = make_token("Bosch", start=10.0)
    token.original_text = "Bosh"
    token.human_corrected = True
    token.text_corrected = True
    token.confidence_strength = 1.0
    saved = make_occurrence(
        token_id="a-forgotten-identifier",
        detected_text="Bosh",
        normalised_text="bosh",
        confidence_strength=0.42,
        reviewed=True,
        replacement="Bosch",
    )

    matched, missing = rematch([saved], make_transcript([token]), RECORDING)

    assert missing == []
    assert matched[0].token_id == token.id
    assert matched[0].detected_text == "Bosh"
    # The 1.0 a correction confers says nothing about the word the person saw.
    assert matched[0].confidence_strength == 0.42


def test_confirming_a_clock_does_not_freeze_the_word_at_its_old_strength():
    """A timing decision is not a text replacement, and must not read as one.

    ``human_corrected`` means "a person decided something about this word",
    and confirming a timing sets it. The strength is only kept from the saved
    occurrence where the *text* was replaced, because that is the only case in
    which the token's own strength is the 1.0 a correction confers rather than
    a measurement. A word whose clock somebody confirmed has never been
    rewritten, so its strength is a real figure and the transcript's is the
    current one.
    """
    token = make_token("Bosch", start=10.0, strength=0.30)
    token.human_corrected = True
    saved = make_occurrence(confidence_strength=0.42, token_id="a-forgotten-identifier")

    matched, _missing = rematch([saved], make_transcript([token]), RECORDING)

    assert matched[0].confidence_strength == 0.30


# -- Answering a new file with what the project already knows -------------


def project_with_rule(**changes) -> ProjectState:
    fields = {
        "id": "rule-1",
        "matched_text": "Bosh",
        "normalised_text": "bosh",
        "replacement": "Bosch",
        "language": "en",
        "created_at": "2026-08-17T09:00:00",
    }
    fields.update(changes)
    return ProjectState(settings=ProjectSettings(), rules=[ReplacementRule(**fields)])


def test_a_rule_answers_a_newly_transcribed_file():
    state = project_with_rule()
    words = spoken(["mister", "Bosh", "signed"], [0.90, 0.42, 0.90])
    transcript = make_transcript(words)

    corrected, answered, queued = apply_rules(state, transcript, RECORDING)

    assert [item.detected_text for item in answered] == ["Bosh"]
    assert answered[0].reviewed is True
    assert answered[0].auto_applied is True
    assert answered[0].applied_rule_id == "rule-1"
    assert answered[0].replacement == "Bosch"
    assert queued == []
    token = corrected.token_by_id(words[1].id)
    assert token.text == "Bosch"
    assert token.original_text == "Bosh"
    assert state.rules[0].occurrence_count == 1
    # The transcript that went in is not the transcript that came out.
    assert transcript.token_by_id(words[1].id).text == "Bosh"


def test_a_form_nobody_has_accepted_falls_into_the_queue_instead():
    """Grouping may reach for "Bosche"; a rule may not."""
    state = project_with_rule()
    transcript = make_transcript(spoken(["Bosche", "signed"], [0.42, 0.90]))

    corrected, answered, queued = apply_rules(state, transcript, RECORDING)

    assert answered == []
    assert [item.detected_text for item in queued] == ["Bosche"]
    assert corrected.tokens[0].text == "Bosche"


def test_a_rule_in_another_language_does_not_fire():
    state = project_with_rule(language="de")
    transcript = make_transcript(
        [make_token("Bosh", strength=0.42, language=Language.ENGLISH)]
    )

    _corrected, answered, queued = apply_rules(state, transcript, RECORDING)

    assert answered == []
    assert [item.detected_text for item in queued] == ["Bosh"]


def test_a_rule_fires_on_a_word_the_services_were_sure_of():
    """The words a service gets confidently wrong are exactly the proper names."""
    state = project_with_rule()
    transcript = make_transcript(spoken(["Bosh"], [0.95]))

    corrected, answered, queued = apply_rules(state, transcript, RECORDING)

    assert [item.detected_text for item in answered] == ["Bosh"]
    assert queued == []
    assert corrected.tokens[0].text == "Bosch"


def test_a_rule_never_overrules_a_correction_made_in_this_file():
    state = project_with_rule()
    token = make_token("Bosh", strength=1.0)
    token.human_corrected = True
    token.text_corrected = True
    transcript = make_transcript([token])

    corrected, answered, queued = apply_rules(state, transcript, RECORDING)

    assert answered == []
    assert queued == []
    assert corrected.tokens[0].text == "Bosh"


def test_confirming_a_clock_does_not_exempt_a_word_from_every_rule_for_ever():
    """The case decision 10 exists for, defeated by an unrelated decision.

    Somebody confirms the timing of the word ``Bosh``, which sets
    ``human_corrected`` and says nothing whatever about the spelling. A rule
    answering ``Bosh`` is accepted later. The rule must still fire: what a
    rule is forbidden to overrule is a *replacement* a person typed into this
    transcript, and no replacement was typed here.
    """
    state = project_with_rule()
    token = make_token("Bosh", strength=0.42)
    token.human_corrected = True
    transcript = make_transcript([token])

    corrected, answered, _queued = apply_rules(state, transcript, RECORDING)

    assert [item.detected_text for item in answered] == ["Bosh"]
    assert corrected.tokens[0].text == "Bosch"


def test_a_confirmed_clock_does_not_hide_a_weak_word_from_the_queue():
    """The same test read the other way, where no rule answers the word.

    Skipping the word outright dropped it from the queue as well, so a weak
    word whose timing somebody had confirmed reached the person by neither
    road: no rule could correct it and no queue could show it.
    """
    state = project_with_rule()
    token = make_token("Bosche", strength=0.42)
    token.human_corrected = True
    transcript = make_transcript([token])

    _corrected, answered, queued = apply_rules(state, transcript, RECORDING)

    assert answered == []
    assert [item.detected_text for item in queued] == ["Bosche"]


def test_a_word_that_already_reads_as_the_replacement_is_left_alone():
    state = project_with_rule(matched_text="Bosch", normalised_text="bosch")
    transcript = make_transcript(spoken(["Bosch"], [0.90]))

    _corrected, answered, queued = apply_rules(state, transcript, RECORDING)

    assert answered == []
    assert queued == []
    assert state.rules[0].occurrence_count == 0


def test_the_tally_is_kept_in_one_place_by_one_function():
    """Two places write a rule's correction, and only one used to count it.

    Whether the number the person is shown was right therefore depended on
    which path a particular word happened to travel by, which is worse than
    no number at all because nothing about it looks wrong.
    """
    state = project_with_rule()
    state.rules.append(
        ReplacementRule(
            id="rule-2",
            matched_text="Bosche",
            normalised_text="bosche",
            replacement="Bosch",
            language="en",
            created_at="2026-08-17T09:00:00",
        )
    )

    note_rule_applied(state, "rule-1")
    note_rule_applied(state, "rule-1")

    assert state.rules[0].occurrence_count == 2
    assert state.rules[1].occurrence_count == 0


def test_counting_against_a_rule_that_has_gone_is_quietly_ignored():
    """A person may delete a rule while its correction is still being written.

    Refusing at that moment would turn a tidy-up into a failure, and nothing
    is at stake: the correction is in the transcript either way and the count
    is only a tally.
    """
    state = project_with_rule()

    note_rule_applied(state, "a rule this project has never held")

    assert state.rules[0].occurrence_count == 0


# -- Running the whole analysis again -------------------------------------


def folder(transcripts: dict[str, Transcript]) -> tuple[list[str], object]:
    """The names and the loader ``reprocess`` takes, from a plain dictionary.

    A test may hold all of its transcripts at once because they are five words
    long. The application may not, which is what the loader is for, and the
    test below on reading one recording at a time is what holds that down.
    """
    return list(transcripts), transcripts.get


def worked_project() -> tuple[ProjectState, dict[str, Transcript]]:
    """A folder somebody has already spent an afternoon on.

    Every kind of manual work is represented exactly once: a replacement, a
    reviewed mark, a correct-as-detected decision, an isolated occurrence and
    a group the person made by hand.
    """
    words = spoken(
        ["Bosch", "Bosh", "Bosche", "ledger", "invoice", "receipt", "quiet"],
        [0.42, 0.45, 0.44, 0.30, 0.50, 0.48, 0.95],
    )
    transcript = make_transcript(words)
    transcripts = {RECORDING: transcript}
    state = reprocess(ProjectState(settings=ProjectSettings()), *folder(transcripts))

    by_text = {item.detected_text: item for item in state.occurrences}
    by_text["Bosch"].replacement = "Bosch AG"
    by_text["Bosh"].reviewed = True
    by_text["ledger"].correct_as_detected = True
    by_text["Bosche"].isolated = True
    state.groups = [
        group for group in state.groups if by_text["Bosche"].id not in group.occurrence_ids
    ]
    for group in state.groups:
        group.occurrence_ids = [
            item for item in group.occurrence_ids if item != by_text["Bosche"].id
        ]
    state.groups = [group for group in state.groups if group.occurrence_ids]
    state.groups.append(
        WordGroup(
            id="group-by-hand",
            representative_text="invoice",
            occurrence_ids=[by_text["invoice"].id],
            user_created=True,
        )
    )
    return state, transcripts


def test_the_sweep_finds_the_weak_words_of_a_folder():
    state, _transcripts = worked_project()

    assert {item.detected_text for item in state.occurrences} == {
        "Bosch",
        "Bosh",
        "Bosche",
        "ledger",
        "invoice",
        "receipt",
    }
    assert state.processed_at != ""


def test_reprocessing_three_times_keeps_every_manual_decision():
    """Somebody exploring a threshold will do this, so it has to be safe.

    The thresholds deliberately move in both directions: 0.35 puts most of
    these words out of scope entirely, 0.60 brings them all back, and 0.45
    lands in between. None of it may touch what the person decided.
    """
    state, transcripts = worked_project()
    decided = {
        item.detected_text: item.id
        for item in state.occurrences
        if item.detected_text in ("Bosch", "Bosh", "Bosche", "ledger", "invoice")
    }

    for threshold in (0.35, 0.60, 0.45):
        state.settings = replace(state.settings, minimum_confidence=threshold)
        state = reprocess(state, *folder(transcripts))

        surviving = {item.id: item for item in state.occurrences}
        for text, identifier in decided.items():
            assert identifier in surviving, f"{text} was lost at {threshold}"

        assert surviving[decided["Bosch"]].replacement == "Bosch AG"
        assert surviving[decided["Bosh"]].reviewed is True
        assert surviving[decided["ledger"]].correct_as_detected is True

        isolated = surviving[decided["Bosche"]]
        assert isolated.isolated is True
        assert state.group_of(isolated.id) is None

        by_hand = state.group("group-by-hand")
        assert by_hand is not None
        assert by_hand.user_created is True
        assert by_hand.occurrence_ids == [decided["invoice"]]


def test_an_isolated_word_is_never_gathered_up_again():
    """Pulling a word out was a statement; rerunning is not new evidence."""
    state, transcripts = worked_project()
    isolated = next(item for item in state.occurrences if item.isolated)

    for _ in range(3):
        state = reprocess(state, *folder(transcripts))
        found = state.occurrence(isolated.id)
        assert found is not None
        assert found.isolated is True
        assert state.group_of(isolated.id) is None


def test_an_automatic_group_that_was_settled_is_kept_whole():
    state, transcripts = worked_project()
    group = next(item for item in state.groups if not item.user_created)
    group.replacement = "Bosch AG"
    group.reviewed = True
    members = list(group.occurrence_ids)

    state = reprocess(state, *folder(transcripts))

    kept = state.group(group.id)
    assert kept is not None
    assert kept.replacement == "Bosch AG"
    assert kept.occurrence_ids == members


def test_an_untouched_word_that_is_no_longer_weak_is_dropped():
    """Automatic groups are rebuilt around whatever the person has not touched."""
    state, transcripts = worked_project()
    untouched = named(state.occurrences, "receipt")

    state.settings = replace(state.settings, minimum_confidence=0.05)
    state = reprocess(state, *folder(transcripts))

    assert state.occurrence(untouched.id) is None


def test_a_recording_that_is_not_loaded_is_left_exactly_as_it_was():
    """An absent file is not evidence that a word has gone."""
    state, transcripts = worked_project()
    elsewhere = make_occurrence(id="occurrence-elsewhere", recording_name="Interview 02.m4a")
    elsewhere.reviewed = True
    state.occurrences.append(elsewhere)

    state = reprocess(state, *folder(transcripts))

    found = state.occurrence("occurrence-elsewhere")
    assert found is not None
    assert found.stale is False
    assert found.reviewed is True


def test_the_flagged_words_survive_the_analysis_running_again():
    """Reprocessing rebuilds the project, so everything it forgets is lost.

    The flagged words are the second block of the first list. Dropping them
    silently gave the caller an empty list rather than an error, which is the
    kind of failure nobody notices until somebody asks where half the review
    went.
    """
    state, transcripts = worked_project()
    state.flagged = [
        FlaggedItem(recording_name=RECORDING, token_id="token-flagged", text="15,000")
    ]

    state = reprocess(state, *folder(transcripts))

    assert [item.token_id for item in state.flagged] == ["token-flagged"]


def test_a_recording_that_has_left_the_folder_takes_its_occurrences_with_it():
    """A file the person deleted is not a file that could not be read.

    Both look the same to the loader, which answers with nothing in either
    case, so the folder listing is what tells them apart. Left in, the words
    of a deleted recording go on being counted in the sentence that says how
    far a replacement reaches, which is read out to somebody who cannot see
    the list and cannot check it.
    """
    state, transcripts = worked_project()
    gone = make_occurrence(id="occurrence-gone", recording_name="Interview 02.m4a")
    gone.reviewed = True
    state.occurrences.append(gone)
    state.flagged = [
        FlaggedItem(recording_name="Interview 02.m4a", token_id="token-gone", text="15,000")
    ]

    state = reprocess(state, *folder(transcripts), present_recordings=[RECORDING])

    assert state.occurrence("occurrence-gone") is None
    assert state.flagged == []


def test_reprocessing_keeps_what_each_transcript_was_when_it_was_read():
    """The analysis builds a fresh project, and this must survive the rebuild.

    What each transcript said its own age was is how the window decides which
    recordings it can skip on the next opening. Rebuilding the project without
    it loses no words and looks harmless, and the folder is then read from end
    to end every time it is opened, which is most of a minute of somebody
    waiting for a window. The one entry that does go is the one belonging to a
    recording that has left the folder, which is the same rule the occurrences
    and the flagged words follow.
    """
    state, transcripts = worked_project()
    state.transcript_times = {RECORDING: 1755421500123456789, "Interview 02.m4a": 17554215004}

    state = reprocess(state, *folder(transcripts), present_recordings=[RECORDING])

    assert state.transcript_times == {RECORDING: 1755421500123456789}


def test_a_deleted_recording_never_inflates_the_sentence_read_out():
    """The defect exactly as it was reproduced: a group of 3 across 2 files."""
    words = spoken(["Bosch", "Bosh"], [0.42, 0.45])
    other = make_transcript(spoken(["Bosche"], [0.44]), "Interview 02.m4a")
    transcripts = {RECORDING: make_transcript(words), "Interview 02.m4a": other}
    state = reprocess(ProjectState(settings=ProjectSettings()), *folder(transcripts))
    group = next(item for item in state.groups if len(item.occurrence_ids) == 3)
    assert affected_summary(state, group.id) == (
        "This replacement applies to 3 occurrences across 2 files."
    )

    del transcripts["Interview 02.m4a"]
    state = reprocess(state, *folder(transcripts), present_recordings=[RECORDING])

    remaining = next(item for item in state.groups if len(item.occurrence_ids) > 1)
    assert affected_summary(state, remaining.id) == (
        "This replacement applies to 2 occurrences in 1 file."
    )
    assert all(item.recording_name == RECORDING for item in state.occurrences)


def test_a_recording_that_could_not_be_read_is_never_treated_as_deleted():
    """The distinction, tested from the other side.

    A busy file or a disconnected drive is still in the folder listing and
    still unreadable, and the person's decisions about it must survive
    untouched. This is the case ``reprocess`` was already careful about, and
    pruning must not undo that care.
    """
    state, transcripts = worked_project()
    unreadable = make_occurrence(id="occurrence-busy", recording_name="Interview 02.m4a")
    unreadable.reviewed = True
    state.occurrences.append(unreadable)
    names = [RECORDING, "Interview 02.m4a"]

    state = reprocess(
        state, names, transcripts.get, present_recordings=names
    )

    found = state.occurrence("occurrence-busy")
    assert found is not None
    assert found.reviewed is True
    assert found.stale is False


def test_nothing_is_pruned_when_the_folder_listing_is_not_known():
    """Silence about the folder must never be read as "the folder is empty".

    A caller that cannot enumerate the folder passes nothing, and nothing is
    then thrown away. Destroying somebody's work on the strength of a question
    that was never asked is the worst thing this function could do.
    """
    state, transcripts = worked_project()
    elsewhere = make_occurrence(id="occurrence-elsewhere", recording_name="Interview 02.m4a")
    elsewhere.reviewed = True
    state.occurrences.append(elsewhere)

    state = reprocess(state, *folder(transcripts))

    assert state.occurrence("occurrence-elsewhere") is not None


def test_the_rules_of_a_deleted_recording_are_kept():
    """Where the person's work actually lives once the recording has gone.

    A rule is not about any one recording. It is the durable form of the
    decision, it answers the next file transcribed, and if the deleted
    recording is restored from a backup its words are found again and the
    rule answers them without anybody being asked twice.
    """
    state, transcripts = worked_project()
    state.rules.append(
        ReplacementRule(
            id="rule-1",
            matched_text="Bosh",
            normalised_text="bosh",
            replacement="Bosch",
            language="en",
            created_at="2026-08-17T09:00:00",
            occurrence_count=4,
        )
    )
    state.occurrences.append(
        make_occurrence(id="occurrence-gone", recording_name="Interview 02.m4a")
    )

    state = reprocess(state, *folder(transcripts), present_recordings=[RECORDING])

    assert [rule.id for rule in state.rules] == ["rule-1"]
    assert state.rules[0].occurrence_count == 4


def test_a_new_word_picks_up_a_rule_the_project_already_has():
    state, transcripts = worked_project()
    state.rules.append(
        ReplacementRule(
            id="rule-1",
            matched_text="invoice",
            normalised_text="invoice",
            replacement="Invoice",
            language="en",
            created_at="2026-08-17T09:00:00",
        )
    )
    words = spoken(["invoice", "ledger"], [0.40, 0.40])
    transcripts["Interview 02.m4a"] = make_transcript(words, "Interview 02.m4a")

    state = reprocess(state, *folder(transcripts))

    fresh = [
        item
        for item in state.occurrences
        if item.recording_name == "Interview 02.m4a" and item.detected_text == "invoice"
    ]
    assert len(fresh) == 1
    assert fresh[0].auto_applied is True
    assert fresh[0].reviewed is True
    assert fresh[0].replacement == "Invoice"
    assert fresh[0].applied_rule_id == "rule-1"


def test_reprocessing_does_not_find_the_same_word_twice():
    state, transcripts = worked_project()
    first = len(state.occurrences)

    state = reprocess(state, *folder(transcripts))
    state = reprocess(state, *folder(transcripts))

    assert len(state.occurrences) == first


def test_reprocessing_puts_the_person_back_where_they_were():
    state, transcripts = worked_project()
    target = named(state.occurrences, "Bosh")
    state.last_occurrence_id = target.id
    state.last_group_id = "a group that will not survive"

    state = reprocess(state, *folder(transcripts))

    assert state.last_occurrence_id == target.id
    group = state.group_of(target.id)
    assert group is not None
    assert state.last_group_id == group.id


def test_a_word_that_has_gone_comes_back_stale_and_out_of_every_group():
    state, transcripts = worked_project()
    target = named(state.occurrences, "Bosh")
    target.reviewed = True
    transcripts[RECORDING] = make_transcript(
        spoken(["Bosch", "ledger", "invoice"], [0.42, 0.30, 0.50])
    )

    state = reprocess(state, *folder(transcripts))

    found = state.occurrence(target.id)
    assert found is not None
    assert found.stale is True
    assert found.reviewed is True
    assert state.group_of(target.id) is None


class StreamingLoader:
    """Builds each transcript when it is asked for, and watches the last one go.

    A weak reference is kept to every transcript handed out, and the number of
    them still alive is counted at the start of the next call. A loader that a
    caller has finished with sees nought; a caller holding on to what it has
    read sees the count climb.
    """

    def __init__(self, recordings: dict[str, list[str]]):
        self._recordings = recordings
        self.names = list(recordings)
        self.read: list[str] = []
        self.still_alive: list[int] = []
        self._handed_out: list[weakref.ref] = []

    def __call__(self, recording_name: str) -> Transcript | None:
        gc.collect()
        self.still_alive.append(sum(1 for ref in self._handed_out if ref() is not None))
        self.read.append(recording_name)
        words = self._recordings.get(recording_name)
        if words is None:
            return None
        transcript = make_transcript(spoken(words), recording_name)
        self._handed_out.append(weakref.ref(transcript))
        return transcript


def test_only_one_recording_is_ever_held_at_a_time():
    """The measurement behind the loader, in a form that fails if it is undone.

    Fifty hour-long recordings read eagerly took 37 seconds and about 1.6 GB.
    Somebody will one day find the loader awkward and pass a dictionary
    comprehension into it, which reads them all up front, looks like a
    tidy-up and rebuilds every byte of that. This test is what notices.
    """
    loader = StreamingLoader(
        {f"Interview {number:02d}.m4a": ["Bosch", "ledger", "invoice"] for number in range(1, 6)}
    )

    state = reprocess(ProjectState(settings=ProjectSettings()), loader.names, loader)

    assert loader.read == loader.names
    assert max(loader.still_alive) == 0
    assert len(state.occurrences) == 15
    assert len({item.recording_name for item in state.occurrences}) == 5


def test_a_name_given_twice_is_only_swept_once():
    loader = StreamingLoader({RECORDING: ["Bosch", "ledger"]})

    state = reprocess(ProjectState(settings=ProjectSettings()), [RECORDING, RECORDING], loader)

    assert loader.read == [RECORDING]
    assert len(state.occurrences) == 2


def test_a_recording_that_cannot_be_read_is_absent_rather_than_damaged():
    """A file being written, or a drive not connected, reads again tomorrow."""
    state, transcripts = worked_project()
    unreadable = "Interview 02.m4a"
    state.occurrences.append(
        make_occurrence(id="occurrence-elsewhere", recording_name=unreadable, reviewed=True)
    )
    names, readable = folder(transcripts)

    def load(recording_name: str) -> Transcript | None:
        return None if recording_name == unreadable else readable(recording_name)

    state = reprocess(state, [*names, unreadable], load)

    found = state.occurrence("occurrence-elsewhere")
    assert found is not None
    assert found.stale is False
    assert found.reviewed is True


# -- Saying what a change will do -----------------------------------------


def summary_state(occurrences: list[Occurrence]) -> ProjectState:
    return ProjectState(
        settings=ProjectSettings(),
        groups=[
            WordGroup(
                id="group-1",
                representative_text="Bosch",
                occurrence_ids=[item.id for item in occurrences],
            )
        ],
        occurrences=occurrences,
    )


def test_the_summary_counts_many_occurrences_across_many_files():
    occurrences = [
        make_occurrence(id=f"occurrence-{position}", recording_name=recording)
        for position, recording in enumerate(
            ["Interview 01.m4a"] * 5 + ["Interview 02.m4a"] * 3
        )
    ]

    assert (
        affected_summary(summary_state(occurrences), "group-1")
        == "This replacement applies to 8 occurrences across 2 files."
    )


def test_the_summary_is_correct_for_one_occurrence_and_one_file():
    occurrences = [make_occurrence(id="occurrence-1")]

    assert (
        affected_summary(summary_state(occurrences), "group-1")
        == "This replacement applies to 1 occurrence in 1 file."
    )


def test_the_summary_says_many_occurrences_in_one_file():
    occurrences = [make_occurrence(id=f"occurrence-{position}") for position in range(4)]

    assert (
        affected_summary(summary_state(occurrences), "group-1")
        == "This replacement applies to 4 occurrences in 1 file."
    )


def test_a_group_that_is_gone_is_described_rather_than_answered_with_silence():
    """An empty string is silence to a screen reader, which reads as a failure."""
    assert (
        affected_summary(ProjectState(settings=ProjectSettings()), "group-1")
        == "This replacement applies to no occurrences."
    )
