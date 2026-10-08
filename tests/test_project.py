"""Tests for what a folder of recordings remembers about its own review.

Two things are being checked here more than anything else. The first is that
no state of the file on the disk stops a project loading, because the Review
window has to open even when the file has been edited by hand into nonsense.
The second is that two folders share nothing at all, which is the rule the
whole design exists to keep.
"""

from __future__ import annotations

import json

from vox_verbatim.json_store import JsonReadStatus
from vox_verbatim.transcription.project import (
    DEFAULT_GROUPING_TOLERANCE,
    DEFAULT_MINIMUM_CONFIDENCE,
    PROJECT_FILE_NAME,
    PROJECT_FORMAT_VERSION,
    FlaggedItem,
    LearnedName,
    Occurrence,
    ProjectSettings,
    ProjectState,
    ProjectStore,
    ReplacementRule,
    SpeakerDoubtItem,
    WordGroup,
    merge_learned_names,
    read_learned_names,
    remove_learned_name,
)


def make_occurrence(identifier: str, **changes) -> Occurrence:
    """An occurrence with everything filled in, so tests only say what matters."""
    fields = {
        "id": identifier,
        "recording_name": "Interview 01.m4a",
        "token_id": f"token-{identifier}",
        "detected_text": "Bosch",
        "normalised_text": "bosch",
        "start": 12.5,
        "end": 13.0,
        "confidence_strength": 0.42,
        "language": "en",
        "context_before": "spoke to mister",
        "context_after": "about the report",
    }
    fields.update(changes)
    return Occurrence(**fields)


def rule(identifier: str, normalised_text: str, language: str) -> ReplacementRule:
    """A rule that replaces one normalised form, so tests only say what matters."""
    return ReplacementRule(
        id=identifier,
        matched_text=normalised_text.title(),
        normalised_text=normalised_text,
        replacement="Bosch",
        language=language,
        created_at="2026-08-17T09:00:00",
    )


def write_project(folder, data) -> None:
    (folder / PROJECT_FILE_NAME).write_text(json.dumps(data), encoding="utf-8")


# -- Opening a folder -----------------------------------------------------


def test_a_folder_with_no_file_gives_an_empty_project(tmp_path):
    store = ProjectStore(tmp_path)

    state = store.load()

    assert state == ProjectState()
    assert state.settings.minimum_confidence == DEFAULT_MINIMUM_CONFIDENCE
    assert state.settings.grouping_tolerance == DEFAULT_GROUPING_TOLERANCE
    # Asking about a folder that has never been reviewed must not write in it.
    assert not (tmp_path / PROJECT_FILE_NAME).exists()


def test_the_file_sits_in_the_audio_folder_itself(tmp_path):
    """The state travels with the recordings, so it lives among them."""
    store = ProjectStore(tmp_path)

    assert store.folder == tmp_path
    assert store.path == tmp_path / PROJECT_FILE_NAME
    assert store.path.parent == tmp_path


def test_a_saved_project_comes_back_unchanged(tmp_path):
    store = ProjectStore(tmp_path)
    state = ProjectState(
        settings=ProjectSettings(
            minimum_confidence=0.6,
            grouping_tolerance=0.1,
            show_other_uncertainties=False,
            show_reviewed=True,
            play_automatically=False,
            show_details=True,
            auto_play_delay_seconds=5,
            smoothing_prompt="Skryf in Afrikaans.\n  Hou dit kort.\n",
        ),
        groups=[
            WordGroup(
                id="group-1",
                representative_text="Bosch",
                occurrence_ids=["one", "two"],
                replacement="Bosch",
                reviewed=True,
                user_created=True,
            )
        ],
        occurrences=[
            make_occurrence("one", reviewed=True, replacement="Bosch"),
            make_occurrence(
                "two",
                detected_text="Bosh",
                auto_applied=True,
                applied_rule_id="rule-1",
                stale=True,
            ),
        ],
        flagged=[
            FlaggedItem(
                recording_name="Interview 01.m4a",
                token_id="token-99",
                text="15,000",
                start=70.0,
                reasons=["numeric_disagreement"],
                confidence="review_required",
                risk_categories=["money"],
            ),
            FlaggedItem(
                recording_name="Interview 02.m4a",
                token_id="token-100",
                text="Jurgen",
                start=None,
                settled=True,
            ),
        ],
        rules=[
            ReplacementRule(
                id="rule-1",
                matched_text="Bosh",
                normalised_text="bosh",
                replacement="Bosch",
                language="en",
                created_at="2026-08-17T09:00:00",
                occurrence_count=3,
                group_id="group-1",
            )
        ],
        processed_at="2026-08-17T09:05:00",
        transcript_times={"Interview 01.m4a": 1755421500123456700},
        last_group_id="group-1",
        last_occurrence_id="two",
    )

    assert store.save(state)
    assert store.load() == state


def test_a_transcript_time_survives_the_file_exactly(tmp_path):
    """It is compared for being the same number, so nothing may round it.

    The number is a filesystem's own account of a file's age in nanoseconds,
    and it is far too large to survive being read back as a floating point
    number. One digit lost at the end turns "this is the file I analysed" into
    "this is a different file", and the recording is then read again on every
    single opening for ever.
    """
    store = ProjectStore(tmp_path)
    exact = 1755421500123456789

    assert store.save(ProjectState(transcript_times={"Interview 01.m4a": exact}))

    assert store.load().transcript_times == {"Interview 01.m4a": exact}


def test_a_transcript_time_that_is_not_a_whole_number_is_forgotten(tmp_path):
    """Forgetting costs a read. Believing it would cost the recording.

    Anything a hand-edited or half-written file offers here that is not a
    plain whole number says nothing about what the file on the disk is, and
    the only safe reading of that is that nobody knows, which sends the
    recording to be read again.
    """
    write_project(
        tmp_path,
        {
            "transcript_times": {
                "Interview 01.m4a": "1755421500123456789",
                "Interview 02.m4a": True,
                "Interview 03.m4a": 12.5,
                "Interview 04.m4a": 1755421500123456789,
            }
        },
    )

    state = ProjectStore(tmp_path).load()

    assert state.transcript_times == {"Interview 04.m4a": 1755421500123456789}


def test_a_transcript_times_field_of_the_wrong_shape_gives_an_empty_one(tmp_path):
    write_project(tmp_path, {"transcript_times": ["Interview 01.m4a"]})

    assert ProjectStore(tmp_path).load().transcript_times == {}


def test_the_saved_file_records_the_format_version(tmp_path):
    store = ProjectStore(tmp_path)

    assert store.save(ProjectState())

    written = json.loads(store.path.read_text(encoding="utf-8"))
    assert written["version"] == PROJECT_FORMAT_VERSION


def test_a_project_written_by_a_newer_version_is_still_read(tmp_path):
    """Refusing would leave somebody unable to review a folder they own."""
    write_project(
        tmp_path,
        {
            "version": PROJECT_FORMAT_VERSION + 1,
            "occurrences": [make_occurrence("one").to_dict()],
            "processed_at": "2026-08-17T09:05:00",
        },
    )

    state = ProjectStore(tmp_path).load()

    assert [item.id for item in state.occurrences] == ["one"]
    assert state.processed_at == "2026-08-17T09:05:00"


# -- Damaged files --------------------------------------------------------


def test_a_file_that_is_not_json_gives_an_empty_project(tmp_path):
    (tmp_path / PROJECT_FILE_NAME).write_text("{ this is not json", encoding="utf-8")

    assert ProjectStore(tmp_path).load() == ProjectState()


def test_a_file_that_is_not_an_object_gives_an_empty_project(tmp_path):
    (tmp_path / PROJECT_FILE_NAME).write_text('["a", "list", "instead"]', encoding="utf-8")

    assert ProjectStore(tmp_path).load() == ProjectState()


def test_a_file_that_is_not_text_gives_an_empty_project(tmp_path):
    (tmp_path / PROJECT_FILE_NAME).write_bytes(b'{"processed_at": "\xff\xfe not text"}')

    assert ProjectStore(tmp_path).load() == ProjectState()


def test_every_field_of_the_wrong_type_falls_back_to_its_default(tmp_path):
    """One field at a time, exactly as the session file degrades."""
    write_project(
        tmp_path,
        {
            "version": "one",
            "settings": {
                "minimum_confidence": "low",
                "grouping_tolerance": None,
                "show_other_uncertainties": "yes",
                "show_reviewed": 1,
                "play_automatically": [],
                "show_details": "yes",
                "auto_play_delay_seconds": "two",
            },
            "groups": "not a list",
            "occurrences": [
                {
                    "id": "one",
                    "recording_name": 42,
                    "token_id": None,
                    "detected_text": ["Bosch"],
                    "normalised_text": {},
                    "start": "12.5",
                    "end": True,
                    "confidence_strength": "0.42",
                    "language": 7,
                    "context_before": 0,
                    "context_after": None,
                    "reviewed": "true",
                    "correct_as_detected": 1,
                    "replacement": 5,
                    "isolated": "no",
                    "stale": [],
                    "auto_applied": "later",
                    "applied_rule_id": 99,
                }
            ],
            "rules": {"not": "a list"},
            "processed_at": 20260817,
            "last_group_id": 3,
            "last_occurrence_id": [],
        },
    )

    state = ProjectStore(tmp_path).load()

    assert state.settings == ProjectSettings()
    assert state.groups == []
    assert state.rules == []
    assert state.processed_at == ""
    assert state.last_group_id is None
    assert state.last_occurrence_id is None
    assert state.occurrences == [
        Occurrence(
            id="one",
            recording_name="",
            token_id="",
            detected_text="",
            normalised_text="",
            start=None,
            end=None,
            confidence_strength=None,
            language="unknown",
            context_before="",
            context_after="",
        )
    ]


def test_a_confidence_outside_the_range_is_pulled_back_into_it(tmp_path):
    """A threshold of five would put every word in front of the person."""
    write_project(
        tmp_path,
        {"settings": {"minimum_confidence": 5.0, "grouping_tolerance": -2.0}},
    )

    settings = ProjectStore(tmp_path).load().settings

    assert settings.minimum_confidence == 1.0
    assert settings.grouping_tolerance == 0.0


def test_a_new_folder_plays_without_a_wait(tmp_path):
    assert ProjectStore(tmp_path).load().settings.auto_play_delay_seconds == 0


def test_the_old_default_wait_in_an_older_file_becomes_no_wait(tmp_path):
    """Two seconds in a version 1 file is the old default, not a choice."""
    write_project(tmp_path, {"version": 1, "settings": {"auto_play_delay_seconds": 2}})
    assert ProjectStore(tmp_path).load().settings.auto_play_delay_seconds == 0


def test_a_wait_the_person_chose_in_an_older_file_is_kept(tmp_path):
    write_project(tmp_path, {"version": 1, "settings": {"auto_play_delay_seconds": 5}})
    assert ProjectStore(tmp_path).load().settings.auto_play_delay_seconds == 5


def test_two_seconds_chosen_after_the_change_is_kept(tmp_path):
    """Once written at the current version, two seconds is the person's own."""
    store = ProjectStore(tmp_path)
    write_project(tmp_path, {"version": 1, "settings": {"auto_play_delay_seconds": 2}})
    state = store.load()
    state.settings.auto_play_delay_seconds = 2
    store.save(state)

    assert store.load().settings.auto_play_delay_seconds == 2


def test_the_wait_before_playing_is_kept_in_whole_seconds(tmp_path):
    """It is a number a person sets for themselves, so it is forgiving.

    Out of range is pulled to the nearest end rather than thrown away,
    because somebody who wrote a large number was plainly reaching for a long
    wait. A fraction is rounded for the same reason: 2.5 means about two and
    a half seconds, and answering it with the default would be baffling.
    """
    write_project(tmp_path, {"settings": {"auto_play_delay_seconds": 900}})
    assert ProjectStore(tmp_path).load().settings.auto_play_delay_seconds == 30

    write_project(tmp_path, {"settings": {"auto_play_delay_seconds": -4}})
    assert ProjectStore(tmp_path).load().settings.auto_play_delay_seconds == 0

    write_project(tmp_path, {"settings": {"auto_play_delay_seconds": 2.5}})
    assert ProjectStore(tmp_path).load().settings.auto_play_delay_seconds == 2

    # Nought is a real answer, not a missing one: somebody with their speech
    # turned off wants the audio at once.
    write_project(tmp_path, {"settings": {"auto_play_delay_seconds": 0}})
    assert ProjectStore(tmp_path).load().settings.auto_play_delay_seconds == 0


def test_an_occurrence_with_no_identifier_is_dropped(tmp_path):
    """Inventing one would show the word as a fresh finding with no history."""
    write_project(
        tmp_path,
        {
            "occurrences": [
                {"recording_name": "Interview 01.m4a", "detected_text": "Bosch"},
                make_occurrence("two").to_dict(),
                "not even an object",
            ]
        },
    )

    state = ProjectStore(tmp_path).load()

    assert [item.id for item in state.occurrences] == ["two"]


def test_two_occurrences_with_one_identifier_keep_the_first(tmp_path):
    write_project(
        tmp_path,
        {
            "occurrences": [
                make_occurrence("one", detected_text="Bosch").to_dict(),
                make_occurrence("one", detected_text="Bosh").to_dict(),
            ]
        },
    )

    state = ProjectStore(tmp_path).load()

    assert [item.detected_text for item in state.occurrences] == ["Bosch"]


def test_a_flagged_word_with_no_recording_or_no_word_is_dropped(tmp_path):
    """The pair is how it is keyed, found and told apart, so neither can be missing.

    Nothing is lost by dropping one. Unlike an occurrence, a flagged word
    holds no decision anybody made, and the next analysis of that recording
    finds it again.
    """
    write_project(
        tmp_path,
        {
            "flagged": [
                {"token_id": "token-1", "text": "15,000"},
                {"recording_name": "Interview 01.m4a", "text": "15,000"},
                "not even an object",
                FlaggedItem(
                    recording_name="Interview 01.m4a",
                    token_id="token-2",
                    text="15,000",
                ).to_dict(),
            ]
        },
    )

    state = ProjectStore(tmp_path).load()

    assert [item.token_id for item in state.flagged] == ["token-2"]


def test_a_flagged_word_falls_back_field_by_field(tmp_path):
    """A word with no text is not damage: sometimes nothing at all was chosen."""
    write_project(
        tmp_path,
        {
            "flagged": [
                {
                    "recording_name": "Interview 01.m4a",
                    "token_id": "token-1",
                    "start": "half past two",
                    "reasons": ["numeric_disagreement", 7, None],
                    "confidence": 0.42,
                    "settled": "yes",
                }
            ]
        },
    )

    item = ProjectStore(tmp_path).load().flagged[0]

    assert item.text == ""
    assert item.start is None
    assert item.reasons == ["numeric_disagreement"]
    # Unresolved rather than high: a category nobody can read must leave the
    # word in front of the person rather than hiding it.
    assert item.confidence == "unresolved"
    assert item.settled is False
    # Written before the kind of value was kept: it is simply not known.
    assert item.risk_categories == []


def test_one_word_flagged_twice_keeps_the_first(tmp_path):
    """The second could never be selected, and would be counted as work waiting."""
    write_project(
        tmp_path,
        {
            "flagged": [
                FlaggedItem("Interview 01.m4a", "token-1", "15,000").to_dict(),
                FlaggedItem("Interview 01.m4a", "token-1", "50,000").to_dict(),
                FlaggedItem("Interview 02.m4a", "token-1", "15,000").to_dict(),
            ]
        },
    )

    state = ProjectStore(tmp_path).load()

    assert [(item.recording_name, item.text) for item in state.flagged] == [
        ("Interview 01.m4a", "15,000"),
        ("Interview 02.m4a", "15,000"),
    ]


def test_a_rule_with_no_identifier_is_kept_and_given_one(tmp_path):
    """Nothing points at a rule by identifier, and the correction is worth keeping."""
    write_project(
        tmp_path,
        {
            "rules": [
                {
                    "matched_text": "Bosh",
                    "normalised_text": "bosh",
                    "replacement": "Bosch",
                    "language": "en",
                    "created_at": "2026-08-17T09:00:00",
                }
            ]
        },
    )

    state = ProjectStore(tmp_path).load()

    assert len(state.rules) == 1
    assert state.rules[0].id
    assert state.rules[0].replacement == "Bosch"
    assert state.rules[0].group_id is None


def test_a_rule_with_nothing_to_match_or_apply_is_dropped(tmp_path):
    write_project(
        tmp_path,
        {
            "rules": [
                {"id": "rule-1", "matched_text": "Bosh", "replacement": ""},
                {"id": "rule-2", "replacement": "Bosch"},
                {
                    "id": "rule-3",
                    "matched_text": "Bosh",
                    "normalised_text": "bosh",
                    "replacement": "Bosch",
                    "language": "en",
                    "created_at": "2026-08-17T09:00:00",
                },
            ]
        },
    )

    state = ProjectStore(tmp_path).load()

    assert [rule.id for rule in state.rules] == ["rule-3"]


# -- Repairing what does not add up ---------------------------------------


def test_a_group_naming_a_missing_occurrence_loses_that_name(tmp_path):
    write_project(
        tmp_path,
        {
            "occurrences": [make_occurrence("one").to_dict()],
            "groups": [
                WordGroup(
                    id="group-1",
                    representative_text="Bosch",
                    occurrence_ids=["one", "gone"],
                ).to_dict()
            ],
        },
    )

    state = ProjectStore(tmp_path).load()

    assert state.groups[0].occurrence_ids == ["one"]
    assert state.occurrences_of("group-1") == [state.occurrence("one")]


def test_a_group_left_holding_nothing_is_dropped(tmp_path):
    """An empty group is a row nobody can open, play or correct."""
    write_project(
        tmp_path,
        {
            "occurrences": [make_occurrence("one").to_dict()],
            "groups": [
                WordGroup(
                    id="group-1", representative_text="Bosch", occurrence_ids=["one"]
                ).to_dict(),
                WordGroup(
                    id="group-2",
                    representative_text="Meyer",
                    occurrence_ids=["gone", "also gone"],
                    user_created=True,
                ).to_dict(),
            ],
        },
    )

    state = ProjectStore(tmp_path).load()

    assert [group.id for group in state.groups] == ["group-1"]
    assert state.group("group-2") is None


def test_an_occurrence_claimed_by_two_groups_stays_in_the_first(tmp_path):
    """First is the only rule that answers the same way every time."""
    write_project(
        tmp_path,
        {
            "occurrences": [make_occurrence("one").to_dict(), make_occurrence("two").to_dict()],
            "groups": [
                WordGroup(
                    id="group-1", representative_text="Bosch", occurrence_ids=["one", "two"]
                ).to_dict(),
                WordGroup(
                    id="group-2", representative_text="Bosh", occurrence_ids=["two"]
                ).to_dict(),
            ],
        },
    )

    state = ProjectStore(tmp_path).load()

    assert state.group("group-1").occurrence_ids == ["one", "two"]
    assert state.group("group-2") is None
    assert state.group_of("two").id == "group-1"


def test_the_same_damaged_file_is_repaired_the_same_way_twice(tmp_path):
    write_project(
        tmp_path,
        {
            "occurrences": [make_occurrence("one").to_dict(), make_occurrence("two").to_dict()],
            "groups": [
                WordGroup(
                    id="group-1", representative_text="Bosch", occurrence_ids=["two", "gone"]
                ).to_dict(),
                WordGroup(
                    id="group-2", representative_text="Bosh", occurrence_ids=["two", "one"]
                ).to_dict(),
            ],
        },
    )
    store = ProjectStore(tmp_path)

    assert store.load() == store.load()


def test_a_marker_pointing_at_something_gone_starts_at_the_beginning(tmp_path):
    """Reprocessing rebuilds the automatic groups, so this is ordinary."""
    write_project(
        tmp_path,
        {
            "occurrences": [make_occurrence("one").to_dict()],
            "last_group_id": "a group from last week",
            "last_occurrence_id": "a word from last week",
        },
    )

    state = ProjectStore(tmp_path).load()

    assert state.last_group_id is None
    assert state.last_occurrence_id is None


def test_a_marker_pointing_at_something_that_is_there_is_kept(tmp_path):
    write_project(
        tmp_path,
        {
            "occurrences": [make_occurrence("one").to_dict()],
            "groups": [
                WordGroup(
                    id="group-1", representative_text="Bosch", occurrence_ids=["one"]
                ).to_dict()
            ],
            "last_group_id": "group-1",
            "last_occurrence_id": "one",
        },
    )

    state = ProjectStore(tmp_path).load()

    assert state.last_group_id == "group-1"
    assert state.last_occurrence_id == "one"


def test_the_rule_that_corrected_a_word_is_kept_and_can_be_reached_again(tmp_path):
    """Undoing an automatic correction has to reach the exact rule behind it."""
    write_project(
        tmp_path,
        {
            "occurrences": [
                make_occurrence(
                    "one", auto_applied=True, applied_rule_id="rule-1"
                ).to_dict()
            ],
            "rules": [
                rule("rule-1", "bosh", "unknown").to_dict(),
                rule("rule-2", "bosh", "en").to_dict(),
            ],
        },
    )

    state = ProjectStore(tmp_path).load()
    occurrence = state.occurrence("one")

    assert occurrence.applied_rule_id == "rule-1"
    # Matching the text and the language again would have found the other
    # rule, which is exactly why the identifier is recorded rather than
    # looked up.
    assert state.rule_for("bosh", "en").id == "rule-2"


def test_a_rule_the_person_has_deleted_leaves_the_correction_standing(tmp_path):
    """The correction did happen; only the way back to the rule is gone."""
    write_project(
        tmp_path,
        {
            "occurrences": [
                make_occurrence(
                    "one", auto_applied=True, applied_rule_id="a rule deleted last week"
                ).to_dict()
            ]
        },
    )

    occurrence = ProjectStore(tmp_path).load().occurrence("one")

    assert occurrence.applied_rule_id is None
    assert occurrence.auto_applied is True


# -- The lookups ----------------------------------------------------------


def build_state() -> ProjectState:
    return ProjectState(
        groups=[
            WordGroup(
                id="group-1", representative_text="Bosch", occurrence_ids=["one", "two"]
            ),
            WordGroup(id="group-2", representative_text="Meyer", occurrence_ids=["three"]),
        ],
        occurrences=[
            make_occurrence("one"),
            make_occurrence("two", detected_text="Bosh"),
            make_occurrence("three", detected_text="Meyer", normalised_text="meyer"),
            make_occurrence("four", detected_text="Maier", isolated=True),
            make_occurrence("five", detected_text="Kruger"),
        ],
    )


def test_the_lookups_find_what_the_window_asks_for():
    state = build_state()

    assert state.occurrence("two").detected_text == "Bosh"
    assert state.occurrence("nothing") is None
    assert state.group("group-2").representative_text == "Meyer"
    assert state.group("nothing") is None
    assert state.group_of("two").id == "group-1"
    assert state.group_of("four") is None
    assert [item.id for item in state.occurrences_of("group-1")] == ["one", "two"]
    assert state.occurrences_of("nothing") == []


def test_occurrences_of_follows_the_order_the_group_lists():
    state = ProjectState(
        groups=[
            WordGroup(id="group-1", representative_text="Bosch", occurrence_ids=["two", "one"])
        ],
        occurrences=[make_occurrence("one"), make_occurrence("two")],
    )

    assert [item.id for item in state.occurrences_of("group-1")] == ["two", "one"]


def test_loose_occurrences_are_the_ones_no_group_holds():
    """Both the word taken out of a group and the word never grouped."""
    state = build_state()

    loose = state.loose_occurrences()

    assert [item.id for item in loose] == ["four", "five"]
    assert loose[0].isolated is True
    assert loose[1].isolated is False


def test_loose_occurrences_is_everything_when_there_are_no_groups():
    state = ProjectState(occurrences=[make_occurrence("one"), make_occurrence("two")])

    assert [item.id for item in state.loose_occurrences()] == ["one", "two"]


def test_rule_for_matches_the_normalised_text_and_the_language():
    state = ProjectState(rules=[rule("rule-1", "bosch", "en"), rule("rule-2", "meyer", "de")])

    assert state.rule_for("bosch", "en").id == "rule-1"
    assert state.rule_for("meyer", "de").id == "rule-2"
    assert state.rule_for("kruger", "en") is None


def test_rule_for_keeps_a_deliberate_language_apart():
    state = ProjectState(rules=[rule("rule-1", "bosch", "de")])

    assert state.rule_for("bosch", "en") is None


def test_rule_for_lets_an_unknown_language_through_on_either_side():
    """An unknown language is missing information, not a difference."""
    known = ProjectState(rules=[rule("rule-1", "bosch", "en")])
    unknown = ProjectState(rules=[rule("rule-2", "bosch", "unknown")])

    assert known.rule_for("bosch", "unknown").id == "rule-1"
    assert unknown.rule_for("bosch", "en").id == "rule-2"


def test_rule_for_prefers_the_matching_language_over_an_unknown_one():
    state = ProjectState(rules=[rule("rule-1", "bosch", "unknown"), rule("rule-2", "bosch", "en")])

    assert state.rule_for("bosch", "en").id == "rule-2"


def test_a_group_makes_one_rule_for_each_detected_form():
    """The shape the later stages build: three spellings, three rules.

    A file transcribed next week which says ``Bosh`` is then answered by a
    person who actually accepted a correction for ``Bosh``, and a spelling
    nobody has accepted is left for the review queue.
    """
    state = ProjectState(
        rules=[
            ReplacementRule(
                id=f"rule-{index}",
                matched_text=form,
                normalised_text=form.lower(),
                replacement="Bosch",
                language="en",
                created_at="2026-08-17T09:00:00",
                group_id="group-1",
            )
            for index, form in enumerate(["Bosch", "Bosh", "Bosche"])
        ]
    )

    assert {rule.group_id for rule in state.rules} == {"group-1"}
    assert state.rule_for("bosh", "en").matched_text == "Bosh"
    assert state.rule_for("bosche", "en").matched_text == "Bosche"
    assert state.rule_for("boshe", "en") is None


# -- Nothing leaks between projects ---------------------------------------


def test_two_folders_keep_entirely_separate_state(tmp_path):
    """A rule accepted in one folder is invisible in the other."""
    first_folder = tmp_path / "Client A"
    second_folder = tmp_path / "Client B"
    first_folder.mkdir()
    second_folder.mkdir()
    first = ProjectStore(first_folder)
    second = ProjectStore(second_folder)

    first.save(
        ProjectState(
            settings=ProjectSettings(minimum_confidence=0.9),
            occurrences=[make_occurrence("one")],
            rules=[rule("rule-1", "bosh", "en")],
        )
    )
    second.save(ProjectState(occurrences=[make_occurrence("two", detected_text="Bosh")]))

    assert first.path != second.path
    assert first.load().rule_for("bosh", "en") is not None
    assert second.load().rule_for("bosh", "en") is None
    assert second.load().settings.minimum_confidence == DEFAULT_MINIMUM_CONFIDENCE
    assert [item.id for item in second.load().occurrences] == ["two"]


def test_a_project_moves_with_the_folder_it_belongs_to(tmp_path):
    """The file is in the audio folder, so copying the folder copies the review."""
    original = tmp_path / "Recordings"
    original.mkdir()
    ProjectStore(original).save(ProjectState(occurrences=[make_occurrence("one")]))

    moved = tmp_path / "Backup"
    moved.mkdir()
    (moved / PROJECT_FILE_NAME).write_bytes((original / PROJECT_FILE_NAME).read_bytes())

    assert [item.id for item in ProjectStore(moved).load().occurrences] == ["one"]


# -- Saving ---------------------------------------------------------------


def test_a_save_that_cannot_be_written_says_so(tmp_path):
    """The caller has to tell the person, so it must never pass quietly."""
    blocked = tmp_path / "not-a-folder"
    blocked.write_text("this is a file, so nothing can be written inside it", encoding="utf-8")

    assert ProjectStore(blocked).save(ProjectState()) is False


def test_a_save_creates_the_folder_when_it_is_not_there(tmp_path):
    store = ProjectStore(tmp_path / "new folder")

    assert store.save(ProjectState(processed_at="2026-08-17T09:05:00")) is True
    assert store.load().processed_at == "2026-08-17T09:05:00"


# -- The names a folder's reviews taught ----------------------------------


def bosch(*wrong_forms: str, language: str = "en") -> LearnedName:
    return LearnedName(
        text="Bosch",
        language=language,
        wrong_forms=list(wrong_forms),
        learned_at="2026-10-07T09:00:00+02:00",
    )


def test_learned_names_survive_a_save_and_a_load(tmp_path):
    store = ProjectStore(tmp_path)
    store.save(ProjectState(learned_names=[bosch("Bosh"), LearnedName("Müller", "de")]))

    loaded = store.load().learned_names

    assert loaded == [bosch("Bosh"), LearnedName("Müller", "de")]


def test_a_file_from_before_learned_names_loads_with_none(tmp_path):
    write_project(tmp_path, {"version": PROJECT_FORMAT_VERSION, "rules": []})

    assert ProjectStore(tmp_path).load().learned_names == []


def test_a_learned_name_with_no_text_is_dropped_and_the_rest_kept(tmp_path):
    write_project(
        tmp_path,
        {
            "learned_names": [
                {"text": "", "language": "en"},
                "not an entry",
                {"text": "Bosch", "wrong_forms": ["Bosh", 3]},
            ]
        },
    )

    [name] = ProjectStore(tmp_path).load().learned_names

    assert name.text == "Bosch"
    assert name.language == "unknown"
    assert name.wrong_forms == ["Bosh"]


def test_a_file_listing_one_name_twice_loads_it_once(tmp_path):
    write_project(
        tmp_path,
        {"learned_names": [bosch("Bosh").to_dict(), {"text": "bosch", "language": "en",
                                                      "wrong_forms": ["Bosj"]}]},
    )

    assert ProjectStore(tmp_path).load().learned_names == [bosch("Bosh", "Bosj")]


def test_a_repeated_name_gains_new_wrong_forms_and_is_listed_once():
    existing = [bosch("Bosh"), LearnedName("Vermeulen", "en", ["Fermeulen"])]

    merged = merge_learned_names(existing, [bosch("bosh", "Bosj", "BOSCH")])

    assert [name.text for name in merged] == ["Bosch", "Vermeulen"]
    # Capitals aside "bosh" is already known, and the correct text is never a wrong form.
    assert merged[0].wrong_forms == ["Bosh", "Bosj"]
    assert existing[0].wrong_forms == ["Bosh"]


def test_one_spelling_in_two_languages_is_two_names():
    merged = merge_learned_names([bosch("Bosh")], [bosch("Bos", language="de")])

    assert [(name.text, name.language) for name in merged] == [("Bosch", "en"), ("Bosch", "de")]


def test_reading_the_names_says_what_it_found(tmp_path):
    assert read_learned_names(tmp_path) == ([], JsonReadStatus.MISSING)

    ProjectStore(tmp_path).save(ProjectState(learned_names=[bosch("Bosh")]))
    assert read_learned_names(tmp_path) == ([bosch("Bosh")], JsonReadStatus.READ)

    (tmp_path / PROJECT_FILE_NAME).write_text("{ not json", encoding="utf-8")
    assert read_learned_names(tmp_path) == ([], JsonReadStatus.DAMAGED)


def test_removing_a_name_keeps_everything_else_in_the_file(tmp_path):
    write_project(
        tmp_path,
        {
            "rules": [],
            "a_key_from_a_newer_version": {"kept": True},
            "learned_names": [bosch("Bosh").to_dict(), LearnedName("Müller", "de").to_dict()],
        },
    )

    assert remove_learned_name(tmp_path, "bosch", "en") is True

    data = json.loads((tmp_path / PROJECT_FILE_NAME).read_text(encoding="utf-8"))
    assert data["a_key_from_a_newer_version"] == {"kept": True}
    assert [entry["text"] for entry in data["learned_names"]] == ["Müller"]


def test_removing_a_name_from_a_damaged_file_writes_nothing(tmp_path):
    path = tmp_path / PROJECT_FILE_NAME
    path.write_text('{"learned_names": [', encoding="utf-8")

    assert remove_learned_name(tmp_path, "Bosch", "en") is False
    assert path.read_text(encoding="utf-8") == '{"learned_names": ['


def test_removing_a_name_that_is_not_there_writes_nothing(tmp_path):
    assert remove_learned_name(tmp_path, "Bosch", "en") is False
    assert not (tmp_path / PROJECT_FILE_NAME).exists()


def test_a_save_does_not_bring_back_a_name_removed_on_disk(tmp_path):
    store = ProjectStore(tmp_path)
    store.save(ProjectState(learned_names=[bosch("Bosh")]))
    state = store.load()

    # Removed by somebody else while this copy was held.
    assert remove_learned_name(tmp_path, "Bosch", "en") is True
    vermeulen = LearnedName("Vermeulen", "en", ["Fermeulen"])
    assert store.save_keeping_learned_names(state, [vermeulen]) is True

    assert [name.text for name in store.load().learned_names] == ["Vermeulen"]
    assert state.learned_names == [vermeulen]


def test_a_save_with_no_file_keeps_the_names_it_holds(tmp_path):
    store = ProjectStore(tmp_path)
    state = ProjectState(learned_names=[bosch("Bosh")])

    assert store.save_keeping_learned_names(state, [bosch("Bosj")]) is True

    assert store.load().learned_names == [bosch("Bosh", "Bosj")]


# -- The stretches whose speaker is in doubt --------------------------------


def test_a_speaker_doubt_survives_the_round_trip(tmp_path):
    state = ProjectState()
    state.speaker_doubts = [
        SpeakerDoubtItem(
            recording_name="Interview 01.m4a",
            start=40.0,
            end=41.4,
            speakers=["Jacques", "Speaker 1"],
            token_ids=["token-1", "token-2"],
        )
    ]
    store = ProjectStore(tmp_path)
    store.save(state)

    loaded = store.load()

    assert loaded.speaker_doubts == state.speaker_doubts


def test_a_project_written_before_speaker_doubts_loads_with_none(tmp_path):
    write_project(tmp_path, {"flagged": []})

    assert ProjectStore(tmp_path).load().speaker_doubts == []


def test_a_speaker_doubt_pointing_at_nothing_is_dropped_and_the_rest_falls_back(tmp_path):
    write_project(
        tmp_path,
        {
            "speaker_doubts": [
                {"token_ids": ["token-1"]},
                {"recording_name": "Interview 01.m4a", "token_ids": []},
                "not even an object",
                {
                    "recording_name": "Interview 01.m4a",
                    "start": "half past two",
                    "speakers": ["Jacques", 7, ""],
                    "token_ids": ["token-3", None],
                },
            ]
        },
    )

    item, = ProjectStore(tmp_path).load().speaker_doubts

    assert item.start is None
    assert item.speakers == ["Jacques"]
    assert item.token_ids == ["token-3"]


def test_one_stretch_listed_twice_keeps_the_first(tmp_path):
    first = SpeakerDoubtItem("Interview 01.m4a", 1.0, 2.0, ["A"], ["token-1"])
    write_project(
        tmp_path,
        {"speaker_doubts": [first.to_dict(), first.to_dict()]},
    )

    assert ProjectStore(tmp_path).load().speaker_doubts == [first]


def test_a_folder_has_no_smoothing_prompt_of_its_own_until_one_is_saved(tmp_path):
    assert ProjectStore(tmp_path).load().settings.smoothing_prompt == ""


def test_a_blank_or_wrong_typed_smoothing_prompt_counts_as_none(tmp_path):
    for value in ("   \n", 42, None):
        write_project(tmp_path, {"settings": {"smoothing_prompt": value}})
        assert ProjectStore(tmp_path).load().settings.smoothing_prompt == ""
