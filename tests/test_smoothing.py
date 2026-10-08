"""Tests for the smooth transcript made with the language model.

A fake client stands in for OpenAI. It answers each part from a function the
test gives it, so each test says exactly what the model returned.
"""

from __future__ import annotations

import json
import re
import threading
import time
from types import SimpleNamespace

from vox_verbatim.settings import (
    DEFAULT_SMOOTHING_PROMPT,
    Settings,
    SmoothingSettings,
)
from vox_verbatim.transcription.model import (
    Candidate,
    Confidence,
    FinalToken,
    Provider,
    Speaker,
    Transcript,
)
from vox_verbatim.transcription.smoothing import (
    FORMAT_INSTRUCTIONS,
    SMOOTH_EXPORT_NAME,
    SMOOTHING_PURPOSE,
    Smoother,
    Turn,
    build_turns,
    check_turn,
    smoother_for,
    split_into_parts,
    style_prompt_for,
    write_smooth_transcript,
)
from vox_verbatim.transcription.project import ProjectStore

_TURN_LINE = re.compile(r"^Turn (\d+) \(([^)]*)\): (.*)$")


def _word(text: str, speaker: str = "0") -> FinalToken:
    return FinalToken(
        text=text,
        normalised_text=text.lower(),
        text_source=Provider.ELEVENLABS,
        text_confidence=Confidence.HIGH,
        speaker=speaker,
    )


def _uncertain(first: str, second: str, speaker: str = "0") -> FinalToken:
    token = _word(first, speaker)
    token.text_confidence = Confidence.UNRESOLVED
    token.candidates = [
        Candidate(text=first, providers=(Provider.ELEVENLABS,)),
        Candidate(text=second, providers=(Provider.OPENAI,)),
    ]
    return token


def _transcript(*turns: tuple[str, str]) -> Transcript:
    tokens: list[FinalToken] = []
    for speaker, text in turns:
        tokens.extend(_word(word, speaker) for word in text.split())
    return Transcript(
        recording_name="interview.wav",
        tokens=tokens,
        speakers=[Speaker(id="0", name="Adrienne"), Speaker(id="1", name="Jacques")],
    )


def _turns_in(arguments: dict) -> list[tuple[int, str, str]]:
    content = arguments["input"][0]["content"]
    found = []
    for line in content.splitlines():
        match = _TURN_LINE.match(line)
        if match:
            found.append((int(match.group(1)), match.group(2), match.group(3)))
    return found


class FakeClient:
    """Answers each request with ``answer(turns, call_index)``."""

    def __init__(self, answer, delay: float = 0.0) -> None:
        self.answer = answer
        self.delay = delay
        self.calls: list[dict] = []
        self.active = 0
        self.most_active = 0
        self._lock = threading.Lock()
        self.responses = self

    def parse(self, **arguments):
        with self._lock:
            index = len(self.calls)
            self.calls.append(arguments)
            self.active += 1
            self.most_active = max(self.most_active, self.active)
        try:
            time.sleep(self.delay)
            turns = self.answer(_turns_in(arguments), index)
            body = json.dumps({"turns": [{"number": n, "text": t} for n, t in turns]})
            return SimpleNamespace(model="gpt-6.1-sol-2026", id=f"resp_{index}", output_text=body)
        finally:
            with self._lock:
                self.active -= 1


def _clean(text: str) -> str:
    """A stand-in for the model: removes fillers and repeated words."""
    kept: list[str] = []
    for word in text.split():
        if word.lower().strip(",.") in {"um", "uh"}:
            continue
        if kept and kept[-1].lower() == word.lower():
            continue
        kept.append(word)
    return " ".join(kept)


def _smoother(client: FakeClient, **extra) -> Smoother:
    return Smoother(
        api_key="sk-test",
        model="gpt-6.1-sol",
        reasoning_effort="low",
        style_prompt=DEFAULT_SMOOTHING_PROMPT,
        client=client,
        **extra,
    )


class FakeStore:
    """Keeps the exports in memory. ``folder`` is where the fingerprint of
    the smoothed text goes, for a test that gets that far."""

    def __init__(self, folder=None) -> None:
        self.written: dict[str, str] = {}
        self.folder = folder

    def write_export(self, name: str, text: str):
        self.written[name] = text
        return name


# -- The file -------------------------------------------------------------


def test_the_file_has_speaker_names_no_times_and_no_fillers(tmp_path):
    transcript = _transcript(
        ("0", "Um we we were living in the old house"),
        ("1", "Uh how long did you stay there"),
    )
    client = FakeClient(lambda turns, _: [(n, _clean(t)) for n, _s, t in turns])
    store = FakeStore(tmp_path)

    outcome = write_smooth_transcript(transcript, store, _smoother(client))

    assert outcome.succeeded
    text = store.written[SMOOTH_EXPORT_NAME]
    assert "Adrienne:\nwe were living in the old house\n" in text
    assert "Jacques:\nhow long did you stay there\n" in text
    assert "Um" not in text and "Uh" not in text
    assert not re.search(r"\[\d+:\d\d\]", text)


def test_the_request_carries_the_style_prompt_then_the_fixed_format_and_a_strict_schema():
    client = FakeClient(lambda turns, _: [(n, t) for n, _s, t in turns])
    _smoother(client).smooth(_transcript(("0", "hello there friend")))

    arguments = client.calls[0]
    assert arguments["instructions"].startswith(DEFAULT_SMOOTHING_PROMPT.rstrip())
    assert arguments["instructions"].endswith(FORMAT_INSTRUCTIONS)
    assert arguments["text"]["format"]["type"] == "json_schema"
    assert arguments["text"]["format"]["strict"] is True
    assert arguments["reasoning"] == {"effort": "low"}
    assert arguments["model"] == "gpt-6.1-sol"


def test_every_uncertain_marker_reaches_the_file_unchanged():
    transcript = _transcript(("0", "um we put it on the"))
    transcript.tokens.append(_uncertain("board", "place"))
    client = FakeClient(lambda turns, _: [(n, _clean(t)) for n, _s, t in turns])

    outcome = _smoother(client).smooth(transcript)

    assert "[UNCERTAIN: board / place]" in outcome.text
    assert not outcome.warnings


def test_a_dropped_reply_between_two_turns_by_one_speaker_leaves_one_joined_turn():
    transcript = _transcript(
        ("0", "We moved to the farm in the spring"),
        ("1", "Mm-hmm"),
        ("0", "and the river flooded that year"),
    )

    def answer(turns, _):
        return [(n, "" if t == "Mm-hmm" else t) for n, _s, t in turns]

    outcome = _smoother(FakeClient(answer)).smooth(transcript)

    assert outcome.text.count("Adrienne:") == 1
    assert "Jacques" not in outcome.text
    assert (
        "We moved to the farm in the spring and the river flooded that year" in outcome.text
    )


def test_a_part_that_misses_a_turn_is_tried_again_and_passes():
    transcript = _transcript(("0", "first thing said"), ("1", "second thing said"))

    def answer(turns, index):
        given = [(n, t) for n, _s, t in turns]
        return given[:1] if index == 0 else given

    client = FakeClient(answer)
    outcome = _smoother(client).smooth(transcript)

    assert len(client.calls) == 2
    assert not outcome.warnings
    assert "WARNING" not in outcome.text
    assert len(outcome.requests) == 2
    assert all(record.purpose == SMOOTHING_PURPOSE for record in outcome.requests)


def test_a_part_that_misses_a_turn_twice_shows_a_warning_at_that_place():
    transcript = _transcript(("0", "first thing said"), ("1", "second thing said"))
    client = FakeClient(lambda turns, _: [(n, t) for n, _s, t in turns][:1])

    outcome = _smoother(client).smooth(transcript)

    assert len(client.calls) == 2
    assert len(outcome.warnings) == 1
    assert "left out turn 2" in outcome.warnings[0]
    body = outcome.text.split("\n\n", 1)[1]
    assert body.startswith("[WARNING:")
    # The turn the model left out keeps its literal words.
    assert "second thing said" in outcome.text


def test_a_failed_request_is_retried_on_its_own():
    transcript = _transcript(("0", "one two three"))
    attempts = {"count": 0}

    def answer(turns, _):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ConnectionError("network down")
        return [(n, t) for n, _s, t in turns]

    client = FakeClient(answer)
    outcome = _smoother(client).smooth(transcript)

    assert len(client.calls) == 2
    assert not outcome.warnings
    assert not outcome.requests[0].succeeded
    assert outcome.requests[1].succeeded


def test_a_long_transcript_is_split_between_turns_and_the_parts_are_sent_together():
    turns = [("0" if index % 2 == 0 else "1", " ".join(["word"] * 30)) for index in range(8)]
    transcript = _transcript(*turns)
    client = FakeClient(lambda turns, _: [(n, t) for n, _s, t in turns], delay=0.2)

    outcome = _smoother(client, part_words=60, parallel_requests=4).smooth(transcript)

    assert len(client.calls) == 4
    assert client.most_active > 1
    sent = [[n for n, _s, _t in _turns_in(call)] for call in client.calls]
    assert sorted(sum(sent, [])) == list(range(1, 9))
    assert all(len(numbers) == 2 for numbers in sent)
    assert outcome.succeeded and not outcome.warnings


def test_each_part_after_the_first_carries_the_previous_turn_as_context_only():
    turns = [Turn(n, "Adrienne", "a b c") for n in range(1, 5)]
    parts = split_into_parts(turns, part_words=6)

    assert [tuple(t.number for t in part.turns) for part in parts] == [(1, 2), (3, 4)]
    assert parts[0].context is None
    assert parts[1].context == turns[1]


def test_a_turn_longer_than_a_part_is_never_cut():
    turns = [Turn(1, "A", "x " * 50), Turn(2, "B", "y")]
    parts = split_into_parts(turns, part_words=10)
    assert [len(part.turns) for part in parts] == [1, 1]


def test_turns_show_unresolved_words_as_transcript_txt_does():
    transcript = _transcript(("0", "on the"))
    transcript.tokens.append(_uncertain("board", "place"))
    assert build_turns(transcript) == [
        Turn(1, "Adrienne", "on the [UNCERTAIN: board / place]")
    ]


# -- The checks ------------------------------------------------------------


def test_a_changed_marker_fails_the_check():
    literal = Turn(1, "A", "on the [UNCERTAIN: board / place] today")
    assert check_turn(literal, "on the board today") is not None
    assert check_turn(literal, "On the [UNCERTAIN: board / place] today.") is None


def test_a_turn_that_loses_its_words_fails_and_a_short_reply_may_be_dropped():
    literal = Turn(1, "A", "We drove to Pretoria with the cattle in the morning")
    assert check_turn(literal, "We drove.") is not None
    assert check_turn(Turn(2, "B", "Mm-hmm, yeah"), "") is None
    assert check_turn(literal, "") is not None


def test_a_marker_the_model_changes_twice_keeps_its_literal_text():
    transcript = _transcript(("0", "um we put it on the"))
    transcript.tokens.append(_uncertain("board", "place"))
    client = FakeClient(lambda turns, _: [(n, "we put it on the board") for n, _s, _t in turns])

    outcome = _smoother(client).smooth(transcript)

    assert outcome.warnings
    assert "[UNCERTAIN: board / place]" in outcome.text


def test_no_key_means_no_request_and_a_plain_reason():
    client = FakeClient(lambda turns, _: [])
    outcome = Smoother(model="gpt-6.1-sol", client=client).smooth(_transcript(("0", "hi")))
    assert outcome.error is not None and "API key" in outcome.error
    assert not client.calls


def test_the_api_key_is_not_in_the_request_record():
    client = FakeClient(lambda turns, _: [(n, t) for n, _s, t in turns])
    outcome = Smoother(
        api_key="sk-secret",
        model="gpt-6.1-sol",
        parameters={"api_key": "sk-secret"},
        client=client,
    ).smooth(_transcript(("0", "hello")))
    assert "sk-secret" not in repr(outcome.requests)


# -- The settings ----------------------------------------------------------


def test_the_smoothing_settings_have_their_defaults():
    settings = Settings().transcription.smoothing
    assert settings.run_after_transcription is True
    assert settings.model == "gpt-6.1-sol"
    assert settings.reasoning_effort == "low"
    assert settings.prompt == DEFAULT_SMOOTHING_PROMPT
    assert "[UNCERTAIN: board / place]" in DEFAULT_SMOOTHING_PROMPT


def test_the_smoothing_settings_survive_a_save_and_load():
    settings = Settings()
    settings.transcription.smoothing = SmoothingSettings(
        run_after_transcription=False, model="other", reasoning_effort="", prompt="Be brief.\n"
    )
    loaded = Settings.from_dict(json.loads(json.dumps(settings.to_dict())))
    assert loaded.transcription.smoothing == settings.transcription.smoothing


def test_an_older_settings_file_and_a_blank_prompt_take_the_defaults():
    loaded = Settings.from_dict({"transcription": {"smoothing": {"prompt": "  "}}})
    assert loaded.transcription.smoothing.prompt == DEFAULT_SMOOTHING_PROMPT
    assert Settings.from_dict({}).transcription.smoothing == SmoothingSettings()


def test_when_every_request_fails_nothing_is_written_and_the_outcome_says_why():
    def answer(turns, _):
        error = RuntimeError("unauthorised")
        error.status_code = 401
        raise error

    store = FakeStore()
    outcome = write_smooth_transcript(
        _transcript(("0", "um we we were there")), store, _smoother(FakeClient(answer))
    )

    assert not outcome.succeeded
    assert outcome.error is not None and "did not answer" in outcome.error
    assert "401" in outcome.error
    assert SMOOTH_EXPORT_NAME not in store.written


def test_a_part_with_no_answer_says_so_while_the_other_parts_are_kept():
    turns = [("0", "first part words here"), ("1", "second part words here")]

    def answer(turns, _):
        if turns[0][0] == 2:
            raise ConnectionError("network down")
        return [(n, t) for n, _s, t in turns]

    outcome = _smoother(FakeClient(answer), part_words=4).smooth(_transcript(*turns))

    assert outcome.succeeded
    assert len(outcome.warnings) == 1
    assert "did not answer for turn 2" in outcome.warnings[0]
    assert "second part words here" in outcome.text


def test_a_turn_that_loses_too_many_words_twice_keeps_the_model_text_with_a_warning():
    literal = "We drove to Pretoria with the cattle in the morning"
    client = FakeClient(lambda turns, _: [(n, "We drove.") for n, _s, _t in turns])

    outcome = _smoother(client).smooth(_transcript(("0", literal)))

    assert len(client.calls) == 2
    assert len(outcome.warnings) == 1
    assert "lost too many of its words" in outcome.warnings[0]
    assert "We drove." in outcome.text
    assert literal not in outcome.text


def test_a_failed_check_then_a_failed_retry_gives_the_check_as_the_reason():
    def answer(turns, index):
        if index == 1:
            raise ConnectionError("network down")
        return [(n, "We drove.") for n, _s, _t in turns]

    outcome = _smoother(FakeClient(answer)).smooth(
        _transcript(("0", "We drove to Pretoria with the cattle in the morning"))
    )

    assert len(outcome.warnings) == 1
    assert "lost too many of its words" in outcome.warnings[0]
    assert "could not be reached" not in outcome.warnings[0]


# -- A folder's own prompt ---------------------------------------------------


def _echo(turns, _):
    return [(number, text) for number, _speaker, text in turns]


def test_a_folder_with_no_prompt_of_its_own_uses_the_apps_prompt(tmp_path):
    settings = Settings().transcription
    settings.smoothing.prompt = "The app's prompt.\n"
    settings.openai_adjudication.api_key = "sk-test"
    client = FakeClient(_echo)
    smoother_for(settings, tmp_path, client=client).smooth(_transcript(("0", "hello there")))
    assert client.calls[0]["instructions"].startswith("The app's prompt.")


def test_a_saved_folder_prompt_is_used_for_that_folder(tmp_path):
    store = ProjectStore(tmp_path)
    state = store.load()
    state.settings.smoothing_prompt = "Skryf in Afrikaans.\n"
    assert store.save(state)
    client = FakeClient(_echo)
    settings = Settings().transcription
    settings.openai_adjudication.api_key = "sk-test"
    smoother = smoother_for(settings, tmp_path, client=client)
    smoother.smooth(_transcript(("0", "hello there")))
    instructions = client.calls[0]["instructions"]
    assert instructions.startswith("Skryf in Afrikaans.")
    assert DEFAULT_SMOOTHING_PROMPT.strip() not in instructions
    assert FORMAT_INSTRUCTIONS in instructions


def test_a_blank_folder_prompt_counts_as_none():
    assert style_prompt_for("App.", "  \n") == "App."
    assert style_prompt_for("App.", "") == "App."
    assert style_prompt_for("App.", "Folder.\n") == "Folder.\n"
