"""The smooth transcript: an edited copy of the literal one that is easy to read.

The literal transcript keeps every filler, stutter and false start, which is
right for a record and tiring to read. This stage asks the OpenAI language
model to edit it into clean text, turn by turn, and writes the result to
``transcript-smooth.txt`` beside ``transcript.txt``. The literal transcript
is never changed.

The person may edit the style prompt. The format the application depends on
is a fixed part that is always added after it, because the answer must come
back turn by turn, by number, with every ``[UNCERTAIN: ...]`` marker as it
was, or it cannot be checked and the file cannot be rebuilt from it.

A long transcript is sent in parts of about 2,000 words, always cut between
turns, and the parts are sent at the same time. Each part carries the last
turn of the part before it as context only, so the model knows what was
being talked about. Each answer is checked: every turn is there, every
marker is unchanged, and almost all the meaningful words of each turn are
kept. A part that fails is tried once more on its own. If it fails again,
the model's text is kept, and a warning line in the file says where to
compare it with the literal transcript.

Nothing here depends on Qt, and the client library is imported only when a
request is made.
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from vox_verbatim.transcription.adjudication import (
    AdjudicationUnavailable,
    _answer_text,
    _stopped_early_errors,
    _text_field,
    build_openai_client,
    without_credentials,
)
from vox_verbatim.transcription.exports import (
    DEFAULT_WRAP_WIDTH,
    _join_tokens,
    _speaker_name,
    _turns,
)
from vox_verbatim.transcription.model import Provider, ProviderRequestRecord, Transcript
from vox_verbatim.transcription.providers.base import describe_api_key_characters

_log = logging.getLogger(__name__)

#: What the smooth transcript is called inside the transcript folder.
SMOOTH_EXPORT_NAME = "transcript-smooth.txt"

#: What these requests are called in provenance.
SMOOTHING_PURPOSE = "smoothing"

#: The name the strict schema is registered under on the request.
SCHEMA_NAME = "transcript_smoothing"

#: About how many words one request carries. Parts are always cut between
#: turns, so a part can be longer than this by up to one turn.
DEFAULT_PART_WORDS = 2_000

#: How many parts are sent at the same time.
DEFAULT_PARALLEL_REQUESTS = 4

#: How many times a part is sent before its answer is kept with a warning.
ATTEMPTS_PER_PART = 2

#: The share of a turn's meaningful words the smooth text must keep.
#: Not yet tuned on a real recording: too strict puts a warning on every
#: part, and too loose misses a dropped sentence.
WORDS_KEPT_THRESHOLD = 0.8

#: A dropped turn may hold at most this many meaningful words. The style
#: prompt asks the model to drop only short replies such as "Mm-hmm"; a
#: dropped turn that said more than this lost something.
MOST_MEANINGFUL_WORDS_IN_A_DROPPED_TURN = 2

#: The fixed format part. It is added after the style prompt, and the
#: person cannot edit it, because the checks below depend on it.
FORMAT_INSTRUCTIONS = """\
Format (fixed; the application depends on it):
- The input lists numbered turns. Return every numbered turn exactly once, by its number, in the order given.
- Put only the edited words of the turn in "text". Do not add the speaker's name or the turn number.
- Copy every marker of the form [UNCERTAIN: a / b] exactly as it is, character for character, in the same turn.
- To leave out a turn, return its number with an empty text. Never leave out a turn that holds an [UNCERTAIN: ...] marker.
- Do not join turns, split them or move words from one turn to another.
- A turn shown as context is only there so you know what came before. Do not return it.
"""

#: The strict schema the answer must follow.
SMOOTHING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["number", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["turns"],
    "additionalProperties": False,
}

_MARKER = re.compile(r"\[UNCERTAIN:[^\]]*\]")
_WORD = re.compile(r"[\w'-]+")

#: Words that carry no meaning of their own for the word-kept check: the
#: fillers the style prompt removes, and the small words a grammar fix may
#: change. Losing one of these is never a reason to warn.
_NOT_MEANINGFUL = frozenset(
    """
    a ah an and are as at be but by er erm for from had has have he her him his
    hm hmm huh i i'm in is it it's its like me mhm mm mm-hmm my no of oh ok okay
    on or our so that the their them then there they this to uh uh-huh um us
    was we well were what yeah yes you you're your know just right
    """.split()
)


# -- Turns and parts ----------------------------------------------------


@dataclass(frozen=True)
class Turn:
    """One turn of the literal transcript: who spoke, and what they said."""

    number: int
    speaker: str
    text: str


@dataclass(frozen=True)
class Part:
    """The turns sent in one request, and the turn before them as context."""

    turns: tuple[Turn, ...]
    context: Turn | None = None


def build_turns(transcript: Transcript) -> list[Turn]:
    """The turns of the literal transcript, numbered from 1.

    The text is what ``transcript.txt`` shows, so an unresolved word reads
    as ``[UNCERTAIN: a / b]`` here too.
    """
    turns: list[Turn] = []
    for speaker_id, tokens in _turns(transcript):
        text = _join_tokens(tokens)
        if not text:
            continue
        turns.append(
            Turn(
                number=len(turns) + 1,
                speaker=_speaker_name(transcript, speaker_id),
                text=text,
            )
        )
    return turns


def split_into_parts(turns: Sequence[Turn], part_words: int = DEFAULT_PART_WORDS) -> list[Part]:
    """Cut the turns into parts of about ``part_words`` words, between turns.

    A turn is never cut, so a single turn longer than a part is a part of
    its own. Each part after the first carries the last turn of the part
    before it as context.
    """
    parts: list[list[Turn]] = []
    words = 0
    for turn in turns:
        count = len(turn.text.split())
        if parts and words + count <= part_words:
            parts[-1].append(turn)
            words += count
            continue
        parts.append([turn])
        words = count
    return [
        Part(turns=tuple(group), context=parts[index - 1][-1] if index else None)
        for index, group in enumerate(parts)
    ]


def build_part_prompt(part: Part) -> str:
    """The user message for one part."""
    lines: list[str] = []
    if part.context is not None:
        lines.append("Context only, from the end of the previous part. Do not return it:")
        lines.append(f"{part.context.speaker}: {part.context.text}")
        lines.append("")
    lines.append("Turns to edit:")
    for turn in part.turns:
        lines.append(f"Turn {turn.number} ({turn.speaker}): {turn.text}")
    return "\n".join(lines)


def instructions_for(style_prompt: str) -> str:
    """The style prompt the person chose, followed by the fixed format part."""
    return style_prompt.rstrip() + "\n\n" + FORMAT_INSTRUCTIONS


# -- Checking an answer -------------------------------------------------


def markers_in(text: str) -> Counter[str]:
    """Every ``[UNCERTAIN: ...]`` marker in ``text``, with how often it occurs."""
    return Counter(_MARKER.findall(text))


def meaningful_words(text: str) -> set[str]:
    """The words of ``text`` that the word-kept check counts.

    Markers are left out, because they are checked on their own, and so are
    the fillers and small words in :data:`_NOT_MEANINGFUL`.
    """
    plain = _MARKER.sub(" ", text).replace("’", "'").lower()
    words = {word.strip("'-") for word in _WORD.findall(plain)}
    return {word for word in words if word and word not in _NOT_MEANINGFUL}


def check_turn(literal: Turn, smooth: str) -> str | None:
    """Say what is wrong with the smooth text of one turn, or None."""
    if markers_in(literal.text) != markers_in(smooth):
        return f"turn {literal.number} changed an [UNCERTAIN] marker"
    kept_from = meaningful_words(literal.text)
    if not smooth.strip():
        if len(kept_from) > MOST_MEANINGFUL_WORDS_IN_A_DROPPED_TURN:
            return f"turn {literal.number} was left out but said more than a short reply"
        return None
    if not kept_from:
        return None
    kept = len(kept_from & meaningful_words(smooth))
    if kept / len(kept_from) < WORDS_KEPT_THRESHOLD:
        return f"turn {literal.number} lost too many of its words"
    return None


def check_answer(part: Part, answer: dict[int, str]) -> list[str]:
    """Every problem with the answer for one part. Empty means it passed."""
    problems: list[str] = []
    wanted = {turn.number for turn in part.turns}
    missing = sorted(wanted - set(answer))
    if missing:
        problems.append("the answer left out turn " + ", ".join(str(n) for n in missing))
    extra = sorted(set(answer) - wanted)
    if extra:
        problems.append("the answer held turn " + ", ".join(str(n) for n in extra))
    for turn in part.turns:
        if turn.number in answer:
            problem = check_turn(turn, answer[turn.number])
            if problem is not None:
                problems.append(problem)
    return problems


def _loses_words_outright(literal: Turn, smooth: str) -> bool:
    """Whether the smooth text changed a marker or emptied a turn that said something."""
    if markers_in(literal.text) != markers_in(smooth):
        return True
    return not smooth.strip() and check_turn(literal, smooth) is not None


def read_answer(response: Any) -> tuple[dict[int, str], str | None]:
    """The text of each turn by its number, or why the answer could not be read."""
    text = _answer_text(response)
    if not text:
        return {}, "the answer was empty"
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return {}, "the answer was not valid JSON"
    turns = parsed.get("turns") if isinstance(parsed, dict) else None
    if not isinstance(turns, list):
        return {}, "the answer had no list of turns"
    answer: dict[int, str] = {}
    duplicates: list[int] = []
    for item in turns:
        if not isinstance(item, dict):
            return {}, "the answer held a turn that was not an object"
        number = item.get("number")
        text_value = item.get("text")
        if isinstance(number, bool) or not isinstance(number, int):
            return {}, "the answer held a turn with no number"
        if not isinstance(text_value, str):
            return {}, "the answer held a turn with no text"
        if number in answer:
            duplicates.append(number)
        answer[number] = text_value.strip()
    if duplicates:
        return answer, "the answer gave turn " + ", ".join(
            str(n) for n in sorted(set(duplicates))
        ) + " more than once"
    return answer, None


# -- The result ---------------------------------------------------------


@dataclass
class PartResult:
    """What one part came back as, after its checks and any retry."""

    part: Part
    texts: dict[int, str]
    warning: str | None = None
    requests: list[ProviderRequestRecord] = field(default_factory=list)

    @property
    def answered(self) -> bool:
        """Whether any request for this part got an answer back."""
        return any(record.succeeded for record in self.requests)


@dataclass
class SmoothOutcome:
    """The smooth transcript, or why there is none."""

    text: str = ""
    warnings: list[str] = field(default_factory=list)
    requests: list[ProviderRequestRecord] = field(default_factory=list)
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and bool(self.text)


# -- The engine ---------------------------------------------------------


class Smoother:
    """Edits a finished transcript into the smooth copy with the OpenAI model.

    Everything it needs arrives through the constructor, as with the
    adjudicator: a caller that reads the settings builds this, and a test
    hands in a fake client.
    """

    provider = Provider.OPENAI

    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        reasoning_effort: str = "",
        style_prompt: str = "",
        parameters: dict[str, Any] | None = None,
        *,
        timeout_seconds: float = 600.0,
        maximum_retries: int = 2,
        part_words: int = DEFAULT_PART_WORDS,
        parallel_requests: int = DEFAULT_PARALLEL_REQUESTS,
        client: Any | None = None,
    ) -> None:
        self._api_key = api_key or ""
        self._model = (model or "").strip()
        self._reasoning_effort = (reasoning_effort or "").strip()
        self._style_prompt = style_prompt or ""
        self._parameters = dict(parameters or {})
        self._timeout_seconds = timeout_seconds
        self._maximum_retries = maximum_retries
        self._part_words = max(1, int(part_words))
        self._parallel_requests = max(1, int(parallel_requests))
        self._client = client

    def describe_configuration_problem(self) -> str | None:
        """What is missing, in words a person can act on, or None if nothing is."""
        if not self._api_key.strip():
            return "no OpenAI API key has been entered"
        key_problem = describe_api_key_characters("OpenAI", self._api_key)
        if key_problem is not None:
            return key_problem
        if not self._model:
            return "no smoothing model has been entered"
        return None

    def smooth(self, transcript: Transcript) -> SmoothOutcome:
        """Make the smooth transcript. Never raises.

        The outcome holds the whole file text, or an error that says why
        there is none.
        """
        problem = self.describe_configuration_problem()
        if problem is not None:
            return SmoothOutcome(error=f"The smooth transcript was not made: {problem}.")
        turns = build_turns(transcript)
        if not turns:
            return SmoothOutcome(error="The smooth transcript was not made: there are no words.")
        try:
            client = self._resolve_client()
        except AdjudicationUnavailable as error:
            return SmoothOutcome(error=str(error))

        parts = split_into_parts(turns, self._part_words)
        workers = min(len(parts), self._parallel_requests)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda part: self._smooth_part(client, part), parts))

        outcome = SmoothOutcome()
        if not any(result.answered for result in results):
            # Every request failed, so there is no edit at all. Writing the
            # literal words under an "Edited" header would replace an earlier
            # good smooth file with nothing better than transcript.txt.
            outcome.requests = [record for result in results for record in result.requests]
            reason = outcome.requests[-1].error if outcome.requests else None
            outcome.error = "The smooth transcript was not made: the language model did not answer"
            outcome.error += f" ({reason})." if reason else "."
            return outcome
        for result in results:
            outcome.requests.extend(result.requests)
            if result.warning is not None:
                outcome.warnings.append(result.warning)
        outcome.text = render_smooth_text(transcript, results, self._model)
        return outcome

    # -- One part -----------------------------------------------------------

    def _smooth_part(self, client: Any, part: Part) -> PartResult:
        """Send one part, check the answer, and try once more if it fails."""
        result = PartResult(part=part, texts={})
        problems: list[str] = []
        for _attempt in range(ATTEMPTS_PER_PART):
            answer, problems, record = self._ask(client, part)
            result.requests.append(record)
            if answer:
                result.texts = answer
            if not problems:
                result.warning = None
                return result
        # The model's text is kept, except where keeping it would lose words
        # outright: a turn it left out or emptied that said something, or a
        # turn whose [UNCERTAIN] markers it changed. Those keep their literal
        # text, so every marker reaches the file.
        for turn in part.turns:
            smooth = result.texts.get(turn.number)
            if smooth is None or _loses_words_outright(turn, smooth):
                result.texts[turn.number] = turn.text
        first, last = part.turns[0].number, part.turns[-1].number
        where = f"turn {first}" if first == last else f"turns {first} to {last}"
        if result.answered:
            result.warning = (
                f"The language model's edit of {where} did not pass its check: "
                + "; ".join(problems)
                + ". Compare it with transcript.txt."
            )
        else:
            result.warning = (
                f"The language model did not answer for {where} ("
                + "; ".join(problems)
                + "), so these turns are the literal words."
            )
        _log.info("A smoothing part failed its check twice: %s", result.warning)
        return result

    def _ask(
        self, client: Any, part: Part
    ) -> tuple[dict[int, str], list[str], ProviderRequestRecord]:
        """One request for one part: the answer, its problems and the record."""
        arguments = self._request_arguments(build_part_prompt(part))
        started_at = datetime.now()
        try:
            response = client.responses.parse(**arguments)
        except _stopped_early_errors() as error:
            reason = f"the model stopped before it finished answering ({type(error).__name__})"
            return {}, [reason], self._record(arguments, None, started_at, part, reason)
        except Exception as error:  # noqa: BLE001 - a failed part is retried, then warned
            _log.warning(
                "The smoothing model did not answer for one part: %s",
                type(error).__name__,
                exc_info=error,
            )
            reason = _failure_reason(error)
            return {}, [reason], self._record(arguments, None, started_at, part, reason)
        record = self._record(arguments, response, started_at, part, None)
        answer, malformed = read_answer(response)
        problems = [malformed] if malformed is not None else []
        problems.extend(check_answer(part, answer))
        return answer, problems, record

    def _request_arguments(self, prompt: str) -> dict[str, Any]:
        """The keyword arguments for one Responses call.

        The extras go on first, so none of them can displace the model, the
        instructions or the strict schema.
        """
        arguments: dict[str, Any] = dict(self._parameters)
        arguments.update(
            {
                "model": self._model,
                "instructions": instructions_for(self._style_prompt),
                "input": [{"role": "user", "content": prompt}],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": SCHEMA_NAME,
                        "strict": True,
                        "schema": SMOOTHING_SCHEMA,
                    }
                },
                "timeout": self._timeout_seconds,
            }
        )
        if self._reasoning_effort:
            arguments["reasoning"] = {"effort": self._reasoning_effort}
        return arguments

    def _record(
        self,
        arguments: dict[str, Any],
        response: Any,
        started_at: datetime,
        part: Part,
        error: str | None,
    ) -> ProviderRequestRecord:
        """Note what was asked, without the transcript's words or the key."""
        recorded: dict[str, Any] = {
            key: value
            for key, value in arguments.items()
            if key not in ("input", "instructions", "text")
        }
        recorded["structured_output"] = SCHEMA_NAME
        recorded["turn_count"] = len(part.turns)
        recorded["reasoning"] = arguments.get("reasoning", "not configured")
        resolved = _text_field(response, "model")
        response_id = _text_field(response, "id")
        if response_id:
            recorded["response_id"] = response_id
        return ProviderRequestRecord(
            provider=self.provider,
            model_identifier=resolved or self._model,
            model_version=resolved or None,
            request_parameters=without_credentials(recorded),
            processing_seconds=(datetime.now() - started_at).total_seconds(),
            started_at=started_at.isoformat(timespec="seconds"),
            succeeded=error is None,
            error=error,
            purpose=SMOOTHING_PURPOSE,
        )

    def _resolve_client(self) -> Any:
        if self._client is None:
            self._client = build_openai_client(
                self._api_key, self._maximum_retries, "Smoothing"
            )
        return self._client


# -- The file -----------------------------------------------------------


def render_smooth_text(
    transcript: Transcript,
    results: Sequence[PartResult],
    model: str = "",
    width: int = DEFAULT_WRAP_WIDTH,
) -> str:
    """The text of ``transcript-smooth.txt``.

    Each speaker's name stands above their text, with no times. Dropped
    turns are removed first, and then consecutive turns by the same speaker
    are joined into one. A warning line stands at the start of a part whose
    answer failed its check twice, and a turn the answer left out keeps its
    literal text there.
    """
    width = max(20, int(width))
    lines: list[str] = [f"Recording: {transcript.recording_name}"]
    edited_by = f" by {model}" if model else ""
    lines.append(
        f"Edited{edited_by} for easy reading. transcript.txt holds the literal words."
    )
    lines.append("")

    # Each entry is a warning line (speaker None) or a speaker and their texts.
    blocks: list[tuple[str | None, list[str]]] = []
    for result in results:
        if result.warning is not None:
            blocks.append((None, [f"[WARNING: {result.warning}]"]))
        for turn in result.part.turns:
            text = result.texts.get(turn.number, "")
            if not text:
                continue
            if blocks and blocks[-1][0] == turn.speaker:
                blocks[-1][1].append(text)
            else:
                blocks.append((turn.speaker, [text]))

    for speaker, texts in blocks:
        if speaker is None:
            lines.extend(textwrap.wrap(texts[0], width=width))
        else:
            lines.append(f"{speaker}:")
            lines.extend(textwrap.wrap(" ".join(texts), width=width) or [""])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_smooth_transcript(transcript: Transcript, store: Any, smoother: Smoother) -> SmoothOutcome:
    """Make the smooth transcript and write it into the transcript folder.

    ``store`` is the recording's :class:`~vox_verbatim.transcription.store.TranscriptStore`.
    Nothing is written when there is no text; the outcome says why.
    """
    outcome = smoother.smooth(transcript)
    if not outcome.text:
        return outcome
    if store.write_export(SMOOTH_EXPORT_NAME, outcome.text) is None:
        outcome.error = f"The {SMOOTH_EXPORT_NAME} file could not be written."
    return outcome


def _failure_reason(error: Exception) -> str:
    status = getattr(error, "status_code", None)
    if status is not None:
        return f"the model answered with an error ({status})"
    return f"the model could not be reached ({type(error).__name__})"
