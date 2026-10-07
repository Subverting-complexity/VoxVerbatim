"""Letting a language model choose between readings, and never write one.

A handful of words in a recording are genuinely contested. The services
disagree, the deterministic rules cannot separate the readings, and the
evidence that would settle it is the sort a person weighs rather than the
sort a rule counts. That is the narrow job this module hands to a language
model, and nothing wider.

The care taken here is not caution for its own sake. It is a response to a
specific and well-earned distrust. A language model asked to improve a
transcript will improve it. It will straighten a broken sentence, drop the
"um", turn a false start into the sentence the speaker meant to say, and
correct an agreement error the speaker actually made. Every one of those
edits reads better than what it replaced, and every one of them is wrong
here, because a verbatim transcript is evidence about a recording rather
than prose. Worse, the damage is invisible afterwards: the tidied sentence
still carries the timestamps of the untidy one, so the words and the audio
have quietly come apart and nothing in the file says so. A person reading
the transcript later has no way to tell.

So the model is never asked to produce text. It is asked to choose. The
candidates it chooses between already exist, having come from the services
themselves or from the known vocabulary. The spans it chooses over already
exist, having been identified as disputed before the model was called. Its
answer arrives as structured edits against those spans, enforced by the
API's own strict schema, and every edit is then checked here against the
spans and the candidates before a single word of the transcript changes.
An edit that fails a check is refused outright rather than repaired,
because a model that has strayed once has no claim on being trusted about
what it meant.

Two of those checks deserve saying out loud. The first is that a
replacement must be a reading somebody actually offered. A word invented
from nothing is exactly the failure this whole module exists to prevent,
and no amount of plausibility earns it a way in. The second is that a
high-risk value - an amount, a date, an account number - is never settled
here at all. The model has only plausibility to offer, and "50,000" and
"15,000" are equally plausible sentences with very different consequences.
Such a span keeps whatever reconciliation left it, is flagged for a person,
and carries the model's opinion alongside as something to read rather than
something to believe. That rule lives in the validation rather than only in
the prompt, because a prompt is a request and validation is a guarantee.

The stage is an enhancement and never a requirement. A missing package, a
missing key, a rate limit, a refusal, a malformed answer: each of them
leaves the disputed words exactly as reconciliation left them, flagged for
human review, and the transcript is finished without them. Nothing here
raises.

OpenAI is the only provider this role will ever have, which the
specification states plainly, so there is no abstraction here inviting a
second one. A single honestly named module is the truthful shape of a
decision that has already been taken.

Nothing here depends on Qt, and the client library is imported inside the
method that needs it, so a machine without it loses adjudication rather
than failing to start.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Sequence

from vox_verbatim.transcription.model import (
    Confidence,
    FinalToken,
    Language,
    Provider,
    ProviderRequestRecord,
    ProviderToken,
    ReviewReason,
    RiskCategory,
)
from vox_verbatim.transcription.normalise import are_equivalent, normalise
from vox_verbatim.transcription.providers.base import describe_api_key_characters

_log = logging.getLogger(__name__)

#: The package that must be installed for adjudication to work at all.
PACKAGE = "openai"

#: What these requests are called in provenance. Adjudication requests sit
#: beside transcription and escalation requests in the same list, and only
#: this word tells them apart.
ADJUDICATION_PURPOSE = "adjudication"

#: The efforts the reasoning models document today. The configured value is
#: not checked against this list, because the model name is editable text
#: and a newer model may accept an effort nobody has heard of yet. An
#: unknown value is passed through and the far end rejects it with a clear
#: message, which is better than this module silently dropping something the
#: user deliberately typed.
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

#: The name the strict schema is registered under on the request.
SCHEMA_NAME = "transcript_adjudication"

#: Settings keys that could hold a credential. Anything named like one of
#: these is dropped before a request is written into provenance, because a
#: transcript folder is copied around and read by people.
_CREDENTIAL_NAMES = ("api_key", "apikey", "key", "token", "secret", "password", "authorization")
"""Matched against the end of each word of a parameter name, never anywhere
inside it: ``max_output_tokens`` contains "token" but is a length, and a
provenance record without it cannot say what was sent. Words are split at
punctuation and at camelCase, and a word only has to end with a name, so
``accessToken``, ``openaiApiKey`` and ``sessiontoken`` are still removed."""


class AdjudicationUnavailable(Exception):
    """The client library is not installed, so this stage cannot run.

    Caught inside :meth:`Adjudicator.adjudicate` and turned into an outcome
    like every other failure. It exists as a type only so that the reason
    survives the trip out of the place that discovered it.
    """


class EditAction(str, Enum):
    """The only three things the model is allowed to say about a span."""

    KEEP = "keep"
    """The text reconciliation chose is the right reading."""

    REPLACE = "replace"
    """Another offered reading is the right one."""

    DECLINE = "decline"
    """The evidence does not settle it, and a person should look."""


class RefusalReason(str, Enum):
    """Why an edit the model returned was thrown away rather than applied.

    These are counted and reported rather than merely logged. A model that
    is routinely refused for the same reason is telling us something about
    the prompt, and a model that is refused for inventing words is telling
    us something rather more serious.
    """

    SPAN_NOT_IN_DISPUTE = "span_not_in_dispute"
    """The edit named a span that was not one of the ones asked about."""

    WRONG_ORIGINAL_TEXT = "wrong_original_text"
    """The edit described the span as saying something it does not say."""

    TEXT_NOT_OFFERED = "text_not_offered"
    """The replacement is not a candidate and not in the known vocabulary."""

    WORD_COUNT_CHANGED = "word_count_changed"
    """The replacement has a different number of words from the span."""

    IDENTITY_WOULD_CHANGE = "identity_would_change"
    """Applying it would have moved a time, a speaker or an identifier."""

    HIGH_RISK_ENTITY = "high_risk_entity"
    """A value that must not be settled by plausibility alone."""

    MALFORMED_ANSWER = "malformed_answer"
    """The answer was not the shape the schema asked for."""

    @property
    def display_name(self) -> str:
        return _REFUSAL_DISPLAY_NAMES[self]


_REFUSAL_DISPLAY_NAMES: dict[RefusalReason, str] = {
    RefusalReason.SPAN_NOT_IN_DISPUTE: "It edited a word it was not asked about",
    RefusalReason.WRONG_ORIGINAL_TEXT: "It misquoted the words it was editing",
    RefusalReason.TEXT_NOT_OFFERED: "It suggested a word nobody heard",
    RefusalReason.WORD_COUNT_CHANGED: "It changed how many words the span holds",
    RefusalReason.IDENTITY_WOULD_CHANGE: "It would have moved the timing or the speaker",
    RefusalReason.HIGH_RISK_ENTITY: "A value that must not be guessed",
    RefusalReason.MALFORMED_ANSWER: "The answer did not have the required shape",
}


#: The shape the model must answer in, enforced by the API rather than only
#: asked for. Strict structured output requires every property to be listed
#: in ``required`` and additional properties to be forbidden, which is
#: convenient here: there is no field for a timestamp, a speaker or an
#: identifier, so the model has nowhere to put one even if it wanted to.
#:
#: It is written out as plain JSON Schema rather than generated from a
#: pydantic model so that importing this module never needs the client
#: library. The API is sent the same thing either way.
ADJUDICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decisions"],
    "properties": {
        "decisions": {
            "type": "array",
            "description": "One decision for each disputed span, and nothing else.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["span_id", "action", "from_text", "to_text", "reason"],
                "properties": {
                    "span_id": {
                        "type": "string",
                        "description": (
                            "The identifier of the disputed span, copied exactly from "
                            "the evidence. Never any other identifier."
                        ),
                    },
                    "action": {
                        "type": "string",
                        "enum": [action.value for action in EditAction],
                        "description": (
                            "keep the current reading, replace it with another offered "
                            "reading, or decline because the evidence does not settle it."
                        ),
                    },
                    "from_text": {
                        "type": "string",
                        "description": "The span's current text, copied exactly.",
                    },
                    "to_text": {
                        "type": "string",
                        "description": (
                            "The chosen reading. For keep and decline, the current text "
                            "again. For replace, one of the candidates offered for this "
                            "span or a term from the known vocabulary, with the same "
                            "number of words as the span."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": "One short sentence saying which evidence decided it.",
                    },
                },
            },
        }
    },
}


#: What the model is told it is for. Deliberately blunt about the things a
#: helpful model would otherwise do, because every one of them has to be
#: refused later and a refusal costs a whole disputed span.
SYSTEM_INSTRUCTIONS = """\
You are settling disagreements between speech-to-text services over a verbatim \
transcript of a real recording.

You are choosing between readings that already exist. You are not writing text, and \
you are not improving it.

Answer exactly once for each disputed span, with one of:
  keep     the reading shown as the current text is right;
  replace  another offered reading is right;
  decline  the evidence does not settle it.

Rules, all of which are checked after you answer:
- A replacement must be one of the candidates offered for that span, or a term from \
the known vocabulary. Never write a word that appears in neither.
- A replacement must have the same number of words as the span it replaces.
- Never change a timestamp, a speaker or an identifier. You are deciding what was \
said and nothing else.
- This is a verbatim transcript. Fillers, false starts, repetitions and ungrammatical \
speech are the point of it. Never tidy them, and never make a sentence read better.
- For a high-risk value, such as money, a date, a time, an account number or a \
quantity, decline unless the evidence is decisive. Plausibility is not evidence.
- Declining is a good answer. A confident wrong answer is not.
"""


# -- What goes in --------------------------------------------------------


@dataclass(frozen=True)
class Dispute:
    """One region deterministic reconciliation could not settle.

    It carries the evidence for that region and nothing else. The whole
    transcript is deliberately not sent: it would cost money on every
    dispute, bury the few lines that actually decide the question, and
    invite the model to range across words nobody asked it about.

    The span is identified by the first of its words, whose identifier
    already exists and is already unique. Giving the model a fresh label
    would let it name a span that does not exist without that being
    obviously wrong; giving it the real identifier means that anything it
    names either is one of the disputes or is out of bounds.
    """

    tokens: tuple[FinalToken, ...]
    """The words in dispute, in transcript order. At least one."""

    backbone_tokens: tuple[ProviderToken, ...] = ()
    """The backbone service's own words for this region, which is where the
    timestamps, the speaker labels and the log probabilities come from."""

    escalation_tokens: tuple[ProviderToken, ...] = ()
    """Second opinions from an escalation service, where one was asked."""

    preceding_text: str = ""
    following_text: str = ""
    """The words either side. A word with no sentence around it cannot be
    judged, and these are what let the model weigh a reading in context
    without being handed the recording."""

    note: str = ""
    """Why reconciliation gave up here, in a few words, where it knows."""

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("A dispute must cover at least one word.")

    @property
    def span_id(self) -> str:
        return self.tokens[0].id

    @property
    def token_ids(self) -> tuple[str, ...]:
        return tuple(token.id for token in self.tokens)

    @property
    def current_text(self) -> str:
        return " ".join(token.text for token in self.tokens if token.text)

    @property
    def candidate_texts(self) -> tuple[str, ...]:
        """Every reading offered for this span, without repeats."""
        texts: list[str] = []
        for token in self.tokens:
            for candidate in token.candidates:
                if candidate.text and candidate.text not in texts:
                    texts.append(candidate.text)
        return tuple(texts)

    @property
    def risk_categories(self) -> tuple[RiskCategory, ...]:
        found: list[RiskCategory] = []
        for token in self.tokens:
            for category in token.risk_categories:
                if category not in found:
                    found.append(category)
        return tuple(found)

    @property
    def is_high_risk(self) -> bool:
        return bool(self.risk_categories)


# -- What comes out ------------------------------------------------------


@dataclass(frozen=True)
class AppliedEdit:
    """One decision that survived validation and changed the transcript."""

    span_id: str
    action: EditAction
    from_text: str
    to_text: str
    reason: str


@dataclass(frozen=True)
class RefusedEdit:
    """One decision that was thrown away, and what was wrong with it."""

    span_id: str
    action: EditAction | None
    to_text: str
    refusal: RefusalReason
    detail: str = ""


@dataclass
class AdjudicationOutcome:
    """Everything that happened in one adjudication pass.

    A failed pass is still an outcome rather than an exception, for the same
    reason a failed provider is: the transcript has to be finished either
    way, and the caller should not have to guard the call.
    """

    applied: tuple[AppliedEdit, ...] = ()
    refused: tuple[RefusedEdit, ...] = ()
    declined: tuple[str, ...] = ()
    """The spans the model would not decide, by span identifier."""

    unanswered: tuple[str, ...] = ()
    """The spans it said nothing about at all, which counts as declining."""

    request: ProviderRequestRecord | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def changed_span_ids(self) -> tuple[str, ...]:
        return tuple(edit.span_id for edit in self.applied if edit.action is EditAction.REPLACE)


# -- The adjudicator -----------------------------------------------------


class Adjudicator:
    """The OpenAI reasoning model, used only to choose between readings.

    Everything it needs arrives through the constructor. It never reads the
    application's settings itself: a caller that does read them builds this,
    which keeps a settings change from reaching in here and lets a test
    build a working adjudicator out of a fake client.

    The model named here is the configured reasoning model and has nothing
    to do with the speech-to-text model. They are two different settings for
    two different jobs, and confusing them would either send audio to a text
    model or ask a transcription model to reason.
    """

    provider = Provider.OPENAI

    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        reasoning_effort: str = "",
        parameters: dict[str, Any] | None = None,
        *,
        timeout_seconds: float = 600.0,
        maximum_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        """Set up the adjudicator.

        ``reasoning_effort`` is left blank by most users, and blank means
        blank: the parameter is left off the request entirely rather than
        being sent as some default nobody chose. Models differ in which
        efforts they accept, and quietly sending one is how a request starts
        failing after a model name is changed.

        ``parameters`` holds the free-form extras from Settings, which is
        what lets a new request parameter be used without an application
        release. They cannot displace the structured-output format or the
        model, because those two are what make the answer safe to apply.

        ``client`` exists so a test, or a caller sharing a connection pool,
        can hand in a ready-made client. Left as ``None`` it means "build a
        real one when first needed", which keeps the library import out of
        application start-up.
        """
        self._api_key = api_key or ""
        self._model = (model or "").strip()
        self._reasoning_effort = (reasoning_effort or "").strip()
        self._parameters = dict(parameters or {})
        self._timeout_seconds = timeout_seconds
        self._maximum_retries = maximum_retries
        self._client = client

    # -- What this is --------------------------------------------------

    @property
    def model_identifier(self) -> str:
        """The model name as configured, before the far end resolves it."""
        return self._model

    @property
    def reasoning_effort(self) -> str:
        """The configured effort, or an empty string when the user left it blank."""
        return self._reasoning_effort

    def is_configured(self) -> bool:
        return self.describe_configuration_problem() is None

    def describe_configuration_problem(self) -> str | None:
        """What is missing, in words a person can act on, or None if nothing is."""
        if not self._api_key.strip():
            return "no API key has been entered for the adjudication model"
        key_problem = describe_api_key_characters("OpenAI", self._api_key)
        if key_problem is not None:
            return key_problem
        if not self._model:
            return "no adjudication model has been entered"
        return None

    # -- Doing the work ------------------------------------------------

    def adjudicate(
        self,
        disputes: Sequence[Dispute],
        *,
        recording_context: str = "",
        vocabulary_terms: Sequence[str] = (),
        languages: Sequence[Language] = (),
    ) -> AdjudicationOutcome:
        """Settle what can be settled, and flag what cannot. Never raises.

        Every dispute leaves this method in one of three states: changed to
        an offered reading the model chose, confirmed as it stood, or left
        exactly as reconciliation had it and flagged for a person. There is
        no fourth state in which the transcript quietly acquires words
        nobody heard.
        """
        if not disputes:
            return AdjudicationOutcome()

        problem = self.describe_configuration_problem()
        if problem is not None:
            # Reported before anything is opened or sent, so that a missing
            # key reads as the plain fact it is rather than as an
            # authentication failure from the far end.
            return self._give_up(disputes, f"Adjudication was skipped: {problem}.")

        prompt = build_prompt(
            disputes,
            recording_context=recording_context,
            vocabulary_terms=vocabulary_terms,
            languages=languages,
        )
        # Debug and no higher, and never in the message of a warning. This
        # string holds what people actually said in the recording, and an
        # ordinary log file is not the place for it.
        _log.debug("Adjudication prompt for %d disputed spans:\n%s", len(disputes), prompt)

        try:
            client = self._resolve_client()
        except AdjudicationUnavailable as error:
            return self._give_up(disputes, str(error))

        started_at = datetime.now()
        arguments = self._request_arguments(prompt)
        # Imported lazily and caught by name, because otherwise the model
        # running out of output tokens part way through a structured answer
        # arrives as a confusing parse error rather than as what it is.
        stopped_early = _stopped_early_errors()
        try:
            response = client.responses.parse(**arguments)
        except stopped_early as error:
            return self._give_up(
                disputes,
                f"The adjudication model stopped before it finished answering: {error}",
                started_at=started_at,
                arguments=arguments,
                dispute_count=len(disputes),
            )
        except Exception as error:  # noqa: BLE001 - every failure becomes an outcome
            _log.warning(
                "The adjudication model did not answer, so the disputed words were left "
                "for review: %s",
                type(error).__name__,
                exc_info=error,
            )
            return self._give_up(
                disputes,
                _failure_sentence(error),
                started_at=started_at,
                arguments=arguments,
                dispute_count=len(disputes),
            )

        record = self._request_record(
            arguments,
            response=response,
            started_at=started_at,
            dispute_count=len(disputes),
            succeeded=True,
            error=None,
        )

        decisions, malformed = _read_decisions(response)
        if malformed is not None:
            outcome = self._give_up(disputes, malformed)
            outcome.request = record
            outcome.refused = (
                RefusedEdit(
                    span_id="",
                    action=None,
                    to_text="",
                    refusal=RefusalReason.MALFORMED_ANSWER,
                    detail=malformed,
                ),
            )
            return outcome

        outcome = apply_decisions(disputes, decisions, vocabulary_terms=vocabulary_terms)
        outcome.request = record
        return outcome

    # -- Building the request -------------------------------------------

    def _request_arguments(self, prompt: str) -> dict[str, Any]:
        """The keyword arguments for one Responses call.

        The extras from Settings go on first and the arguments that matter
        go on top of them. That order is deliberate: a stray parameter must
        not be able to switch off the strict schema, because the schema is
        what makes the answer safe to apply at all.
        """
        arguments: dict[str, Any] = dict(self._parameters)
        arguments.update(
            {
                "model": self._model,
                "instructions": SYSTEM_INSTRUCTIONS,
                "input": [{"role": "user", "content": prompt}],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": SCHEMA_NAME,
                        "strict": True,
                        "schema": ADJUDICATION_SCHEMA,
                    }
                },
                "timeout": self._timeout_seconds,
            }
        )
        if self._reasoning_effort:
            # Nested, because this is the Responses API. Chat Completions
            # takes the same idea as a flat ``reasoning_effort``, and sending
            # that shape here does nothing at all: the request succeeds and
            # the model simply reasons at its default.
            arguments["reasoning"] = {"effort": self._reasoning_effort}
        return arguments

    # -- Recording what happened ----------------------------------------

    def _request_record(
        self,
        arguments: dict[str, Any] | None,
        *,
        response: Any = None,
        started_at: datetime | None,
        dispute_count: int,
        succeeded: bool,
        error: str | None,
    ) -> ProviderRequestRecord:
        """Note what was asked and what came back, without the credential.

        The model name recorded is the one the response reports, which is
        the resolved snapshot rather than the alias that was sent. Both are
        kept, because a transcript that only recorded the alias could not
        later tell a changed model from a changed application.

        The prompt is not recorded. It holds what was said in the recording,
        and provenance is copied around and read by people who have no
        business reading the transcript twice over.
        """
        started = started_at or datetime.now()
        sent = dict(arguments or {})
        recorded: dict[str, Any] = {
            key: value
            for key, value in sent.items()
            # The prompt is the recording's content; the schema is a fixed
            # constant that would swamp the record it was written into.
            if key not in ("input", "instructions", "text")
        }
        recorded["structured_output"] = SCHEMA_NAME
        recorded["disputed_span_count"] = dispute_count
        recorded["reasoning"] = sent.get("reasoning", "not configured")
        resolved = _text_field(response, "model")
        response_id = _text_field(response, "id")
        if response_id:
            recorded["response_id"] = response_id
        return ProviderRequestRecord(
            provider=self.provider,
            model_identifier=resolved or self._model,
            model_version=resolved,
            request_parameters=without_credentials(recorded),
            processing_seconds=(datetime.now() - started).total_seconds(),
            started_at=started.isoformat(timespec="seconds"),
            succeeded=succeeded,
            error=error,
            purpose=ADJUDICATION_PURPOSE,
        )

    def _give_up(
        self,
        disputes: Sequence[Dispute],
        reason: str,
        *,
        started_at: datetime | None = None,
        arguments: dict[str, Any] | None = None,
        dispute_count: int = 0,
    ) -> AdjudicationOutcome:
        """Leave every dispute as it was, flagged for a person, and say why.

        This is the path taken whenever the stage cannot do its job, and it
        is the reason the stage can never cost anybody a transcript.
        """
        _log.info("Adjudication settled nothing this time: %s", reason)
        for dispute in disputes:
            # The model did not decline these; it was never heard. The words
            # keep the confidence reconciliation gave them and are flagged,
            # rather than being marked unresolved, because a request that
            # failed is not evidence that a word is wrong, and marking it so
            # would print every one of them as [UNCERTAIN] in the export.
            _decline(dispute, reason, lower_confidence=False)
        record = None
        if arguments is not None:
            record = self._request_record(
                arguments,
                started_at=started_at,
                dispute_count=dispute_count or len(disputes),
                succeeded=False,
                error=reason,
            )
        return AdjudicationOutcome(
            unanswered=tuple(dispute.span_id for dispute in disputes),
            request=record,
            error=reason,
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
        installed. That machine loses adjudication and is told why.
        """
        try:
            from openai import OpenAI
        except ImportError as error:
            raise AdjudicationUnavailable(
                f"Adjudication needs the {PACKAGE} package, which is not installed. "
                f"Run: pip install {PACKAGE}"
            ) from error
        return OpenAI(api_key=self._api_key, max_retries=self._maximum_retries)


# -- The prompt ----------------------------------------------------------


def build_prompt(
    disputes: Sequence[Dispute],
    *,
    recording_context: str = "",
    vocabulary_terms: Sequence[str] = (),
    languages: Sequence[Language] = (),
) -> str:
    """Lay out the evidence for the disputed regions, and nothing else.

    Everything the specification lists is here: what each service heard, the
    backbone's times, speakers and log probabilities, any second opinion
    that was fetched, the known vocabulary, the words either side, the
    language evidence, how well the words lined up, the high-risk flags and
    where each reading came from. What is not here is the rest of the
    transcript, and that omission is doing real work. It keeps the cost
    proportional to the number of disputes rather than to the length of the
    recording, and it keeps the few lines that decide the question from
    being buried in thousands that do not.
    """
    lines: list[str] = []
    if recording_context.strip():
        lines.append(f"About this recording: {recording_context.strip()}")
    if languages:
        names = ", ".join(language.display_name for language in languages)
        lines.append(f"Languages that may occur: {names}")
    terms = [term for term in vocabulary_terms if term and term.strip()]
    if terms:
        lines.append(f"Known vocabulary: {', '.join(terms)}")
    lines.append(f"Disputed spans: {len(disputes)}. Answer once for each.")

    for dispute in disputes:
        lines.append("")
        lines.extend(_dispute_lines(dispute))
    return "\n".join(lines)


def _dispute_lines(dispute: Dispute) -> list[str]:
    lines = [
        f"--- span {dispute.span_id}",
        f"words in span: {len(dispute.tokens)}",
        f"current text: {dispute.current_text!r}",
    ]
    if dispute.preceding_text.strip():
        lines.append(f"before: ...{dispute.preceding_text.strip()}")
    if dispute.following_text.strip():
        lines.append(f"after: {dispute.following_text.strip()}...")
    if dispute.note.strip():
        lines.append(f"why it is disputed: {dispute.note.strip()}")
    if dispute.is_high_risk:
        lines.append(
            "HIGH RISK: "
            + ", ".join(category.display_name for category in dispute.risk_categories)
            + ". Decline unless the evidence is decisive."
        )

    lines.append("candidates:")
    if dispute.candidate_texts:
        for token in dispute.tokens:
            for candidate in token.candidates:
                lines.append(f"  {_candidate_line(candidate)}")
    else:
        lines.append("  (none beyond the current text)")

    lines.extend(_evidence_lines(dispute))
    return lines


def _candidate_line(candidate: Any) -> str:
    parts = [f"{candidate.text!r}"]
    if candidate.providers:
        parts.append("heard by " + ", ".join(p.display_name for p in candidate.providers))
    if candidate.in_vocabulary:
        parts.append("in the known vocabulary")
    if candidate.score:
        parts.append(f"score {candidate.score:.3f}")
    return "; ".join(parts)


def _evidence_lines(dispute: Dispute) -> list[str]:
    lines: list[str] = []
    for token in dispute.tokens:
        detail = [f"  {token.id}: {token.text!r}"]
        if token.start is not None and token.end is not None:
            detail.append(f"{token.start:.2f}-{token.end:.2f}s")
        if token.speaker:
            detail.append(f"speaker {token.speaker}")
        detail.append(f"alignment {token.alignment_status.value}")
        detail.append(f"timing {token.timing_status.value}")
        detail.append(f"confidence {token.text_confidence.value}")
        if token.text_source is not None:
            detail.append(f"text from {token.text_source.display_name}")
        if token.language is not Language.UNKNOWN:
            detail.append(f"language {token.language.display_name}")
        scores = token.language_evidence.scores
        if scores:
            detail.append(
                "language evidence "
                + ", ".join(f"{key.value} {value:.2f}" for key, value in scores.items())
            )
        lines.append(" ".join(detail))
    if lines:
        lines.insert(0, "words in the span:")

    backbone = [_provider_token_line(token) for token in dispute.backbone_tokens]
    if backbone:
        lines.append("backbone measurements:")
        lines.extend(backbone)
    escalation = [_provider_token_line(token) for token in dispute.escalation_tokens]
    if escalation:
        lines.append("second opinion on this region:")
        lines.extend(escalation)
    return lines


def _provider_token_line(token: ProviderToken) -> str:
    parts = [f"  {token.provider.display_name}: {token.text!r}"]
    if token.start is not None and token.end is not None:
        parts.append(f"{token.start:.2f}-{token.end:.2f}s")
    if token.speaker:
        parts.append(f"speaker {token.speaker}")
    if token.log_probability is not None:
        parts.append(f"log probability {token.log_probability:.3f}")
    if token.confidence is not None:
        parts.append(f"confidence {token.confidence:.3f}")
    return " ".join(parts)


# -- Validating and applying ---------------------------------------------


def apply_decisions(
    disputes: Sequence[Dispute],
    decisions: Sequence[dict[str, Any]],
    *,
    vocabulary_terms: Sequence[str] = (),
) -> AdjudicationOutcome:
    """Check each decision against the evidence, and apply only what survives.

    This is the part that actually protects the transcript, and it is
    separate from the request on purpose: it can be reasoned about, and
    tested, without a client anywhere near it. A decision that fails any
    check is refused whole rather than trimmed into something acceptable,
    because a model that has strayed once has no claim on being trusted
    about what it meant.
    """
    by_span = {dispute.span_id: dispute for dispute in disputes}
    applied: list[AppliedEdit] = []
    refused: list[RefusedEdit] = []
    declined: list[str] = []
    answered: set[str] = set()

    for decision in decisions:
        span_id = _text_of(decision.get("span_id"))
        action = _action_of(decision.get("action"))
        to_text = _text_of(decision.get("to_text"))
        from_text = _text_of(decision.get("from_text"))
        reason = _text_of(decision.get("reason"))

        dispute = by_span.get(span_id)
        if dispute is None or action is None:
            # Either it named a span nobody asked about, which may well be a
            # real word elsewhere in the transcript, or it invented an
            # action. Both are out of bounds and neither is repairable.
            refused.append(
                RefusedEdit(
                    span_id=span_id,
                    action=action,
                    to_text=to_text,
                    refusal=(
                        RefusalReason.MALFORMED_ANSWER
                        if action is None
                        else RefusalReason.SPAN_NOT_IN_DISPUTE
                    ),
                    detail=(
                        "The answer named a span that was not in dispute."
                        if action is not None
                        else "The answer named an action that does not exist."
                    ),
                )
            )
            continue

        if span_id in answered:
            # A second opinion on a span already settled is not a decision,
            # it is a rewrite of one, and taking the later of two answers
            # would make the result depend on their order.
            refused.append(
                RefusedEdit(
                    span_id=span_id,
                    action=action,
                    to_text=to_text,
                    refusal=RefusalReason.SPAN_NOT_IN_DISPUTE,
                    detail="The answer decided the same span twice.",
                )
            )
            continue
        answered.add(span_id)

        if action is EditAction.DECLINE:
            _decline(dispute, reason or "The adjudicating model would not decide.")
            declined.append(span_id)
            continue

        refusal = _refusal_for(dispute, action, from_text, to_text, vocabulary_terms)
        if refusal is not None:
            _refuse(dispute, refusal, reason)
            refused.append(
                RefusedEdit(
                    span_id=span_id,
                    action=action,
                    to_text=to_text,
                    refusal=refusal,
                    detail=refusal.display_name,
                )
            )
            continue

        if not _write_text(dispute, action, to_text, reason):
            # Only reachable if applying the edit would have moved something
            # that is not text. The words are put back and the span is
            # flagged, which is the same outcome as any other refusal.
            _refuse(dispute, RefusalReason.IDENTITY_WOULD_CHANGE, reason)
            refused.append(
                RefusedEdit(
                    span_id=span_id,
                    action=action,
                    to_text=to_text,
                    refusal=RefusalReason.IDENTITY_WOULD_CHANGE,
                    detail=RefusalReason.IDENTITY_WOULD_CHANGE.display_name,
                )
            )
            continue

        applied.append(
            AppliedEdit(
                span_id=span_id,
                action=action,
                from_text=dispute.current_text if action is EditAction.KEEP else from_text,
                to_text=to_text,
                reason=reason,
            )
        )

    unanswered = tuple(
        dispute.span_id for dispute in disputes if dispute.span_id not in answered
    )
    for span_id in unanswered:
        # Silence about a span is a declination. It is recorded as one so
        # that the word reaches a person rather than sitting in the
        # transcript looking settled.
        # Silence is not a judgement. Nothing was learned about the word, so
        # it keeps the confidence the evidence gave it and is flagged.
        _decline(
            by_span[span_id],
            "The adjudicating model did not answer about this span.",
            lower_confidence=False,
        )

    return AdjudicationOutcome(
        applied=tuple(applied),
        refused=tuple(refused),
        declined=tuple(declined),
        unanswered=unanswered,
    )


def _refusal_for(
    dispute: Dispute,
    action: EditAction,
    from_text: str,
    to_text: str,
    vocabulary_terms: Sequence[str],
) -> RefusalReason | None:
    """Say why this decision cannot be applied, or None if it can."""
    if not _quotes_the_span(dispute, from_text):
        # A model that misquotes what it is editing has read something other
        # than the span it names, and its conclusion is about that other
        # thing rather than about this one.
        return RefusalReason.WRONG_ORIGINAL_TEXT

    if action is EditAction.KEEP:
        # Keeping is only allowed to keep. A "keep" that quietly carries
        # different text is a replacement wearing the safer of the two words.
        if to_text and not are_equivalent(to_text, dispute.current_text):
            return RefusalReason.TEXT_NOT_OFFERED
        return None

    if not to_text.strip():
        return RefusalReason.TEXT_NOT_OFFERED

    if dispute.is_high_risk:
        # Not because this particular answer looks wrong, but because the
        # model has only plausibility to offer and plausibility is exactly
        # what must not settle an amount, a date or an account number. The
        # rule lives here rather than only in the prompt, because a prompt is
        # a request and this has to be a guarantee.
        return RefusalReason.HIGH_RISK_ENTITY

    if len(to_text.split()) != len(dispute.tokens):
        # A span of two words cannot become one word without somebody
        # deciding where the join between the two original spans now falls,
        # and that decision belongs to forced alignment against the audio,
        # not to a model that has never heard it.
        return RefusalReason.WORD_COUNT_CHANGED

    if not _was_offered(to_text, dispute, vocabulary_terms):
        return RefusalReason.TEXT_NOT_OFFERED
    return None


def _quotes_the_span(dispute: Dispute, from_text: str) -> bool:
    """Whether the decision describes the span as it actually reads.

    Compared through the normalisation rules rather than character for
    character, so that a difference in punctuation or capitals is not
    treated as a misquote. A genuinely different sentence still is.
    """
    if not from_text.strip():
        return True
    return are_equivalent(from_text, dispute.current_text)


def _was_offered(to_text: str, dispute: Dispute, vocabulary_terms: Sequence[str]) -> bool:
    """Whether this reading came from somewhere rather than from nowhere.

    A reading counts as offered when it matches a candidate for the span or
    a known vocabulary term, either as a whole phrase or word by word. The
    word-by-word test is what lets the specification's own example work:
    "Jurgen Miller" becomes "Jürgen Müller" because both of those words are
    in the vocabulary, even though that exact pair was never offered
    together.

    The comparison is the normalisation module's, so a candidate re-spelled
    with its proper umlaut is still the same offered reading. A word that
    nobody heard and that nobody wrote down is not.
    """
    offered_phrases = list(dispute.candidate_texts)
    offered_phrases.append(dispute.current_text)
    offered_phrases.extend(token.text for token in dispute.tokens if token.text)
    offered_phrases.extend(term for term in vocabulary_terms if term and term.strip())

    if any(are_equivalent(to_text, phrase) for phrase in offered_phrases):
        return True

    offered_words = {
        word for phrase in offered_phrases for word in phrase.split() if word
    }
    return all(
        any(are_equivalent(word, offered) for offered in offered_words)
        for word in to_text.split()
    )


def _write_text(dispute: Dispute, action: EditAction, to_text: str, reason: str) -> bool:
    """Apply a validated decision, and prove nothing but the text moved.

    The identity of every word - its identifier, its times, its speaker and
    where those came from - is photographed before the change and compared
    afterwards. That guards against the model, which has no field to put a
    timestamp in and so cannot ask for one directly, and equally against a
    future change to this function that starts writing more than it should.
    A mismatch puts everything back and reports the failure.
    """
    before = [_identity_of(token) for token in dispute.tokens]
    previous = [token.text for token in dispute.tokens]
    previous_sources = [token.text_source for token in dispute.tokens]
    previous_text = " ".join(word for word in previous if word)
    words = to_text.split() if action is EditAction.REPLACE else previous
    # Written where a person reviewing the word will read it, so it says what
    # was chosen as well as why. "replace" on its own would send them looking
    # for the reading it replaced.
    note = (
        f"Chose {to_text!r} over {previous_text!r}."
        if action is EditAction.REPLACE
        else f"Kept {previous_text!r}."
    )

    offered_by = (
        _service_that_offered_phrase(to_text, dispute)
        if action is EditAction.REPLACE
        else None
    )
    for token, word in zip(dispute.tokens, words):
        if action is EditAction.REPLACE and word != token.text:
            token.text = word
            token.normalised_text = normalise(word)
            # The transcript says which service each answer came from, and
            # the service that read the word before has just lost. A reading
            # only the vocabulary offered came from no service at all.
            token.text_source = offered_by or _service_that_offered(word, token, dispute)
        # Adjudicated, not proven. The model weighed evidence a rule could
        # not, which is worth more than an unresolved word and less than a
        # person's own reading. So the word improves as far as "review
        # suggested" and no further, and stays visible in the review queue
        # rather than being marked settled on a model's say-so.
        #
        # A high-risk word - an amount, a date, an account number - is never
        # raised by the model at all, not even when it only confirms the
        # reading. Raising an unresolved amount to "review suggested" would
        # drop its [UNCERTAIN: a / b] marker from the export and show one
        # plain number, which settles it in effect. So its confidence stays
        # what the evidence gave it, and it is flagged for a person.
        if dispute.is_high_risk:
            token.flag(ReviewReason.HIGH_RISK_ENTITY)
        else:
            token.text_confidence = Confidence.REVIEW_SUGGESTED
        token.llm_decision = f"{note} {reason}".strip()

    after = [_identity_of(token) for token in dispute.tokens]
    if after != before:
        for token, original, source in zip(dispute.tokens, previous, previous_sources):
            token.text = original
            token.normalised_text = normalise(original)
            token.text_source = source
        return False
    return True


def _service_that_offered_phrase(to_text: str, dispute: Dispute) -> Provider | None:
    """The first service that offered the whole chosen reading, or None.

    Reconciliation often keeps a reading of several words as one candidate,
    such as "I scream", on every word it covers. No single word of it equals
    that candidate, so the whole reading is looked for first.
    """
    for token in dispute.tokens:
        for candidate in token.candidates:
            if candidate.providers and are_equivalent(candidate.text, to_text):
                return candidate.providers[0]
    heard = " ".join(word.text for word in dispute.escalation_tokens if word.text)
    if dispute.escalation_tokens and are_equivalent(heard, to_text):
        return dispute.escalation_tokens[0].provider
    return None


def _service_that_offered(word: str, token: FinalToken, dispute: Dispute) -> Provider | None:
    """The first service that heard this word here, or None if none did."""
    for candidate in token.candidates:
        if candidate.providers and are_equivalent(candidate.text, word):
            return candidate.providers[0]
    for heard in dispute.escalation_tokens:
        if are_equivalent(heard.text, word):
            return heard.provider
    return None


def _identity_of(token: FinalToken) -> tuple[Any, ...]:
    """Everything about a word that adjudication must never touch."""
    return (
        token.id,
        token.start,
        token.end,
        token.speaker,
        token.speaker_source,
        token.timing_source,
        token.timing_status,
        token.timing_confidence,
    )


def _decline(dispute: Dispute, reason: str, lower_confidence: bool = True) -> None:
    """Leave the span exactly as it was, and make sure a person sees it.

    A decline the model actually made lowers the word to unresolved: the
    best-placed judge looked at the readings and would not choose, which is
    the definition of an open question. ``lower_confidence`` is False when
    the model was never reached, since then nothing has been learned about
    the word and its confidence should stay what the evidence gave it.
    """
    for token in dispute.tokens:
        if lower_confidence:
            token.text_confidence = Confidence.UNRESOLVED
        token.flag(ReviewReason.ADJUDICATION_DECLINED)
        if dispute.is_high_risk:
            token.flag(ReviewReason.HIGH_RISK_ENTITY)
        token.llm_decision = reason


def _refuse(dispute: Dispute, refusal: RefusalReason, model_reason: str) -> None:
    """Throw an edit away, keeping the model's opinion where a person can read it.

    The opinion is kept deliberately. It is worth something to somebody
    deciding the span by hand, and it is worth nothing at all in the
    transcript itself, so it goes into the review note rather than into the
    words.
    """
    for token in dispute.tokens:
        token.text_confidence = Confidence.UNRESOLVED
        if refusal is RefusalReason.HIGH_RISK_ENTITY:
            token.flag(ReviewReason.HIGH_RISK_ENTITY)
        else:
            token.flag(ReviewReason.ESCALATION_UNRESOLVED)
        note = refusal.display_name
        token.llm_decision = f"{note}: {model_reason}" if model_reason else note


# -- Reading the answer --------------------------------------------------


def _read_decisions(response: Any) -> tuple[list[dict[str, Any]], str | None]:
    """Pull the decisions out of the answer, or say what was wrong with it.

    The schema is strict, so a well-behaved API cannot return the wrong
    shape. This still checks, because "cannot happen" is a poor reason to
    apply an unexamined answer to somebody's transcript.
    """
    text = _answer_text(response)
    if not text:
        return [], "The adjudication model returned an empty answer."
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError) as error:
        return [], f"The adjudication model's answer was not valid JSON: {error}"
    if not isinstance(parsed, dict):
        return [], "The adjudication model's answer was not an object."
    decisions = parsed.get("decisions")
    if not isinstance(decisions, list):
        return [], "The adjudication model's answer had no list of decisions."
    if not all(isinstance(decision, dict) for decision in decisions):
        return [], "The adjudication model returned a decision that was not an object."
    return list(decisions), None


def _answer_text(response: Any) -> str:
    """The JSON the model wrote, whichever shape the library hands it over in."""
    text = _text_field(response, "output_text")
    if text:
        return text
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        for piece in getattr(item, "content", None) or []:
            value = _text_field(piece, "text")
            if value:
                parts.append(value)
    return "".join(parts)


def _text_field(source: Any, name: str) -> str:
    if source is None:
        return ""
    value = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
    return "" if value is None else str(value)


def _text_of(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _action_of(value: Any) -> EditAction | None:
    try:
        return EditAction(str(value).strip().lower())
    except ValueError:
        return None


# -- Helpers -------------------------------------------------------------


def without_credentials(parameters: dict[str, Any]) -> dict[str, Any]:
    """Strip anything that looks like a secret out of a recorded parameter set.

    A request record is written into the recording's folder, which is copied
    around and read by people. An API key that reached it would be a leak
    nobody would ever notice.
    """
    cleaned: dict[str, Any] = {}
    for key, value in parameters.items():
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(key)).lower()
        words = [word for word in re.split(r"[^a-z0-9]+", name) if word]
        if any(word.endswith(_CREDENTIAL_NAMES) for word in words):
            continue
        cleaned[key] = without_credentials(value) if isinstance(value, dict) else value
    return cleaned


def _stopped_early_errors() -> tuple[type[BaseException], ...]:
    """The two errors that mean the answer was cut short rather than refused.

    Imported here rather than at the top of the module, for the same reason
    the client is. An empty tuple is a legal thing to catch and matches
    nothing, which is the right behaviour when the package is missing: there
    is then no client either, and the call never happened.
    """
    try:
        from openai import ContentFilterFinishReasonError, LengthFinishReasonError
    except ImportError:
        return ()
    return (LengthFinishReasonError, ContentFilterFinishReasonError)


def _failure_sentence(error: Exception) -> str:
    """One sentence about why the model did not answer, for the outcome.

    The library's exception classes are not imported to tell them apart.
    Reading the status code off whatever was raised covers every class it
    has, and doing it that way means this module never imports the package
    merely to catch its errors, which would undo the point of importing it
    lazily.
    """
    status = getattr(error, "status_code", None)
    detail = str(error).strip() or type(error).__name__
    if status is not None:
        return f"The adjudication model answered with an error ({status}): {detail}"
    return f"The adjudication model could not be reached: {detail}"


def summarise(outcome: AdjudicationOutcome) -> str:
    """One line about a pass, for the warnings a person actually reads."""
    if outcome.error is not None:
        return f"Adjudication did not run: {outcome.error}"
    parts = [f"{len(outcome.applied)} settled"]
    if outcome.refused:
        parts.append(f"{len(outcome.refused)} refused")
    if outcome.declined or outcome.unanswered:
        parts.append(f"{len(outcome.declined) + len(outcome.unanswered)} left for review")
    return "Adjudication: " + ", ".join(parts) + "."
