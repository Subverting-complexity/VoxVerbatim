"""The two documents a person actually reads when the transcription is done.

Everything else in this package is built for the application to reason
about. These two are built for a human being, and they answer two different
questions.

The first is "what was said". That is the plain text export: the words,
grouped into turns and labelled with who was speaking, wrapped so it can be
read in any editor and pasted anywhere. Where the services could not agree
on a word, it says so in the text itself rather than choosing one and hiding
the doubt, because a wrong number that reads perfectly well is far more
dangerous than a visible gap.

The second is "can I trust this". That is the Markdown review report, and it
is the more important of the two. It says which services answered and which
did not, how long the run took, what it cost, how much of the transcript is
solid, and then every single place that still needs a person, with the time
it happens, why it was flagged, what each service heard there and what the
application settled on. Somebody reading it should be able to decide whether
to sign the transcript off or sit down with the audio, and they should be
able to decide that without knowing anything about how the application
works. That is why the confidence categories are explained in the report
itself every time: nobody remembers what four category names mean weeks
after they last saw them.

Subtitle formats are deliberately absent. SRT and WebVTT were considered and
ruled out; these two exports are what was asked for, and adding the others
"while we are here" would mean four formats to keep correct instead of two.

Times are always written the way a person reads a clock, as ``12:04`` or
``1:03:22``, using the same formatting the rest of the application uses for
durations. A raw ``742.6`` is not something anybody can find in a recording.
"""

from __future__ import annotations

import logging
import textwrap
from datetime import datetime
from typing import Any, Iterable, Sequence

from vox_verbatim.formatting import format_duration, spoken_duration
from vox_verbatim.transcription.cost import CostEstimate, describe_money
from vox_verbatim.transcription.diarisation import speaker_doubt_stretches
from vox_verbatim.transcription.model import (
    Confidence,
    FinalToken,
    Provider,
    ProviderRequestRecord,
    ReviewReason,
    Transcript,
    _attaches_to_previous,
    review_reason_text,
)

_log = logging.getLogger(__name__)

#: What the two exports are called inside the transcript folder.
TEXT_EXPORT_NAME = "transcript.txt"
REPORT_EXPORT_NAME = "review-report.md"

#: The smooth transcript's file name, its request purpose and how its
#: failure message begins. They are defined here rather than in
#: :mod:`.smoothing`, because that module imports this one and the report
#: below needs them.
SMOOTH_EXPORT_NAME = "transcript-smooth.txt"
SMOOTHING_PURPOSE = "smoothing"
SMOOTHING_NOT_MADE = "The smooth transcript was not made"

#: Where the text export wraps. Wide enough not to look like a poem, narrow
#: enough to read comfortably in a plain editor and in a screen magnifier.
DEFAULT_WRAP_WIDTH = 78

#: Where an estimated cost is looked for when the caller does not pass one.
#: The pipeline records it with the rest of the run's settings, so a report
#: rebuilt from a stored transcript months later still shows what the run
#: cost.
COST_KEYS = ("estimated_cost", "estimated_cost_usd")

_UNKNOWN_SPEAKER = "Unknown speaker"

#: The services that place each word in the recording. The others return
#: text only, so their failing never explains a transcript with no times.
_TIMING_SERVICES: tuple[Provider, ...] = (
    Provider.ELEVENLABS,
    Provider.ASSEMBLYAI,
    Provider.DEEPGRAM,
)

#: The share of the words that must have nothing to play before the report
#: explains it once instead of listing each word. Below this the words with
#: no time are exceptions a person should look at one by one; above it they
#: are one fault, and a list of hundreds of identical rows hides the words
#: that need a decision for any other reason.
_UNTIMED_SHARE_FOR_ONE_STATEMENT = 0.5

#: What each confidence category means, in the words the report uses. They
#: are here rather than on the enumeration because they are an explanation
#: for a reader, not a label for a control.
_CONFIDENCE_MEANINGS: dict[Confidence, str] = {
    Confidence.HIGH: (
        "The evidence agreed. These words can be read as they stand."
    ),
    Confidence.REVIEW_SUGGESTED: (
        "Something was slightly off, such as one service hearing it differently "
        "or a timing that had to be inferred. Usually right, worth a glance."
    ),
    Confidence.REVIEW_REQUIRED: (
        "The evidence genuinely conflicts, or the word is the kind that must "
        "not be guessed at, such as an amount or a date. Check these."
    ),
    Confidence.UNRESOLVED: (
        "Nothing was chosen at all. The transcript shows the alternatives in "
        "square brackets, and only you can settle them."
    ),
}


# -- Small shared helpers -------------------------------------------------


def format_timestamp(seconds: float | None) -> str:
    """Return a clock time such as ``12:04``, or ``--:--`` when there is none.

    A word with no timing is a real state in this application, not a bug, so
    it needs a way of being written down that does not pretend to be a time.
    """
    if seconds is None:
        return "--:--"
    return format_duration(seconds)


def _token_time(token: FinalToken) -> float | None:
    """The best time to quote for a word.

    A word's own start is used where it has one. Where it does not, the
    wider span it is believed to sit in is the next best thing, and it is
    exactly the unaligned words that a person most needs to find.
    """
    if token.start is not None:
        return token.start
    span = token.source_audio_span
    return span.start if span is not None else None


def _speaker_name(transcript: Transcript, speaker_id: str | None) -> str:
    if speaker_id is None:
        return _UNKNOWN_SPEAKER
    speaker = transcript.speaker_for(speaker_id)
    if speaker is not None:
        return speaker.display_name
    # A label the transcript never registered still names a real person in
    # the recording, so it is shown rather than thrown away.
    return f"Speaker {speaker_id}"


def render_token_text(token: FinalToken) -> str:
    """What one word looks like in an export.

    A word the evidence never settled is written out as the choice it still
    is, in the form the specification asks for, rather than as whichever
    candidate happened to score highest.
    """
    alternatives = token.is_uncertain_between()
    if alternatives:
        return "[UNCERTAIN: " + " / ".join(alternatives) + "]"
    return token.text


def _join_tokens(tokens: Iterable[FinalToken]) -> str:
    """Join words into running text, reattaching separated punctuation.

    Some services return a full stop as a word of its own with its own
    timing. Joining everything with spaces would give " ." at the end of
    every sentence, so punctuation goes back onto the word in front of it.
    The rule itself lives with the model, so that the transcript's own
    verbatim text and this export can never drift apart.
    """
    parts: list[str] = []
    for token in tokens:
        text = render_token_text(token)
        if not text:
            continue
        if parts and _attaches_to_previous(text):
            parts[-1] = parts[-1] + text
        else:
            parts.append(text)
    return " ".join(parts)


def _turns(transcript: Transcript) -> list[tuple[str | None, list[FinalToken]]]:
    """Group consecutive words into runs by the person speaking them.

    Punctuation and the odd word with no speaker are kept with the turn they
    fall inside rather than starting one of their own, which is what stops a
    stray full stop from being labelled as somebody else talking.
    """
    grouped: list[tuple[str | None, list[FinalToken]]] = []
    for token in transcript.tokens:
        speaker = token.speaker
        if grouped and (speaker == grouped[-1][0] or speaker is None):
            grouped[-1][1].append(token)
            continue
        grouped.append((speaker, [token]))
    return grouped


def _services_that_answered(transcript: Transcript) -> list[Provider]:
    return [
        provider
        for provider in Provider
        if provider in transcript.provider_results
        and transcript.provider_results[provider].succeeded
    ]


def _count(count: int, singular: str, plural: str | None = None) -> str:
    """Return a count and its noun, agreeing with each other.

    Written out because a report that says "1 words" reads as something a
    machine produced and nobody checked, which is the opposite of the
    impression this document needs to give.
    """
    if count == 1:
        return f"1 {singular}"
    return f"{count:,} {plural or singular + 's'}"


def _service_list(providers: Sequence[Provider]) -> str:
    names = [provider.display_name for provider in providers]
    if not names:
        return "none"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


# -- The plain text export ------------------------------------------------


def render_plain_text(transcript: Transcript, width: int = DEFAULT_WRAP_WIDTH) -> str:
    """Return the transcript as readable text with the speakers named.

    The header is deliberately short. It says which recording this is, when
    it was transcribed and which services contributed, which is what somebody
    needs to know when the file turns up on its own six months later, and
    nothing more. The detail belongs in the review report.
    """
    width = max(20, int(width))
    lines: list[str] = [f"Recording: {transcript.recording_name}"]
    if transcript.canonical_audio is not None:
        lines.append(f"Length: {format_duration(transcript.canonical_audio.duration)}")
    when = transcript.completed_at or transcript.created_at
    if when:
        lines.append(f"Transcribed: {_readable_time(when)}")
    lines.append(f"Services: {_service_list(_services_that_answered(transcript))}")
    lines.append("")

    body = _turns(transcript)
    if not body:
        lines.append("This recording produced no words.")
        return "\n".join(lines) + "\n"

    for speaker_id, tokens in body:
        text = _join_tokens(tokens)
        if not text:
            continue
        started = _token_time(tokens[0])
        lines.append(f"[{format_timestamp(started)}] {_speaker_name(transcript, speaker_id)}:")
        lines.extend(textwrap.wrap(text, width=width) or [""])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _readable_time(stamp: str) -> str:
    """Turn a stored timestamp into something a person reads without effort.

    The stored form is ISO 8601 because that is what sorts and parses. It is
    not what anybody wants to read, so it becomes ``17 August 2026 at 10:04``
    where it can be understood, and is left exactly as it was where it
    cannot, since an odd-looking timestamp is better than none.
    """
    parsed = _parse_time(stamp)
    if parsed is None:
        return stamp
    return parsed.strftime("%d %B %Y at %H:%M")


def _parse_time(stamp: str) -> datetime | None:
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp)
    except ValueError:
        return None


# -- The review report ----------------------------------------------------


def render_review_report(
    transcript: Transcript,
    estimated_cost: float | CostEstimate | None = None,
) -> str:
    """Return the Markdown report a person reads to decide whether to trust this.

    ``estimated_cost`` is what the run is believed to have cost, either as a
    plain amount or as the estimate the costing module works out, which also
    knows the currency and which services it could not price. The pipeline
    passes it in where it has one; otherwise it is looked for among the run's
    recorded settings, so that rebuilding this report from a stored
    transcript months later does not silently drop the figure.
    """
    cost = estimated_cost if estimated_cost is not None else _recorded_cost(transcript)
    parts = [
        _report_opening(transcript),
        _report_untimed(transcript),
        _report_services(transcript, cost),
        _report_confidence(transcript),
        _report_review_queue(transcript),
        _report_speaker_doubts(transcript),
        _report_escalation(transcript),
        _report_smoothing(transcript),
        _report_warnings(transcript),
    ]
    return "\n".join(part for part in parts if part).rstrip() + "\n"


def write_exports(
    transcript: Transcript,
    store: Any,
    patient: bool = True,
    names: Sequence[str] = (),
) -> list[str]:
    """Write the two readable documents for a transcript, and name any that failed.

    ``store`` is the recording's
    :class:`~vox_verbatim.transcription.store.TranscriptStore`. Both the
    pipeline, at the end of a run, and the review window, after each saved
    correction, write through this, so the files always match the transcript
    they were made from. Each file goes to a temporary name and is then moved
    over the old one, so a reader never sees a half-written file.

    A file that cannot be written, for example because another program holds
    it open, is named in the answer rather than raised. The transcript is
    already saved by the time this runs, and an export is a convenience
    rebuilt from it, so a failure here must never undo a correction.

    ``patient`` is passed on to the store: False writes once, without waiting
    for a program that holds a file open to let go of it.

    ``names`` limits the writing to those documents; empty means both. The
    pipeline writes transcript.txt before the smooth transcript is made and
    the report after it, so the report can say what smoothing did.
    """
    failed: list[str] = []
    for name, render in (
        (TEXT_EXPORT_NAME, render_plain_text),
        (REPORT_EXPORT_NAME, render_review_report),
    ):
        if names and name not in names:
            continue
        try:
            written = store.write_export(name, render(transcript), patient=patient)
        except Exception:  # an export is a convenience, never the transcript
            _log.exception("Could not write the %s export.", name)
            written = None
        if written is None:
            failed.append(name)
    return failed


def _recorded_cost(transcript: Transcript) -> float | None:
    for key in COST_KEYS:
        value = transcript.effective_settings.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        return float(value)
    return None


def _report_opening(transcript: Transcript) -> str:
    lines = [f"# Review report: {transcript.recording_name}", ""]
    when = transcript.completed_at or transcript.created_at
    sentence = (
        "This report explains how this transcript was produced and what, if "
        "anything, still needs you."
    )
    if when:
        sentence += f" The transcription finished on {_readable_time(when)}."
    lines.append(sentence)
    lines.append("")
    if transcript.canonical_audio is not None:
        audio = transcript.canonical_audio
        lines.append(
            f"The recording is {spoken_duration(audio.duration)} long. Every time in "
            "this report is measured from the start of it."
        )
        lines.append("")
    return "\n".join(lines)


def _untimed_words(transcript: Transcript) -> tuple[int, int]:
    """How many spoken words have nothing to play, out of how many words."""
    words = [
        token for token in transcript.tokens
        if token.text.strip() and not _attaches_to_previous(token.text)
    ]
    return sum(1 for token in words if token.audible_span is None), len(words)


def _most_words_are_untimed(transcript: Transcript) -> bool:
    untimed, total = _untimed_words(transcript)
    return total > 0 and untimed / total > _UNTIMED_SHARE_FOR_ONE_STATEMENT


def _report_untimed(transcript: Transcript) -> str:
    """Say once, before anything else, that the words could not be placed.

    When no service timed the words, every one of them is flagged for the
    same reason. Listing each would bury the real disagreements under
    hundreds of identical rows, and leave the cause to a line at the end of
    the report, so the cause and the remedy are given here instead and the
    list further down leaves those rows out.
    """
    if not _most_words_are_untimed(transcript):
        return ""
    untimed, total = _untimed_words(transcript)
    lines = ["## No word could be placed in the recording", ""]
    share = "None" if untimed == total else f"Only {total - untimed:,}"
    lines.append(
        f"{share} of the {_count(total, 'word')} in this transcript can be played "
        "back, because no service said where in the recording the words were "
        "spoken. Without that, the review window cannot play a word, and the "
        "language model and the second opinions have no audio to check a "
        "disagreement against."
    )
    lines.append("")

    failed = [
        transcript.provider_results[provider]
        for provider in _TIMING_SERVICES
        if provider in transcript.provider_results
        and not transcript.provider_results[provider].succeeded
    ]
    realign_failed = sum(
        1 for token in transcript.tokens
        if ReviewReason.FORCED_ALIGNMENT_FAILED in token.review_reasons
    )
    if failed or realign_failed:
        lines.append("The sources of word times that failed:")
        lines.append("")
        for result in failed:
            reason = result.error or "no reason was reported"
            lines.append(f"- **{result.provider.display_name}**: {reason}")
        if realign_failed:
            lines.append(
                f"- **Forced alignment**: it could not measure "
                f"{_count(realign_failed, 'word')}."
            )
        lines.append("")
    else:
        lines.append(
            "None of the services that time their words "
            f"({_service_list(_TIMING_SERVICES)}) gave word times for this recording."
        )
        lines.append("")
    lines.append(
        "Fix the cause, such as an API key in Settings, and then transcribe the "
        "recording again. Forced alignment only measures words that a service "
        "has already placed roughly, so it cannot place these words on its own. "
        "The list of words that need a decision leaves out the words whose only "
        "fault is that they have no time."
    )
    lines.append("")
    return "\n".join(lines)


def _report_services(transcript: Transcript, cost: float | None) -> str:
    lines = ["## Which services answered", ""]
    results = [
        transcript.provider_results[provider]
        for provider in Provider
        if provider in transcript.provider_results
    ]
    if not results:
        lines.append("No speech-to-text service was called for this recording.")
        lines.append("")
        return "\n".join(lines)

    answered = [result for result in results if result.succeeded]
    failed = [result for result in results if not result.succeeded]
    if not failed:
        lines.append(
            f"Every service asked came back with words: "
            f"{_service_list([result.provider for result in answered])}."
        )
    else:
        lines.append(
            f"{len(answered)} of the {_count(len(results), 'service')} asked came back "
            "with words. A service that does not answer costs a little accuracy and "
            "nothing else: the transcript is built from whatever did answer."
        )
    lines.append("")
    lines.append("| Service | Answered | Words | Model | Time taken | Note |")
    lines.append("| --- | --- | ---: | --- | ---: | --- |")
    for result in results:
        request = result.request
        lines.append(
            "| {service} | {answered} | {words} | {model} | {took} | {note} |".format(
                service=_cell(result.provider.display_name),
                answered="Yes" if result.succeeded else "No",
                words=f"{len(result.word_tokens):,}" if result.succeeded else "-",
                model=_cell(_model_description(request)),
                took=format_duration(request.processing_seconds) if request else "-",
                note=_cell(result.error or ""),
            )
        )
    lines.append("")

    if failed:
        lines.append("The services that did not answer, and why:")
        lines.append("")
        for result in failed:
            reason = result.error or "no reason was reported"
            lines.append(f"- **{result.provider.display_name}**: {reason}")
        lines.append("")

    lines.append(_timing_sentence(transcript))
    if cost is not None:
        lines.append("")
        lines.append(_cost_sentence(cost))
    lines.append("")
    return "\n".join(lines)


def _cost_sentence(cost: float | CostEstimate) -> str:
    """What the run cost, and honestly what the figure leaves out.

    A total that looks complete and is not would be worse than no total at
    all, so a service whose rate nobody has set is named rather than quietly
    counted as free.
    """
    if isinstance(cost, CostEstimate):
        amount = describe_money(cost.total, cost.currency)
        unpriced = cost.unpriced_providers
    else:
        amount = describe_money(float(cost))
        unpriced = ()
    sentence = (
        f"The estimated cost of this run is {amount}. It is worked out from the "
        "amount of audio each service was sent, so it is an estimate rather than "
        "a bill."
    )
    if unpriced:
        has = "has" if len(unpriced) == 1 else "have"
        sentence += (
            f" It leaves out {_service_list(unpriced)}, which {has} no rate set in "
            "Settings, so the real figure is higher than this one."
        )
    return sentence


def _model_description(request: ProviderRequestRecord | None) -> str:
    """The exact model, which is the thing that explains a changed result later."""
    if request is None or not request.model_identifier:
        return "not recorded"
    if request.model_version:
        return f"{request.model_identifier} ({request.model_version})"
    return request.model_identifier


def _timing_sentence(transcript: Transcript) -> str:
    total = sum(request.processing_seconds for request in transcript.requests)
    parts: list[str] = []
    if transcript.requests:
        made = "was made" if len(transcript.requests) == 1 else "were made"
        parts.append(
            f"{_count(len(transcript.requests), 'request')} {made} in all, "
            f"taking {spoken_duration(total)} of service time in total."
        )
    started = _parse_time(transcript.created_at)
    finished = _parse_time(transcript.completed_at)
    if started is not None and finished is not None and finished >= started:
        elapsed = (finished - started).total_seconds()
        sentence = f"From start to finish the run took {spoken_duration(elapsed)}"
        if parts and elapsed < total:
            # The two figures disagree, and a reader who noticed that without
            # being told would reasonably wonder which one to believe. When
            # local work takes longer than the services, the run is the larger
            # figure and the reason below would be false.
            sentence += (
                ", which is less than the total above because the services were "
                "called at the same time as each other"
            )
        parts.append(sentence + ".")
    return " ".join(parts)


def _report_confidence(transcript: Transcript) -> str:
    counts = transcript.counts_by_confidence()
    total = sum(counts.values())
    lines = ["## How much of this can be trusted", ""]
    if not total:
        lines.append("There are no words in this transcript, so there is nothing to rate.")
        lines.append("")
        return "\n".join(lines)

    lines.append(
        "Every word is rated in one of four categories. The services report "
        "confidence on scales that mean different things and cannot be compared "
        "with each other, so the application does not turn them into a single "
        "percentage. It sorts the words into these four instead."
    )
    lines.append("")
    lines.append("| Rating | Words | Share | What it means |")
    lines.append("| --- | ---: | ---: | --- |")
    for category in Confidence:
        count = counts.get(category, 0)
        share = (count / total) * 100.0
        lines.append(
            "| {name} | {count:,} | {share:.1f}% | {meaning} |".format(
                name=_cell(category.display_name),
                count=count,
                share=share,
                meaning=_cell(_CONFIDENCE_MEANINGS[category]),
            )
        )
    lines.append("")
    settled = counts.get(Confidence.HIGH, 0)
    if settled == total:
        lines.append(f"Every one of the {_count(total, 'word')} came out at high confidence.")
    else:
        lines.append(
            f"That is {settled:,} of {_count(total, 'word')} rated high confidence, "
            f"leaving {_count(total - settled, 'word')} worth a look."
        )
    lines.append("")
    return "\n".join(lines)


def _report_review_queue(transcript: Transcript) -> str:
    lines = ["## What still needs you", ""]
    # A word held back only for its speaker is not listed here. Its text is
    # not in doubt, and the section after this one lists the speaker doubts
    # once for each stretch of speech rather than once for every word.
    flagged = transcript.word_review_tokens
    pending = flagged
    if _most_words_are_untimed(transcript):
        # The statement at the top already explains these words, once.
        pending = [
            token for token in pending
            if token.review_reasons != [ReviewReason.UNALIGNED_WORD]
        ]
    if not pending and flagged:
        lines.append(
            "Nothing else. Every word flagged for review was flagged only because "
            "it has no time, which the start of this report explains."
        )
        lines.append("")
        return "\n".join(lines)
    if not pending and transcript.review_tokens:
        lines.append(
            "Nothing about the words. The services agreed on the text everywhere, "
            "including every value that must not be guessed at. Only who said "
            "some of it is in doubt, which the next section lists."
        )
        lines.append("")
        return "\n".join(lines)
    if not pending:
        lines.append(
            "Nothing. No word in this transcript was flagged for review, which "
            "means the services agreed everywhere, including on every value "
            "that must not be guessed at. You can read the transcript as it "
            "stands."
        )
        lines.append("")
        return "\n".join(lines)

    needs = "One place needs" if len(pending) == 1 else f"{len(pending):,} places need"
    lines.append(
        f"{needs} a decision. Each one gives the time it happens in the recording, "
        "what the transcript says now, why it was flagged, and what each service "
        "actually heard there, so you can play that moment and judge it for "
        "yourself."
    )
    lines.append("")
    lines.append("| Time | In the transcript | Why | What the services heard |")
    lines.append("| --- | --- | --- | --- |")
    for token in pending:
        lines.append(
            "| {time} | {chosen} | {why} | {heard} |".format(
                time=format_timestamp(_token_time(token)),
                chosen=_cell(render_token_text(token) or "(nothing)"),
                why=_cell(_reasons_in_words(token)),
                heard=_cell(_what_the_services_heard(transcript, token)),
            )
        )
    lines.append("")
    risky = [token for token in pending if token.risk_categories]
    if risky:
        kinds = sorted({category.display_name for token in risky
                        for category in token.risk_categories})
        opening = (
            "One of these is a value"
            if len(risky) == 1
            else f"{len(risky):,} of these are values"
        )
        lines.append(
            f"{opening} that must never be settled by what sounds plausible "
            f"({', '.join(kinds).lower()}). Getting one of those wrong is invisible "
            "afterwards, because the wrong value reads perfectly well, so they are "
            "flagged whenever any service heard a different value, and settled "
            "only when every service that heard them agreed."
        )
        lines.append("")
    return "\n".join(lines)


def _report_speaker_doubts(transcript: Transcript) -> str:
    """List each stretch of speech whose speaker is in doubt, once.

    Left out entirely when there is none. The text of these words is not in
    question, which is why they are not in the table above; what is in
    question is which of two people said them, and that is one question for
    a whole stretch rather than one for each word in it.
    """
    stretches = speaker_doubt_stretches(transcript.tokens)
    if not stretches:
        return ""
    lines = ["## Where the speaker is in doubt", ""]
    count = (
        "One stretch of speech has"
        if len(stretches) == 1
        else f"{len(stretches):,} stretches of speech have"
    )
    lines.append(
        f"{count} words the services agree on but heard from different people. "
        "Each row gives where the stretch starts and ends and the two speakers "
        "the services heard, so you can play it and decide who was speaking."
    )
    lines.append("")
    lines.append("| From | To | Speakers the services heard |")
    lines.append("| --- | --- | --- |")
    for stretch in stretches:
        lines.append(
            "| {start} | {end} | {speakers} |".format(
                start=format_timestamp(stretch.start),
                end=format_timestamp(stretch.end),
                speakers=_cell(_doubted_speakers_in_words(transcript, stretch.speakers)),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _doubted_speakers_in_words(transcript: Transcript, speakers: Sequence[str]) -> str:
    names = [_speaker_name(transcript, label) for label in speakers] or [_UNKNOWN_SPEAKER]
    if len(names) < 2:
        names.append("another speaker")
    return " or ".join(names)


def _reasons_in_words(token: FinalToken) -> str:
    if not token.review_reasons:
        return "Waiting for a decision"
    return "; ".join(_reason_text(token, reason) for reason in token.review_reasons)


def _reason_text(token: FinalToken, reason: ReviewReason) -> str:
    try:
        return review_reason_text(token, reason)
    except KeyError:
        # A reason added to the model without a description should still be
        # readable rather than crashing the report somebody is waiting on.
        return reason.value.replace("_", " ").capitalize()


def _what_the_services_heard(transcript: Transcript, token: FinalToken) -> str:
    """What each service put in this place, including what was thrown out.

    The rejected words are shown too. A service whose word was discarded is
    still evidence about that moment, and hiding it would leave the reader
    wondering why only two of four services appear.
    """
    heard: dict[Provider, list[str]] = {}

    def remember(provider: Provider, text: str) -> None:
        if not text:
            return
        said = heard.setdefault(provider, [])
        if text not in said:
            said.append(text)

    for reference in token.source_tokens:
        result = transcript.provider_results.get(reference.provider)
        source = result.token_at(reference.index) if result else None
        if source is not None:
            remember(reference.provider, source.text)
    for candidate in token.candidates:
        for provider in candidate.providers:
            remember(provider, candidate.text)
    for reference in token.rejected_tokens:
        result = transcript.provider_results.get(reference.provider)
        source = result.token_at(reference.index) if result else None
        if source is not None:
            remember(reference.provider, f"{source.text} (rejected)")

    if not heard:
        return "nothing recorded"
    return "; ".join(
        f"{provider.display_name}: {' / '.join(heard[provider])}"
        for provider in Provider
        if provider in heard
    )


def _report_escalation(transcript: Transcript) -> str:
    """What was asked a second time, and what the language model made of it."""
    second_opinions = [
        request
        for request in transcript.requests
        if request.purpose not in ("transcription", SMOOTHING_PURPOSE)
    ]
    decided = [token for token in transcript.tokens if token.llm_decision]
    if not second_opinions and not decided:
        return ""

    lines = ["## Second opinions and adjudication", ""]
    if second_opinions:
        by_purpose: dict[tuple[str, Provider], list[ProviderRequestRecord]] = {}
        for request in second_opinions:
            by_purpose.setdefault((request.purpose, request.provider), []).append(request)
        sent_audio = any(request.chunks for request in second_opinions)
        if sent_audio:
            lines.append(
                "Where the first pass could not settle something, a short window of "
                "audio around it was sent again, with several seconds of speech "
                "either side so the service had a sentence to work with rather than "
                "one bare word."
            )
        else:
            lines.append("These are the requests made after the first pass over the audio.")
        lines.append("")
        lines.append("| What for | Service | Requests | Audio sent |")
        lines.append("| --- | --- | ---: | --- |")
        for (purpose, provider), requests in sorted(
            by_purpose.items(), key=lambda item: (item[0][0], item[0][1].value)
        ):
            windows = [
                chunk for request in requests for chunk in request.chunks
            ]
            covered = sum(chunk.canonical_end - chunk.canonical_start for chunk in windows)
            lines.append(
                "| {purpose} | {service} | {count} | {covered} |".format(
                    purpose=_cell(purpose.replace("_", " ").capitalize()),
                    service=_cell(provider.display_name),
                    count=len(requests),
                    # Adjudication sends the words and no audio at all, so
                    # "none" here is the truth rather than a gap in the record.
                    covered=spoken_duration(covered) if windows else "none",
                )
            )
        lines.append("")

    if decided:
        lines.append(
            f"The language model was asked about {_count(len(decided), 'word')} and "
            "gave an answer for each. It is never allowed to invent a word that no service "
            "heard, and it is never allowed to settle an amount, a date or anything "
            "else that must not be guessed at."
        )
        lines.append("")
        lines.append("| Time | What it decided | Now in the transcript | Settled |")
        lines.append("| --- | --- | --- | --- |")
        for token in decided:
            lines.append(
                "| {time} | {decision} | {text} | {settled} |".format(
                    time=format_timestamp(_token_time(token)),
                    decision=_cell(token.llm_decision or ""),
                    text=_cell(render_token_text(token)),
                    settled="No" if token.needs_review else "Yes",
                )
            )
        lines.append("")
    else:
        lines.append("The language model was not asked to decide anything here.")
        lines.append("")
    return "\n".join(lines)


def _report_smoothing(transcript: Transcript) -> str:
    """Whether the smooth transcript was made, what it took and what it cost.

    Left out when smoothing was not asked for, so a run without it says
    nothing about it.
    """
    requests = [
        request for request in transcript.requests if request.purpose == SMOOTHING_PURPOSE
    ]
    failure = next(
        (warning for warning in transcript.warnings if warning.startswith(SMOOTHING_NOT_MADE)),
        None,
    )
    if not requests and failure is None:
        return ""

    lines = ["## The smooth transcript", ""]
    if failure is not None:
        lines.append(f"{failure} No {SMOOTH_EXPORT_NAME} was left beside this transcript.")
    else:
        lines.append(
            f"{SMOOTH_EXPORT_NAME} is an edited copy of this transcript for easy reading. "
            f"{TEXT_EXPORT_NAME} keeps the literal words."
        )
    lines.append("")
    if requests:
        answered = sum(1 for request in requests if request.succeeded)
        took = sum(request.processing_seconds for request in requests)
        models = sorted({_model_description(request) for request in requests})
        sentence = (
            f"It took {_count(len(requests), 'request')} to the language model "
            f"({', '.join(models)}), of which {answered} answered, in "
            f"{format_duration(took)} of request time."
        )
        rate = _smoothing_rate(transcript)
        if rate is not None:
            sentence += (
                f" At the rate in Settings that is an estimated "
                f"{describe_money(len(requests) * rate)}."
            )
        lines.append(sentence)
        lines.append("")
    return "\n".join(lines)


def _smoothing_rate(transcript: Transcript) -> float | None:
    """The price of one smoothing request this run was made with, if recorded."""
    cost = transcript.effective_settings.get("cost")
    if not isinstance(cost, dict):
        return None
    value = cost.get("smoothing_per_request")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _report_warnings(transcript: Transcript) -> str:
    if not transcript.warnings:
        return ""
    lines = ["## Things worth knowing", ""]
    lines.append(
        "These did not stop the transcription, but they change how much of it "
        "rests on how many opinions."
    )
    lines.append("")
    for warning in transcript.warnings:
        lines.append(f"- {warning}")
    lines.append("")
    return "\n".join(lines)


def _cell(value: Any) -> str:
    """Make text safe to put in a Markdown table cell.

    A vertical bar in a transcript would end the cell early and tear the
    table apart, and a line break would end the row, so both are dealt with
    here rather than being hoped against.
    """
    text = str(value).replace("|", "\\|")
    return " ".join(text.split()) or " "
