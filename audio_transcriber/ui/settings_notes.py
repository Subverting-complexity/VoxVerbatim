"""What each setting in the Settings dialog means, written out for the user to read.

This is the same idea as :mod:`audio_transcriber.ui.enhance_notes`, and for
the same reasons. Every setting carries a **summary** of a sentence or two,
which becomes the control's accessible description and its tooltip, and a
full **note**, which is too long to hear on every focus and so is shown in a
panel that follows the focus.

It matters more here than it does there. The Enhance Audio settings are
about loudness, which most people have some feeling for. These settings are
about somebody else's web service: chunk targets, canonical overlap,
reasoning effort, an Azure API version. Nobody should have to already know
what a chunk overlap is before they can use this application, and a label
saying "Chunk overlap" teaches them nothing. So each note says what the
setting actually changes, what it costs in money or in quality, and when it
is worth touching at all.

Two things here differ from the Enhance Audio notes, both because there are
around sixty settings rather than eight.

The first is that a note knows which category it belongs to. The dialog
builds its category list from these, and the guide groups itself by them, so
a note and the page it appears on cannot drift apart.

The second is how a note is looked up. There is no named constant per note,
because sixty constants is a wall of names that says nothing the names
themselves do not. Instead the key of a note **is** the setting it explains,
written as ``section.field`` with the real field names from
:mod:`audio_transcriber.settings`: ``elevenlabs.api_key``,
``processing.provider_timeout_seconds``, ``cost.deepgram_per_minute``. The
settings that live directly on :class:`~audio_transcriber.settings.Settings`
rather than in a section use ``general.`` as their section. A key that names
nothing raises ``KeyError`` the moment the dialog is built, which the tests
catch, so a typo cannot reach a user as a silently missing explanation.

The repeated shapes are built rather than typed out. Every service has an
API key and a free-form parameter box, and those two notes say very nearly
the same thing six times over. Written out by hand they would drift: one
would be corrected and the other five left as they were. Written once and
given the facts that differ, they cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# -- The categories, which are also the pages of the dialog ---------------
#
# The dialog shows these in this order, and the guide reads them in this
# order. They are text rather than an enumeration because they are shown to
# the user exactly as they are written here.

GENERAL = "General"
TRANSCRIPTION = "Transcription"
VOCABULARY = "Vocabulary"
STATISTICS = "Statistics"
COSTS = "Costs"
ELEVENLABS = "ElevenLabs"
OPENAI_TRANSCRIPTION = "OpenAI transcription"
OPENAI_ADJUDICATION = "OpenAI adjudication"
MICROSOFT = "Microsoft MAI"
ASSEMBLYAI = "AssemblyAI"
DEEPGRAM = "Deepgram"

#: Every category, in the order the dialog lists them: the settings that
#: are about the application first, then the ones that are about money, and
#: then one page for each service it talks to.
CATEGORIES: tuple[str, ...] = (
    GENERAL,
    TRANSCRIPTION,
    VOCABULARY,
    STATISTICS,
    COSTS,
    ELEVENLABS,
    OPENAI_TRANSCRIPTION,
    OPENAI_ADJUDICATION,
    MICROSOFT,
    ASSEMBLYAI,
    DEEPGRAM,
)

#: What each page is for, in one sentence. It becomes the accessible
#: description of the page itself, so a screen reader user who has just
#: moved onto a page hears what it holds before working through it.
CATEGORY_SUMMARIES: dict[str, str] = {
    GENERAL: (
        "How the application starts up, and how far the skip buttons move the "
        "playing position."
    ),
    TRANSCRIPTION: (
        "How a transcription run behaves: how long it waits for a service, how it "
        "handles disagreement, and where it writes what it produces."
    ),
    VOCABULARY: (
        "The names and specialised words the services would otherwise get wrong, "
        "kept in lists you choose from before a recording is transcribed."
    ),
    STATISTICS: (
        "What the application has learned about each service from your own "
        "corrections. Nothing here is changed by hand; it is shown so that you can "
        "see it."
    ),
    COSTS: (
        "What each service charges, so a run can be priced before it starts. These "
        "are your figures, checked against what the services publish today."
    ),
    ELEVENLABS: (
        "How the application talks to ElevenLabs Scribe, which supplies the word "
        "timings and the speaker labels every transcript is built on."
    ),
    OPENAI_TRANSCRIPTION: (
        "How the application asks OpenAI what was said. It is not asked for timing "
        "or for speakers; those come from ElevenLabs."
    ),
    OPENAI_ADJUDICATION: (
        "How the application asks an OpenAI reasoning model to settle a word the "
        "other services could not agree on."
    ),
    MICROSOFT: (
        "How the application talks to Microsoft MAI-Transcribe through your own "
        "Azure AI Foundry resource. A third opinion on English and German."
    ),
    ASSEMBLYAI: (
        "How the application asks AssemblyAI for a second opinion on short, "
        "disputed passages, including the ones that may be in Afrikaans."
    ),
    DEEPGRAM: (
        "How the application talks to Deepgram, the optional challenger. Nothing "
        "depends on it, and it is switched off until you ask for it."
    ),
}


@dataclass(frozen=True)
class SettingNote:
    """What one setting is called, and what there is to say about it."""

    key: str
    """The setting this explains, written as ``section.field``."""

    category: str
    """The page it appears on, which is one of :data:`CATEGORIES`."""

    title: str
    """The heading it appears under, and the control's accessible name."""

    summary: str
    """One or two sentences, for the accessible description and the tooltip."""

    note: str
    """The full explanation, for the panel and the guide."""


def _reflowable(text: str) -> str:
    """Put each paragraph on one line, so it can be wrapped to fit.

    The notes are written wrapped in the source below, because that is where
    they are read and reviewed. On screen the wrapping has to follow the
    width of the panel and the size of the user's font instead, and a screen
    reader reads a text box a line at a time, so a paragraph broken across
    eight lines is read as eight lines with a pause in each gap.

    This is deliberately a copy of the same helper in
    :mod:`audio_transcriber.ui.enhance_notes` rather than an import of a
    private name from it. The two sets of notes have nothing to do with each
    other, and five lines of duplication is a smaller price than a
    dependency between them.
    """
    paragraphs = text.strip().split("\n\n")
    return "\n\n".join(
        " ".join(line.strip() for line in paragraph.splitlines()).strip()
        for paragraph in paragraphs
    )


# -- The shapes that repeat once per service ------------------------------


def _api_key_note(
    category: str, section: str, service: str, where: str, billing: str
) -> SettingNote:
    """The note for one service's API key.

    Every one of these says the same four things, so they are said once:
    what the key is and where to get it, why the Show box exists, what
    happens to the key afterwards, and what an empty box means.
    """
    return SettingNote(
        key=f"{section}.api_key",
        category=category,
        title=f"{service} API key",
        summary=(
            f"The credential that lets this application call {service}. Use the Show "
            "box beside it to check what you pasted."
        ),
        note=f"""\
This is the credential that lets this application call {service}. Without it
nothing can be sent, and the page says so in words rather than leaving you to
notice an empty box.

Get it from {where}

The box hides what you type, the way a password box does. That is the right
default and a poor way to check anything, so there is a Show box beside it.
Ticking it reveals the key as ordinary text, which is the only way to be
certain that what you pasted is what you meant to paste: a masked box is read
out as a row of dots, and a key with a space on the end looks exactly like a
key without one. Untick it when you have checked.

{billing}

The key is written to your settings file on this computer, in plain text, so
treat that file the way you would treat a written-down password. It never
leaves the machine except in requests to {service} itself. In particular it is
never written into a transcript: the record of how a transcript was made
strips out anything named like a credential, by rule rather than by a list
that somebody has to remember to update.

Clearing the box switches the service off in practice. It will be reported as
not set up rather than called, which is a clearer failure than a request sent
with no credential.""",
    )


def _parameters_note(
    category: str, key: str, title: str, what_for: str, example: str
) -> SettingNote:
    """The note for one free-form request parameter box.

    These are the boxes that let a user use a parameter the application has
    never heard of, which is how a service can add one without waiting for a
    release of this application. The warning that comes with that freedom is
    the same every time, so it is written once.
    """
    return SettingNote(
        key=key,
        category=category,
        title=title,
        summary=(
            "Named values added to every request, written as JSON. Leave it empty "
            "unless the service has a parameter this application does not know about."
        ),
        note=f"""\
{what_for}

Write it as a JSON object: braces around named values, with a comma between
them. For example:

{example}

This exists so that you are never held up by us. Services add parameters
between releases of this application, and a parameter that arrived this
morning can be typed in this afternoon rather than waited for. That is the
whole benefit, and it is a real one.

The cost is that nothing here is checked against what the service actually
accepts, because we do not know what it accepts. A misspelt name is usually
rejected by the service with a message that says so. A name that is real but
a value that is wrong is worse: the request succeeds and quietly gives you
something other than what you wanted.

What is checked is that the text is JSON and is an object rather than a list
or a bare number. If it is not, OK will not close the dialog. You are told
what is wrong and where, the focus is put back in this box, and every
character you typed is left exactly as you typed it. Nothing you wrote is
thrown away because it did not parse.

An empty box means no extra parameters, which is the right answer almost
always. Leave it empty unless you have a specific reason and the service's
own documentation in front of you.""",
    )


def _rate_note(key: str, service: str, charged_for: str, when_used: str) -> SettingNote:
    """The note for one per-minute rate on the Costs page."""
    return SettingNote(
        key=key,
        category=COSTS,
        title=f"{service} rate per minute",
        summary=(
            f"What one minute of audio sent to {service} is expected to cost, in US "
            "dollars. Used for the estimate, never for billing."
        ),
        note=f"""\
This is what you expect to pay {service} for one minute of audio. {charged_for}

It is used for one thing: working out what a recording will cost before
anything is sent, so that you find out before rather than after. Nobody is
billed from this figure, and changing it changes no invoice. A wrong figure
here does not cost you money; it costs you an estimate you cannot trust,
which is worse than having no estimate at all.

{when_used}

The figure this box starts with was right when it was written and is not
necessarily right now. Prices change, and they change without telling us,
which is exactly why this is a setting rather than a number buried in the
code. Check it against the price the service publishes today, and put the
price you are actually on into the box. If you are on a volume or committed
rate, that is the number to use here, not the list price.""",
    )


_WRITTEN_NOTES: tuple[SettingNote, ...] = (
    # -- General ----------------------------------------------------------
    SettingNote(
        key="general.reopen_last_folder",
        category=GENERAL,
        title="Reopen the last folder when the application starts",
        summary=(
            "Whether the folder you were working in opens again on start-up. With "
            "this off, the application starts with nothing open and waits for you."
        ),
        note="""\
With this on, the application opens the folder you were last working in and
reads it again, so you carry on where you stopped.

With it off, the application starts with no folder open and waits for you to
choose one. The folder is still remembered either way: switching this back on
brings it back rather than starting from nothing.

Switch it off if you work in a different folder every time, or if the folder
lives on a drive that is not always connected and you would rather not wait
for the application to find that out on every start-up.""",
    ),
    SettingNote(
        key="general.short_skip_seconds",
        category=GENERAL,
        title="Short skip",
        summary=(
            "How far the short skip buttons and their keyboard shortcuts move, in "
            "seconds. Between 1 and 3600."
        ),
        note="""\
This is how far the short pair of skip buttons moves the playing position,
and how far the keyboard shortcuts for them move it.

The short skip is the one for going back over a phrase you did not catch, so
a few seconds is usually right. Around 15 seconds suits speech: long enough
to clear a sentence, short enough that two presses do not overshoot the
thing you were listening to.

There are three pairs so that you can move at three scales without stopping
to think: a phrase, a paragraph and a section. Setting all three to similar
numbers loses that. The names of the buttons and their spoken names both
follow whatever you choose here, so a screen reader says the real interval
rather than the word "short".""",
    ),
    SettingNote(
        key="general.medium_skip_seconds",
        category=GENERAL,
        title="Medium skip",
        summary=(
            "How far the medium skip buttons and their keyboard shortcuts move, in "
            "seconds. Between 1 and 3600."
        ),
        note="""\
This is how far the middle pair of skip buttons moves the playing position.

It is the one for moving past a passage rather than back over a phrase.
Around two minutes suits a meeting recording, where it lands you roughly one
topic away from where you were.

Keep it clearly larger than the short skip and clearly smaller than the long
one. Three intervals that are close together give you one interval and two
buttons that do nearly nothing.""",
    ),
    SettingNote(
        key="general.long_skip_seconds",
        category=GENERAL,
        title="Long skip",
        summary=(
            "How far the long skip buttons and their keyboard shortcuts move, in "
            "seconds. Between 1 and 3600."
        ),
        note="""\
This is how far the long pair of skip buttons moves the playing position.

It is for crossing a recording rather than moving about inside a passage.
Around five minutes gets you across an hour-long recording in a dozen
presses, which is fast enough to find a section you half remember.

Very large values are allowed, up to an hour, but they stop being useful
long before that: past a few minutes a skip is no longer navigation, and
dragging the position or typing a time is the better tool.""",
    ),
    # -- Transcription: how a run behaves ---------------------------------
    SettingNote(
        key="processing.default_afrikaans_enabled",
        category=TRANSCRIPTION,
        title="Expect Afrikaans in new recordings",
        summary=(
            "Whether new recordings are treated as possibly containing Afrikaans. "
            "Off switches the whole Afrikaans path out rather than weighting it down."
        ),
        note="""\
This is what new recordings start with. It can still be changed for one
recording; this is only the answer that recording begins with.

Off does not merely make Afrikaans less likely. It switches the whole
Afrikaans path out. Nothing is sent to the model chosen for Afrikaans, and no
passage is considered as Afrikaans at all. That is deliberate: German and
Afrikaans are close enough that a German recording checked for Afrikaans will
sooner or later have a passage handed to the wrong model, and a confidently
wrong Afrikaans transcript of German speech is much harder to spot than an
obviously missing one.

On costs you something real. Passages that may be Afrikaans go to a second,
older model as well, because the better model does not hear Afrikaans at all,
so those passages are transcribed twice and charged twice.

Switch it on once if you record in Afrikaans, and leave it off if you do not.
This is not a setting to toggle per run.""",
    ),
    SettingNote(
        key="processing.default_expected_speaker_count",
        category=TRANSCRIPTION,
        title="Expected number of speakers",
        summary=(
            "How many people a new recording is expected to contain. It is a hint to "
            "the services that tell speakers apart, not a limit."
        ),
        note="""\
This is how many people a new recording is expected to contain. Like the
Afrikaans setting, it is the answer a recording starts with and can be
changed for one recording.

It is passed to the services that separate speakers, which do better when
they are told roughly how many voices to expect than when they have to work
it out from nothing. It is a hint rather than a rule: a recording that turns
out to have four voices is not forced into three.

One is honest for dictation and for a recorded phone note, which is what most
recordings are. For a meeting, count the people who actually speak rather
than the people who attended.

Being roughly right helps. Being precisely right helps very little more, so
do not hold up a run trying to remember whether somebody spoke.""",
    ),
    SettingNote(
        key="processing.escalation_context_seconds_before",
        category=TRANSCRIPTION,
        title="Context before a disputed word",
        summary=(
            "How much audio before a disputed word is sent when it goes out for a "
            "second opinion. Around 5 seconds gives a service a sentence to work with."
        ),
        note="""\
When the services disagree about a word, that word is sent out again on its
own for a second opinion. This is how much of the recording before it goes
with it.

A word sent with no audio around it is a word with no sentence around it, and
a speech service asked to transcribe half a second of audio does markedly
worse than the same service given the whole recording. The context is what
lets it hear the word the way a person would: in a phrase that means
something.

Too much is its own mistake. Send thirty seconds around a disputed word and
the second opinion is no longer about that word; it is another transcript of
a passage, and the passage may contain other disputes that muddy the answer.

Around 5 seconds is a sentence or two of speech, which is what the second
opinion needs. Raise it towards 10 for slow speech or a poor recording. The
audio is charged for, so this and the setting after it are the two numbers
that decide what escalation costs.""",
    ),
    SettingNote(
        key="processing.escalation_context_seconds_after",
        category=TRANSCRIPTION,
        title="Context after a disputed word",
        summary=(
            "How much audio after a disputed word is sent with it. The same trade-off "
            "as the context before it, and usually the same figure."
        ),
        note="""\
This is the other half of the window sent for a second opinion: how much of
the recording after the disputed word goes with it.

What comes after a word settles it more often than you would expect. A
surname is confirmed by the sentence that follows it, and an acronym is
confirmed by what it is said to have done. So this is not filler; it is
evidence.

The same trade-off applies as for the context before. Too little and the
service has nothing to work with; too much and the request stops being about
the disputed word and starts being another transcript.

Keeping the two the same is the sensible default, and around 5 seconds each
gives a ten-second window. Both are charged for, so both are worth knowing
about before a long recording with many disputes.""",
    ),
    SettingNote(
        key="processing.provider_timeout_seconds",
        category=TRANSCRIPTION,
        title="How long to wait for a service",
        summary=(
            "How long one request may take before it is given up on. An hour of audio "
            "takes a service minutes, so short timeouts fail healthy requests."
        ),
        note="""\
This is how long one request to one service may take before the application
stops waiting and treats it as failed.

It has to be generous. A service sent an hour of audio takes minutes to
answer, and it is working the whole time. A timeout that feels reasonable for
a web page is far too short here: it would abandon perfectly healthy requests
and then retry them, so you would pay twice for work that was going to
succeed.

15 minutes is the default and covers a long recording sent to a slow service
comfortably. Raise it if you regularly transcribe recordings of several
hours. Lower it only if you would genuinely rather have a failure than a long
wait, and remember that the retry setting means a failure is not the end of
it.

A request that times out is not billed by most services, but a request that
was going to succeed and was abandoned near the end may well be. That is the
real cost of setting this too low.""",
    ),
    SettingNote(
        key="processing.provider_retry_attempts",
        category=TRANSCRIPTION,
        title="Retry attempts",
        summary=(
            "How many times a failed request is tried again. Zero is allowed and "
            "means a failure is reported rather than paid for twice."
        ),
        note="""\
This is how many extra times a request that failed is sent again before the
application gives up on it.

Most failures against these services are temporary: a rate limit, a busy
moment, a connection that dropped. Trying again a moment later fixes those
without anybody being told, which is why the default is 2 rather than 0.

Zero is a real answer rather than an odd one. These services are charged per
request, and a user watching what a long run costs may honestly prefer to be
told that something failed than to have the application quietly pay for two
more attempts.

Large numbers help less than they look as though they should. If two retries
did not fix it, the fault is usually a wrong key, a wrong endpoint or a
service that is properly down, and none of those is fixed by asking eight
times.""",
    ),
    SettingNote(
        key="processing.provider_retry_backoff_seconds",
        category=TRANSCRIPTION,
        title="Wait before retrying",
        summary=(
            "How long to wait before the first retry. The wait grows for each further "
            "attempt, so that a busy service is given time to recover."
        ),
        note="""\
This is how long the application waits before trying a failed request again.
The wait grows for each further attempt, so a second retry waits longer than
the first.

Waiting is the point. The commonest reason for a request to fail is that the
service asked you to slow down, and retrying immediately is the one response
guaranteed not to help: it arrives while the service is still refusing, uses
up an attempt, and may extend the period you are being held back for.

Two seconds is enough for a rate limit and short enough not to be noticed on
a run that is going well. Raise it to five or ten if you often send many
recordings at once and see requests being refused.

Zero is allowed, and means the retry goes out immediately. It is only
sensible when you are testing something, not for real work.""",
    ),
    SettingNote(
        key="processing.provider_chunk_overlap_seconds",
        category=TRANSCRIPTION,
        title="Overlap between chunks",
        summary=(
            "How much of the previous chunk each chunk repeats, so that a word cut in "
            "half by a boundary is still heard whole by one request."
        ),
        note="""\
A recording that is too large to send in one request is cut into chunks. This
is how much of the previous chunk each one repeats.

The reason it cannot be zero is what happens at the join. Cut a recording at
an arbitrary moment and you will sooner or later cut through the middle of a
word. The request before the cut hears the first half of it and the request
after hears the second half, so neither hears the word, and it comes back as
two fragments that are not words at all.

With an overlap, the boundary is crossed twice: the seconds either side of it
appear at the end of one chunk and at the start of the next, so at least one
request hears the whole word. The duplicated words are then matched up and
dropped afterwards, which is done for you and is why you never see the
repeated seconds in a transcript.

Two seconds covers any single word and most short phrases. Raising it costs
money, because the overlapping audio is sent and charged twice, and it buys
very little past a few seconds. Lowering it to zero saves a trivial amount
and reintroduces broken words at every join, which is a bad trade.""",
    ),
    SettingNote(
        key="processing.forced_alignment_enabled",
        category=TRANSCRIPTION,
        title="Measure the timing of corrected words",
        summary=(
            "Whether a word that changed has its position in the audio measured "
            "again. Off leaves those words pointing at a wider, honest span."
        ),
        note="""\
Every word in a transcript points at the moment in the recording where it was
said, which is what lets you click a word and hear it. When reconciliation
changes a word, the timing that came with the old word no longer belongs to
the new one.

With this on, the changed word is measured against the audio again, so it
points at the moment it was actually said. This is a separate, cheap request
to ElevenLabs, which is the service that does the measuring.

With it off, nothing is measured again and the affected words keep the widest
span that can honestly be defended, and say so. That is not a hidden penalty:
clicking such a word plays a slightly larger piece of the recording rather
than sending you to the wrong place.

Leave it on unless you are counting every request. The cost is small and the
alternative is a transcript whose words point at approximately the right
moment.""",
    ),
    SettingNote(
        key="processing.escalation_enabled",
        category=TRANSCRIPTION,
        title="Send disputed words for a second opinion",
        summary=(
            "Whether words the services disagree about are sent out again with "
            "context. Off sends them straight to the review queue instead."
        ),
        note="""\
When the services disagree about a word and the rules cannot settle it, that
word can be sent out again on its own, with a few seconds of audio around it,
for a service to listen to properly.

This is where most of the quality above a single service comes from, and it
is also where most of the extra cost comes from. Each escalation is a
separate charged request, and a difficult recording can produce a great many
of them, which is what the ceiling on the next page down is for.

With this off, nothing is escalated. Words the services could not agree on go
straight to the review queue for a person to settle by listening. Nothing is
lost and nothing is guessed at; the work simply moves from a service to you.

Switch it off for a rough transcript of a clear recording where you intend to
read it through anyway. Leave it on when the transcript matters more than the
few pence it costs.""",
    ),
    SettingNote(
        key="processing.adjudication_enabled",
        category=TRANSCRIPTION,
        title="Let a reasoning model settle what is left",
        summary=(
            "Whether an OpenAI reasoning model decides between candidate readings the "
            "rules and the second opinions could not. Off sends them to review."
        ),
        note="""\
Some disputes survive everything else. Two services heard two plausible
words, the second opinion agreed with neither, and no rule prefers one over
the other. This decides what happens then.

With it on, the candidate readings are put to an OpenAI reasoning model along
with the surrounding text, and it chooses. This is a reasoning task rather
than a listening one, which is why it is a different model from the one used
for transcription, and it has its own page in this dialog.

With it off, those words go to the review queue instead, marked as unsettled,
and a person decides. That is not a failure state: it is the honest one, and
every choice a person makes there is remembered and used to help next time.

It needs the OpenAI adjudication page to be set up. If it is switched on and
that page is not filled in, you will be told before a run starts rather than
part way through it.""",
    ),
    SettingNote(
        key="processing.maximum_escalations_per_recording",
        category=TRANSCRIPTION,
        title="Most escalations for one recording",
        summary=(
            "A ceiling on how many second opinions one recording may ask for. "
            "Reaching it is not an error: the rest go to the review queue."
        ),
        note="""\
This is a ceiling on how many disputed words one recording may send out for a
second opinion.

It exists because escalation is charged per request and a bad recording can
produce an extraordinary number of disputes. A recording made in a noisy
room, or one where the microphone was in somebody's pocket, can confuse every
service on nearly every word. Without a ceiling that recording would quietly
send thousands of paid requests before anybody noticed.

Reaching the ceiling is not an error and does not stop the run. The
escalations that did happen are used, and the disputes that are left go to
the review queue for a person to settle, exactly as they would if escalation
were switched off. You are told that the ceiling was reached, so a recording
that hit it can be looked at rather than silently half-processed.

200 is a generous allowance for an ordinary recording and a firm stop for a
hopeless one. Lower it if you are working to a budget. Raise it only for a
recording you know is difficult and genuinely want fully processed.""",
    ),
    SettingNote(
        key="processing.transcript_folder_suffix",
        category=TRANSCRIPTION,
        title="Transcript folder name ending",
        summary=(
            'What is added to a recording\'s name to make the folder written beside '
            'it. The default is ".transcript".'
        ),
        note="""\
Everything a run produces for one recording goes into a folder beside that
recording: the transcript itself, the untouched replies from each service,
and the record of how the transcript was made. This is what that folder is
called: the recording's own name with this on the end.

So a recording called meeting.m4a gets a folder called meeting.m4a.transcript
beside it, with the default ending.

Beside the recording rather than in one central place, deliberately. It means
that moving, copying or backing up a recording takes its transcript with it,
and that a drive full of recordings is self-describing rather than depending
on a database somewhere else that may not have come along.

The ending must be usable as part of a folder name. A backslash, a colon or a
".." would put those files somewhere other than beside the recording, and you
would have no idea where your transcripts had gone, so anything of that shape
is refused with a message rather than accepted and quietly changed.""",
    ),
    # -- Vocabulary --------------------------------------------------------
    SettingNote(
        key="vocabulary.profiles",
        category=VOCABULARY,
        title="Vocabulary profiles",
        summary=(
            "Named lists of words the services would otherwise get wrong, kept at "
            "four levels: global, client, project and speaker."
        ),
        note="""\
Speech services are good at ordinary words and weak at exactly the words that
matter most in your recordings: a client's surname, a product nobody else
sells, an acronym that sounds like an everyday word. Every service will accept
a list of words to listen out for, and being told makes a real difference to
whether they come back spelled correctly.

The lists are kept at four levels rather than in one big list. A global list
applies to everything. A client list applies to recordings of that client's
work, a project list to one project, and a speaker list to one person. You
choose which lists apply before a recording is transcribed.

Four levels rather than one because the same word is not equally likely in
every recording. A surname that is almost certain in one client's meeting is
a distraction in somebody else's, and a long list of distractions makes a
service worse rather than better, not merely no better.

When a service will not accept the whole list, the narrower levels are kept
and the wider ones are cut, and within a level the words you have confirmed
most often are kept. So a name on a speaker's own list survives, and a rarely
used entry on the global list is what gets dropped.""",
    ),
    SettingNote(
        key="vocabulary.terms",
        category=VOCABULARY,
        title="Terms in this profile",
        summary=(
            "The words in the selected profile, exactly as they should be spelled. "
            "Add, edit and remove them with the buttons beneath the table."
        ),
        note="""\
These are the words in the profile selected above, spelled exactly as you
want them to appear in a finished transcript. The spelling here is what the
services are asked to listen for and what a corrected word is corrected to.

A term can carry more than its spelling. Its category says what kind of thing
it is, which is used when weighing up a disagreement: two services disagreeing
about something you marked as a person's name is treated more carefully than a
disagreement about an ordinary word. Its language marks a term that belongs to
one language only, which most names do not. Pronunciation hints say how it
sounds, for the services that can use that.

What it usually gets heard as is the most useful part and the least obvious.
Recording that "Kotze" comes back as "cause a" lets reconciliation recognise a
wrong reading as a known mistake rather than as an ordinary rival spelling,
which is a far stronger reason to reject it.

Keep the lists short and specific. Twenty names that really occur in a
client's recordings are worth more than two hundred words added in case they
come up.""",
    ),
    SettingNote(
        key="vocabulary.corrections",
        category=VOCABULARY,
        title="What has been learned from your corrections",
        summary=(
            "Words you have corrected while reviewing transcripts, counted. Ones you "
            "have corrected more than once are offered to the services automatically."
        ),
        note="""\
Every time you change a word while reviewing a transcript, that change is
recorded here along with which service got it wrong.

These are the strongest entries the application has. A word you typed while
listening to your own audio is evidence of two things at once: that the right
spelling is worth listening for next time, and that a particular service
mishears that word, which is used later when weighing one service's reading
against another's.

Corrections are counted rather than listed twice. The same correction made
once is a slip; the same correction made every week is a word the services
cannot hear. Only the count tells those apart, which is why a correction has
to have been made more than once before it is offered to the services as a
term. A service told to listen for a word that was only ever corrected by
accident is being pushed towards a mistake rather than away from one.

Nothing here needs managing. It is shown so that you can see what the
application has learned, and so that a word you keep having to fix is
something you can see rather than something you only feel.""",
    ),
    # -- Statistics ---------------------------------------------------------
    SettingNote(
        key="statistics.accuracy",
        category=STATISTICS,
        title="What has been learned about each service",
        summary=(
            "How often each service was used and how often you corrected it. It is "
            "shown so you can see it; there is nothing here to change."
        ),
        note="""\
This page has nothing to set. It shows what the application has learned about
each service from the corrections you have made while reviewing transcripts.

It is here because this is the one part of the application that quietly changes
what it does on the strength of your own edits. A service you have had to
correct often is believed less readily when the services disagree. That is
worth having, and it is only worth having if you can see it: a person who
cannot see why one service is now preferred has no way to tell learning from
drift, and no way to notice when the numbers say something they know to be
wrong.

Every figure is given with the number of words it was worked out from, and
never as a rate on its own. Three words out of three is a hundred per cent, and
a percentage with nothing behind it invites you to believe it. Each row also
says in words how much evidence sits behind it, and a service you have never
used keeps its row and says so rather than being left out, because a missing
row reads as a faultless service.

The reliability weights are used when the services disagree about a word. They
never override a clear reading, and they cannot make the application ignore a
service altogether.""",
    ),
    # -- Costs -------------------------------------------------------------
    SettingNote(
        key="cost.confirm_before_running",
        category=COSTS,
        title="Show the estimated cost before a run starts",
        summary=(
            "Whether a run stops to show what it is expected to cost and wait for you "
            "to agree. On is strongly recommended."
        ),
        note="""\
With this on, starting a run works out what it is expected to cost from the
length of the recording and the rates on this page, shows you the figure, and
waits for you to agree before anything is sent.

Five metered services and a reasoning model add up to a real amount of money
on a long recording, and finding that out afterwards is not something anybody
should have to do. The estimate is also the moment you notice that you asked
for a three-hour recording by mistake, or that Deepgram is switched on when
you thought it was not.

The figure is an estimate and is honest about that. Escalation and
adjudication depend on how much the services disagree, which cannot be known
before they have listened, so the estimate gives a range for those rather
than pretending to a single number.

Switch it off only when you are running many recordings you have already
priced and the confirmation is genuinely in the way.""",
    ),
    _rate_note(
        "cost.elevenlabs_per_minute",
        "ElevenLabs Scribe",
        "Every minute of every recording goes to it, because it supplies the timings "
        "and speaker labels the whole transcript is built on.",
        "It applies to the whole length of every recording you transcribe, so it is "
        "usually the largest single line in an estimate.",
    ),
    _rate_note(
        "cost.openai_transcription_per_minute",
        "OpenAI transcription",
        "Every minute of every recording goes to it as well, as the second reading "
        "that ElevenLabs is checked against.",
        "It applies to the whole length of every recording, the same as ElevenLabs. "
        "If your recordings are chunked, the overlap between chunks is sent twice and "
        "is charged twice, so a large overlap shows up here.",
    ),
    _rate_note(
        "cost.microsoft_per_minute",
        "Microsoft MAI",
        "It is a third opinion on English and German, and it is the most expensive of "
        "the transcription services per minute.",
        "It applies only when Microsoft MAI is switched on, and only to recordings in "
        "the languages it is trusted with. Switching it off removes this line "
        "entirely.",
    ),
    _rate_note(
        "cost.assemblyai_per_minute",
        "AssemblyAI",
        "It is not sent whole recordings. It is sent short windows around disputed "
        "words, so what it charges follows how much the other services disagreed.",
        "The minutes counted here are the escalation windows rather than the "
        "recording, so this line depends far more on how difficult a recording is "
        "than on how long it is.",
    ),
    _rate_note(
        "cost.deepgram_per_minute",
        "Deepgram",
        "It is the optional challenger, brought in when the others remain split and "
        "for comparing services against each other.",
        "It applies only when Deepgram is switched on, which it is not by default. "
        "Nothing in a transcript depends on it.",
    ),
    SettingNote(
        key="cost.adjudication_per_request",
        category=COSTS,
        title="Adjudication cost per request",
        summary=(
            "What one call to the reasoning model is expected to cost, averaged. It "
            "is charged per request rather than per minute of audio."
        ),
        note="""\
This is what you expect one adjudication to cost: one call to the reasoning
model to settle one word the other services could not agree on.

It is charged per request rather than per minute, because no audio is sent. What
goes is the candidate readings and the text around them, so the price follows
how much evidence the disputed passage carries and how hard the model was asked
to think, not how long the recording is.

That makes this figure an average of very unequal requests rather than a price.
A word disputed in a short, clear sentence costs a fraction of one disputed in
a dense passage with three candidate readings, and raising the reasoning effort
on the adjudication page raises it again.

It matters most on difficult recordings, where the number of requests is large.
Multiply it by the ceiling on escalations to see the worst case a single
recording can reach.""",
    ),
    # -- ElevenLabs --------------------------------------------------------
    SettingNote(
        key="elevenlabs.enabled",
        category=ELEVENLABS,
        title="Use ElevenLabs Scribe",
        summary=(
            "Whether ElevenLabs is called at all. It supplies the timings and speaker "
            "labels every transcript is built on, so a run without it is not a run."
        ),
        note="""\
This switches ElevenLabs Scribe on and off.

Off is allowed but is close to meaningless for real work. Scribe is the
structural backbone of a transcript: the word timings, the speaker labels and
the confidence figures that decide what goes to the review queue all come from
it, and nothing else here supplies them. A run without it has no timings to
click, no speakers to name and nothing to measure against.

If you switch it off, you are told before a run starts that a transcript
cannot be made, rather than getting a transcript that quietly lacks half of
what a transcript is.

The switch is here for one honest reason: so that a service having a bad day
can be taken out of the picture while you work out what is happening, without
your key being deleted.""",
    ),
    _api_key_note(
        ELEVENLABS,
        "elevenlabs",
        "ElevenLabs",
        "your ElevenLabs account page, under your profile. It is shown once when it "
        "is created, so copy it then.",
        "Everything sent with this key is billed to your ElevenLabs account. Every "
        "minute of every recording you transcribe goes to this service, so this is "
        "usually the largest part of what a run costs.",
    ),
    SettingNote(
        key="elevenlabs.transcription_model",
        category=ELEVENLABS,
        title="ElevenLabs transcription model",
        summary=(
            "Which Scribe model is asked to transcribe. Free text, so a model "
            "released tomorrow can be used tomorrow."
        ),
        note="""\
This is the name of the ElevenLabs model that does the transcribing, sent with
every request exactly as you type it.

It is a box you type into rather than a list to choose from, and that is the
point. These services rename and replace their models far more often than this
application is released. A list would mean that using a model announced this
morning needs a new version of this application, which is a poor reason to be
unable to use something you are paying for.

The cost of that freedom is that a name with a typing mistake in it is not
caught here. It is caught by ElevenLabs, which rejects it with a message that
says the model does not exist, and you will see that message rather than a
silent failure.

Change it when ElevenLabs publishes a newer Scribe model and you have read
what changed. Timings, speaker labels and confidence figures all come from
here, so a model that transcribes slightly better but times words worse is not
an improvement for this application.""",
    ),
    SettingNote(
        key="elevenlabs.diarise",
        category=ELEVENLABS,
        title="Tell the speakers apart",
        summary=(
            "Whether Scribe is asked which speaker said each word. On, because who "
            "said something is one of the things a transcript is for."
        ),
        note="""\
With this on, Scribe is asked to work out which speaker said each word, and
the transcript can say so.

Who said something is one of the three questions a transcript answers, along
with what was said and when. Scribe is the only service here that is asked for
it, so switching this off means no transcript from this application has
speakers in it, however many people were in the room.

It is on by default for that reason. The number of speakers to expect is taken
from the Transcription page, and telling it roughly how many voices there are
makes it noticeably better at the job.

Switch it off only for recordings that genuinely have one voice and where you
would rather not pay for the extra work, such as dictation. Even then the
saving is small.""",
    ),
    SettingNote(
        key="elevenlabs.tag_audio_events",
        category=ELEVENLABS,
        title="Mark laughter, applause and other sounds",
        summary=(
            "Whether non-speech sounds come back marked as what they are. Off makes "
            "them arrive as ordinary-looking words nobody said."
        ),
        note="""\
With this on, laughter, applause, coughing and similar sounds come back marked
as events rather than as words.

Off does not make them disappear. It makes them arrive as ordinary-looking
words, because the service transcribes what it hears and a laugh sounds
somewhat like speech. That puts things nobody said into what is supposed to be
a verbatim transcript, and there is then no way to tell them from things
somebody did say.

Marked, they can be shown, hidden, or counted as you prefer, and they are
never confused with speech during reconciliation.

Leave it on. The only reason to switch it off is a service charging extra for
it, and none of them currently does.""",
    ),
    _parameters_note(
        ELEVENLABS,
        "elevenlabs.transcription_parameters",
        "Extra ElevenLabs transcription parameters",
        "These are added to every transcription request sent to ElevenLabs, on top of "
        "the model, the diarisation and the audio-event settings above.",
        '{"language_code": "en", "num_speakers": 3}',
    ),
    _parameters_note(
        ELEVENLABS,
        "elevenlabs.forced_alignment_parameters",
        "Extra ElevenLabs alignment parameters",
        "These are added to the separate, much cheaper requests that measure where a "
        "corrected word actually falls in the audio. Alignment takes no model name of "
        "its own: it is the same service listening for a word you already know.",
        '{"language_code": "en"}',
    ),
    # -- OpenAI transcription ---------------------------------------------
    SettingNote(
        key="openai_transcription.enabled",
        category=OPENAI_TRANSCRIPTION,
        title="Use OpenAI for transcription",
        summary=(
            "Whether OpenAI is asked what was said. It is the second full reading "
            "that ElevenLabs is checked against, so a run needs it."
        ),
        note="""\
This switches OpenAI transcription on and off.

Its job in the ensemble is to work out what was probably said, as a second
full reading of the whole recording. Almost everything this application does
beyond a single service depends on having two independent readings to compare:
without a second one there is nothing to disagree, nothing to escalate and
nothing to adjudicate, and the confidence figures have nothing to check
themselves against.

So a run with this off is not a run. You are told before anything is sent
rather than at the end.

It is asked for words only. It is not asked for timing or for speakers, both
of which come from ElevenLabs, so nothing on this page mentions either.""",
    ),
    _api_key_note(
        OPENAI_TRANSCRIPTION,
        "openai_transcription",
        "OpenAI",
        "the API keys page of the OpenAI platform site. It is shown once when it is "
        "created, so copy it then.",
        "Everything sent with this key is billed to your OpenAI account. This page and "
        "the adjudication page can use the same key or different ones; separate keys "
        "on separate projects are worth the small trouble, because they let you see "
        "what transcription costs and what adjudication costs as two figures rather "
        "than one.",
    ),
    SettingNote(
        key="openai_transcription.model",
        category=OPENAI_TRANSCRIPTION,
        title="OpenAI transcription model",
        summary=(
            "Which OpenAI model transcribes. This must stay in the gpt-transcribe "
            "family; Whisper and the older models are not to be used here."
        ),
        note="""\
This is the name of the OpenAI model that transcribes, sent with every request
exactly as you type it.

It is free text for the same reason as every other model name here: the
services rename their models far faster than this application is released.

Unlike the others, this one has a rule attached that is worth knowing. The
transcription stage is to use the current gpt-transcribe family and nothing
else. Not Whisper, and not the older GPT-4o transcription models, even to gain
a feature such as word timing. Timing comes from ElevenLabs, so there is
nothing to be gained by reaching for an older model that offers it, and a good
deal to be lost in accuracy.

Change it when OpenAI publishes a newer model in that family.""",
    ),
    SettingNote(
        key="openai_transcription.chunk_target_bytes",
        category=OPENAI_TRANSCRIPTION,
        title="Chunk size to aim for",
        summary=(
            "How large each piece of a long recording is aimed at, in bytes. OpenAI "
            "refuses anything over 25 MB, so this aims well under it."
        ),
        note="""\
OpenAI will not accept a request larger than 25 million bytes, which is about
25 MB. A recording bigger than that has to be cut into pieces, and this is the
size each piece is aimed at.

It is aimed at rather than set to, and that is the whole reason this is a
setting. How large a piece of audio will be once it has been encoded cannot be
known exactly until it has been encoded, so the target has to sit below the
limit with room to spare. A chunk that comes out slightly over the limit is
not a slightly large chunk; it is a failed request.

20 million bytes leaves 5 million bytes of room, which is enough for any
ordinary recording. Lower it if you see requests being refused for size, which
happens with unusual encodings that expand more than expected.

Raising it towards the limit saves very little. Fewer, larger chunks mean
slightly less overlap sent twice, but the risk of a chunk crossing the limit
rises much faster than the saving does.""",
    ),
    _parameters_note(
        OPENAI_TRANSCRIPTION,
        "openai_transcription.parameters",
        "Extra OpenAI transcription parameters",
        "These are added to every transcription request sent to OpenAI, on top of the "
        "model name above.",
        '{"language": "en", "temperature": 0}',
    ),
    # -- OpenAI adjudication ------------------------------------------------
    SettingNote(
        key="openai_adjudication.enabled",
        category=OPENAI_ADJUDICATION,
        title="Use OpenAI for adjudication",
        summary=(
            "Whether this service is available to settle disputes. The Transcription "
            "page decides whether adjudication is used at all."
        ),
        note="""\
This switches the adjudication service on and off.

There are two switches for adjudication and they mean different things. The
one on the Transcription page decides whether disputes are adjudicated at all.
This one decides whether this particular service is available to do it. In
practice you want both on or the feature off, and switching this one off while
the other stays on is what produces the warning that adjudication is switched
on but not set up.

Nothing in a transcript fails when adjudication is unavailable. The disputes
that would have been settled go to the review queue instead, marked as
unsettled, and a person decides. Every decision made there is remembered.

The specification allows no other company's model in this role, so there is no
alternative service to point this at.""",
    ),
    _api_key_note(
        OPENAI_ADJUDICATION,
        "openai_adjudication",
        "OpenAI",
        "the API keys page of the OpenAI platform site, the same place as the "
        "transcription key.",
        "Everything sent with this key is billed to your OpenAI account, per request "
        "rather than per minute of audio. A separate key from the transcription one, "
        "on its own project, is worth the small trouble: it is the only easy way to "
        "see what adjudication is costing you on its own.",
    ),
    SettingNote(
        key="openai_adjudication.model",
        category=OPENAI_ADJUDICATION,
        title="Adjudication model",
        summary=(
            "Which OpenAI model settles disputes. It is a reasoning task, so this is a "
            "different and freer choice than the transcription model."
        ),
        note="""\
This is the model asked to decide between candidate readings when the rules and
the second opinions could not.

It is a different model from the transcription one, and the choice is a freer
one. Deciding which of two plausible words fits a sentence is a reasoning task
rather than a listening one: the model never hears the audio. It is given the
candidate readings, what each service reported, and the text around them.

That means the reasoning models are the right family here, and they move
faster than anything else named in this dialog, which is why this is plain text
rather than a list. A name that does not exist is rejected by OpenAI with a
message saying so.

A more capable model settles more disputes correctly and costs more per
request. On a difficult recording with hundreds of disputes that difference is
noticeable, so this is one of the few settings where a run's cost and its
quality really are traded directly against each other.""",
    ),
    SettingNote(
        key="openai_adjudication.reasoning_effort",
        category=OPENAI_ADJUDICATION,
        title="Reasoning effort",
        summary=(
            "How hard the model is asked to think. Leave it empty to leave the "
            "parameter out of the request, which some models require."
        ),
        note="""\
This says how much thinking the adjudicating model should do before it answers.
More thinking settles hard cases better and costs more per request, because
what the model thinks is billed along with what it says.

Empty is a real answer rather than a missing one, and it is the reason this is
a box rather than a list. Models differ in whether they accept a reasoning
control at all, and sending one to a model that has none is an error that fails
the request. Clearing this box is how you say "do not send it", which is what a
model without the parameter needs.

The words a model accepts are its own business and have changed more than once,
which is the other reason this is free text. Recent models take low, medium and
high, and some take minimal as well. Check what the model you have chosen
accepts.

Medium is a sensible starting point. Raise it if adjudications are coming back
wrong on cases you think are settleable; clear it if the model rejects the
request because it does not know the parameter.""",
    ),
    _parameters_note(
        OPENAI_ADJUDICATION,
        "openai_adjudication.parameters",
        "Extra adjudication parameters",
        "These are added to every adjudication request, on top of the model and the "
        "reasoning effort above.",
        '{"max_output_tokens": 2000}',
    ),
    # -- Microsoft MAI ------------------------------------------------------
    SettingNote(
        key="microsoft.enabled",
        category=MICROSOFT,
        title="Use Microsoft MAI",
        summary=(
            "Whether Microsoft MAI is called. It is an optional third opinion on "
            "English and German, and nothing depends on it."
        ),
        note="""\
This switches Microsoft MAI-Transcribe on and off.

It is a third opinion rather than a necessity. Where the two required services
disagree, a third independent reading often settles the matter without anything
having to be escalated or adjudicated, which is where it earns its cost.

It is never depended upon, deliberately. It is a preview service, and a
transcript must not fail because a preview service did not answer. If it is
switched on and does not respond, the run carries on without it and says so.

It is not to be trusted on Afrikaans, so Afrikaans passages do not go to it
whatever this is set to.

It is the most expensive of the transcription services per minute, so on a long
recording this switch is a real decision rather than a free improvement.""",
    ),
    _api_key_note(
        MICROSOFT,
        "microsoft",
        "Microsoft MAI",
        "the Keys and Endpoint page of your Azure AI Foundry resource, in the Azure "
        "portal. Azure gives you two keys and either will work, so that you can "
        "rotate one while the other is in use.",
        "Everything sent with this key is billed to the Azure subscription that owns "
        "the resource. MAI is the most expensive of the transcription services per "
        "minute of audio.",
    ),
    SettingNote(
        key="microsoft.endpoint",
        category=MICROSOFT,
        title="Azure resource endpoint",
        summary=(
            "The web address of your own Azure AI Foundry resource. There is no "
            "default, because the address is yours rather than ours."
        ),
        note="""\
This is the address the requests are sent to, and it is the setting most likely
to be the reason this service is not working.

Unlike the other services here, Microsoft MAI has no fixed address we could put
in the code. You create a resource in your own Azure subscription and it gets
an address of its own, something like
https://your-resource.cognitiveservices.azure.com. That is why this box starts
empty: there is no sensible value to guess, and a plausible-looking wrong one
would be worse than nothing.

Copy it from the Keys and Endpoint page of your resource in the Azure portal,
the same page the key comes from. Paste the address as it is given, without a
path on the end.

A key with no endpoint has nowhere to go, so this service is reported as not
set up until both are filled in.""",
    ),
    SettingNote(
        key="microsoft.model",
        category=MICROSOFT,
        title="Microsoft model",
        summary=(
            "Which model your Azure resource is asked for. It is named in the request "
            "rather than in the address, so it is a setting of its own."
        ),
        note="""\
This is the name of the model your Azure resource is asked for.

Azure works differently from the other services here: the model is named in the
body of the request rather than being part of the address. That is why the
endpoint and the model are two settings rather than one, and why changing model
does not mean creating a new resource.

It is free text, like every other model name in this dialog, so a newer
MAI-Transcribe can be used the day it appears.

The name has to match a model your resource has actually been given access to.
An unknown name comes back from Azure as a clear error rather than as a wrong
transcript.""",
    ),
    SettingNote(
        key="microsoft.api_version",
        category=MICROSOFT,
        title="Azure API version",
        summary=(
            "Which revision of the Azure API to speak. Azure requires it on every "
            "request and changes it independently of the model."
        ),
        note="""\
Azure requires every request to say which revision of its API it is written
against, as a date. This is that date.

It is a separate setting because Azure changes it on its own schedule, without
reference to the model. A new API version can arrive while your model stays the
same, and a new model can arrive that needs a newer API version than the one
you are on.

Pinning it is the point. A request that says which revision it speaks keeps
working when Azure changes what a newer revision means, which is exactly the
protection you want on a service you depend on. The alternative, always using
the newest, means your transcripts can change because somebody else shipped
something.

Change it when the model you want needs a newer one, and read what changed
before you do. A version Azure does not recognise is refused with a message
that lists the ones it does.""",
    ),
    _parameters_note(
        MICROSOFT,
        "microsoft.parameters",
        "Extra Microsoft parameters",
        "These are added to every request sent to your Azure resource, on top of the "
        "model and the API version above.",
        '{"locales": ["en-GB", "de-DE"]}',
    ),
    # -- AssemblyAI ---------------------------------------------------------
    SettingNote(
        key="assemblyai.enabled",
        category=ASSEMBLYAI,
        title="Use AssemblyAI",
        summary=(
            "Whether AssemblyAI is called. It is the service that listens again to "
            "disputed passages, and the only one that can hear Afrikaans."
        ),
        note="""\
This switches AssemblyAI on and off.

Its job is different from the other services here. It is not sent whole
recordings. It is sent short windows of audio around the words the others could
not agree on, and around the speaker boundaries that look wrong, and asked to
listen again with fresh ears.

That makes it cheap in a way the per-minute rate hides: what it charges follows
how much the services disagreed rather than how long the recording is. A clear
recording barely uses it at all.

It is also the only service here that can hear Afrikaans, through its older
model. Switching it off with Afrikaans switched on leaves Afrikaans passages
with nothing to check them, so those two settings belong together.

Nothing fails if it is switched off. Disputes that would have gone to it go to
the review queue instead.""",
    ),
    _api_key_note(
        ASSEMBLYAI,
        "assemblyai",
        "AssemblyAI",
        "your AssemblyAI account dashboard, where it is shown on the home page and "
        "can be copied at any time.",
        "Everything sent with this key is billed to your AssemblyAI account. Because "
        "it is sent short windows rather than whole recordings, what it costs follows "
        "how much the other services disagreed and how much context you send with "
        "each disputed word.",
    ),
    SettingNote(
        key="assemblyai.primary_model",
        category=ASSEMBLYAI,
        title="AssemblyAI primary model",
        summary=(
            "The better model, used for everything except Afrikaans. It is more "
            "accurate and covers fewer languages."
        ),
        note="""\
This is the model used for second opinions in every language except Afrikaans.

It is the more accurate of the two and the one that should do nearly all the
work. Its limitation is the number of languages it covers, which is what the
second model name below exists for.

It is free text, like every model name here, so a newer model can be used as
soon as AssemblyAI publishes one.

If you never record Afrikaans, this is the only model name on this page that
matters.""",
    ),
    SettingNote(
        key="assemblyai.afrikaans_model",
        category=ASSEMBLYAI,
        title="AssemblyAI Afrikaans model",
        summary=(
            "The older model, used only for passages that may be Afrikaans, because "
            "the better model does not cover that language."
        ),
        note="""\
This is the model used for passages that may be Afrikaans, and only for those.

There are two model names on this page because the better model does not hear
Afrikaans at all. That is not a matter of it being worse at it: the language is
not in its list. So a passage that may be Afrikaans has to go to the older
model, which does cover it, or go nowhere.

Keeping the two apart as named settings is what lets that choice be made per
passage rather than per account. An English recording is not dragged down to
the older model because you sometimes record in Afrikaans, and an Afrikaans
passage is not sent to a model that cannot hear it.

It is only used when Afrikaans is switched on for a recording, on the
Transcription page or for that recording alone. If you never record Afrikaans,
this box is never read.""",
    ),
    _parameters_note(
        ASSEMBLYAI,
        "assemblyai.parameters",
        "Extra AssemblyAI parameters",
        "These are added to every request sent to AssemblyAI, on top of whichever of "
        "the two model names above applies.",
        '{"punctuate": true, "format_text": true}',
    ),
    # -- Deepgram -----------------------------------------------------------
    SettingNote(
        key="deepgram.enabled",
        category=DEEPGRAM,
        title="Use Deepgram",
        summary=(
            "Whether Deepgram is called at all. It is the optional challenger and is "
            "off until you ask for it."
        ),
        note="""\
This switches Deepgram on and off, and it is the one service here that starts
switched off.

Nothing in this application depends on it. It is brought in when the other
services remain split and there is still a decision to make, and it is useful
for comparing the services against each other on your own recordings rather
than on somebody's benchmark.

Switching it on adds a per-minute charge and, usually, a small improvement on
difficult passages. Whether that is worth it depends on your recordings, which
is precisely what you can find out by switching it on for a few of them and
looking at what it changed.

It stays off by default because a service nobody asked for should not appear on
an invoice.""",
    ),
    _api_key_note(
        DEEPGRAM,
        "deepgram",
        "Deepgram",
        "your Deepgram console, under API keys, where you can create one and copy it "
        "when it is shown.",
        "Everything sent with this key is billed to your Deepgram account. Deepgram is "
        "switched off by default, so this key costs you nothing until you switch the "
        "service on.",
    ),
    SettingNote(
        key="deepgram.model",
        category=DEEPGRAM,
        title="Deepgram model",
        summary=(
            "Which Deepgram model is asked to transcribe. Free text, so a newer Nova "
            "can be used the day it appears."
        ),
        note="""\
This is the name of the Deepgram model used, sent with every request exactly as
you type it.

Free text, like every model name in this dialog, because Deepgram publishes new
models on its own schedule and waiting for a release of this application to use
one would be a poor reason not to.

Nova-3 is the current general model and a sensible default. Deepgram also
publishes models specialised for particular kinds of audio, such as telephone
recordings, and those are worth trying if your recordings are all of one kind.

One thing follows from this choice that is easy to miss. Your vocabulary lists
reach Deepgram as key terms, and only Nova-3 and Flux accept those. Naming an
older model here means your names and specialised words are not sent at all,
and the transcript comes back with them spelled as the model guessed. The
application says so in its log when it happens, but the model name is where the
decision is made.

An unknown name is refused by Deepgram with a message saying so, so a typing
mistake shows up as an error rather than as a poor transcript.""",
    ),
    _parameters_note(
        DEEPGRAM,
        "deepgram.parameters",
        "Extra Deepgram parameters",
        "These are added to every request sent to Deepgram, on top of the model name "
        "above.",
        '{"smart_format": true, "filler_words": false}',
    ),
)

#: The notes as the rest of the application sees them, with each paragraph
#: on a single line ready to be wrapped to whatever width it is shown at.
_NOTES: tuple[SettingNote, ...] = tuple(
    replace(note, note=_reflowable(note.note)) for note in _WRITTEN_NOTES
)

NOTES: dict[str, SettingNote] = {note.key: note for note in _NOTES}


def note_for(key: str) -> SettingNote:
    """Return the note filed under ``key``.

    Raises:
        KeyError: if there is no such note, which is a mistake in the code
            rather than anything the user can cause. The dialog looks every
            note up as it builds each control, so a wrong key fails on the
            first test that opens the dialog rather than reaching a user as
            a control with no explanation.
    """
    return NOTES[key]


def summary_of(key: str) -> str:
    """The short form, for an accessible description and a tooltip."""
    return NOTES[key].summary


def note_text(key: str) -> str:
    """The full explanation, for the panel and the guide."""
    return NOTES[key].note


def notes_in(category: str) -> tuple[SettingNote, ...]:
    """Every note belonging to one page, in the order it was written."""
    return tuple(note for note in _NOTES if note.category == category)


def guide_text() -> str:
    """Every note, grouped by page, for reading straight through.

    The panel in the dialog shows one note at a time, which suits somebody
    working through the settings. This is the same words laid out for
    somebody who would rather read a page before touching anything, or copy
    the lot somewhere else.
    """
    parts = ["Settings: what each one does", ""]
    for category in CATEGORIES:
        parts.append(category.upper())
        parts.append("=" * len(category))
        parts.append("")
        parts.append(_reflowable(CATEGORY_SUMMARIES[category]))
        parts.append("")
        for note in notes_in(category):
            parts.append(note.title)
            parts.append("-" * len(note.title))
            parts.append("")
            parts.append(note.note)
            parts.append("")
    return "\n".join(parts).rstrip() + "\n"
