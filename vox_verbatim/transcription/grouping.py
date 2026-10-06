"""Turning a folder of transcripts into work a person can actually do.

A transcript of an hour of speech holds perhaps eight thousand words, and a
few hundred of them will be shaky. Presented as a list, that is an afternoon
of clicking through the same surname forty times. Presented as thirty groups
of related words, it is twenty minutes, because the surname is decided once
and the decision lands on all forty. This module builds those groups, keeps
them across a re-transcription, and answers a file transcribed next week with
what the person has already decided.

**Grouping is generous and rules are strict, and the difference is the whole
design.** They look like the same question -- "are these two the same word?"
-- and they are not, because what happens to the answer is different.

A group is a *suggestion*. It is put in front of a person who has the audio
in their ears and can see every member, and if the analysis has swept in a
word that does not belong, the person takes it out in one keystroke and
nothing was ever at stake. So grouping may reach: ``Bosch``, ``Bosh`` and
``Bosche`` are gathered together on a similarity score, and being wrong about
that occasionally costs somebody two seconds.

A replacement rule *rewrites a word in a file nobody has opened*. There is no
person, no audio and no chance to object. So a rule fires only on an exact
match of the normalised form, with the language agreeing where both are
known, and on nothing else.

Somebody will eventually notice that this module already contains a perfectly
good similarity measure and propose reusing it for rule matching, on the
grounds that the feature would then feel cleverer: a rule for ``Bosch`` would
also answer ``Bosche`` and the queue would be shorter. **That would be a
mistake, and it is worth being precise about why.** The similarity threshold
below is set where it is because a wrong grouping is cheap. Wired to rules,
the same threshold silently rewrites a word on evidence chosen for a case
where being wrong did not matter, in a file the person has not read, under a
correction they accepted for a different spelling. The damage is invisible:
the transcript reads perfectly well, and nothing in it says that a machine
picked the word. The shorter queue is bought with exactly the kind of quiet
error the whole review workflow exists to prevent. A form nobody has ever
accepted a replacement for belongs in the queue, next to the forms that are
already settled, where one action settles it too.

**A word with no saved strength is left out of the sweep entirely.** It
cannot be thresholded honestly, and inventing a value for it would put a word
in front of a person for a reason nobody could explain. Old transcripts, made
before ``confidence_strength`` existed, are full of such words, and so are
two live paths through the current pipeline: a word inserted by the
reconciliation rules, and a word the adjudicating language model moved, are
both flagged by rule rather than by assessment and keep no strength.

That is not a hole and must not be filled. Those words all carry review
reasons, so they reach the person by the other road: they appear as existing
uncertainty items, which the Review window shows by default. The two paths
cover the ground between them. The sweep finds words that are *quietly* weak,
with nothing but a low number to say so. The uncertainty items carry words
that were flagged *for a stated reason*. Giving the second kind an invented
strength so that the sweep could also find them would put the same word in
front of the person twice, the second time under a number nobody could
justify.

**Occurrence identity survives a re-transcription; token identifiers do not.**
``FinalToken.id`` is a fresh uuid on every run, so it is worthless for
deciding whether the word in this transcript is the word somebody made a
decision about last week. Re-matching works on the recording, the normalised
text, a start time within :data:`OCCURRENCE_TIME_TOLERANCE`, and the
surrounding words where two candidates are otherwise equally good. Anything
that cannot be matched confidently comes back marked stale and is never
attached to the nearest available word, because applying somebody's decision
to a word they never looked at is worse than telling them the decision could
not be kept, and it is worse precisely because it is invisible.

Nothing here imports Qt, so the whole analysis can be tested without a
desktop, and nothing here talks to a network or to a language model. The same
folder analysed twice gives the same answer twice.
"""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime
from difflib import SequenceMatcher

from vox_verbatim.transcription.model import FinalToken, Language, Transcript
from vox_verbatim.transcription.normalise import (
    _compound_form,
    _has_german_letter,
    is_punctuation_only,
    normalise,
)
from vox_verbatim.transcription.project import (
    Occurrence,
    ProjectState,
    ReplacementRule,
    WordGroup,
)

#: How alike two normalised forms must be before grouping will suggest that
#: they are the same intended word. The figure is a
#: :class:`difflib.SequenceMatcher` ratio, which is twice the number of
#: matching characters over the combined length of both strings.
#:
#: It is set where it is by the case the specification names. ``bosch`` and
#: ``bosh`` score 0.89, ``bosch`` and ``bosche`` score 0.91, and both are
#: comfortably above this line; ``bosh`` and ``bosche`` score exactly 0.80 and
#: are **not** joined directly, but end up in the same group anyway because
#: each of them is joined to ``bosch``. That is the intended behaviour and it
#: is why the clustering below links rather than compares every pair against a
#: single centre.
#:
#: The trade in both directions. Lower it and chains grow: at 0.75 a single
#: changed letter in a four-letter word qualifies, which sweeps ``form`` and
#: ``from`` together and, worse, lets a chain of near neighbours run from one
#: genuine word to another through a middle that resembles both. A person then
#: opens a group of nine and finds three different words in it, which is more
#: work than no grouping at all. Raise it and the opposite happens: at 0.90 a
#: dropped letter in a short surname no longer qualifies, ``Bosh`` sits in its
#: own group of one, and the person decides the same surname three times. The
#: cost of grouping too eagerly is a moment's annoyance; the cost of grouping
#: too timidly is the repetition this whole feature exists to remove, so the
#: line sits nearer the eager end.
SIMILARITY_THRESHOLD = 0.82

#: Similarity is only consulted when the shorter of the two normalised forms
#: is at least this long. One character out of three is a third of a word, and
#: at the threshold above ``the`` and ``they``, ``and`` and ``land``, ``our``
#: and ``four`` all qualify, which is a group of common short words that have
#: nothing to do with each other. Short words are still gathered by an exact
#: normalised match and by :func:`~vox_verbatim.transcription.normalise.are_equivalent`,
#: which are the two tests that cannot be wrong; only the guessing stage is
#: withheld from them.
SIMILARITY_MINIMUM_LENGTH = 4

#: How far a word may have moved and still be the same word, in seconds. A
#: re-transcription shifts word boundaries by a few tens of milliseconds where
#: nothing changed and by rather more where a neighbouring word was split or
#: merged. Three quarters of a second is wide enough to absorb that and narrow
#: enough that the same word said twice in one sentence does not fall inside
#: it.
OCCURRENCE_TIME_TOLERANCE = 0.75

#: Two candidates whose start times are this close to each other are treated
#: as equally good, and the surrounding words decide between them. Without a
#: margin, a difference of a millisecond would settle the question, which is a
#: precision neither service nor re-transcription has.
OCCURRENCE_TIME_TIE = 0.05

#: How many words each side are kept for re-matching. Enough to tell two
#: mentions of the same surname in one recording apart, few enough that a
#: re-transcription changing a neighbouring word does not wipe the context
#: out.
CONTEXT_WORDS = 4

#: The language value a word carries when the language was never established.
_UNKNOWN_LANGUAGE = Language.UNKNOWN.value


# -- Finding the words that want a person --------------------------------


def occurrences_in(
    transcript: Transcript,
    recording_name: str,
    minimum_confidence: float,
) -> list[Occurrence]:
    """Every word of one transcript weak enough to want a person.

    A word qualifies when its ``confidence_strength`` is not ``None`` and
    falls below the threshold. A word with no saved strength is left out
    entirely: it cannot be thresholded honestly, and guessing a value for it
    would put words in front of a person for a reason nobody could explain.
    The module docstring says what happens to those words instead, and why
    the gap is not a gap.

    The sweep looks at every word, not only the words already flagged for
    review. That is decision 5 of the specification: a word can be quietly
    weak without any rule having noticed it, and those are exactly the words a
    person never finds by reading a queue.
    """
    contexts = _contexts_of(transcript)
    found: list[Occurrence] = []
    for token in transcript.tokens:
        if is_punctuation_only(token.text):
            continue
        strength = token.confidence_strength
        if strength is None or strength >= minimum_confidence:
            continue
        found.append(_occurrence_for(token, recording_name, contexts))
    return found


def _occurrence_for(
    token: FinalToken,
    recording_name: str,
    contexts: dict[str, tuple[str, str]],
) -> Occurrence:
    """Make this layer's record of one word of one transcript.

    ``detected_text`` is taken from ``original_text`` where the token has one,
    because that field holds what the services actually produced before
    anybody changed it, and decision 7 rewrites an occurrence from what it
    originally said rather than from what it currently says.
    """
    detected = token.original_text if token.original_text is not None else token.text
    before, after = contexts.get(token.id, ("", ""))
    return Occurrence(
        id=_new_id(),
        recording_name=recording_name,
        token_id=token.id,
        detected_text=detected,
        normalised_text=_comparison_form(detected),
        start=token.start,
        end=token.end,
        confidence_strength=token.confidence_strength,
        language=token.language.value,
        context_before=before,
        context_after=after,
    )


def _comparison_form(text: str) -> str:
    """The normalised form of a piece of text, worked out rather than read.

    ``FinalToken.normalised_text`` is deliberately **not** used, anywhere in
    this module, even though it now holds the right answer.

    It did not always. Until this feature was built,
    :meth:`~vox_verbatim.transcription.model.Transcript.with_correction`
    left the field describing the word as it *was*, so on a corrected word it
    named the very spelling the project had just answered, and matching a rule
    against it would have rewritten that word again on every reprocess. That
    is fixed, and there is a test holding it.

    Working the form out from the live text is kept regardless, because it
    cannot go stale whatever any other module does later, and because it costs
    a cached string lookup. A field that has to be maintained in step by
    somebody else is a promise; a value derived from the text in front of you
    is a fact.
    """
    return normalise(text)


def _contexts_of(transcript: Transcript) -> dict[str, tuple[str, str]]:
    """The few words each side of every word, ready for re-matching.

    Built once for the whole transcript rather than walked afresh for each
    word, because the sweep asks for it on every weak word and re-matching
    asks for it on every candidate.
    """
    spoken = [token for token in transcript.tokens if not is_punctuation_only(token.text)]
    contexts: dict[str, tuple[str, str]] = {}
    for position, token in enumerate(spoken):
        before = spoken[max(0, position - CONTEXT_WORDS):position]
        after = spoken[position + 1:position + 1 + CONTEXT_WORDS]
        contexts[token.id] = (
            " ".join(item.text for item in before),
            " ".join(item.text for item in after),
        )
    return contexts


def _new_id() -> str:
    return uuid.uuid4().hex


def _now() -> str:
    """When the analysis ran, as local time carrying its own offset.

    The offset is included because a project file travels with the folder it
    sits in, so the machine that reads "processed at half past two" is often
    not the machine that wrote it and may not be in the same country.
    """
    return datetime.now().astimezone().isoformat(timespec="seconds")


# -- Gathering them into groups ------------------------------------------


@dataclass(eq=False)
class _Cluster:
    """Occurrences being gathered, and the facts that decide what may join.

    A cluster keeps its own range of confidences rather than one figure,
    because the tolerance is a question about how far apart two words are and
    the answer has to be measured against the nearest member rather than
    against an average that no member actually has.

    Clusters are compared by identity rather than by content, so that the
    index in :class:`_Relations` can hold them in sets while their contents
    change underneath it.
    """

    members: list[Occurrence] = field(default_factory=list)
    forms: set[str] = field(default_factory=set)
    """The distinct normalised forms in this cluster."""

    detected_forms: set[str] = field(default_factory=set)
    """The distinct forms as the services spelled them.

    Kept as well as the normalised forms because
    :func:`~vox_verbatim.transcription.normalise.are_equivalent` has to
    see the original spelling to recognise a dropped umlaut. By the time a
    text has been normalised, "Jürgen" has already become ``juergen`` and the
    German rule has nothing left to notice.
    """

    language: str = _UNKNOWN_LANGUAGE
    """The one known language in this cluster, or unknown where there is none."""

    lowest: float | None = None
    highest: float | None = None

    position: int = 0
    """Where this cluster stood in the order the clusters were made.

    Survivors keep their relative order as clusters are absorbed, so this
    stays the cluster's rank for the rest of the run.
    """

    alive: bool = True
    """False once this cluster has been absorbed into an earlier one."""

    def absorb(self, occurrence: Occurrence) -> None:
        self.members.append(occurrence)
        self.forms.add(occurrence.normalised_text)
        self.detected_forms.add(occurrence.detected_text)
        if self.language == _UNKNOWN_LANGUAGE:
            self.language = occurrence.language
        strength = occurrence.confidence_strength
        if strength is None:
            return
        self.lowest = strength if self.lowest is None else min(self.lowest, strength)
        self.highest = strength if self.highest is None else max(self.highest, strength)


def group_occurrences(
    occurrences: list[Occurrence],
    tolerance: float,
) -> list[WordGroup]:
    """Gather occurrences that probably mean the same intended word.

    The work is done in the order the contract lays down, and each step is
    there for a reason the step before it cannot cover.

    **First, an exact match on the normalised form.** ``normalise`` already
    folds case, punctuation, apostrophes, German spelling habits, number
    formats and compound word boundaries, so this catches far more than it
    sounds like it does: ``Bosch``, ``bosch,`` and ``BOSCH`` are one form
    here, and so are ``twenty-five`` and ``25``. Everything that meets at this
    step is the same word beyond argument, which is what earns it the
    exemption described below.

    **Second, :func:`~vox_verbatim.transcription.normalise.are_equivalent`.**
    This is the application's existing answer to "the same word once spelling
    habits are set aside", and it is used rather than reimplemented so that
    grouping and reconciliation cannot drift into disagreeing about what
    counts as the same word. It catches the one thing normalising cannot,
    which is a dropped umlaut: ``Jürgen`` and ``Jurgen``.

    **Third, similarity of the normalised form.** Neither of the first two
    steps has anything to say about ``Bosch``, ``Bosh`` and ``Bosche``, which
    is the case the whole feature was asked for. A plain edit-distance ratio
    from the standard library does say something, and
    :data:`SIMILARITY_THRESHOLD` records where the line is drawn and what it
    costs on each side.

    Two occurrences are kept apart, whatever the text says, when their
    languages are both known and different. An unknown language is missing
    information rather than a difference, so it never keeps anything apart --
    but it is not allowed to act as a bridge either, or an English word and a
    German word would end up in one group by way of a word whose language
    nobody established.

    They are also kept apart when their confidences differ by more than the
    tolerance, **except where the text is an exact normalised match**. Two
    spellings of the same surname at 42 and 61 percent are still the same
    surname, and splitting them would make the person decide it twice for no
    reason anybody could explain. That exemption needs no special case in the
    code below: an exact match is settled in the first step, inside a single
    cluster, and the tolerance is only ever asked about afterwards, when two
    *different* forms are considered for joining.

    The tolerance is a boundary between neighbours rather than a diameter. A
    chain of words each within the tolerance of the next can end up in one
    group spanning more than the tolerance from end to end, and that is
    deliberate: the confidences of two spellings of one word are a smear
    rather than a figure, and a group is a suggestion a person can undo.
    """
    clusters = _initial_clusters(occurrences)
    _join_clusters(clusters, tolerance)
    return [
        WordGroup(
            id=_new_id(),
            representative_text=_representative_text(cluster.members),
            occurrence_ids=[member.id for member in cluster.members],
        )
        for cluster in clusters
    ]


def _initial_clusters(occurrences: list[Occurrence]) -> list[_Cluster]:
    """One cluster per normalised form, split only where languages conflict.

    This is step one, and it is also where the confidence exemption lives. An
    occurrence joins the first existing cluster that holds its exact form and
    whose language it does not contradict, whatever its confidence says, so
    every group of identical words is settled before any tolerance is
    consulted. First rather than best-fitting because first is the only rule
    that gives the same answer every time, and a grouping that changed between
    two runs over the same folder would move the person's work about
    underneath them.
    """
    clusters: list[_Cluster] = []
    by_form: dict[str, list[_Cluster]] = {}
    for occurrence in occurrences:
        candidates = by_form.setdefault(occurrence.normalised_text, [])
        home = next(
            (
                cluster
                for cluster in candidates
                if _languages_agree(cluster.language, occurrence.language)
            ),
            None,
        )
        if home is None:
            home = _Cluster()
            candidates.append(home)
            clusters.append(home)
        home.absorb(occurrence)
    return clusters


def _join_clusters(clusters: list[_Cluster], tolerance: float) -> None:
    """Merge clusters that may join, until nothing else can, in place.

    The result is defined by a simple rule: find the earliest pair of
    clusters that may join, merge the later one into the earlier one, and
    start again from the front, until no pair is left. "Earliest" is by the
    position of the first cluster and then of the second, so the outcome
    depends on the order the occurrences arrived in and on nothing else.

    That rule is not the same as "merge everything that is related to
    anything", and the difference matters, which is why this is not a
    union-find over a relation worked out once:

    * A cluster carries one known language, taken from the first member that
      had one. A cluster of unknown language that reaches both an English
      cluster and a German one joins whichever the rule reaches first and is
      then English or German, and closed to the other. Joining everything
      related would put the two known languages in one group by way of the
      word nobody could place.
    * A cluster's confidence range widens as it grows, and the tolerance is
      measured against that range. A form within tolerance of nothing on its
      own can fall inside the range of a cluster that has grown around it.
      The merge order decides what has grown by the time each pair is
      looked at.

    So the rule is kept exactly, and what changed is how much work finding
    the earliest pair costs. The obvious loop restarts from the front after
    every merge and tests every pair again, which is cubic in the number of
    clusters and, with a string similarity at the bottom of each test, took
    over a minute and a half for two thousand words. Three facts make most
    of that work unnecessary:

    * Whether two forms answer to each other never changes, so the
      relation between forms is worked out once, up front, by
      :class:`_Relations`, and a cluster's candidates are read off it
      rather than found by trying every other cluster in turn.
    * After a merge, only the cluster that grew has changed. Every pair that
      was rejected before and does not involve it is still rejected. So the
      next earliest pair either involves the grown cluster, or sits beyond
      the row the scan had reached. Each merge is therefore followed by
      settling the grown cluster -- trying it against everything before it
      (merging upward and trying again from the front, as the rule
      requires), then against everything after it -- and the scan resumes
      where it was, skipping the rows already known to be clean.
    * Survivors keep their relative order, so a cluster's position in the
      original list serves as its position for the rest of the run, and
      nothing has to be renumbered when a cluster is absorbed.
    """
    relations = _Relations(clusters)
    for row in clusters:
        if not row.alive:
            continue
        partner = relations.first_partner_after(row, tolerance)
        if partner is None:
            continue
        relations.merge(row, partner)
        relations.settle(row, tolerance)
        # The loop carries on with the cluster after ``row`` in the original
        # order. Every live cluster before it, ``row`` included if it is
        # still alive, is now known to have no partner anywhere.
    clusters[:] = [cluster for cluster in clusters if cluster.alive]


class _Relations:
    """Which clusters could join which, kept current as clusters merge.

    Built once from the forms every cluster starts with. The relation
    between two forms never changes, so it is worked out here in full and
    the clustering then only asks which clusters currently hold the forms
    related to a given cluster's forms, which is a few dictionary lookups
    rather than a walk over every cluster with a string comparison at each
    step.

    Two relations are kept, one for each way two texts can answer to each
    other:

    * ``equivalent``, over the forms as the services spelled them, answers
      to :func:`~vox_verbatim.transcription.normalise.are_equivalent`.
      That function says yes when the normalised forms agree, or when one
      text has a German letter and the forms with the umlaut dropped agree,
      so the detected forms are bucketed by those two keys and the buckets
      give the answer without calling it once per pair.
    * ``similar`` and its reverse, over the normalised forms, answer to
      :func:`_similar`, which is a string similarity ratio. Measuring every
      pair of forms would be most of the old cost all over again, so pairs
      are first ruled out by two cheap upper bounds on the ratio -- the two
      lengths alone, and the number of characters the forms share in any
      order -- and only the pairs that survive both are measured. The ratio
      is not promised to be the same with the strings swapped, so each pair
      is measured in both orders and the two directions are kept apart.
    """

    def __init__(self, clusters: list[_Cluster]) -> None:
        for position, cluster in enumerate(clusters):
            cluster.position = position
        self._holding_form: dict[str, set[_Cluster]] = {}
        self._holding_detected: dict[str, set[_Cluster]] = {}
        for cluster in clusters:
            for form in cluster.forms:
                self._holding_form.setdefault(form, set()).add(cluster)
            for detected in cluster.detected_forms:
                self._holding_detected.setdefault(detected, set()).add(cluster)
        self._equivalent = _equivalence_partners(list(self._holding_detected))
        self._similar, self._similar_reverse = _similarity_partners(list(self._holding_form))

    def merge(self, keep: _Cluster, absorbed: _Cluster) -> None:
        """Fold ``absorbed`` into ``keep`` and mark it gone."""
        for member in absorbed.members:
            keep.absorb(member)
        for form in absorbed.forms:
            holders = self._holding_form[form]
            holders.discard(absorbed)
            holders.add(keep)
        for detected in absorbed.detected_forms:
            holders = self._holding_detected[detected]
            holders.discard(absorbed)
            holders.add(keep)
        absorbed.alive = False

    def settle(self, grown: _Cluster, tolerance: float) -> None:
        """Carry out every merge the rule demands because ``grown`` grew.

        Repeats until the grown cluster has no partner before it and none
        after it, which is exactly when the front-to-back rule would have
        moved past it. A partner before it takes the grown cluster in, and
        the search starts from the front again because that earlier cluster
        has now grown.
        """
        while True:
            earlier = self.first_partner_before(grown, tolerance)
            if earlier is not None:
                self.merge(earlier, grown)
                grown = earlier
                continue
            later = self.first_partner_after(grown, tolerance)
            if later is None:
                return
            self.merge(grown, later)

    def first_partner_before(self, cluster: _Cluster, tolerance: float) -> _Cluster | None:
        """The earliest live cluster before this one that it may join."""
        best: _Cluster | None = None
        for candidate in self._candidates(cluster, self._similar_reverse):
            if candidate.position >= cluster.position:
                continue
            if best is not None and candidate.position >= best.position:
                continue
            if _may_join(candidate, cluster, tolerance):
                best = candidate
        return best

    def first_partner_after(self, cluster: _Cluster, tolerance: float) -> _Cluster | None:
        """The earliest live cluster after this one that it may join."""
        best: _Cluster | None = None
        for candidate in self._candidates(cluster, self._similar):
            if candidate.position <= cluster.position:
                continue
            if best is not None and candidate.position >= best.position:
                continue
            if _may_join(cluster, candidate, tolerance):
                best = candidate
        return best

    def _candidates(
        self,
        cluster: _Cluster,
        similar: dict[str, set[str]],
    ) -> set[_Cluster]:
        """Every live cluster holding a form related to one of this cluster's.

        ``similar`` is the direction of :func:`_similar` that matches which
        side of the pair this cluster is on: an earlier cluster's forms are
        always the first argument.
        """
        found: set[_Cluster] = set()
        for detected in cluster.detected_forms:
            for partner in self._equivalent.get(detected, ()):
                found.update(self._holding_detected[partner])
        for form in cluster.forms:
            for partner in similar.get(form, ()):
                found.update(self._holding_form[partner])
        found.discard(cluster)
        return found


def _equivalence_partners(detected_forms: list[str]) -> dict[str, set[str]]:
    """For each detected form, the others :func:`are_equivalent` accepts.

    Worked out through the same two keys that function compares, so that
    the answer is the one it would give, pair by pair, without asking it
    pair by pair. The German direction is one-sided by design: a form with
    no German letter in it is only reached through the dropped-umlaut key
    by a form that has one.
    """
    by_normalised: dict[str, list[str]] = {}
    by_dropped: dict[str, list[str]] = {}
    german: set[str] = set()
    for detected in detected_forms:
        by_normalised.setdefault(normalise(detected), []).append(detected)
        by_dropped.setdefault(_compound_form(detected, True), []).append(detected)
        if _has_german_letter(detected):
            german.add(detected)
    partners: dict[str, set[str]] = {}
    for detected in detected_forms:
        found = set(by_normalised[normalise(detected)])
        dropped = by_dropped[_compound_form(detected, True)]
        if detected in german:
            found.update(dropped)
        else:
            found.update(other for other in dropped if other in german)
        found.discard(detected)
        if found:
            partners[detected] = found
    return partners


def _similarity_partners(
    forms: list[str],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """For each normalised form, the others :func:`_similar` accepts.

    Returns the relation in both directions: ``forward[a]`` holds every
    ``b`` with ``_similar(a, b)``, and ``reverse[b]`` holds every ``a`` for
    the same pairs.

    The ratio :func:`_similar` measures is twice the matched characters
    over the two lengths, and the matched characters can be no more than
    the shorter length and no more than the characters the two forms have
    in common counted without regard to order. Both give an upper bound on
    the ratio, both are cheap, and a pair that fails either cannot pass the
    threshold, so only the pairs that pass both are measured for real.
    """
    forward: dict[str, set[str]] = {}
    reverse: dict[str, set[str]] = {}
    # A form with a digit is never similar to anything, so it is left out
    # of the measuring altogether rather than measured and turned down.
    eligible = [
        form
        for form in forms
        if len(form) >= SIMILARITY_MINIMUM_LENGTH and not _has_digit(form)
    ]
    by_length: dict[int, list[str]] = {}
    for form in eligible:
        by_length.setdefault(len(form), []).append(form)
    counts = {form: Counter(form) for form in eligible}
    lengths = sorted(by_length)
    for shorter_length in lengths:
        # The longer form may be at most this long before the length bound
        # alone rules the pair out: 2 * shorter / (shorter + longer) must
        # reach the threshold.
        longest = int(shorter_length * (2 - SIMILARITY_THRESHOLD) / SIMILARITY_THRESHOLD)
        for longer_length in lengths:
            if longer_length < shorter_length or longer_length > longest:
                continue
            shorter_forms = by_length[shorter_length]
            longer_forms = by_length[longer_length]
            total = shorter_length + longer_length
            for index, shorter in enumerate(shorter_forms):
                shorter_counts = counts[shorter]
                # Two forms of the same length meet once, as shorter and
                # longer, and both directions are measured then.
                if longer_length == shorter_length:
                    others = longer_forms[index + 1 :]
                else:
                    others = longer_forms
                for longer in others:
                    longer_counts = counts[longer]
                    shared = sum(
                        min(count, longer_counts.get(character, 0))
                        for character, count in shorter_counts.items()
                    )
                    if 2 * shared / total < SIMILARITY_THRESHOLD:
                        continue
                    if _similar(shorter, longer):
                        forward.setdefault(shorter, set()).add(longer)
                        reverse.setdefault(longer, set()).add(shorter)
                    if _similar(longer, shorter):
                        forward.setdefault(longer, set()).add(shorter)
                        reverse.setdefault(shorter, set()).add(longer)
    return forward, reverse


def _may_join(first: _Cluster, second: _Cluster, tolerance: float) -> bool:
    """Whether two clusters whose texts are related may become one.

    Only the language and the confidence gap are asked here. Whether the
    texts are related at all is settled once, up front, by
    :class:`_Relations`, which only ever offers a pair whose texts are; the
    two tests left are the ones whose answer changes as clusters grow.
    """
    if not _languages_agree(first.language, second.language):
        return False
    return _confidence_gap(first, second) <= tolerance


def _languages_agree(first: str, second: str) -> bool:
    """Whether two languages are compatible, treating unknown as no obstacle.

    Because a cluster carries the one known language of everything in it, an
    unknown-language word joins whichever cluster reaches it first and then
    stops being a bridge. That is what prevents an English cluster and a
    German cluster merging through a word nobody could place.
    """
    if first == _UNKNOWN_LANGUAGE or second == _UNKNOWN_LANGUAGE:
        return True
    return first == second


def _confidence_gap(first: _Cluster, second: _Cluster) -> float:
    """How far apart the nearest members of two clusters are.

    Nought where the ranges overlap, and nought as well where either cluster
    has no measured strength at all. An unmeasured strength is not a distance
    of nought, but the tolerance exists to keep genuinely different-looking
    words apart on the evidence of their confidences, and there is no such
    evidence here. Splitting a group on a number nobody has is the same
    mistake as thresholding on one.
    """
    if first.lowest is None or first.highest is None:
        return 0.0
    if second.lowest is None or second.highest is None:
        return 0.0
    return max(0.0, max(first.lowest, second.lowest) - min(first.highest, second.highest))


def _similar(first: str, second: str) -> bool:
    """Whether two normalised forms are close enough to suggest one word.

    This is the measure :class:`_Relations` builds its similarity index
    from. It is only ever asked about a pair in one particular order -- the
    earlier cluster's form first -- because the matcher's answer is not
    guaranteed to be the same with the two strings swapped.

    A form with a digit in it is never similar to anything. For a number, a
    near spelling is a different value: "150000" and "160000" score well
    above the threshold, and settling them as one group would rewrite one
    amount as another in every file. Number words are already digits by the
    time a form reaches here, so "one hundred and fifty thousand" is covered
    too. The same number written the same way still groups, because an exact
    match and an equivalence never come through this measure.
    """
    if not first or not second:
        return False
    if _has_digit(first) or _has_digit(second):
        return False
    if min(len(first), len(second)) < SIMILARITY_MINIMUM_LENGTH:
        return False
    return SequenceMatcher(None, first, second).ratio() >= SIMILARITY_THRESHOLD


def _has_digit(form: str) -> bool:
    """Whether a normalised form contains a digit, and so names a number."""
    return any(character.isdigit() for character in form)


def _representative_text(members: list[Occurrence]) -> str:
    """The form to show for a group: the commonest, weakest where it is a tie.

    Commonest because it is the spelling the person will recognise from the
    recording. Weakest to break a tie because where two spellings occur the
    same number of times, the one the application is least sure of is the one
    most likely to be wrong, and showing it is what makes the group's reason
    for existing visible at a glance.

    A form with no measured strength never wins a tie. "Nobody measured this"
    is not "this is weak", and letting an unmeasured form take the tie-break
    would decide the group's name on an absence.
    """
    if not members:
        return ""
    counts = Counter(member.detected_text for member in members)
    order = {}
    weakest: dict[str, float] = {}
    for position, member in enumerate(members):
        order.setdefault(member.detected_text, position)
        strength = member.confidence_strength
        if strength is None:
            continue
        current = weakest.get(member.detected_text)
        if current is None or strength < current:
            weakest[member.detected_text] = strength

    def rank(text: str) -> tuple[int, int, float, int]:
        strength = weakest.get(text)
        return (
            -counts[text],
            1 if strength is None else 0,
            0.0 if strength is None else strength,
            order[text],
        )

    return min(counts, key=rank)


# -- Keeping a decision across a re-transcription -------------------------


def rematch(
    saved: list[Occurrence],
    transcript: Transcript,
    recording_name: str,
) -> tuple[list[Occurrence], list[Occurrence]]:
    """Re-attach saved occurrences to a transcript that has been redone.

    Returns the ones re-attached and the ones that could not be. Token
    identifiers are fresh uuids on every run, so they are worthless across a
    re-transcription. Matching is on the recording, the normalised text, a
    start time within :data:`OCCURRENCE_TIME_TOLERANCE`, and the surrounding
    words where two candidates are otherwise equally good.

    An occurrence that cannot be matched confidently is returned marked
    ``stale`` rather than being attached to the nearest thing available.
    Applying somebody's correction to a word they never looked at is worse in
    every way than telling them a decision could not be kept, and it is worse
    because it is invisible: the transcript reads perfectly well afterwards
    and nothing in it says that the word was chosen by a near miss.

    There is one exception to "token identifiers are worthless", and it is not
    a contradiction. Where the transcript still contains the very token an
    occurrence points at, the transcript has not been redone at all -- this is
    an ordinary reprocessing run over the same file -- and the identifier is
    then conclusive rather than useless. Taking that path first is also what
    stops a reprocessing run marking every already-corrected occurrence stale,
    which is what would happen if a word whose text has since been replaced
    had to be found again by its old spelling.

    An occurrence naming another recording is handed straight back in the
    second list, without the stale flag. This transcript has nothing to say
    about it either way, and saying "stale" would be a claim rather than a
    silence.
    """
    contexts = _contexts_of(transcript)
    by_form: dict[str, list[FinalToken]] = {}
    by_id: dict[str, FinalToken] = {}
    for token in transcript.tokens:
        if is_punctuation_only(token.text):
            continue
        by_id[token.id] = token
        # Indexed under what the token *originally* said, because that is the
        # spelling a saved occurrence remembers. A token already corrected by
        # a person or by a rule would otherwise be unfindable by the very
        # occurrence that records the correction.
        detected = token.original_text if token.original_text is not None else token.text
        by_form.setdefault(_comparison_form(detected), []).append(token)

    matched: list[Occurrence] = []
    missing: list[Occurrence] = []
    claimed: set[str] = set()
    for occurrence in saved:
        if occurrence.recording_name != recording_name:
            missing.append(occurrence)
            continue
        token = by_id.get(occurrence.token_id)
        if token is None or token.id in claimed:
            form = occurrence.normalised_text or _comparison_form(occurrence.detected_text)
            token = _best_candidate(occurrence, by_form.get(form, []), claimed, contexts)
        if token is None:
            missing.append(replace(occurrence, stale=True))
            continue
        claimed.add(token.id)
        matched.append(_reattached(occurrence, token, contexts))
    return matched, missing


def _best_candidate(
    occurrence: Occurrence,
    candidates: list[FinalToken],
    claimed: set[str],
    contexts: dict[str, tuple[str, str]],
) -> FinalToken | None:
    """The one word this occurrence certainly belongs to, or nothing at all.

    Every step here can refuse. That is the point of the function: it returns
    a word only when there is one word it could be, and hands back ``None`` at
    the first sign of a choice it cannot justify.
    """
    available = [token for token in candidates if token.id not in claimed]
    if not available:
        return None

    if occurrence.start is None:
        # A saved occurrence with no start time can still be re-matched, but
        # only on its text and its context, so the bar is the same and the
        # evidence is thinner. A word with no timing is the ordinary case for
        # the services that do not time their words at all.
        near = available
    else:
        distances = {
            token.id: abs(token.start - occurrence.start)
            for token in available
            if token.start is not None
        }
        distances = {
            token_id: gap
            for token_id, gap in distances.items()
            if gap <= OCCURRENCE_TIME_TOLERANCE
        }
        if not distances:
            return None
        closest = min(distances.values())
        near = [
            token
            for token in available
            if token.id in distances and distances[token.id] <= closest + OCCURRENCE_TIME_TIE
        ]
    if len(near) == 1:
        return near[0]

    # Two or more words of the same text at effectively the same moment. The
    # words around them are the only thing left that can tell them apart, and
    # if they cannot, nothing can: a tie is answered with a refusal rather
    # than with the first of them.
    scored = [(_context_score(occurrence, contexts.get(token.id, ("", ""))), token)
              for token in near]
    best_score = max(score for score, _token in scored)
    if best_score <= 0:
        return None
    winners = [token for score, token in scored if score == best_score]
    if len(winners) != 1:
        return None
    return winners[0]


def _context_score(occurrence: Occurrence, context: tuple[str, str]) -> int:
    """How many of the surrounding words still agree, counted outwards.

    Words are compared in their normalised form, so a re-transcription that
    changed only the punctuation or the capitals of a neighbour does not read
    as a different sentence. Counting outwards from the word itself matters
    because a re-transcription that inserted or dropped a word further away
    should not throw the nearer agreement away.
    """
    before, after = context
    return _agreeing(occurrence.context_before.split()[::-1], before.split()[::-1]) + _agreeing(
        occurrence.context_after.split(), after.split()
    )


def _agreeing(first: list[str], second: list[str]) -> int:
    count = 0
    for left, right in zip(first, second):
        if _comparison_form(left) != _comparison_form(right):
            break
        count += 1
    return count


def _reattached(
    occurrence: Occurrence,
    token: FinalToken,
    contexts: dict[str, tuple[str, str]],
) -> Occurrence:
    """The saved occurrence, now pointing at this word of this transcript.

    Everything the person decided survives untouched: the reviewed mark, the
    replacement, the correct-as-detected decision and the isolation. What is
    refreshed is everything that describes where the word sits and what it
    says, because those are facts about the transcript rather than decisions
    about the word.

    The strength is the one exception among the facts. Where the token's text
    has already been replaced, its strength is the 1.0 that a correction
    confers and says nothing about the word the person originally saw, so the
    saved measurement is kept instead. Overwriting it would make an occurrence
    the project itself corrected look like a word the services were sure of.

    The test is ``text_corrected`` and deliberately not ``human_corrected``.
    The reasoning above is entirely about the text having been replaced, and
    ``human_corrected`` is also set by a decision about the timing, which
    changes no text and confers no 1.0. Asking the broad question froze such a
    word at whatever strength it had when it was first found, so a
    re-transcription that made the word markedly better or worse was never
    reflected in the list the person sorts by weakness.
    """
    before, after = contexts.get(token.id, ("", ""))
    detected = token.original_text if token.original_text is not None else token.text
    strength = occurrence.confidence_strength if token.text_corrected else token.confidence_strength
    return replace(
        occurrence,
        token_id=token.id,
        detected_text=detected,
        normalised_text=_comparison_form(detected),
        start=token.start,
        end=token.end,
        confidence_strength=strength,
        language=token.language.value,
        context_before=before,
        context_after=after,
        stale=False,
    )


# -- Answering a new file with what the project already knows -------------


def apply_rules(
    state: ProjectState,
    transcript: Transcript,
    recording_name: str,
) -> tuple[Transcript, list[Occurrence], list[Occurrence]]:
    """Answer a newly transcribed file with what the project already knows.

    Returns the transcript with the automatic corrections made, the
    occurrences a rule answered, and the weak occurrences no rule could
    answer. The second list leaves the review queue at once; the third joins
    it.

    Rule matching here is strict, and deliberately stricter than grouping. A
    rule fires only on an exact match of the normalised form, with the
    language agreeing where both are known, which is exactly what
    :meth:`~vox_verbatim.transcription.project.ProjectState.rule_for`
    does and the reason this asks that method rather than deciding for itself.
    Grouping is allowed to be generous because its output is a suggestion a
    person then judges; this is allowed no generosity at all, because its
    output is a word silently rewritten in a file nobody has opened. A form
    nobody has ever accepted a replacement for falls into the queue instead,
    grouped beside the forms that are already settled, which is exactly where
    a person can decide about it in one action.

    An occurrence answered this way is marked ``auto_applied`` and
    ``reviewed`` and records which rule answered it, and the word it rewrote
    records what it originally said, so the correction can be explained and
    undone like any other.

    A rule fires on a strong word as readily as on a weak one, and that is on
    purpose. The words a service gets confidently wrong are proper names, and
    a rule that only touched the words already in doubt would miss precisely
    the case decision 10 was written for. What makes that safe is that nothing
    is lost: the original text is kept on the token, the occurrence names the
    rule, and the rule belongs to this folder alone.

    Two things are never touched. A word whose text a person has already
    replaced in this very file is left exactly as they left it, because a rule
    is an older and weaker statement than somebody typing into this
    transcript. And a word that already reads as the replacement is left alone
    as well, because there is nothing to correct and recording an automatic
    correction that changed nothing would put a phantom in the person's list
    of decisions.

    "Already replaced" is read from ``text_corrected`` and emphatically not
    from ``human_corrected``, and this is the sharpest edge in the module.
    ``human_corrected`` is also set when somebody confirms a word's *timing*,
    which is a statement about the clock and says nothing at all about the
    spelling. Reading the broad flag here made confirming a clock exempt the
    word from every replacement rule the project would ever learn, for the
    rest of the project's life -- and the word did not fall into the queue
    either, because the same test skipped it before the threshold was ever
    consulted, so it reached the person by neither road. That is precisely the
    case decision 10 exists for: a proper name somebody has already settled
    elsewhere, silently left saying the wrong thing.

    The one thing this changes in the given state is the ``occurrence_count``
    of the rules that fired, which is counted here because here is where the
    correction is actually made. It is counted through
    :func:`note_rule_applied` rather than by hand, because the window writes a
    rule's correction too and the tally must not depend on which of the two
    did the work.
    """
    contexts = _contexts_of(transcript)
    known_forms = {rule.normalised_text for rule in state.rules}
    minimum = state.settings.minimum_confidence

    corrected = transcript
    answered: list[Occurrence] = []
    queued: list[Occurrence] = []
    for token in transcript.tokens:
        if is_punctuation_only(token.text) or token.text_corrected:
            continue
        form = _comparison_form(token.text)
        rule = _matching_rule(state, known_forms, form, token.language.value)
        if rule is not None and rule.replacement != token.text:
            occurrence = _occurrence_for(token, recording_name, contexts)
            occurrence.replacement = rule.replacement
            occurrence.reviewed = True
            occurrence.auto_applied = True
            occurrence.applied_rule_id = rule.id
            note_rule_applied(state, rule.id)
            corrected = corrected.with_correction(token.id, text=rule.replacement)
            answered.append(occurrence)
            continue
        strength = token.confidence_strength
        if strength is not None and strength < minimum:
            queued.append(_occurrence_for(token, recording_name, contexts))
    return corrected, answered, queued


def note_rule_applied(state: ProjectState, rule_id: str) -> None:
    """Record that a rule's correction was written into a transcript.

    There are two places in the application where a rule's replacement is
    actually put into a file: :func:`apply_rules` here, which answers a newly
    transcribed recording, and the Review window, which writes the corrections
    the analysis only wrote down. Only one of them was keeping the tally, so
    whether a rule's ``occurrence_count`` was right depended on which of the
    two paths a particular word happened to travel by -- and the person is
    shown that number as a statement of how much work the rule has saved them.
    A count that is right by accident is worse than no count, because nothing
    about it looks wrong.

    Both callers use this and neither counts by hand. Being asked about a rule
    the project no longer holds is not an error and is deliberately silent: a
    person may delete a rule while a transcript that its correction has just
    been written into is still being saved, and refusing at that moment would
    turn a tidy-up into a failure. Nothing is lost either way, because the
    correction itself is in the transcript and the count is only a tally.
    """
    for rule in state.rules:
        if rule.id == rule_id:
            rule.occurrence_count += 1
            return


def _matching_rule(
    state: ProjectState,
    known_forms: set[str],
    form: str,
    language: str,
) -> ReplacementRule | None:
    """The rule for this word, asked of the project rather than worked out here.

    The set of forms is only a way of not walking the whole rule list for
    every word of a long recording. The decision itself stays in one place, so
    that the rule for what counts as a match cannot come to differ between the
    window, the project and this module.
    """
    if form not in known_forms:
        return None
    return state.rule_for(form, language)


# -- Running the whole analysis again -------------------------------------


def reprocess(
    state: ProjectState,
    recording_names: Iterable[str],
    load_transcript: Callable[[str], Transcript | None],
    present_recordings: Iterable[str] | None = None,
) -> ProjectState:
    """Run the whole analysis again without losing manual work.

    Isolated occurrences stay isolated. User-created groups keep their
    members. Reviewed decisions, replacements, correct-as-detected marks and
    project rules all survive. Automatic groups are rebuilt around whatever is
    left. New occurrences pick up any project rule that matches them.

    This is called whenever the person moves the threshold or the tolerance,
    which is to say often, and casually, by somebody exploring what a
    different setting would show them. It therefore has to be safe to run
    fifty times. The rule it keeps is simple to state: **reprocessing may
    change what the analysis suggests and may never change what a person
    decided.**

    **Recordings are named, not handed over, and this is the whole reason for
    the loader.** An earlier version took a ``dict[str, Transcript]`` and was
    measured on a realistic folder: fifty hour-long recordings took 37 seconds
    to read and held about 1.6 GB of live objects. That is what pressing
    Ctrl+R would have cost, every time, and it is not affordable. Reading one
    transcript, taking the occurrences out of it and letting it go before
    reading the next makes the peak one transcript rather than fifty, and the
    project file already holds everything the window's two lists need, so
    opening a folder reads no transcripts at all.

    That cost comes straight back the moment somebody finds the loader awkward
    and writes ``reprocess(state, names, {name: read(name) for name in
    names}.get)``. It looks like a tidy-up, it passes every test in this file,
    and it rebuilds the 1.6 GB. **The loader must read one recording when it
    is asked for it, and not before.** Anything that reads them all up front
    has undone this, however it is spelled.

    An isolated occurrence is left out of every automatic group, not only out
    of the one it was pulled from. Taking a word out of a group was a
    statement about that word, and running the analysis again with a different
    threshold is not new evidence about it; gathering it up again would
    silently overrule a decision the person made deliberately.

    A recording the loader answers with ``None`` is treated as absent for this
    run rather than as damage, and its occurrences are kept exactly as they
    were. That covers a file being written by another program, a disconnected
    network drive and a transcript folder somebody has moved, and all three
    read perfectly well tomorrow. Marking those occurrences stale would tell
    the person their decisions had been lost when nothing of the kind had
    happened, and the same goes for a recording that is simply not in
    ``recording_names`` at all.

    **A recording that has left the folder is the one case that rule gets
    wrong, and ``present_recordings`` is how it is told apart.** "Could not be
    read just now" and "deleted last week" look identical from inside this
    function, because the loader answers with nothing in both cases, so the
    evidence has to come from outside it: ``present_recordings`` names every
    recording the folder still holds a transcript for. An occurrence naming a
    recording that is not in that list describes a word in a file that is no
    longer there, and it is dropped, along with the flagged words of the same
    recording.

    Dropping somebody's work is not a small thing to do, so three guards sit
    around it. It happens only when the caller actually supplies a listing:
    ``None`` means "I do not know what is in the folder", which is the honest
    answer from a caller that could not enumerate it, and nothing is thrown
    away on the strength of a question that was never asked. It is decided on
    the file existing rather than on the file parsing, so a damaged transcript
    is never mistaken for a deleted one. And the rules are untouched.

    That last point is what makes dropping the right answer rather than merely
    the tidy one. An occurrence is a pointer at a word in a file: with the
    file gone it can no longer be played, corrected or confirmed, so keeping
    it leaves a row that does nothing, and worse, it goes on being counted in
    :func:`affected_summary` -- the sentence read out to somebody who cannot
    see the list and cannot check it, saying a replacement reaches eight
    occurrences across two files when one of the two no longer exists. The
    durable half of the person's work is not in the occurrence at all. It is
    in the rules, which belong to the folder rather than to any recording,
    which answer the next file transcribed, and which mean that a recording
    restored from a backup has its words found again and answered
    automatically. So what is lost is a pointer, and what is kept is the
    decision.

    This never edits a transcript. Where a project rule answers a newly found
    word, the decision is recorded on the occurrence and the caller writes it
    into the file, which is the same path an ordinary group replacement takes.
    :func:`apply_rules` is the function that both decides and rewrites, and it
    is the one that counts the rule as having fired.

    The flagged words are carried across unchanged, apart from that pruning.
    They are a property of the transcripts rather than of the analysis, and
    rebuilding the state without them handed the caller an empty second block
    instead of an error -- a failure whose only symptom is half a review
    quietly missing.
    """
    settings = state.settings
    present = None if present_recordings is None else set(present_recordings)
    protected = _protected_ids(state)
    saved_by_recording: dict[str, list[Occurrence]] = {}
    for occurrence in state.occurrences:
        if _has_left_the_folder(occurrence.recording_name, present):
            continue
        saved_by_recording.setdefault(occurrence.recording_name, []).append(occurrence)

    updated: dict[str, Occurrence] = {}
    fresh: list[Occurrence] = []
    for recording_name in _in_order(recording_names):
        transcript = load_transcript(recording_name)
        if transcript is None:
            continue
        _reprocess_recording(
            state,
            transcript,
            recording_name,
            saved_by_recording.get(recording_name, []),
            updated,
            fresh,
        )
        # Let the transcript go before the next one is read. Without this the
        # name still holds the previous recording while the loader builds the
        # next, so the peak would be two rather than one, which on an
        # hour-long recording is tens of megabytes for no reason at all.
        del transcript

    # The project's own order is kept for the words that were already there,
    # rather than the order the recordings happened to be read in, so that
    # reprocessing does not shuffle the list under a person who has just
    # learned where things are.
    kept = [
        updated.get(occurrence.id, occurrence)
        for occurrence in state.occurrences
        if not _has_left_the_folder(occurrence.recording_name, present)
    ]
    surviving = [
        occurrence
        for occurrence in kept
        if _is_manual_work(occurrence, protected)
        or (not occurrence.stale and _is_weak(occurrence, settings.minimum_confidence))
    ]

    occurrences = surviving + fresh
    groups = _rebuild_groups(state, occurrences, settings.grouping_tolerance)
    rebuilt = ProjectState(
        settings=settings,
        groups=groups,
        occurrences=occurrences,
        flagged=[
            item
            for item in state.flagged
            if not _has_left_the_folder(item.recording_name, present)
        ],
        rules=state.rules,
        processed_at=_now(),
        # Carried across untouched, apart from the recordings that have left
        # the folder, exactly as the flagged words are. What each transcript
        # was when it was last analysed is not this function's business -- it
        # is not told which files it was given, only what they said -- but a
        # fresh project built without it would be a project that has forgotten
        # which transcripts it has read, and every recording in the folder
        # would then be read again on the next opening.
        transcript_times={
            name: age
            for name, age in state.transcript_times.items()
            if not _has_left_the_folder(name, present)
        },
    )
    _restore_markers(state, rebuilt)
    return rebuilt


def _has_left_the_folder(recording_name: str, present: set[str] | None) -> bool:
    """Whether this recording is known to be gone, rather than merely unread.

    ``None`` is not an empty folder. It is a caller that did not say, and the
    only safe reading of silence is that everything is still there: an empty
    set would otherwise wipe an entire project the first time somebody called
    this without a listing.
    """
    return present is not None and recording_name not in present


def _in_order(recording_names: Iterable[str]) -> list[str]:
    """The names to consider, each of them once, in the order they arrived.

    A name given twice would otherwise be swept twice and every weak word in
    it found twice, once under each of two identifiers, which is a duplicate
    the person would have to settle twice and could not tell apart.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for recording_name in recording_names:
        if recording_name in seen:
            continue
        seen.add(recording_name)
        ordered.append(recording_name)
    return ordered


def _reprocess_recording(
    state: ProjectState,
    transcript: Transcript,
    recording_name: str,
    saved: list[Occurrence],
    updated: dict[str, Occurrence],
    fresh: list[Occurrence],
) -> None:
    """Take everything wanted from one transcript, so it can then be dropped.

    Nothing this leaves behind refers to the transcript. Occurrences are plain
    data holding strings and numbers, so once this returns, the only thing
    keeping the recording in memory is the caller's own name for it.

    A word that a saved occurrence was just re-matched to must not also be
    found by the sweep, or the same word would appear twice, once carrying the
    person's decision and once looking like a fresh finding. An occurrence
    that could not be re-matched is deliberately not counted as claiming
    anything: it points at nothing now, and the word it used to describe is
    free to be found afresh.
    """
    matched, missing = rematch(saved, transcript, recording_name)
    for occurrence in matched + missing:
        updated[occurrence.id] = occurrence
    claimed = {occurrence.token_id for occurrence in matched}
    for occurrence in occurrences_in(
        transcript, recording_name, state.settings.minimum_confidence
    ):
        if occurrence.token_id in claimed:
            continue
        _answer_with_rule(state, occurrence)
        fresh.append(occurrence)


def _protected_ids(state: ProjectState) -> set[str]:
    """Occurrences that survive because of the group they are in.

    Two kinds of group are the person's work rather than the analysis's. One
    they made by hand, and one the analysis made that they then settled, by
    giving it a replacement, marking it reviewed or saying it was right as
    detected. Rebuilding either would throw that away, so their members are
    kept even when a moved threshold no longer finds the words weak.
    """
    return {
        occurrence_id
        for group in state.groups
        if _group_is_manual(group)
        for occurrence_id in group.occurrence_ids
    }


def _group_is_manual(group: WordGroup) -> bool:
    return (
        group.user_created
        or group.reviewed
        or group.correct_as_detected
        or group.replacement is not None
    )


def _is_manual_work(occurrence: Occurrence, protected: set[str]) -> bool:
    """Whether anything about this occurrence is a person's decision.

    ``auto_applied`` counts, even though nobody looked at this particular
    word, because the rule behind it is a decision a person made and the
    occurrence is the only record that it was applied here.
    """
    return (
        occurrence.reviewed
        or occurrence.correct_as_detected
        or occurrence.replacement is not None
        or occurrence.isolated
        or occurrence.auto_applied
        or occurrence.id in protected
    )


def _is_weak(occurrence: Occurrence, minimum_confidence: float) -> bool:
    strength = occurrence.confidence_strength
    return strength is not None and strength < minimum_confidence


def _answer_with_rule(state: ProjectState, occurrence: Occurrence) -> None:
    """Let a project rule answer a newly found word, where one can.

    The same strict test as :func:`apply_rules`, and for the same reason. What
    is different is only that nothing is rewritten here; the occurrence
    carries the answer and the caller applies it.
    """
    rule = state.rule_for(occurrence.normalised_text, occurrence.language)
    if rule is None:
        return
    occurrence.replacement = rule.replacement
    occurrence.reviewed = True
    occurrence.auto_applied = True
    occurrence.applied_rule_id = rule.id


def _rebuild_groups(
    state: ProjectState,
    occurrences: list[Occurrence],
    tolerance: float,
) -> list[WordGroup]:
    """Keep the groups that are a person's work; rebuild the rest.

    A kept group keeps its identifier, so the marker saying where the person
    had got to still points at it, and keeps every decision made about it. Its
    membership shrinks to the occurrences that are still there, and its
    representative text is worked out again from those, because that text
    describes the group rather than being a decision about it and a group
    whose commonest form has gone should not still be named after it. A group
    left with nothing in it is dropped, as it would be on loading.
    """
    surviving = {occurrence.id for occurrence in occurrences}
    groups: list[WordGroup] = []
    grouped: set[str] = set()
    by_id = {occurrence.id: occurrence for occurrence in occurrences}
    for group in state.groups:
        if not _group_is_manual(group):
            continue
        members = [
            occurrence_id
            for occurrence_id in group.occurrence_ids
            if occurrence_id in surviving and occurrence_id not in grouped
        ]
        if not members:
            continue
        grouped.update(members)
        groups.append(
            replace(
                group,
                occurrence_ids=members,
                representative_text=_representative_text([by_id[item] for item in members]),
            )
        )

    # Isolated words are left out on purpose, and so are stale ones: a stale
    # occurrence describes a word that is no longer in the file, and letting
    # it into a group would put a decision that cannot be applied beside ones
    # that can.
    loose = [
        occurrence
        for occurrence in occurrences
        if occurrence.id not in grouped and not occurrence.isolated and not occurrence.stale
    ]
    groups.extend(group_occurrences(loose, tolerance))
    return groups


def _restore_markers(previous: ProjectState, rebuilt: ProjectState) -> None:
    """Put the person back where they were, as far as that still means anything.

    The occurrence is the thing they were actually looking at, so it leads: if
    it is still in the project, the group marker is set to whichever group now
    holds it, which may well not be the group it was in before. Where the
    occurrence has gone, both markers are cleared rather than one of them
    being kept, because landing somebody in the right group on the wrong word
    is more confusing than starting them at the top.
    """
    occurrence_id = previous.last_occurrence_id
    if occurrence_id is None or rebuilt.occurrence(occurrence_id) is None:
        return
    rebuilt.last_occurrence_id = occurrence_id
    group = rebuilt.group_of(occurrence_id)
    rebuilt.last_group_id = None if group is None else group.id


# -- Saying what a change will do, before it is made ----------------------


def affected_summary(state: ProjectState, group_id: str) -> str:
    """"This replacement applies to 8 occurrences across 2 files."

    Counted rather than estimated, and correct for one occurrence and one
    file. The wording matters more than it looks: this sentence is what a
    person has to go on before they change a word in files they have not
    opened, and it is read out to somebody who cannot see the group. "Applies
    to 8" with the noun left off, or a bare pair of numbers, tells a listener
    almost nothing.

    A group that is gone, or has nothing left in it, is described as applying
    to nothing rather than answered with an empty string. A screen reader says
    nothing at all for an empty string, and silence in the place where a
    consequence should be is indistinguishable from the application having
    failed to answer.
    """
    occurrences = state.occurrences_of(group_id)
    if not occurrences:
        return "This replacement applies to no occurrences."
    files = len({occurrence.recording_name for occurrence in occurrences})
    count = len(occurrences)
    words = "1 occurrence" if count == 1 else f"{count} occurrences"
    # "across 1 file" is not something anybody says, and a sentence that reads
    # as though it were assembled by a machine invites the reader to distrust
    # the number in it.
    where = "in 1 file" if files == 1 else f"across {files} files"
    return f"This replacement applies to {words} {where}."
