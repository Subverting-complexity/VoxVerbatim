"""The pages of the Settings dialog, one per category.

The dialog itself is a shell: a list of categories on the left, a page on
the right, a panel explaining whichever setting has the focus, and two
buttons. Everything that knows what a particular setting *is* lives here,
one class per page, so that adding a service means adding a page and a line
to the list of them rather than editing a thousand-line dialog.

Four things are the same on every page, and they are the same because each
of them is an accessibility rule rather than a matter of taste.

Every input is named, described, and has a real label pointing at it with
``setBuddy``. The name and the description both come from
:mod:`vox_verbatim.ui.settings_notes`, so a control cannot exist
without an explanation, and the explanation cannot be written twice.

Every API key box hides what you type and has a Show box beside it. Hiding
is right and checking is necessary, and a masked box read out as a row of
dots is no way to check that you pasted the right key. The Show box is
named for what it does and says so when it changes.

Every free-form parameter box is validated when OK is pressed, never
silently. A service can add a request parameter between releases of this
application, so each service has a box of JSON that is passed through as it
is written. That freedom is worth having and it means the text can be
wrong, so it is checked, and a mistake stops the dialog closing, says
exactly what is wrong and where, and leaves every character the user typed
exactly where it was.

Every page that describes a service says in words whether that service is
set up. The wording comes from the settings themselves, through
``is_configured`` and each section's ``requirements`` phrase, rather than
from sentences written here that would drift from what the code actually
requires.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from vox_verbatim.settings import (
    MAXIMUM_CHUNK_OVERLAP_SECONDS,
    MAXIMUM_CHUNK_TARGET_BYTES,
    MAXIMUM_COST,
    MAXIMUM_ESCALATION_CONTEXT_SECONDS,
    MAXIMUM_ESCALATIONS_PER_RECORDING,
    MAXIMUM_EXPECTED_SPEAKER_COUNT,
    MAXIMUM_PROVIDER_RETRY_ATTEMPTS,
    MAXIMUM_PROVIDER_RETRY_BACKOFF_SECONDS,
    MAXIMUM_PROVIDER_TIMEOUT_SECONDS,
    MAXIMUM_SKIP_SECONDS,
    MINIMUM_CHUNK_OVERLAP_SECONDS,
    MINIMUM_CHUNK_TARGET_BYTES,
    MINIMUM_COST,
    MINIMUM_ESCALATION_CONTEXT_SECONDS,
    MINIMUM_ESCALATIONS_PER_RECORDING,
    MINIMUM_EXPECTED_SPEAKER_COUNT,
    MINIMUM_PROVIDER_RETRY_ATTEMPTS,
    MINIMUM_PROVIDER_RETRY_BACKOFF_SECONDS,
    MINIMUM_PROVIDER_TIMEOUT_SECONDS,
    MINIMUM_SKIP_SECONDS,
    AssemblyAiSettings,
    CostSettings,
    DeepgramSettings,
    ElevenLabsSettings,
    MicrosoftMaiSettings,
    OpenAiAdjudicationSettings,
    OpenAiTranscriptionSettings,
    ProcessingSettings,
    Settings,
)
from vox_verbatim.transcription.model import Language
from vox_verbatim.transcription.vocabulary import (
    TermCategory,
    Vocabulary,
    VocabularyLevel,
    VocabularyProfile,
    VocabularyTerm,
)
from vox_verbatim.ui import settings_notes as notes
from vox_verbatim.ui.accessibility import announce, describe

_log = logging.getLogger(__name__)

#: How many lines of JSON a parameter box shows before it scrolls. Enough
#: for two or three parameters, which is what these boxes really hold.
_PARAMETER_LINES = 5

#: How many rows of the terms table are visible before it scrolls.
_TERM_ROWS = 6


@dataclass(frozen=True)
class Problem:
    """Something that must be put right before the dialog can close.

    The widget travels with the message because the message on its own is
    not enough to act on. A user who is told that some JSON is wrong needs
    to be put back in the box that holds it, on the page that holds the box,
    and that cannot be worked out from the words afterwards.
    """

    message: str
    widget: QWidget


# -- The free-form parameter dictionaries ---------------------------------


def _refuse_constant(name: str) -> Any:
    """Reject the values Python will read as JSON but nothing else will.

    ``json.loads`` accepts ``NaN``, ``Infinity`` and ``-Infinity`` by
    default, and Python will happily hold them. They are not JSON, and the
    settings file refuses to write them, so a dictionary containing one
    would be thrown away whole the next time the settings were loaded. The
    user would have typed something the dialog accepted and then found it
    gone, which is exactly the silent loss this box is meant not to have.
    """
    raise ValueError(
        f"{name} is not a value JSON can carry. Use an ordinary number, or leave the "
        "value out altogether"
    )


def _kind_of(value: Any) -> str:
    """What a piece of parsed JSON is, in words a person can act on."""
    if isinstance(value, list):
        return "a list"
    if isinstance(value, bool):
        return "a true or false value"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return "a piece of text"
    if value is None:
        return "nothing at all"
    return "something else"


def parse_parameters(text: str, title: str) -> tuple[dict[str, Any] | None, str]:
    """Read a parameter box, or say exactly what is wrong with it.

    Returns the parameters and an empty message when the text is usable, and
    ``None`` with a message that can be read out when it is not. Nothing is
    ever half-read: a box that cannot be understood leaves the settings
    exactly as they were rather than losing part of what was typed.

    An empty box is not an error. It means no extra parameters, which is the
    right answer for almost everybody.
    """
    stripped = text.strip()
    if not stripped:
        return {}, ""
    try:
        value = json.loads(stripped, parse_constant=_refuse_constant)
    except json.JSONDecodeError as error:
        # This has to come first: JSONDecodeError is a kind of ValueError,
        # so catching ValueError above it would swallow every syntax error
        # and report it without the line and column that make it findable.
        return None, (
            f"{title}: this is not valid JSON. {error.msg}, at line {error.lineno}, "
            f"column {error.colno}. Correct it or clear the box. Nothing you typed "
            "has been changed."
        )
    except ValueError as error:
        return None, (
            f"{title}: {error}. Correct it or clear the box. Nothing you typed has "
            "been changed."
        )
    if not isinstance(value, dict):
        return None, (
            f"{title}: this must be a JSON object, which is named values inside "
            'braces, such as {"temperature": 0}. What is there now is '
            f"{_kind_of(value)}. Correct it or clear the box. Nothing you typed has "
            "been changed."
        )
    return value, ""


def parameters_as_text(parameters: dict[str, Any]) -> str:
    """Lay a parameter dictionary out for editing, or return nothing at all.

    An empty dictionary becomes an empty box rather than a pair of braces,
    because an empty box is what "there is nothing extra here" looks like,
    and because a screen reader reading "open brace close brace" is being
    told about punctuation rather than about the setting.
    """
    if not parameters:
        return ""
    return json.dumps(parameters, indent=2, sort_keys=True)


# -- The page every category shares ---------------------------------------


class SettingsPage(QWidget):
    """One page of the Settings dialog.

    A page knows how to show a copy of the settings, how to write itself
    back into one, and what is wrong with what it currently holds. It does
    not know about the dialog, the category list or the note panel, which is
    what lets a page be built and tested on its own.
    """

    def __init__(self, category: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.category = category
        # The page itself is named, because in this layout the page is a
        # thing the user moves onto rather than an invisible container. A
        # screen reader announces it as they arrive, which is what ties the
        # category they chose on the left to the controls on the right.
        describe(self, f"{category} settings", notes.CATEGORY_SUMMARIES[category])

        #: Which note belongs to which control. The dialog reads this to
        #: make its explanation panel follow the focus.
        self.note_keys: dict[QWidget, str] = {}
        self._parameter_boxes: list[QPlainTextEdit] = []
        self._parameter_titles: dict[QPlainTextEdit, str] = {}
        self._last_good_parameters: dict[QPlainTextEdit, dict[str, Any]] = {}
        self._required_texts: list[tuple[QLineEdit, str]] = []

    # -- What a page must be able to do -----------------------------------

    def show_settings(self, settings: Settings) -> None:
        """Put the values from ``settings`` into this page's controls."""

    def apply_to(self, settings: Settings) -> None:
        """Write this page's controls back into ``settings``, in place."""

    def status_message(self) -> str:
        """What this page has to say about itself, or nothing.

        A page that describes a service says whether that service is set up.
        Everything else stays quiet.
        """
        return ""

    def problems(self) -> list[Problem]:
        """Everything that would have to be corrected before OK can close.

        The base answer covers the two mistakes that are possible on every
        page: a parameter box that is not JSON, and a name that a request
        cannot be sent without. Handling them here rather than page by page
        is deliberate. A new service added a year from now gets both checks
        without anybody remembering to ask for them, and a service whose
        author forgot would otherwise fail silently.

        Required names are only checked while :meth:`requires_texts` says
        so. Broken JSON is always reported, because it is lost on saving
        whether or not anything reads it.
        """
        found: list[Problem] = []
        for box in self._parameter_boxes:
            _, message = parse_parameters(box.toPlainText(), self._parameter_titles[box])
            if message:
                found.append(Problem(message, box))
        if self.requires_texts():
            for edit, message in self._required_texts:
                if not edit.text().strip():
                    found.append(Problem(message, edit))
        return found

    def requires_texts(self) -> bool:
        """Whether an empty required name has to stop OK closing.

        Most pages are always in use, so the answer is yes. A page for a
        service that can be switched off answers for itself.
        """
        return True

    # -- Building controls -------------------------------------------------

    def _new_form(self, parent: QWidget | None = None) -> QFormLayout:
        """A form whose long rows wrap rather than forcing the dialog wider."""
        form = QFormLayout(parent if parent is not None else self)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        return form

    def explain(self, widget: QWidget, key: str) -> QWidget:
        """Name a control from its note, and have the panel follow it.

        The short summary becomes the accessible description and the
        tooltip, so a screen reader reads it on focus and a mouse user sees
        it on hover. The full note is left to the panel, because hearing
        four paragraphs every time the focus moves would be unusable.
        """
        note = notes.note_for(key)
        describe(widget, note.title, note.summary)
        self.note_keys[widget] = key
        return widget

    def add_check(self, form: QFormLayout, text: str, key: str) -> QCheckBox:
        box = QCheckBox(text, self)
        self.explain(box, key)
        form.addRow(box)
        return box

    def add_row(self, form: QFormLayout, label_text: str, widget: QWidget, key: str) -> QWidget:
        """Add a labelled control, with the label really pointing at it.

        A label with no buddy names nothing. The field beside it is then
        read out as an unnamed edit box with a value in it, and the words on
        screen belong to no control at all.
        """
        label = QLabel(label_text, self)
        self.explain(widget, key)
        label.setBuddy(widget)
        form.addRow(label, widget)
        return widget

    def add_text(
        self, form: QFormLayout, label_text: str, key: str, required_because: str = ""
    ) -> QLineEdit:
        edit = QLineEdit(self)
        self.add_row(form, label_text, edit, key)
        if required_because:
            self._required_texts.append((edit, required_because))
        return edit

    def add_key(self, form: QFormLayout, service: str, key: str) -> tuple[QLineEdit, QCheckBox]:
        """A password box for an API key, with a box that reveals it.

        Both halves are needed. Hiding the key is right, because a settings
        dialog is opened in front of other people. Revealing it is also
        right, because a key is pasted rather than typed and a paste can
        silently go wrong: the wrong clipboard entry, a trailing space, half
        of it. A sighted user checks by looking, and a masked box gives a
        screen reader nothing but a row of dots to read, so without the Show
        box a screen reader user simply cannot check their own key.
        """
        edit = QLineEdit(self)
        edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.explain(edit, key)

        show_box = QCheckBox("Sho&w the key", self)
        describe(
            show_box,
            f"Show the {service} API key",
            "Shows the key as ordinary text instead of dots, so you can check that "
            "what you pasted is what you meant to paste.",
        )
        # Standing on the Show box is standing on the key, as far as the
        # explanation goes, so the panel does not clear when you reach it.
        self.note_keys[show_box] = key
        show_box.toggled.connect(
            lambda shown: self._set_key_visible(edit, show_box, service, shown)
        )

        label = QLabel("&API key", self)
        label.setBuddy(edit)
        row = QWidget(self)
        inner = QHBoxLayout(row)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.addWidget(edit, 1)
        inner.addWidget(show_box, 0)
        form.addRow(label, row)
        return edit, show_box

    def _set_key_visible(
        self, edit: QLineEdit, show_box: QCheckBox, service: str, shown: bool
    ) -> None:
        """Reveal or hide a key, and say which has just happened.

        The checkbox reports its own tick state, but a tick is not the point:
        whether the key is on screen is. So the change is announced in the
        words that matter, which is also what tells a user in a shared office
        that they have left a credential showing.
        """
        edit.setEchoMode(
            QLineEdit.EchoMode.Normal if shown else QLineEdit.EchoMode.Password
        )
        announce(
            show_box,
            f"The {service} API key is now {'shown' if shown else 'hidden'}.",
        )

    def add_number(
        self,
        form: QFormLayout,
        label_text: str,
        key: str,
        lowest: float,
        highest: float,
        suffix: str,
        decimals: int = 1,
        step: float = 1.0,
    ) -> QDoubleSpinBox:
        box = QDoubleSpinBox(self)
        box.setRange(lowest, highest)
        box.setDecimals(decimals)
        box.setSingleStep(step)
        box.setSuffix(suffix)
        # The box will not accept a value outside its range, so there is no
        # error state to report and nothing for the user to correct.
        self.add_row(form, label_text, box, key)
        return box

    def add_count(
        self,
        form: QFormLayout,
        label_text: str,
        key: str,
        lowest: int,
        highest: int,
        suffix: str,
        step: int = 1,
    ) -> QSpinBox:
        box = QSpinBox(self)
        box.setRange(lowest, highest)
        box.setSingleStep(step)
        box.setSuffix(suffix)
        # Large counts are grouped, so that twenty million bytes is read out
        # as a number rather than as eight digits in a row.
        box.setGroupSeparatorShown(True)
        self.add_row(form, label_text, box, key)
        return box

    def add_parameters(self, form: QFormLayout, label_text: str, key: str) -> QPlainTextEdit:
        """A multi-line box of JSON, and the promise to check it on the way out."""
        box = QPlainTextEdit(self)
        # Without this, Tab types a tab character into the box and a
        # keyboard user is trapped in it with no way out but the mouse.
        box.setTabChangesFocus(True)
        box.setFixedHeight(self.fontMetrics().lineSpacing() * _PARAMETER_LINES)
        note = notes.note_for(key)
        self.add_row(form, label_text, box, key)
        self._parameter_boxes.append(box)
        self._parameter_titles[box] = note.title
        self._last_good_parameters[box] = {}
        return box

    # -- Parameters in and out ---------------------------------------------

    def show_parameters(self, box: QPlainTextEdit, parameters: dict[str, Any]) -> None:
        box.setPlainText(parameters_as_text(parameters))
        self._last_good_parameters[box] = dict(parameters)

    def parameters_of(self, box: QPlainTextEdit) -> dict[str, Any]:
        """What a parameter box holds, or what it last held that made sense.

        Falling back matters because these settings are read while the
        dialog is still open, to work out what is still missing for a run.
        Text that does not parse is reported by :meth:`problems` and stops
        the dialog closing; until then the last good value stands in, so
        nothing else has to cope with half a dictionary.
        """
        value, _ = parse_parameters(box.toPlainText(), self._parameter_titles[box])
        if value is None:
            return dict(self._last_good_parameters.get(box, {}))
        return value


class ProviderPage(SettingsPage):
    """A page describing one service, which says whether it is set up.

    Whether a service can be called is the single most useful thing this
    page can tell somebody, and it must be told in words. A greyed-out
    field, a red border or a warning triangle says nothing to a screen
    reader and nothing to anybody who cannot make out the colour.

    The words come from the settings rather than from here. Each section of
    the settings knows whether it is configured and carries a phrase saying
    what it needs, so a service that gains a required field says so on this
    page without this file being touched.
    """

    #: The box that switches the service on and off. Every service page
    #: builds one, because every service section of the settings has one.
    _enabled_box: QCheckBox

    def __init__(self, category: str, service_name: str, parent: QWidget | None = None) -> None:
        super().__init__(category, parent)
        self.service_name = service_name
        # A label carries a message, so it is left unnamed on purpose. A
        # label has no accessible value of its own: its text is its name,
        # and naming it would hide what it says.
        self._status_label = QLabel(self)
        self._status_label.setWordWrap(True)

    def _current_section(self) -> Any:
        """This page's controls, as the settings object they describe."""
        raise NotImplementedError

    def status_message(self) -> str:
        section = self._current_section()
        if not section.enabled:
            return (
                f"{self.service_name} is switched off. It will not be called, and "
                "nothing here is used until you switch it on."
            )
        if section.is_configured:
            return f"{self.service_name} is switched on and is set up."
        return (
            f"{self.service_name} is switched on but is not set up. It needs "
            f"{section.requirements}."
        )

    def requires_texts(self) -> bool:
        """Only while the service is switched on.

        A service that is switched off is never called, so a name it would
        need for a request does not matter. Refusing to save over it would
        trap somebody who cleared a box on a service they do not use: the
        only way out would be to type a model name for that service. The
        empty name is replaced by the default the next time the settings are
        loaded, which is what that person needs if they switch it on later.
        """
        return self._enabled_box.isChecked()

    def refresh_status(self) -> None:
        self._status_label.setText(self.status_message())

    def watch(self, *widgets: QWidget) -> None:
        """Update the "is it set up" message as the user types.

        Every control that can change the answer reports to here, so the
        message is right the moment a key is pasted rather than the next
        time the page is opened.
        """
        for widget in widgets:
            if isinstance(widget, QLineEdit):
                widget.textChanged.connect(self.refresh_status)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self.refresh_status)


# -- General ---------------------------------------------------------------


class GeneralPage(SettingsPage):
    """How the application starts, and how far the skip buttons move."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(notes.GENERAL, parent)
        layout = QVBoxLayout(self)
        form = self._new_form(None)
        layout.addLayout(form)

        self._reopen_box = self.add_check(
            form,
            "&Reopen the last folder when the application starts",
            "general.reopen_last_folder",
        )

        group = QGroupBox("Skip intervals", self)
        skip_form = self._new_form(group)
        self._short_box = self._add_skip(skip_form, "&Short skip", "general.short_skip_seconds")
        self._medium_box = self._add_skip(
            skip_form, "&Medium skip", "general.medium_skip_seconds"
        )
        self._long_box = self._add_skip(skip_form, "&Long skip", "general.long_skip_seconds")
        layout.addWidget(group)
        layout.addStretch(1)

    def _add_skip(self, form: QFormLayout, label_text: str, key: str) -> QSpinBox:
        return self.add_count(
            form, label_text, key, MINIMUM_SKIP_SECONDS, MAXIMUM_SKIP_SECONDS, " seconds", step=5
        )

    def show_settings(self, settings: Settings) -> None:
        self._reopen_box.setChecked(settings.reopen_last_folder)
        self._short_box.setValue(settings.short_skip_seconds)
        self._medium_box.setValue(settings.medium_skip_seconds)
        self._long_box.setValue(settings.long_skip_seconds)

    def apply_to(self, settings: Settings) -> None:
        settings.reopen_last_folder = self._reopen_box.isChecked()
        settings.short_skip_seconds = self._short_box.value()
        settings.medium_skip_seconds = self._medium_box.value()
        settings.long_skip_seconds = self._long_box.value()


# -- Transcription ---------------------------------------------------------


class TranscriptionPage(SettingsPage):
    """How a run behaves, as opposed to who it talks to."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(notes.TRANSCRIPTION, parent)
        layout = QVBoxLayout(self)

        recordings = QGroupBox("New recordings", self)
        form = self._new_form(recordings)
        self._afrikaans_box = self.add_check(
            form, "Expect &Afrikaans in new recordings", "processing.default_afrikaans_enabled"
        )
        self._speaker_box = self.add_count(
            form,
            "&Expected number of speakers",
            "processing.default_expected_speaker_count",
            MINIMUM_EXPECTED_SPEAKER_COUNT,
            MAXIMUM_EXPECTED_SPEAKER_COUNT,
            " speakers",
        )
        layout.addWidget(recordings)

        services = QGroupBox("Talking to the services", self)
        form = self._new_form(services)
        self._timeout_box = self.add_number(
            form,
            "How long to &wait for a service",
            "processing.provider_timeout_seconds",
            MINIMUM_PROVIDER_TIMEOUT_SECONDS,
            MAXIMUM_PROVIDER_TIMEOUT_SECONDS,
            " seconds",
            step=30.0,
        )
        self._retry_box = self.add_count(
            form,
            "&Retry attempts",
            "processing.provider_retry_attempts",
            MINIMUM_PROVIDER_RETRY_ATTEMPTS,
            MAXIMUM_PROVIDER_RETRY_ATTEMPTS,
            " attempts",
        )
        self._backoff_box = self.add_number(
            form,
            "Wai&t before retrying",
            "processing.provider_retry_backoff_seconds",
            MINIMUM_PROVIDER_RETRY_BACKOFF_SECONDS,
            MAXIMUM_PROVIDER_RETRY_BACKOFF_SECONDS,
            " seconds",
            step=0.5,
        )
        self._overlap_box = self.add_number(
            form,
            "&Overlap between chunks",
            "processing.provider_chunk_overlap_seconds",
            MINIMUM_CHUNK_OVERLAP_SECONDS,
            MAXIMUM_CHUNK_OVERLAP_SECONDS,
            " seconds",
            step=0.5,
        )
        layout.addWidget(services)

        disputes = QGroupBox("Settling disagreements", self)
        form = self._new_form(disputes)
        self._escalation_box = self.add_check(
            form, "Send &disputed words for a second opinion", "processing.escalation_enabled"
        )
        self._before_box = self.add_number(
            form,
            "Context &before a disputed word",
            "processing.escalation_context_seconds_before",
            MINIMUM_ESCALATION_CONTEXT_SECONDS,
            MAXIMUM_ESCALATION_CONTEXT_SECONDS,
            " seconds",
            step=0.5,
        )
        self._after_box = self.add_number(
            form,
            "Context a&fter a disputed word",
            "processing.escalation_context_seconds_after",
            MINIMUM_ESCALATION_CONTEXT_SECONDS,
            MAXIMUM_ESCALATION_CONTEXT_SECONDS,
            " seconds",
            step=0.5,
        )
        self._maximum_box = self.add_count(
            form,
            "Most escalations &per recording",
            "processing.maximum_escalations_per_recording",
            MINIMUM_ESCALATIONS_PER_RECORDING,
            MAXIMUM_ESCALATIONS_PER_RECORDING,
            " escalations",
            step=10,
        )
        self._adjudication_box = self.add_check(
            form,
            "&Let a reasoning model settle what is left",
            "processing.adjudication_enabled",
        )
        layout.addWidget(disputes)

        results = QGroupBox("What a run produces", self)
        form = self._new_form(results)
        self._alignment_box = self.add_check(
            form,
            "&Measure the timing of corrected words",
            "processing.forced_alignment_enabled",
        )
        self._suffix_edit = self.add_text(
            form,
            "Transcript folder name e&nding",
            "processing.transcript_folder_suffix",
            required_because=(
                "The transcript folder name ending is empty. It is added to a "
                "recording's own name to make the folder written beside it, so it "
                "has to say something."
            ),
        )
        layout.addWidget(results)

        needed = QGroupBox("Before a run can be made", self)
        inner = QVBoxLayout(needed)
        # Another message-carrying label, so it is left unnamed.
        self._requirements_label = QLabel(self)
        self._requirements_label.setWordWrap(True)
        inner.addWidget(self._requirements_label)
        layout.addWidget(needed)
        layout.addStretch(1)

    def show_settings(self, settings: Settings) -> None:
        processing = settings.transcription.processing
        self._afrikaans_box.setChecked(processing.default_afrikaans_enabled)
        self._speaker_box.setValue(processing.default_expected_speaker_count)
        self._timeout_box.setValue(processing.provider_timeout_seconds)
        self._retry_box.setValue(processing.provider_retry_attempts)
        self._backoff_box.setValue(processing.provider_retry_backoff_seconds)
        self._overlap_box.setValue(processing.provider_chunk_overlap_seconds)
        self._escalation_box.setChecked(processing.escalation_enabled)
        self._before_box.setValue(processing.escalation_context_seconds_before)
        self._after_box.setValue(processing.escalation_context_seconds_after)
        self._maximum_box.setValue(processing.maximum_escalations_per_recording)
        self._adjudication_box.setChecked(processing.adjudication_enabled)
        self._alignment_box.setChecked(processing.forced_alignment_enabled)
        self._suffix_edit.setText(processing.transcript_folder_suffix)

    def apply_to(self, settings: Settings) -> None:
        settings.transcription.processing = ProcessingSettings(
            default_afrikaans_enabled=self._afrikaans_box.isChecked(),
            default_expected_speaker_count=self._speaker_box.value(),
            escalation_context_seconds_before=self._before_box.value(),
            escalation_context_seconds_after=self._after_box.value(),
            provider_timeout_seconds=self._timeout_box.value(),
            provider_retry_attempts=self._retry_box.value(),
            provider_retry_backoff_seconds=self._backoff_box.value(),
            provider_chunk_overlap_seconds=self._overlap_box.value(),
            forced_alignment_enabled=self._alignment_box.isChecked(),
            escalation_enabled=self._escalation_box.isChecked(),
            adjudication_enabled=self._adjudication_box.isChecked(),
            maximum_escalations_per_recording=self._maximum_box.value(),
            transcript_folder_suffix=self._suffix_edit.text().strip(),
        )

    def show_requirements(self, transcription: Any) -> None:
        """Say what would stop a run, in the settings' own words.

        The sentences come from the settings rather than being written here,
        so that a rule about what a run needs is stated in one place. This
        page only decides when to show them.
        """
        problems = transcription.missing_requirements()
        if not problems:
            self._requirements_label.setText(
                "Everything a run needs is set up. Each service that is switched on "
                "has what it needs to be called."
            )
            return
        self._requirements_label.setText("\n".join(problems))

    def problems(self) -> list[Problem]:
        """The usual checks, and one that only the settings themselves can make.

        The folder name ending decides where a transcript is written, so a
        value that would not survive being saved must not be accepted here.
        Rather than repeating the rules about separators and dots, which
        would be correct until somebody changed them in one place and not
        the other, the value is offered to the settings and the answer is
        compared with what was typed. Anything the settings would replace is
        something this dialog must refuse, whatever the reason.
        """
        found = super().problems()
        typed = self._suffix_edit.text().strip()
        if typed:
            kept = ProcessingSettings.from_dict(
                {"transcript_folder_suffix": typed}
            ).transcript_folder_suffix
            if kept != typed:
                found.append(
                    Problem(
                        f'"{typed}" cannot be used as the transcript folder name '
                        "ending. It becomes part of a folder name, so it cannot "
                        "contain a backslash, a forward slash, a colon, a star, a "
                        'question mark, a quotation mark, a bar, or "..". Saving it '
                        f'would have quietly written "{kept}" instead.',
                        self._suffix_edit,
                    )
                )
        return found


# -- Costs -----------------------------------------------------------------


class CostsPage(SettingsPage):
    """What each service charges, so a run can be priced before it starts."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(notes.COSTS, parent)
        layout = QVBoxLayout(self)

        rates = QGroupBox("Rates in United States dollars", self)
        form = self._new_form(rates)
        self._elevenlabs_box = self._add_rate(
            form, "&ElevenLabs rate per minute", "cost.elevenlabs_per_minute"
        )
        self._openai_box = self._add_rate(
            form, "&OpenAI transcription rate per minute", "cost.openai_transcription_per_minute"
        )
        self._microsoft_box = self._add_rate(
            form, "&Microsoft MAI rate per minute", "cost.microsoft_per_minute"
        )
        self._assemblyai_box = self._add_rate(
            form, "&AssemblyAI rate per minute", "cost.assemblyai_per_minute"
        )
        self._deepgram_box = self._add_rate(
            form, "&Deepgram rate per minute", "cost.deepgram_per_minute"
        )
        self._adjudication_box = self._add_rate(
            form, "Ad&judication cost per request", "cost.adjudication_per_request"
        )
        self._smoothing_box = self._add_rate(
            form, "Smoot&hing cost per request", "cost.smoothing_per_request"
        )
        layout.addWidget(rates)

        form = self._new_form(None)
        self._confirm_box = self.add_check(
            form, "Show the estimated cost &before a run starts", "cost.confirm_before_running"
        )
        layout.addLayout(form)

        # A message-carrying label, so it is left unnamed. It says out loud
        # what the rates cannot say for themselves.
        warning = QLabel(
            "These rates are yours to keep right. They were correct when they were "
            "written and the services change their prices without telling us, so an "
            "estimate is only as honest as the figures above.",
            self,
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)
        layout.addStretch(1)

    def _add_rate(self, form: QFormLayout, label_text: str, key: str) -> QDoubleSpinBox:
        # Four decimal places, because these are fractions of a cent per
        # minute and rounding them to two would turn most of them into zero.
        return self.add_number(
            form, label_text, key, MINIMUM_COST, MAXIMUM_COST, " US dollars",
            decimals=4, step=0.001,
        )

    def show_settings(self, settings: Settings) -> None:
        cost = settings.transcription.cost
        self._elevenlabs_box.setValue(cost.elevenlabs_per_minute)
        self._openai_box.setValue(cost.openai_transcription_per_minute)
        self._microsoft_box.setValue(cost.microsoft_per_minute)
        self._assemblyai_box.setValue(cost.assemblyai_per_minute)
        self._deepgram_box.setValue(cost.deepgram_per_minute)
        self._adjudication_box.setValue(cost.adjudication_per_request)
        self._smoothing_box.setValue(cost.smoothing_per_request)
        self._confirm_box.setChecked(cost.confirm_before_running)

    def apply_to(self, settings: Settings) -> None:
        settings.transcription.cost = CostSettings(
            elevenlabs_per_minute=self._elevenlabs_box.value(),
            openai_transcription_per_minute=self._openai_box.value(),
            microsoft_per_minute=self._microsoft_box.value(),
            assemblyai_per_minute=self._assemblyai_box.value(),
            deepgram_per_minute=self._deepgram_box.value(),
            adjudication_per_request=self._adjudication_box.value(),
            smoothing_per_request=self._smoothing_box.value(),
            confirm_before_running=self._confirm_box.isChecked(),
        )


# -- One page per service --------------------------------------------------


class ElevenLabsPage(ProviderPage):
    """ElevenLabs Scribe: the timings, the speakers and the confidence figures."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(notes.ELEVENLABS, "ElevenLabs Scribe", parent)
        layout = QVBoxLayout(self)
        form = self._new_form(None)
        layout.addLayout(form)

        self._enabled_box = self.add_check(form, "&Use ElevenLabs Scribe", "elevenlabs.enabled")
        self._key_edit, self._show_key_box = self.add_key(form, "ElevenLabs", "elevenlabs.api_key")
        self._model_edit = self.add_text(
            form,
            "&Transcription model",
            "elevenlabs.transcription_model",
            required_because=(
                "The ElevenLabs transcription model name is empty. A request cannot "
                "be sent without one, and an empty name would be replaced by the "
                "default the next time the settings were loaded. Type a model name, "
                "or switch the service off."
            ),
        )
        self._diarise_box = self.add_check(
            form, "Tell the &speakers apart", "elevenlabs.diarise"
        )
        self._events_box = self.add_check(
            form, "&Mark laughter, applause and other sounds", "elevenlabs.tag_audio_events"
        )
        self._transcription_parameters = self.add_parameters(
            form, "Extra transcription &parameters", "elevenlabs.transcription_parameters"
        )
        self._alignment_parameters = self.add_parameters(
            form, "Extra alignment param&eters", "elevenlabs.forced_alignment_parameters"
        )
        layout.addWidget(self._status_label)
        layout.addStretch(1)
        self.watch(self._enabled_box, self._key_edit, self._model_edit)

    def show_settings(self, settings: Settings) -> None:
        section = settings.transcription.elevenlabs
        self._enabled_box.setChecked(section.enabled)
        self._key_edit.setText(section.api_key)
        self._model_edit.setText(section.transcription_model)
        self._diarise_box.setChecked(section.diarise)
        self._events_box.setChecked(section.tag_audio_events)
        self.show_parameters(self._transcription_parameters, section.transcription_parameters)
        self.show_parameters(self._alignment_parameters, section.forced_alignment_parameters)
        self.refresh_status()

    def _current_section(self) -> ElevenLabsSettings:
        return ElevenLabsSettings(
            api_key=self._key_edit.text().strip(),
            transcription_model=self._model_edit.text().strip(),
            transcription_parameters=self.parameters_of(self._transcription_parameters),
            forced_alignment_parameters=self.parameters_of(self._alignment_parameters),
            diarise=self._diarise_box.isChecked(),
            tag_audio_events=self._events_box.isChecked(),
            enabled=self._enabled_box.isChecked(),
        )

    def apply_to(self, settings: Settings) -> None:
        settings.transcription.elevenlabs = self._current_section()


class OpenAiTranscriptionPage(ProviderPage):
    """OpenAI asked what was said, as the second full reading of a recording."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(notes.OPENAI_TRANSCRIPTION, "OpenAI transcription", parent)
        layout = QVBoxLayout(self)
        form = self._new_form(None)
        layout.addLayout(form)

        self._enabled_box = self.add_check(
            form, "&Use OpenAI for transcription", "openai_transcription.enabled"
        )
        self._key_edit, self._show_key_box = self.add_key(
            form, "OpenAI transcription", "openai_transcription.api_key"
        )
        self._model_edit = self.add_text(
            form,
            "&Model",
            "openai_transcription.model",
            required_because=(
                "The OpenAI transcription model name is empty. A request cannot be "
                "sent without one, and an empty name would be replaced by the "
                "default the next time the settings were loaded. Type a model name, "
                "or switch the service off."
            ),
        )
        self._chunk_box = self.add_count(
            form,
            "Chunk si&ze to aim for",
            "openai_transcription.chunk_target_bytes",
            MINIMUM_CHUNK_TARGET_BYTES,
            MAXIMUM_CHUNK_TARGET_BYTES,
            " bytes",
            step=1_000_000,
        )
        self._parameters = self.add_parameters(
            form, "Extra &parameters", "openai_transcription.parameters"
        )
        layout.addWidget(self._status_label)
        layout.addStretch(1)
        self.watch(self._enabled_box, self._key_edit, self._model_edit)

    def show_settings(self, settings: Settings) -> None:
        section = settings.transcription.openai_transcription
        self._enabled_box.setChecked(section.enabled)
        self._key_edit.setText(section.api_key)
        self._model_edit.setText(section.model)
        self._chunk_box.setValue(section.chunk_target_bytes)
        self.show_parameters(self._parameters, section.parameters)
        self.refresh_status()

    def _current_section(self) -> OpenAiTranscriptionSettings:
        return OpenAiTranscriptionSettings(
            api_key=self._key_edit.text().strip(),
            model=self._model_edit.text().strip(),
            parameters=self.parameters_of(self._parameters),
            chunk_target_bytes=self._chunk_box.value(),
            enabled=self._enabled_box.isChecked(),
        )

    def apply_to(self, settings: Settings) -> None:
        settings.transcription.openai_transcription = self._current_section()


class OpenAiAdjudicationPage(ProviderPage):
    """The reasoning model that settles what the rules could not."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(notes.OPENAI_ADJUDICATION, "OpenAI adjudication", parent)
        layout = QVBoxLayout(self)
        form = self._new_form(None)
        layout.addLayout(form)

        self._enabled_box = self.add_check(
            form, "&Use OpenAI for adjudication", "openai_adjudication.enabled"
        )
        self._key_edit, self._show_key_box = self.add_key(
            form, "OpenAI adjudication", "openai_adjudication.api_key"
        )
        self._model_edit = self.add_text(
            form,
            "&Model",
            "openai_adjudication.model",
            required_because=(
                "The adjudication model name is empty. A request cannot be sent "
                "without one, and an empty name would be replaced by the default "
                "the next time the settings were loaded. Type a model name, or "
                "switch adjudication off."
            ),
        )
        # Deliberately not marked as required: an empty reasoning effort is
        # a real answer, meaning "leave the parameter out of the request",
        # which is what a model that does not accept one needs.
        self._effort_edit = self.add_text(
            form, "&Reasoning effort", "openai_adjudication.reasoning_effort"
        )
        self._parameters = self.add_parameters(
            form, "Extra &parameters", "openai_adjudication.parameters"
        )
        layout.addWidget(self._status_label)
        layout.addStretch(1)
        self.watch(self._enabled_box, self._key_edit, self._model_edit)

    def show_settings(self, settings: Settings) -> None:
        section = settings.transcription.openai_adjudication
        self._enabled_box.setChecked(section.enabled)
        self._key_edit.setText(section.api_key)
        self._model_edit.setText(section.model)
        self._effort_edit.setText(section.reasoning_effort)
        self.show_parameters(self._parameters, section.parameters)
        self.refresh_status()

    def _current_section(self) -> OpenAiAdjudicationSettings:
        return OpenAiAdjudicationSettings(
            api_key=self._key_edit.text().strip(),
            model=self._model_edit.text().strip(),
            reasoning_effort=self._effort_edit.text().strip(),
            parameters=self.parameters_of(self._parameters),
            enabled=self._enabled_box.isChecked(),
        )

    def apply_to(self, settings: Settings) -> None:
        settings.transcription.openai_adjudication = self._current_section()


class MicrosoftPage(ProviderPage):
    """Microsoft MAI-Transcribe, through the user's own Azure resource."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(notes.MICROSOFT, "Microsoft MAI", parent)
        layout = QVBoxLayout(self)
        form = self._new_form(None)
        layout.addLayout(form)

        self._enabled_box = self.add_check(form, "&Use Microsoft MAI", "microsoft.enabled")
        self._key_edit, self._show_key_box = self.add_key(
            form, "Microsoft MAI", "microsoft.api_key"
        )
        # Not marked as required: an empty endpoint is how a user who has
        # not created an Azure resource yet leaves this service alone, and
        # the status message already says that it cannot be called.
        self._endpoint_edit = self.add_text(form, "Resource &endpoint", "microsoft.endpoint")
        self._model_edit = self.add_text(
            form,
            "&Model",
            "microsoft.model",
            required_because=(
                "The Microsoft model name is empty. Azure names the model in the "
                "request rather than in the address, so a request cannot be sent "
                "without one. Type a model name, or switch the service off."
            ),
        )
        self._version_edit = self.add_text(
            form,
            "API &version",
            "microsoft.api_version",
            required_because=(
                "The Azure API version is empty. Azure requires it on every "
                "request, so a request cannot be sent without one."
            ),
        )
        self._parameters = self.add_parameters(form, "Extra &parameters", "microsoft.parameters")
        layout.addWidget(self._status_label)
        layout.addStretch(1)
        self.watch(self._enabled_box, self._key_edit, self._endpoint_edit, self._model_edit)

    def show_settings(self, settings: Settings) -> None:
        section = settings.transcription.microsoft
        self._enabled_box.setChecked(section.enabled)
        self._key_edit.setText(section.api_key)
        self._endpoint_edit.setText(section.endpoint)
        self._model_edit.setText(section.model)
        self._version_edit.setText(section.api_version)
        self.show_parameters(self._parameters, section.parameters)
        self.refresh_status()

    def _current_section(self) -> MicrosoftMaiSettings:
        return MicrosoftMaiSettings(
            api_key=self._key_edit.text().strip(),
            endpoint=self._endpoint_edit.text().strip(),
            model=self._model_edit.text().strip(),
            api_version=self._version_edit.text().strip(),
            parameters=self.parameters_of(self._parameters),
            enabled=self._enabled_box.isChecked(),
        )

    def apply_to(self, settings: Settings) -> None:
        settings.transcription.microsoft = self._current_section()


class AssemblyAiPage(ProviderPage):
    """AssemblyAI, asked to listen again to short disputed passages."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(notes.ASSEMBLYAI, "AssemblyAI", parent)
        layout = QVBoxLayout(self)
        form = self._new_form(None)
        layout.addLayout(form)

        self._enabled_box = self.add_check(form, "&Use AssemblyAI", "assemblyai.enabled")
        self._key_edit, self._show_key_box = self.add_key(form, "AssemblyAI", "assemblyai.api_key")
        self._primary_edit = self.add_text(
            form,
            "&Primary model",
            "assemblyai.primary_model",
            required_because=(
                "The AssemblyAI primary model name is empty. It is the model used "
                "for every language except Afrikaans, so a request cannot be sent "
                "without one. Type a model name, or switch the service off."
            ),
        )
        self._afrikaans_edit = self.add_text(
            form,
            "A&frikaans model",
            "assemblyai.afrikaans_model",
            required_because=(
                "The AssemblyAI Afrikaans model name is empty. The primary model "
                "cannot hear Afrikaans at all, so passages that may be Afrikaans "
                "would have nowhere to go. Type a model name."
            ),
        )
        self._parameters = self.add_parameters(
            form, "Extra re&quest parameters", "assemblyai.parameters"
        )
        layout.addWidget(self._status_label)
        layout.addStretch(1)
        self.watch(self._enabled_box, self._key_edit, self._primary_edit)

    def show_settings(self, settings: Settings) -> None:
        section = settings.transcription.assemblyai
        self._enabled_box.setChecked(section.enabled)
        self._key_edit.setText(section.api_key)
        self._primary_edit.setText(section.primary_model)
        self._afrikaans_edit.setText(section.afrikaans_model)
        self.show_parameters(self._parameters, section.parameters)
        self.refresh_status()

    def _current_section(self) -> AssemblyAiSettings:
        return AssemblyAiSettings(
            api_key=self._key_edit.text().strip(),
            primary_model=self._primary_edit.text().strip(),
            afrikaans_model=self._afrikaans_edit.text().strip(),
            parameters=self.parameters_of(self._parameters),
            enabled=self._enabled_box.isChecked(),
        )

    def apply_to(self, settings: Settings) -> None:
        settings.transcription.assemblyai = self._current_section()


class DeepgramPage(ProviderPage):
    """Deepgram, the optional challenger nothing depends on."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(notes.DEEPGRAM, "Deepgram", parent)
        layout = QVBoxLayout(self)
        form = self._new_form(None)
        layout.addLayout(form)

        self._enabled_box = self.add_check(form, "&Use Deepgram", "deepgram.enabled")
        self._key_edit, self._show_key_box = self.add_key(form, "Deepgram", "deepgram.api_key")
        self._model_edit = self.add_text(
            form,
            "&Model",
            "deepgram.model",
            required_because=(
                "The Deepgram model name is empty. A request cannot be sent without "
                "one. Type a model name, or switch the service off."
            ),
        )
        self._parameters = self.add_parameters(form, "Extra &parameters", "deepgram.parameters")
        layout.addWidget(self._status_label)
        layout.addStretch(1)
        self.watch(self._enabled_box, self._key_edit, self._model_edit)

    def show_settings(self, settings: Settings) -> None:
        section = settings.transcription.deepgram
        self._enabled_box.setChecked(section.enabled)
        self._key_edit.setText(section.api_key)
        self._model_edit.setText(section.model)
        self.show_parameters(self._parameters, section.parameters)
        self.refresh_status()

    def _current_section(self) -> DeepgramSettings:
        return DeepgramSettings(
            api_key=self._key_edit.text().strip(),
            model=self._model_edit.text().strip(),
            parameters=self.parameters_of(self._parameters),
            enabled=self._enabled_box.isChecked(),
        )

    def apply_to(self, settings: Settings) -> None:
        settings.transcription.deepgram = self._current_section()


# -- Vocabulary ------------------------------------------------------------

COLUMN_TERM = 0
COLUMN_CATEGORY = 1
COLUMN_LANGUAGE = 2
COLUMN_HEARD_AS = 3
COLUMN_CONFIRMED = 4

_TERM_HEADINGS = ("Term", "Category", "Language", "Usually heard as", "Times confirmed")


class TermTableModel(QAbstractTableModel):
    """The terms of one profile, as a standard Qt table model.

    A model behind a plain :class:`QTableView` rather than anything painted
    by hand, because Qt's own model and view already tell Windows what every
    cell is, which row and column it sits in, and how many of each there
    are. A custom widget would have to reproduce all of that to be usable at
    all, and would get it slightly wrong.

    Every column says something in words. A term with no category reads as
    "Not set" rather than as an empty cell, because an empty cell is silence
    to a screen reader and indistinguishable from the end of the table.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._terms: list[VocabularyTerm] = []

    def set_terms(self, terms: list[VocabularyTerm]) -> None:
        self.beginResetModel()
        self._terms = list(terms)
        self.endResetModel()

    def terms(self) -> list[VocabularyTerm]:
        return list(self._terms)

    def term_at(self, row: int) -> VocabularyTerm | None:
        if 0 <= row < len(self._terms):
            return self._terms[row]
        return None

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._terms)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(_TERM_HEADINGS)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        term = self.term_at(index.row())
        if term is None:
            return None
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return _term_column_text(term, index.column())
        return None

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return _TERM_HEADINGS[section]
        return section + 1

    def add_term(self, term: VocabularyTerm) -> int:
        row = len(self._terms)
        self.beginInsertRows(QModelIndex(), row, row)
        self._terms.append(term)
        self.endInsertRows()
        return row

    def replace_term(self, row: int, term: VocabularyTerm) -> None:
        if not 0 <= row < len(self._terms):
            return
        self._terms[row] = term
        first = self.index(row, 0)
        last = self.index(row, self.columnCount() - 1)
        self.dataChanged.emit(first, last)

    def remove_term(self, row: int) -> None:
        if not 0 <= row < len(self._terms):
            return
        self.beginRemoveRows(QModelIndex(), row, row)
        del self._terms[row]
        self.endRemoveRows()


def _term_column_text(term: VocabularyTerm, column: int) -> str:
    if column == COLUMN_TERM:
        return term.text
    if column == COLUMN_CATEGORY:
        return term.category.display_name if term.category else "Not set"
    if column == COLUMN_LANGUAGE:
        return term.language.display_name if term.language else "Any language"
    if column == COLUMN_HEARD_AS:
        return ", ".join(term.common_misrecognitions) or "Nothing recorded"
    if column == COLUMN_CONFIRMED:
        return str(term.confirmation_count)
    return ""


def _split_list(text: str) -> tuple[str, ...]:
    """Turn a comma-separated box into the entries it names."""
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _language_from(value: Any) -> Language | None:
    """Turn what a combo box hands back into a language, or into nothing.

    The categories and levels in this file are enumerations that are also
    strings, and Qt does not carry a Python object through a combo box
    unchanged: what goes in as ``VocabularyLevel.SPEAKER`` comes back as
    ``"speaker"``. Everywhere one of these is read back it therefore has to
    be turned into the enumeration again rather than trusted to still be
    one, which is what this and the ``from_value`` calls beside it are for.

    ``None`` is a real answer here and means the term belongs to every
    language, which most names do.
    """
    if not value:
        return None
    language = Language.from_code(str(value))
    return None if language is Language.UNKNOWN else language


class TermDialog(QDialog):
    """Add or change one term.

    A small dialog rather than editing in the table, because typing into a
    cell means the meaning of the cell has to be inferred from its column,
    and a category or a language chosen from a list in a cell is awkward
    with a screen reader. Here every field has a label of its own.
    """

    def __init__(self, term: VocabularyTerm | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit a term" if term else "Add a term")

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._text_edit = QLineEdit(self)
        describe(
            self._text_edit,
            "Term",
            "The word or phrase, spelled exactly as it should appear in a transcript.",
        )
        text_label = QLabel("&Term", self)
        text_label.setBuddy(self._text_edit)
        form.addRow(text_label, self._text_edit)

        self._category_box = _combo_of(
            self,
            [("Not set", None)] + [(item.display_name, item) for item in TermCategory],
            "Category",
            "What kind of thing this is. A disagreement over a person's name is "
            "weighed more carefully than one over an ordinary word.",
        )
        category_label = QLabel("&Category", self)
        category_label.setBuddy(self._category_box)
        form.addRow(category_label, self._category_box)

        self._language_box = _combo_of(
            self,
            [("Any language", None)]
            + [
                (language.display_name, language)
                for language in Language
                if language is not Language.UNKNOWN
            ],
            "Language",
            "Set this only for a term that belongs to one language. Most names "
            "belong everywhere, which is what Any language means.",
        )
        language_label = QLabel("&Language", self)
        language_label.setBuddy(self._language_box)
        form.addRow(language_label, self._language_box)

        self._hints_edit = QLineEdit(self)
        describe(
            self._hints_edit,
            "Pronunciation hints",
            "How the term sounds, for the services that can use that. Separate "
            "several with commas.",
        )
        hints_label = QLabel("&Pronunciation hints", self)
        hints_label.setBuddy(self._hints_edit)
        form.addRow(hints_label, self._hints_edit)

        self._heard_edit = QLineEdit(self)
        describe(
            self._heard_edit,
            "Usually heard as",
            "What the services usually return instead. Recording this lets a wrong "
            "reading be recognised as a known mistake. Separate several with commas.",
        )
        heard_label = QLabel("Usually &heard as", self)
        heard_label.setBuddy(self._heard_edit)
        form.addRow(heard_label, self._heard_edit)

        self._message_label = QLabel(self)
        self._message_label.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._message_label)
        layout.addWidget(buttons)

        self._confirmation_count = 0
        if term is not None:
            self._show_term(term)
        self._text_edit.setFocus(Qt.FocusReason.TabFocusReason)

    def _show_term(self, term: VocabularyTerm) -> None:
        self._text_edit.setText(term.text)
        self._category_box.setCurrentIndex(max(0, self._category_box.findData(term.category)))
        self._language_box.setCurrentIndex(max(0, self._language_box.findData(term.language)))
        self._hints_edit.setText(", ".join(term.pronunciation_hints))
        self._heard_edit.setText(", ".join(term.common_misrecognitions))
        # Carried through rather than shown. It counts how often a person
        # confirmed this term while reviewing real transcripts, so it is not
        # something to type in, and losing it on an edit would throw away
        # the evidence that decides which terms survive a service's limit.
        self._confirmation_count = term.confirmation_count

    def accept(self) -> None:
        """Refuse a term with no text, and say so where it can be heard."""
        if not self._text_edit.text().strip():
            message = "A term needs some text. Type the word or phrase, or press Cancel."
            self._message_label.setText(message)
            announce(self._text_edit, message, urgent=True)
            self._text_edit.setFocus(Qt.FocusReason.OtherFocusReason)
            return
        super().accept()

    def chosen_term(self) -> VocabularyTerm:
        return VocabularyTerm(
            text=self._text_edit.text().strip(),
            category=TermCategory.from_value(self._category_box.currentData()),
            language=_language_from(self._language_box.currentData()),
            pronunciation_hints=_split_list(self._hints_edit.text()),
            common_misrecognitions=_split_list(self._heard_edit.text()),
            confirmation_count=self._confirmation_count,
        )


class ProfileDialog(QDialog):
    """Name a profile and say how narrowly it applies."""

    def __init__(
        self,
        name: str = "",
        level: VocabularyLevel = VocabularyLevel.GLOBAL,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rename this profile" if name else "Add a profile")

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._name_edit = QLineEdit(self)
        self._name_edit.setText(name)
        describe(
            self._name_edit,
            "Profile name",
            "What to call this list, such as the client, the project or the person "
            "it belongs to.",
        )
        name_label = QLabel("&Name", self)
        name_label.setBuddy(self._name_edit)
        form.addRow(name_label, self._name_edit)

        self._level_box = _combo_of(
            self,
            [(item.display_name, item) for item in VocabularyLevel],
            "Level",
            "How narrowly this list applies. Narrower lists are kept when a service "
            "will not accept every term.",
        )
        self._level_box.setCurrentIndex(max(0, self._level_box.findData(level)))
        level_label = QLabel("&Level", self)
        level_label.setBuddy(self._level_box)
        form.addRow(level_label, self._level_box)

        self._message_label = QLabel(self)
        self._message_label.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._message_label)
        layout.addWidget(buttons)
        self._name_edit.setFocus(Qt.FocusReason.TabFocusReason)

    def accept(self) -> None:
        if not self._name_edit.text().strip():
            message = "A profile needs a name. Type one, or press Cancel."
            self._message_label.setText(message)
            announce(self._name_edit, message, urgent=True)
            self._name_edit.setFocus(Qt.FocusReason.OtherFocusReason)
            return
        super().accept()

    def chosen_name(self) -> str:
        return self._name_edit.text().strip()

    def chosen_level(self) -> VocabularyLevel:
        return VocabularyLevel.from_value(self._level_box.currentData()) or VocabularyLevel.GLOBAL


def _combo_of(
    parent: QWidget, entries: list[tuple[str, Any]], name: str, description: str
) -> QComboBox:
    box = QComboBox(parent)
    for label, value in entries:
        box.addItem(label, value)
    describe(box, name, description)
    return box


def _identifier_for(name: str, level: VocabularyLevel, taken: set[str]) -> str:
    """Make a stable, readable identifier for a new profile.

    The identifier is what a recording refers to, so it has to be unique and
    has to survive the profile being renamed. It is built from the name only
    because a readable one is far easier to recognise in a settings file
    than a random string when something has to be repaired by hand.
    """
    base = "-".join(part for part in name.lower().split() if part) or "profile"
    candidate = f"{level.value}-{base}"
    if candidate not in taken:
        return candidate
    number = 2
    while f"{candidate}-{number}" in taken:
        number += 1
    return f"{candidate}-{number}"


class VocabularyPage(SettingsPage):
    """The lists of words the services would otherwise get wrong.

    The vocabulary is not part of the settings. It is kept in its own file,
    because it grows with use and is edited by reviewing transcripts as well
    as by this page. So this page works on its own copy of it, handed in at
    the start and handed back when the dialog is accepted, exactly as the
    settings are.
    """

    def __init__(self, vocabulary: Vocabulary | None = None, parent: QWidget | None = None) -> None:
        super().__init__(notes.VOCABULARY, parent)
        self._vocabulary = vocabulary if vocabulary is not None else Vocabulary()

        layout = QVBoxLayout(self)

        profiles = QGroupBox("Profiles", self)
        profiles_layout = QVBoxLayout(profiles)
        profile_label = QLabel("Vocabulary &profiles", profiles)
        self._profile_list = QListWidget(profiles)
        self._profile_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.explain(self._profile_list, "vocabulary.profiles")
        profile_label.setBuddy(self._profile_list)
        profiles_layout.addWidget(profile_label)
        profiles_layout.addWidget(self._profile_list)

        self._add_profile_button = self._button(
            "&Add profile...",
            "Add a profile",
            "Opens a dialog for naming a new list of terms and saying how narrowly "
            "it applies.",
            "vocabulary.profiles",
        )
        self._rename_profile_button = self._button(
            "&Rename profile...",
            "Rename the selected profile",
            "Opens a dialog for changing the name and level of the profile selected "
            "in the list.",
            "vocabulary.profiles",
        )
        self._remove_profile_button = self._button(
            "Re&move profile",
            "Remove the selected profile",
            "Removes the selected profile and every term in it. Nothing is written "
            "until you press OK.",
            "vocabulary.profiles",
        )
        profiles_layout.addLayout(
            _button_row(
                self._add_profile_button, self._rename_profile_button, self._remove_profile_button
            )
        )
        layout.addWidget(profiles)

        terms = QGroupBox("Terms", self)
        terms_layout = QVBoxLayout(terms)
        terms_label = QLabel("&Terms in this profile", terms)
        self._terms_model = TermTableModel(self)
        self._terms_view = QTableView(terms)
        self._terms_view.setModel(self._terms_model)
        self._terms_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._terms_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        # Terms are changed in a dialog with a label on every field rather
        # than by typing into cells, so the cells themselves are read-only.
        self._terms_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._terms_view.setAlternatingRowColors(False)
        self._terms_view.setWordWrap(False)
        self._terms_view.horizontalHeader().setStretchLastSection(True)
        self._terms_view.setMinimumHeight(self.fontMetrics().lineSpacing() * (_TERM_ROWS + 2))
        self.explain(self._terms_view, "vocabulary.terms")
        terms_label.setBuddy(self._terms_view)
        terms_layout.addWidget(terms_label)
        terms_layout.addWidget(self._terms_view)

        self._add_term_button = self._button(
            "Add &new term...",
            "Add a term",
            "Opens a dialog for typing a new word or phrase into the selected profile.",
            "vocabulary.terms",
        )
        self._edit_term_button = self._button(
            "&Edit term...",
            "Edit the selected term",
            "Opens the selected term for changing.",
            "vocabulary.terms",
        )
        self._remove_term_button = self._button(
            "Remo&ve term",
            "Remove the selected term",
            "Removes the selected term from this profile. Nothing is written until "
            "you press OK.",
            "vocabulary.terms",
        )
        terms_layout.addLayout(
            _button_row(self._add_term_button, self._edit_term_button, self._remove_term_button)
        )
        layout.addWidget(terms)

        learned = QGroupBox("Learned from your corrections", self)
        learned_layout = QVBoxLayout(learned)
        # A message-carrying label, so it is left unnamed.
        self._learned_label = QLabel(learned)
        self._learned_label.setWordWrap(True)
        learned_layout.addWidget(self._learned_label)
        layout.addWidget(learned)
        layout.addStretch(1)

        self._profile_list.currentRowChanged.connect(self._on_profile_changed)
        self._add_profile_button.clicked.connect(self.add_profile)
        self._rename_profile_button.clicked.connect(self.rename_profile)
        self._remove_profile_button.clicked.connect(self.remove_profile)
        self._add_term_button.clicked.connect(self.add_term)
        self._edit_term_button.clicked.connect(self.edit_term)
        self._remove_term_button.clicked.connect(self.remove_term)

        self.show_vocabulary(self._vocabulary)

    def _button(self, text: str, name: str, description: str, note_key: str) -> QPushButton:
        button = QPushButton(text, self)
        describe(button, name, description)
        # Standing on a button that acts on a list is standing on the list,
        # as far as the explanation panel goes.
        self.note_keys[button] = note_key
        button.setAutoDefault(False)
        return button

    # -- Showing and handing back ------------------------------------------

    def show_vocabulary(self, vocabulary: Vocabulary) -> None:
        self._vocabulary = vocabulary
        self._fill_profile_list(select=0)
        self._show_learned()

    def chosen_vocabulary(self) -> Vocabulary:
        """The vocabulary as the user left it."""
        self._save_terms()
        return self._vocabulary

    def _fill_profile_list(self, select: int) -> None:
        self._profile_list.clear()
        for profile in self._vocabulary.profiles:
            item = QListWidgetItem(_profile_label(profile))
            item.setData(Qt.ItemDataRole.UserRole, profile.id)
            self._profile_list.addItem(item)
        if self._profile_list.count():
            self._profile_list.setCurrentRow(min(select, self._profile_list.count() - 1))
        else:
            self._terms_model.set_terms([])

    def _show_learned(self) -> None:
        count = len(self._vocabulary.corrections.corrections)
        if not count:
            self._learned_label.setText(
                "Nothing has been learned yet. Words you correct while reviewing a "
                "transcript are remembered here, and the ones you correct more than "
                "once are offered to the services automatically."
            )
            return
        counted = "1 correction has" if count == 1 else f"{count} corrections have"
        self._learned_label.setText(
            f"{counted} been learned from your reviewing. The ones you have made more "
            "than once are offered to the services as terms to listen for, along with "
            "what they usually get wrong instead."
        )

    def _selected_profile(self) -> VocabularyProfile | None:
        item = self._profile_list.currentItem()
        if item is None:
            return None
        return self._vocabulary.profile(str(item.data(Qt.ItemDataRole.UserRole)))

    def _on_profile_changed(self, row: int) -> None:
        profile = self._selected_profile()
        if profile is None:
            self._terms_model.set_terms([])
            return
        self._terms_model.set_terms(profile.terms)
        if self._terms_model.rowCount():
            self._terms_view.setCurrentIndex(self._terms_model.index(0, COLUMN_TERM))
        count = self._terms_model.rowCount()
        word = "term" if count == 1 else "terms"
        announce(
            self._profile_list,
            f"{profile.display_name}, {profile.level.display_name} level, {count} {word}.",
        )

    def _save_terms(self) -> None:
        """Copy the table back into the profile it belongs to."""
        profile = self._selected_profile()
        if profile is not None:
            profile.terms = self._terms_model.terms()

    # -- Acting -------------------------------------------------------------

    def add_profile(self) -> None:
        dialog = ProfileDialog(parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._save_terms()
        taken = {profile.id for profile in self._vocabulary.profiles}
        name = dialog.chosen_name()
        level = dialog.chosen_level()
        profile = VocabularyProfile(
            id=_identifier_for(name, level, taken), level=level, name=name
        )
        self._vocabulary.profiles.append(profile)
        self._fill_profile_list(select=len(self._vocabulary.profiles) - 1)
        announce(self._profile_list, f"Added the profile {name}.", urgent=True)

    def rename_profile(self) -> None:
        profile = self._selected_profile()
        if profile is None:
            self._refuse_without_profile()
            return
        dialog = ProfileDialog(name=profile.name, level=profile.level, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._save_terms()
        profile.name = dialog.chosen_name()
        profile.level = dialog.chosen_level()
        row = self._profile_list.currentRow()
        self._fill_profile_list(select=row)
        announce(
            self._profile_list,
            f"Renamed to {profile.display_name}, {profile.level.display_name} level.",
            urgent=True,
        )

    def remove_profile(self) -> None:
        profile = self._selected_profile()
        if profile is None:
            self._refuse_without_profile()
            return
        row = self._profile_list.currentRow()
        self._vocabulary.profiles.remove(profile)
        self._fill_profile_list(select=max(0, row - 1))
        announce(
            self._profile_list,
            f"Removed the profile {profile.display_name}. Press Cancel to get it back.",
            urgent=True,
        )

    def add_term(self) -> None:
        if self._selected_profile() is None:
            self._refuse_without_profile()
            return
        dialog = TermDialog(parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        term = dialog.chosen_term()
        row = self._terms_model.add_term(term)
        self._save_terms()
        self._terms_view.setCurrentIndex(self._terms_model.index(row, COLUMN_TERM))
        announce(self._terms_view, f"Added the term {term.text}.", urgent=True)

    def edit_term(self) -> None:
        row = self._terms_view.currentIndex().row()
        term = self._terms_model.term_at(row)
        if term is None:
            self._refuse_without_term()
            return
        dialog = TermDialog(term=term, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        changed = dialog.chosen_term()
        self._terms_model.replace_term(row, changed)
        self._save_terms()
        announce(self._terms_view, f"Changed the term to {changed.text}.", urgent=True)

    def remove_term(self) -> None:
        row = self._terms_view.currentIndex().row()
        term = self._terms_model.term_at(row)
        if term is None:
            self._refuse_without_term()
            return
        self._terms_model.remove_term(row)
        self._save_terms()
        announce(
            self._terms_view,
            f"Removed the term {term.text}. Press Cancel to get it back.",
            urgent=True,
        )

    def _refuse_without_profile(self) -> None:
        """Say why nothing happened, rather than switching the button off.

        A button that is disabled when there is nothing to act on looks
        tidier and is worse. Disabling a control that holds the focus takes
        the focus with it, and a control that is simply absent from the
        keyboard order gives a user nothing to ask about. So the button
        stays reachable and explains itself.
        """
        announce(
            self._profile_list,
            "No profile is selected. Choose one in the list, or add one first.",
            urgent=True,
        )

    def _refuse_without_term(self) -> None:
        announce(
            self._terms_view,
            "No term is selected. Choose one in the table, or add one first.",
            urgent=True,
        )


def _profile_label(profile: VocabularyProfile) -> str:
    """Name a profile and say its level, because the level is not a column here."""
    return f"{profile.display_name} ({profile.level.display_name} level)"


def _button_row(*buttons: QPushButton) -> QHBoxLayout:
    row = QHBoxLayout()
    for button in buttons:
        row.addWidget(button)
    row.addStretch(1)
    return row


# -- Statistics ------------------------------------------------------------


def _statistics_widget(statistics: Any, parent: QWidget) -> QWidget | None:
    """Build the statistics page, or return nothing if it cannot be built yet.

    The page itself belongs to another part of the application and is
    imported here rather than at the top of the file, on purpose. Settings
    is the one dialog a user opens to fix things, and it holds every API key
    in the application. If the statistics page or anything it depends on
    cannot be imported, the right outcome by a long way is a Settings dialog
    with one page missing and a sentence saying so, not a Settings dialog
    that refuses to open at all and takes every other page down with it.
    """
    try:
        from vox_verbatim.ui.statistics_page import StatisticsPage

        return StatisticsPage(statistics, parent)
    except Exception:  # a page that will not build must not close the dialog
        _log.debug("The statistics page could not be built.", exc_info=True)
        return None


class StatisticsSettingsPage(SettingsPage):
    """What the application has learned about each service, read only.

    There is nothing to set here, which makes it the odd page out. It is in
    Settings because this is where a person comes to find out why the
    application is behaving as it is, and because the numbers explain a
    behaviour that is otherwise invisible: a service you have corrected
    often is believed less readily than one you have not.

    The contents come from another module. This class is the part that makes
    it a page of this dialog: a category, an accessible name, a note for the
    explanation panel, and somewhere to stand if the contents are missing.
    """

    def __init__(self, statistics: Any = None, parent: QWidget | None = None) -> None:
        super().__init__(notes.STATISTICS, parent)
        layout = QVBoxLayout(self)
        self._page = _statistics_widget(statistics, self)
        if self._page is None:
            # A message-carrying label, so it is left unnamed. It says in
            # words what is missing, because an empty page says nothing at
            # all to somebody who cannot see that it is empty.
            missing = QLabel(
                "The service statistics cannot be shown in this build of the "
                "application. Nothing is wrong with your settings, and no other page "
                "is affected.",
                self,
            )
            missing.setWordWrap(True)
            layout.addWidget(missing)
            layout.addStretch(1)
        else:
            layout.addWidget(self._page, 1)
        # Registering the page itself rather than the controls inside it.
        # The panel finds a note by walking up from whatever has the focus,
        # so anything inside this page finds this one note, which is right:
        # the page has one thing to explain and none of its controls is a
        # setting with an explanation of its own.
        self.note_keys[self] = "statistics.accuracy"

    def show_statistics(self, statistics: Any) -> None:
        """Show a fresh set of figures, if there is a page to show them on."""
        if self._page is not None:
            self._page.set_statistics(statistics)


def build_pages(
    vocabulary: Vocabulary | None = None, statistics: Any = None
) -> list[SettingsPage]:
    """Every page, in the order the category list shows them.

    The order matches :data:`vox_verbatim.ui.settings_notes.CATEGORIES`,
    and a test checks that it does, so a page cannot be added here without a
    category and its notes arriving with it.
    """
    return [
        GeneralPage(),
        TranscriptionPage(),
        VocabularyPage(vocabulary),
        StatisticsSettingsPage(statistics),
        CostsPage(),
        ElevenLabsPage(),
        OpenAiTranscriptionPage(),
        OpenAiAdjudicationPage(),
        MicrosoftPage(),
        AssemblyAiPage(),
        DeepgramPage(),
    ]
