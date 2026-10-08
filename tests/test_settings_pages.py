"""Tests for the individual pages of the Settings dialog.

These are about one page at a time: whether it shows what it was given,
whether it hands back exactly that, and whether it refuses what would be
lost if it were accepted. The dialog that holds the pages, and everything
about moving between them, is tested in test_settings_dialog.py.

Several of these are written as loops over every page rather than as a test
per page. That is deliberate. There are eleven pages and there will be more,
and a rule that is checked page by page is a rule that the twelfth page will
quietly not follow.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableView,
)

from vox_verbatim.settings import (
    DEFAULT_ELEVENLABS_TRANSCRIPTION_MODEL,
    DEFAULT_SMOOTHING_PROMPT,
    AssemblyAiSettings,
    CostSettings,
    DeepgramSettings,
    ElevenLabsSettings,
    MicrosoftMaiSettings,
    OpenAiAdjudicationSettings,
    OpenAiTranscriptionSettings,
    ProcessingSettings,
    Settings,
    SmoothingSettings,
    TranscriptionSettings,
)
from vox_verbatim.transcription.vocabulary import (
    TermCategory,
    Vocabulary,
    VocabularyLevel,
    VocabularyProfile,
    VocabularyTerm,
)
from vox_verbatim.ui import settings_notes as notes
from vox_verbatim.ui import settings_pages as pages_module
from vox_verbatim.ui.settings_pages import (
    ProviderPage,
    StatisticsSettingsPage,
    TermTableModel,
    VocabularyPage,
    build_pages,
    parameters_as_text,
    parse_parameters,
)

#: Every kind of widget a user types into or chooses from. Anything on this
#: list needs a name, a description and a label pointing at it.
INPUT_KINDS = (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QTableView,
)


def inputs_of(page) -> list:
    """Every control on a page that a user puts a value into.

    Spin boxes hold a line edit of their own, and combo boxes may hold one
    too. Those are parts of a control rather than controls, and they are
    named by the control around them, so they are left out.
    """
    found = []
    for kind in INPUT_KINDS:
        for widget in page.findChildren(kind):
            if isinstance(widget.parent(), (QSpinBox, QDoubleSpinBox, QComboBox)):
                continue
            found.append(widget)
    return found


@pytest.fixture
def all_pages(qapp):
    built = build_pages()
    yield built
    for page in built:
        page.deleteLater()


# -- The rules that hold on every page ------------------------------------


def test_there_is_a_page_for_every_category_and_nothing_else(all_pages):
    """The list, the pages and the notes have to agree on what exists."""
    assert [page.category for page in all_pages] == list(notes.CATEGORIES)


def test_every_input_on_every_page_has_a_name(all_pages):
    """Written as a loop so that a new page cannot arrive without names.

    An unnamed control is read out by a screen reader as its type and its
    value and nothing else: "edit, blank". The user is then being asked to
    fill in a box whose purpose is invisible to them.
    """
    unnamed = [
        f"{page.category}: {type(widget).__name__}"
        for page in all_pages
        for widget in inputs_of(page)
        if not widget.accessibleName()
    ]

    assert unnamed == []


def test_every_input_on_every_page_says_what_it_is_for(all_pages):
    """A name alone says what a setting is called, not what it does.

    These settings are somebody else's web service. "Chunk overlap" is a
    name; it teaches nobody what to put in the box.

    The statistics page is left out. Its contents belong to another module
    and are read-only rather than settings, so there is nothing there to
    explain how to fill in.
    """
    undescribed = [
        f"{page.category}: {widget.accessibleName()}"
        for page in all_pages
        if not isinstance(page, StatisticsSettingsPage)
        for widget in inputs_of(page)
        if not widget.accessibleDescription()
    ]

    assert undescribed == []


def test_every_input_on_every_page_is_reached_by_its_own_label(all_pages):
    """A label with no buddy names nothing, and its field is read as blank."""
    orphans = []
    for page in all_pages:
        labelled = {label.buddy() for label in page.findChildren(QLabel) if label.buddy()}
        for widget in inputs_of(page):
            # A checkbox carries its own text, so it is its own label.
            if isinstance(widget, QCheckBox):
                continue
            if widget not in labelled:
                orphans.append(f"{page.category}: {widget.accessibleName()}")

    assert orphans == []


def test_every_input_on_every_page_explains_itself_in_the_panel(all_pages):
    """Every control is filed under a note, and every note exists.

    The lookup raises rather than returning nothing, so a key with a typing
    mistake in it fails here rather than reaching a user as a control the
    explanation panel goes blank on.
    """
    for page in all_pages:
        for widget in inputs_of(page):
            key = page.note_keys.get(widget)
            # The statistics page has nothing to set, so its contents are
            # covered by the page's own note rather than one each.
            if key is None and isinstance(page, StatisticsSettingsPage):
                continue
            assert key is not None, f"{page.category}: {widget.accessibleName()} has no note"
            assert notes.note_for(key).title


def test_the_name_and_description_of_a_control_come_from_its_note(all_pages):
    """So the same sentence is never written twice and cannot drift."""
    for page in all_pages:
        for widget, key in page.note_keys.items():
            if not isinstance(widget, INPUT_KINDS):
                continue
            note = notes.note_for(key)
            # The Show box beside an API key is filed under the key, but is
            # a different control and is named for what it does.
            if isinstance(widget, QCheckBox) and "Show" in widget.accessibleName():
                continue
            assert widget.accessibleName() == note.title
            assert widget.accessibleDescription() == note.summary
            # The same words on the tooltip, so a mouse user and a screen
            # reader user are told the same thing.
            assert widget.toolTip() == note.summary


def test_no_two_controls_on_a_page_answer_the_same_alt_key(all_pages):
    """Two controls on one Alt letter means neither is reliably reachable.

    Only within a page, plus the buttons that are always on screen. Qt
    ignores a mnemonic whose control is on a hidden page, so the same letter
    may be reused from one page to the next, which with sixty settings it
    has to be.
    """
    # The category list, the Guide button and the two dialog buttons are
    # visible whichever page is showing. OK and Cancel carry no mnemonic of
    # their own on Windows, so only these two are in the way.
    always_visible = ["c", "g"]
    for page in all_pages:
        letters = list(always_visible)
        for kind in (QLabel, QPushButton, QCheckBox):
            for widget in page.findChildren(kind):
                text = widget.text()
                if "&" in text and not text.endswith("&"):
                    letters.append(text[text.index("&") + 1].casefold())

        assert sorted(letters) == sorted(set(letters)), (
            f"repeated Alt letters on the {page.category} page: {sorted(letters)}"
        )


def test_a_parameter_box_never_traps_the_tab_key(all_pages):
    """A multi-line box that eats Tab leaves a keyboard user stuck in it."""
    boxes = [
        box
        for page in all_pages
        for box in page.findChildren(QPlainTextEdit)
        if box.isEnabled() and not box.isReadOnly()
    ]

    assert boxes, "there should be parameter boxes to check"
    for box in boxes:
        assert box.tabChangesFocus() is True


def test_every_page_hands_back_what_it_was_shown(all_pages):
    """Shown a set of settings, a page must write back exactly those."""
    settings = every_setting_changed()

    changed = Settings()
    for page in all_pages:
        page.show_settings(settings)
        page.apply_to(changed)

    assert changed == settings


# -- The free-form parameter dictionaries ---------------------------------


def test_an_empty_parameter_box_means_no_parameters():
    assert parse_parameters("", "Extra parameters") == ({}, "")
    assert parse_parameters("   \n ", "Extra parameters") == ({}, "")


def test_a_parameter_box_is_read_as_written():
    value, message = parse_parameters('{"temperature": 0, "language": "en"}', "Extra")

    assert value == {"temperature": 0, "language": "en"}
    assert message == ""


def test_broken_json_says_where_it_is_broken():
    value, message = parse_parameters('{"a": 1,}', "Extra parameters")

    assert value is None
    assert "Extra parameters" in message
    assert "line 1" in message and "column" in message
    assert "Nothing you typed has been changed" in message


def test_json_that_is_not_an_object_says_what_it_is_instead():
    """A list is valid JSON and useless as a set of request parameters."""
    value, message = parse_parameters("[1, 2]", "Extra parameters")

    assert value is None
    assert "must be a JSON object" in message
    assert "a list" in message


@pytest.mark.parametrize("text", ["NaN", '{"a": Infinity}', '{"a": -Infinity}'])
def test_the_values_python_reads_but_json_cannot_carry_are_refused(text):
    """Python will read these happily and the settings file will not write them.

    Accepting one here would mean the whole dictionary was thrown away the
    next time the settings were loaded, which is exactly the silent loss
    this box is written to avoid.
    """
    value, message = parse_parameters(text, "Extra parameters")

    assert value is None
    assert "JSON can carry" in message


def test_parameters_are_laid_out_for_reading_and_an_empty_set_is_an_empty_box():
    assert parameters_as_text({}) == ""
    assert parameters_as_text({"b": 1, "a": 2}) == '{\n  "a": 2,\n  "b": 1\n}'


def test_a_broken_parameter_box_is_reported_by_the_page_that_holds_it(all_pages):
    page = _page(all_pages, notes.DEEPGRAM)
    page.show_settings(Settings())
    page._parameters.setPlainText("not json at all")

    problems = page.problems()

    assert [problem.widget for problem in problems] == [page._parameters]
    assert "not valid JSON" in problems[0].message


def test_a_broken_parameter_box_keeps_the_last_value_that_made_sense(all_pages):
    """Nothing else has to cope with half a dictionary while it is being typed."""
    page = _page(all_pages, notes.DEEPGRAM)
    page.show_settings(
        Settings(
            transcription=TranscriptionSettings(deepgram=DeepgramSettings(parameters={"a": 1}))
        )
    )
    page._parameters.setPlainText("{oops")

    settings = Settings()
    page.apply_to(settings)

    assert settings.transcription.deepgram.parameters == {"a": 1}


# -- API keys --------------------------------------------------------------


def test_every_api_key_starts_hidden_and_can_be_shown(all_pages):
    """Hiding is right; being unable to check what you pasted is not.

    A masked box is read out as a row of dots, so without the Show box a
    screen reader user has no way at all to tell a good key from a key with
    half of itself missing.
    """
    provider_pages = [page for page in all_pages if isinstance(page, ProviderPage)]
    assert len(provider_pages) == 6

    for page in provider_pages:
        assert page._key_edit.echoMode() == QLineEdit.EchoMode.Password
        assert page._show_key_box.isChecked() is False
        assert page._show_key_box.accessibleName().startswith("Show the ")

        page._show_key_box.setChecked(True)
        assert page._key_edit.echoMode() == QLineEdit.EchoMode.Normal

        page._show_key_box.setChecked(False)
        assert page._key_edit.echoMode() == QLineEdit.EchoMode.Password


def test_showing_a_key_is_said_out_loud(qapp, monkeypatch):
    """The tick state is not the point; whether the key is on screen is."""
    said: list[str] = []
    monkeypatch.setattr(
        pages_module, "announce", lambda widget, message, urgent=False: said.append(message)
    )
    page = build_pages()[-1]  # Deepgram

    page._show_key_box.setChecked(True)
    page._show_key_box.setChecked(False)

    assert said == [
        "The Deepgram API key is now shown.",
        "The Deepgram API key is now hidden.",
    ]


# -- The smooth transcript page --------------------------------------------


def test_the_smoothing_page_shows_the_saved_values_and_saves_a_change(all_pages):
    page = _page(all_pages, notes.SMOOTHING)
    settings = Settings()
    settings.transcription.smoothing = SmoothingSettings(
        run_after_transcription=False, model="edit-1", reasoning_effort="high", prompt="Be brief."
    )

    page.show_settings(settings)

    assert page._run_box.isChecked() is False
    assert page._model_edit.text() == "edit-1"
    assert page._effort_edit.text() == "high"
    assert page._prompt_box.toPlainText() == "Be brief."

    page._run_box.setChecked(True)
    page._prompt_box.setPlainText("Keep it plain.\n\nNo summaries.")
    changed = Settings()
    page.apply_to(changed)
    assert changed.transcription.smoothing.run_after_transcription is True
    # Kept as typed, line breaks and all.
    assert changed.transcription.smoothing.prompt == "Keep it plain.\n\nNo summaries."


def test_restore_default_prompt_puts_the_default_back_and_says_so(all_pages, monkeypatch):
    said: list[str] = []
    monkeypatch.setattr(
        pages_module, "announce", lambda widget, message, urgent=False: said.append(message)
    )
    page = _page(all_pages, notes.SMOOTHING)
    page.show_settings(Settings())
    page._prompt_box.setPlainText("Something else.")

    page._restore_button.click()

    assert page._prompt_box.toPlainText() == DEFAULT_SMOOTHING_PROMPT
    assert said == ["The default style prompt is back in the box."]


def test_an_empty_smoothing_model_or_prompt_is_refused(all_pages):
    page = _page(all_pages, notes.SMOOTHING)
    page.show_settings(Settings())
    page._model_edit.setText("  ")
    page._prompt_box.setPlainText("  \n ")

    problems = page.problems()

    assert [problem.widget for problem in problems] == [page._model_edit, page._prompt_box]
    assert "model name is empty" in problems[0].message
    assert "style prompt is empty" in problems[1].message


def test_an_empty_smoothing_reasoning_effort_is_accepted(all_pages):
    page = _page(all_pages, notes.SMOOTHING)
    page.show_settings(Settings())
    page._effort_edit.setText("")

    assert page.problems() == []


def test_the_restore_button_is_named_and_explained(all_pages):
    page = _page(all_pages, notes.SMOOTHING)

    assert page._restore_button.accessibleName() == "Restore default prompt"
    assert page._restore_button.accessibleDescription()
    assert page.note_keys[page._restore_button] == "smoothing.prompt"


def test_the_smoothing_notes_say_which_key_it_uses_and_who_adds_the_format():
    assert "OpenAI API key already entered" in notes.note_text(
        "smoothing.run_after_transcription"
    )
    assert "added by the application" in notes.note_text("smoothing.prompt")


# -- Saying whether a service is set up ------------------------------------


def test_a_service_that_is_not_set_up_says_so_in_words(all_pages):
    """Never by colour, an icon or an empty box.

    The wording comes from the settings themselves, so a service that gains
    a required field starts saying so here without this page being touched.
    """
    page = _page(all_pages, notes.ELEVENLABS)
    page.show_settings(Settings())

    message = page._status_label.text()

    assert "is not set up" in message
    assert ElevenLabsSettings.requirements in message


def test_a_service_that_is_set_up_says_that_too(all_pages):
    page = _page(all_pages, notes.ELEVENLABS)
    page.show_settings(
        Settings(
            transcription=TranscriptionSettings(elevenlabs=ElevenLabsSettings(api_key="k"))
        )
    )

    assert page._status_label.text() == "ElevenLabs Scribe is switched on and is set up."


def test_a_service_that_is_switched_off_says_that_rather_than_complaining(all_pages):
    page = _page(all_pages, notes.DEEPGRAM)
    page.show_settings(Settings())

    assert "is switched off" in page._status_label.text()


def test_the_message_changes_as_the_key_is_typed(all_pages):
    """The moment a key is pasted, not the next time the page is opened."""
    page = _page(all_pages, notes.DEEPGRAM)
    page.show_settings(Settings())
    page._enabled_box.setChecked(True)
    assert "is not set up" in page._status_label.text()

    page._key_edit.setText("dg-key")

    assert page._status_label.text() == "Deepgram is switched on and is set up."


def test_the_transcription_page_lists_what_would_stop_a_run(all_pages):
    """In the settings' own sentences, so the rule is stated in one place."""
    page = _page(all_pages, notes.TRANSCRIPTION)
    transcription = TranscriptionSettings()

    page.show_requirements(transcription)

    shown = page._requirements_label.text()
    assert transcription.missing_requirements()
    for sentence in transcription.missing_requirements():
        assert sentence in shown


def test_the_transcription_page_says_when_nothing_is_missing(all_pages):
    page = _page(all_pages, notes.TRANSCRIPTION)
    ready = TranscriptionSettings(
        elevenlabs=ElevenLabsSettings(api_key="k"),
        openai_transcription=OpenAiTranscriptionSettings(api_key="k"),
        openai_adjudication=OpenAiAdjudicationSettings(api_key="k"),
        # The optional services are switched off rather than given keys,
        # because a service nobody asked for is not something missing.
        microsoft=MicrosoftMaiSettings(enabled=False),
        assemblyai=AssemblyAiSettings(enabled=False),
        deepgram=DeepgramSettings(enabled=False),
    )

    page.show_requirements(ready)

    assert "Everything a run needs is set up" in page._requirements_label.text()


# -- Values that would be lost if they were accepted -----------------------


def test_a_model_name_cannot_be_left_empty(all_pages):
    """An empty name is replaced by the default on the next load.

    Accepting it would mean the user cleared a box, pressed OK, and found
    the old value back the next time they looked.
    """
    page = _page(all_pages, notes.ELEVENLABS)
    page.show_settings(Settings())
    page._model_edit.setText("   ")

    problems = page.problems()

    assert [problem.widget for problem in problems] == [page._model_edit]
    assert "empty" in problems[0].message
    # And the default really would have come back, which is why it is refused.
    assert (
        ElevenLabsSettings.from_dict({"transcription_model": ""}).transcription_model
        == DEFAULT_ELEVENLABS_TRANSCRIPTION_MODEL
    )


@pytest.mark.parametrize(
    ("category", "field"),
    [
        (notes.ELEVENLABS, "_model_edit"),
        (notes.OPENAI_TRANSCRIPTION, "_model_edit"),
        (notes.OPENAI_ADJUDICATION, "_model_edit"),
        (notes.MICROSOFT, "_model_edit"),
        (notes.MICROSOFT, "_version_edit"),
        (notes.ASSEMBLYAI, "_primary_edit"),
        (notes.ASSEMBLYAI, "_afrikaans_edit"),
        (notes.DEEPGRAM, "_model_edit"),
    ],
)
def test_an_empty_name_on_a_service_that_is_switched_off_does_not_stop_saving(
    all_pages, category, field
):
    """A service that is switched off is never called, so its names do not matter.

    Refusing here would leave somebody who cleared a box on a service they
    do not use unable to save until they typed a name they do not need.
    """
    page = _page(all_pages, category)
    page.show_settings(Settings())
    page._enabled_box.setChecked(False)
    getattr(page, field).setText("")

    assert page.problems() == []


@pytest.mark.parametrize(
    ("category", "field"),
    [
        (notes.ELEVENLABS, "_model_edit"),
        (notes.OPENAI_TRANSCRIPTION, "_model_edit"),
        (notes.OPENAI_ADJUDICATION, "_model_edit"),
        (notes.MICROSOFT, "_model_edit"),
        (notes.MICROSOFT, "_version_edit"),
        (notes.ASSEMBLYAI, "_primary_edit"),
        (notes.ASSEMBLYAI, "_afrikaans_edit"),
        (notes.DEEPGRAM, "_model_edit"),
    ],
)
def test_the_same_empty_name_on_a_service_that_is_switched_on_still_stops_saving(
    all_pages, category, field
):
    page = _page(all_pages, category)
    page.show_settings(Settings())
    page._enabled_box.setChecked(True)
    edit = getattr(page, field)
    edit.setText("")

    problems = page.problems()

    assert [problem.widget for problem in problems] == [edit]
    assert "empty" in problems[0].message


def test_broken_json_on_a_service_that_is_switched_off_is_still_reported(all_pages):
    """Unlike an empty name, text that is not JSON is lost on saving."""
    page = _page(all_pages, notes.DEEPGRAM)
    page.show_settings(Settings())
    page._enabled_box.setChecked(False)
    page._parameters.setPlainText("{not json")

    problems = page.problems()

    assert [problem.widget for problem in problems] == [page._parameters]


def test_an_empty_reasoning_effort_is_a_real_answer_rather_than_a_mistake(all_pages):
    """Empty means "leave the parameter out", which some models require."""
    page = _page(all_pages, notes.OPENAI_ADJUDICATION)
    page.show_settings(Settings())
    page._effort_edit.setText("")

    assert page.problems() == []

    settings = Settings()
    page.apply_to(settings)
    assert settings.transcription.openai_adjudication.reasoning_effort == ""


def test_an_empty_azure_endpoint_is_reported_rather_than_refused(all_pages):
    """A key with no endpoint has nowhere to go, and a wrong guess is worse.

    So the page says the service is not set up and names the endpoint among
    what it needs, rather than refusing to close over a box that somebody
    with no Azure resource has every reason to leave empty.
    """
    page = _page(all_pages, notes.MICROSOFT)
    page.show_settings(Settings())
    page._key_edit.setText("ms-key")

    assert page.problems() == []
    assert "is not set up" in page._status_label.text()
    assert "resource endpoint" in page._status_label.text()


@pytest.mark.parametrize("suffix", [".trans/cript", "..", "a..b", 'x"y', "C:evil"])
def test_a_folder_name_ending_that_would_move_the_transcripts_is_refused(all_pages, suffix):
    """It becomes part of a folder name, so it decides where files land."""
    page = _page(all_pages, notes.TRANSCRIPTION)
    page.show_settings(Settings())
    page._suffix_edit.setText(suffix)

    problems = page.problems()

    assert [problem.widget for problem in problems] == [page._suffix_edit]
    assert suffix in problems[0].message


def test_an_ordinary_folder_name_ending_is_accepted(all_pages):
    page = _page(all_pages, notes.TRANSCRIPTION)
    page.show_settings(Settings())
    page._suffix_edit.setText(".out")

    assert page.problems() == []


# -- The vocabulary page ---------------------------------------------------


def a_vocabulary() -> Vocabulary:
    vocabulary = Vocabulary(
        profiles=[
            VocabularyProfile(
                id="global-main",
                level=VocabularyLevel.GLOBAL,
                name="Everything",
                terms=[
                    VocabularyTerm(
                        text="Kotze",
                        category=TermCategory.PERSON,
                        common_misrecognitions=("cause a",),
                        confirmation_count=4,
                    )
                ],
            ),
            VocabularyProfile(id="client-acme", level=VocabularyLevel.CLIENT, name="Acme"),
        ]
    )
    vocabulary.corrections.record("cause a", "Kotze")
    return vocabulary


def test_the_profile_list_says_the_level_rather_than_showing_it(qapp):
    """Level is not a colour or a position here, it is part of the text."""
    page = VocabularyPage(a_vocabulary())

    shown = [page._profile_list.item(row).text() for row in range(page._profile_list.count())]

    assert shown == ["Everything (Global level)", "Acme (Client level)"]


def test_the_terms_table_says_something_in_every_cell(qapp):
    """An empty cell is silence to a screen reader, and reads as the end."""
    page = VocabularyPage(a_vocabulary())
    model = page._terms_model

    row = [model.data(model.index(0, column)) for column in range(model.columnCount())]

    assert row == ["Kotze", "Person", "Any language", "cause a", "4"]
    assert all(text for text in row)


def test_the_terms_table_names_its_columns(qapp):
    page = VocabularyPage(a_vocabulary())
    model = page._terms_model

    headings = [
        model.headerData(column, Qt.Orientation.Horizontal)
        for column in range(model.columnCount())
    ]

    assert headings == ["Term", "Category", "Language", "Usually heard as", "Times confirmed"]


def test_choosing_a_profile_shows_its_terms(qapp):
    page = VocabularyPage(a_vocabulary())

    page._profile_list.setCurrentRow(1)

    assert page._terms_model.rowCount() == 0

    page._profile_list.setCurrentRow(0)

    assert page._terms_model.rowCount() == 1


def test_a_term_can_be_added_edited_and_removed(qapp, monkeypatch):
    page = VocabularyPage(a_vocabulary())

    _accept_term_dialog(monkeypatch, "Van der Merwe")
    page.add_term()
    assert [term.text for term in page.chosen_vocabulary().profiles[0].terms] == [
        "Kotze",
        "Van der Merwe",
    ]

    _accept_term_dialog(monkeypatch, "Van der Merwe (SA)")
    page.edit_term()
    assert page.chosen_vocabulary().profiles[0].terms[1].text == "Van der Merwe (SA)"

    page.remove_term()
    assert [term.text for term in page.chosen_vocabulary().profiles[0].terms] == ["Kotze"]


def test_editing_a_term_keeps_what_a_person_confirmed(qapp, monkeypatch):
    """The count is evidence from real transcripts, not something to retype.

    It decides which terms survive when a service will not take them all, so
    losing it on a spelling change would quietly weaken the list.
    """
    page = VocabularyPage(a_vocabulary())
    _accept_term_dialog(monkeypatch, "Kotzé")

    page._terms_view.setCurrentIndex(page._terms_model.index(0, 0))
    page.edit_term()

    term = page.chosen_vocabulary().profiles[0].terms[0]
    assert term.text == "Kotzé"
    assert term.confirmation_count == 4
    assert term.common_misrecognitions == ("cause a",)


def test_a_term_with_no_text_is_refused_out_loud(qapp, monkeypatch):
    said: list[str] = []
    monkeypatch.setattr(
        pages_module, "announce", lambda widget, message, urgent=False: said.append(message)
    )
    dialog = pages_module.TermDialog()

    dialog.accept()

    assert dialog.result() != QDialog.DialogCode.Accepted
    assert said and "needs some text" in said[0]
    assert "needs some text" in dialog._message_label.text()


def test_a_profile_can_be_added_and_removed(qapp, monkeypatch):
    page = VocabularyPage(a_vocabulary())
    monkeypatch.setattr(
        pages_module.ProfileDialog,
        "exec",
        lambda self: (
            self._name_edit.setText("Bosch"),
            self._level_box.setCurrentIndex(self._level_box.findData(VocabularyLevel.SPEAKER)),
            QDialog.DialogCode.Accepted,
        )[2],
    )

    page.add_profile()

    profiles = page.chosen_vocabulary().profiles
    assert [profile.name for profile in profiles] == ["Everything", "Acme", "Bosch"]
    assert profiles[-1].level is VocabularyLevel.SPEAKER
    assert profiles[-1].id == "speaker-bosch"

    page.remove_profile()
    assert [profile.name for profile in page.chosen_vocabulary().profiles] == [
        "Everything",
        "Acme",
    ]


def test_acting_with_nothing_selected_says_why_rather_than_going_quiet(qapp, monkeypatch):
    """The buttons stay reachable instead of being switched off.

    Disabling a control that holds the focus takes the focus with it, and a
    control that is simply absent from the keyboard order gives the user
    nothing to ask about.
    """
    said: list[str] = []
    monkeypatch.setattr(
        pages_module, "announce", lambda widget, message, urgent=False: said.append(message)
    )
    page = VocabularyPage(Vocabulary())

    assert page._add_term_button.isEnabled()
    page.add_term()
    page.edit_term()

    assert "No profile is selected. Choose one in the list, or add one first." in said


def test_the_terms_table_is_not_edited_in_its_cells(qapp):
    """Terms are changed in a dialog where every field has a label."""
    page = VocabularyPage(a_vocabulary())

    assert page._terms_view.editTriggers() == QTableView.EditTrigger.NoEditTriggers


def test_a_term_model_with_no_terms_reports_nothing_rather_than_failing(qapp):
    model = TermTableModel()

    assert model.rowCount() == 0
    assert model.term_at(0) is None
    assert model.data(model.index(0, 0)) is None


# -- The statistics page ---------------------------------------------------


def test_the_statistics_page_is_shown_when_it_can_be_built(all_pages):
    page = _page(all_pages, notes.STATISTICS)

    assert isinstance(page, StatisticsSettingsPage)
    assert page._page is not None


def test_a_statistics_page_that_cannot_be_built_says_so_in_words(qapp, monkeypatch):
    """One page missing must not take the whole Settings dialog down.

    Settings is where a person comes to fix things, and it holds every API
    key in the application. A page that will not import has to cost that
    page and nothing else.
    """
    monkeypatch.setattr(
        pages_module, "_statistics_widget", lambda statistics, parent: None
    )
    page = StatisticsSettingsPage()

    said = [label.text() for label in page.findChildren(QLabel) if label.text()]

    assert any("cannot be shown" in text for text in said)
    assert any("no other page is affected" in text.lower() for text in said)


# -- Helpers ---------------------------------------------------------------


def _page(built, category: str):
    return next(page for page in built if page.category == category)


def _accept_term_dialog(monkeypatch, text: str) -> None:
    monkeypatch.setattr(
        pages_module.TermDialog,
        "exec",
        lambda self: (self._text_edit.setText(text), QDialog.DialogCode.Accepted)[1],
    )


def every_setting_changed() -> Settings:
    """A Settings object in which nothing is left at its default.

    Every field is given a value of its own, so that a page which reads the
    right field and writes the wrong one cannot pass by accident.
    """
    return Settings(
        reopen_last_folder=False,
        short_skip_seconds=7,
        medium_skip_seconds=95,
        long_skip_seconds=444,
        transcription=TranscriptionSettings(
            openai_transcription=OpenAiTranscriptionSettings(
                api_key="ot-key",
                model="transcribe-1",
                parameters={"language": "en"},
                chunk_target_bytes=12_000_000,
                enabled=False,
            ),
            openai_adjudication=OpenAiAdjudicationSettings(
                api_key="oa-key",
                model="reason-1",
                reasoning_effort="high",
                parameters={"max_output_tokens": 2000},
                enabled=False,
            ),
            elevenlabs=ElevenLabsSettings(
                api_key="el-key",
                transcription_model="scribe-9",
                transcription_parameters={"num_speakers": 3},
                forced_alignment_parameters={"language_code": "de"},
                diarise=False,
                tag_audio_events=False,
                enabled=False,
            ),
            microsoft=MicrosoftMaiSettings(
                api_key="ms-key",
                endpoint="https://example.cognitiveservices.azure.com",
                model="mai-9",
                api_version="2030-01-01",
                parameters={"locales": ["en-GB"]},
                enabled=True,
            ),
            assemblyai=AssemblyAiSettings(
                api_key="aa-key",
                primary_model="universal-9",
                afrikaans_model="universal-1",
                parameters={"punctuate": True},
                enabled=False,
            ),
            deepgram=DeepgramSettings(
                api_key="dg-key",
                model="nova-9",
                parameters={"smart_format": {"nested": 1}},
                enabled=True,
            ),
            processing=ProcessingSettings(
                default_afrikaans_enabled=True,
                default_expected_speaker_count=6,
                escalation_context_seconds_before=3.5,
                escalation_context_seconds_after=7.5,
                provider_timeout_seconds=123.0,
                provider_retry_attempts=7,
                provider_retry_backoff_seconds=9.5,
                provider_chunk_overlap_seconds=4.5,
                forced_alignment_enabled=False,
                escalation_enabled=False,
                adjudication_enabled=False,
                maximum_escalations_per_recording=1234,
                transcript_folder_suffix=".out",
            ),
            cost=CostSettings(
                elevenlabs_per_minute=0.0123,
                openai_transcription_per_minute=0.0456,
                microsoft_per_minute=0.5,
                assemblyai_per_minute=0.001,
                deepgram_per_minute=0.0099,
                adjudication_per_request=1.25,
                smoothing_per_request=0.75,
                confirm_before_running=False,
            ),
            smoothing=SmoothingSettings(
                run_after_transcription=False,
                model="edit-9",
                reasoning_effort="medium",
                prompt="Edit gently.\n\nKeep every name.",
            ),
        ),
    )
