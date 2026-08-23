"""Tests for the Settings dialog and the way the window uses it.

The dialog is a category list beside a stack of pages. That pattern asks
more of a screen reader than tabs do, because nothing in Qt says the two
controls are related, so a good deal of what is tested here is the work that
makes the relationship explicit: what is announced when a category is
chosen, what each page is called, and where the focus goes.

The pages themselves are tested in test_settings_pages.py.
"""

from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QTableView,
)

from vox_verbatim.session import SessionStore
from vox_verbatim.settings import (
    SETTINGS_FILE_NAME,
    ElevenLabsSettings,
    Settings,
    SettingsStore,
    TranscriptionSettings,
)
from vox_verbatim.transcription.vocabulary import (
    Vocabulary,
    VocabularyLevel,
    VocabularyProfile,
    VocabularyTerm,
)
from vox_verbatim.ui import settings_dialog as settings_dialog_module
from vox_verbatim.ui import settings_notes as notes
from vox_verbatim.ui import settings_pages as settings_pages_module
from vox_verbatim.ui.help_dialogs import keyboard_shortcuts_text
from vox_verbatim.ui.main_window import MainWindow
from vox_verbatim.ui.settings_dialog import SettingsDialog
from tests.conftest import wait_until, write_fake_audio
from tests.test_settings_pages import every_setting_changed, inputs_of


def general_page(dialog: SettingsDialog):
    return dialog.page(notes.GENERAL)


def test_the_dialog_opens_showing_the_settings_in_force(qapp):
    settings = Settings(
        reopen_last_folder=False,
        short_skip_seconds=10,
        medium_skip_seconds=45,
        long_skip_seconds=200,
    )

    dialog = SettingsDialog(settings)

    page = general_page(dialog)
    assert page._reopen_box.isChecked() is False
    assert page._short_box.value() == 10
    assert page._medium_box.value() == 45
    assert page._long_box.value() == 200


def test_the_dialog_hands_back_what_the_user_changed(qapp):
    dialog = SettingsDialog(Settings())
    page = general_page(dialog)

    page._reopen_box.setChecked(False)
    page._short_box.setValue(30)

    chosen = dialog.chosen_settings()
    assert chosen.reopen_last_folder is False
    assert chosen.short_skip_seconds == 30
    # Anything untouched keeps the value it had.
    assert chosen.medium_skip_seconds == 120


def test_the_dialog_refuses_a_skip_outside_what_the_buttons_are_for(qapp):
    dialog = SettingsDialog(Settings())
    page = general_page(dialog)

    page._short_box.setValue(0)
    assert page._short_box.value() >= 1

    page._long_box.setValue(999_999)
    assert page._long_box.value() <= 3600


def test_the_dialog_starts_on_the_category_list_rather_than_the_ok_button(qapp):
    """Deliberately changed from starting on the first setting.

    With one flat page, the first setting was the top of the dialog. With
    eleven pages it is not: the categories are, and landing there means a
    screen reader reads out where the user is and what their choices are
    before anything else happens. Landing on a setting would put them in the
    middle of a structure they had not been told about.
    """
    dialog = SettingsDialog(Settings())
    dialog.show()

    assert dialog.focusWidget() is dialog._categories
    assert dialog._categories.currentRow() == 0


def test_new_intervals_reach_the_buttons_the_menu_and_the_help(qapp, tmp_path):
    """A changed interval must change everything that quotes it."""
    window = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        window.apply_settings(Settings(short_skip_seconds=30, medium_skip_seconds=90))

        forward_short = window._player_panel._forward_buttons[0]
        assert forward_short.text() == "+ 30 sec"
        assert forward_short.accessibleName() == "Forward 30 seconds"

        back_medium = [
            action.text() for action, interval, direction in window._skip_actions
            if interval == 1 and direction == -1
        ]
        assert back_medium == ["Back 1 minute 30 seconds"]

        assert "Back 30 seconds" in keyboard_shortcuts_text(window.settings)
    finally:
        window.close()


def test_a_changed_interval_is_how_far_the_button_moves(qapp, tmp_path):
    window = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        requested: list[int] = []
        window._player_panel.skipRequested.connect(requested.append)
        # The transport controls are switched off until a file is loaded,
        # and a switched-off button does nothing when it is clicked.
        window._player_panel.set_media_loaded(True)
        window.apply_settings(Settings(short_skip_seconds=30))

        window._player_panel._forward_buttons[0].click()
        window._player_panel._back_buttons[-1].click()

        assert requested == [30_000, -30_000]
    finally:
        window.close()


def test_settings_are_saved_beside_the_session(qapp, tmp_path):
    window = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        window.apply_settings(Settings(short_skip_seconds=25))
    finally:
        window.close()

    assert SettingsStore(tmp_path / SETTINGS_FILE_NAME).load().short_skip_seconds == 25


def test_settings_survive_a_restart(qapp, tmp_path):
    store = SessionStore(tmp_path / "session.json")
    window = MainWindow(store)
    try:
        window.apply_settings(Settings(reopen_last_folder=False, long_skip_seconds=42))
    finally:
        window.close()

    reopened = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        assert reopened.settings.reopen_last_folder is False
        assert reopened.settings.long_skip_seconds == 42
        assert reopened._player_panel._back_buttons[0].accessibleName() == "Back 42 seconds"
    finally:
        reopened.close()


def test_switching_off_reopening_leaves_the_folder_closed_on_start(qapp, tmp_path):
    """The folder is still remembered; it is simply not opened again."""
    folder = tmp_path / "recordings"
    folder.mkdir()
    write_fake_audio(folder / "one.m4a")

    window = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        window._folder_panel.folderChosen.emit(str(folder))
        assert wait_until(qapp, lambda: window._model.rowCount() == 1)
        window.apply_settings(Settings(reopen_last_folder=False))
    finally:
        window.close()

    reopened = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        assert reopened._model.rowCount() == 0
        assert "switched off in the settings" in reopened._status_label.text()
    finally:
        reopened.close()

    # Closing a run that never opened a folder must not throw the remembered
    # one away. The check has to come after that window has closed and
    # written its session out, which is where it would be lost.
    third_run = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        assert third_run._session.folder == str(folder)
    finally:
        third_run.close()

    # And with reopening switched back on, the folder really does come back.
    third_run_settings = SettingsStore(tmp_path / SETTINGS_FILE_NAME)
    third_run_settings.save(Settings(reopen_last_folder=True))
    fourth_run = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        assert wait_until(qapp, lambda: fourth_run._model.rowCount() == 1)
    finally:
        fourth_run.close()


def test_a_folder_that_is_not_connected_keeps_its_checked_files(qapp, tmp_path):
    """A disconnected drive must not clear the marks the user made.

    The file list cannot be read, so it says nothing about which files were
    checked. Saving it as empty would throw the marks away for good.
    """
    folder = tmp_path / "recordings"
    folder.mkdir()
    write_fake_audio(folder / "one.m4a")
    write_fake_audio(folder / "two.m4a")

    window = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        window._folder_panel.folderChosen.emit(str(folder))
        assert wait_until(qapp, lambda: window._model.rowCount() == 2)
        window._model.set_checked_names(["two.m4a"])
    finally:
        window.close()

    # The drive goes away, and the application is opened and closed again.
    for child in folder.iterdir():
        child.unlink()
    folder.rmdir()
    while_disconnected = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        assert "not available" in while_disconnected._status_label.text()
    finally:
        while_disconnected.close()

    # The drive comes back.
    folder.mkdir()
    write_fake_audio(folder / "one.m4a")
    write_fake_audio(folder / "two.m4a")
    reconnected = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        assert wait_until(qapp, lambda: reconnected._model.rowCount() == 2)
        assert reconnected._model.checked_names() == ["two.m4a"]
    finally:
        reconnected.close()


def test_the_log_file_is_reported_rather_than_opened_when_it_is_missing(qapp, tmp_path):
    window = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        opened: list[str] = []
        window._open_with_windows = lambda path, description: opened.append(description)

        window.open_log_file()

        # There is no log file in a test run, so the user is told where one
        # would appear rather than being handed an error from Windows.
        assert opened == []
        assert "no log file yet" in window._status_label.text()
    finally:
        window.close()


def test_the_settings_folder_that_opens_is_the_one_actually_in_use(qapp, tmp_path):
    window = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        asked_for: list = []
        window._open_with_windows = lambda path, description: asked_for.append(path)

        window.open_settings_folder()

        assert asked_for == [tmp_path]
    finally:
        window.close()


def test_cancelling_the_dialog_changes_nothing(qapp, tmp_path, monkeypatch):
    window = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        before = window.settings
        monkeypatch.setattr(SettingsDialog, "exec", lambda self: QDialog.DialogCode.Rejected)

        window.show_settings()

        assert window.settings == before
        assert not (tmp_path / SETTINGS_FILE_NAME).exists()
    finally:
        window.close()


def test_accepting_the_dialog_applies_what_was_chosen(qapp, tmp_path, monkeypatch):
    window = MainWindow(SessionStore(tmp_path / "session.json"))
    try:
        def accept_with_a_longer_skip(dialog):
            general_page(dialog)._short_box.setValue(20)
            return QDialog.DialogCode.Accepted

        monkeypatch.setattr(SettingsDialog, "exec", accept_with_a_longer_skip)

        window.show_settings()

        assert window.settings.short_skip_seconds == 20
        assert window._player_panel._forward_buttons[0].text() == "+ 20 sec"
    finally:
        window.close()


# -- Moving between the categories ----------------------------------------


def test_every_category_has_a_page_and_every_page_is_named_for_it(qapp):
    """A stacked page is a place the user arrives at, so it needs a name.

    With tabs, the tab names the page. Here the list and the stack are two
    unrelated controls, so the page has to name itself or a user who tabs
    into it is told only that they are in a group of controls.
    """
    dialog = SettingsDialog(Settings())
    try:
        shown = [
            dialog._categories.item(row).text() for row in range(dialog._categories.count())
        ]
        assert shown == list(notes.CATEGORIES)

        for row, page in enumerate(dialog.pages):
            dialog._categories.setCurrentRow(row)
            assert dialog._stack.currentWidget().widget() is page
            assert page.accessibleName() == f"{page.category} settings"
            assert page.accessibleDescription() == notes.CATEGORY_SUMMARIES[page.category]
    finally:
        dialog.close()


def test_choosing_a_category_says_which_page_is_now_showing(qapp, monkeypatch):
    """The sentence that pays for choosing a list and a stack over tabs.

    Nothing in Qt ties the two controls together, so without this a user
    arrowing down the list hears the name of a list item and has no
    confirmation that anything happened on the other side of the dialog.
    """
    said: list[str] = []
    monkeypatch.setattr(
        settings_dialog_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    dialog = SettingsDialog(Settings())
    try:
        dialog.show_category(notes.COSTS)

        assert said[-1] == "Showing Costs settings."
    finally:
        dialog.close()


def test_arriving_on_a_service_says_whether_it_is_set_up(qapp, monkeypatch):
    """It is what somebody wants to know on arriving at the page.

    The alternative is tabbing through every control to work it out, and a
    control that is empty because nothing was ever typed reads exactly like
    a control that is empty because something went wrong.
    """
    said: list[str] = []
    monkeypatch.setattr(
        settings_dialog_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )
    dialog = SettingsDialog(Settings())
    try:
        dialog.show_category(notes.ELEVENLABS)
        assert said[-1] == (
            "Showing ElevenLabs settings. ElevenLabs Scribe is switched on but is "
            "not set up. It needs an API key and a transcription model name."
        )

        dialog.page(notes.ELEVENLABS)._key_edit.setText("el-key")
        dialog.show_category(notes.GENERAL)
        dialog.show_category(notes.ELEVENLABS)

        assert said[-1] == (
            "Showing ElevenLabs settings. ElevenLabs Scribe is switched on and is "
            "set up."
        )
    finally:
        dialog.close()


def test_the_transcription_page_answers_for_the_pages_beside_it(qapp):
    """What would stop a run depends on every service page at once.

    So the answer is worked out again from the settings as they stand each
    time the page is arrived at, rather than from the ones the dialog was
    opened with. Typing a key on one page changes what another page says.
    """
    dialog = SettingsDialog(Settings())
    try:
        page = dialog.page(notes.TRANSCRIPTION)
        dialog.show_category(notes.TRANSCRIPTION)
        assert "ElevenLabs Scribe is switched on but is not set up" in (
            page._requirements_label.text()
        )

        dialog.show_category(notes.ELEVENLABS)
        dialog.page(notes.ELEVENLABS)._key_edit.setText("el-key")
        dialog.show_category(notes.TRANSCRIPTION)

        assert "ElevenLabs Scribe is switched on but is not set up" not in (
            page._requirements_label.text()
        )
        # And the services that are still missing are still named.
        assert "OpenAI transcription is switched on but is not set up" in (
            page._requirements_label.text()
        )
    finally:
        dialog.close()


# -- Getting about with the keyboard --------------------------------------


def test_tab_leads_from_the_categories_through_the_page_to_the_buttons(qapp):
    """The order through the dialog must be the order it reads."""
    dialog = SettingsDialog(Settings())
    try:
        dialog.show()
        dialog.show_category(notes.GENERAL)
        dialog._categories.setFocus(Qt.FocusReason.TabFocusReason)

        landed = []
        for _ in range(9):
            landed.append(dialog.focusWidget())
            dialog.focusNextChild()

        page = general_page(dialog)
        assert landed == [
            dialog._categories,
            page._reopen_box,
            page._short_box,
            page._medium_box,
            page._long_box,
            dialog._notes_text,
            dialog._guide_button,
            dialog._ok_button,
            dialog._cancel_button,
        ]
        # And round again, rather than stopping at the end.
        assert dialog.focusWidget() is dialog._categories
    finally:
        dialog.close()


def test_every_control_on_every_page_can_be_reached_with_tab_alone(qapp):
    """Written as a loop, so a new page cannot arrive with a control stranded.

    A control that only the mouse can reach is a control that does not
    exist, for the user this application is written for.
    """
    dialog = SettingsDialog(Settings())
    try:
        dialog.show()
        for row, page in enumerate(dialog.pages):
            dialog._categories.setCurrentRow(row)
            dialog._categories.setFocus(Qt.FocusReason.TabFocusReason)

            reached = set()
            for _ in range(40):
                dialog.focusNextChild()
                widget = dialog.focusWidget()
                if widget is dialog._categories:
                    break
                reached.add(widget)

            missed = [
                control.accessibleName()
                for control in inputs_of(page)
                if control not in reached
            ]
            assert missed == [], f"unreachable on the {page.category} page: {missed}"
    finally:
        dialog.close()


def test_the_scrolling_areas_are_not_stops_on_the_way_through(qapp):
    """A container that answers no keys must not take the focus."""
    dialog = SettingsDialog(Settings())
    try:
        for scroller in dialog._scrollers.values():
            assert scroller.focusPolicy() == Qt.FocusPolicy.NoFocus
        assert dialog._panes.focusPolicy() == Qt.FocusPolicy.NoFocus
    finally:
        dialog.close()


def test_the_pages_scroll_rather_than_pushing_the_buttons_off_the_screen(qapp):
    """At a large text size a provider page is taller than a laptop screen.

    Growing the dialog to fit would put OK and Cancel out of reach, so the
    page scrolls and the buttons stay where they were.
    """
    dialog = SettingsDialog(Settings())
    try:
        dialog.show()

        assert dialog.height() <= dialog._room_to_grow()
        for page, scroller in dialog._scrollers.items():
            assert scroller.widget() is page
            assert scroller.widgetResizable() is True
        assert not dialog._stack.isAncestorOf(dialog._ok_button)
        assert not dialog._stack.isAncestorOf(dialog._cancel_button)
        assert not dialog._stack.isAncestorOf(dialog._notes_text)
    finally:
        dialog.close()


# -- The panel that explains the setting you are standing on ---------------


def test_the_note_panel_follows_the_focus_onto_every_setting(qapp):
    """Including the spin boxes, which hand the focus to a field inside them."""
    dialog = SettingsDialog(Settings())
    try:
        dialog.show()
        for row, page in enumerate(dialog.pages):
            dialog._categories.setCurrentRow(row)
            for widget, key in page.note_keys.items():
                if not widget.isVisible() or widget.focusPolicy() == Qt.FocusPolicy.NoFocus:
                    continue
                widget.setFocus(Qt.FocusReason.TabFocusReason)
                qapp.processEvents()

                assert dialog._showing_note == key, (
                    f"{page.category}: {widget.accessibleName()} shows the wrong note"
                )
                assert dialog._notes_text.toPlainText() == notes.note_text(key)
                assert notes.note_for(key).title in dialog._notes_group.title()
    finally:
        dialog.close()


def test_the_note_panel_covers_a_page_that_has_no_settings_of_its_own(qapp):
    """The statistics page has one thing to explain rather than one per control."""
    dialog = SettingsDialog(Settings())
    try:
        dialog.show()
        dialog.show_category(notes.STATISTICS)
        page = dialog.page(notes.STATISTICS)
        table = page.findChildren(QTableView)[0]

        table.setFocus(Qt.FocusReason.TabFocusReason)
        qapp.processEvents()

        assert dialog._showing_note == "statistics.accuracy"
    finally:
        dialog.close()


def test_reading_a_note_does_not_change_the_note(qapp):
    """The panel takes focus so it can be read, and must hold still when it does."""
    dialog = SettingsDialog(Settings())
    try:
        dialog.show()
        dialog.show_category(notes.OPENAI_ADJUDICATION)
        effort = dialog.page(notes.OPENAI_ADJUDICATION)._effort_edit
        effort.setFocus(Qt.FocusReason.TabFocusReason)
        qapp.processEvents()

        dialog._notes_text.setFocus(Qt.FocusReason.TabFocusReason)
        qapp.processEvents()

        assert dialog._showing_note == "openai_adjudication.reasoning_effort"
    finally:
        dialog.close()


def test_a_closed_dialog_stops_watching_the_focus(qapp):
    """Otherwise every opening of Settings leaves a watcher behind for good."""
    dialogs = [SettingsDialog(Settings()) for _ in range(3)]
    try:
        for dialog in dialogs:
            dialog.show()
            qapp.processEvents()
        assert [dialog._watching_focus for dialog in dialogs] == [True, True, True]

        # reject is what Cancel and Escape both do, and what exec returns
        # through, and it hides without raising a close event.
        for dialog in dialogs:
            dialog.reject()
            qapp.processEvents()

        assert [dialog._watching_focus for dialog in dialogs] == [False, False, False]
    finally:
        for dialog in dialogs:
            dialog.close()


def test_the_whole_guide_can_be_opened_from_the_dialog(qapp, monkeypatch):
    opened: list = []
    monkeypatch.setattr(
        settings_dialog_module.SettingsGuideDialog, "exec", lambda self: opened.append(self)
    )
    dialog = SettingsDialog(Settings())
    try:
        dialog.show()

        dialog._guide_button.click()
        dialog._guide_shortcut.activated.emit()

        assert len(opened) == 2
        assert opened[0]._text.toPlainText() == notes.guide_text()
        # Every category is a heading in it, so nothing is explained only in
        # a panel the user has to go and find.
        for category in notes.CATEGORIES:
            assert category.upper() in opened[0]._text.toPlainText()
    finally:
        dialog.close()


# -- Pressing OK on something that cannot be saved -------------------------


def test_bad_json_keeps_the_dialog_open_on_the_box_that_holds_it(qapp, monkeypatch):
    """OK must never appear to do nothing.

    A user who presses OK, hears silence and finds the dialog still open
    concludes the button is broken. So the problem is announced over
    whatever was being said, written where a sighted user reads the same
    words, and the focus is put in the offending box on its own page.
    """
    said: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        settings_dialog_module,
        "announce",
        lambda widget, message, urgent=False: said.append((message, urgent)),
    )
    dialog = SettingsDialog(Settings())
    try:
        dialog.show()
        box = dialog.page(notes.DEEPGRAM)._parameters
        box.setPlainText('{"smart_format": true,}')
        dialog.show_category(notes.GENERAL)

        dialog.accept()

        assert dialog.result() != QDialog.DialogCode.Accepted
        assert dialog.isVisible()
        # Taken to the page, and to the box on it.
        assert dialog._categories.currentItem().text() == notes.DEEPGRAM
        assert dialog.focusWidget() is box
        # Told what is wrong, urgently, and on screen as well.
        message, urgent = said[-1]
        assert urgent is True
        assert "not valid JSON" in message and "line 1" in message
        assert message == dialog._notes_text.toPlainText()
        # And nothing the user typed was thrown away.
        assert box.toPlainText() == '{"smart_format": true,}'
    finally:
        dialog.close()


def test_correcting_the_mistake_lets_ok_close_the_dialog(qapp):
    dialog = SettingsDialog(Settings())
    try:
        box = dialog.page(notes.DEEPGRAM)._parameters
        box.setPlainText("rubbish")
        dialog.accept()
        assert dialog.result() != QDialog.DialogCode.Accepted

        box.setPlainText('{"smart_format": true}')
        dialog.accept()

        assert dialog.result() == QDialog.DialogCode.Accepted
        assert dialog.chosen_settings().transcription.deepgram.parameters == {
            "smart_format": True
        }
    finally:
        dialog.close()


def test_the_note_for_the_offending_control_comes_back_afterwards(qapp):
    """The panel is borrowed to carry the message, not taken over."""
    dialog = SettingsDialog(Settings())
    try:
        dialog.show()
        box = dialog.page(notes.DEEPGRAM)._parameters
        box.setPlainText("rubbish")
        dialog.accept()
        qapp.processEvents()

        box.setPlainText("{}")
        dialog.page(notes.DEEPGRAM)._model_edit.setFocus(Qt.FocusReason.TabFocusReason)
        qapp.processEvents()
        box.setFocus(Qt.FocusReason.TabFocusReason)
        qapp.processEvents()

        assert dialog._notes_text.toPlainText() == notes.note_text("deepgram.parameters")
    finally:
        dialog.close()


# -- Working on a copy -----------------------------------------------------


def test_the_dialog_never_writes_into_the_settings_it_was_given(qapp):
    """Cancel has to leave everything as it was, including the nested parts.

    The settings hold eight nested objects, several with dictionaries of
    free-form parameters inside them. A shallow copy would leave the dialog
    writing into the very objects the caller is still holding, and the
    changes would survive Cancel in exactly the places nobody checks.
    """
    original = Settings(
        transcription=TranscriptionSettings(
            elevenlabs=ElevenLabsSettings(
                api_key="original-key", transcription_parameters={"num_speakers": 2}
            )
        )
    )
    before = deepcopy(original)
    dialog = SettingsDialog(original)
    try:
        page = dialog.page(notes.ELEVENLABS)
        page._key_edit.setText("changed-key")
        page._transcription_parameters.setPlainText('{"num_speakers": 9}')
        page._enabled_box.setChecked(False)
        general_page(dialog)._short_box.setValue(99)

        # The dialog holds the changes.
        chosen = dialog.chosen_settings()
        assert chosen.transcription.elevenlabs.api_key == "changed-key"
        assert chosen.transcription.elevenlabs.transcription_parameters == {"num_speakers": 9}
        assert chosen.short_skip_seconds == 99
        # The object it was handed holds none of them.
        assert original == before
    finally:
        dialog.close()


def test_asking_twice_gives_the_same_answer(qapp):
    """And leaves alone everything no page edits, such as Enhance Audio."""
    settings = Settings()
    settings.enhance.output_folder = "D:/Enhanced"
    dialog = SettingsDialog(settings)
    try:
        first = dialog.chosen_settings()
        second = dialog.chosen_settings()

        assert first == second
        assert first.enhance.output_folder == "D:/Enhanced"
    finally:
        dialog.close()


def test_the_vocabulary_is_edited_on_a_copy_too(qapp, monkeypatch):
    """It is kept in its own file, and Cancel must leave that file's contents alone."""
    original = Vocabulary(
        profiles=[
            VocabularyProfile(
                id="global-main",
                level=VocabularyLevel.GLOBAL,
                name="Everything",
                terms=[VocabularyTerm(text="Kotze")],
            )
        ]
    )
    dialog = SettingsDialog(Settings(), vocabulary=original)
    try:
        monkeypatch.setattr(
            settings_pages_module.TermDialog,
            "exec",
            lambda self: (
                self._text_edit.setText("Van der Merwe"),
                QDialog.DialogCode.Accepted,
            )[1],
        )
        dialog.page(notes.VOCABULARY).add_term()

        assert [term.text for term in dialog.chosen_vocabulary().profiles[0].terms] == [
            "Kotze",
            "Van der Merwe",
        ]
        assert [term.text for term in original.profiles[0].terms] == ["Kotze"]
    finally:
        dialog.close()


def test_every_setting_survives_the_trip_through_the_dialog(qapp):
    """Shown a full set of settings, the dialog must hand back exactly those.

    This is the test that catches a page reading one field and writing
    another, which no amount of looking at the screen would show.
    """
    settings = every_setting_changed()
    dialog = SettingsDialog(settings)
    try:
        assert dialog.chosen_settings() == settings
    finally:
        dialog.close()
