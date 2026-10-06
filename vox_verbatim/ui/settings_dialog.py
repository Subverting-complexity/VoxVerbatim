"""The Settings dialog.

The dialog works on a copy of the settings and hands the copy back only if
the user presses OK, so Cancel really does leave everything as it was.

It used to be one flat page, which was right when there were four settings
on it. There are now around sixty, across ten categories, and most of them
belong to one speech service or another. So it is a list of categories on
the left and one page on the right, with the pages themselves living in
:mod:`vox_verbatim.ui.settings_pages`, one class each.

That pattern was chosen over tabs, and it costs something a screen reader
user pays: with tabs, the relationship between the tab and what it reveals
is built into the widget and every screen reader announces it. A list beside
a stack is two ordinary controls that Qt has no reason to think are related
at all. Everything below that matters is about paying that cost back.

The relationship is made explicit rather than left to be inferred. Choosing
a category announces which page is now showing, and says whether the service
on it is set up, because that is the sentence somebody wants at that moment.
Each page carries the category as its own accessible name, so arriving on it
by Tab confirms where you are.

The order through the dialog is the order it reads: the category list, then
the controls of the page, then the panel that explains them, then OK and
Cancel. The arrow keys move down the categories, as they do in any list, and
Tab crosses into the page.

The pages scroll rather than growing past the bottom of the screen. At a
large Windows text size the ElevenLabs page alone is taller than a laptop
screen, and a dialog that grows to fit it would put OK and Cancel out of
reach entirely.

Pressing OK checks what the user typed before closing. The free-form
parameter boxes are the reason: they hold JSON, JSON can be wrong, and the
one thing that must never happen is that OK quietly does nothing. So a
mistake keeps the dialog open, moves to the page and the box holding it,
says exactly what is wrong and where, and interrupts the screen reader to
say so, because silence would be read as "the button is broken".
"""

from __future__ import annotations

import copy
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from vox_verbatim.settings import Settings
from vox_verbatim.transcription.vocabulary import Vocabulary
from vox_verbatim.ui import settings_notes as notes
from vox_verbatim.ui.accessibility import announce, describe
from vox_verbatim.ui.settings_pages import (
    Problem,
    SettingsPage,
    StatisticsSettingsPage,
    TranscriptionPage,
    build_pages,
)

#: How much of a note is on show at once. Enough to read a paragraph and see
#: that there is more, without the panel crowding out the settings.
_NOTE_LINES = 6

#: Used when Qt cannot say which screen the dialog is on, which happens in a
#: test run with no desktop.
_FALLBACK_MAXIMUM_HEIGHT = 900

#: A starting size that shows a provider page without scrolling at an
#: ordinary text size. Both are clamped to the screen before being used.
_PREFERRED_WIDTH = 980
_PREFERRED_HEIGHT = 760


class SettingsGuideDialog(QDialog):
    """Every setting explained, one after another, in one readable window.

    The panel in the Settings dialog shows one note at a time, which suits
    somebody working down a page. This is the same words grouped by
    category, for somebody who would rather read a page through before
    touching anything, or copy the lot into a document of their own.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Guide to the settings")

        self._text = QPlainTextEdit(self)
        self._text.setPlainText(notes.guide_text())
        self._text.setReadOnly(True)
        describe(
            self._text,
            "Guide to the settings",
            "Explains every setting in this dialog, grouped by category. Use the "
            "arrow keys to read through it.",
        )

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(self._text)
        layout.addWidget(buttons)

        self.resize(680, 640)
        # The text is the point of this window, so it starts with the focus
        # and a screen reader lands on something worth reading.
        self._text.setFocus()


class SettingsDialog(QDialog):
    """Shows the settings by category, and returns the changed ones."""

    def __init__(
        self,
        settings: Settings,
        parent: QWidget | None = None,
        vocabulary: Vocabulary | None = None,
        statistics: Any = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")

        # Every page edits a copy, and the copy is deep. The settings hold
        # eight nested objects and a dictionary of free-form parameters
        # inside several of them, so a shallow copy would leave the dialog
        # writing into the very objects the caller is still using, and
        # Cancel would leave changes behind in exactly the places nobody
        # thinks to check.
        self._settings = copy.deepcopy(settings)
        self._vocabulary = copy.deepcopy(vocabulary) if vocabulary is not None else Vocabulary()

        # Which note belongs to which control, gathered from every page, and
        # which note is on show.
        self._note_keys: dict[QWidget, str] = {}
        self._showing_note: str | None = None
        self._watching_focus = False
        # The control that OK found a problem in, while the problem is on
        # show in the note panel. See _report.
        self._problem_widget: QWidget | None = None

        self._pages: list[SettingsPage] = build_pages(self._vocabulary, statistics)
        self._pages_by_category = {page.category: page for page in self._pages}

        self._build_ui()
        self._connect_signals()
        self._show_settings()
        self._open_at_a_sensible_size()

        # The category list starts with the focus. It is the top of the
        # structure rather than the first setting, and landing there means a
        # screen reader reads out where the user is and what their choices
        # are before anything else happens.
        self._categories.setCurrentRow(0)
        self._categories.setFocus(Qt.FocusReason.TabFocusReason)

    # -- Building the dialog ---------------------------------------------

    def _build_ui(self) -> None:
        panes = QSplitter(Qt.Orientation.Horizontal, self)
        # A splitter so the category list can be widened at a large text
        # size, which matters under magnification. It is not something to
        # land on, so it stays out of the keyboard order and is left
        # unnamed; the two panes inside it are named instead.
        panes.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        panes.setChildrenCollapsible(False)
        panes.addWidget(self._build_categories())
        panes.addWidget(self._build_pages())
        panes.setStretchFactor(0, 0)
        panes.setStretchFactor(1, 1)
        self._panes = panes

        layout = QVBoxLayout(self)
        layout.addWidget(panes, 1)
        layout.addWidget(self._build_notes())
        # The buttons stay outside everything that scrolls, so OK and Cancel
        # are always in the same place however long a page is.
        layout.addWidget(self._build_buttons())

    def _build_categories(self) -> QWidget:
        holder = QWidget(self)
        label = QLabel("&Categories", holder)

        self._categories = QListWidget(holder)
        self._categories.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._categories.addItems([page.category for page in self._pages])
        describe(
            self._categories,
            "Settings categories",
            "Use the arrow keys to choose a category. The settings for it appear to "
            "the right, and Tab moves into them.",
        )
        label.setBuddy(self._categories)
        # As wide as the longest category needs and no wider, so the pages
        # keep the room. It follows the font, so it grows with the text size
        # rather than cutting a category name in half at 200 per cent.
        widest = self._categories.sizeHintForColumn(0)
        if widest > 0:
            self._categories.setMaximumWidth(widest + 2 * self._categories.frameWidth() + 40)
        self._categories.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label)
        layout.addWidget(self._categories, 1)
        return holder

    def _build_pages(self) -> QWidget:
        """One scrolling area per page, held in a stack.

        Each page gets its own scrolling area rather than the stack being
        put inside one. A stack asks for as much room as its tallest page
        whatever is showing, so one arrangement would give the General page,
        which has four settings on it, the height of the Transcription page,
        which has thirteen. Qt brings whatever takes the focus into view by
        itself, so tabbing through a long page still works exactly as it
        reads.
        """
        self._stack = QStackedWidget(self)
        self._scrollers: dict[SettingsPage, QScrollArea] = {}
        for page in self._pages:
            scroller = QScrollArea(self._stack)
            scroller.setWidget(page)
            scroller.setWidgetResizable(True)
            scroller.setFrameShape(QScrollArea.Shape.NoFrame)
            scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            # The scrolling area is a container, not somewhere to stand.
            # Naming it would put a stop on the way through that answers no
            # keys of its own, between the category list and the settings.
            scroller.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._stack.addWidget(scroller)
            self._scrollers[page] = scroller
            self._note_keys.update(page.note_keys)
        return self._stack

    def _build_notes(self) -> QWidget:
        """The panel that explains whichever setting the focus is on.

        These settings need this more than the Enhance Audio settings did,
        not less. Loudness is something most people have a feeling for; a
        canonical chunk overlap is not, and neither is an Azure API version.
        A label saying "Chunk overlap" teaches nobody anything, a tooltip
        needs a mouse and times out, and reading four paragraphs aloud every
        time the focus moves would be unbearable.

        So the explanation goes in a plain read-only text box that follows
        the focus. Tab onto a setting and its note appears; tab into the box
        to read it at leisure, where it stays put rather than changing under
        you.
        """
        group = QGroupBox("About this setting", self)
        self._notes_group = group

        self._notes_text = QPlainTextEdit(group)
        self._notes_text.setReadOnly(True)
        self._notes_text.setFixedHeight(self.fontMetrics().lineSpacing() * _NOTE_LINES)
        describe(
            self._notes_text,
            "About this setting",
            "Explains the setting the focus is on. Use the arrow keys to read "
            "through it. It stays on the last setting while you read, and the Guide "
            "button shows all of them together.",
        )

        self._guide_button = QPushButton("&Guide to these settings...", group)
        describe(
            self._guide_button,
            "Guide to these settings",
            "Opens a window explaining every setting in this dialog, grouped by "
            "category, so they can be read straight through. F1 does the same.",
        )
        self._guide_button.setAutoDefault(False)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self._guide_button, 0)

        inner = QVBoxLayout(group)
        inner.addWidget(self._notes_text)
        inner.addLayout(row)
        group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        return group

    def _build_buttons(self) -> QWidget:
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self._ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        describe(
            self._ok_button,
            "OK",
            "Saves everything on every page. If something cannot be saved you are "
            "told what and taken to it, and the dialog stays open.",
        )
        self._cancel_button = self._buttons.button(QDialogButtonBox.StandardButton.Cancel)
        describe(
            self._cancel_button,
            "Cancel",
            "Closes without changing anything, on every page, including the "
            "vocabulary.",
        )
        self._cancel_button.setAutoDefault(False)
        return self._buttons

    def _connect_signals(self) -> None:
        self._categories.currentRowChanged.connect(self._on_category_changed)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._guide_button.clicked.connect(self.show_guide)

        # F1 asks for help on what is in front of you, everywhere else in
        # Windows, so it does here too.
        self._guide_shortcut = QShortcut(QKeySequence(Qt.Key.Key_F1), self)
        self._guide_shortcut.activated.connect(self.show_guide)

    def _open_at_a_sensible_size(self) -> None:
        """Start out big enough to work in, and never taller than the screen."""
        self.resize(
            max(_PREFERRED_WIDTH, self.sizeHint().width()),
            min(_PREFERRED_HEIGHT, self._room_to_grow()),
        )

    def _room_to_grow(self) -> int:
        """The tallest this dialog may sensibly be on the screen it is on."""
        screen = self.screen()
        if screen is None:
            return _FALLBACK_MAXIMUM_HEIGHT
        # A little short of the working area, so the title bar and the edge
        # of the screen are not fighting for the same pixels.
        return int(screen.availableGeometry().height() * 0.9)

    # -- Pages ------------------------------------------------------------

    @property
    def pages(self) -> list[SettingsPage]:
        """Every page, in the order the category list shows them."""
        return list(self._pages)

    def page(self, category: str) -> SettingsPage:
        """The page for one category.

        Raises:
            KeyError: if there is no such category, which is a mistake in
                the code rather than anything a user can cause.
        """
        return self._pages_by_category[category]

    def show_category(self, category: str) -> None:
        """Show one category's page, as though it had been chosen in the list."""
        items = self._categories.findItems(category, Qt.MatchFlag.MatchExactly)
        if items:
            self._categories.setCurrentRow(self._categories.row(items[0]))

    def _on_category_changed(self, row: int) -> None:
        """Show the chosen page, and say plainly which one is now showing.

        This is the sentence that pays for choosing a list and a stack over
        tabs. Nothing in Qt ties these two controls together, so a user who
        arrows down the list would otherwise hear only the name of a list
        item and have no confirmation that anything at all happened on the
        other side of the dialog.

        A page that has something to say about itself says it here too. On a
        service page that is whether the service is set up, which is exactly
        what somebody wants to know on arriving, and the alternative is
        making them tab through the whole page to find out.
        """
        if not 0 <= row < len(self._pages):
            return
        page = self._pages[row]
        self._stack.setCurrentIndex(row)
        self._refresh_page(page)
        message = f"Showing {page.category} settings."
        status = page.status_message()
        if status:
            message = f"{message} {status}"
        announce(self._categories, message)

    def _refresh_page(self, page: SettingsPage) -> None:
        """Bring a page up to date with what the other pages now hold.

        Only the Transcription page needs this. What would stop a run
        depends on every service page, so the answer it shows has to be
        worked out from the settings as they stand this moment rather than
        from the ones the dialog opened with.
        """
        if isinstance(page, TranscriptionPage):
            page.show_requirements(self.chosen_settings().transcription)

    # -- Settings in and out ----------------------------------------------

    def _show_settings(self) -> None:
        for page in self._pages:
            page.show_settings(self._settings)
        self._refresh_page(self.page(notes.TRANSCRIPTION))

    def chosen_settings(self) -> Settings:
        """Return the settings as the user left them.

        Built from a fresh copy each time rather than from the object the
        pages were shown, so that asking twice gives the same answer and so
        that anything no page edits, such as the Enhance Audio settings,
        survives untouched.
        """
        chosen = copy.deepcopy(self._settings)
        for page in self._pages:
            page.apply_to(chosen)
        return chosen

    def chosen_vocabulary(self) -> Vocabulary:
        """Return the vocabulary as the user left it.

        It is separate from the settings because it is kept in its own file:
        it grows with use and is added to by reviewing transcripts as well
        as by this dialog. It is still edited on a copy here, so Cancel
        leaves it exactly as it was.
        """
        page = self.page(notes.VOCABULARY)
        return page.chosen_vocabulary()

    def show_statistics(self, statistics: Any) -> None:
        """Show a fresh set of service statistics, if that page could be built."""
        page = self.page(notes.STATISTICS)
        if isinstance(page, StatisticsSettingsPage):
            page.show_statistics(statistics)

    # -- The explanation panel ---------------------------------------------

    def showEvent(self, event) -> None:
        """Start watching the focus while the dialog is actually on screen."""
        super().showEvent(event)
        self._watch_focus(True)

    def hideEvent(self, event) -> None:
        """Stop watching as soon as it is not.

        The watch is on the application, which outlives this dialog, and the
        dialog may be kept by the window that opened it rather than thrown
        away when it closes. Left connected, every opening of Settings would
        add another watcher for the life of the application, each one
        looking at every focus change anywhere in it. Closing is also not
        the same as being closed: Cancel and Escape both go through reject,
        which hides the dialog without raising a close event at all.
        """
        self._watch_focus(False)
        super().hideEvent(event)

    def _watch_focus(self, watching: bool) -> None:
        """Follow the focus, or stop following it.

        The application is asked about the focus rather than each control
        being watched, because the widget that actually takes focus is often
        a part of a control rather than the control itself.
        """
        application = QApplication.instance()
        if application is None or watching == self._watching_focus:
            return
        self._watching_focus = watching
        if watching:
            application.focusChanged.connect(self._on_focus_changed)
        else:
            application.focusChanged.disconnect(self._on_focus_changed)

    def _on_focus_changed(self, _old: QWidget | None, new: QWidget | None) -> None:
        """Show the note for whatever now has the focus, if it has one.

        The search walks up from the focused widget rather than looking it
        up directly, because the widget that actually takes focus is often a
        part of the control rather than the control itself: a spin box hands
        it to the text field inside it. Walking up finds the control the
        user thinks they are on, and finds the page for a control that has
        no note of its own.

        Focus that belongs to nothing in particular, including the note
        panel itself, leaves the note where it is. That is what lets the
        note be read: moving into it to read it must not change it.

        A problem reported by OK stays on show while the focus is on the
        control that holds it, because _report puts the focus there and the
        control's own note would otherwise replace the problem at once. The
        first move anywhere else lets the notes follow the focus again.
        """
        problem = self._problem_widget
        if problem is not None:
            if new is not None and (new is problem or problem.isAncestorOf(new)):
                return
            self._problem_widget = None
        widget = new
        while widget is not None and widget is not self:
            key = self._note_keys.get(widget)
            if key is not None:
                self._show_note(key)
                return
            widget = widget.parentWidget()

    def _show_note(self, key: str) -> None:
        note = notes.note_for(key)
        if self._showing_note == key:
            return
        self._showing_note = key
        self._notes_group.setTitle(f"About: {note.title}")
        self._notes_text.setPlainText(note.note)
        # Back to the top, so reading starts at the first line rather than
        # wherever the last note was left.
        self._notes_text.moveCursor(QTextCursor.MoveOperation.Start)
        self._notes_text.verticalScrollBar().setValue(0)

    def show_guide(self) -> None:
        """Open the window that explains every setting one after another."""
        SettingsGuideDialog(self).exec()

    # -- Closing ------------------------------------------------------------

    def problems(self) -> list[Problem]:
        """Everything that would have to be corrected before OK can close."""
        found: list[Problem] = []
        for page in self._pages:
            found.extend(page.problems())
        return found

    def accept(self) -> None:
        """Check what was typed, and close only if it can all be saved.

        Silence is the failure to avoid here. A user who presses OK and
        finds the dialog still open, with nothing said, concludes that the
        button is broken and presses it again. So the first problem is
        announced over whatever the screen reader was saying, written on
        screen where a sighted user sees the same words, and the focus is
        put in the box that holds it, on the page that holds the box.

        Only the first problem is reported. Being told six things at once
        that you cannot see is not being helped; correcting one and pressing
        OK again brings the next.
        """
        found = self.problems()
        if not found:
            super().accept()
            return
        self._report(found[0])

    def _report(self, problem: Problem) -> None:
        page = self._page_holding(problem.widget)
        if page is not None:
            self.show_category(page.category)
        self._notes_group.setTitle("About: this could not be saved")
        self._notes_text.setPlainText(problem.message)
        self._notes_text.moveCursor(QTextCursor.MoveOperation.Start)
        # Cleared so that moving back onto the offending control shows its
        # explanation again rather than deciding nothing has changed.
        self._showing_note = None
        # Held until the focus leaves the control. Putting the focus there
        # below would otherwise show the control's note over the problem
        # straight away: a screen reader still hears the announcement, but
        # sighted and ZoomText users would see OK do nothing at all.
        self._problem_widget = problem.widget
        problem.widget.setFocus(Qt.FocusReason.OtherFocusReason)
        announce(problem.widget, problem.message, urgent=True)

    def _page_holding(self, widget: QWidget) -> SettingsPage | None:
        for page in self._pages:
            if page is widget or page.isAncestorOf(widget):
                return page
        return None
