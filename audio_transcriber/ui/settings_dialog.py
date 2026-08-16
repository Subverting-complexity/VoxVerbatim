"""The Settings dialog.

The dialog works on a copy of the settings and hands the copy back only if
the user presses OK, so Cancel really does leave everything as it was.

There are only a few settings so far. The point of the dialog is that the
way settings are shown, changed, checked and saved is settled, so that
adding the next one means adding a row here and a field to
:class:`~audio_transcriber.settings.Settings`, and nothing else.
"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from audio_transcriber.settings import MAXIMUM_SKIP_SECONDS, MINIMUM_SKIP_SECONDS, Settings
from audio_transcriber.ui.accessibility import describe


class SettingsDialog(QDialog):
    """Shows the settings and returns the changed ones."""

    def __init__(self, settings: Settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self._settings = settings

        self._reopen_box = QCheckBox("&Reopen the last folder when the application starts", self)
        self._reopen_box.setChecked(settings.reopen_last_folder)
        describe(
            self._reopen_box,
            "Reopen the last folder when the application starts",
            "When this is off, the application starts with no folder open and waits "
            "for you to choose one.",
        )

        skip_group = QGroupBox("Skip intervals", self)
        skip_layout = QFormLayout(skip_group)
        skip_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self._short_box = self._add_skip_row(
            skip_layout, "&Short skip", "Short skip", settings.short_skip_seconds
        )
        self._medium_box = self._add_skip_row(
            skip_layout, "&Medium skip", "Medium skip", settings.medium_skip_seconds
        )
        self._long_box = self._add_skip_row(
            skip_layout, "&Long skip", "Long skip", settings.long_skip_seconds
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._reopen_box)
        layout.addWidget(skip_group)
        layout.addStretch(1)
        layout.addWidget(buttons)

        # The first setting takes the focus, so a screen reader starts on
        # something worth reading rather than on the OK button.
        self._reopen_box.setFocus(Qt.FocusReason.TabFocusReason)

    def _add_skip_row(
        self,
        layout: QFormLayout,
        label_text: str,
        accessible_name: str,
        value: int,
    ) -> QSpinBox:
        label = QLabel(label_text, self)
        box = QSpinBox(self)
        box.setRange(MINIMUM_SKIP_SECONDS, MAXIMUM_SKIP_SECONDS)
        box.setValue(value)
        box.setSuffix(" seconds")
        # The box will not accept a value outside its range, so there is no
        # error state to report and nothing for the user to correct.
        describe(
            box,
            accessible_name,
            f"How far the {accessible_name.lower()} buttons move, in seconds. "
            f"Between {MINIMUM_SKIP_SECONDS} and {MAXIMUM_SKIP_SECONDS}.",
        )
        label.setBuddy(box)
        layout.addRow(label, box)
        return box

    def chosen_settings(self) -> Settings:
        """Return the settings as the user left them."""
        return replace(
            self._settings,
            reopen_last_folder=self._reopen_box.isChecked(),
            short_skip_seconds=self._short_box.value(),
            medium_skip_seconds=self._medium_box.value(),
            long_skip_seconds=self._long_box.value(),
        )
