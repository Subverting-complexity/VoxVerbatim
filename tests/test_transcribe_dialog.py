"""Tests for the Transcribe dialog.

The pipeline is replaced in every test that runs one. These tests are about
what the dialog asks, what it refuses, what it says a run will cost and what
it reports afterwards. A real run would call four paid services over the
network, so nothing here imports a provider or opens a socket.
"""

from __future__ import annotations

import threading

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
)

from audio_transcriber.audio.library import AudioFile
from audio_transcriber import settings as settings_module
from audio_transcriber.settings import TranscriptionSettings
from audio_transcriber.transcription import pipeline
from audio_transcriber.transcription.model import (
    FinalToken,
    Provider,
    ReviewStatus,
    Transcript,
)
from audio_transcriber.transcription.vocabulary import (
    Vocabulary,
    VocabularyLevel,
    VocabularyProfile,
)
from audio_transcriber.ui import transcribe_dialog as transcribe_dialog_module
from audio_transcriber.ui.transcribe_dialog import AFRIKAANS, TranscribeDialog

from tests.conftest import wait_until


@pytest.fixture(autouse=True)
def every_library_loads(monkeypatch) -> None:
    """Pretend every vendor library can be imported.

    The dialog refuses to start a run whose libraries are not installed, and
    this environment does not have all of them. The tests that are about that
    refusal put a failing probe back on purpose.
    """
    monkeypatch.setattr(settings_module, "_probe_library", lambda module, attribute: None)


@pytest.fixture
def recordings(tmp_path) -> list[AudioFile]:
    """Two recordings of a known length. Nothing reads the files themselves."""
    return [
        AudioFile(path=tmp_path / "alpha.m4a", size_bytes=1000, duration_seconds=60.0),
        AudioFile(path=tmp_path / "beta.m4a", size_bytes=2000, duration_seconds=120.0),
    ]


@pytest.fixture
def vocabulary() -> Vocabulary:
    return Vocabulary(
        profiles=[
            VocabularyProfile(id="acme", level=VocabularyLevel.CLIENT, name="Acme Limited"),
            VocabularyProfile(id="rollout", level=VocabularyLevel.PROJECT, name="Rollout"),
        ]
    )


def configured_settings() -> TranscriptionSettings:
    """Settings with everything filled in, so nothing blocks a run."""
    settings = TranscriptionSettings()
    settings.elevenlabs.api_key = "eleven-key"
    settings.openai_transcription.api_key = "openai-key"
    settings.microsoft.api_key = "microsoft-key"
    settings.microsoft.endpoint = "https://example.invalid"
    settings.assemblyai.api_key = "assembly-key"
    settings.openai_adjudication.api_key = "adjudication-key"
    assert settings.missing_requirements() == []
    return settings


def make_transcript(name: str, words: int = 4, needing_review: int = 0) -> Transcript:
    transcript = Transcript(recording_name=name)
    for index in range(words):
        token = FinalToken(text=f"word{index}")
        if index < needing_review:
            token.review_status = ReviewStatus.PENDING
        transcript.tokens.append(token)
    return transcript


def open_dialog(recordings, settings=None, vocabulary=None) -> TranscribeDialog:
    return TranscribeDialog(
        recordings,
        settings if settings is not None else configured_settings(),
        vocabulary=vocabulary,
    )


def silence_message_boxes(monkeypatch, confirm: bool = True) -> list[str]:
    """Answer the two message boxes in code, and keep what they were told."""
    shown: list[str] = []
    monkeypatch.setattr(
        TranscribeDialog,
        "_show_completion_message",
        lambda self, message: shown.append(message),
    )
    monkeypatch.setattr(TranscribeDialog, "confirm_cost", lambda self, message: confirm)
    return shown


def fake_pipeline(monkeypatch, transcribe) -> None:
    monkeypatch.setattr(pipeline, "transcribe_recording", transcribe)


def run_and_wait(qapp, dialog: TranscribeDialog) -> None:
    assert dialog.start() is True
    assert wait_until(qapp, lambda: dialog.summary is not None)


def interactive_widgets(dialog: TranscribeDialog) -> list:
    """Every control a person can land on, gathered by type rather than by name.

    The text field inside a spin box is left out. It is a part of the spin
    box rather than a control of its own, and Qt names the spin box.
    """
    kinds = (QPushButton, QLineEdit, QCheckBox, QSpinBox, QPlainTextEdit, QListWidget)
    return [
        widget
        for kind in kinds
        for widget in dialog.findChildren(kind)
        if not isinstance(widget.parentWidget(), QAbstractSpinBox)
    ]


# -- What the dialog asks ------------------------------------------------


def test_the_dialog_lists_the_recordings_with_their_lengths(qapp, recordings):
    dialog = open_dialog(recordings)
    try:
        assert dialog._recording_list.count() == 2
        assert dialog._recording_list.item(0).text().startswith("alpha.m4a")
        assert "1 minute" in dialog._recording_list.item(0).text()
        assert "2 recordings" in dialog._recording_group.title()
    finally:
        dialog.close()


def test_a_recording_whose_length_is_unknown_says_so_rather_than_saying_nothing(
    qapp, tmp_path
):
    unmeasured = [AudioFile(path=tmp_path / "alpha.m4a", size_bytes=1000)]
    dialog = open_dialog(unmeasured)
    try:
        assert "length unknown" in dialog._recording_list.item(0).text()
    finally:
        dialog.close()


def test_afrikaans_starts_switched_off_by_default(qapp, recordings):
    dialog = open_dialog(recordings)
    try:
        assert dialog._afrikaans_box.isChecked() is False
        assert dialog.chosen_configuration().afrikaans_enabled is False
    finally:
        dialog.close()


def test_afrikaans_starts_switched_on_where_the_settings_say_so(qapp, recordings):
    settings = configured_settings()
    settings.processing.default_afrikaans_enabled = True
    settings.processing.default_expected_speaker_count = 4

    dialog = open_dialog(recordings, settings)
    try:
        assert dialog._afrikaans_box.isChecked() is True
        assert dialog._speaker_count_box.value() == 4
    finally:
        dialog.close()


def test_the_dialog_collects_every_configuration_value(qapp, recordings, vocabulary):
    dialog = open_dialog(recordings, vocabulary=vocabulary)
    try:
        dialog._afrikaans_box.setChecked(True)
        dialog._speaker_count_box.setValue(3)
        dialog._speaker_names_edit.setText("Anna Bosch, Piet Muller ,  ")
        dialog._context_edit.setPlainText("A quarterly review with the auditors.\n")
        dialog._profile_list.item(1).setCheckState(Qt.CheckState.Checked)

        configuration = dialog.chosen_configuration()

        assert configuration.afrikaans_enabled is True
        assert configuration.expected_speaker_count == 3
        # The empty entry left by a trailing comma is dropped rather than
        # being sent to the services as a nameless speaker.
        assert configuration.known_speakers == ["Anna Bosch", "Piet Muller"]
        assert configuration.recording_context == "A quarterly review with the auditors."
        assert configuration.vocabulary_profile_ids == ["rollout"]
    finally:
        dialog.close()


def test_the_configuration_reaches_the_pipeline_options(qapp, recordings, vocabulary):
    dialog = open_dialog(recordings, vocabulary=vocabulary)
    try:
        dialog._afrikaans_box.setChecked(True)
        options = dialog.chosen_options()

        assert options.configuration.afrikaans_enabled is True
        assert options.vocabulary is vocabulary
        assert options.settings is dialog._settings
    finally:
        dialog.close()


def test_a_vocabulary_with_no_profiles_says_where_to_make_one(qapp, recordings):
    dialog = open_dialog(recordings)
    try:
        assert dialog._profile_list.count() == 1
        assert "Settings" in dialog._profile_list.item(0).text()
        # It is not a profile, so it cannot be chosen as one.
        assert dialog.chosen_profile_ids() == []
    finally:
        dialog.close()


# -- What it will cost ---------------------------------------------------


def test_the_cost_estimate_names_every_service_and_the_total(qapp, recordings):
    dialog = open_dialog(recordings)
    try:
        text = dialog.cost_text()

        assert "Total audio length: 3 minutes." in text
        for provider in (Provider.ELEVENLABS, Provider.OPENAI, Provider.MICROSOFT):
            assert provider.display_name in text
        assert "Estimated total:" in text
        # The honesty about what the figure leaves out is not optional.
        assert "cannot be worked out in advance" in text
        assert "rates are typed into Settings by hand" in text
        assert dialog._cost_display.toPlainText() == text
    finally:
        dialog.close()


def test_only_the_services_that_are_switched_on_are_counted(qapp, recordings):
    settings = configured_settings()
    settings.microsoft.enabled = False

    dialog = open_dialog(recordings, settings)
    try:
        assert dialog.services_that_will_run() == [Provider.ELEVENLABS, Provider.OPENAI]
        assert Provider.MICROSOFT.display_name not in dialog.cost_text()
    finally:
        dialog.close()


class PartialRates:
    """Cost settings that know every price except Microsoft's."""

    confirm_before_running = False
    adjudication_per_request = 0.0

    def per_minute_for(self, provider):
        return None if provider is Provider.MICROSOFT else 0.01


def test_a_service_with_no_rate_is_reported_as_unpriced_rather_than_free(qapp, recordings):
    settings = configured_settings()
    settings.cost = PartialRates()

    dialog = open_dialog(recordings, settings)
    try:
        text = dialog.cost_text()

        assert "Microsoft MAI: unpriced" in text
        assert "It is not in the total." in text
        # Three minutes at a penny a minute for each of the two priced
        # services, and nothing invented for the third.
        assert "Estimated total: 0.06 USD." in text
    finally:
        dialog.close()


def test_a_recording_of_unknown_length_is_named_as_missing_from_the_estimate(
    qapp, tmp_path
):
    recordings = [
        AudioFile(path=tmp_path / "alpha.m4a", size_bytes=1000, duration_seconds=60.0),
        AudioFile(path=tmp_path / "beta.m4a", size_bytes=2000),
    ]

    dialog = open_dialog(recordings)
    try:
        text = dialog.cost_text()

        assert "Total audio length: 1 minute." in text
        assert "the real cost will be higher: beta.m4a" in text
    finally:
        dialog.close()


# -- Agreeing to spend the money -----------------------------------------


def test_the_run_asks_before_spending_anything_and_can_be_refused(
    qapp, monkeypatch, recordings
):
    asked: list[str] = []
    monkeypatch.setattr(
        TranscribeDialog,
        "confirm_cost",
        lambda self, message: asked.append(message) or False,
    )
    fake_pipeline(monkeypatch, lambda *args, **kwargs: pytest.fail("Nothing should be sent."))

    dialog = open_dialog(recordings)
    try:
        assert dialog.start() is False

        assert len(asked) == 1
        assert "Estimated total:" in asked[0]
        assert dialog.is_running is False
        assert dialog._progress_label.text() == "The run was not started."
    finally:
        dialog.close()


def test_nothing_is_asked_where_the_user_has_switched_the_confirmation_off(
    qapp, monkeypatch, recordings
):
    settings = configured_settings()
    settings.cost.confirm_before_running = False
    silence_message_boxes(monkeypatch, confirm=False)
    fake_pipeline(monkeypatch, lambda recording, *args, **kwargs: make_transcript(recording.name))

    dialog = open_dialog(recordings, settings)
    try:
        # Refusing would stop the run, and the run starts, so nothing asked.
        run_and_wait(qapp, dialog)
        assert dialog.summary.transcribed == 2
    finally:
        dialog.close()


# -- Refusing to start ---------------------------------------------------


def test_a_run_that_cannot_succeed_is_refused_before_anything_is_sent(
    qapp, monkeypatch, recordings
):
    fake_pipeline(monkeypatch, lambda *args, **kwargs: pytest.fail("Nothing should be sent."))
    monkeypatch.setattr(
        TranscribeDialog, "confirm_cost", lambda self, message: pytest.fail("Asked too late.")
    )

    dialog = open_dialog(recordings, TranscriptionSettings())
    try:
        assert dialog.start() is False

        assert dialog.is_running is False
        report = dialog._report_text.toPlainText()
        assert "ElevenLabs Scribe is switched on but is not set up" in report
        assert "cannot start" in dialog._progress_label.text()
    finally:
        dialog.close()


def test_a_missing_elevenlabs_is_explained_as_the_backbone_it_is(qapp, recordings):
    dialog = open_dialog(recordings, TranscriptionSettings())
    try:
        dialog.start()

        report = dialog._report_text.toPlainText()
        assert "no word timings and no initial speaker labels" in report
    finally:
        dialog.close()


def test_a_library_that_cannot_be_loaded_refuses_the_run_like_a_missing_key(
    qapp, monkeypatch, recordings
):
    """A package never installed is otherwise found half way through a paid run."""
    fake_pipeline(monkeypatch, lambda *args, **kwargs: pytest.fail("Nothing should be sent."))
    monkeypatch.setattr(
        TranscribeDialog, "confirm_cost", lambda self, message: pytest.fail("Asked too late.")
    )
    monkeypatch.setattr(
        settings_module,
        "_probe_library",
        lambda module, attribute: "No module named 'assemblyai'." if module == "assemblyai" else None,
    )

    dialog = open_dialog(recordings)
    try:
        assert dialog.start() is False

        assert dialog.is_running is False
        report = dialog._report_text.toPlainText()
        assert "AssemblyAI is switched on, but the assemblyai library" in report
        assert "No module named 'assemblyai'." in report
        assert "cannot start" in dialog._progress_label.text()
    finally:
        dialog.close()


def test_a_transcript_folder_that_cannot_be_written_refuses_the_run(
    qapp, monkeypatch, recordings
):
    from audio_transcriber.transcription.store import TranscriptStore

    fake_pipeline(monkeypatch, lambda *args, **kwargs: pytest.fail("Nothing should be sent."))
    monkeypatch.setattr(
        TranscribeDialog, "confirm_cost", lambda self, message: pytest.fail("Asked too late.")
    )
    monkeypatch.setattr(
        TranscriptStore,
        "probe_writable",
        lambda self: f"The transcript folder for {self.recording_path.name} cannot be written to.",
    )

    dialog = open_dialog(recordings)
    try:
        assert dialog.start() is False

        report = dialog._report_text.toPlainText()
        assert "The transcript folder for alpha.m4a cannot be written to." in report
    finally:
        dialog.close()


def test_the_folder_is_probed_before_any_money_is_spent(qapp, monkeypatch, recordings):
    """The probe is made, and removed, and the run then goes ahead."""
    silence_message_boxes(monkeypatch)
    fake_pipeline(monkeypatch, lambda recording, *args, **kwargs: make_transcript(recording.name))

    dialog = open_dialog(recordings)
    try:
        run_and_wait(qapp, dialog)
        folder = recordings[0].path.parent / "alpha.m4a.transcript"
        assert folder.is_dir()
        assert list(folder.iterdir()) == []
    finally:
        dialog.close()


def test_a_run_with_nothing_to_transcribe_says_so(qapp, monkeypatch):
    silence_message_boxes(monkeypatch)
    dialog = open_dialog([])
    try:
        assert dialog.start() is False
        assert "no recordings" in dialog._progress_label.text()
    finally:
        dialog.close()


# -- Running -------------------------------------------------------------


def test_a_finished_run_reports_every_recording_and_what_needs_review(
    qapp, monkeypatch, recordings
):
    def transcribe(recording, options, progress=None, cancelled=None):
        progress(pipeline.PipelineProgress(fraction=0.5, stage=f"Working on {recording.name}."))
        return make_transcript(recording.name, words=10, needing_review=2)

    fake_pipeline(monkeypatch, transcribe)
    shown = silence_message_boxes(monkeypatch)

    dialog = open_dialog(recordings)
    try:
        run_and_wait(qapp, dialog)

        assert dialog.summary.transcribed == 2
        assert dialog.summary.review_count == 4
        report = dialog._report_text.toPlainText()
        assert "alpha.m4a" in report and "beta.m4a" in report
        assert "4 words to review" in shown[0]
        assert dialog._progress_bar.value() == 100
        # The two things worth doing next are now available.
        assert dialog._review_button.isEnabled() is True
        assert dialog._folder_button.isEnabled() is True
    finally:
        dialog.close()


def test_the_stage_sentence_is_shown_while_the_run_goes_on(qapp, monkeypatch, recordings):
    def transcribe(recording, options, progress=None, cancelled=None):
        progress(pipeline.PipelineProgress(fraction=0.5, stage=f"Working on {recording.name}."))
        return make_transcript(recording.name)

    fake_pipeline(monkeypatch, transcribe)
    silence_message_boxes(monkeypatch)

    dialog = open_dialog(recordings)
    stages: list[str] = []
    dialog._runner.progressChanged.connect(lambda percentage, stage: stages.append(stage))
    try:
        run_and_wait(qapp, dialog)

        assert stages == ["Working on alpha.m4a.", "Working on beta.m4a."]
    finally:
        dialog.close()


def test_the_controls_come_back_when_the_run_ends(qapp, monkeypatch, recordings):
    fake_pipeline(monkeypatch, lambda recording, *args, **kwargs: make_transcript(recording.name))
    silence_message_boxes(monkeypatch)

    dialog = open_dialog(recordings)
    try:
        run_and_wait(qapp, dialog)

        assert dialog._configuration_group.isEnabled() is True
        assert dialog._start_button.isEnabled() is True
        assert dialog._close_button.accessibleName() == "Close"
    finally:
        dialog.close()


def test_a_recording_that_failed_is_named_in_the_report(qapp, monkeypatch, recordings):
    def transcribe(recording, options, progress=None, cancelled=None):
        if recording.name == "alpha.m4a":
            raise OSError("The file is not readable.")
        return make_transcript(recording.name)

    fake_pipeline(monkeypatch, transcribe)
    silence_message_boxes(monkeypatch)

    dialog = open_dialog(recordings)
    try:
        run_and_wait(qapp, dialog)

        assert "The file is not readable." in dialog._report_text.toPlainText()
        assert dialog.summary.failed == 1
    finally:
        dialog.close()


def test_cancelling_stops_the_run(qapp, monkeypatch, recordings):
    release = threading.Event()

    def transcribe(recording, options, progress=None, cancelled=None):
        release.wait(10.0)
        return make_transcript(recording.name)

    fake_pipeline(monkeypatch, transcribe)
    silence_message_boxes(monkeypatch)

    dialog = open_dialog(recordings)
    try:
        assert dialog.start() is True
        assert wait_until(qapp, lambda: dialog.is_running)
        dialog.cancel()
        release.set()

        assert wait_until(qapp, lambda: dialog.summary is not None)
        assert dialog.summary.cancelled is True
        assert len(dialog.summary.results) == 1
    finally:
        dialog.close()


def test_escape_stops_a_running_job_instead_of_closing_on_top_of_it(
    qapp, monkeypatch, recordings
):
    release = threading.Event()

    def transcribe(recording, options, progress=None, cancelled=None):
        release.wait(10.0)
        return make_transcript(recording.name)

    fake_pipeline(monkeypatch, transcribe)
    silence_message_boxes(monkeypatch)

    dialog = open_dialog(recordings)
    try:
        dialog.start()
        assert wait_until(qapp, lambda: dialog.is_running)
        dialog.reject()

        assert dialog.isVisible() is False or dialog.result() != QDialog.DialogCode.Rejected
        release.set()
        assert wait_until(qapp, lambda: dialog.summary is not None)
        assert dialog.summary.cancelled is True
    finally:
        dialog.close()


def test_a_second_cancel_closes_the_dialog_and_leaves_the_run_to_stop(
    qapp, monkeypatch, recordings
):
    """A request already with a service can take minutes to come back.

    The first Cancel asks the run to stop and says so. Refusing to close
    until the service answered would hold the person in front of the dialog
    for as long as the slowest service took, so the second press closes it,
    and the run stops by itself afterwards without anything raising.
    """
    release = threading.Event()

    def transcribe(recording, options, progress=None, cancelled=None):
        release.wait(10.0)
        return make_transcript(recording.name)

    fake_pipeline(monkeypatch, transcribe)
    shown = silence_message_boxes(monkeypatch)
    said: list[str] = []
    monkeypatch.setattr(
        transcribe_dialog_module,
        "announce",
        lambda widget, message, urgent=False: said.append(message),
    )

    dialog = open_dialog(recordings)
    try:
        dialog.show()
        dialog.start()
        assert wait_until(qapp, lambda: dialog.is_running)

        dialog.reject()
        assert dialog.isVisible() is True
        assert "Press Cancel again" in dialog._progress_label.text()
        assert any("Press Cancel again" in message for message in said)
        assert "Press this again" in dialog._close_button.accessibleDescription()

        dialog.reject()
        assert dialog.isVisible() is False
        assert dialog.is_stopping_in_background is True

        release.set()
        assert wait_until(qapp, lambda: dialog.summary is not None)
        # The run finished into a closed dialog: no message box was raised on
        # top of whatever the person is doing now.
        assert shown == []
        assert dialog.summary.cancelled is True
        assert dialog.is_running is False
    finally:
        # A detached dialog gets rid of itself once the summary is in, so by
        # now it may already have gone.
        try:
            dialog.close()
        except RuntimeError:
            pass


def test_closing_a_dialog_on_a_run_that_was_never_cancelled_stops_it_first(
    qapp, monkeypatch, recordings
):
    release = threading.Event()

    def transcribe(recording, options, progress=None, cancelled=None):
        release.wait(10.0)
        return make_transcript(recording.name)

    fake_pipeline(monkeypatch, transcribe)
    silence_message_boxes(monkeypatch)

    dialog = open_dialog(recordings)
    try:
        dialog.show()
        dialog.start()
        assert wait_until(qapp, lambda: dialog.is_running)

        # The close box is the same decision as Cancel, answered the same way.
        assert dialog.close() is False
        assert dialog.isVisible() is True
        assert dialog.close() is True
        assert dialog.isVisible() is False
    finally:
        release.set()
        wait_until(qapp, lambda: not dialog.is_running)


def test_a_finished_run_comes_to_the_front_before_it_speaks(qapp, monkeypatch, recordings):
    """A run lasts long enough for anybody to go and do something else."""
    fake_pipeline(monkeypatch, lambda recording, *args, **kwargs: make_transcript(recording.name))
    order: list[str] = []
    monkeypatch.setattr(
        TranscribeDialog, "_show_completion_message", lambda self, message: order.append("box")
    )
    monkeypatch.setattr(TranscribeDialog, "confirm_cost", lambda self, message: True)
    monkeypatch.setattr(
        TranscribeDialog, "_come_to_the_front", lambda self: order.append("front")
    )

    dialog = open_dialog(recordings)
    try:
        run_and_wait(qapp, dialog)
        assert order == ["front", "box"]
    finally:
        dialog.close()


def test_coming_to_the_front_flashes_the_taskbar(qapp, monkeypatch, recordings):
    from PySide6.QtWidgets import QApplication

    alerted: list[object] = []
    monkeypatch.setattr(QApplication, "alert", staticmethod(lambda widget, msec=0: alerted.append(widget)))
    dialog = open_dialog(recordings)
    try:
        dialog._come_to_the_front()
        assert alerted == [dialog]
    finally:
        dialog.close()


# -- What to do afterwards -----------------------------------------------


def test_the_review_button_asks_for_the_recording_that_needs_it(
    qapp, monkeypatch, recordings
):
    def transcribe(recording, options, progress=None, cancelled=None):
        needing = 2 if recording.name == "beta.m4a" else 0
        return make_transcript(recording.name, needing_review=needing)

    fake_pipeline(monkeypatch, transcribe)
    silence_message_boxes(monkeypatch)

    dialog = open_dialog(recordings)
    try:
        run_and_wait(qapp, dialog)
        assert dialog.review_request is None

        dialog.open_review_window()

        assert dialog.review_request is not None
        assert dialog.review_request.name == "beta.m4a"
        # The dialog closes so that the review window is not opened behind it.
        assert dialog.result() == QDialog.DialogCode.Accepted
    finally:
        dialog.close()


def test_where_nothing_needs_review_the_first_transcript_is_offered(
    qapp, monkeypatch, recordings
):
    fake_pipeline(monkeypatch, lambda recording, *args, **kwargs: make_transcript(recording.name))
    silence_message_boxes(monkeypatch)

    dialog = open_dialog(recordings)
    try:
        run_and_wait(qapp, dialog)

        assert dialog.outcome_to_open().name == "alpha.m4a"
        assert "act on alpha.m4a" in dialog._report_text.toPlainText()
    finally:
        dialog.close()


def test_the_transcript_folder_button_opens_the_folder_beside_the_recording(
    qapp, monkeypatch, recordings
):
    fake_pipeline(monkeypatch, lambda recording, *args, **kwargs: make_transcript(recording.name))
    silence_message_boxes(monkeypatch)
    opened: list[str] = []
    monkeypatch.setattr(
        "audio_transcriber.ui.transcribe_dialog.QDesktopServices.openUrl",
        lambda url: opened.append(url.toLocalFile()) or True,
    )

    dialog = open_dialog(recordings)
    try:
        run_and_wait(qapp, dialog)
        dialog.open_transcript_folder()

        assert opened[0].endswith("alpha.m4a.transcript")
    finally:
        dialog.close()


# -- Accessibility -------------------------------------------------------


def test_every_control_in_the_dialog_has_an_accessible_name(qapp, recordings, vocabulary):
    """Gathered by type, so a control added later cannot slip through unnamed."""
    dialog = open_dialog(recordings, vocabulary=vocabulary)
    try:
        unnamed = [
            f"{widget.__class__.__name__} in {widget.parentWidget().__class__.__name__}"
            for widget in interactive_widgets(dialog)
            if not widget.accessibleName()
        ]

        assert unnamed == []
    finally:
        dialog.close()


def test_every_control_in_the_dialog_can_be_reached_from_the_keyboard(qapp, recordings):
    dialog = open_dialog(recordings)
    try:
        unreachable = [
            widget.accessibleName()
            for widget in interactive_widgets(dialog)
            if not (widget.focusPolicy() & Qt.FocusPolicy.TabFocus)
        ]

        assert unreachable == []
    finally:
        dialog.close()


def test_every_input_has_a_label_pointing_at_it(qapp, recordings):
    dialog = open_dialog(recordings)
    try:
        buddies = {label.buddy() for label in dialog.findChildren(QLabel) if label.buddy()}
        inputs = (
            dialog._speaker_count_box,
            dialog._speaker_names_edit,
            dialog._context_edit,
            dialog._profile_list,
        )

        for field in inputs:
            assert field in buddies, f"{field.accessibleName()} has no label pointing at it"
    finally:
        dialog.close()


def test_no_two_controls_answer_the_same_alt_key(qapp, recordings):
    """Two controls on one Alt letter means neither is reliably reachable."""
    dialog = open_dialog(recordings)
    try:
        letters = [
            text[text.index("&") + 1].casefold()
            for text in (
                widget.text()
                for kind in (QLabel, QPushButton, QCheckBox)
                for widget in dialog.findChildren(kind)
            )
            if "&" in text and not text.endswith("&")
        ]

        assert sorted(letters) == sorted(set(letters)), f"repeated Alt keys in {letters}"
    finally:
        dialog.close()


def test_the_progress_bar_is_named_and_the_message_labels_are_not(qapp, recordings):
    """A label's text is its name, so naming one hides whatever it says."""
    dialog = open_dialog(recordings)
    try:
        assert dialog.findChild(QProgressBar).accessibleName() == "Overall progress"
        assert dialog._progress_label.accessibleName() == ""
        assert dialog._stage_label.accessibleName() == ""
    finally:
        dialog.close()


def test_the_explanation_panel_follows_the_focus_and_holds_still_when_focused(
    qapp, recordings
):
    dialog = open_dialog(recordings)
    dialog.show()
    try:
        dialog._speaker_names_edit.setFocus()
        qapp.processEvents()
        assert "Known speaker names" in dialog._notes_group.title()
        standing = dialog._notes_text.toPlainText()

        # Moving into the panel to read it must not change what it says.
        dialog._notes_text.setFocus()
        qapp.processEvents()
        assert dialog._notes_text.toPlainText() == standing
    finally:
        dialog.close()


def test_the_afrikaans_note_explains_what_switching_it_on_actually_costs(qapp, recordings):
    dialog = open_dialog(recordings)
    try:
        dialog._show_note(AFRIKAANS)
        note = dialog._notes_text.toPlainText()

        assert "not one of the available answers" in note
        assert "cheaper and more accurate" in note
        assert "Microsoft and Deepgram do not support it at all" in note
        assert "25 and 50 per cent" in note
    finally:
        dialog.close()
