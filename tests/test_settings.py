"""Tests for the user's settings and how they are kept."""

from __future__ import annotations

import json

from vox_verbatim.formatting import compact_interval
from vox_verbatim.settings import EnhanceSettings, Settings, SettingsStore


def test_saved_settings_come_back_unchanged(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    settings = Settings(
        reopen_last_folder=False,
        short_skip_seconds=10,
        medium_skip_seconds=60,
        long_skip_seconds=600,
    )

    assert store.save(settings)

    assert store.load() == settings


def test_no_settings_file_gives_the_defaults(tmp_path):
    loaded = SettingsStore(tmp_path / "not-there.json").load()

    assert loaded == Settings()
    assert loaded.reopen_last_folder is True
    assert loaded.skip_seconds == (15, 120, 300)


def test_a_damaged_settings_file_gives_the_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{ not json at all", encoding="utf-8")

    assert SettingsStore(path).load() == Settings()


def test_a_settings_file_that_is_not_text_gives_the_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_bytes(b'{"reopen_last_folder": \xff\xfe}')

    assert SettingsStore(path).load() == Settings()


def test_a_nonsense_value_falls_back_only_for_that_setting(tmp_path):
    """One bad value must not throw away the settings that are fine."""
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "reopen_last_folder": "yes please",
                "short_skip_seconds": 0,
                "medium_skip_seconds": 99999,
                "long_skip_seconds": 45,
            }
        ),
        encoding="utf-8",
    )

    loaded = SettingsStore(path).load()

    assert loaded.reopen_last_folder is True
    assert loaded.short_skip_seconds == 15
    assert loaded.medium_skip_seconds == 120
    assert loaded.long_skip_seconds == 45


def test_a_skip_of_true_is_not_a_skip_of_one_second(tmp_path):
    """True counts as 1 in Python, and would slip through a plain number check."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"short_skip_seconds": True}), encoding="utf-8")

    assert SettingsStore(path).load().short_skip_seconds == 15


def test_the_three_intervals_are_offered_shortest_first():
    settings = Settings(short_skip_seconds=5, medium_skip_seconds=30, long_skip_seconds=90)

    assert settings.skip_seconds == (5, 30, 90)


def test_interval_labels_are_short_enough_for_a_button():
    assert compact_interval(15) == "15 sec"
    assert compact_interval(60) == "1 min"
    assert compact_interval(120) == "2 min"
    assert compact_interval(90) == "1 min 30 sec"
    assert compact_interval(3600) == "60 min"


def test_the_enhancement_settings_are_saved_and_come_back(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    settings = Settings(
        enhance=EnhanceSettings(
            output_folder=r"D:\Enhanced",
            target_lufs=-16.0,
            ceiling_dbtp=-2.0,
            maximum_gain_db=18.0,
            use_limiter=True,
            output_format="flac",
            replace_existing=True,
        )
    )

    assert store.save(settings)

    assert store.load() == settings


def test_settings_written_before_enhancement_existed_still_load(tmp_path):
    """An older settings file has no enhancement section at all."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"short_skip_seconds": 20}), encoding="utf-8")

    loaded = SettingsStore(path).load()

    assert loaded.short_skip_seconds == 20
    assert loaded.enhance == EnhanceSettings()


def test_nonsense_enhancement_values_fall_back_to_the_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "enhance": {
                    "output_folder": 42,
                    "target_lufs": "loud",
                    "ceiling_dbtp": 20.0,
                    "maximum_gain_db": -5.0,
                    "use_limiter": "sometimes",
                    "output_format": "mp3",
                    "replace_existing": None,
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = SettingsStore(path).load().enhance

    assert loaded == EnhanceSettings()
    # A format the application cannot write must not survive being edited
    # into the file, or the next run would ask for an encoder that is not
    # there.
    assert loaded.output_format == "wav16"
