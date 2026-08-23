"""Tests for the on-screen and spoken forms of durations and sizes."""

from __future__ import annotations

import pytest

from vox_verbatim.formatting import (
    UNKNOWN_TEXT,
    format_duration,
    format_position,
    format_size,
    spoken_duration,
    spoken_position,
    spoken_size,
)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0:00"),
        (9, "0:09"),
        (65, "1:05"),
        (600, "10:00"),
        (3600, "1:00:00"),
        (3903, "1:05:03"),
        (59.6, "1:00"),
        (-5, "0:00"),
    ],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


def test_format_duration_without_a_value():
    assert format_duration(None) == UNKNOWN_TEXT


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0 seconds"),
        (1, "1 second"),
        (65, "1 minute 5 seconds"),
        (120, "2 minutes"),
        (3600, "1 hour"),
        (3903, "1 hour 5 minutes 3 seconds"),
    ],
)
def test_spoken_duration(seconds, expected):
    assert spoken_duration(seconds) == expected


def test_spoken_duration_without_a_value():
    assert spoken_duration(None) == UNKNOWN_TEXT


@pytest.mark.parametrize(
    ("num_bytes", "expected"),
    [
        (0, "0 bytes"),
        (1, "1 byte"),
        (999, "999 bytes"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (4 * 1024 * 1024, "4.0 MB"),
        (3 * 1024**3, "3.0 GB"),
    ],
)
def test_format_size(num_bytes, expected):
    assert format_size(num_bytes) == expected


def test_spoken_size_uses_whole_words():
    assert spoken_size(4 * 1024 * 1024) == "4.0 megabytes"
    assert spoken_size(2048) == "2.0 kilobytes"
    assert spoken_size(12) == "12 bytes"
    assert spoken_size(None) == UNKNOWN_TEXT


def test_positions_are_read_from_milliseconds():
    assert format_position(65_000) == "1:05"
    assert spoken_position(65_000) == "1 minute 5 seconds"
    assert format_position(None) == UNKNOWN_TEXT
