"""Tests for the shareable Windows build.

These cannot build the application; that takes minutes and a working
internet connection. What they can do is catch the one mistake that is easy
to make and expensive to find, which is adding a library to the application
and forgetting to tell the build about it. The application then works
perfectly on the machine it was written on and fails on the machine it was
sent to, weeks later, in front of somebody else.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from audio_transcriber import APPLICATION_NAME, ORGANISATION_NAME

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC_FILE = REPOSITORY_ROOT / "packaging" / "audio-transcriber.spec"
PUBLISH_SCRIPT = REPOSITORY_ROOT / "publish.cmd"
READ_ME = REPOSITORY_ROOT / "packaging" / "Read me first.txt"

#: Libraries the application needs that are deliberately not collected in
#: full by the specification, and why each one is safe to leave out.
#:
#: PySide6 is by far the largest thing in the build, and PyInstaller ships
#: its own rules for it that work out which parts of Qt are actually used.
#: Collecting it whole would add hundreds of megabytes of Qt that this
#: application never touches.
#:
#: Mutagen is ordinary Python that says what it imports, which is exactly
#: the case PyInstaller handles by reading the code.
COLLECTED_BY_OTHER_MEANS = {"PySide6", "mutagen"}


def _required_distributions() -> set[str]:
    """The libraries listed in requirements.txt, without their versions."""
    text = (REPOSITORY_ROOT / "requirements.txt").read_text(encoding="utf-8")
    names = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0])
    return names


def _spec_value(name: str):
    """Read one setting out of the specification without running it.

    The specification cannot simply be imported. PyInstaller runs it with
    several names already defined, ``SPECPATH`` among them, so running it
    here would fail before reaching anything worth reading. Reading the
    syntax tree asks the file what it says instead of what it does.
    """
    tree = ast.parse(SPEC_FILE.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if name in targets:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is no longer defined in {SPEC_FILE.name}.")


def _import_names() -> dict[str, str]:
    """Installed name against imported name, taken from the specification.

    The mapping is kept there rather than here, because the build is the
    thing that has to act on it. This test only has to agree with it.
    """
    distributions = _spec_value("DISTRIBUTION_NAMES")
    return {distribution: package for package, distribution in distributions.items()}


def test_every_required_library_is_accounted_for_in_the_build() -> None:
    """A new library must be either collected or consciously left out.

    This is the test the file exists for. Adding a transcription service
    means adding its client library to requirements.txt, and a library the
    build has never been told about is one that goes missing the moment
    somebody presses Transcribe in a copy of the published folder.
    """
    bundled = set(_spec_value("BUNDLED_PACKAGES"))
    import_names = _import_names()
    for distribution in _required_distributions():
        if distribution in COLLECTED_BY_OTHER_MEANS:
            continue
        package = import_names.get(distribution, distribution)
        assert package in bundled, (
            f"{distribution} is needed to run the application but the build "
            f"does not collect it. Add {package!r} to BUNDLED_PACKAGES in "
            f"{SPEC_FILE.name}, or, if PyInstaller can be trusted to find it "
            f"by reading the imports, add {distribution!r} to "
            "COLLECTED_BY_OTHER_MEANS here and say why."
        )


def test_the_build_does_not_collect_libraries_that_are_no_longer_needed() -> None:
    """The other direction: a library dropped from the application.

    Left in the specification it would go on being downloaded and copied
    into every published folder, and the build would fail outright once it
    was no longer installed anywhere.
    """
    import_names = _import_names()
    required = {import_names.get(name, name) for name in _required_distributions()}
    for package in _spec_value("BUNDLED_PACKAGES"):
        assert package in required, (
            f"The build collects {package!r}, which is no longer in "
            "requirements.txt. Remove it from BUNDLED_PACKAGES in "
            f"{SPEC_FILE.name}."
        )


@pytest.mark.parametrize(
    "expected",
    [
        '"packaging\\audio-transcriber.spec"',
        '"packaging\\Read me first.txt"',
        'set "PUBLISH_DIR=publish"',
    ],
)
def test_the_publish_script_still_refers_to_the_files_it_needs(expected: str) -> None:
    """Catch a file renamed or moved without the script being changed.

    A batch file names its files as text, so nothing else would notice. Each
    of these is matched as it is written in the script rather than as a bare
    word, so that the same word appearing in a comment cannot stand in for
    the line that does the work.
    """
    assert expected in PUBLISH_SCRIPT.read_text(encoding="utf-8")


def test_the_publish_script_looks_for_the_program_the_build_produces() -> None:
    """The built program is named from the package, and the script is not.

    The specification takes the application's name from
    ``audio_transcriber/__init__.py``, so the folder and the .exe are named
    from there too. The batch file cannot read Python, so it has the name
    written into it. Renaming the application would leave the script looking
    for a program PyInstaller never produced, and it would report a failure
    after a build that had in fact succeeded.
    """
    application_name = APPLICATION_NAME
    script = PUBLISH_SCRIPT.read_text(encoding="utf-8")
    assert f"{application_name}.exe" in script
    assert f"PUBLISH_DIR%\\{application_name}" in script


def test_the_published_folder_carries_a_note_for_whoever_receives_it() -> None:
    """The folder is sent to people who did not build it.

    They meet an unsigned program from somebody they know, so the note has
    to cover the SmartScreen warning they will see, or the application looks
    like something Windows caught.
    """
    text = READ_ME.read_text(encoding="utf-8")
    assert "SmartScreen" in text
    assert f"{APPLICATION_NAME}.exe" in text


def test_the_note_sends_people_to_the_folder_their_settings_are_really_in() -> None:
    """The note writes the settings path out, and the path is built from names.

    Windows works out where per-user files go from the organisation name and
    the application name. Changing either moves the folder, and a note that
    still names the old one sends somebody looking for a folder that is not
    there.
    """
    text = READ_ME.read_text(encoding="utf-8")
    assert f"{ORGANISATION_NAME}\\{APPLICATION_NAME}" in text
