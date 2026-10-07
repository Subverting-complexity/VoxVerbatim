# -*- mode: python ; coding: utf-8 -*-
"""How the shareable Windows build of VoxVerbatim is put together.

This is a PyInstaller specification. It is read by ``publish.cmd``, which is
the only thing anybody needs to run. The result is a folder holding
``VoxVerbatim.exe`` and everything it needs, including a private copy
of Python itself, so that the folder can be copied to a computer that has
never had Python on it and still work.

Two decisions here are worth explaining, because both were choices between
things that all work.

The build is a folder rather than a single file. PyInstaller can produce one
self-contained ``.exe``, which looks tidier, but a single file has to unpack
its several hundred megabytes into a temporary folder on every start. That
turns a one-second start into a ten-second one, and it is exactly the shape
that virus scanners are most suspicious of. A folder starts immediately and
looks like ordinary software.

The client libraries for the transcription services are collected whole,
rather than being left to PyInstaller to work out. PyInstaller finds what a
program needs by reading its ``import`` statements, which works for code that
says what it wants. These libraries do not always: they build class names
from strings, read their own version out of their installation metadata, and
carry data files next to their code. Collecting each one in full costs disk
space, which is cheap, and removes a whole family of failures that only ever
appear on somebody else's computer, which is not.
"""

import importlib.util
import os
import re

from PyInstaller.utils.hooks import collect_all, copy_metadata

#: The repository root. ``SPECPATH`` is set by PyInstaller to the folder this
#: file lives in, so this holds wherever the repository has been cloned to
#: and whatever folder the build was started from.
REPOSITORY_ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))

#: The libraries collected in full, for the reason given at the top of the
#: file. These are the ones the application reaches for at the moment a
#: service is called rather than when it starts, so a mistake here does not
#: show up until somebody presses Transcribe.
#:
#: ``av`` is in the list for its own submodules, not for the FFmpeg libraries
#: it decodes with. Those live in a folder of their own beside the package,
#: ``av.libs``, which this collects nothing from; they arrive because
#: PyInstaller ships a rule for PyAV that knows about that layout. Anybody
#: chasing a missing codec on somebody else's machine should look there and
#: not here.
#:
#: ``tests/test_packaging.py`` checks this list against ``requirements.txt``,
#: so a library added to the application and forgotten here fails the test
#: suite instead of failing quietly in somebody's copy of the build.
BUNDLED_PACKAGES = (
    "av",
    "openai",
    "elevenlabs",
    "assemblyai",
    "deepgram",
    "httpx",
)

#: What a library is imported as, against what it is installed as, for the
#: ones where the two differ. Deepgram publishes ``deepgram-sdk`` and you
#: write ``import deepgram``. The build needs both names: one to find the
#: code, the other to find the installation record that goes with it.
DISTRIBUTION_NAMES = {"deepgram": "deepgram-sdk"}

#: Left out on purpose. Tkinter is a second, unused user interface toolkit
#: that Python installations carry, and pytest belongs to working on the
#: application rather than to running it. Neither is imported by anything
#: here, but both have a habit of being pulled in by something else.
EXCLUDED_PACKAGES = ("tkinter", "pytest")


def _package_constant(name):
    """Read one constant out of the package without importing it.

    Importing ``vox_verbatim`` would pull in PySide6 and the whole
    application while the build is still deciding what to build. Reading the
    one line is enough.

    Everything the built ``.exe`` says about itself is read this way rather
    than written down again here. The package is where the application's name,
    its organisation and its version are decided; a second copy in this file
    would be a second place to forget, and the two would disagree in the file
    properties of something already sent to somebody.
    """
    init_file = os.path.join(REPOSITORY_ROOT, "vox_verbatim", "__init__.py")
    with open(init_file, encoding="utf-8") as handle:
        source = handle.read()
    match = re.search(rf'^{name} = "([^"]*)"', source, re.MULTILINE)
    if match is None:
        raise SystemExit(
            f"{name} could not be found in vox_verbatim/__init__.py. "
            f'It is expected on a line of its own, as {name} = "something".'
        )
    return match.group(1)


APPLICATION_NAME = _package_constant("APPLICATION_NAME")
ORGANISATION_NAME = _package_constant("ORGANISATION_NAME")


def _version_resource_file(version):
    """Write the Windows version resource and return where it was written.

    This is what fills in the Details tab of the file properties dialog, and
    it is also the name SmartScreen reads out when it stops the application
    the first time. Without it, somebody being asked whether to trust a
    program sees a file name where the name of the program should be.

    Windows wants four numbers; the project uses three, so the fourth is
    zero. Anything in the version that is not a number is dropped from those
    four and kept in the version people read, so ``1.2.0rc1`` still builds.

    The names and the numbers go in as Python literals rather than as text
    dropped between quotation marks, so that an apostrophe in a name cannot
    turn the resource into something PyInstaller fails to parse.
    """
    numbers = []
    for part in version.split(".")[:3]:
        digits = re.match(r"\d+", part)
        numbers.append(int(digits.group()) if digits else 0)
    while len(numbers) < 4:
        numbers.append(0)
    numbers = tuple(numbers)
    company = repr(ORGANISATION_NAME)
    application = repr(APPLICATION_NAME)
    executable = repr(f"{APPLICATION_NAME}.exe")
    readable_version = repr(version)
    resource = f"""VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={numbers},
    prodvers={numbers},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0),
  ),
  kids=[
    StringFileInfo([
      StringTable(
        '040904B0',
        [
          StringStruct('CompanyName', {company}),
          StringStruct('FileDescription', {application}),
          StringStruct('FileVersion', {readable_version}),
          StringStruct('InternalName', {application}),
          StringStruct('OriginalFilename', {executable}),
          StringStruct('ProductName', {application}),
          StringStruct('ProductVersion', {readable_version}),
        ],
      ),
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])]),
  ],
)
"""
    os.makedirs(workpath, exist_ok=True)
    path = os.path.join(workpath, "windows-version-resource.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(resource)
    return path


datas = []
binaries = []
hidden_imports = []
for package in BUNDLED_PACKAGES:
    distribution = DISTRIBUTION_NAMES.get(package, package)
    # Ask whether the library is there before asking for its contents.
    # ``collect_all`` does not complain about a name it cannot find: it logs
    # a warning nobody reads in a thousand lines of build output and returns
    # nothing. A misspelling, or an installed name written where an imported
    # name belongs, would then produce a build that is quietly missing a
    # service, which is the exact failure this file exists to prevent.
    if importlib.util.find_spec(package) is None:
        raise SystemExit(
            f"{package} cannot be imported, so the build would silently "
            f"leave it out. Either {distribution} is not installed in the "
            "environment the build is running in, which publish.cmd takes "
            f"care of, or {package!r} is not the name the library is "
            "imported under. BUNDLED_PACKAGES holds imported names; "
            "DISTRIBUTION_NAMES is where the two are joined up."
        )
    try:
        # ``include_py_files`` is off. The libraries' Python code is compiled
        # into the archive inside the .exe, and collecting it again as data
        # would put a second, readable copy of every source file into the
        # folder that gets sent to people.
        package_datas, package_binaries, package_hidden_imports = collect_all(
            package, include_py_files=False
        )
        # The installation record as well as the code. Several of these
        # libraries look up their own version while they run, and a library
        # that cannot find out how old it is tends to raise rather than
        # shrug.
        package_datas += copy_metadata(distribution, recursive=True)
    except Exception as error:
        raise SystemExit(
            f"{distribution} could not be collected: {error}\n"
            "The libraries the application needs must be installed in the "
            "same environment the build runs in. Running publish.cmd does "
            "that for you."
        ) from error
    datas += package_datas
    binaries += package_binaries
    hidden_imports += package_hidden_imports

version = _package_constant("__version__")

analysis = Analysis(
    [os.path.join(SPECPATH, "launch.py")],
    pathex=[REPOSITORY_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=list(EXCLUDED_PACKAGES),
    noarchive=False,
    optimize=0,
)

archive = PYZ(analysis.pure)

executable = EXE(
    archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name=APPLICATION_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # No console window. The application already reports a failure it cannot
    # recover from in a standard message box, which a screen reader reads,
    # and writes the detail to its log file. A console window behind the
    # application would add nothing and would be one more thing in the way.
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version=_version_resource_file(version),
)

COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    # UPX compression is off. It makes the folder smaller and the start
    # slower, and compressed executables are one of the things virus
    # scanners judge software by. This application is already going to be
    # unsigned on somebody else's computer; there is no sense adding to the
    # suspicion to save disk space that nobody is short of.
    upx=False,
    upx_exclude=[],
    name=APPLICATION_NAME,
    # ``contents_directory`` is deliberately left at its default of
    # ``_internal``. That default is why the folder somebody receives holds
    # three things rather than fifteen hundred: the program, one folder of
    # supporting files, and the note. Setting it to "." would tip every DLL
    # out into the top of the folder, and finding the program by arrowing
    # down that list, with a screen reader reading each name, would be
    # miserable. It is a tidiness setting with an accessibility cost.
)
