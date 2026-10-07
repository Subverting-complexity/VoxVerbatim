<# : batch part. PowerShell reads this line as the start of a comment.
@echo off
rem ----------------------------------------------------------------------
rem Installs the Claude Code companion tools that .claude\ecosystem.md
rem describes, for the person who runs it:
rem
rem   1. Graphify and Headroom, with pip, into the user's Python.
rem   2. The user's Python Scripts folder on the user PATH, so the
rem      graphify and headroom commands (and the Stop hook in
rem      .claude\settings.json) can find them.
rem   3. RTK, from its official GitHub release, checked against the
rem      release checksums, and its folder on the user PATH.
rem   4. RTK's global Claude Code hook (rtk init -g), after asking.
rem
rem Safe to run again: each step skips what is already in place.
rem
rem Everything installs per user, so no step needs administrator rights,
rem and the script does not ask for them by default. Elevating would be
rem worse than useless: when a standard user types an administrator's
rem password at the prompt, the elevated process runs as that other
rem account, and the PATH and Claude settings would land in the wrong
rem profile. The script elevates only when told to install RTK into a
rem folder the user cannot write to, for example:
rem
rem   install-companion-tools.cmd "C:\Program Files\rtk"
rem
rem The batch part below only hands this same file to PowerShell, which
rem runs everything after the closing comment marker.
rem ----------------------------------------------------------------------
setlocal
set "SELF=%~f0"
set "RTK_DIR=%~1"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s = [IO.File]::ReadAllText($env:SELF); Invoke-Expression $s"
set "RESULT=%ERRORLEVEL%"
echo.
pause
exit /b %RESULT%
#>

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is very slow with the progress bar on
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$script:Failures = @()

function Write-Step($text) { Write-Host ''; Write-Host "== $text" -ForegroundColor Cyan }
function Write-Ok($text)   { Write-Host "   OK: $text" -ForegroundColor Green }
function Write-Note($text) { Write-Host "   $text" }
function Write-Fail($text) { Write-Host "   FAILED: $text" -ForegroundColor Red; $script:Failures += $text }

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal $id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-Writable($dir) {
    try {
        if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        $probe = Join-Path $dir ('.write-test-' + [guid]::NewGuid())
        [IO.File]::WriteAllText($probe, '')
        Remove-Item -LiteralPath $probe -Force
        return $true
    } catch { return $false }
}

# Adds a folder to the user PATH in the registry and to this process.
# The registry value is read and written raw, as REG_EXPAND_SZ, so that
# entries such as %USERPROFILE%\bin stay unexpanded. The obvious
# [Environment]::SetEnvironmentVariable(..., 'User') call in Windows
# PowerShell writes REG_SZ instead, which silently breaks such entries.
function Add-UserPath($dir) {
    $dir = $dir.TrimEnd('\')
    $key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $true)
    try {
        $raw = [string]$key.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        $entries = @($raw -split ';' | Where-Object { $_ -ne '' })
        $already = $entries | Where-Object {
            [Environment]::ExpandEnvironmentVariables($_).TrimEnd('\') -ieq $dir
        }
        if ($already) {
            Write-Ok "$dir is already on the user PATH."
        } else {
            $key.SetValue('Path', (($entries + $dir) -join ';'), [Microsoft.Win32.RegistryValueKind]::ExpandString)
            Write-Ok "Added $dir to the user PATH."
            $script:PathChanged = $true
        }
    } finally { $key.Close() }

    if (-not (($env:Path -split ';') | Where-Object { $_.TrimEnd('\') -ieq $dir })) {
        $env:Path = "$env:Path;$dir"
    }
}

# Tells open programs, such as Explorer, that the environment changed,
# so new terminals started from them see the new PATH without a sign-out.
function Send-EnvironmentChange {
    if (-not ('Win32.NativeEnv' -as [type])) {
        Add-Type -Namespace Win32 -Name NativeEnv -MemberDefinition @'
[DllImport("user32.dll", SetLastError = true, CharSet = CharSet.Auto)]
public static extern IntPtr SendMessageTimeout(IntPtr hWnd, uint Msg, UIntPtr wParam, string lParam,
    uint fuFlags, uint uTimeout, out UIntPtr lpdwResult);
'@
    }
    $result = [UIntPtr]::Zero
    [void][Win32.NativeEnv]::SendMessageTimeout([IntPtr]0xffff, 0x1A, [UIntPtr]::Zero, 'Environment', 2, 5000, [ref]$result)
}

# Runs a native command and returns its exit code, without letting
# stderr output turn into a terminating PowerShell error.
function Invoke-Native([string]$exe, [string[]]$arguments) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $exe @arguments 2>&1 | ForEach-Object { Write-Note "$_" } ; return $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
}

function Test-Command([string]$exe, [string[]]$arguments) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $exe @arguments *> $null; return ($LASTEXITCODE -eq 0) }
    catch { return $false }
    finally { $ErrorActionPreference = $old }
}

# ---------------------------------------------------------------------
# Elevation, only when RTK is to go into a folder this user cannot write.
# ---------------------------------------------------------------------
$rtkDir = if ($env:RTK_DIR) { $env:RTK_DIR.Trim('"') } else { Join-Path $env:LOCALAPPDATA 'Programs\rtk' }

if (-not (Test-Writable $rtkDir)) {
    if (Test-Admin) {
        Write-Host "Cannot write to $rtkDir, even as administrator." -ForegroundColor Red
        exit 1
    }
    Write-Host "Installing RTK into $rtkDir needs administrator rights. Asking Windows to elevate..."
    Write-Host "Use your own administrator account at the prompt. Another account's settings would be changed instead of yours."
    try {
        $p = Start-Process -FilePath $env:SELF -ArgumentList "`"$rtkDir`"" -Verb RunAs -Wait -PassThru
        exit $p.ExitCode
    } catch {
        Write-Host 'Elevation was cancelled. Nothing was installed.' -ForegroundColor Red
        exit 1
    }
}

Write-Host 'Installing the Claude Code companion tools for' $env:USERNAME
if (Test-Admin) { Write-Note '(Running as administrator. That is not needed, but it does no harm for your own account.)' }

# ---------------------------------------------------------------------
# 1. Python
# ---------------------------------------------------------------------
Write-Step 'Finding Python'
$python = $null
$pythonArgs = @()
if (Test-Command 'py' @('-3', '--version'))     { $python = 'py'; $pythonArgs = @('-3') }
elseif (Test-Command 'python' @('--version'))   { $python = 'python' }

if (-not $python) {
    Write-Fail 'Python 3 was not found. Install it from https://www.python.org/downloads/ (tick "Add python.exe to PATH"), then run this script again.'
} else {
    $version = (& $python @pythonArgs --version 2>&1) -join ' '
    Write-Ok "Using $version"
    $minor = [int](& $python @pythonArgs -c 'import sys; print(sys.version_info[1])')
    if ($minor -lt 10) { Write-Fail 'Headroom needs Python 3.10 or later.' }

    # ------------------------------------------------------------------
    # 2. Graphify and Headroom
    # ------------------------------------------------------------------
    Write-Step 'Installing Graphify and Headroom with pip'
    $code = Invoke-Native $python ($pythonArgs + @('-m', 'pip', 'install', '--user', '--disable-pip-version-check', '--quiet', 'graphifyy', 'headroom-ai[all]'))
    if ($code -eq 0) { Write-Ok 'Graphify and Headroom are installed.' }
    else { Write-Fail "pip exited with code $code." }

    # ------------------------------------------------------------------
    # 3. The user Scripts folder on PATH
    # ------------------------------------------------------------------
    Write-Step 'Putting the Python Scripts folder on your PATH'
    $scripts = (& $python @pythonArgs -c "import os, sysconfig; print(sysconfig.get_path('scripts', os.name + '_user'))").Trim()
    if (-not (Test-Path -LiteralPath $scripts)) {
        Write-Fail "The Python Scripts folder $scripts does not exist after the install."
    } else {
        Add-UserPath $scripts
    }
}

# ---------------------------------------------------------------------
# 4. RTK
# ---------------------------------------------------------------------
Write-Step 'Installing RTK'
# 'rtk gain' is the real test. A bare 'rtk --version' also succeeds for
# the unrelated "Rust Type Kit", which has the same command name.
$rtkExe = Join-Path $rtkDir 'rtk.exe'
if ((Test-Path -LiteralPath $rtkExe) -and (Test-Command $rtkExe @('gain'))) {
    Write-Ok "RTK is already installed in $rtkDir."
} else {
    try {
        $headers = @{ 'User-Agent' = 'install-companion-tools'; 'Accept' = 'application/vnd.github+json' }
        $release = Invoke-RestMethod -Uri 'https://api.github.com/repos/rtk-ai/rtk/releases/latest' -Headers $headers
        $assetName = 'rtk-x86_64-pc-windows-msvc.zip'
        $asset = $release.assets | Where-Object { $_.name -eq $assetName }
        $sums  = $release.assets | Where-Object { $_.name -eq 'checksums.txt' }
        if (-not $asset) { throw "Release $($release.tag_name) has no $assetName." }
        Write-Note "Downloading RTK $($release.tag_name)..."

        $temp = Join-Path ([IO.Path]::GetTempPath()) ('rtk-' + [guid]::NewGuid())
        New-Item -ItemType Directory -Path $temp | Out-Null
        try {
            $zip = Join-Path $temp $assetName
            Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip -Headers $headers -UseBasicParsing

            if ($sums) {
                $sumFile = Join-Path $temp 'checksums.txt'
                Invoke-WebRequest -Uri $sums.browser_download_url -OutFile $sumFile -Headers $headers -UseBasicParsing
                $line = Get-Content -LiteralPath $sumFile | Where-Object { $_ -match [regex]::Escape($assetName) + '\s*$' } | Select-Object -First 1
                if (-not $line) { throw "checksums.txt has no entry for $assetName." }
                $expected = ($line -split '\s+')[0]
                $actual = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash
                if ($actual -ine $expected) { throw "Checksum mismatch for $assetName. Expected $expected, got $actual. Nothing was installed." }
                Write-Ok 'The download matches the published checksum.'
            } else {
                Write-Note 'This release publishes no checksums, so the download was not verified.'
            }

            $unpacked = Join-Path $temp 'unpacked'
            Expand-Archive -LiteralPath $zip -DestinationPath $unpacked -Force
            $found = Get-ChildItem -LiteralPath $unpacked -Recurse -Filter 'rtk.exe' | Select-Object -First 1
            if (-not $found) { throw "The archive has no rtk.exe." }
            Copy-Item -LiteralPath $found.FullName -Destination $rtkExe -Force
        } finally {
            Remove-Item -LiteralPath $temp -Recurse -Force -ErrorAction SilentlyContinue
        }

        if (Test-Command $rtkExe @('gain')) { Write-Ok "RTK $($release.tag_name) is installed in $rtkDir." }
        else { Write-Fail "RTK was copied to $rtkDir, but 'rtk gain' does not run." }
    } catch {
        Write-Fail "RTK install: $($_.Exception.Message)"
    }
}

if (Test-Path -LiteralPath $rtkExe) {
    Add-UserPath $rtkDir

    # Warn when another program called rtk would win on PATH.
    $first = Get-Command rtk -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($first -and ($first.Source -ine $rtkExe)) {
        Write-Note "WARNING: $($first.Source) comes earlier on PATH than $rtkExe."
        Write-Note "If that is the unrelated 'Rust Type Kit', remove it (cargo uninstall rtk) or the RTK hook will not work."
    }

    # ------------------------------------------------------------------
    # 5. RTK's global Claude Code hook
    # ------------------------------------------------------------------
    Write-Step "RTK's Claude Code hook"
    Write-Note "Current RTK setup:"
    [void](Invoke-Native $rtkExe @('init', '--show'))
    Write-Note ''
    Write-Note "'rtk init -g' adds RTK's hook to your global Claude Code settings (~\.claude\settings.json)."
    Write-Note 'If the setup above already shows the hook, answer N.'
    $answer = Read-Host '   Run rtk init -g now? [Y/n]'
    if ($answer -eq '' -or $answer -match '^[Yy]') {
        $code = Invoke-Native $rtkExe @('init', '-g')
        if ($code -eq 0) { Write-Ok 'RTK hook installed.' } else { Write-Fail "rtk init -g exited with code $code." }
    } else {
        Write-Note "Skipped. Run 'rtk init -g' later to add the hook."
    }
}

# ---------------------------------------------------------------------
# Finish
# ---------------------------------------------------------------------
if ($script:PathChanged) { Send-EnvironmentChange }

Write-Step 'Checking the installed tools'
foreach ($check in @(@('graphify', @('--version')), @('headroom', @('--version')), @('rtk', @('gain')))) {
    if (Test-Command $check[0] $check[1]) { Write-Ok "$($check[0]) runs." }
    else { Write-Fail "$($check[0]) does not run from PATH." }
}

Write-Host ''
if ($script:Failures.Count -eq 0) {
    Write-Host 'All companion tools are installed.' -ForegroundColor Green
    if ($script:PathChanged) {
        Write-Host 'Your PATH changed. Close and reopen your terminals and the Claude app so they see it.'
    }
    Write-Host "To use Headroom, start Claude Code with: headroom wrap claude"
    exit 0
} else {
    Write-Host "$($script:Failures.Count) step(s) failed:" -ForegroundColor Red
    $script:Failures | ForEach-Object { Write-Host "  - $_" }
    Write-Host 'Fix the problem and run this script again. Steps that already worked are skipped.'
    exit 1
}
