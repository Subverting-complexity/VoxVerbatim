# Review Configuration — VoxVerbatim

## Repository

- Org: Subverting-complexity
- Repo: VoxVerbatim
- Default branch: main

## Labels

State labels use the `review` prefix so they are easy to filter. The **Purpose** column is the stable identity skills resolve against; it never changes even if the prefix does.

State labels are mutually exclusive: exactly one is applied per review.

| Purpose             | Label                      | Type   | Meaning                                                                     |
| ------------------- | -------------------------- | ------ | --------------------------------------------------------------------------- |
| `needs-review`      | `review-needs-review`      | State  | Open PR awaiting its first review (entry state, applied at creation)        |
| `reviewing`         | `review-reviewing`         | State  | Review in progress — prevents concurrent reviews                            |
| `approved`          | `review-approved`          | State  | No remaining issues, ready for human merge                                  |
| `changes-requested` | `review-changes-requested` | State  | Concrete problems remain that a human must address                          |
| `needs-discussion`  | `review-needs-discussion`  | State  | Architectural or scope questions need human judgment                        |
| `needs-re-review`   | `review-needs-re-review`   | State  | New commits pushed since last review — re-review required                   |
| `failed`            | `review-failed`            | State  | Review could not be completed (checkout failed, PR too large)               |
| `updating`          | `review-updating`          | State  | A builder agent is addressing review feedback — prevents concurrent updates |
| `fixes-applied`     | `review-fixes-applied`     | Action | Claude pushed fix commits to the PR branch (sticky across runs)             |

These labels are managed by `/synergy:pr-review` and are the single source of truth for PR review state.

## Auto-Merge on Approval

| Setting                      | Value      |
| ---------------------------- | ---------- |
| auto-merge-on-approval       | `disabled` |
| require-ci-before-merge      | `false`    |
| bypass-ci-on-billing-failure | `false`    |
| bypass-ci-when-no-pipeline   | `false`    |

An approved PR is left for a person to merge.

## Hard Non-Compliance Gates

Any of these force a `Changes Requested` verdict regardless of all other findings.

- **No linked issue**: the PR does not reference the GitHub issue it implements.
- **Untested non-trivial logic**: new or changed logic in `vox_verbatim/transcription/`, `vox_verbatim/audio/` or the settings and storage modules ships without tests.
- **Secrets in code**: API keys or tokens for any speech service committed to the repository, written to a log, or written into a transcript's provenance record.
- **Scope creep**: the PR makes substantial changes unrelated to its linked issue.
- **Quality gate not run or failing**: `ruff check .` or `pytest` fails.
- **Accessibility regression**: a UI change leaves a control without an accessible name, breaks keyboard access or `Tab` order, loses focus on opening or closing a dialog, hides a status change from screen readers, or conveys information by colour or position alone. `CLAUDE.md` treats these as functional bugs.

## Tech Stack Review Rules

Stack: Python 3.11+, PySide6 (Qt 6 Widgets), PyAV (FFmpeg), the vendor SDKs for ElevenLabs, OpenAI, AssemblyAI and Deepgram, and `httpx` for Microsoft.

- **Standard Qt widgets**: prefer standard widgets and model/view classes over custom-painted controls; flag Qt Quick or QML.
- **UI thread only**: worker threads must not touch widgets; results cross back by signal.
- **Thread and runner lifetime**: a runner or thread must be stopped before the object that owns it is destroyed, and a window must stop playback and timers on close.
- **Vendor SDKs imported lazily**: a service library is imported only when that service is called, so a missing package switches off one service rather than the application.
- **AssemblyAI parameters**: AssemblyAI code must match the current API (`https://www.assemblyai.com/docs/llms.txt`), not remembered parameter names.
- **Writes are atomic**: JSON written to disk goes through `json_store` so an interrupted save leaves the previous file intact.
- **Time units**: canonical times are seconds; flag milliseconds leaking through an adapter and an offset added twice.

## Architecture Rules

- **Layer boundaries**: `vox_verbatim/transcription/` and `vox_verbatim/audio/` must not import from `vox_verbatim/ui/`.
- **Adapters stay at the edge**: nothing above `transcription/providers/` names a vendor library or a vendor's field names.
- **Numbers go to a person**: no change may let an amount, date or account number be settled by scoring or by the language model. The README states this rule and it is the product's core promise.
- **A folder is a project**: what a review learns stays in that folder's project file unless the PR documents a deliberate exception.

## Security Specifics

- No API key reaches a log line, an exception message shown to the user, a raw-response file or a provenance record.
- Recordings are sent only to the services the user switched on.

## Test Expectations

- New behaviour in the transcription pipeline has a unit test that runs without network access.
- UI changes have a test that runs under `QT_QPA_PLATFORM=offscreen` and checks accessible names and keyboard behaviour where they change.
- A bug fix includes a test that fails without the fix.

## Review Comment Footer

```
---
Reviewed at <SHA>
🤖 Reviewed with Claude Code
```

The `Reviewed at <SHA>` line is machine-parsed by future runs to detect whether the PR has changed since the last review.
