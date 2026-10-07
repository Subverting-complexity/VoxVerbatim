# Review Configuration — VoxVerbatim

## Repository

- Org: Subverting-complexity
- Repo: VoxVerbatim
- Default branch: main

## Labels

The **Purpose** column is the stable identity skills resolve against. State labels are mutually exclusive — exactly one is applied per review.

| Purpose | Label | Type | Meaning |
| ------- | ----- | ---- | ------- |
| `needs-review` | `review-needs-review` | State | Open PR awaiting its first review (entry state, applied at creation) |
| `reviewing` | `review-reviewing` | State | Review in progress — prevents concurrent reviews |
| `approved` | `review-approved` | State | No remaining issues, ready to merge |
| `changes-requested` | `review-changes-requested` | State | Concrete problems remain that a human must address |
| `needs-discussion` | `review-needs-discussion` | State | Architectural or scope questions need human judgment |
| `needs-re-review` | `review-needs-re-review` | State | New commits pushed since last review — re-review required |
| `failed` | `review-failed` | State | Review could not be completed (checkout failed, PR too large) |
| `updating` | `review-updating` | State | A builder agent is addressing review feedback — prevents concurrent updates |
| `fixes-applied` | `review-fixes-applied` | Action | Claude pushed fix commits to the PR branch (sticky across runs) |

These labels are managed by the `/synergy:pr-review` skill and are the single source of truth for PR review state.

## Custom Labels

| Label | When to apply |
| ----- | ------------- |
| `accessibility` | The PR changes anything a person sees or operates: files under `vox_verbatim/ui/`, or any other Qt widget, dialog, menu, shortcut or status message. Review these against the Accessibility section of `CLAUDE.md`. |

## Auto-Merge on Approval

| Setting                      | Value     |
| ---------------------------- | --------- |
| auto-merge-on-approval       | `enabled` |
| require-ci-before-merge      | `true`    |
| bypass-ci-on-billing-failure | `true`    |
| bypass-ci-when-no-pipeline   | `false`   |

An approved PR is squash-merged and its branch deleted once the review comment is posted. It must have green CI (the `CI` workflow: Ruff and pytest). The one exception: when GitHub Actions cannot run because of a billing or account problem, and the local quality gate passed on the head commit, the PR merges anyway. A genuine red check is never bypassed, and a merge conflict is never bypassed.

## Hard Non-Compliance Gates

Any of these force a `Changes Requested` verdict regardless of all other findings.

- The PR links no issue it closes.
- Non-trivial logic changes with no new or updated tests.
- Secrets, API keys or tokens in code, tests, fixtures or logs. Service keys (OpenAI, ElevenLabs, AssemblyAI, Deepgram, Microsoft) come from the user's settings, never from the repository.
- Scope creep: changes that belong to none of the issues the PR closes.
- An accessibility regression: an interactive control with no accessible name, a function that cannot be reached from the keyboard, broken `Tab` order, a removed focus indicator, information given by colour or position alone, or an error or status change not exposed to screen readers.
- A custom-painted control or Qt Quick/QML where a standard Qt Widget would do.

## Tech Stack Review Rules

- Desktop UI uses PySide6 with Qt 6 Widgets. Prefer standard widgets and Qt model/view controls for lists, tables and trees.
- Every interactive control has an accessible name, role, value and state; labels are associated with their inputs (`setBuddy` or equivalent); dialogs set and restore focus correctly.
- Editable text exposes the real caret so ZoomText can track it.
- Layouts work with large fonts, Windows scaling and magnification: no fixed pixel sizes that clip text.
- Long work (transcription, audio enhancement, network calls) runs off the UI thread and reports progress and completion in a way a screen reader announces.
- AssemblyAI code must match the current API (https://www.assemblyai.com/docs/llms.txt), not memorised parameter names.
- Code passes `ruff check .` with the settings in `pyproject.toml` (line length 100, Python 3.11 target).

## Architecture Rules

- Keep audio (`vox_verbatim/audio/`), transcription (`vox_verbatim/transcription/`) and UI (`vox_verbatim/ui/`) separate: audio and transcription code does not import from the UI.
- Service adapters live under `vox_verbatim/transcription/providers/` and share one interface; service-specific details do not leak into reconciliation or the UI.
- Settings and stored state go through `settings.py` and `json_store.py`.

## Security Specifics

- No API key, token or recording content in logs, exceptions shown to the user, or crash output.
- Files written for the user (exports, enhanced copies, project files) never overwrite the original recording.
- Paths built from user input or service responses are not trusted blindly.

## Test Expectations

- New logic in audio, transcription and reconciliation code has unit tests under `tests/`.
- UI changes have tests where practical, running headless (`QT_QPA_PLATFORM=offscreen`), including accessible names of new controls.
- Tests never call a real speech service or the network; service responses are faked.
- `pytest` passes and `ruff check .` is clean.

## Review Comment Footer

```
---
Reviewed at <SHA>
🤖 Reviewed with Claude Code
```

The `Reviewed at <SHA>` line is machine-parsed by future runs to detect whether the PR has changed since the last review.
