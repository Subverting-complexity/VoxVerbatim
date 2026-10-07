# CLAUDE.md

## Desktop UI

Use PySide6 with Qt 6 Widgets for the desktop UI.

## Accessibility

Accessibility is a core requirement. The application must work well with JAWS, NVDA and ZoomText Magnifier/Reader on Windows.

When building or changing the UI:

* Prefer standard Qt widgets over custom-painted controls.
* Ensure every interactive control has a clear accessible name, role, value and state.
* Associate labels correctly with input controls.
* Ensure all functionality is available from the keyboard.
* Maintain logical `Tab` and `Shift+Tab` navigation.
* Handle focus correctly when opening and closing dialogs.
* Ensure errors, validation messages and important status changes are exposed to screen readers.
* Do not convey information using colour, icons or visual position alone.
* Use flexible layouts that work with large fonts, Windows scaling and magnification.
* Ensure editable text exposes the real caret so ZoomText can track it.
* Do not remove visible focus indicators.
* Prefer standard Qt model/view controls for tables, lists and trees.
* Avoid Qt Quick/QML unless there is a strong technical reason to use it.

Treat accessibility failures as functional bugs. Where there is a choice between custom visual behaviour and standard accessible Qt behaviour, favour accessibility.

## AssemblyAI

Always fetch https://www.assemblyai.com/docs/llms.txt before writing AssemblyAI code.
The API has changed — do not rely on memorized parameter names.

## Filing issues

`wf issue-apply` sometimes fails to create issues in this repository. GitHub answers the `createIssue` request that carries the issue fields with a general "Something went wrong" error, and no issue is filed. It happened on 2026-10-07 for every request that created 2 or 3 issues at once; a request that created 1 issue the same day worked. The cause is not known. The same create works in `Subverting-complexity/claude-plugins`, so it is specific to this repository or its field values.

When it happens, create the issue without its fields, then run `wf issue-apply` again with the new issue's `number` on the spec entry. That second run sets the fields and the stage without an error. No issue tracks this fault. To find the cause, first check which field value GitHub refuses during a create.


## Supplementary Files

| File | Purpose |
| ---- | ------- |
| `ClaudeProject.md` | Settings for the synergy plugin: identity, quality gate, branch convention, issue fields |
| `docs/review.config.md` | PR review labels, merge rules and review checks |
| `.claude/ecosystem.md` | Companion tools available to Claude Code |
