# Project Configuration

<!-- ClaudeProject schema: v1 -->
<!-- Format spec: docs/claudeproject-spec.md in the claude-plugins repo -->

Settings for the `synergy` plugin. All commands and the execute skill read this file. Keep it lean: it is auto-loaded into context on every workflow command, so prefer values over prose and remove sections the project does not use.

## Identity

| Setting        | Value                   |
| -------------- | ----------------------- |
| org            | `Subverting-complexity` |
| repo           | `VoxVerbatim`           |
| default-branch | `main`                  |

## Package Manager

`python` (pip, `requirements.txt` and `requirements-dev.txt`)

## Quality Gate

Command to run before each commit, from the repository root, with the development requirements and `ruff` installed:

```
python -m ruff check . && QT_QPA_PLATFORM=offscreen python -m pytest
```

This is the same pair of checks `.github/workflows/ci.yml` runs. The tests need no display: `QT_QPA_PLATFORM=offscreen` lets the Qt widget tests run headless.

## Branch Convention

Pattern for feature branches:

```
feature/{number}/{short-description}
```

Example: `feature/42/fix-user-login-broken`

## Issue Types & Fields

**Required.** "This org has none" and "nobody wrote this section" look identical at runtime, so the section is always present and always says which it is. `/synergy:setup` writes it from `wf org-capabilities`; re-run it after enabling issue types or adding a field.

### Capability

| Setting      | Value |
| ------------ | ----- |
| type-capable | `yes` |

The owner is an organisation with native GitHub issue types enabled (Bug, Feature, User Story, Epic, Chore). The native type is the classification.

### Field names

| Purpose key                  | Field name               |
| ---------------------------- | ------------------------ |
| field-stage                  | `Stage`                  |
| field-priority               | `Priority`               |
| field-effort                 | `Effort`                 |
| field-type                   | `Classification`         |
| field-origin                 | `Origin`                 |
| field-ownership              | `Ownership`              |
| field-user-release-notes     | `User release notes`     |
| field-internal-release-notes | `Internal release notes` |
| field-shipped-version        | `Shipped in version`     |

`field-priority`, `field-effort` and `field-ownership` are required: `wf issue-apply` refuses to create an issue without them. `Stage` has all nine options (`Backlog`, `In Progress`, `In Review`, `Blocked`, `Non-code`, `Needs refinement`, `Parked`, `Done`, `Area`); a blank `Stage` means available.

### Missing

| Field    | Consequence |
| -------- | ----------- |
| _(none)_ | —           |

## Refinement

| Setting          | Value               |
| ---------------- | ------------------- |
| refinement-skill | `feature-discovery` |

## Session Budget

Target ~100k tokens per session. One story per session, run start-to-finish. Commit and push early so work survives an unexpected end.

## Story Template

Issues should include at minimum: **Context** (what/why), **Requirements** (acceptance criteria + constraints), and optionally **Notes** (dependencies, references, edge cases). The repository's issue template (Summary, Changes, Acceptance criteria) takes precedence for headings.

## Reference Docs (optional)

- `README.md`: intended behaviour of every feature, and the architecture map
- `CLAUDE.md`: PySide6 and accessibility rules (JAWS, NVDA, ZoomText)

## Bundled Skills

Available as `/synergy:*`: acceptance-criteria, build, bulk-execute, code-architect, pr-review, ecosystem-setup, execute, feature-discovery, grill, pr-body, preflight, spec-hardening, support-request, tone, user-facing-communication, user-story, verify-feature, writing-github-issues.
