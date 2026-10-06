# Project Configuration

<!-- ClaudeProject schema: v1 -->
<!-- Format spec: docs/claudeproject-spec.md in the claude-plugins repo -->

Settings for the `synergy` plugin. All commands and the execute skill read this file. Keep it lean — it is auto-loaded into context on every workflow command, so prefer values over prose and remove sections your project does not use.

## Identity

| Setting        | Value                   |
| -------------- | ----------------------- |
| org            | `Subverting-complexity` |
| repo           | `VoxVerbatim`           |
| default-branch | `main`                  |

## Package Manager

`python (pip)`

## Quality Gate

Command to run before each commit. It runs the same checks as CI (`ruff check .`, then `pytest`), using the shared `.venv` in the main checkout:

```
bash scripts/quality-gate.sh
```

## Branch Convention

Pattern for feature branches:

```
feature/{number}/{short-desc}
```

Example: `feature/41/voice-picker`

## Issue Types & Fields

Written by `/synergy:setup` from `wf org-capabilities`. Re-run it after enabling issue types or adding a field.

### Capability

| Setting | Value |
| ------- | ----- |
| type-capable | `yes` |

The owner is an organization with native GitHub issue types enabled (Bug, Feature, User Story, Epic, Chore). The native type is the classification, and `wf issue-apply` strips any `type-*` label off an issue it writes.

### Field names

| Purpose key          | Field name       |
| -------------------- | ---------------- |
| field-stage          | `Stage`          |
| field-priority       | `Priority`       |
| field-effort         | `Effort`         |
| field-type           | `Classification` |
| field-origin         | `Origin`         |
| field-ownership      | `Ownership`      |
| field-user-release-notes     | `User release notes`     |
| field-internal-release-notes | `Internal release notes` |
| field-shipped-version        | `Shipped in version`     |

`field-priority`, `field-effort` and `field-ownership` are required: the pool's order, its size ceiling, and whether a code agent may take the issue. `field-type` and `field-origin` are optional. `field-stage` is the issue's state, with nine options (`Backlog`, `In Progress`, `In Review`, `Blocked`, `Non-code`, `Needs refinement`, `Parked`, `Done`, `Area`); a blank `Stage` means available, the same as `Backlog`.

### Missing

| Field | Consequence |
| ----- | ----------- |
| _(none)_ | — |

Option ids are not copied here. Run `wf org-capabilities` for the live values.

## Refinement

| Setting          | Value               |
| ---------------- | ------------------- |
| refinement-skill | `feature-discovery` |

Skill the execute flow offers when a story is too thin to implement: `feature-discovery` (default). It runs the `grill` interview and turns the answers into a fuller spec with acceptance criteria. A story a person has not approved yet belongs at `Stage` `Needs refinement`, which keeps it out of the pool without needing a label.

## Session Budget

Target ~100k tokens per session. One story per session, run start-to-finish. Commit and push early so work survives an unexpected end.

## Story Template

Issues should include at minimum: **Context** (what/why), **Requirements** (acceptance criteria + constraints), and optionally **Notes** (dependencies, references, edge cases).

## Bundled Skills

Available as `/synergy:*`: acceptance-criteria, build, bulk-execute, code-architect, pr-review, ecosystem-setup, execute, feature-discovery, grill, pr-body, preflight, spec-hardening, support-request, tone, user-facing-communication, user-story, verify-feature, writing-github-issues.
