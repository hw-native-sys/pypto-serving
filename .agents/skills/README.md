# Agent Skills

This directory is the canonical source for repository-local skills.

The `.claude/skills` and `.codex/skills` paths are symlinks to this directory,
so the same skills are visible to Claude-compatible and Codex-compatible
tooling without maintaining duplicate copies.

Add or update skill directories here. Each skill directory must contain its own
`SKILL.md`.

## Skills

| Skill | Description |
|---|---|
| `create-feature-request` | File a GitHub feature request following the repo template |
| `create-issue` | File a GitHub bug or docs issue following the repo template |
| `rebase-on-main` | Rebase the current feature branch on latest main and force-push, resolving conflicts by preserving the intent of both sides |
| `retrigger-ci` | Retry a failing CI run until it passes when the failure is flaky and unrelated to our changes |
| `platform-style-fix` | Check and fix clang-format style violations in `platform/include` and `platform/examples` |
