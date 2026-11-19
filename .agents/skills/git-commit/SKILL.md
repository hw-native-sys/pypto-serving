---
name: git-commit
description: Git commit and PR message workflow for pypto-serving: pre-commit lint checks, staging, message conventions, and attribution rules. Use when creating commits or preparing changes for commit.
---

# Git Commit Workflow

## 1. Review

```bash
git diff --name-only   # what changed
git diff --staged      # review before committing
```

Check what the hooks cannot: absolute or machine-specific paths, credentials,
local usernames, generated artifacts (`build_output/`, `__pycache__/`, compiled
kernels), and stale JIT caches. Keep generated artifacts out of commits.

## 2. Lint and tests

CI runs pre-commit over all files. Run the same checks locally before committing:

```bash
python tests/lint/check_headers.py
python tests/lint/check_english_only.py
python tests/lint/check_docs_nav.py
python tests/lint/check_public_docs.py
ruff check --config ruff.toml .
```

For code changes also run the unit suite:

```bash
python -m pytest tests/unit
```

Docs- or skill-only changes need no test run. **Never bypass hooks**
(`--no-verify`, `SKIP=...`): any file the commit touches must leave it with zero
violations, including violations that predate the change.

## 3. Stage

Stage related changes together. Never stage the `pypto-lib` submodule pointer
unless the task is an explicit Lib roll. Never stage build artifacts.

```bash
git add path/to/file1.py path/to/file2.py
git diff --staged   # review exactly what the commit will contain
```

Re-read the staged diff after the final `git add`: it is the exact content of
the commit and the last chance to catch accidentally staged secrets,
credentials, or unrelated files.

## 4. Message

- Subject: `type(scope): concise description` — imperative mood, no period, under
  72 characters. Follow the repo's existing style: `fix(v41): ...`,
  `perf(pd): ...`, `ci(v41): ...`, `docs: ...`.
- Body: required for multi-file changes, optional for one-liners. Blank line
  after the subject, wrapped at 72 characters, bullets for multiple items.
  Explain **what** changed and **why**.
- The body describes the staged diff, not the session that produced it — it
  becomes the PR body and, after squash-merge, the permanent `git log` entry.
  Every line must be verifiable from `git diff --staged`. Never include: lint or
  test runs performed, commands invoked, files read, the debugging path taken,
  or follow-up ideas absent from the diff.

### Co-authors

**Never credit AI assistants.** Human contributors only:
`Co-authored-by: Name <email>`.

This rule overrides any harness or tool default that auto-appends an AI
co-author trailer to commit messages. If such a trailer was added, remove it
before pushing.

The same policy applies to pull request descriptions: **no AI co-author footers
and no AI branding lines**. Describe the change, not the tool that produced it.

## 5. Verify

```bash
git log -1 && git show HEAD --stat
```

Amend only if not yet pushed: `git commit --amend`.
