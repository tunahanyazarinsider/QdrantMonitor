# QdrantMonitor

## Code review

After writing or changing code, run the built-in `/code-review` on the diff before reporting the work as done or committing it. Fix confirmed findings, then re-run the checks.

This is the built-in `/code-review`, not the `code-review:pre-commit-review` plugin skill.

## Changelog

Every commit that changes what a user sees or can do adds a line under **Unreleased** in `CHANGELOG.md`, in the same commit. When the user pushes, move the Unreleased lines under a heading with that day's date.
