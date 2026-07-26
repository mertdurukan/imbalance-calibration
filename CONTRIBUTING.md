# Contributing

This repository accompanies a pre-registered study. The commit history is part
of the research artifact, so history rewriting (force-push, rebase, amend on
`main`) is not accepted — see `.cursorrules`.

## Reporting problems

Open an issue describing the expected and observed behaviour, with the command
you ran and the relevant output.

## Pull requests

1. Open an issue first for anything beyond a typo fix.
2. Keep changes focused; one concern per pull request.
3. Run the test suite before submitting (`make test` or `pytest`).
4. Analyses that deviate from `PREREG.md` must be recorded in `DEVIATIONS.md`.

## Scope

Results reported in `paper.md` are frozen at tag `v1.0.0`. Corrections to those
results are welcome as issues; silent edits to the reported numbers are not.
