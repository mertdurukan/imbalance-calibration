# DEVIATIONS LOG

Append-only. Every departure from `PREREG.md` is recorded here with an ISO-8601 timestamp,
what changed, why, and whether it was decided BEFORE or AFTER seeing the affected results.

The "before/after" field is the important one: a deviation decided after seeing results is
a potential source of bias and must be reported as such in the paper.

Format:

## YYYY-MM-DDTHH:MM:SSZ — <short title>
- **Changed:** what, precisely
- **Reason:** why it was unavoidable
- **Decided:** BEFORE seeing affected results / AFTER seeing affected results
- **Impact on hypotheses:** none / H1 / H2 / H3

---

## 2026-07-14T05:37:00Z — conda → venv+pip environment
- **Changed:** Replaced `environment.yml` / conda `make setup` with `requirements.txt` + `python3.11 -m venv .venv` + pip. All make targets now invoke `./.venv/bin/python` explicitly. Added `make verify` for import/libomp smoke check. SPEC §6 referenced conda.
- **Reason:** Target machine (macOS Apple Silicon) has no conda installed.
- **Decided:** BEFORE seeing affected results
- **Impact on hypotheses:** none
