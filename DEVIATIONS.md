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

## 2026-07-14 — N_DATASETS: 10 -> 8 (pool exhausted)
- **Changed:** PREREG §4.1 specified "the first 10 datasets" from OpenML-CC18 ∪ tag:imbalanced. The pool mechanically yields only 8 datasets meeting the criteria (minority rate 1–20%, 2k–200k rows, <=30% missing). The `imbalanced` tag is currently empty on OpenML, so the effective pool was CC18 alone.
- **Reason:** design error in PREREG §4.1 — the specified pool cannot deliver 10 datasets. This was not discoverable without querying OpenML.
- **Decided:** BEFORE seeing any results. No model has been fit; only dataset metadata was inspected.
- **Action taken:** proceeded with all 8 that qualify. The pool was NOT broadened and no threshold was relaxed, to preserve the anti-cherry-picking guarantee.
- **Impact on hypotheses:** none on validity. Reduces breadth of generalisation; to be reported as a limitation.

## 2026-07-14 — Ambiguity resolution: definition of "missing-value rate"
- **Ambiguity:** PREREG §4.1 specified "no missing-value rate above 30%" without defining whether the rate is cell-level (missing cells / rows×features) or instance-level (rows with >=1 missing / rows). This is a gap in the pre-registration, not a change to it.
- **Resolved as:** cell-level, per the literal meaning of "rate" and because the filter's intent is to exclude sparse datasets, not datasets with one sparse column.
- **Disclosure:** the consequence was known at decision time (cell-level retains OpenML dataset 38 `sick`, giving N=8; instance-level would give N=7). The choice was made on the reasoning above, not on the count. Stated openly so the reader can judge.
- **Decided:** BEFORE seeing any results. No model has been fit.
- **Impact on hypotheses:** none.

## 2026-07-14 — Dataset pool: excursion and reversion to PREREG §4.1
- **What was tried:** The pool rule was briefly broadened beyond PREREG §4.1 — all active OpenML datasets with 2 classes, rows 2k–200k, minority 1–20%, INSTANCE-level missing rate <=30%, NumberOfFeatures <= 500, deduplicated by name, first 10 by ascending ID. N_DATASETS was kept at 10. That pool yielded 107 candidates; the first 10 were IDs 310, 312, 316, 958, 962, 971, 976, 977, 978, 980.
- **Why it was abandoned:** 6 of those 10 are OCR/character tasks binarized from multiclass, and three of them (mfeat-morphological, mfeat-fourier, mfeat-factors) are the SAME 2000 handwritten digits represented in three different feature spaces — identical n_rows and identical 0.10 event rate. They are not independent data-generating processes, which the paired analysis assumes. Root cause: "first N by ascending ID" is pathological on a broad pool, because it returns OpenML's oldest datasets, i.e. the classic UCI/OCR sets.
- **Resolution:** Reverted to the pre-registered pool (OpenML-CC18 ∪ tag:imbalanced, filters as written in PREREG §4.1, CELL-LEVEL missing-value rate, sort by ID ascending, take all that qualify) → 8 datasets. N_DATASETS = 8. No new rule was invented; reverting REMOVES a researcher degree of freedom rather than adding one.
- **Bookkeeping disclosure:** A DEVIATIONS.md entry describing the broadened pool was written and then deleted from the working tree before it was ever committed, so no git history was rewritten. Rather than reconstruct that deleted text and pass it off as the contemporaneous record, the full excursion is disclosed here in a single entry. Nothing about the excursion is concealed.
- **Decided:** BEFORE any model was fit. No experimental results existed at any point during this excursion.
- **Impact on hypotheses:** none. Breadth of generalisation is limited to 8 datasets; to be reported as a limitation.
