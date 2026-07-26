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

## 2026-07-14 — Provenance: src/runner.py authored by an unattributed process
- **What happened:** src/runner.py appeared in the working tree during the Task 4 session, authored by a process other than the reviewing agent (most likely a background subagent). Authorship cannot be attributed with certainty.
- **Why it was kept:** the file passes tests/test_runner_leakage.py, whose ability to DETECT a leak was independently proven by a mutation test (a deliberately leaky run_cell was rejected with max sentinel id 10191.0 >= 10000). The guarantee rests on the proven test, not on the authorship of the code.
- **Action:** audited line-by-line against SPEC §3–§5 before any results were used. Audit recorded in the repository.
- **Decided:** BEFORE any results were used.
- **Impact on hypotheses:** none.

## 2026-07-14 — Data limitation: all-missing `TBG` column in dataset 38 `sick`
- **What happened:** OpenML dataset 38 (`sick`) contains a feature column `TBG` that is entirely missing (all-NaN). Median imputation cannot impute a column with no observed values, so `SimpleImputer` (default `keep_empty_features=False`) drops it inside the CV pipeline, emitting a `UserWarning` on every fit.
- **Action:** the fact is logged ONCE per (dataset, column) in `src/runner.py` (`_log_all_nan_columns`) and the repeated `SimpleImputer` warning is suppressed at fit time only (narrowly, by message + `UserWarning`) to keep the ~3,000-cell run readable. The column is NOT dropped by our code and is NOT silently excluded (.cursorrules #3); the drop is the imputer's documented behaviour and is disclosed here.
- **Decided:** BEFORE any results were used.
- **Impact on hypotheses:** none. To be reported in the paper's limitations (dataset 38 effectively has one fewer usable feature).

## 2026-07-14 — Results-directory hygiene (engineering, pre-results)
- **Changed:** (a) tests now write to a temporary directory; a test artifact (dataset_id 999999, from the leakage test's synthetic dataset) had been written into results/cells/ and would have been reported as a failed experimental cell. (b) Per-item probability files moved from results/cells/{cid}.yprob.parquet to results/yprob/{cid}.parquet, because the shared glob namespace silently mixed them with cell results.
- **Reason:** both defects would have corrupted the analysis without raising an error.
- **Decided:** BEFORE any experimental results were used. Only a pilot had been run, and it was discarded and re-run after the fix. No hypothesis-relevant number was inspected before deciding.
- **Impact on hypotheses:** none.

## 2026-07-14 — H1 FAIL for MLP: diagnosis (no design change)
- **Observation:** H1 (|ΔAUROC| < 0.01) FAILED for mlp: ros +0.055, smote +0.049. The 95% intervals span ~0.5, and mlp/none shows negative calibration slopes in some replicates — indicating a collapsing baseline rather than a genuine improvement from resampling.
- **Action taken:** DIAGNOSIS ONLY. No hyperparameter, model, or exclusion rule was changed. MLP remains in the study exactly as pre-registered, with early_stopping=True as specified in PREREG §4.2.
- **Why nothing was changed:** modifying the MLP configuration after seeing that it failed would be a results-dependent design change — precisely what this pre-registration exists to prevent. The failure is reported as a finding, not engineered away.
- **Decided:** AFTER seeing results. Disclosed as such.
- **Impact on hypotheses:** H1 is reported as FAILED for MLP in Table 1. The diagnosis is reported as an exploratory finding and as a limitation of the pre-registered MLP specification.

## 2026-07-14 — MLP early-stopping mechanism verification (read-only, no design change)
- **What was done:** Added `scripts/verify_mlp_mechanism.py`, a read-only diagnostic that EXACTLY reproduces (same dataset, same `StratifiedKFold(shuffle=True, random_state=seed)` split & fold, same frozen `make_pipeline`/hyperparameters from `src/config.py`) the 17 broken `mlp/none` replicates (frozen AUROC < 0.6), the 17 matched `mlp/SMOTE` replicates, and all 25 `mlp/none` replicates on the high-event-rate control dataset jm1 (1053). For each it records `MLPClassifier.n_iter_`, `best_validation_score_`, and the held-out predicted-probability distribution (min/max/mean/std). Output: `results/diagnostics/report6..report9`.
- **Why it is not a deviation from PREREG:** no hyperparameter, model, condition, split, seed, or exclusion rule was changed; the refits are in-memory and discarded, and NOTHING was written to `results/results.parquet`, `results/cells/`, or `results/yprob/`. The reproduced held-out AUROC equals the frozen value in all 42 checked cells (exact match), confirming these are reproductions of the frozen run, not new fits. This diagnostic does not feed the pre-registered H1/H2/H3 tables.
- **Ordering note:** the pre-registered Tables 1–3 (`src/analyze.py`) were regenerated and reported to the author BEFORE this mechanism diagnostic was discussed, correcting the earlier ordering in which diagnostics were seen first.
- **Decided:** AFTER seeing results. Disclosed as such.
- **Impact on hypotheses:** none. The pre-registered result stands exactly as computed; this diagnostic explains the MLP H1 FAIL, it does not replace it. The post-hoc exclusion recompute remains EXPLORATORY and clearly labelled (`scripts/diagnose_mlp.py` Report 5).

## Repository re-creation (2026-07-26)

The GitHub repository was deleted and re-created under the same name on
2026-07-26 to clear a stale cached contributor entry that GitHub Support could
not correct. The git history was pushed back unchanged: all commits, authors,
dates, messages, and the `v1.0.0` tag are identical to the original. No commit
was rewritten, added, or removed. As a consequence the repository's GitHub
creation date now reads 2026-07-26 rather than the original date; the commit
timestamps in the history remain the authoritative record. A `--mirror` backup
of the original repository is retained offline.
