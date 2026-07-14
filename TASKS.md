# TASKS — ordered implementation plan for Cursor

Work **one task at a time**. Do not start task N+1 until task N's acceptance criteria pass.
Each task below contains a prompt you can paste into Cursor verbatim.

> **Golden rule for every task:** open a fresh Cursor composer session, and start the
> prompt with: *"Read `.cursorrules`, `SPEC.md`, and `METRICS.md` first. Then:"*
> Cursor forgets context. This line is your context anchor.

---

## Task 0 — Scaffold ✅ acceptance: `make test` runs (and fails, with 0 tests collected is NOT ok — see below)

**Prompt:**
> Read `.cursorrules`, `SPEC.md`, and `METRICS.md` first. Then create the repository
> skeleton exactly as specified in SPEC.md §1: all directories, all empty modules with
> their docstrings and type-hinted function signatures from SPEC.md §3, raising
> `NotImplementedError` in every body. Create `environment.yml` (python 3.11, pinned:
> scikit-learn, imbalanced-learn, xgboost, statsmodels, pandas, pyarrow, openml, pytest,
> matplotlib). Create the `Makefile` with the targets in SPEC.md §6. Do not implement any
> logic yet.

**Acceptance:** `make setup` succeeds; `python -c "import src.runner"` succeeds;
every function raises `NotImplementedError`.

---

## Task 1 — Metrics (TEST FIRST) ✅ acceptance: `make test` passes, ≥ 8 tests

⚠️ **This is the highest-risk module. It is implemented first, and tests come before code.**

**Prompt:**
> Read `.cursorrules`, `SPEC.md`, and `METRICS.md` first. Then:
> Step 1: write `tests/test_metrics.py` implementing EVERY known-answer test described in
> METRICS.md (§2 slope/intercept convergence, §3 ECE three cases, §4 Net Benefit against a
> hand-computed 10-item example you construct in the test). Do not write the implementation yet.
> Step 2: show me the tests and STOP. I will review before you implement.

Then, after review:
> Now implement `src/metrics.py` to pass those tests. Use statsmodels GLM with `offset=`
> for the calibration intercept as METRICS.md §2 specifies. Do not change any test to make
> it pass.

**Acceptance:** all metric tests green. Calibration slope on a well-specified synthetic
model is 1.00 ± 0.05. Net Benefit matches your hand-computed example exactly.

---

## Task 2 — Dataset selection ✅ acceptance: `datasets.txt` exists, has 10 IDs, is committed

**Prompt:**
> Read `.cursorrules` and SPEC.md §3 first. Then implement `src/datasets.py::select_datasets`
> exactly per PREREG §4.1: query OpenML, filter on the criteria in `config.py`, sort by
> dataset ID ASCENDING, take the first `N_DATASETS`. It must be deterministic — running it
> twice returns the identical list. Implement `make datasets` so it REFUSES to run if
> `datasets.txt` already exists (print an error and exit 1). Then implement `load_dataset`.

**Acceptance:** `make datasets` produces exactly 10 IDs. Running it again exits with an
error. **Commit `datasets.txt` immediately** — this is the anti-cherry-picking guarantee.
Then eyeball the list: if a dataset looks wrong, you may NOT remove it. Log it in
`DEVIATIONS.md` and discuss.

---

## Task 3 — Leakage contract test (TEST BEFORE PIPELINE) ✅ acceptance: test fails, then passes

**Prompt:**
> Read `.cursorrules` first. Then write `tests/test_leakage.py` with these tests:
> 1. `test_resampler_is_inside_pipeline`: `make_pipeline("logreg","smote",0)` returns an
>    `imblearn.pipeline.Pipeline` whose resampler step is an instance of `SMOTE`.
> 2. `test_no_resampling_on_none`: the "none" pipeline contains NO resampler step.
> 3. `test_validation_fold_untouched`: construct a dataset with a unique sentinel value in
>    each row. Run cross-validation. Assert the number of items scored in each validation
>    fold equals the original fold size — i.e. resampling never inflated the validation set.
> 4. `test_class_balance_after_resample`: after SMOTE inside the pipeline, the TRAINING data
>    seen by the model is balanced, but the VALIDATION data retains the original imbalance.
> Write the tests first, then implement `src/conditions.py` and `src/models.py` to pass them.

**Acceptance:** all 4 leakage tests green. Test 3 and 4 are the ones that actually matter —
they are what stop the single most common error in this literature.

---

## Task 4 — Pilot ✅ acceptance: one end-to-end result row, hand-checked

**Prompt:**
> Read `SPEC.md` §3–§5 first. Then implement `src/runner.py` (`cell_id`, `run_cell`,
> `run_all`) with the caching in SPEC.md §5 and the exact output schema in SPEC.md §4,
> including saving per-item probabilities to `y_prob_path`. Then implement `make pilot`:
> the FIRST dataset in datasets.txt × xgboost × all 4 conditions × seed 0 × all 5 folds.
> Print nothing but a completion summary; results go to `results/cells/`.

**Acceptance — do this by hand, do not delegate it:**
- Open `results/results.parquet`. 20 rows (4 conditions × 5 folds). `status == "ok"` for all.
- **Sanity check:** the `smote` rows should show `cal_intercept < 0` (overestimation of
  risk). If they don't, something is wrong — SMOTE is *supposed* to break calibration this
  way. If the sign is wrong, your metric implementation is wrong, not the world.
- Time one cell. Multiply by 3,000. **If the projection exceeds ~8 hours, stop and tell me
  before scaling up.**

---

## Task 5 — Full run ✅ acceptance: 3,000 rows, 0 unexplained failures

**Prompt:**
> Run `make reproduce`. It must be resumable — kill it halfway and restart; it must not
> recompute completed cells. Report: total rows, count by `status`, and the full text of
> every distinct error for failed cells.

**Acceptance:** every `status == "failed"` row has a written explanation in
`DEVIATIONS.md`. **Failed cells are reported in the paper, never dropped.**

---

## Task 6 — Analysis ✅ acceptance: the three H1/H2/H3 tables

**Prompt:**
> Read `METRICS.md` §4 and §5 first. Then implement `src/analyze.py` to produce, from
> `results.parquet` and the saved `y_prob` files:
> - **Table 1 (H1):** paired ΔAUROC and ΔAUPRC vs `none`, per model × condition, with 95% CIs.
> - **Table 2 (H2):** calibration slope, intercept, ECE, Brier per model × condition, with CIs.
> - **Table 3 (H3):** Net Benefit at each threshold, including the `none_threshold` arm
>   computed from saved probabilities per METRICS.md §4.1.
> - **Figure 1:** decision curves (all conditions + treat-all + treat-none reference lines).
> - **Figure 2:** calibration curves for one representative dataset, all 4 conditions.
> Use paired differences per METRICS.md §5. Do NOT run t-tests across CV folds.

**Acceptance:** each table maps 1:1 to a pre-registered hypothesis. Nothing extra in the
main tables. Exploratory findings go in a section explicitly labeled "Exploratory".

---

## Anti-drift checklist — run this before EVERY commit

- [ ] Did I add a hyperparameter search anywhere? → revert
- [ ] Did I drop or skip a dataset/fold/condition? → revert
- [ ] Did I change a metric definition or a test? → revert
- [ ] Did I edit `PREREG.md`? → revert, use `DEVIATIONS.md`
- [ ] Is every number in the output accompanied by a CI? → if not, fix
- [ ] Does `make reproduce` still work from a clean checkout?
