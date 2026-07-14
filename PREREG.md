# Pre-Registration — Study 1

## Class Imbalance Corrections in Modern Tabular ML: Discrimination, Calibration, and Decisions

**Status:** PRE-REGISTERED — no experiments have been run at the time of this commit.
**Registered:** [DATE — set to the commit date]
**Author:** [NAME]
**Repository rule:** This document is frozen at first commit. Any deviation during the study is recorded in `DEVIATIONS.md` with a timestamp and a reason. Deviations are reported in the final write-up.

---

## 1. Background and motivation

Van den Goorbergh et al. (JAMIA, 2022) showed that for **logistic regression**, class-imbalance corrections (random undersampling, random oversampling, SMOTE) did not improve discrimination (AUROC) but substantially damaged probability calibration, and that gains in sensitivity/specificity trade-offs were reproducible by simple decision-threshold adjustment at zero cost. Carriero et al. (Statistics in Medicine, 2025) extended this line to ML methods and explicitly state that the effect of imbalance corrections on the calibration of **flexible ML models** remains an open question.

This study tests that open question directly on modern tabular learners.

## 2. Research questions

- **RQ1.** Do imbalance corrections improve discrimination (AUROC, AUPRC) for XGBoost and MLP models on tabular data?
- **RQ2.** Do imbalance corrections damage probability calibration (calibration slope, intercept, ECE, Brier) for these models?
- **RQ3.** Is any classification benefit of imbalance corrections reproducible by decision-threshold adjustment alone?
- **RQ4.** Do corrections change the ranking of models under a decision-analytic metric (Net Benefit) relative to AUROC ranking?

## 3. Hypotheses (falsifiable, stated before data)

- **H1.** Imbalance corrections will produce **no meaningful AUROC improvement** (|ΔAUROC| < 0.01 on average) for any model class.
- **H2.** SMOTE and oversampling will **degrade calibration slope away from 1.0 and intercept away from 0.0**, and increase ECE, for all model classes.
- **H3.** The `none + threshold shift` condition will match or exceed the Net Benefit of every correction condition at clinically/operationally relevant thresholds.
- **Falsification criteria:** H1 is falsified if any correction improves mean AUROC by ≥ 0.01 with a 95% CI excluding zero. H2 is falsified if calibration metrics under corrections are statistically indistinguishable from the uncorrected condition. H3 is falsified if any correction beats threshold-shifting on Net Benefit with a CI excluding zero. **All outcomes, including full or partial falsification, will be published.**

## 4. Design (frozen)

### 4.1 Datasets
- Source: **OpenML**. Selection procedure (mechanical, to prevent cherry-picking): binary classification tasks from the OpenML-CC18 benchmark suite plus OpenML tags `imbalanced`, filtered to: minority class between **1% and 20%**, n between **2,000 and 200,000**, no missing-value rate above 30%, tabular (no image/text). Take the **first 10** datasets meeting criteria when sorted by OpenML dataset ID ascending.
- The final dataset list is recorded in `datasets.txt` at selection time, before any model is fit.

### 4.2 Models (fixed hyperparameters — no tuning anywhere)
- **Logistic Regression** (sklearn defaults, `max_iter=5000`) — anchor/replication arm.
- **XGBoost** (`n_estimators=500`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`)
- **MLP** (sklearn, `hidden_layer_sizes=(64,32)`, `early_stopping=True`, `max_iter=500`)

Rationale: hyperparameters are held **constant across conditions**; the estimand is the *difference between conditions*, not peak performance. Tuning per condition would confound the comparison.

### 4.3 Conditions (5)
1. `none` — no correction
2. `rus` — random undersampling to 1:1
3. `ros` — random oversampling to 1:1
4. `smote` — SMOTE (k=5) to 1:1
5. `none + threshold` — no correction; decision threshold set to the event rate (and swept across thresholds for decision-curve analysis)

Corrections are applied **inside the CV training folds only** (leakage guard: resampling never sees validation data).

### 4.4 Resampling & repetition
- 5-fold **stratified** cross-validation × **5 seeds** (seeds 0–4, fixed here).
- Total fits: 10 datasets × 3 models × 4 resampling conditions × 25 fold-seed pairs = **3,000 fits** (threshold condition reuses `none` fits).

### 4.5 Outcome metrics (all reported; none optional)
- Discrimination: **AUROC**, **AUPRC**
- Calibration: **calibration slope**, **calibration intercept**, **ECE (15 equal-frequency bins)**, **Brier score**
- Decision: **Net Benefit** at thresholds {event rate, 0.05, 0.1, 0.2} + full decision curves
- Uncertainty: mean ± 95% CI across seed×fold replicates; dataset-level and pooled summaries.

### 4.6 Analysis plan
- Primary comparison: each correction vs `none`, per model class, paired across fold-seed replicates.
- No significance-hunting: the pre-specified estimands are the H1–H3 contrasts only. Anything else is labeled exploratory.

## 5. Compute & environment
- Single machine (Apple M4 Pro, CPU only). `n_jobs` capped at 8 to avoid memory pressure.
- Environment pinned in `environment.yml`; single-command reproduction via `make reproduce`.

## 6. Scope limits (stated up front)
- Tabular data only; findings do not extend to deep models on unstructured data.
- Fixed hyperparameters; we do not claim results hold under per-condition tuning (see §4.2 rationale).
- Class-weighting approaches (e.g., `scale_pos_weight`) are **out of scope** for v1 and noted as follow-up work.

## 7. Deviations
Any change to the above after first commit → logged in `DEVIATIONS.md`, reported in the paper.
