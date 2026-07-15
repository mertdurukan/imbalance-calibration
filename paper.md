# Imbalance Corrections Don't Help Modern Tabular ML — and the One Case Where They Seem To Is a Broken Baseline

*Working draft. All numeric values are final, taken verbatim from the frozen pre-registered analysis (2,400 fits, commit history intact). Prose is drafted for revision, not final submission.*

---

## Abstract

Class-imbalance corrections — SMOTE, random over-sampling, random under-sampling — are near-universal in applied tabular machine learning. The evidence that they help is surprisingly thin, and concentrated on logistic regression. Van den Goorbergh et al. (2022) showed that for logistic regression, imbalance corrections do not improve discrimination and substantially damage calibration; Carriero et al. (2025) extended the analysis to machine-learning models but stated explicitly that the effect on the calibration of flexible ML methods remained unknown.

We pre-registered and executed a factorial study — 8 OpenML datasets × 3 model families (logistic regression, gradient-boosted trees, multilayer perceptron) × 4 conditions × 25 replicates = 2,400 model fits — measuring discrimination, calibration, and decision quality. We find that corrections do not improve discrimination for logistic regression or gradient-boosted trees (|ΔAUROC| < 0.01 across all six contrasts), that they severely damage calibration in every model family (calibration-in-the-large moves from approximately 0 to approximately −2; expected calibration error rises up to fivefold), and that in all 36 decision-analytic contrasts, no correction outperforms simply shifting the decision threshold on the uncorrected model.

The one apparent exception — the MLP, where corrections raise AUROC by up to 0.05 on average — is not a benefit of resampling. We show mechanistically that accuracy-monitored early stopping halts training after roughly twelve iterations on low-prevalence data, because a majority-class predictor already achieves validation accuracy equal to one minus the event rate. Resampling accidentally restores a usable training signal. The apparent gain is repair of a broken baseline, not improvement.

We further find that under standard decision thresholds, corrections can make models actively harmful: random under-sampling produced a negative net benefit — worse than treating no one — in 118 of 200 MLP replicates at a threshold of 0.20, converting a useful uncorrected model into a harmful one in 89 of them.

The recommendation is simple: do not resample. If the decision operating point is wrong, move the threshold. It is free, reversible, and leaves the model's probabilities intact.

---

## 1. Introduction

Applied tabular machine learning treats class-imbalance correction as hygiene. Faced with a dataset where the positive class is rare, the reflex is to rebalance — most often with SMOTE — before training. The practice is codified in tutorials, default pipelines, and reviewer expectations.

The empirical basis for this reflex is weaker than its ubiquity suggests. The most careful evidence concerns logistic regression, where van den Goorbergh et al. (2022) demonstrated that imbalance corrections do not improve the area under the ROC curve, damage probability calibration, and that any apparent classification benefit is reproducible for free by adjusting the decision threshold. Carriero et al. (2025) carried the question into machine learning, but were explicit that the effect of imbalance corrections on the calibration of flexible ML models was not yet known.

That gap is the target of this study. We ask, for three representative model families spanning linear, tree-ensemble, and neural approaches: do imbalance corrections improve discrimination, what do they do to calibration, and do they improve the quality of the decisions the model is ultimately used to make?

We answer all three under pre-registration, and we report an outcome that falsifies part of our own stated hypothesis. That falsification turns out to be the most informative result in the study, because tracing its cause reveals the mechanism by which practitioners come to believe that imbalance corrections work.

## 2. Related work

*(to be written — anchor on the two papers below, paraphrased, minimal quotation)*

Van den Goorbergh et al. (JAMIA, 2022): for clinical prediction with logistic regression, imbalance corrections did not improve discrimination, degraded calibration, and the classification gains were recoverable by threshold adjustment.

Carriero et al. (Statistics in Medicine, 2025): extended the analysis toward ML methods and named the open question — the calibration behaviour of flexible ML methods under imbalance corrections remained undetermined.

The contribution here is a factorial study across three model families with a decision-analytic outcome (net benefit), plus a mechanistic account of the one case where corrections appear to help.

## 3. Methods

### 3.1 Pre-registration
The full design — datasets, models, conditions, metrics, hypotheses, and falsification criteria — was frozen before any data was queried. The pre-registration is the first commit in the repository, timestamped before every result. Two deviations arose and are logged: the pre-registered pool yielded eight datasets rather than the intended ten (the pool was not broadened, to preserve the mechanical selection), and the phrase "missing-value rate" was resolved to a cell-level definition. Both were decided before any model was fit.

### 3.2 Datasets
Eight datasets were selected mechanically from OpenML: binary targets, minority-class prevalence between 1% and 20%, between 2,000 and 200,000 rows, no more than 30% missing cells, sorted by dataset ID ascending, with all qualifying datasets taken. There was no hand-curation; the dataset list was committed before any model was fit. The datasets span software-defect prediction (jm1, kc1), medical screening (sick), bank marketing, environmental sensing (ozone), customer churn, ad classification, and remote-sensing land cover (wilt), with event rates from 0.054 to 0.193.

### 3.3 Models
Logistic regression, gradient-boosted trees (XGBoost), and a multilayer perceptron, each with fixed hyperparameters transcribed from the pre-registration. Hyperparameters are held constant across all conditions: the quantity being estimated is the *difference between conditions*, not peak performance, and per-condition tuning would confound that difference. No hyperparameter search occurs anywhere in the codebase; a contract test enforces this.

### 3.4 Conditions
Four training conditions — no correction, random under-sampling, random over-sampling, and SMOTE — each balancing the training data to a 1:1 ratio. A fifth analysis-only condition, *no correction with a shifted decision threshold*, is computed at analysis time from the saved per-item probabilities of the uncorrected model.

### 3.5 Leakage guard, proven rather than asserted
Resampling is confined to training folds by placing it inside the cross-validation pipeline. We do not merely claim this. A mutation test establishes that our leakage test can detect a violation: a deliberately leaky implementation, which resamples the full dataset before splitting, was rejected because a held-out sentinel identifier reached the model's training data. The guard is verified, not assumed.

### 3.6 Metrics
Discrimination is measured by AUROC and average precision. Calibration is measured by the calibration slope and calibration-in-the-large (fitted by a generalized linear model with the linear predictor as an offset), by expected calibration error using fifteen equal-mass bins, and by the Brier score. Equal-mass bins are used deliberately: with concentrated probabilities, equal-width bins leave most bins empty and understate calibration error. Decision quality is measured by net benefit (Vickers and Elkin) across a range of thresholds. Every reported quantity carries a 95% interval from the replicate distribution; no t-tests are computed across cross-validation folds, which are not independent.

## 4. Results

### 4.1 Discrimination: no gain (H1)

For the two model families without early stopping, imbalance corrections leave discrimination essentially unchanged. Every paired ΔAUROC versus the uncorrected model falls in the third decimal place: logistic regression ranges from −0.0022 to +0.0008, and gradient-boosted trees from −0.0097 to −0.0003. Both satisfy the pre-registered criterion of |ΔAUROC| < 0.01.

The MLP is the sole exception, with mean ΔAUROC up to +0.055 — failing the criterion. Section 4.4 shows this failure is not a benefit of correction.

*(Table 1 here.)*

### 4.2 Calibration: destroyed (H2)

Corrections damage calibration in every model family. The clearest signal is calibration-in-the-large, which is zero for a perfectly calibrated model. The uncorrected logistic-regression baseline is nearly perfect at +0.013, as maximum-likelihood theory predicts. After correction it moves to approximately −2.2 across all three resamplers — a large, systematic overestimation of risk. Expected calibration error rises from 0.034 to as high as 0.19, roughly fivefold. The same qualitative pattern holds for the tree ensemble and the MLP.

Twelve of thirteen calibration verdicts confirm the hypothesis. The single exception (the gradient-boosted-tree/SMOTE intercept) is reported rather than concealed: its magnitude fell while its sign flipped. Figure 2 shows the corrected curves bowing below the diagonal — the visual signature of over-confident, over-high predicted risk.

*(Table 2, Figure 2 here.)*

### 4.3 Decisions: threshold shifting is free and sufficient (H3)

In all 36 decision-analytic contrasts — three models, three corrections, four thresholds — no correction beats the uncorrected model with a shifted decision threshold, with a 95% interval excluding zero. Everything a correction buys in decision terms, a free change of the decision threshold buys as well. Figure 1 shows the uncorrected curve lying on or above every corrected curve across the threshold range.

*(Table 3, Figure 1 here.)*

### 4.4 The MLP "gain" is a broken baseline, with a measured mechanism

The MLP is the only model configured with accuracy-monitored early stopping. Seventeen of two hundred uncorrected MLP replicates have AUROC below 0.6, with a minimum of 0.385 — worse than chance. All seventeen fall on the two lowest-prevalence datasets, wilt (0.054) and ozone (0.063); no corrected condition produces a single broken replicate.

Exact reproduction of those seventeen cells, holding seed, fold, and hyperparameters fixed, reveals the cause. The broken replicates stop training after twelve to thirteen iterations, with a best internal-validation accuracy of 0.936 to 0.946 and a maximum predicted probability never exceeding 0.515. The matched SMOTE replicates train for eighteen to fifty-four iterations, reach validation accuracy near 0.99, and produce confident probabilities.

The decisive number is the validation accuracy. The majority-class accuracy of the two affected datasets is 0.9369 (ozone) and 0.9461 (wilt), and the broken replicates' validation scores sit exactly on that band. The network is predicting the majority class, achieving the accuracy a constant predictor would achieve, and early stopping — which monitors accuracy — reads this as convergence and halts an untrained model. Resampling to 1:1 restores an informative accuracy signal, training proceeds, and the model learns.

Practitioners observe SMOTE deliver up to +0.48 AUROC on these replicates. What they observe is a race in which the baseline was disabled before the start. Figure 3 makes the mechanism visible at a glance: the broken group's iteration count is pinned at the floor while its validation accuracy lands precisely on the majority-class line.

*(Figure 3 here.)*

### 4.5 Corrections can make models actively harmful

Beyond failing to help, corrections can harm. Using the sign of the pre-registered net benefit — a negative value means the model is worse than treating no one — random under-sampling on the MLP at a threshold of 0.20 produced negative net benefit in 118 of 200 replicates, and in 89 of them it converted a model that had been useful without correction into a harmful one. Logistic regression under the same correction and threshold flipped 82 of 200. The uncorrected model was almost never harmful for logistic regression or gradient-boosted trees (0 of 200 at every threshold). Gradient-boosted trees were the most robust; SMOTE was the least damaging correction for the ML models.

*(Table 4 here — descriptive, no verdict.)*

## 5. Discussion

For a practitioner, the operational takeaway is a diagnostic reversal. If imbalance correction appears to help your model, the first hypothesis should be that your baseline is broken — as it was here for the MLP, silently, through a default early-stopping rule interacting with low prevalence. The apparent gain may be repair, not improvement, and the repair is better achieved by fixing the training procedure than by distorting the data.

For the field, the mechanism explains the folklore. The persistent belief that resampling helps does not require resampling to help. It requires only that default training procedures fail quietly on imbalanced data in a way that resampling happens to mask. Discrimination is unchanged for models that were training correctly all along; the "gains" accrue precisely where the baseline was broken.

The consistent cost is calibration. Every correction, in every model family, moved calibration-in-the-large sharply negative — a systematic overestimation of risk. If the model's probabilities inform a decision, and if they do not one should ask why the model exists, then corrections make the decision worse, and at standard thresholds can make it actively harmful.

The alternative is threshold adjustment. It achieves everything the corrections achieve in decision terms, costs nothing, is reversible, and leaves the probabilities untouched.

## 6. Limitations

The study covers eight tabular datasets. The pre-registered pool could not supply ten; we took all eight that qualified and declined to broaden the pool, to preserve the mechanical, cherry-picking-proof selection. The findings are tabular and do not extend to unstructured data.

Hyperparameters were fixed. We do not claim these results hold under per-condition tuning; holding them fixed is a deliberate design choice that isolates the effect of the correction from the effect of tuning. The anticipated objection — that better MLP settings would remove the broken baseline, and with it the apparent SMOTE gain — is exactly the point. The configuration used is a standard default. The finding is that under standard defaults, on imbalanced data, the apparent benefit of correction is an artifact of a silently mistrained baseline.

Class-weighting approaches, which reweight the loss rather than resample the data, are out of scope and are the natural next study. One dataset contains an entirely missing column, which the imputer drops; this is disclosed rather than silently handled.

## 7. Reproducibility

A single command reproduces the entire study from a clean checkout. The pre-registration, the deviations log, every per-item probability, and all 2,400 cells are in the repository. The leakage guard is mutation-tested; every reported quantity carries a 95% interval; failed cells, had there been any, would be reported rather than dropped.

---

## Appendix — deferred writing tasks

- §2 Related work: write out in prose, ≤15-word quotes, mostly paraphrase.
- Insert Table 1–4 and Figure 1–3 at marked points; Figure S1 (per-dataset decision curves) as supplement.
- Retitle for venue. Candidate framing: a short methods/─position paper for a stats-in-ML or clinical-ML venue where the Van Calster / van Smeden / Carriero line of work lives.
- Circulate to the authors whose open question this closes — they are the natural amplifiers.
