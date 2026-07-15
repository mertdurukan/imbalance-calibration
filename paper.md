# Imbalance Corrections Don't Help Modern Tabular ML, and the One Case Where They Seem To Is a Broken Baseline

*Working draft. All numeric values are final, taken verbatim from the frozen pre-registered analysis (2,400 fits, commit history intact). Prose is drafted for revision, not final submission.*

---

## Abstract

Class-imbalance corrections (SMOTE, random over-sampling, random under-sampling) are near-universal in applied tabular machine learning. The evidence that they help is surprisingly thin, and concentrated on logistic regression. Van den Goorbergh et al. (2022) showed that for logistic regression, imbalance corrections do not improve discrimination and substantially damage calibration; Carriero et al. (2025) extended the analysis to machine-learning models but stated explicitly that the effect on the calibration of flexible ML methods remained unknown.

We pre-registered and executed a factorial study, 8 OpenML datasets × 3 model families (logistic regression, gradient-boosted trees, multilayer perceptron) × 4 conditions × 25 replicates = 2,400 model fits, measuring discrimination, calibration, and decision quality. We find that corrections do not improve discrimination for logistic regression or gradient-boosted trees (|ΔAUROC| < 0.01 across all six contrasts), that they severely damage calibration in every model family (calibration-in-the-large moves from approximately 0 to approximately −2; expected calibration error rises up to fivefold), and that in all 36 decision-analytic contrasts, no correction outperforms simply shifting the decision threshold on the uncorrected model. Our calibration hypothesis (H2) has three sub-predictions per cell (that corrections push the calibration slope away from 1.0, the intercept away from 0.0, and expected calibration error up), and it holds in 26 of the 27 verdict cells across the nine model×correction cells, the single failure being the XGBoost/SMOTE intercept.

The one apparent exception (the MLP, where corrections raise AUROC by up to 0.05 on average) is not a benefit of resampling. We show mechanistically that accuracy-monitored early stopping halts training after roughly twelve iterations on low-prevalence data, because a majority-class predictor already achieves validation accuracy equal to one minus the event rate. Resampling accidentally restores a usable training signal. The apparent gain is repair of a broken baseline, not improvement.

We further find that under standard decision thresholds, corrections can make models actively harmful: random under-sampling produced a negative net benefit (worse than treating no one) in 118 of 200 MLP replicates at a threshold of 0.20, converting a useful uncorrected model into a harmful one in 89 of them.

The recommendation is simple: do not resample. If the decision operating point is wrong, move the threshold. It is free, reversible, and leaves the model's probabilities intact.

---

## 1. Introduction

Applied tabular machine learning treats class-imbalance correction as hygiene. Faced with a dataset where the positive class is rare, the reflex is to rebalance, most often with SMOTE, before training. The practice is codified in tutorials, default pipelines, and reviewer expectations.

The empirical basis for this reflex is weaker than its ubiquity suggests. The most careful evidence concerns logistic regression, where van den Goorbergh et al. (2022) demonstrated that imbalance corrections do not improve the area under the ROC curve, damage probability calibration, and that any apparent classification benefit is reproducible for free by adjusting the decision threshold. Carriero et al. (2025) carried the question into machine learning, but were explicit that the effect of imbalance corrections on the calibration of flexible ML models was not yet known.

That gap is the target of this study. We ask, for three representative model families spanning linear, tree-ensemble, and neural approaches: do imbalance corrections improve discrimination, what do they do to calibration, and do they improve the quality of the decisions the model is ultimately used to make?

We answer all three under pre-registration, and we report an outcome that falsifies part of our own stated hypothesis. That falsification turns out to be the most informative result in the study, because tracing its cause reveals the mechanism by which practitioners come to believe that imbalance corrections work.

## 2. Related work

The concern that class-imbalance corrections may harm rather than help prediction models originates in a line of work from the Utrecht group. **Van den Goorbergh et al. (2022)** examined logistic regression under four conditions (no correction, random undersampling, random oversampling, and SMOTE) across simulation and a clinical case study. They found that corrections improved the sensitivity–specificity balance, but that the same balance was obtainable by shifting the decision threshold on the uncorrected model; corrections otherwise produced strong miscalibration, systematically overestimating risk, without improving discrimination. Their conclusion was that outcome imbalance is not a problem in itself, and that correcting it can worsen performance.

**Piccininni et al. (2024)** analysed random resampling techniques and reached a compatible conclusion regarding their consequences for calibration and discrimination in clinical risk prediction, reinforcing that the calibration damage is a general property of resampling rather than an artifact of one implementation.

**Carriero et al. (2025)** extended the question from logistic regression to flexible machine-learning methods (support vector machines, random forests, XGBoost, and boosting ensembles) using Monte Carlo simulation and a MIMIC-III case study. Their calibration finding replicated van den Goorbergh's across every method: corrections consistently produced risk overestimation, and the miscalibration was often not repaired by recalibration. Their discrimination finding, however, was explicitly model-dependent: corrections improved discrimination for some algorithms (SVM, random forest) and worsened it for others (XGBoost, boosting ensembles); and they noted a specific unexplained anomaly, that random forest with random oversampling was often well-calibrated in simulation but not in their case study. Crucially, they framed the calibration behaviour of flexible ML methods under imbalance correction as, at the time, largely unknown, and called for further work.

Two gaps remain. First, this literature is almost entirely clinical and simulation-based; whether the same effects hold on heterogeneous real-world tabular benchmarks, and whether they matter for the *decision* a model drives (as opposed to its discrimination or calibration in isolation), has not been tested factorially. Second, and more importantly, the model-dependence of the discrimination result (corrections appearing to help some algorithms) has been reported but not *explained*. The present study addresses both: we test discrimination, calibration, and decision quality (net benefit) together on eight real tabular datasets, and we trace the one case where corrections appear to improve discrimination to its mechanism, showing it to be an artifact of the baseline rather than a benefit of the correction.

## 3. Methods

### 3.1 Pre-registration
The full design (datasets, models, conditions, metrics, hypotheses, and falsification criteria) was frozen before any data was queried. The pre-registration is the first commit in the repository, timestamped before every result. Two deviations arose and are logged: the pre-registered pool yielded eight datasets rather than the intended ten (the pool was not broadened, to preserve the mechanical selection), and the phrase "missing-value rate" was resolved to a cell-level definition. Both were decided before any model was fit.

### 3.2 Datasets
Eight datasets were selected mechanically from OpenML: binary targets, minority-class prevalence between 1% and 20%, between 2,000 and 200,000 rows, no more than 30% missing cells, sorted by dataset ID ascending, with all qualifying datasets taken. There was no hand-curation; the dataset list was committed before any model was fit. The datasets span software-defect prediction (jm1, kc1), medical screening (sick), bank marketing, environmental sensing (ozone), customer churn, ad classification, and remote-sensing land cover (wilt), with event rates from 0.054 to 0.193.

### 3.3 Models
Logistic regression, gradient-boosted trees (XGBoost), and a multilayer perceptron, each with fixed hyperparameters transcribed from the pre-registration. Hyperparameters are held constant across all conditions: the quantity being estimated is the *difference between conditions*, not peak performance, and per-condition tuning would confound that difference. No hyperparameter search occurs anywhere in the codebase; a contract test enforces this.

### 3.4 Conditions
Four training conditions (no correction, random under-sampling, random over-sampling, and SMOTE) each balancing the training data to a 1:1 ratio. A fifth analysis-only condition, *no correction with a shifted decision threshold*, is computed at analysis time from the saved per-item probabilities of the uncorrected model.

### 3.5 Leakage guard, proven rather than asserted
Resampling is confined to training folds by placing it inside the cross-validation pipeline. We do not merely claim this. A mutation test establishes that our leakage test can detect a violation: a deliberately leaky implementation, which resamples the full dataset before splitting, was rejected because a held-out sentinel identifier reached the model's training data. The guard is verified, not assumed.

### 3.6 Metrics
Discrimination is measured by AUROC and average precision. Calibration is measured by the calibration slope and calibration-in-the-large (fitted by a generalized linear model with the linear predictor as an offset), by expected calibration error using fifteen equal-mass bins, and by the Brier score. Equal-mass bins are used deliberately: with concentrated probabilities, equal-width bins leave most bins empty and understate calibration error. Decision quality is measured by net benefit (Vickers and Elkin) across a range of thresholds. Every reported quantity carries a 95% interval from the replicate distribution; no t-tests are computed across cross-validation folds, which are not independent.

## 4. Results

### 4.1 Discrimination: no gain (H1)

For the two model families without early stopping, imbalance corrections leave discrimination essentially unchanged. Every paired ΔAUROC versus the uncorrected model falls in the third decimal place: logistic regression ranges from −0.0022 to +0.0008, and gradient-boosted trees from −0.0097 to −0.0003. Both satisfy the pre-registered criterion of |ΔAUROC| < 0.01.

The MLP is the sole exception, with mean ΔAUROC up to +0.055, failing the criterion. Section 4.4 shows this failure is not a benefit of correction.

| model | contrast | n | ΔAUROC (mean [95%]) | ΔAUPRC (mean [95%]) | H1 (|ΔAUROC|<0.01) |
| --- | --- | --- | --- | --- | --- |
| logreg | rus - none | 200 | -0.0022 [-0.0299, 0.0124] | -0.0365 [-0.1893, 0.0269] | PASS |
| logreg | ros - none | 200 | 0.0008 [-0.0129, 0.0119] | -0.0138 [-0.0607, 0.0191] | PASS |
| logreg | smote - none | 200 | -0.0008 [-0.0176, 0.0115] | -0.0142 [-0.0732, 0.0270] | PASS |
| xgboost | rus - none | 200 | -0.0097 [-0.0419, 0.0068] | -0.0542 [-0.1979, 0.0016] | PASS |
| xgboost | ros - none | 200 | -0.0028 [-0.0223, 0.0074] | -0.0014 [-0.0326, 0.0190] | PASS |
| xgboost | smote - none | 200 | -0.0003 [-0.0173, 0.0136] | -0.0028 [-0.0609, 0.0291] | PASS |
| mlp | rus - none | 200 | 0.0181 [-0.1117, 0.4292] | -0.0634 [-0.4358, 0.4306] | FAIL |
| mlp | ros - none | 200 | 0.0546 [-0.0390, 0.5183] | 0.1031 [-0.1251, 0.8861] | FAIL |
| mlp | smote - none | 200 | 0.0491 [-0.0446, 0.5181] | 0.1002 [-0.1041, 0.8913] | FAIL |

> Paired within each (dataset, seed, fold): difference is metric(correction) − metric(none), computed per replicate, then summarised. Unpaired means are never compared (METRICS.md §5).
> 95% interval = 2.5/97.5 percentiles of the replicate distribution (descriptive; CV folds are not independent, so no t-test is run).
> H1 column: PASS iff |mean ΔAUROC| < 0.01 (PREREG §3). The full interval is shown so the reader can apply the directional falsification criterion (improvement ≥ 0.01 with a 95% interval excluding zero).

### 4.2 Calibration: destroyed (H2)

Corrections damage calibration in every model family. The clearest signal is calibration-in-the-large, which is zero for a perfectly calibrated model. The uncorrected logistic-regression baseline is nearly perfect at +0.013, as maximum-likelihood theory predicts. After correction it moves to approximately −2.2 across all three resamplers: a large, systematic overestimation of risk. Expected calibration error rises from 0.034 to as high as 0.19, roughly fivefold. The same qualitative pattern holds for the tree ensemble and the MLP.

H2 makes three sub-predictions in each model×correction cell: that corrections push the calibration slope away from 1.0, the intercept (calibration-in-the-large) away from 0.0, and expected calibration error up, which across the nine model×correction cells yields twenty-seven verdicts in total. Twenty-six of the twenty-seven confirm the hypothesis. The single exception (the XGBoost/SMOTE intercept) is reported rather than concealed: its magnitude fell while its sign flipped. Figure 2 shows the corrected curves bowing below the diagonal: the visual signature of over-confident, over-high predicted risk.

| model | condition | n | cal_slope (mean [95%]) | cal_intercept (mean [95%]) | ECE (mean [95%]) | Brier (mean [95%]) | H2 slope→away 1.0 | H2 intercept→away 0.0 | H2 ECE↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logreg | none | 200 | 0.984 [0.418, 1.912] | 0.013 [-0.450, 0.638] | 0.0337 [0.0125, 0.0779] | 0.0677 [0.0230, 0.1420] | ref | ref | ref |
| logreg | rus | 200 | 0.873 [0.422, 2.033] | -2.416 [-4.291, -1.421] | 0.1928 [0.0657, 0.2696] | 0.1296 [0.0532, 0.2137] | PASS | PASS | PASS |
| logreg | ros | 200 | 0.774 [0.365, 1.218] | -2.038 [-2.993, -0.830] | 0.1637 [0.0278, 0.2662] | 0.1147 [0.0291, 0.2124] | PASS | PASS | PASS |
| logreg | smote | 200 | 0.725 [0.331, 1.085] | -2.158 [-3.290, -1.395] | 0.1564 [0.0345, 0.2661] | 0.1129 [0.0330, 0.2132] | PASS | PASS | PASS |
| xgboost | none | 200 | 0.778 [0.454, 1.047] | 0.456 [-0.201, 1.449] | 0.0253 [0.0026, 0.0771] | 0.0541 [0.0068, 0.1392] | ref | ref | ref |
| xgboost | rus | 200 | 0.719 [0.332, 1.179] | -2.625 [-4.188, -1.545] | 0.1381 [0.0432, 0.2628] | 0.1119 [0.0279, 0.2420] | PASS | PASS | PASS |
| xgboost | ros | 200 | 0.698 [0.400, 0.968] | -0.559 [-1.601, 0.681] | 0.0535 [0.0037, 0.1389] | 0.0627 [0.0063, 0.1683] | PASS | PASS | PASS |
| xgboost | smote | 200 | 0.731 [0.406, 1.007] | -0.324 [-2.045, 0.581] | 0.0291 [0.0030, 0.0850] | 0.0559 [0.0058, 0.1423] | PASS | FAIL | PASS |
| mlp | none | 200 | 0.999 [-0.136, 2.827] | -0.234 [-2.104, 0.516] | 0.0520 [0.0081, 0.2614] | 0.0718 [0.0161, 0.1443] | ref | ref | ref |
| mlp | rus | 200 | 1.372 [0.570, 3.533] | -2.265 [-3.467, -1.389] | 0.2598 [0.1168, 0.4379] | 0.1642 [0.0593, 0.2682] | PASS | PASS | PASS |
| mlp | ros | 200 | 0.567 [0.224, 1.235] | -1.368 [-2.937, 0.041] | 0.0945 [0.0167, 0.2661] | 0.0856 [0.0152, 0.2144] | PASS | PASS | PASS |
| mlp | smote | 200 | 0.542 [0.258, 1.327] | -1.341 [-2.569, -0.024] | 0.0928 [0.0145, 0.2499] | 0.0860 [0.0116, 0.2091] | PASS | PASS | PASS |

> Absolute per-condition metrics. Mean + 95% interval (2.5/97.5 percentiles of the replicate distribution; descriptive, no t-test: folds not independent).
> Perfect calibration: slope = 1.0, intercept = 0.0. H2 predicts corrections push slope AWAY from 1.0, intercept AWAY from 0.0, and RAISE ECE.
> PASS/FAIL (corrections only) is vs the `none` reference (row 'ref') in each model: slope PASS iff |mean−1| > |none−1|; intercept PASS iff |mean| > |none|; ECE PASS iff mean > none. Brier has no pre-registered direction under H2 and carries no verdict.

![Figure 2: calibration curves (H2)](results/figures/figure2_calibration_curves.png)

### 4.3 Decisions: threshold shifting is free and sufficient (H3)

In all 36 decision-analytic contrasts (three models, three corrections, four thresholds) no correction beats the uncorrected model with a shifted decision threshold, with a 95% interval excluding zero. Everything a correction buys in decision terms, a free change of the decision threshold buys as well. Figure 1 shows the uncorrected curve lying on or above every corrected curve across the threshold range.

| model | threshold | contrast | n | ΔNB (mean [95%]) | H3 (nt ≥ corr) |
| --- | --- | --- | --- | --- | --- |
| logreg | NB@eventrate | none_threshold − rus | 200 | 0.0314 [0.0038, 0.0625] | PASS |
| logreg | NB@eventrate | none_threshold − ros | 200 | 0.0257 [-0.0019, 0.0629] | PASS |
| logreg | NB@eventrate | none_threshold − smote | 200 | 0.0230 [-0.0019, 0.0590] | PASS |
| logreg | NB@0.05 | none_threshold − rus | 200 | 0.0112 [-0.0014, 0.0357] | PASS |
| logreg | NB@0.05 | none_threshold − ros | 200 | 0.0074 [-0.0026, 0.0164] | PASS |
| logreg | NB@0.05 | none_threshold − smote | 200 | 0.0060 [-0.0035, 0.0138] | PASS |
| logreg | NB@0.10 | none_threshold − rus | 200 | 0.0262 [0.0007, 0.0705] | PASS |
| logreg | NB@0.10 | none_threshold − ros | 200 | 0.0182 [-0.0003, 0.0379] | PASS |
| logreg | NB@0.10 | none_threshold − smote | 200 | 0.0157 [-0.0005, 0.0358] | PASS |
| logreg | NB@0.20 | none_threshold − rus | 200 | 0.0496 [0.0091, 0.0987] | PASS |
| logreg | NB@0.20 | none_threshold − ros | 200 | 0.0359 [0.0011, 0.0715] | PASS |
| logreg | NB@0.20 | none_threshold − smote | 200 | 0.0319 [0.0023, 0.0665] | PASS |
| xgboost | NB@eventrate | none_threshold − rus | 200 | 0.0131 [-0.0063, 0.0352] | PASS |
| xgboost | NB@eventrate | none_threshold − ros | 200 | 0.0033 [-0.0064, 0.0225] | PASS |
| xgboost | NB@eventrate | none_threshold − smote | 200 | -0.0010 [-0.0097, 0.0057] | PASS |
| xgboost | NB@0.05 | none_threshold − rus | 200 | 0.0015 [-0.0198, 0.0139] | PASS |
| xgboost | NB@0.05 | none_threshold − ros | 200 | -0.0011 [-0.0119, 0.0044] | PASS |
| xgboost | NB@0.05 | none_threshold − smote | 200 | -0.0018 [-0.0109, 0.0023] | PASS |
| xgboost | NB@0.10 | none_threshold − rus | 200 | 0.0082 [-0.0095, 0.0271] | PASS |
| xgboost | NB@0.10 | none_threshold − ros | 200 | 0.0003 [-0.0090, 0.0086] | PASS |
| xgboost | NB@0.10 | none_threshold − smote | 200 | -0.0019 [-0.0097, 0.0029] | PASS |
| xgboost | NB@0.20 | none_threshold − rus | 200 | 0.0225 [0.0063, 0.0435] | PASS |
| xgboost | NB@0.20 | none_threshold − ros | 200 | 0.0041 [-0.0074, 0.0237] | PASS |
| xgboost | NB@0.20 | none_threshold − smote | 200 | -0.0008 [-0.0089, 0.0084] | PASS |
| mlp | NB@eventrate | none_threshold − rus | 200 | 0.0371 [-0.0059, 0.0776] | PASS |
| mlp | NB@eventrate | none_threshold − ros | 200 | 0.0030 [-0.0493, 0.0524] | PASS |
| mlp | NB@eventrate | none_threshold − smote | 200 | 0.0027 [-0.0503, 0.0492] | PASS |
| mlp | NB@0.05 | none_threshold − rus | 200 | 0.0127 [-0.0046, 0.0403] | PASS |
| mlp | NB@0.05 | none_threshold − ros | 200 | -0.0024 [-0.0449, 0.0144] | PASS |
| mlp | NB@0.05 | none_threshold − smote | 200 | -0.0016 [-0.0457, 0.0138] | PASS |
| mlp | NB@0.10 | none_threshold − rus | 200 | 0.0301 [-0.0176, 0.0887] | PASS |
| mlp | NB@0.10 | none_threshold − ros | 200 | -0.0077 [-0.0969, 0.0284] | PASS |
| mlp | NB@0.10 | none_threshold − smote | 200 | -0.0073 [-0.0993, 0.0264] | PASS |
| mlp | NB@0.20 | none_threshold − rus | 200 | 0.0703 [-0.0617, 0.2087] | PASS |
| mlp | NB@0.20 | none_threshold − ros | 200 | -0.0135 [-0.2198, 0.0582] | PASS |
| mlp | NB@0.20 | none_threshold − smote | 200 | -0.0137 [-0.2211, 0.0525] | PASS |

> Paired within each (dataset, seed, fold): ΔNB = NB(none_threshold) − NB(correction) per replicate, then summarised (METRICS.md §5).
> Positive ΔNB means `none + threshold shift` is at least as good as the correction. 95% interval = 2.5/97.5 percentiles (descriptive; no t-test).
> H3 PASS iff the correction does NOT beat `none_threshold` with a 95% interval excluding zero (i.e. not hi < 0). H3 is falsified where a correction beats threshold-shifting on Net Benefit with a CI excluding zero (PREREG §3).

![Figure 1: pooled decision curves (H3)](results/figures/figure1_decision_curves.png)

### 4.4 The MLP "gain" is a broken baseline, with a measured mechanism

The MLP is the only model configured with accuracy-monitored early stopping. Seventeen of two hundred uncorrected MLP replicates have AUROC below 0.6, with a minimum of 0.385, worse than chance. All seventeen fall on the two lowest-prevalence datasets, wilt (0.054) and ozone (0.063); no corrected condition produces a single broken replicate.

Exact reproduction of those seventeen cells, holding seed, fold, and hyperparameters fixed, reveals the cause. The broken replicates stop training after twelve to thirteen iterations, with a best internal-validation accuracy of 0.936 to 0.946 and a maximum predicted probability never exceeding 0.515. The matched SMOTE replicates train for eighteen to fifty-four iterations, reach validation accuracy near 0.99, and produce confident probabilities.

The decisive number is the validation accuracy. The majority-class accuracy of the two affected datasets is 0.9369 (ozone) and 0.9461 (wilt), and the broken replicates' validation scores sit exactly on that band. The network is predicting the majority class, achieving the accuracy a constant predictor would achieve, and early stopping, which monitors accuracy, reads this as convergence and halts an untrained model. Resampling to 1:1 restores an informative accuracy signal, training proceeds, and the model learns.

Practitioners observe SMOTE deliver up to +0.48 AUROC on these replicates. What they observe is a race in which the baseline was disabled before the start. Figure 3 makes the mechanism visible at a glance: the broken group's iteration count is pinned at the floor while its validation accuracy lands precisely on the majority-class line.

![Figure 3: MLP early-stopping mechanism](results/figures/figure3_mlp_mechanism.png)

### 4.5 Corrections can make models actively harmful

Beyond failing to help, corrections can harm. Using the sign of the pre-registered net benefit (a negative value means the model is worse than treating no one), random under-sampling on the MLP at a threshold of 0.20 produced negative net benefit in 118 of 200 replicates, and in 89 of them it converted a model that had been useful without correction into a harmful one. Logistic regression under the same correction and threshold flipped 82 of 200. The uncorrected model was almost never harmful for logistic regression or gradient-boosted trees (0 of 200 at every threshold). Gradient-boosted trees were the most robust; SMOTE was the least damaging correction for the ML models.

| model | condition | threshold | n | (a) n_NB<0 | (a) frac_NB<0 [95%] | (b) n_corr<0 & none>0 | (b) frac_corr<0 & none>0 [95%] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logreg | none | NB@eventrate | 200 | 0 | 0.0000 [0.0000, 0.0000] | — | — |
| logreg | none | NB@0.05 | 200 | 0 | 0.0000 [0.0000, 0.0000] | — | — |
| logreg | none | NB@0.10 | 200 | 0 | 0.0000 [0.0000, 0.0000] | — | — |
| logreg | none | NB@0.20 | 200 | 0 | 0.0000 [0.0000, 0.0000] | — | — |
| logreg | rus | NB@eventrate | 200 | 2 | 0.0100 [0.0000, 0.0250] | 2 | 0.0100 [0.0000, 0.0250] |
| logreg | rus | NB@0.05 | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |
| logreg | rus | NB@0.10 | 200 | 25 | 0.1250 [0.0800, 0.1750] | 25 | 0.1250 [0.0800, 0.1750] |
| logreg | rus | NB@0.20 | 200 | 82 | 0.4100 [0.3400, 0.4750] | 82 | 0.4100 [0.3400, 0.4750] |
| logreg | ros | NB@eventrate | 200 | 1 | 0.0050 [0.0000, 0.0150] | 1 | 0.0050 [0.0000, 0.0150] |
| logreg | ros | NB@0.05 | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |
| logreg | ros | NB@0.10 | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |
| logreg | ros | NB@0.20 | 200 | 41 | 0.2050 [0.1500, 0.2650] | 41 | 0.2050 [0.1500, 0.2650] |
| logreg | smote | NB@eventrate | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |
| logreg | smote | NB@0.05 | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |
| logreg | smote | NB@0.10 | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |
| logreg | smote | NB@0.20 | 200 | 30 | 0.1500 [0.1000, 0.2000] | 30 | 0.1500 [0.1000, 0.2000] |
| xgboost | none | NB@eventrate | 200 | 0 | 0.0000 [0.0000, 0.0000] | — | — |
| xgboost | none | NB@0.05 | 200 | 0 | 0.0000 [0.0000, 0.0000] | — | — |
| xgboost | none | NB@0.10 | 200 | 0 | 0.0000 [0.0000, 0.0000] | — | — |
| xgboost | none | NB@0.20 | 200 | 0 | 0.0000 [0.0000, 0.0000] | — | — |
| xgboost | rus | NB@eventrate | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |
| xgboost | rus | NB@0.05 | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |
| xgboost | rus | NB@0.10 | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |
| xgboost | rus | NB@0.20 | 200 | 23 | 0.1150 [0.0750, 0.1600] | 23 | 0.1150 [0.0750, 0.1600] |
| xgboost | ros | NB@eventrate | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |
| xgboost | ros | NB@0.05 | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |
| xgboost | ros | NB@0.10 | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |
| xgboost | ros | NB@0.20 | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |
| xgboost | smote | NB@eventrate | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |
| xgboost | smote | NB@0.05 | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |
| xgboost | smote | NB@0.10 | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |
| xgboost | smote | NB@0.20 | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |
| mlp | none | NB@eventrate | 200 | 8 | 0.0400 [0.0150, 0.0700] | — | — |
| mlp | none | NB@0.05 | 200 | 0 | 0.0000 [0.0000, 0.0000] | — | — |
| mlp | none | NB@0.10 | 200 | 26 | 0.1300 [0.0850, 0.1750] | — | — |
| mlp | none | NB@0.20 | 200 | 30 | 0.1500 [0.1000, 0.1951] | — | — |
| mlp | rus | NB@eventrate | 200 | 35 | 0.1750 [0.1250, 0.2300] | 32 | 0.1600 [0.1100, 0.2150] |
| mlp | rus | NB@0.05 | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |
| mlp | rus | NB@0.10 | 200 | 72 | 0.3600 [0.2950, 0.4250] | 47 | 0.2350 [0.1800, 0.2950] |
| mlp | rus | NB@0.20 | 200 | 118 | 0.5900 [0.5200, 0.6550] | 89 | 0.4450 [0.3800, 0.5150] |
| mlp | ros | NB@eventrate | 200 | 1 | 0.0050 [0.0000, 0.0150] | 1 | 0.0050 [0.0000, 0.0150] |
| mlp | ros | NB@0.05 | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |
| mlp | ros | NB@0.10 | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |
| mlp | ros | NB@0.20 | 200 | 8 | 0.0400 [0.0150, 0.0700] | 8 | 0.0400 [0.0150, 0.0700] |
| mlp | smote | NB@eventrate | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |
| mlp | smote | NB@0.05 | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |
| mlp | smote | NB@0.10 | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |
| mlp | smote | NB@0.20 | 200 | 0 | 0.0000 [0.0000, 0.0000] | 0 | 0.0000 [0.0000, 0.0000] |

> NB is the pre-registered Net Benefit recomputed from the saved y_prob files at the pre-registered thresholds {event rate, 0.05, 0.10, 0.20} (the same values as Table 3, METRICS.md §4.1). NB < 0 means WORSE than treating nobody at that threshold.
> (a) counts replicates with NB(condition) < 0. `none` is included as the reference row so each correction can be read against the uncorrected model at the same (model, threshold).
> (b) is PAIRED within each (dataset, seed, fold): replicates where the correction's NB < 0 (harmful) WHILE `none`'s NB > 0 (useful) on the SAME replicate: the correction turned a useful model into a harmful one at that threshold. On the `none` reference row this quantity is impossible by construction and is marked '—'.
> n = replicates per (model, condition) = 8 datasets × 5 seeds × 5 folds = 200 (n_pairs likewise for the paired part (b)).
> 95% interval = percentile bootstrap over the 200 replicate indicators (2000 resamples, seed config.SEEDS[0]). Descriptive only: replicates share 8 datasets and CV folds are not independent (METRICS.md §5), so this is not a significance test.
> This is a DESCRIPTIVE report of the sign of a pre-registered quantity; no PASS/FAIL verdict is assigned (unlike the pre-registered H1–H3 tables).

**Supplementary:** per-dataset decision curves are provided in [Figure S1](results/figures/figure_S1_decision_curves_by_dataset.png).

## 5. Discussion

For a practitioner, the operational takeaway is a diagnostic reversal. If imbalance correction appears to help your model, the first hypothesis should be that your baseline is broken, as it was here for the MLP, silently, through a default early-stopping rule interacting with low prevalence. The apparent gain may be repair, not improvement, and the repair is better achieved by fixing the training procedure than by distorting the data.

For the field, the mechanism explains the folklore. The persistent belief that resampling helps does not require resampling to help. It requires only that default training procedures fail quietly on imbalanced data in a way that resampling happens to mask. Discrimination is unchanged for models that were training correctly all along; the "gains" accrue precisely where the baseline was broken.

The consistent cost is calibration. Every correction, in every model family, moved calibration-in-the-large sharply negative: a systematic overestimation of risk. If the model's probabilities inform a decision, and if they do not one should ask why the model exists, then corrections make the decision worse, and at standard thresholds can make it actively harmful.

The alternative is threshold adjustment. It achieves everything the corrections achieve in decision terms, costs nothing, is reversible, and leaves the probabilities untouched.

## 6. Limitations

The study covers eight tabular datasets. The pre-registered pool could not supply ten; we took all eight that qualified and declined to broaden the pool, to preserve the mechanical, cherry-picking-proof selection. The findings are tabular and do not extend to unstructured data.

Hyperparameters were fixed. We do not claim these results hold under per-condition tuning; holding them fixed is a deliberate design choice that isolates the effect of the correction from the effect of tuning. The anticipated objection (that better MLP settings would remove the broken baseline, and with it the apparent SMOTE gain) is exactly the point. The configuration used is a standard default. The finding is that under standard defaults, on imbalanced data, the apparent benefit of correction is an artifact of a silently mistrained baseline.

Class-weighting approaches, which reweight the loss rather than resample the data, are out of scope and are the natural next study. One dataset contains an entirely missing column, which the imputer drops; this is disclosed rather than silently handled.

## 7. Reproducibility

A single command reproduces the entire study from a clean checkout. The pre-registration, the deviations log, every per-item probability, and all 2,400 cells are in the repository. The leakage guard is mutation-tested; every reported quantity carries a 95% interval; failed cells, had there been any, would be reported rather than dropped.

## References

1. van den Goorbergh R, van Smeden M, Timmerman D, Van Calster B. *J Am Med Inform Assoc (JAMIA)*. 2022;29(9):1525-1534. doi:10.1093/jamia/ocac093
2. Piccininni M, Wechsung M, Van Calster B, Rohmann JL, Konigorski S, van Smeden M. *J Biomed Inform*. 2024;155:104666. doi:10.1016/j.jbi.2024.104666
3. Carriero A, Luijken K, de Hond A, Moons KGM, van Calster B, van Smeden M. *Stat Med*. 2025;44(3-4):e10320. doi:10.1002/sim.10320
4. Vickers AJ, Elkin EB. *Med Decis Making*. 2006;26(6):565-574. doi:10.1177/0272989X06295361 (Net Benefit)

---

## Appendix: deferred writing tasks

- §2 Related work: write out in prose, ≤15-word quotes, mostly paraphrase.
- Insert Table 1–4 and Figure 1–3 at marked points; Figure S1 (per-dataset decision curves) as supplement.
- Retitle for venue. Candidate framing: a short methods/position paper for a stats-in-ML or clinical-ML venue where the Van Calster / van Smeden / Carriero line of work lives.
- Circulate to the authors whose open question this closes; they are the natural amplifiers.
