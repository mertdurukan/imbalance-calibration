# METRICS — exact definitions

Cursor: implement these literally. Do NOT substitute library defaults. Do NOT simplify.
Every metric below has a known-answer test in `tests/test_metrics.py` — write the test
BEFORE the implementation.

Notation: `y ∈ {0,1}^n` true labels, `p ∈ (0,1)^n` predicted probability of class 1.

---

## 1. Discrimination

- **AUROC** = `sklearn.metrics.roc_auc_score(y, p)`
- **AUPRC** = `sklearn.metrics.average_precision_score(y, p)` (average precision, NOT
  trapezoidal PR area — they differ; average precision is the specified estimand.)
- **Brier** = `mean((p - y)**2)`

---

## 2. Calibration slope & intercept ⚠️ most-often-implemented-wrong

These are **NOT** `sklearn.calibration` outputs. They come from clinical prediction
literature (Van Calster et al.) and are defined by two logistic regressions on the
**linear predictor** `L = logit(p) = log(p / (1-p))`.

**Clipping:** clip `p` to `[1e-6, 1 - 1e-6]` before taking the logit, to avoid ±inf.

**Calibration slope** — fit `y ~ a + b·L` by logistic regression (no regularization,
`penalty=None`). The slope is `b`.
- Perfect = **1.0**
- `b < 1` → predictions are **over-extreme / overfit** (too confident at both ends)
- `b > 1` → predictions are under-extreme
- **This is the metric that should degrade under SMOTE.** SMOTE inflates predicted
  probabilities for the minority class → systematic overestimation.

**Calibration intercept** ("calibration-in-the-large") — fit `y ~ a + offset(L)`, i.e. a
logistic regression with the slope FIXED at 1 and only an intercept free. The intercept
is `a`.
- Perfect = **0.0**
- `a > 0` → systematic **underestimation** of risk
- `a < 0` → systematic **overestimation** of risk (expected under oversampling)

Implementation note: `statsmodels.GLM` with `family=Binomial()` and the `offset=` argument
does this directly. Using sklearn requires manual handling — prefer statsmodels here.

**Known-answer test:** if `p` is generated from a true logistic model and `y` sampled from
it, slope → 1.0 and intercept → 0.0 as n → ∞. Test with n=100_000, tolerance 0.05.

---

## 3. ECE (Expected Calibration Error) — equal-MASS bins

⚠️ Most implementations use equal-WIDTH bins. We use **equal-mass (quantile) bins**,
because with imbalanced data most predictions cluster near 0 and equal-width bins leave
most bins empty, which understates calibration error.

```
1. Sort items by p.
2. Split into ECE_N_BINS (=15) bins of EQUAL COUNT (np.array_split on sorted order).
3. For each bin b: conf_b = mean(p in b), acc_b = mean(y in b), n_b = |b|
4. ECE = sum_b (n_b / n) * |acc_b - conf_b|
```

**Known-answer test:** a perfectly calibrated synthetic set → ECE ≈ 0 (tolerance 0.01).
A model that always predicts 0.5 on a 50/50 balanced set → ECE ≈ 0. A model that always
predicts 0.9 on a 50/50 set → ECE ≈ 0.4.

---

## 4. Net Benefit (decision curve analysis) ⚠️ this is the headline metric

From Vickers & Elkin. For a decision threshold `pt`:

```
predicted_positive = (p >= pt)
TP = sum(predicted_positive & (y == 1))
FP = sum(predicted_positive & (y == 0))
n  = len(y)

NB(pt) = TP/n - (FP/n) * (pt / (1 - pt))
```

Interpretation: net true positives per patient/case, penalizing false positives by the
odds of the threshold. Higher is better.

**Reference lines that MUST appear on every decision curve:**
- `treat-all`: NB = event_rate - (1 - event_rate) * (pt / (1 - pt))
- `treat-none`: NB = 0

A model is only useful where its curve sits **above both** reference lines.

### 4.1 The `none_threshold` condition
This is NOT a separate model fit. It reuses the `condition="none"` predictions and simply
evaluates NB across the threshold sweep. **This is the entire point of hypothesis H3:**
if `none` at a shifted threshold matches `smote` at its default threshold, then SMOTE
bought nothing that a free threshold change could not.

Because of this, **the decision curve is computed from the saved `y_prob` files at
analysis time**, not during fitting. This is why `y_prob_path` in the output schema is
mandatory.

---

## 5. Uncertainty

Every reported number = point estimate + 95% CI.

- **Within-cell** (one dataset/model/condition/seed/fold): percentile bootstrap over
  ITEMS, `BOOTSTRAP_N=2000` resamples, seed fixed.
- **Across replicates** (pooling 25 seed×fold cells): report mean and the 2.5/97.5
  percentiles of the replicate distribution. Do NOT compute a t-test across folds —
  CV folds are not independent; this is a known statistical error. Report the interval
  descriptively and say so in the paper.
- **Primary contrast**: paired difference `metric(condition) - metric(none)`, computed
  within each (dataset, model, seed, fold), then summarized. Pairing matters — do not
  compare unpaired means.
