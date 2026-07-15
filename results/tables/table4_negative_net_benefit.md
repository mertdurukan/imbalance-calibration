# DESCRIPTIVE — sign of the pre-registered Net Benefit. Not a new hypothesis test; no PASS/FAIL verdict is assigned.

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
> (b) is PAIRED within each (dataset, seed, fold): replicates where the correction's NB < 0 (harmful) WHILE `none`'s NB > 0 (useful) on the SAME replicate — the correction turned a useful model into a harmful one at that threshold. On the `none` reference row this quantity is impossible by construction and is marked '—'.
> n = replicates per (model, condition) = 8 datasets × 5 seeds × 5 folds = 200 (n_pairs likewise for the paired part (b)).
> 95% interval = percentile bootstrap over the 200 replicate indicators (2000 resamples, seed config.SEEDS[0]). Descriptive only: replicates share 8 datasets and CV folds are not independent (METRICS.md §5), so this is not a significance test.
> This is a DESCRIPTIVE report of the sign of a pre-registered quantity; no PASS/FAIL verdict is assigned (unlike the pre-registered H1–H3 tables).
