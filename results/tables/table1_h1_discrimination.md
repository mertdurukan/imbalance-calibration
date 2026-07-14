# Table 1 — H1 (discrimination): paired ΔAUROC / ΔAUPRC vs `none`

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
