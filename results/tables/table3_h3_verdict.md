# Table 3 (H3 verdict): paired (none_threshold − correction) Net Benefit

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
