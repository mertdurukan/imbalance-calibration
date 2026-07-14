# Table 2 — H2 (calibration): slope, intercept, ECE, Brier per model × condition

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

> Absolute per-condition metrics. Mean + 95% interval (2.5/97.5 percentiles of the replicate distribution; descriptive, no t-test — folds not independent).
> Perfect calibration: slope = 1.0, intercept = 0.0. H2 predicts corrections push slope AWAY from 1.0, intercept AWAY from 0.0, and RAISE ECE.
> PASS/FAIL (corrections only) is vs the `none` reference (row 'ref') in each model: slope PASS iff |mean−1| > |none−1|; intercept PASS iff |mean| > |none|; ECE PASS iff mean > none. Brier has no pre-registered direction under H2 and carries no verdict.
