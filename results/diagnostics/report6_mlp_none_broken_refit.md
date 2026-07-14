# Report 6 — EXACT REFIT of the 17 broken mlp/none replicates (frozen AUROC < 0.6)

| dataset_id | dataset_name | event_rate | seed | fold | n_iter_ | best_val_score_ | prob_min | prob_max | prob_mean | prob_std | auroc_frozen | auroc_repro | exact_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 40983 | wilt | 0.0539 | 0 | 0 | 12 | 0.9459 | 0.0072 | 0.5086 | 0.3552 | 0.072836 | 0.5136 | 0.5136 | yes |
| 40983 | wilt | 0.0539 | 0 | 1 | 12 | 0.9459 | 0.0025 | 0.5033 | 0.3486 | 0.080754 | 0.4802 | 0.4802 | yes |
| 40983 | wilt | 0.0539 | 0 | 2 | 13 | 0.9459 | 0.0000 | 0.3356 | 0.2019 | 0.070874 | 0.5304 | 0.5304 | yes |
| 40983 | wilt | 0.0539 | 0 | 4 | 13 | 0.9459 | 0.0001 | 0.3405 | 0.2050 | 0.074607 | 0.5850 | 0.5850 | yes |
| 40983 | wilt | 0.0539 | 2 | 0 | 12 | 0.9459 | 0.0429 | 0.4240 | 0.3050 | 0.057311 | 0.4302 | 0.4302 | yes |
| 40983 | wilt | 0.0539 | 2 | 1 | 12 | 0.9459 | 0.0849 | 0.4198 | 0.3029 | 0.058476 | 0.3851 | 0.3851 | yes |
| 40983 | wilt | 0.0539 | 2 | 2 | 12 | 0.9459 | 0.1011 | 0.4172 | 0.3028 | 0.055289 | 0.4518 | 0.4518 | yes |
| 40983 | wilt | 0.0539 | 2 | 3 | 12 | 0.9459 | 0.0422 | 0.4133 | 0.3037 | 0.057450 | 0.4492 | 0.4492 | yes |
| 40983 | wilt | 0.0539 | 2 | 4 | 12 | 0.9459 | 0.0950 | 0.4194 | 0.3019 | 0.057456 | 0.4369 | 0.4369 | yes |
| 40983 | wilt | 0.0539 | 4 | 0 | 12 | 0.9459 | 0.0447 | 0.4468 | 0.3194 | 0.047604 | 0.5254 | 0.5254 | yes |
| 40983 | wilt | 0.0539 | 4 | 1 | 12 | 0.9459 | 0.0297 | 0.4821 | 0.3141 | 0.053562 | 0.5668 | 0.5668 | yes |
| 40983 | wilt | 0.0539 | 4 | 2 | 12 | 0.9459 | 0.0728 | 0.4426 | 0.3151 | 0.047863 | 0.4884 | 0.4884 | yes |
| 40983 | wilt | 0.0539 | 4 | 3 | 12 | 0.9459 | 0.0412 | 0.4429 | 0.3159 | 0.049156 | 0.5063 | 0.5063 | yes |
| 1487 | ozone-level-8hr | 0.0631 | 0 | 0 | 13 | 0.9360 | 0.0048 | 0.5147 | 0.2045 | 0.122723 | 0.4674 | 0.4674 | yes |
| 1487 | ozone-level-8hr | 0.0631 | 0 | 2 | 13 | 0.9360 | 0.0014 | 0.5135 | 0.1944 | 0.115065 | 0.4366 | 0.4366 | yes |
| 1487 | ozone-level-8hr | 0.0631 | 0 | 4 | 13 | 0.9360 | 0.0022 | 0.5146 | 0.2015 | 0.116369 | 0.5175 | 0.5175 | yes |
| 1487 | ozone-level-8hr | 0.0631 | 2 | 3 | 13 | 0.9409 | 0.0021 | 0.5145 | 0.1726 | 0.126316 | 0.5432 | 0.5432 | yes |

> Exact reproduction: same dataset, same StratifiedKFold(shuffle=True, random_state=seed) split & fold, same frozen pipeline & hyperparameters (src.config). No hyperparameter changed; fits are in-memory and discarded.
> n_iter_ = MLP iterations run before early stopping fired. best_val_score_ = MLPClassifier.best_validation_score_ (accuracy on its own internal early-stopping validation split, monitored because early_stopping=True per PREREG §4.2).
> prob_* = distribution of held-out predicted P(class=1). auroc_repro vs auroc_frozen: exact_match=yes confirms the reproduction reproduces the frozen cell.
