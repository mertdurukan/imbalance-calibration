# Report 7 — EXACT REFIT of the 17 matched mlp/SMOTE replicates (same dataset/seed/fold as Report 6)

| dataset_id | dataset_name | event_rate | seed | fold | n_iter_ | best_val_score_ | prob_min | prob_max | prob_mean | prob_std | auroc_repro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 40983 | wilt | 0.0539 | 0 | 0 | 25 | 0.9877 | 0.0000 | 0.9999 | 0.0838 | 0.242348 | 0.9963 |
| 40983 | wilt | 0.0539 | 0 | 1 | 22 | 0.9891 | 0.0000 | 0.9997 | 0.0731 | 0.221327 | 0.9978 |
| 40983 | wilt | 0.0539 | 0 | 2 | 19 | 0.9945 | 0.0000 | 0.9987 | 0.1044 | 0.235230 | 0.9914 |
| 40983 | wilt | 0.0539 | 0 | 4 | 18 | 0.9891 | 0.0000 | 0.9983 | 0.1163 | 0.235347 | 0.9966 |
| 40983 | wilt | 0.0539 | 2 | 0 | 19 | 0.9877 | 0.0000 | 0.9991 | 0.1176 | 0.241466 | 0.9936 |
| 40983 | wilt | 0.0539 | 2 | 1 | 35 | 0.9877 | 0.0000 | 0.9999 | 0.0689 | 0.231737 | 0.9979 |
| 40983 | wilt | 0.0539 | 2 | 2 | 34 | 0.9905 | 0.0000 | 0.9999 | 0.0748 | 0.240746 | 0.9962 |
| 40983 | wilt | 0.0539 | 2 | 3 | 31 | 0.9945 | 0.0000 | 0.9999 | 0.0658 | 0.228510 | 0.9866 |
| 40983 | wilt | 0.0539 | 2 | 4 | 24 | 0.9877 | 0.0000 | 0.9998 | 0.0765 | 0.237757 | 0.9983 |
| 40983 | wilt | 0.0539 | 4 | 0 | 24 | 0.9959 | 0.0000 | 0.9999 | 0.0747 | 0.235236 | 0.9873 |
| 40983 | wilt | 0.0539 | 4 | 1 | 28 | 0.9918 | 0.0000 | 0.9998 | 0.0795 | 0.239135 | 0.9978 |
| 40983 | wilt | 0.0539 | 4 | 2 | 20 | 0.9877 | 0.0000 | 0.9995 | 0.0822 | 0.221179 | 0.9975 |
| 40983 | wilt | 0.0539 | 4 | 3 | 44 | 0.9959 | 0.0000 | 1.0000 | 0.0756 | 0.244373 | 0.9961 |
| 1487 | ozone-level-8hr | 0.0631 | 0 | 0 | 43 | 0.9737 | 0.0000 | 0.9995 | 0.0730 | 0.219881 | 0.9350 |
| 1487 | ozone-level-8hr | 0.0631 | 0 | 2 | 54 | 0.9868 | 0.0000 | 0.9995 | 0.0686 | 0.221222 | 0.8793 |
| 1487 | ozone-level-8hr | 0.0631 | 0 | 4 | 46 | 0.9789 | 0.0000 | 0.9999 | 0.0819 | 0.237774 | 0.9206 |
| 1487 | ozone-level-8hr | 0.0631 | 2 | 3 | 49 | 0.9789 | 0.0000 | 0.9988 | 0.0675 | 0.206308 | 0.8646 |

> Same (dataset, seed, fold) keys as the broken mlp/none set, condition SMOTE. Exact reproduction of the frozen mlp/smote fits; same frozen hyperparameters. n_iter_ is the contrast quantity vs Report 6.
> best_val_score_ here is MLP accuracy on its internal validation split of the SMOTE-BALANCED (1:1) training fold, so it is not comparable in level to Report 6's (which validates on an imbalanced split); reported for completeness.
