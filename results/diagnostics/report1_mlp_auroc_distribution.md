# Report 1 — mlp AUROC distribution by condition

| condition | n | auroc_mean_[95%] | auroc_min | auroc_max | n_auroc_lt_0.6 |
| --- | --- | --- | --- | --- | --- |
| none | 200 | 0.8393 [0.4518, 0.9897] | 0.3851 | 0.9978 | 17 |
| rus | 200 | 0.8574 [0.7035, 0.9810] | 0.6572 | 0.9914 | 0 |
| ros | 200 | 0.8939 [0.6988, 0.9979] | 0.6848 | 0.9987 | 0 |
| smote | 200 | 0.8884 [0.6884, 0.9978] | 0.6682 | 0.9988 | 0 |

> n_auroc_lt_0.6: count of replicates with AUROC < 0.6 (post-hoc near-chance flag; not pre-registered).
> 95% interval = 2.5/97.5 percentiles of the replicate distribution.
