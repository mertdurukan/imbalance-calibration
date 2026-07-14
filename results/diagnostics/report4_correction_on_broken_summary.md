# Report 4 (summary) — paired ΔAUROC on broken mlp/none replicates (none AUROC < 0.6)

| contrast | n_pairs | delta_auroc_mean_[95%] |
| --- | --- | --- |
| rus - none  (broken subset only) | 17 | 0.3658 [0.1986, 0.5168] |
| ros - none  (broken subset only) | 17 | 0.4833 [0.3582, 0.5939] |
| smote - none  (broken subset only) | 17 | 0.4835 [0.3541, 0.5930] |

> Paired ΔAUROC = AUROC(correction) − AUROC(none) within each broken (dataset, seed, fold) replicate, then summarised. Descriptive only; this is a post-hoc subset, not a pre-registered estimand.
