# Report 5 — EXPLORATORY mlp H1 recompute (post-hoc exclusion of broken mlp/none replicates)

| model | contrast | n_pairs_kept | delta_auroc_mean_[95%] | H1_recomputed (|ΔAUROC|<0.01) |
| --- | --- | --- | --- | --- |
| mlp | rus - none | 183 | -0.0142 [-0.1160, 0.1624] | FAIL |
| mlp | ros - none | 183 | 0.0148 [-0.0392, 0.2863] | FAIL |
| mlp | smote - none | 183 | 0.0087 [-0.0468, 0.2881] | PASS |

> EXPLORATORY — post-hoc exclusion, not pre-registered, reported for diagnosis only. The pre-registered H1 result stands as reported in Table 1 (results/tables/table1_h1_discrimination.csv).
> Exclusion rule: drop the (dataset, seed, fold) replicates whose mlp/none AUROC < 0.6. Excluded 17 of 200 mlp/none replicates.
> Paired ΔAUROC = AUROC(correction) − AUROC(none) within each kept replicate, then summarised (2.5/97.5 percentile interval).
