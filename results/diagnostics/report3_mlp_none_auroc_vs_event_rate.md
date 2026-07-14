# Report 3 — mlp/none AUROC vs dataset event rate

| dataset_id | dataset_name | event_rate | n | mlp_none_auroc_mean_[95%] | n_auroc_lt_0.6 |
| --- | --- | --- | --- | --- | --- |
| 40983 | wilt | 0.0539 | 25 | 0.6319 [0.4122, 0.9974] | 13 |
| 38 | sick | 0.0612 | 25 | 0.9555 [0.9186, 0.9795] | 0 |
| 1487 | ozone-level-8hr | 0.0631 | 25 | 0.8147 [0.4551, 0.9368] | 4 |
| 1461 | bank-marketing | 0.1170 | 25 | 0.9220 [0.9089, 0.9296] | 0 |
| 40978 | Internet-Advertisements | 0.1400 | 25 | 0.9788 [0.9601, 0.9905] | 0 |
| 40701 | churn | 0.1414 | 25 | 0.9065 [0.8804, 0.9295] | 0 |
| 1067 | kc1 | 0.1546 | 25 | 0.7908 [0.7238, 0.8460] | 0 |
| 1053 | jm1 | 0.1935 | 25 | 0.7145 [0.6908, 0.7411] | 0 |

> One row per dataset, sorted by event_rate ascending (most imbalanced first). 95% interval = 2.5/97.5 percentiles over the 25 seed×fold replicates.
