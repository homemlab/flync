# Dataset versioning
During hyperparameter optimizations I will use the following logic to tag the datasets:
- `--dataset-suffix` flag will help understand which dataset type is used: E.g. `redux-no_bb_names-no_cpat`; `redux-remap_names`; etc...
- `--dataset-version` flag will be used to fine-tune feature selection:

| version | meaning |
| --- | --- |
| 1.0 | All features used (refer to `--dataset-sufix` for clarity on which) |
| 2.0 | Automatic removal of correlated features using `--analyze-correlations` and `--corelation-threshold`|
| 3.0 | Manually curated features using upstream processes and/or `--drop-features-file` |