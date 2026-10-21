# Experiment Protocol (Draft)

- Main design: 3 models × 3 imbalance strategies.
- Main split: chronological 60% train / 20% validation / 20% test.
- Primary metric: average precision.
- Secondary metrics: precision, recall, F1, ROC-AUC, confusion matrix.
- Fit preprocessing and apply SMOTE only within training partitions/folds.
- Select thresholds using validation data only; do not tune on test labels.
- Document duplicate-handling decisions and timestamp tie handling.
- The dataset spans roughly two days; do not claim long-term drift evidence.
- Review-cost scenarios are hypothetical proxies, not actual bank losses.
