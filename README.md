# Credit Card Fraud Detection

MSc research project: Credit Card Fraud Detection under Severe Class Imbalance.

## Dataset
Place the original dataset at `data/raw/creditcard.csv`. Do not commit raw data to a public repository.

## Workflow
1. EDA and dataset audit
2. Duplicate investigation and split validation
3. Leakage-safe preprocessing
4. Logistic Regression, Decision Tree, and Random Forest
5. Compare original data, class weighting, and SMOTE
6. Evaluate using average precision, precision, recall, F1, ROC-AUC, confusion matrices, and validation-selected thresholds

Never use test labels to choose models, hyperparameters, or thresholds. Apply SMOTE only to training data / training folds.
