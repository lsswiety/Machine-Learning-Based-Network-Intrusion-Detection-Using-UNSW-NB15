# UNSW-NB15 Intrusion Detection using ML and Feature Extraction

A clean, modular preprocessing and machine learning pipeline for the **UNSW-NB15 network intrusion detection dataset**, covering preprocessing, feature selection, model training, hyperparameter tuning, and performance analysis compared with the NSL-KDD dataset.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Pipeline Steps](#pipeline-steps)
- [Feature Selection](#feature-selection)
- [Models and Evaluation](#models-and-evaluation)
- [Grid Search and Hyperparameter Tuning](#grid-search-and-hyperparameter-tuning)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Performance Analysis](#performance-analysis)
- [Output Artifacts](#output-artifacts)
- [Configuration](#configuration)
- [Dependencies](#dependencies)

---

## Project Overview

This project prepares the UNSW-NB15 dataset for multiclass intrusion detection and evaluates different feature selection methods and machine learning models. The pipeline generates a final feature matrix containing exactly **194 features**, then evaluates five feature subsets across multiple classifiers using both baseline and tuned configurations.

A comparison with the **NSL-KDD dataset** is also included to study how dataset complexity and imbalance affect intrusion detection performance.

---

## Dataset

**UNSW-NB15** is a network traffic dataset created by the Australian Centre for Cyber Security (ACCS). It contains both normal and attack traffic across 10 categories with 231,286 total samples.

| File | Description |
|------|-------------|
| `UNSW_NB15_training-set.csv` | Raw training data |
| `UNSW_NB15_testing-set.csv` | Raw testing data |

> Download from the [official UNSW page](https://research.unsw.edu.au/projects/unsw-nb15-dataset).

### Attack Categories

`Normal`, `Fuzzers`, `Analysis`, `Backdoors`, `DoS`, `Exploits`, `Generic`, `Reconnaissance`, `Shellcode`, `Worms`

> **Class imbalance challenge:** The dataset is imbalanced, where Normal traffic makes up about 61% of the test data while Worms contains only 44 samples. The goal was to handle this imbalance and evaluate its effect on detecting minority attack classes. Because of this, accuracy alone can be misleading, as a model may achieve high accuracy while failing to detect rare but important attacks.

---

## Pipeline Steps

### 1. Load Data

- Reads training and testing CSV files into pandas DataFrames
- Keeps only columns shared between train and test sets

### 2. Drop ID and Address Columns

Removed columns:

```text
id, srcip, sport, dstip, dsport
```

Also removes the binary `label` column since the task is multiclass classification.

### 3. Remove Duplicates

- Removes exact duplicate rows independently from train and test sets

### 4. Label Engineering

- Extracts `attack_cat` as the target column
- Cleans whitespace
- Applies `LabelEncoder` fitted only on training data

### 5. Drop Correlated Features

Removed highly correlated features:

```text
sloss, dloss, dpkts, dwin, ltime, ct_srv_dst, ct_src_dport_ltm, ct_dst_src_ltm
```

### 6. Log1p Transformation

Applied `log1p` to skewed numerical columns:

```text
dur, sbytes, dbytes, sload, dload, spkts, stcpb, dtcpb,
smeansz, dmeansz, sjit, djit
```

### 7. Data Cleaning

- Fills missing values using train-set modes
- Replaces blank placeholder values
- Preserves `'-'` as a valid categorical value
- Clips binary columns to `{0,1}`
- Converts incorrect object dtypes to numeric

### 8. Scaling and One-Hot Encoding

- `MinMaxScaler` applied to numerical columns
- `OneHotEncoder` applied to categorical columns

| Category | Features |
|---|---|
| Protocol categories | 133 |
| Service categories | 13 |
| State categories | 16 |

### 9. Final Feature Matrix

| Component | Count |
|---|---|
| Numerical features | 32 |
| Protocol OHE features | 133 |
| Service OHE features | 13 |
| State OHE features | 16 |
| **Total** | **194** |

---

## Feature Selection

All feature selection methods were fitted only on the training set to avoid data leakage.

| Feature Set | Method |
|---|---|
| All Features | Baseline |
| MI k=20 | Mutual Information |
| Chi2 k=20 | Chi-Square |
| RFE k=20 | Recursive Feature Elimination |
| SFS k=20 | Sequential Feature Selection |

### Key Findings

- MI performed best among filter methods
- RFE performed best among wrapper methods
- Chi2 often selected categorical OHE features and reduced performance
- Naïve Bayes improved significantly after feature selection

---

## Models and Evaluation

Three classifiers were evaluated across all feature subsets.

### SVM

```python
make_pipeline(
    StandardScaler(),
    LinearSVC(C=10, max_iter=5000, random_state=42)
)
```

> Trained on a subsample of 5,000 rows because of computational cost.

### Random Forest

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=22,
    criterion='gini',
    min_samples_split=6,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'
)
```

### XGBoost

```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='multi:softmax'
)
```

### Baseline Results

| Model | Accuracy | Weighted F1 |
|---|---|---|
| XGBoost | 75.38% | 0.7798 |
| MLP | 72.50% | 0.7481 |
| Decision Tree | 66.79% | 0.7183 |
| Random Forest | 66.16% | 0.7145 |
| SVM | 63.60% | 0.6683 |
| Naïve Bayes | 16.12% | 0.2261 |

---

## Grid Search and Hyperparameter Tuning

GridSearchCV was applied using weighted F1 scoring and StratifiedKFold cross validation.

### Tuned Results

| Model | Before | After |
|---|---|---|
| Decision Tree | 66.79% | 74.83% |
| Random Forest | 66.16% | 71.80% |
| XGBoost | 75.38% | 75.58% |
| MLP | 72.50% | 70.97% |

> XGBoost achieved the highest cross-validation F1 score and showed minimal improvement after tuning, indicating that the original configuration was already strong.

---

## Exploratory Data Analysis

Included visualizations:

- Class distribution plots
- Correlation heatmaps
- Feature selection comparisons
- Per-class performance analysis

---

## Performance Analysis

### UNSW-NB15 vs NSL-KDD

| | NSL-KDD | UNSW-NB15 |
|---|---|---|
| Best baseline | Decision Tree | XGBoost |
| Best tuned model | SVM | XGBoost |
| Easier dataset | Yes | No |

NSL-KDD produced better Macro F1 scores because it contains fewer classes and clearer attack boundaries. UNSW-NB15 is harder because of severe imbalance and more complex attack distributions.

### Rare Class Weaknesses

| Attack Class | Test Samples | F1 Score |
|---|---|---|
| Analysis | few | 0.18 |
| Backdoor | few | 0.19 |
| Worms | 44 | 0.43 |

> High accuracy can still hide poor detection performance on rare attack categories.

### Computational Costs

| Operation | Time |
|---|---|
| SFS feature selection | ~1405s |
| RFE feature selection | ~59.5s |
| XGBoost grid search | ~1102s |

---

## Output Artifacts

### Preprocessing Outputs

| File | Description |
|---|---|
| `final_train.pkl` | Processed training data |
| `final_test.pkl` | Processed testing data |
| `train_preprocessed.csv` | CSV training output |
| `test_preprocessed.csv` | CSV testing output |

### ML Outputs

| File | Description |
|---|---|
| `feature_selection_dashboard.png` | Model and feature selection dashboard |

---

## Configuration

```python
TRAIN_FILE  = "train_preprocessed.csv"
TEST_FILE   = "test_preprocessed.csv"
TARGET_COL  = "attack_cat"
K_VALUES    = [20]
K_WRAPPER   = 20
SAMPLE_SIZE = None
```

---

## Dependencies

```text
numpy
pandas
matplotlib
scikit-learn
xgboost
```

Install all dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost
```
