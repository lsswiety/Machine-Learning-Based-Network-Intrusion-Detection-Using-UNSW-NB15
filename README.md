# UNSW-NB15 Intrusion Detection — ML & Feature Extraction

A clean, modular preprocessing and machine learning pipeline for the **UNSW-NB15 network intrusion detection dataset**, covering end-to-end preprocessing, feature selection (filter and wrapper methods), model training, hyperparameter tuning, and a full performance analysis comparing UNSW-NB15 against the NSL-KDD dataset.

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

This project prepares the UNSW-NB15 dataset for multiclass classification and runs a full comparative experiment across feature selection methods and classifiers. The pipeline produces a final feature matrix of exactly **194 features**, then evaluates five feature subsets across three models (SVM, Random Forest, XGBoost) with both baseline and grid-search-tuned configurations.

A companion analysis also benchmarks results against the **NSL-KDD dataset** to assess how dataset complexity affects model performance.

---

## Dataset

**UNSW-NB15** is a network traffic dataset created by the Australian Centre for Cyber Security (ACCS). It contains both normal and attack traffic across 10 categories, with 231,286 total samples (train + test).

| File | Description |
|------|-------------|
| `UNSW_NB15_training-set.csv` | Raw training data |
| `UNSW_NB15_testing-set.csv` | Raw testing data |

> Download from the [official UNSW page](https://research.unsw.edu.au/projects/unsw-nb15-dataset).

### Attack Categories (10-class Target)

`Normal`, `Fuzzers`, `Analysis`, `Backdoors`, `DoS`, `Exploits`, `Generic`, `Reconnaissance`, `Shellcode`, `Worms`

> **Class imbalance note:** Normal accounts for ~61% of test samples. Worms has only 44 test samples. This makes accuracy misleading — a model can reach 75% accuracy while completely failing on rare but critical attack types.

---

## Pipeline Steps

### 1. Load Data

Reads `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` into pandas DataFrames. Column alignment is enforced — only columns present in both splits are kept, preventing train/test mismatch downstream.

### 2. Drop ID & Address Columns

Drops `id`, `srcip`, `sport`, `dstip`, `dsport` **before** deduplication to avoid unique IPs masking true duplicates. Also drops the binary `label` column (not used in multiclass).

### 3. Remove Duplicates

Removes exact duplicate rows from both train and test sets independently.

### 4. Label Engineering

Extracts `attack_cat` as the multiclass target, strips whitespace, and applies `LabelEncoder` fitted on train only — never on test data.

### 5. Drop Correlated Features

Drops 8 highly correlated features (threshold > 0.95) identified during EDA:

```
sloss, dloss, dpkts, dwin, ltime, ct_srv_dst, ct_src_dport_ltm, ct_dst_src_ltm
```

### 6. Log1p Transform

Applies `log1p` to 12 skewed numerical columns to reduce the effect of outliers before scaling:

```
dur, sbytes, dbytes, sload, dload, spkts, stcpb, dtcpb, smeansz, dmeansz, sjit, djit
```

### 7. Data Cleaning

- Fills nulls with train-set mode values
- Replaces blank placeholder values (e.g. `' '`) with the column mode
- **Preserves `'-'` in categorical columns** (`proto`, `service`, `state`) — `service = '-'` is a valid category meaning no application-layer service was detected, and carries real signal for the models
- Clips binary columns (`is_sm_ips_ports`, `is_ftp_login`, `ct_flw_http_mthd`) to `{0, 1}`
- Coerces numeric columns mistakenly read as object dtype

### 8. Scaling & One-Hot Encoding

- **MinMaxScaler** fitted on 32 numerical columns (scales all values to `[0, 1]`)
- **OneHotEncoder** with fixed category lists to guarantee reproducible output:
  - `proto` → 133 categories
  - `service` → 13 categories (includes `'-'` as a valid category)
  - `state` → 16 categories

### 9. Final Feature Matrix

| Component | Count |
|-----------|-------|
| Numerical features (scaled) | 32 |
| Protocol OHE features | 133 |
| Service OHE features | 13 |
| State OHE features | 16 |
| **Total** | **194** |

---

## Feature Selection

All feature selectors are fitted **exclusively on the training set** to prevent data leakage. Five feature subsets are evaluated (k=20 for all methods):

| Feature Set | Method | Key Detail |
|-------------|--------|------------|
| All Features | Baseline | All 194 features; serves as the performance ceiling |
| MI k=20 | Filter — Mutual Information | `SelectKBest(mutual_info_classif, k=20)`; correctly prioritises continuous traffic features (duration, sbytes, dbytes, rate) |
| Chi2 k=20 | Filter — Chi-Square | `SelectKBest(chi2, k=20)`; requires non-negative input, so features are clipped to `[0, ∞)` before scoring; tends to select OHE protocol/service columns |
| RFE k=20 | Wrapper — RFE | `RFE(LinearSVC(C=1), n_features_to_select=20, step=0.3)`; step=0.3 eliminates 30% of features per round for speed |
| SFS k=20 | Wrapper — SFS | `SequentialFeatureSelector(RandomForestClassifier(n_estimators=20, max_depth=8), cv=2, direction='forward')`; 1,405s vs 59.5s for RFE |

### Feature Overlap Analysis

| Pair | Overlap |
|------|---------|
| MI ∩ RFE | 13/20 (65%) — both favour continuous flow metrics like `sbytes`, `tcprtt` |
| Chi2 ∩ RFE | 7/20 (35%) |
| Chi2 ∩ SFS | 7/20 (35%) |
| Chi2 ∩ MI | 6/20 (30%) |
| MI ∩ SFS | 3/20 (15%) — SFS selects a notably different feature profile |

---

## Models and Evaluation

Three classifiers are trained on each of the five feature subsets (15 combinations total). All models use `sklearn`'s `clone()` inside the evaluation loop to ensure no state carry-over between runs.

### Model Configurations

**SVM**
```python
make_pipeline(StandardScaler(), LinearSVC(C=10, max_iter=5000, random_state=42))
```
> Trained on a subsample of 5,000 rows due to the prohibitive cost of kernel methods at full scale on 194 features.

**Random Forest**
```python
RandomForestClassifier(
    n_estimators=300, max_depth=22, criterion='gini',
    min_samples_split=6, n_jobs=-1, random_state=42,
    class_weight='balanced'
)
```

**XGBoost**
```python
XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    objective='multi:softmax', num_class=<n_classes>
)
```

### Baseline Results (Before Grid Search)

| Model | Accuracy | Weighted F1 |
|-------|----------|-------------|
| XGBoost | 75.38% | 0.7798 ← best |
| MLP | 72.50% | 0.7481 |
| Decision Tree | 66.79% | 0.7183 |
| Random Forest | 66.16% | 0.7145 |
| SVM* | 63.60% | 0.6683 |
| Naïve Bayes | 16.12% | 0.2261 |

*SVM subsampled to 5,000 rows.

### Feature Selection Impact

- **MI was the best filter method** XGBoost with MI: 75.50% vs 68.88% with Chi2
- **RFE was the best wrapper method**  Random Forest with RFE: F1 0.7287 vs 0.6958 with SFS
- **Chi2 consistently hurt UNSW-NB15 models** by selecting OHE protocol/service columns over informative continuous features
- **Naïve Bayes benefited most** from feature selection: 16.12% → 46.62% with MI

---

## Grid Search and Hyperparameter Tuning

GridSearchCV with `StratifiedKFold(n_splits=3)` and weighted F1 scoring, run on all features.

### Parameter Grids

**Decision Tree**
```python
{
    'criterion':         ['gini', 'entropy'],
    'max_depth':         [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 5],
    'class_weight':      ['balanced', None]
}
```

**Random Forest**
```python
{
    'n_estimators':      [100, 300],
    'max_depth':         [10, 22, None],
    'min_samples_split': [2, 6],
    'class_weight':      ['balanced', None]
}
```

**XGBoost** (72 combinations × 3 folds = 216 fits, ~1,102s)
```python
{
    'n_estimators':     [50, 100],
    'max_depth':        [3, 6, 10],
    'learning_rate':    [0.01, 0.1, 0.3],
    'subsample':        [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0]
}
```

**K-Means** (clustering sweep)
```
k ∈ {9, 10, 11}, init ∈ {'k-means++', 'random'}, max_iter ∈ {100, 300}
Evaluated by ARI and NMI against ground-truth labels.
```

### Tuned Results

| Model | Before (Acc) | After (Acc) | Change |
|-------|-------------|-------------|--------|
| Decision Tree | 66.79% | 74.83% | +8.0 pp |
| Random Forest | 66.16% | 71.80% | +5.6 pp |
| XGBoost | 75.38% | 75.58% | +0.2 pp (near ceiling) |
| MLP | 72.50% | 70.97% | −1.5 pp |

> XGBoost's best CV F1 was **0.8125** — the highest of any model — confirming it had well-chosen parameters from the start. Test Weighted F1 of 0.7820 reflects the normal drop from CV to held-out data.

---

## Exploratory Data Analysis

Included visualisations:

- **Class distribution** bar charts for train and test sets
- **Correlation heatmap** of selected numerical features (used to identify the 8 dropped correlated columns)

---

## Performance Analysis

### UNSW-NB15 vs NSL-KDD

| | NSL-KDD (best) | UNSW-NB15 (best) |
|--|----------------|------------------|
| Before tuning | Decision Tree: 79.89% acc, Macro F1: 0.6289 | XGBoost: 75.38% acc, Macro F1: 0.5405 |
| After tuning | SVM: 76.76% acc, Weighted F1: 0.7313 | XGBoost: 75.58% acc, Weighted F1: 0.7820 |

NSL-KDD consistently produced higher Macro F1 scores because it has only 5 classes with clearer attack boundaries. UNSW-NB15's 10-class problem with extreme imbalance makes classification harder high accuracy masks complete failure on rare but critical attack types.

### Per-Class Weaknesses (UNSW-NB15)

Even the best tuned XGBoost model shows severe gaps on rare classes:

| Attack Class | Test Samples | F1 Score |
|---|---|---|
| Generic | many | 0.92 |
| Normal | majority | high |
| Analysis | few | 0.18 |
| Backdoor | few | 0.19 |
| Worms | **44** | 0.43 |

> High overall accuracy masks total failure on the rarest and most dangerous attack types. This is the core danger in imbalanced intrusion detection datasets.

### Computational Costs

| Operation | Time |
|-----------|------|
| SFS feature selection | ~1,405s |
| RFE feature selection | ~59.5s |
| XGBoost grid search (216 fits) | ~1,102s |

**Recommended production combination:** MI feature selection + XGBoost or Decision Tree — strong accuracy, interpretable behaviour, manageable training time.

---

## Output Artifacts

After running the preprocessing pipeline:

| File | Description |
|------|-------------|
| `final_train.pkl` | Preprocessed X_train + y_train (pickle) |
| `final_test.pkl` | Preprocessed X_test + y_test (pickle) |
| `train_preprocessed.csv` | Preprocessed training set (CSV) — input to ML notebook |
| `test_preprocessed.csv` | Preprocessed testing set (CSV) — input to ML notebook |

After running the ML notebook:

| File | Description |
|------|-------------|
| `feature_selection_dashboard.png` | 4×3 chart dashboard: per-model metrics, F1 heatmap, training time, per-class F1 |

---

## Configuration

The ML notebook exposes a top-level config block — edit these without touching any function:

```python
TRAIN_FILE  = "train_preprocessed.csv"
TEST_FILE   = "test_preprocessed.csv"
TARGET_COL  = "attack_cat"
K_VALUES    = [20]       # k for filter methods (MI, Chi2)
K_WRAPPER   = 20         # k for wrapper methods (RFE, SFS)
SAMPLE_SIZE = None       # set an int to subsample training data
```

---

## Dependencies

```
numpy
pandas
matplotlib
scikit-learn
xgboost          # pip install xgboost  (optional; notebook degrades gracefully without it)
```

Install all at once:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost
```
