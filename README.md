# UNSW-NB15 Preprocessing Pipeline — Multiclass Classification

A clean, modular preprocessing pipeline for the **UNSW-NB15 network intrusion detection dataset**, designed for multiclass classification using `attack_cat` as the target label.

---

## Project Overview

This project prepares the UNSW-NB15 dataset for machine learning by performing end-to-end preprocessing: deduplication, feature engineering, correlation-based feature selection, scaling, and one-hot encoding — producing a final feature matrix of exactly **194 features** ready for model training.

---

## Dataset

**UNSW-NB15** is a network traffic dataset created by the Australian Centre for Cyber Security (ACCS). It contains both normal and attack traffic across 10 categories.

| File | Description |
|------|-------------|
| `UNSW_NB15_training-set.csv` | Raw training data |
| `UNSW_NB15_testing-set.csv` | Raw testing data |

> Download the dataset from the [official UNSW page](https://research.unsw.edu.au/projects/unsw-nb15-dataset).

### Attack Categories (Multiclass Target)

`Normal`, `Fuzzers`, `Analysis`, `Backdoors`, `DoS`, `Exploits`, `Generic`, `Reconnaissance`, `Shellcode`, `Worms`

---

## Pipeline Steps

### 1. Load Data
Reads the train and test CSV files into pandas DataFrames.

### 2. Drop ID & Address Columns
Drops `id`, `srcip`, `sport`, `dstip`, `dsport` **before** deduplication to avoid unique IPs masking true duplicates. Also drops the binary `label` column (not used in multiclass).

### 3. Remove Duplicates
Removes exact duplicate rows from both train and test sets independently.

### 4. Label Engineering
Extracts `attack_cat` as the multiclass target, strips whitespace, and applies `LabelEncoder` fitted on train only.

### 5. Drop Correlated Features
Drops 8 highly correlated features (threshold > 0.95) identified during EDA:

```
sloss, dloss, dpkts, dwin, ltime, ct_srv_dst, ct_src_dport_ltm, ct_dst_src_ltm
```

### 6. Log1p Transform
Applies `log1p` to 12 skewed numerical columns to reduce the effect of outliers:

```
dur, sbytes, dbytes, sload, dload, spkts, stcpb, dtcpb, smeansz, dmeansz, sjit, djit
```

### 7. Data Cleaning
- Fills nulls with train-set mode values
- Replaces bad placeholder values (e.g. `' '`, `'-'`)
- Clips binary columns (`is_sm_ips_ports`, `is_ftp_login`, `ct_flw_http_mthd`) to `{0, 1}`
- Coerces numeric columns mistakenly read as object

### 8. Scaling & One-Hot Encoding
- **StandardScaler** fitted on 32 numerical columns
- **OneHotEncoder** with fixed category lists to guarantee reproducible output:
  - `proto` → 133 categories
  - `service` → 13 categories
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

## Exploratory Data Analysis (EDA)

Included visualizations:
- **Class distribution** bar charts for train and test sets
- **Correlation heatmap** of selected numerical features

---

## Output Artifacts

After running the pipeline, the following files are saved:

| File | Description |
|------|-------------|
| `final_train.pkl` | Preprocessed X_train + y_train (pickle) |
| `final_test.pkl` | Preprocessed X_test + y_test (pickle) |
| `train_preprocessed.csv` | Preprocessed training set (CSV) |
| `test_preprocessed.csv` | Preprocessed testing set (CSV) |
