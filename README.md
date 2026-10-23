# Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/SMOTE-imblearn-red)](https://imbalanced-learn.org/)

An educational machine-learning project that compares **Logistic Regression** and **Random Forest** for highly imbalanced transaction classification. The training pipeline generates a reproducible **synthetic dataset** (100,000 rows; 172 fraud examples) and applies SMOTE to training data.

This is not the original ULB/Kaggle dataset and is not suitable for real transaction decisions. The browser detector uses the trained models through the local Flask API; it does not simulate predictions when the API is unavailable.

---

## 🚀 Key Deliverables

1. **Working ML Training Pipeline (`train_models.py`)**:
   - Generates 100,000 reproducible synthetic transactions with 28 PCA-style features, Time, Amount, and a Class label.
   - Standardizes `Time` and `Amount` using `StandardScaler`.
   - Splits 80/20 with Stratification (preserving exact fraud ratio).
   - Applies **SMOTE** on the training partition to balance minority representation without data leakage.
   - Trains and exports **Logistic Regression** and **Random Forest** models (`.joblib`).
   - Exports high-resolution evaluation figures (Confusion Matrices, ROC curves, PR curves, Feature Importances).

2. **Live CLI Inference Engine (`detect_cli.py`)**:
   - Run predictions on generated sample transactions or custom parameters.

3. **REST API Backend (`detect_api.py`)**:
   - Flask API serving `/api/predict`, `/api/samples`, `/api/metadata`, and `/api/health`.

4. **Academic Jupyter Notebook (`fraud_detection.ipynb`)**:
   - Complete exploratory data analysis (EDA), code cells, visualizations, cross-validation, and narrative for lecturer review.

5. **3-Page Web Portfolio & Interactive Detector**:
   - `index.html`: Project Overview + **Live Interactive Fraud Detection Lab** (instant 1-click test scenarios + custom slider scanner).
   - `methodology.html`: Complete 6-stage ML pipeline, SMOTE theory, and model hyperparameters.
   - `results.html`: Real evaluation results, confusion matrices, and embedded Matplotlib telemetry plots.

---

## Model benchmark results

Evaluation conducted on a stratified held-out test partition of **20,000 synthetic transactions** (34 fraud cases):

| Metric | Logistic Regression | Random Forest | Better result |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 99.99% | 99.98% | **LR** |
| **Precision** | 97.06% | **100.00%** | **RF** |
| **Recall (Sensitivity)** | **97.06%** | 88.24% | **LR** |
| **F1 Score** | **97.06%** | 93.75% | **LR** |
| **ROC-AUC** | 0.9999 | **1.0000** | **RF** |
| **Average Precision (PR-AUC)** | 0.9851 | 0.9861 | **RF** |
| **False Positives (Alarms)** | 1 | **0** | **RF** |
| **False Negatives (Missed Fraud)** | **1** | 4 | **LR** |

These unusually strong scores are results on generated data whose fraud patterns were deliberately constructed. They should not be interpreted as evidence of real-world performance.

---

## ⚡ Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Models & Generate Plots
```bash
python train_models.py
```
This trains the models, saves `models/*.joblib`, exports `models/detection_engine.json`, and generates evaluation plots in `assets/plots/`.

### 3. Run the model API
```bash
python detect_api.py
```

### 4. Run CLI inference
Test a confirmed fraudulent transaction:
```bash
python detect_cli.py --sample fraud_1
```
Test a generated legitimate sample:
```bash
python detect_cli.py --sample legit_1
```
List all available test scenarios:
```bash
python detect_cli.py --list-samples
```
Test custom transaction parameters:
```bash
python detect_cli.py --amount 1500.0 --v14 -6.2 --v4 4.1
```

### 5. Launch the portfolio and detector
In a second terminal, from the project root:
```bash
python -m http.server 3000
```
Open [http://localhost:3000](http://localhost:3000). Keep the Flask API running in the first terminal for model-backed predictions. Custom browser scans populate four PCA features; the other PCA values are set to zero.

---

## 📁 Repository Structure

```text
├── assets/
│   └── plots/
│       ├── confusion_matrices.png   # Matplotlib confusion matrix comparison
│       ├── feature_importances.png  # Top 10 predictive fraud drivers
│       ├── pr_curves.png            # Precision-Recall curves
│       └── roc_curves.png           # Receiver Operating Characteristic
├── models/
│   ├── detection_engine.json        # Exported metrics, coefficients & metadata
│   ├── logistic_regression.joblib   # Serialized Logistic Regression model
│   ├── random_forest.joblib         # Serialized Random Forest model
│   └── scaler.joblib                # Fitted StandardScaler
├── build_notebook.py                # Generator for academic Jupyter notebook
├── creditcard.csv                   # 100,000 generated synthetic transactions
├── detect_api.py                    # REST API backend for live predictions
├── detect_cli.py                    # Command-line fraud detection scanner
├── fraud_detection.ipynb            # Academic Jupyter Notebook for submission
├── index.html                       # Page 1: Overview + Live Fraud Detection Lab
├── methodology.html                 # Page 2: Pipeline, SMOTE & hyperparameters
├── results.html                     # Page 3: Benchmark metrics & analysis
├── sample_test_transactions.json    # Verified sample transactions (fraud & legit)
├── script.js                        # Particle canvas, animations & JS inference
├── style.css                        # White and blue design system
├── train_models.py                  # Full ML training & evaluation pipeline
└── requirements.txt                 # Project dependencies
```

---

## Deploying the portfolio

The static portfolio can be hosted with GitHub Pages (Settings → Pages → deploy the `main` branch from `/ (root)`). GitHub Pages does not host the Flask API. The detector therefore requires the API running on the visitor's machine; it will show an offline message rather than make up predictions if the API is unavailable.

---

## 👤 Author
- **Jabulani Mayenda**
- GitHub: [jabulani-mayenda](https://github.com/jabulani-mayenda)
- Repository: [fruad-detection](https://github.com/jabulani-mayenda/fruad-detection)
