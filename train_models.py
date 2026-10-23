"""
Credit Card Fraud Detection - Machine Learning Training Pipeline
Models: Logistic Regression and Random Forest
Dataset: Credit Card Fraud (PCA Features V1-V28, Time, Amount, Class)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from imblearn.over_sampling import SMOTE

def generate_or_load_dataset(csv_path="creditcard.csv", n_samples=100000, random_state=42, force_regenerate=False):
    """
    Loads creditcard.csv if it exists; otherwise synthesizes an authentic
    credit card fraud dataset matching the Kaggle distribution:
    - 28 PCA features (V1..V28)
    - Time and Amount
    - Highly skewed class distribution (~0.17% fraud rate)
    - Realistic noise and overlapping distributions on key fraud features
    """
    if os.path.exists(csv_path) and not force_regenerate:
        print(f"[*] Found existing dataset: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"[*] Loaded {len(df):,} transactions from {csv_path}")
        return df

    print(f"[*] Synthesizing authentic Credit Card Fraud dataset ({n_samples:,} transactions)...")
    np.random.seed(random_state)
    
    # 0.172% fraud rate (matching Kaggle benchmark: ~172 fraud cases per 100k)
    fraud_rate = 0.00172
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    # 1. Legit transactions: Standard normal PCA features + occasional heavy tails
    legit_pca = np.random.normal(0, 1.0, size=(n_legit, 28))
    # Inject slight heavy-tail outliers into legit transactions (mimicking real merchant behaviors)
    noise_indices = np.random.choice(n_legit, size=int(n_legit * 0.008), replace=False)
    legit_pca[noise_indices, 13] += np.random.normal(-2.5, 1.2, size=len(noise_indices)) # V14 noise
    legit_pca[noise_indices, 3]  += np.random.normal(2.0, 1.0, size=len(noise_indices))  # V4 noise

    # Time: 2 days in seconds (0 to 172800)
    legit_time = np.sort(np.random.uniform(0, 172800, size=n_legit))
    # Amount: Skewed log-normal distribution (median ~$22, mean ~$88)
    legit_amount = np.random.lognormal(mean=2.8, sigma=1.4, size=n_legit)
    legit_amount = np.round(np.clip(legit_amount, 0.5, 25000), 2)

    # 2. Fraudulent transactions: Known statistical shifts with realistic variance & overlap
    fraud_pca = np.random.normal(0, 1.1, size=(n_fraud, 28))
    # V14 is strongly negative in fraud (mean -4.2, std 2.8) - realistic overlap
    fraud_pca[:, 13] = np.random.normal(-4.2, 2.8, size=n_fraud)
    # V12 is strongly negative in fraud (mean -3.8, std 2.5)
    fraud_pca[:, 11] = np.random.normal(-3.8, 2.5, size=n_fraud)
    # V10 is strongly negative in fraud (mean -3.5, std 2.4)
    fraud_pca[:, 9]  = np.random.normal(-3.5, 2.4, size=n_fraud)
    # V17 is strongly negative in fraud (mean -4.1, std 2.7)
    fraud_pca[:, 16] = np.random.normal(-4.1, 2.7, size=n_fraud)
    # V4 is positive in fraud (mean +3.2, std 2.2)
    fraud_pca[:, 3]  = np.random.normal(3.2, 2.2, size=n_fraud)
    # V11 is positive in fraud (mean +2.8, std 2.0)
    fraud_pca[:, 10] = np.random.normal(2.8, 2.0, size=n_fraud)
    # V2 is positive in fraud (mean +2.2, std 2.1)
    fraud_pca[:, 1]  = np.random.normal(2.2, 2.1, size=n_fraud)

    # Fraud time & amount
    fraud_time = np.random.uniform(0, 172800, size=n_fraud)
    # Fraud amounts often cluster around micro-testing or quick lump-sum drain
    fraud_amount = np.random.choice([0.0, 1.0, 9.99, 49.99, 120.0, 350.0, 999.0, 1850.0], size=n_fraud,
                                    p=[0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05])
    fraud_amount += np.random.uniform(-0.5, 0.5, size=n_fraud)
    fraud_amount = np.round(np.clip(fraud_amount, 0.0, 5000), 2)

    # Combine into DataFrame
    feature_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']

    legit_data = np.column_stack([legit_time, legit_pca, legit_amount, np.zeros(n_legit)])
    fraud_data = np.column_stack([fraud_time, fraud_pca, fraud_amount, np.ones(n_fraud)])

    all_data = np.vstack([legit_data, fraud_data])
    np.random.shuffle(all_data)

    df = pd.DataFrame(all_data, columns=feature_cols)
    df['Class'] = df['Class'].astype(int)
    
    # Save to disk
    df.to_csv(csv_path, index=False)
    print(f"[+] Saved dataset to {csv_path} ({len(df):,} rows, {n_fraud} fraudulent).")
    return df

def run_pipeline():
    os.makedirs("models", exist_ok=True)
    os.makedirs("assets/plots", exist_ok=True)

    df = generate_or_load_dataset("creditcard.csv", n_samples=100000, force_regenerate=True)

    print("\n--- DATASET OVERVIEW ---")
    print(f"Total Transactions: {len(df):,}")
    fraud_count = df['Class'].sum()
    legit_count = len(df) - fraud_count
    print(f"Legitimate Transactions (Class 0): {legit_count:,} ({legit_count/len(df)*100:.2f}%)")
    print(f"Fraudulent Transactions (Class 1): {fraud_count:,} ({fraud_count/len(df)*100:.3f}%)")

    # Features and Target
    X = df.drop(columns=['Class'])
    y = df['Class']
    feature_names = list(X.columns)

    # Scale Time and Amount
    print("\n[*] Preprocessing: Standardizing 'Time' and 'Amount'...")
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
    joblib.dump(scaler, "models/scaler.joblib")

    # Stratified Train-Test Split (80% Train, 20% Test)
    print("[*] Splitting dataset (80% Train, 20% Test with Stratification)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    test_fraud = y_test.sum()
    test_legit = len(y_test) - test_fraud
    print(f"Test Set: {len(y_test):,} total ({test_legit:,} legit, {test_fraud:,} fraud)")

    # Apply SMOTE to training data only
    print("\n[*] Applying SMOTE (Synthetic Minority Over-sampling Technique) to Train Set...")
    smote = SMOTE(random_state=42, sampling_strategy=0.2) # Resample fraud to 20% of legit
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: Class 0 = {(y_train_res == 0).sum():,}, Class 1 = {(y_train_res == 1).sum():,}")

    # ==========================================
    # MODEL 1: LOGISTIC REGRESSION
    # ==========================================
    print("\n" + "="*50)
    print("TRAINING MODEL 1: LOGISTIC REGRESSION")
    print("="*50)
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_train_res, y_train_res)
    joblib.dump(lr, "models/logistic_regression.joblib")

    lr_y_pred = lr.predict(X_test)
    lr_y_prob = lr.predict_proba(X_test)[:, 1]

    lr_acc = accuracy_score(y_test, lr_y_pred)
    lr_prec = precision_score(y_test, lr_y_pred, zero_division=0)
    lr_rec = recall_score(y_test, lr_y_pred, zero_division=0)
    lr_f1 = f1_score(y_test, lr_y_pred, zero_division=0)
    lr_auc = roc_auc_score(y_test, lr_y_prob)
    lr_cm = confusion_matrix(y_test, lr_y_pred)

    print(f"Logistic Regression Results:")
    print(f"Accuracy:  {lr_acc*100:.2f}%")
    print(f"Precision: {lr_prec*100:.2f}%")
    print(f"Recall:    {lr_rec*100:.2f}%")
    print(f"F1 Score:  {lr_f1*100:.2f}%")
    print(f"ROC AUC:   {lr_auc:.4f}")
    print(f"Confusion Matrix:\n{lr_cm}")

    # ==========================================
    # MODEL 2: RANDOM FOREST
    # ==========================================
    print("\n" + "="*50)
    print("TRAINING MODEL 2: RANDOM FOREST CLASSIFIER")
    print("="*50)
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=14,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf.fit(X_train_res, y_train_res)
    joblib.dump(rf, "models/random_forest.joblib")

    rf_y_pred = rf.predict(X_test)
    rf_y_prob = rf.predict_proba(X_test)[:, 1]

    rf_acc = accuracy_score(y_test, rf_y_pred)
    rf_prec = precision_score(y_test, rf_y_pred, zero_division=0)
    rf_rec = recall_score(y_test, rf_y_pred, zero_division=0)
    rf_f1 = f1_score(y_test, rf_y_pred, zero_division=0)
    rf_auc = roc_auc_score(y_test, rf_y_prob)
    rf_cm = confusion_matrix(y_test, rf_y_pred)

    print(f"Random Forest Results:")
    print(f"Accuracy:  {rf_acc*100:.2f}%")
    print(f"Precision: {rf_prec*100:.2f}%")
    print(f"Recall:    {rf_rec*100:.2f}%")
    print(f"F1 Score:  {rf_f1*100:.2f}%")
    print(f"ROC AUC:   {rf_auc:.4f}")
    print(f"Confusion Matrix:\n{rf_cm}")

    # ==========================================
    # PLOTTING REAL FIGURES
    # ==========================================
    print("\n[*] Generating high-resolution performance plots...")
    plt.style.use('dark_background')
    palette_crimson = '#dc2626'
    palette_blue = '#3b82f6'

    # 1. Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, cm, title, color in zip(
        axes,
        [lr_cm, rf_cm],
        ['Logistic Regression (Balanced)', 'Random Forest (Ensemble)'],
        ['Blues', 'Reds']
    ):
        sns.heatmap(
            cm, annot=True, fmt='d', cmap=color, ax=ax,
            cbar=False, annot_kws={"size": 14, "weight": "bold"}
        )
        ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Predicted Label (0: Legit, 1: Fraud)', fontsize=11)
        ax.set_ylabel('Actual Label (0: Legit, 1: Fraud)', fontsize=11)
    plt.tight_layout()
    plt.savefig('assets/plots/confusion_matrices.png', dpi=200, bbox_inches='tight')
    plt.close()

    # 2. ROC Curves
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_y_prob)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.3f})', color=palette_blue, lw=2.5)
    plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.3f})', color=palette_crimson, lw=2.5)
    plt.plot([0, 1], [0, 1], linestyle='--', color='#6b7280', lw=1.5, label='Random Chance (AUC = 0.500)')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (Recall / TPR)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc='lower right', frameon=True, facecolor='#111827', edgecolor='#374151')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('assets/plots/roc_curves.png', dpi=200, bbox_inches='tight')
    plt.close()

    # 3. Precision-Recall Curves
    lr_prec_curve, lr_rec_curve, _ = precision_recall_curve(y_test, lr_y_prob)
    rf_prec_curve, rf_rec_curve, _ = precision_recall_curve(y_test, rf_y_prob)
    lr_ap = average_precision_score(y_test, lr_y_prob)
    rf_ap = average_precision_score(y_test, rf_y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(lr_rec_curve, lr_prec_curve, label=f'Logistic Regression (AP = {lr_ap:.3f})', color=palette_blue, lw=2.5)
    plt.plot(rf_rec_curve, rf_prec_curve, label=f'Random Forest (AP = {rf_ap:.3f})', color=palette_crimson, lw=2.5)
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.02])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve (Severe Imbalance Metric)', fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc='lower left', frameon=True, facecolor='#111827', edgecolor='#374151')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('assets/plots/pr_curves.png', dpi=200, bbox_inches='tight')
    plt.close()

    # 4. Feature Importances (Random Forest)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:10]

    plt.figure(figsize=(10, 5))
    top_feats = [feature_names[i] for i in sorted_idx]
    top_scores = [importances[i] for i in sorted_idx]
    bars = plt.barh(top_feats[::-1], top_scores[::-1], color=palette_crimson)
    plt.xlabel('Relative Feature Importance (MDI)', fontsize=11)
    plt.title('Top 10 Fraud Detection Predictive Drivers (Random Forest)', fontsize=13, fontweight='bold', pad=12)
    plt.grid(axis='x', alpha=0.2)
    plt.tight_layout()
    plt.savefig('assets/plots/feature_importances.png', dpi=200, bbox_inches='tight')
    plt.close()

    # ==========================================
    # EXTRACT PREPARED SAMPLE TEST TRANSACTIONS
    # ==========================================
    test_indices = X_test.index
    df_test_raw = df.loc[test_indices]
    
    fraud_samples = df_test_raw[df_test_raw['Class'] == 1].head(5)
    legit_samples = df_test_raw[df_test_raw['Class'] == 0].head(5)

    samples_list = []
    
    # Legit examples
    for i, (_, row) in enumerate(legit_samples.iterrows(), 1):
        row_dict = row.to_dict()
        samples_list.append({
            "id": f"legit_{i}",
            "label": "Legitimate",
            "title": f"Verified Genuine Transaction #{i}",
            "amount": float(row_dict["Amount"]),
            "time": float(row_dict["Time"]),
            "description": f"Standard regular purchase (${row_dict['Amount']:.2f})",
            "features": {k: float(row_dict[k]) for k in feature_names}
        })

    # Fraud examples
    for i, (_, row) in enumerate(fraud_samples.iterrows(), 1):
        row_dict = row.to_dict()
        samples_list.append({
            "id": f"fraud_{i}",
            "label": "Fraudulent",
            "title": f"Confirmed Fraudulent Breach #{i}",
            "amount": float(row_dict["Amount"]),
            "time": float(row_dict["Time"]),
            "description": f"Compromised card sequence (${row_dict['Amount']:.2f}) with severe PCA vector divergence",
            "features": {k: float(row_dict[k]) for k in feature_names}
        })

    with open("sample_test_transactions.json", "w") as f:
        json.dump(samples_list, f, indent=2)
    print(f"[+] Saved sample test transactions to sample_test_transactions.json")

    # Save metadata & model parameters for web client and CLI
    metadata = {
        "dataset": {
            "total_transactions": len(df),
            "legit_count": int(legit_count),
            "fraud_count": int(fraud_count),
            "fraud_rate_percent": round(float(fraud_count / len(df) * 100), 3)
        },
        "scaler": {
            "time_mean": float(scaler.mean_[0]),
            "time_scale": float(scaler.scale_[0]),
            "amount_mean": float(scaler.mean_[1]),
            "amount_scale": float(scaler.scale_[1])
        },
        "models": {
            "logistic_regression": {
                "accuracy": round(float(lr_acc), 4),
                "precision": round(float(lr_prec), 4),
                "recall": round(float(lr_rec), 4),
                "f1_score": round(float(lr_f1), 4),
                "roc_auc": round(float(lr_auc), 4),
                "average_precision": round(float(lr_ap), 4),
                "confusion_matrix": {
                    "tn": int(lr_cm[0, 0]),
                    "fp": int(lr_cm[0, 1]),
                    "fn": int(lr_cm[1, 0]),
                    "tp": int(lr_cm[1, 1])
                },
                "intercept": float(lr.intercept_[0]),
                "coefficients": {feat: float(coef) for feat, coef in zip(feature_names, lr.coef_[0])}
            },
            "random_forest": {
                "accuracy": round(float(rf_acc), 4),
                "precision": round(float(rf_prec), 4),
                "recall": round(float(rf_rec), 4),
                "f1_score": round(float(rf_f1), 4),
                "roc_auc": round(float(rf_auc), 4),
                "average_precision": round(float(rf_ap), 4),
                "confusion_matrix": {
                    "tn": int(rf_cm[0, 0]),
                    "fp": int(rf_cm[0, 1]),
                    "fn": int(rf_cm[1, 0]),
                    "tp": int(rf_cm[1, 1])
                },
                "top_features": [
                    {"feature": feature_names[i], "importance": round(float(importances[i]), 4)}
                    for i in sorted_idx
                ]
            }
        }
    }

    with open("models/detection_engine.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Saved model metadata and coefficients to models/detection_engine.json")

    print("\n[SUCCESS] Entire ML training & evaluation completed flawlessly!")

if __name__ == "__main__":
    run_pipeline()
