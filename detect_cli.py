"""
Credit Card Fraud Detection - Command Line Interface (CLI) Detector
Runs live inference using trained Random Forest and Logistic Regression models.
Usage:
    python detect_cli.py --sample fraud_1
    python detect_cli.py --sample legit_1
    python detect_cli.py --list-samples
    python detect_cli.py --amount 1250.0 --v14 -6.2 --v4 4.1
"""

import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd

def load_resources():
    try:
        rf_model = joblib.load("models/random_forest.joblib")
        lr_model = joblib.load("models/logistic_regression.joblib")
        scaler = joblib.load("models/scaler.joblib")
        with open("sample_test_transactions.json", "r") as f:
            samples = json.load(f)
        with open("models/detection_engine.json", "r") as f:
            metadata = json.load(f)
        return rf_model, lr_model, scaler, samples, metadata
    except Exception as e:
        print(f"Error loading model resources: {e}")
        print("Please run `python train_models.py` first to generate models and metadata.")
        sys.exit(1)

def format_prediction_output(result, sample_name=None):
    print("\n" + "="*60)
    print("      REAL-TIME CREDIT CARD FRAUD DETECTION SCAN")
    print("="*60)
    if sample_name:
        print(f"Sample Tested:         {sample_name}")
    print(f"Transaction Amount:    ${result['amount']:,.2f}")
    print(f"Transaction Time:      {result['time']:,.0f} seconds")
    print("-" * 60)
    
    # Status Banner
    if result['is_fraud']:
        print(f"[!] STATUS: FRAUD DETECTED - {result['risk_level']} RISK")
    else:
        print(f"[OK] STATUS: TRANSACTION APPROVED (LEGITIMATE)")
    print("-" * 60)
    print(f"Random Forest Probability:     {result['rf_fraud_prob']*100:.2f}%")
    print(f"Logistic Regression Prob:      {result['lr_fraud_prob']*100:.2f}%")
    print(f"Recommended System Action:     {result['recommendation']}")
    print("-" * 60)
    print("Top Feature Anomaly Drivers:")
    for feat in result['anomalies'][:4]:
        print(f"  * {feat['feature']:<5} = {feat['value']:>7.2f}  [{feat['signal']}]")
    print("="*60 + "\n")

def analyze_transaction(features_dict, rf_model, lr_model, scaler, feature_names):
    df_single = pd.DataFrame([features_dict], columns=feature_names)
    amount_raw = float(df_single['Amount'].iloc[0])
    time_raw = float(df_single['Time'].iloc[0])

    # Scale Time and Amount
    df_scaled = df_single.copy()
    df_scaled[['Time', 'Amount']] = scaler.transform(df_single[['Time', 'Amount']])

    # Predict Probabilities
    rf_prob = float(rf_model.predict_proba(df_scaled)[0, 1])
    lr_prob = float(lr_model.predict_proba(df_scaled)[0, 1])
    
    # Combined consensus probability
    consensus_prob = (rf_prob * 0.7) + (lr_prob * 0.3)
    is_fraud = consensus_prob >= 0.50

    if consensus_prob >= 0.80:
        risk_level = "CRITICAL"
        rec = "BLOCK CARD & DISPATCH IMMEDIATE SMS / 2FA CHALLENGE"
    elif consensus_prob >= 0.50:
        risk_level = "HIGH"
        rec = "SUSPEND TRANSACTION FOR SECONDARY FRAUD TEAM REVIEW"
    elif consensus_prob >= 0.20:
        risk_level = "ELEVATED"
        rec = "ALLOW WITH PASSIVE VELOCITY MONITORING"
    else:
        risk_level = "LOW"
        rec = "APPROVE INSTANTLY (LOW RISK PROFILE)"

    # Identify top anomalous signals (especially known fraud vectors: V14, V4, V12, V10, V17)
    anomalies = []
    for col in ['V14', 'V4', 'V12', 'V10', 'V17', 'V11', 'V2']:
        if col in features_dict:
            val = float(features_dict[col])
            if col in ['V14', 'V12', 'V10', 'V17']:
                signal = "HEAVY NEGATIVE DEFLECTION" if val < -3.0 else "NORMAL RANGE"
            else:
                signal = "SPIKE IN VECTOR MAGNITUDE" if val > 2.5 else "NORMAL RANGE"
            anomalies.append({"feature": col, "value": val, "signal": signal})

    anomalies.sort(key=lambda x: abs(x['value']), reverse=True)

    return {
        "amount": amount_raw,
        "time": time_raw,
        "is_fraud": is_fraud,
        "rf_fraud_prob": rf_prob,
        "lr_fraud_prob": lr_prob,
        "consensus_prob": consensus_prob,
        "risk_level": risk_level,
        "recommendation": rec,
        "anomalies": anomalies
    }

def main():
    parser = argparse.ArgumentParser(description="Live Credit Card Fraud Detection Inference CLI")
    parser.add_argument("--sample", type=str, help="ID of pre-configured sample transaction (e.g. fraud_1, legit_1)")
    parser.add_argument("--list-samples", action="store_true", help="List available sample test transactions")
    parser.add_argument("--amount", type=float, default=120.0, help="Transaction amount in USD")
    parser.add_argument("--time", type=float, default=45000.0, help="Elapsed time in seconds")
    parser.add_argument("--v14", type=float, default=0.0, help="PCA feature V14 value (negative values often indicate fraud)")
    parser.add_argument("--v4", type=float, default=0.0, help="PCA feature V4 value (positive values often indicate fraud)")
    parser.add_argument("--v12", type=float, default=0.0, help="PCA feature V12 value")
    parser.add_argument("--v10", type=float, default=0.0, help="PCA feature V10 value")

    args = parser.parse_args()

    rf_model, lr_model, scaler, samples, metadata = load_resources()
    feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

    if args.list_samples:
        print("\n--- AVAILABLE SAMPLE TRANSACTIONS ---")
        for s in samples:
            print(f"[{s['id']}] {s['label'].upper():<11} | ${s['amount']:>8.2f} | {s['title']}")
        print("\nRun with: python detect_cli.py --sample <sample_id>\n")
        return

    if args.sample:
        target = next((s for s in samples if s['id'].lower() == args.sample.lower()), None)
        if not target:
            print(f"Error: Sample '{args.sample}' not found. Use --list-samples to see options.")
            return
        result = analyze_transaction(target['features'], rf_model, lr_model, scaler, feature_names)
        format_prediction_output(result, sample_name=f"{target['title']} (Expected: {target['label']})")
        return

    # Custom transaction inference
    features = {feat: 0.0 for feat in feature_names}
    features['Amount'] = args.amount
    features['Time'] = args.time
    features['V14'] = args.v14
    features['V4'] = args.v4
    features['V12'] = args.v12
    features['V10'] = args.v10

    result = analyze_transaction(features, rf_model, lr_model, scaler, feature_names)
    format_prediction_output(result, sample_name="Custom User Input Transaction")

if __name__ == "__main__":
    main()
