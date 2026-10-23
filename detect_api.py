"""
Credit Card Fraud Detection - Real-time REST API
Provides endpoints for running real inference on transactions and retrieving test cases.
Run with:
    python detect_api.py
"""

import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load trained models & metadata
print("[*] Initializing Fraud Detection Engine...")
try:
    rf_model = joblib.load("models/random_forest.joblib")
    lr_model = joblib.load("models/logistic_regression.joblib")
    scaler = joblib.load("models/scaler.joblib")
    with open("sample_test_transactions.json", "r") as f:
        SAMPLE_TRANSACTIONS = json.load(f)
    with open("models/detection_engine.json", "r") as f:
        DETECTION_METADATA = json.load(f)
    FEATURE_NAMES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    print("[+] Models and metadata successfully loaded.")
except Exception as e:
    print(f"[!] Warning: could not load all model files: {e}")
    rf_model, lr_model, scaler = None, None, None
    SAMPLE_TRANSACTIONS = []
    DETECTION_METADATA = {}
    FEATURE_NAMES = []

def score_transaction(features_dict):
    df_single = pd.DataFrame([features_dict], columns=FEATURE_NAMES)
    amount_raw = float(df_single['Amount'].iloc[0])
    time_raw = float(df_single['Time'].iloc[0])

    df_scaled = df_single.copy()
    df_scaled[['Time', 'Amount']] = scaler.transform(df_single[['Time', 'Amount']])

    rf_prob = float(rf_model.predict_proba(df_scaled)[0, 1])
    lr_prob = float(lr_model.predict_proba(df_scaled)[0, 1])
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

    anomalies = []
    for col in ['V14', 'V4', 'V12', 'V10', 'V17', 'V11', 'V2']:
        if col in features_dict:
            val = float(features_dict[col])
            if col in ['V14', 'V12', 'V10', 'V17']:
                sig = "HEAVY NEGATIVE DEFLECTION" if val < -3.0 else "NORMAL RANGE"
            else:
                sig = "SPIKE IN VECTOR MAGNITUDE" if val > 2.5 else "NORMAL RANGE"
            anomalies.append({"feature": col, "value": round(val, 2), "signal": sig})

    anomalies.sort(key=lambda x: abs(x['value']), reverse=True)

    return {
        "amount": amount_raw,
        "time": time_raw,
        "is_fraud": is_fraud,
        "rf_fraud_prob": round(rf_prob, 4),
        "lr_fraud_prob": round(lr_prob, 4),
        "consensus_prob": round(consensus_prob, 4),
        "risk_level": risk_level,
        "recommendation": rec,
        "anomalies": anomalies
    }

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "models_ready": rf_model is not None and lr_model is not None,
        "total_test_samples": len(SAMPLE_TRANSACTIONS)
    })

@app.route('/api/samples', methods=['GET'])
def get_samples():
    return jsonify(SAMPLE_TRANSACTIONS)

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    return jsonify(DETECTION_METADATA)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # Check if sample ID provided
        if "sample_id" in data:
            sample = next((s for s in SAMPLE_TRANSACTIONS if s["id"] == data["sample_id"]), None)
            if not sample:
                return jsonify({"error": f"Sample ID {data['sample_id']} not found"}), 404
            features = sample["features"]
            result = score_transaction(features)
            result["sample_info"] = {
                "id": sample["id"],
                "title": sample["title"],
                "expected": sample["label"]
            }
            return jsonify(result)

        # Custom transaction
        features = {feat: 0.0 for feat in FEATURE_NAMES}
        if "features" in data and isinstance(data["features"], dict):
            features.update(data["features"])
        else:
            for k in FEATURE_NAMES:
                if k in data:
                    features[k] = float(data[k])

        if "amount" in data:
            features["Amount"] = float(data["amount"])
        if "time" in data:
            features["Time"] = float(data["time"])

        result = score_transaction(features)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    print("[*] Starting Fraud Detection Inference API on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)
