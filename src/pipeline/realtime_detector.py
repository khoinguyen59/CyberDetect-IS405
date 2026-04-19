"""
Real-time Cyberattack Detection and Alert Generation.

Paper references:
  - Algorithm 4 (line 839): Real-time cyberattack detection and alert generation
  - Line 862-863: "logs detection results, including timestamps and confidence scores"
  - Line 947: "stored in a distributed NoSQL database such as MongoDB or Apache Cassandra"
  - Line 948-949: "Each detected attack instance triggers an automated alert mechanism"
  - Line 950: "device ID, attack type, timestamp, and severity score"
"""

import pandas as pd
import json
import os
from datetime import datetime
from tensorflow.keras.models import load_model


# Severity mapping for attack types (paper line 950: "severity score")
SEVERITY_MAP = {
    0: {"level": "INFO", "severity": 0.0, "type": "Normal"},
    1: {"level": "HIGH", "severity": 0.9, "type": "DDoS"},
    2: {"level": "CRITICAL", "severity": 1.0, "type": "Ransomware"},
    3: {"level": "MEDIUM", "severity": 0.6, "type": "Reconnaissance"},
    4: {"level": "HIGH", "severity": 0.8, "type": "Backdoor"},
    5: {"level": "MEDIUM", "severity": 0.5, "type": "Injection"},
    6: {"level": "HIGH", "severity": 0.7, "type": "XSS"},
    7: {"level": "MEDIUM", "severity": 0.6, "type": "Password"},
    8: {"level": "LOW", "severity": 0.3, "type": "Scanning"},
    9: {"level": "HIGH", "severity": 0.85, "type": "MITM"},
}


def _try_mongodb_insert(results, mongo_uri="mongodb://localhost:27017",
                        db_name="cyberdetect", collection_name="alerts"):
    """
    Attempt to insert detection results into MongoDB.

    Paper line 947: "stored in a distributed NoSQL database such as MongoDB"

    Returns True if successful, False otherwise (triggers JSON fallback).
    """
    try:
        from pymongo import MongoClient
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        # Test connection
        client.server_info()
        db = client[db_name]
        collection = db[collection_name]
        collection.insert_many(results)
        client.close()
        print(f"✅ {len(results)} alerts stored in MongoDB ({db_name}.{collection_name})")
        return True
    except ImportError:
        print("⚠️ pymongo not installed. Falling back to JSON file storage.")
        return False
    except Exception as e:
        print(f"⚠️ MongoDB connection failed ({e}). Falling back to JSON file storage.")
        return False


def simulate_detection(model_path, input_csv, output_json):
    """
    Real-time detection simulation — Paper Algorithm 4 (line 839).

    Pipeline:
      1. Load trained model (line 860)
      2. Preprocess input (normalization + feature selection already done)
      3. Forward propagation → class probabilities (line 861)
      4. Classify via argmax (line 861)
      5. Log to NoSQL (MongoDB) with timestamps + confidence (line 862-863, 947)
      6. Generate alerts for critical detections (line 948-949)
    """
    model = load_model(model_path)
    df = pd.read_csv(input_csv)
    X = df.drop('label', axis=1).values

    preds = model.predict(X)
    labels = preds.argmax(axis=1)
    probs = preds.max(axis=1)

    # Paper line 862: "timestamps and confidence scores"
    # Paper line 950: "device ID, attack type, timestamp, and severity score"
    results = []
    critical_alerts = []

    for i in range(len(labels)):
        pred_class = int(labels[i])
        severity_info = SEVERITY_MAP.get(pred_class, {"level": "UNKNOWN", "severity": 0.0, "type": "Unknown"})

        record = {
            "device_id": f"IoT-{i:06d}",
            "predicted_class": pred_class,
            "attack_type": severity_info["type"],
            "confidence": float(probs[i]),
            "severity_score": severity_info["severity"],
            "severity_level": severity_info["level"],
            "timestamp": datetime.now().isoformat(),
        }
        results.append(record)

        # Paper line 950: "For critical attacks such as ransomware or DDoS,
        # the system generates high-priority alerts"
        if severity_info["level"] in ("CRITICAL", "HIGH"):
            critical_alerts.append(record)

    # Attempt MongoDB storage first (paper line 947), fallback to JSON
    mongo_success = _try_mongodb_insert(results)

    if not mongo_success:
        # JSON fallback
        os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else ".", exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ {len(results)} alerts saved to {output_json}")

    # Print critical alert summary
    print(f"\n🚨 Critical/High-priority alerts: {len(critical_alerts)}/{len(results)}")
    if critical_alerts[:5]:
        print("   Top 5 critical alerts:")
        for alert in critical_alerts[:5]:
            print(f"   - Device {alert['device_id']}: {alert['attack_type']} "
                  f"(confidence={alert['confidence']:.4f}, severity={alert['severity_score']})")

    print(f"✅ Detection pipeline complete. Total predictions: {len(results)}")
