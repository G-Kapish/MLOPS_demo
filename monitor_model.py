import os
import warnings
import joblib
import mlflow
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import accuracy_score, classification_report

# Suppress version warnings when loading pickle files
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# 1. Expected schema
FEATURE_COLUMNS = [
    "Temperature_C",
    "Battery_Voltage",
    "Power_Consumption_W",
    "Signal_Strength_Percent",
    "Fan_Speed_RPM",
    "Humidity_Percent",
    "Traffic_Load",
    "Tower_Age_Years",
]
TARGET_COLUMN = "Failure_Within_48Hrs"
MODEL_PATH = "telecom_tower_model.pkl"
TELEMETRY_PATH = "new_tower_telemetry.xlsx"

# 2. Load trained model
print("Loading model...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

# 3. Load production telemetry data
print("Loading new telemetry data...")
new_data = pd.read_excel(TELEMETRY_PATH)
print(f"Loaded {len(new_data)} telemetry records.")

# 4. Extract features and actual target
X_new = new_data[FEATURE_COLUMNS]
y_new = new_data[TARGET_COLUMN]

# 5. Make predictions & calculate accuracy
predictions = model.predict(X_new)
prod_accuracy = float(accuracy_score(y_new, predictions))

print("\n" + "=" * 40)
print("       MODEL DRIFT MONITORING")
print("=" * 40)
print(f"Production Accuracy: {prod_accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_new, predictions))

# 6. MLflow Tracking
mlflow.set_experiment("Telecom_Tower_Model_Monitoring")

with mlflow.start_run():
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_metric("production_accuracy", prod_accuracy)

    # 7. Drift / Degradation Alert Check
    threshold = 0.80
    if prod_accuracy < threshold:
        alert_msg = f"WARNING: Accuracy ({prod_accuracy:.2%}) dropped below threshold ({threshold:.2%}). Retraining recommended."
        print(f"\n[ALERT] {alert_msg}")
        mlflow.log_param("drift_alert", True)
    else:
        print("\n[OK] Model performance is stable.")
        mlflow.log_param("drift_alert", False)

print("\nMetrics successfully logged to MLflow.")