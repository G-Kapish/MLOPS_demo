import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Load dataset (relative path)
df = pd.read_excel("Telecom_Tower_Failure_Dataset_10000-1.xlsx")

# 2. Separate features and target (drop identifier columns if present)
features = [
    "Temperature_C",
    "Battery_Voltage",
    "Power_Consumption_W",
    "Signal_Strength_Percent",
    "Fan_Speed_RPM",
    "Humidity_Percent",
    "Traffic_Load",
    "Tower_Age_Years",
]

X = df[features]
y = df["Failure_Within_48Hrs"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4. Train model
print("Training Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate on test set
y_pred = model.predict(X_test)
accuracy = float(accuracy_score(y_test, y_pred))

print(f"Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Save model artifact and metrics once
joblib.dump(model, "telecom_tower_model.pkl")

with open("metrics.json", "w") as f:
    json.dump({"accuracy": accuracy}, f, indent=4)

print("Training Completed Successfully!")