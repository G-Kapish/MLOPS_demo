import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Expected feature sequence matching training dataset
EXPECTED_FEATURES = [
    "Temperature_C",
    "Battery_Voltage",
    "Power_Consumption_W",
    "Signal_Strength_Percent",
    "Fan_Speed_RPM",
    "Humidity_Percent",
    "Traffic_Load",
    "Tower_Age_Years",
]

# Load model artifact
MODEL_PATH = "telecom_tower_model.pkl"
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "status": "online",
            "message": "Telecom Tower Prediction API is running successfully!",
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid or missing JSON payload"}), 400

        # Support single dictionary or list of records
        if isinstance(data, dict):
            input_df = pd.DataFrame([data])
        elif isinstance(data, list):
            input_df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Payload must be a JSON object or array of objects"}), 400

        # Validate missing features
        missing_cols = [col for col in EXPECTED_FEATURES if col not in input_df.columns]
        if missing_cols:
            return (
                jsonify({"error": f"Missing required feature columns: {missing_cols}"}),
                400,
            )

        # Enforce exact column order expected by Scikit-Learn
        input_features = input_df[EXPECTED_FEATURES]

        # Generate predictions
        predictions = model.predict(input_features).tolist()

        return jsonify(
            {
                "predictions": predictions,
                "count": len(predictions),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)