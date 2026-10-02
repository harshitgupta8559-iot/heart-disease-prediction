"""
Heart Disease Prediction System - Flask Backend API
===================================================

Educational Machine Learning Demonstration
------------------------------------------
DISCLAIMER: This application is an educational machine-learning demonstration
and is NOT a certified medical diagnostic system. Predictions must not be used
for actual clinical diagnosis or treatment planning.

Beginner-Friendly Concepts Explained:
-------------------------------------
1. What does Flask do?
   Flask is a lightweight, flexible web framework written in Python. It allows us
   to create a web server that listens for incoming HTTP requests (from web browsers,
   mobile apps, or frontend user interfaces) and returns appropriate responses
   (such as web pages or JSON data). In our architecture, Flask acts as the bridge
   between the machine learning model and the user interface.

2. What does the '/' route do?
   The root route ('/') handles GET requests sent to the base URL of our server.
   It renders and serves our interactive web interface (templates/index.html),
   allowing users to input the 13 clinical features and view real-time predictions.

3. What does the '/predict' route do?
   The '/predict' route is an API endpoint that handles HTTP POST requests. It expects
   a JSON payload containing a patient's 13 clinical measurements. The endpoint:
   - Validates that all 13 required fields are provided and numeric.
   - Converts the data into a pandas DataFrame with the exact feature order.
   - Scales the input using our pre-fitted scaler.
   - Uses the trained machine learning model to produce a prediction (0 or 1)
     and its associated probability.
   - Returns a structured JSON response to the client.

4. Why must the scaler be reused?
   Machine learning algorithms (such as Logistic Regression) are sensitive to feature
   scales. During training, StandardScaler calculated the exact mean and standard
   deviation for each feature from the training dataset. We MUST reuse this exact
   fitted scaler (loaded from model/scaler.joblib) using `scaler.transform()` so that
   new patient inputs are normalized using the exact same baseline.
   NEVER re-fit (`fit()` or `fit_transform()`) on a single patient sample, because that
   would destroy the relationship learned by the model.

5. Why must feature order remain identical?
   Machine learning models do not understand feature names like "age" or "chol"; they
   only process an array of numbers at specific indices (column 0, column 1, etc.).
   If the order of features changes (for example, swapping 'age' and 'chol'), the model
   would mistakenly interpret a cholesterol reading of 240 as the patient's age, and
   an age of 55 as the cholesterol level, resulting in completely incorrect predictions.
"""

import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS) for future frontend integration
# This ensures that frontend applications running on different ports or origins
# (e.g., Live Server or file://) can communicate with our Flask API without browser CORS errors.
CORS(app)

# Resolve paths to model artifacts
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "heart_disease_model.joblib"
SCALER_PATH = BASE_DIR / "model" / "scaler.joblib"
METADATA_PATH = BASE_DIR / "model" / "model_metadata.json"

# Load trained model, scaler, and metadata at server startup
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # EXACT 13 features expected in exact order
    EXPECTED_FEATURES = metadata.get("features", [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ])
    print(f"[+] Loaded model: {metadata.get('model_name', 'Trained Model')}")
    print(f"[+] Loaded scaler: {SCALER_PATH.name}")
    print(f"[+] Expected features ({len(EXPECTED_FEATURES)}): {EXPECTED_FEATURES}")
except Exception as e:
    print(f"[!] Critical Error loading model artifacts: {e}")
    model = None
    scaler = None
    EXPECTED_FEATURES = []


@app.route("/", methods=["GET"])
def home():
    """
    GET /: Renders and serves the frontend interface (templates/index.html).
    """
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict: API route for generating heart disease predictions.

    Expects a JSON body with exactly 13 numeric fields:
    - age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal

    Returns:
    - JSON response with prediction, probability, result description, and educational disclaimer.
    - HTTP 400 for missing or invalid inputs.
    - HTTP 500 if the server encounters an internal error.
    """
    # Verify that model and scaler were successfully loaded
    if model is None or scaler is None:
        return jsonify({
            "success": False,
            "error": "Model or scaler artifact is not loaded on the server."
        }), 500

    # 1. Validate that the incoming request contains JSON data
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "Request body must be valid JSON with 'Content-Type: application/json'."
        }), 400

    try:
        data = request.get_json(silent=True)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to parse JSON body: {str(e)}"
        }), 400

    if not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": "Request JSON body must be a key-value object containing the 13 clinical features."
        }), 400

    # 2. Check that all 13 expected fields are present
    missing_fields = [feature for feature in EXPECTED_FEATURES if feature not in data]
    if missing_fields:
        return jsonify({
            "success": False,
            "error": f"Missing required field(s): {', '.join(missing_fields)}"
        }), 400

    # 3. Validate that each field contains a valid numeric value
    cleaned_input = {}
    for feature in EXPECTED_FEATURES:
        val = data[feature]

        # Reject null, None, or empty string values
        if val is None or val == "":
            return jsonify({
                "success": False,
                "error": f"Field '{feature}' cannot be null or empty."
            }), 400

        # Attempt conversion to float
        try:
            numeric_val = float(val)
        except (ValueError, TypeError):
            return jsonify({
                "success": False,
                "error": f"Invalid numeric value for field '{feature}': received '{val}'"
            }), 400

        # Reject NaN or Infinite values
        if np.isnan(numeric_val) or np.isinf(numeric_val):
            return jsonify({
                "success": False,
                "error": f"Field '{feature}' cannot be NaN or Infinity."
            }), 400

        cleaned_input[feature] = numeric_val

    # 4. Convert input into a pandas DataFrame using the EXACT feature order from training
    try:
        input_df = pd.DataFrame([cleaned_input], columns=EXPECTED_FEATURES)

        # 5. Apply the pre-fitted scaler (transform ONLY, never fit)
        input_scaled = scaler.transform(input_df)

        # 6. Generate model prediction (0 or 1)
        prediction = int(model.predict(input_scaled)[0])

        # 7. Generate prediction probability if supported by the model
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_scaled)[0]
            # Probability of the predicted class (confidence score)
            probability = round(float(probabilities[prediction]), 4)
        else:
            probability = 1.0

        # 8. Formulate descriptive result text
        # Educational reminder: prediction 0 indicates no heart disease detected according
        # to dataset criteria, but does NOT medically prove a patient is healthy.
        if prediction == 1:
            result_text = "Heart disease prediction: Positive"
        else:
            result_text = "Heart disease prediction: Negative (No heart disease indicated according to dataset label)"

        # 9. Return the successful JSON response
        return jsonify({
            "success": True,
            "prediction": prediction,
            "result": result_text,
            "probability": probability,
            "disclaimer": "Educational prediction only. Not a medical diagnosis."
        }), 200

    except Exception as e:
        # Catch unexpected prediction errors without crashing the Flask server
        return jsonify({
            "success": False,
            "error": f"An error occurred during prediction processing: {str(e)}"
        }), 500


if __name__ == "__main__":
    # Start the Flask development server on 127.0.0.1:5000
    print("[*] Starting Flask application on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
