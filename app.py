import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

import shap

from data_preprocessing import DataPreprocessor, FEATURE_NAMES

app = Flask(__name__)

# ========================
# Load model + scaler once at startup
# ========================
MODEL_PATH  = "model/weights.pkl"
SCALER_PATH = "model/scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError(
        "Trained model not found.\n"
        "Run:  python training.py  then  python shap_analysis.py  first."
    )

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

scaler_data = DataPreprocessor.load_scaler(SCALER_PATH)
scaler      = scaler_data["scaler"]
medians     = scaler_data["medians"]
iqr_bounds  = scaler_data["iqr_bounds"]
feat_stats  = scaler_data.get("feature_stats", {})

# ========================
# SHAP explainer
# Background = real training data loaded from scaler file,
# or re-derived from the dataset if not cached there.
# ========================
def _build_background():
    """Load a small real background for SHAP from the training dataset."""
    try:
        dp = DataPreprocessor("dataset/diabetes.csv")
        X_train, _, _, _, _, _ = dp.preprocess(smote=False)
        rng   = np.random.default_rng(42)
        idx   = rng.choice(len(X_train), size=min(100, len(X_train)), replace=False)
        return X_train[idx]
    except Exception:
        # Last-resort fallback: zeros (better than random noise)
        print("⚠️  Could not load real background for SHAP; using zeros.")
        return np.zeros((50, len(FEATURE_NAMES)))

background = _build_background()

def model_fn(X):
    return model.predict_proba(X)

explainer = shap.Explainer(model_fn, background)


# ========================
# Input preprocessing
# ========================
def preprocess_input(raw_values: list) -> np.ndarray:
    """
    Apply the same transforms as training:
      1. Replace physiologically-impossible zeros with train medians
      2. IQR clipping
      3. Z-score scaling
    """
    arr = np.array(raw_values, dtype=float)

    for i, col in enumerate(FEATURE_NAMES):
        # Zero imputation
        if col in medians and arr[i] == 0.0:
            arr[i] = medians[col]
        # IQR clipping
        if col in iqr_bounds:
            lo, hi = iqr_bounds[col]
            arr[i] = np.clip(arr[i], lo, hi)

    # Z-score normalization
    arr_scaled = scaler.transform(arr.reshape(1, -1))
    return arr_scaled


# ========================
# Routes
# ========================
@app.route("/")
def home():
    return render_template("index.html", feature_stats=feat_stats)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        raw_values = [float(request.form[feat]) for feat in FEATURE_NAMES]
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    try:
        input_scaled = preprocess_input(raw_values)

        # Prediction
        pred = int(model.predict(input_scaled)[0])
        prob = float(model.predict_proba(input_scaled)[0, 1])

        # SHAP for this specific input
        shap_vals  = explainer(input_scaled)
        # shap_vals.values shape: (1, n_features, 2) — take class 1 slice
        shap_class1 = shap_vals.values[0, :, 1].tolist()
        shap_dict = dict(sorted(
            ((feat, round(v, 5)) for feat, v in zip(FEATURE_NAMES, shap_class1)),
            key=lambda kv: abs(kv[1]),
            reverse=True
        ))

        return render_template(
            "result.html",
            prediction  = "Diabetic" if pred == 1 else "Non-Diabetic",
            probability = round(prob * 100, 2),
            shap_values = shap_dict,
            input_values= dict(zip(FEATURE_NAMES, raw_values)),
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)