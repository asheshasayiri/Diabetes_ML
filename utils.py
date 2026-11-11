"""
utils.py — Shared helper functions for DiabetesXAI.
"""

import numpy as np


FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]

# Typical physiological ranges for basic input validation
FEATURE_RANGES = {
    "Pregnancies":              (0,   20),
    "Glucose":                  (0,  300),
    "BloodPressure":            (0,  200),
    "SkinThickness":            (0,  100),
    "Insulin":                  (0,  900),
    "BMI":                      (0,   70),
    "DiabetesPedigreeFunction": (0,    3),
    "Age":                      (1,  120),
}


def validate_input(values: dict) -> list:
    """
    Validate a dict of {feature_name: value} against physiological ranges.
    Returns a list of warning strings (empty = all OK).
    """
    warnings = []
    for feat, (lo, hi) in FEATURE_RANGES.items():
        v = values.get(feat)
        if v is None:
            warnings.append(f"Missing value for '{feat}'.")
        elif not (lo <= float(v) <= hi):
            warnings.append(
                f"'{feat}' = {v} is outside expected range [{lo}, {hi}]."
            )
    return warnings


def print_metrics(metrics: dict, label: str = ""):
    """Pretty-print a metrics dict returned by NeuralNetwork.evaluate()."""
    header = f"  {label}" if label else "  Metrics"
    print(header)
    print(f"    Accuracy    : {metrics['accuracy']:.4f}")
    print(f"    Precision   : {metrics['precision']:.4f}")
    print(f"    Recall      : {metrics['recall']:.4f}")
    print(f"    F1 Score    : {metrics['f1']:.4f}")
    print(f"    Specificity : {metrics['specificity']:.4f}")


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))
