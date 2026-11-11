# DiabetesXAI 🧠

> A feedforward neural network built **from scratch using pure NumPy** — no TensorFlow, no PyTorch — trained on real clinical data to predict diabetes risk, with SHAP-based explainability and an interactive Flask web interface.

---

## Why This Project Exists

Most ML projects use high-level frameworks that hide what actually happens inside a model. This project builds every component manually — forward pass, backpropagation, gradient descent — so every line of code maps directly to the math. SHAP explainability is layered on top to show *why* the model makes each prediction, which is critical for any healthcare application.

---

## Project Structure

```
DiabetesXAI/
│
├── app.py                   # Flask backend
├── neural_network.py        # NumPy neural network (core engine)
├── training.py              # Training pipeline
├── data_preprocessing.py    # Cleaning, normalization, SMOTE
├── shap_analysis.py         # SHAP explainability
├── utils.py                 # Shared helpers + input validation
│
├── model/                   # Generated after training (gitignored)
│   ├── weights.pkl          # Saved model object
│   ├── scaler.pkl           # Scaler + medians + IQR bounds
│   ├── shap_values.pkl      # Precomputed SHAP values
│   ├── loss_curve.png       # Training vs validation loss plot
│   ├── shap_summary.png     # SHAP beeswarm summary plot
│   └── shap_bar.png         # SHAP mean |value| bar chart
│
├── dataset/
│   └── diabetes.csv         # Pima Indians Diabetes dataset
│
├── templates/
│   ├── index.html           # Input form
│   └── result.html          # Prediction + SHAP bar chart
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

**Pima Indians Diabetes Database**
- 768 patients, 8 clinical features, 1 binary target (0 = No diabetes, 1 = Diabetes)
- Source: [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Place the CSV at `dataset/diabetes.csv`

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration (2-hr oral glucose test) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-hour serum insulin (μU/mL) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Diabetes likelihood based on family history |
| Age | Age in years |

---

## Installation

```bash
git clone https://github.com/yourusername/DiabetesXAI.git
cd DiabetesXAI
pip install -r requirements.txt
```

---

## Run Order

```bash
# Step 1 — Train the model
python training.py

# Step 2 — Run SHAP analysis (generates plots)
python shap_analysis.py

# Step 3 — Start the web app
python app.py
```

Then open `http://localhost:5000` in your browser.

---

## Module Details

### `neural_network.py` — Core Engine

The entire neural network implemented in pure NumPy. No ML framework touches the model.

**Architecture**
```
Input (8) → Hidden (16, Leaky ReLU) → Hidden (8, Leaky ReLU) → Output (1, Sigmoid)
```

**Loss:** Binary Cross-Entropy + L2 regularization

| Feature | Detail |
|---|---|
| He initialization | `√(2/fan_in)` — prevents vanishing gradients with ReLU activations |
| Leaky ReLU | slope=0.01 for Z<0 — avoids dead neurons from plain ReLU |
| L2 regularization | Applied to loss and backprop gradients; controls overfitting |
| Mini-batch SGD | Shuffled batches per epoch; `batch_size=32` default |
| Learning rate decay | `lr / (1 + decay × epoch)` — fast start, fine-tune finish |
| Gradient clipping | Caps gradient norm to 5.0 — prevents NaN loss on imbalanced data |
| Early stopping | Monitors val loss; restores best weights when patience runs out |
| Gradient check | Finite-difference vs analytical gradients; relative error < 1e-5 confirms backprop is correct |
| `evaluate()` | Returns Accuracy, Precision, Recall, F1, Specificity — pure NumPy |
| `save()` / `load()` | Persists trained model via pickle |

**Key usage**
```python
model = NeuralNetwork(input_size=8)

# Verify backprop before training (run on small sample)
model.gradient_check(X_train[:20], y_train[:20])

# Train with early stopping
train_losses, val_losses = model.train(
    X_train, y_train,
    X_val, y_val,
    epochs=300, lr=0.01, batch_size=32,
    lr_decay=0.001, patience=25
)

# Evaluate
model.evaluate(X_test, y_test)

# Save / load
model.save("model/weights.pkl")
model = NeuralNetwork.load("model/weights.pkl")

# Predict
labels = model.predict(X_test)             # shape (m,), values 0 or 1
probs  = model.predict_proba(X_test)       # shape (m, 2), columns: [P(0), P(1)]
```

---

### `data_preprocessing.py` — Data Pipeline

**Pipeline order** (order matters — changing it causes data leakage)

```
Load CSV → validate columns → stratified train/test split →
clean zeros (train medians only) → validation split →
IQR outlier capping (train bounds) → z-score normalize (train stats) →
SMOTE (train only) → sanity check → save scaler
```

| Feature | Detail |
|---|---|
| Stratified split | Preserves class ratio (65/35) in train and test |
| Zero imputation | Zeros in Glucose/BP/Insulin/BMI/SkinThickness are medians, not real — replaced with **train-only** medians |
| IQR outlier capping | Insulin reaches 846 in raw data; IQR bounds computed on train, applied to all splits |
| Data leakage guard | Medians, IQR bounds, and scaler mean/std all computed on train indices only |
| Validation split | 10% of total held out for early stopping; test set untouched until final eval |
| SMOTE | Synthetic minority oversampling on train only; test set keeps real distribution |
| `sanity_check()` | Asserts: no NaN, mean≈0 and std≈1 post-normalization, feature count=8 |
| `get_feature_stats()` | Min/max/mean/median per feature (used by frontend for input hints) |
| `save_scaler()` / `load_scaler()` | Persists scaler + medians + IQR bounds — Flask uses these for live inference |

**Scaler payload saved to `model/scaler.pkl`**
```python
{
    "scaler":        StandardScaler,   # fit on train only
    "medians":       dict,             # per-feature train median
    "iqr_bounds":    dict,             # per-feature (lo, hi) IQR bounds
    "feature_stats": dict,             # min/max/mean/median for frontend
    "features":      list              # FEATURE_NAMES in order
}
```

---

### `shap_analysis.py` — Explainability

Uses [SHAP](https://shap.readthedocs.io/) (SHapley Additive exPlanations) to explain individual predictions.

- **Background data:** 100-sample subsample of real training data — not random noise, which would produce meaningless attributions
- **Output:** Per-feature SHAP values for each test sample, class-1 slice (diabetic probability)
- **Plots saved:** `shap_summary.png` (beeswarm), `shap_bar.png` (mean |SHAP|)

---

### `app.py` — Flask Web App

- Loads model and scaler on startup (no retraining per request)
- Applies the same preprocessing pipeline to live user input: zero imputation → IQR clipping → z-score scaling
- Runs SHAP on each prediction using real training data as background
- Returns prediction, probability, and per-feature SHAP contributions to `result.html`

---

### `utils.py` — Helpers

| Function | Purpose |
|---|---|
| `validate_input(values)` | Checks user inputs against physiological ranges; returns warning strings |
| `print_metrics(metrics)` | Pretty-prints a metrics dict from `evaluate()` |
| `sigmoid(z)` | Numerically stable sigmoid (clip to ±500) |

---

## Target Metrics

| Metric | Target |
|---|---|
| Test Accuracy | ≥ 80% |
| F1 Score | ≥ 0.76 |
| Precision | ≥ 0.73 |
| Recall | ≥ 0.73 |

---

## Key Design Decisions

**Why NumPy only for the neural network?**
Frameworks like Keras abstract away backpropagation. Implementing it manually means every weight update, every gradient, every activation function maps directly to the underlying calculus. The gradient check function verifies this is mathematically correct before training begins.

**Why SHAP for explainability?**
In healthcare, a prediction without explanation is not useful. SHAP (SHapley Additive exPlanations) assigns each feature a contribution value for each individual prediction — grounded in game theory. It answers: *"For this specific patient, how much did Glucose vs BMI vs Age contribute to the prediction?"*

**Why IQR capping before normalization?**
Z-score normalization is sensitive to outliers. A single extreme value (Insulin=846) compresses the rest of the distribution. Capping first ensures normalization works as intended.

**Why SMOTE only on training data?**
Applying SMOTE before splitting means synthetic samples derived from the same original point can appear in both train and test — making test evaluation unreliable. Applied to train only, the test set reflects real-world class distribution.

**Why real background data for SHAP (not random)?**
SHAP Shapley values represent the marginal contribution of each feature relative to a baseline. If that baseline is random noise instead of the actual data distribution, the attributions have no meaningful interpretation.

---

## Interview Talking Points

- *"Why not Keras?"* — Wanted to implement and understand backpropagation at the mathematical level. Used a gradient check to verify the analytical gradients match finite-difference numerical gradients before training.
- *"What is SHAP and why does it matter in healthcare?"* — Model trust and explainability; clinicians need to understand why a model flagged a patient, not just that it did.
- *"How did you handle class imbalance?"* — SMOTE on train only; test set kept at real distribution so evaluation reflects real-world performance.
- *"What is data leakage and where could it occur here?"* — Median imputation, IQR bounds, and normalization statistics must be computed on training data only. Computing them on the full dataset before splitting leaks test distribution into training.
- *"How did you verify your backpropagation?"* — Numerical gradient check: finite differences vs analytical gradients; relative error < 1e-5 confirms correctness.
- *"Why IQR capping before z-score normalization?"* — Insulin has extreme outliers (up to 846). A single outlier at ~10σ dominates gradient updates. Capping first keeps the normalization meaningful.
