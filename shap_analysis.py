import shap
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt

from data_preprocessing import DataPreprocessor, FEATURE_NAMES


if __name__ == "__main__":

    # ========================
    # 1. Load Data
    # ========================
    dp = DataPreprocessor("dataset/diabetes.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = dp.preprocess()

    # ========================
    # 2. Load Model
    # ========================
    with open("model/weights.pkl", "rb") as f:
        model = pickle.load(f)

    # ========================
    # 3. SHAP Explainer
    #    Background = real training data (subsample for speed).
    #    Using random noise as background produces meaningless attributions.
    # ========================
    rng         = np.random.default_rng(42)
    n_bg        = min(100, len(X_train))
    bg_idx      = rng.choice(len(X_train), size=n_bg, replace=False)
    background  = X_train[bg_idx]

    # Wrap model so SHAP gets (m, 2) output — P(class=0), P(class=1)
    def model_fn(X):
        return model.predict_proba(X)

    print("Building SHAP explainer …")
    explainer   = shap.Explainer(model_fn, background)
    shap_values = explainer(X_test)

    # ========================
    # 4. Save SHAP values
    # ========================
    with open("model/shap_values.pkl", "wb") as f:
        pickle.dump(shap_values, f)
    print("SHAP values saved → model/shap_values.pkl")

    # ========================
    # 5. Summary Plot  (class 1 = diabetic)
    # ========================
    shap.summary_plot(
        shap_values[:, :, 1],   # slice for class 1
        X_test,
        feature_names=FEATURE_NAMES,
        show=False
    )
    plt.tight_layout()
    plt.savefig("model/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("SHAP summary plot saved → model/shap_summary.png")

    # ========================
    # 6. Bar Plot (mean |SHAP|)
    # ========================
    shap.summary_plot(
        shap_values[:, :, 1],
        X_test,
        feature_names=FEATURE_NAMES,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig("model/shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("SHAP bar plot saved → model/shap_bar.png")

    print("\nSHAP analysis complete.")
