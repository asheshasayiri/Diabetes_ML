import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# Try importing SMOTE; fall back gracefully if not installed
# ------------------------------------------------------------------
try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False
    print("⚠️  imbalanced-learn not found — SMOTE will be skipped. "
          "Run: pip install imbalanced-learn")

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]

# Columns where 0 is physiologically impossible → treat as missing
ZERO_IMPUTE_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


class DataPreprocessor:
    def __init__(self, filepath="dataset/diabetes.csv"):
        self.filepath = filepath
        self.scaler   = StandardScaler()

        # Stats computed on train only (used by Flask for live inference)
        self.train_medians  = {}
        self.iqr_bounds     = {}
        self.feature_stats  = {}

    # ========================
    # LOAD
    # ========================
    def load_data(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"Dataset not found at '{self.filepath}'.\n"
                "Download from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database"
            )
        df = pd.read_csv(self.filepath)

        required = FEATURE_NAMES + ["Outcome"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        return df

    # ========================
    # CLEAN  (train-safe)
    # ========================
    def clean_data(self, df, train_idx):
        """
        Replace 0s with TRAIN medians only — no leakage.
        train_idx: integer index array for training rows.
        """
        df = df.copy()
        for col in ZERO_IMPUTE_COLS:
            train_median = df.loc[train_idx, col].replace(0, np.nan).median()
            self.train_medians[col] = train_median
            df[col] = df[col].replace(0, np.nan).fillna(train_median)
        return df

    # ========================
    # IQR OUTLIER CAPPING  (train-safe)
    # ========================
    def cap_outliers(self, X_train, X_val, X_test, feature_names):
        """
        Cap extreme values using IQR bounds computed on X_train only.
        Returns DataFrames with the same columns.
        """
        X_train = pd.DataFrame(X_train, columns=feature_names)
        X_val   = pd.DataFrame(X_val,   columns=feature_names)
        X_test  = pd.DataFrame(X_test,  columns=feature_names)

        for col in feature_names:
            Q1  = X_train[col].quantile(0.25)
            Q3  = X_train[col].quantile(0.75)
            IQR = Q3 - Q1
            lo  = Q1 - 1.5 * IQR
            hi  = Q3 + 1.5 * IQR
            self.iqr_bounds[col] = (lo, hi)

            for df in [X_train, X_val, X_test]:
                df[col] = df[col].clip(lo, hi)

        return X_train.values, X_val.values, X_test.values

    # ========================
    # SCALE  (train-safe)
    # ========================
    def scale_data(self, X_train, X_val, X_test):
        X_train = self.scaler.fit_transform(X_train)
        X_val   = self.scaler.transform(X_val)
        X_test  = self.scaler.transform(X_test)
        return X_train, X_val, X_test

    # ========================
    # FEATURE STATS
    # ========================
    def compute_feature_stats(self, df):
        """Min / max / mean / median per feature from the full dataset.
        Used by the frontend to set slider ranges."""
        stats = {}
        for col in FEATURE_NAMES:
            stats[col] = {
                "min":    float(df[col].min()),
                "max":    float(df[col].max()),
                "mean":   float(df[col].mean()),
                "median": float(df[col].median()),
            }
        self.feature_stats = stats
        return stats

    # ========================
    # SANITY CHECK
    # ========================
    def sanity_check(self, X_train, X_val, X_test, y_train, y_test):
        assert not np.isnan(X_train).any(), "NaN in X_train!"
        assert not np.isnan(X_val).any(),   "NaN in X_val!"
        assert not np.isnan(X_test).any(),  "NaN in X_test!"

        train_mean = X_train.mean()
        train_std  = X_train.std()
        assert abs(train_mean) < 0.1,      f"X_train mean far from 0: {train_mean:.4f}"
        assert abs(train_std  - 1.0) < 0.1, f"X_train std far from 1: {train_std:.4f}"

        assert X_train.shape[1] == 8, f"Expected 8 features, got {X_train.shape[1]}"

        print("✅ Sanity check passed.")

    # ========================
    # MAIN PIPELINE
    # ========================
    def preprocess(self, smote=True):
        """
        Full pipeline:
          Load → validate → get train indices →
          clean zeros (train medians) → stratified split →
          IQR capping (train bounds) → z-score (train stats) →
          SMOTE (train only) → sanity check → save scaler
        """

        # 1. Load
        df = self.load_data()

        # 2. Feature stats (before any transforms — for frontend sliders)
        self.compute_feature_stats(df)

        # 3. Train / test split indices FIRST (so cleaning uses train medians only)
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["Outcome"]
        )
        train_idx = train_df.index

        # 4. Clean zeros using train medians
        df_clean = self.clean_data(df, train_idx)

        # 5. Rebuild train/test from cleaned df
        train_df = df_clean.loc[train_df.index]
        test_df  = df_clean.loc[test_df.index]

        # 6. Further split train → train + val
        X_tr = train_df[FEATURE_NAMES]
        y_tr = train_df["Outcome"]
        X_te = test_df[FEATURE_NAMES]
        y_te = test_df["Outcome"]

        X_train, X_val, y_train, y_val = train_test_split(
            X_tr, y_tr, test_size=0.125, random_state=42, stratify=y_tr
        )  # 0.125 of 0.8 ≈ 10% of total

        # 7. IQR capping (train bounds only)
        X_train_arr, X_val_arr, X_test_arr = self.cap_outliers(
            X_train.values, X_val.values, X_te.values, FEATURE_NAMES
        )

        # 8. Z-score normalization (train mean/std only)
        X_train_s, X_val_s, X_test_s = self.scale_data(
            X_train_arr, X_val_arr, X_test_arr
        )

        y_train = y_train.values.reshape(-1)
        y_val   = y_val.values.reshape(-1)
        y_test  = y_te.values.reshape(-1)

        # 9. SMOTE on train only
        if smote and _SMOTE_AVAILABLE:
            sm = SMOTE(random_state=42)
            X_train_s, y_train = sm.fit_resample(X_train_s, y_train)
            print(f"SMOTE applied → train size: {len(y_train)} "
                  f"(pos: {y_train.sum()}, neg: {(1-y_train).sum()})")
        else:
            if smote and not _SMOTE_AVAILABLE:
                print("Skipping SMOTE (imbalanced-learn not installed).")

        # 10. Sanity check
        self.sanity_check(X_train_s, X_val_s, X_test_s, y_train, y_test)

        # 11. Save scaler + metadata
        self.save_scaler("model/scaler.pkl")

        return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test

    # ========================
    # SAVE / LOAD SCALER
    # ========================
    def save_scaler(self, path="model/scaler.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "scaler":        self.scaler,
            "medians":       self.train_medians,
            "iqr_bounds":    self.iqr_bounds,
            "feature_stats": self.feature_stats,
            "features":      FEATURE_NAMES,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"Scaler saved → {path}")

    @staticmethod
    def load_scaler(path="model/scaler.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_feature_stats(self):
        """Return min/max/mean/median per feature (for frontend slider ranges)."""
        if not self.feature_stats:
            raise RuntimeError("Call preprocess() first.")
        return self.feature_stats
