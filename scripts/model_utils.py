"""Shared model classes — imported by both 05_train_model.py and predict_match.py
so that joblib can deserialize the LGBMPipeline from either context."""
import numpy as np


class _FeatureNames:
    """Minimal scaler stand-in that exposes feature_names_in_ for predict_match.py."""
    def __init__(self, features):
        self.feature_names_in_ = list(features)


class LGBMPipeline:
    """Wraps any calibrated classifier (LightGBM or LogReg) with optional StandardScaler.

    The scaler kwarg allows LogisticRegression models to scale inputs at inference
    without changing the joblib-serialised class name (backward-compatible).
    """

    def __init__(self, calibrated_model, features, scaler=None):
        self.calibrated_model = calibrated_model
        self.features = list(features)
        self.scaler = scaler
        self.named_steps = {"scaler": _FeatureNames(self.features)}

    def predict_proba(self, X):
        if hasattr(X, "columns"):
            arr = X[self.features].values
        else:
            arr = np.asarray(X)
        if self.scaler is not None:
            arr = self.scaler.transform(arr)
        return self.calibrated_model.predict_proba(arr)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
