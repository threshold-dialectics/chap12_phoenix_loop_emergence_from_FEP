import joblib
import numpy as np
import pandas as pd

from phoenix_loop_ML_classifier import create_ml_features, BURN_IN_PERIOD, FEATURE_WINDOW_SIZE

class PhoenixLoopMLClassifier:
    """Wrapper around the pre-trained Phoenix Loop RandomForest model."""

    def __init__(self, model_path='phoenix_rf_classifier.joblib', scaler_path='phoenix_feature_scaler.joblib'):
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.loaded = True
            print('INFO (ML Classifier): loaded pre-trained model and scaler')
        except FileNotFoundError:
            self.model = None
            self.scaler = None
            self.loaded = False
            print('WARNING (ML Classifier): pre-trained model or scaler not found')

    def predict_phases_for_run(self, diagnostics_dict):
        """Predict Phoenix Loop phase labels for a single run."""
        if not self.loaded:
            return np.full(len(diagnostics_dict.get('SpeedIndex', [])), 0, dtype=int)

        X_df, _, _ = create_ml_features([diagnostics_dict], FEATURE_WINDOW_SIZE)
        X_df = X_df.dropna()
        if X_df.empty:
            return np.full(len(diagnostics_dict.get('SpeedIndex', [])), 0, dtype=int)

        X_scaled = self.scaler.transform(X_df)
        y_pred = self.model.predict(X_scaled)

        start_offset = BURN_IN_PERIOD + FEATURE_WINDOW_SIZE - 1
        preds_full = np.full(len(diagnostics_dict.get('SpeedIndex', [])), 0, dtype=int)
        for idx, pred in zip(X_df.index, y_pred):
            time_idx = start_offset + idx
            if time_idx < len(preds_full):
                preds_full[time_idx] = pred
        return preds_full
