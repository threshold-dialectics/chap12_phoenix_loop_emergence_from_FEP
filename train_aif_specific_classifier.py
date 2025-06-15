# train_aif_specific_classifier.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
import os
import matplotlib.pyplot as plt
# Set consistent font sizes across all plots
plt.rcParams.update({'font.size': 16,
                     'xtick.labelsize': 12,
                     'ytick.labelsize': 12})
from scipy.signal import savgol_filter # For derivatives
from scipy.stats import linregress
from collections import Counter # For model entropy calculation (if needed directly here, though usually from history)
import json

# --- Configuration for Feature Engineering & Training ---
AIF_FEATURE_WINDOW_SIZE = 20  # Window size for mean/std stats
AIF_SMOOTHING_WINDOW = 15     # For calculating derivatives if needed for features
AIF_SMOOTHING_POLYORDER = 2
AIF_BURN_IN_PERIOD_FOR_FEATURES = 30 # Skip initial unstable diagnostic period for feature generation
                                     # Manual labels ideally start considering meaningful phases after this burn-in.
LONG_SLOPE_WINDOW = 25  # Window size for long-term slope and stability calculations

# Keys to calculate long-term slopes and stability metrics for
SLOPE_TARGET_KEYS = ['avg_beta_system', 'avg_fcrit_system', 'rhoE', 'avg_PE_system']
STABILITY_STD_TARGET_KEYS = ['avg_beta_system', 'avg_fcrit_system', 'avg_PE_system']

# Define base feature names to extract and process
# These should be keys present in your AIF simulation 'history' dictionaries
BASE_FEATURE_KEYS_FROM_AIF_HISTORY = [
    'avg_PE_system',
    'avg_beta_system',
    'avg_fcrit_system',
    'active_agents_count',
    'SpeedIndex',
    'CoupleIndex',
    'model_entropy_system',
    'rhoE'
]

# Optional: Define keys for which to calculate derivatives if they are not pre-calculated
# and you want to include them as base features before windowing.
DERIVATIVE_FEATURE_KEYS = [
    'avg_PE_system', 'avg_beta_system', 'avg_fcrit_system',
    'SpeedIndex', 'model_entropy_system', 'rhoE', 'active_agents_count'

]




# --- Helper Function: Generate Features from AIF History ---
def generate_features_from_aif_history(run_id_str, single_aif_history_dict, manual_labels_for_run):
    history = single_aif_history_dict
    num_steps_total = len(history['time'])
    
    X_run_features = []
    y_run_labels = []
    run_ids_for_samples = []

    processed_series = {}
    feature_names_generated_base = []

    for key in BASE_FEATURE_KEYS_FROM_AIF_HISTORY:
        if key in history and len(history[key]) == num_steps_total:
            series = np.array(history[key])
            processed_series[key] = np.nan_to_num(series) # Ensure NaNs are handled before processing
            feature_names_generated_base.append(key)
        else:
            print(f"Warning: Key {key} not found or length mismatch in history for {run_id_str}. Skipping this base feature.")
            continue # Skip if key is missing or has wrong length


    for key_to_derive in DERIVATIVE_FEATURE_KEYS: # Use your defined list
        if key_to_derive in processed_series:
            series = processed_series[key_to_derive]
            dot_key = f"dot_{key_to_derive}"
            if len(series) > AIF_SMOOTHING_WINDOW: # Ensure enough data for smoothing
                smoothed_arr = savgol_filter(series, AIF_SMOOTHING_WINDOW, AIF_SMOOTHING_POLYORDER, mode='mirror')
                processed_series[dot_key] = np.gradient(smoothed_arr)
            else:
                processed_series[dot_key] = np.zeros_like(series) # Fallback if too short

            if dot_key not in feature_names_generated_base: # Avoid duplicates if already there
                feature_names_generated_base.append(dot_key)
        # else:
            # print(f"Warning: Base key {key_to_derive} for derivative calculation not found.")

    # --- Long-term slope features ---
    for slope_key in SLOPE_TARGET_KEYS:
        if slope_key in processed_series:
            series = processed_series[slope_key]
            slope_vals = np.full(num_steps_total, np.nan, dtype=float)
            for idx in range(num_steps_total):
                if idx >= LONG_SLOPE_WINDOW - 1:
                    window = series[idx - LONG_SLOPE_WINDOW + 1 : idx + 1]
                    if np.any(~np.isnan(window)):
                        slope, _, _, _, _ = linregress(np.arange(LONG_SLOPE_WINDOW), window)
                        slope_vals[idx] = slope
                    else:
                        slope_vals[idx] = 0.0
            new_key = f"slope_{slope_key}"
            processed_series[new_key] = slope_vals
            feature_names_generated_base.append(new_key)

    # --- Stability (std dev) metrics ---
    for stab_key in STABILITY_STD_TARGET_KEYS:
        if stab_key in processed_series:
            series = processed_series[stab_key]
            std_vals = np.full(num_steps_total, np.nan, dtype=float)
            for idx in range(num_steps_total):
                if idx >= LONG_SLOPE_WINDOW - 1:
                    window = series[idx - LONG_SLOPE_WINDOW + 1 : idx + 1]
                    std_vals[idx] = np.nanstd(window)
            new_key = f"stability_std_{stab_key}"
            processed_series[new_key] = std_vals
            feature_names_generated_base.append(new_key)

    # --- Time since shock feature ---
    t_shock = history.get('t_shock', 150)
    time_since_shock = np.array([max(0, t - t_shock) for t in range(num_steps_total)])
    processed_series['time_since_shock'] = time_since_shock
    feature_names_generated_base.append('time_since_shock')

    start_idx_for_samples = AIF_BURN_IN_PERIOD_FOR_FEATURES + AIF_FEATURE_WINDOW_SIZE -1

    for t in range(start_idx_for_samples, num_steps_total):
        feature_vector_t = []
        valid_sample = True
        for base_feature_name in feature_names_generated_base:
            if base_feature_name not in processed_series: # Should not happen if logic above is correct
                feature_vector_t.extend([np.nan, np.nan, np.nan]) # Placeholder for missing base feature
                valid_sample = False
                continue

            series = processed_series[base_feature_name]
            current_value = series[t]
            window_data = series[t - AIF_FEATURE_WINDOW_SIZE + 1 : t + 1]
            
            if len(window_data) == AIF_FEATURE_WINDOW_SIZE:
                window_mean = np.nanmean(window_data) # Use nanmean in case NaNs from smoothing edges
                window_std = np.nanstd(window_data)   # Use nanstd
                feature_vector_t.extend([np.nan_to_num(current_value), np.nan_to_num(window_mean), np.nan_to_num(window_std)])
            else:
                feature_vector_t.extend([np.nan, np.nan, np.nan])
                valid_sample = False
        
        # Manual labels are 1,2,3,4. Classifier might prefer 0-indexed if not handled by class_weight or labels param.
        # For sklearn, it's generally fine to use labels like 1,2,3,4 directly.
        if valid_sample and t < len(manual_labels_for_run) and not np.isnan(manual_labels_for_run[t]) and manual_labels_for_run[t] != 0 : # Assuming 0 is not a valid phase label
            X_run_features.append(feature_vector_t)
            y_run_labels.append(manual_labels_for_run[t])
            run_ids_for_samples.append(run_id_str)

    df_column_names = []
    for name in feature_names_generated_base:
        df_column_names.append(name + "_t")
        df_column_names.append(name + "_win_mean")
        df_column_names.append(name + "_win_std")
        
    return pd.DataFrame(X_run_features, columns=df_column_names), np.array(y_run_labels), np.array(run_ids_for_samples)


# --- Main Training Logic ---
def train_aif_classifier():
    print("--- Starting AIF-Specific Classifier Training ---")

    # --- 1. Load Histories ---

    # --- Main Training Logic ---
def train_aif_classifier():
    print("--- Starting AIF-Specific Classifier Training ---")

    # --- 1. Load Manual Labels and Histories ---
    # (Your existing code for loading all_manual_labels_dict_10_runs and loaded_histories_dict)
    # ... (previous loading code as in your script) ...
    num_total_steps_sim = 800 
    manual_labels_run1 = np.zeros(num_total_steps_sim, dtype=int); manual_labels_run1[0:150] = 4; manual_labels_run1[150:186] = 1; manual_labels_run1[186:291] = 2; manual_labels_run1[291:551] = 3; manual_labels_run1[551:800] = 4
    manual_labels_run2 = np.zeros(num_total_steps_sim, dtype=int); manual_labels_run2[0:150] = 4; manual_labels_run2[150:191] = 1; manual_labels_run2[191:301] = 2; manual_labels_run2[301:561] = 3; manual_labels_run2[561:800] = 4
    manual_labels_run3 = np.zeros(num_total_steps_sim, dtype=int); manual_labels_run3[0:150] = 4; manual_labels_run3[150:181] = 1; manual_labels_run3[181:296] = 2; manual_labels_run3[296:541] = 3; manual_labels_run3[541:800] = 4
    manual_labels_run4 = np.zeros(num_total_steps_sim, dtype=int); manual_labels_run4[0:150] = 4; manual_labels_run4[150:191] = 1; manual_labels_run4[191:311] = 2; manual_labels_run4[311:571] = 3; manual_labels_run4[571:800] = 4
    manual_labels_run5 = np.zeros(num_total_steps_sim, dtype=int); manual_labels_run5[0:150] = 4; manual_labels_run5[150:186] = 1; manual_labels_run5[186:291] = 2; manual_labels_run5[291:551] = 3; manual_labels_run5[551:800] = 4
    manual_labels_run6 = np.zeros(num_total_steps_sim, dtype=int); manual_labels_run6[0:150] = 4; manual_labels_run6[150:186] = 1; manual_labels_run6[186:301] = 2; manual_labels_run6[301:561] = 3; manual_labels_run6[561:800] = 4
    manual_labels_run7 = np.zeros(num_total_steps_sim, dtype=int); manual_labels_run7[0:150] = 4; manual_labels_run7[150:191] = 1; manual_labels_run7[191:291] = 2; manual_labels_run7[291:551] = 3; manual_labels_run7[551:800] = 4
    manual_labels_run8 = np.zeros(num_total_steps_sim, dtype=int); manual_labels_run8[0:150] = 4; manual_labels_run8[150:181] = 1; manual_labels_run8[181:286] = 2; manual_labels_run8[286:541] = 3; manual_labels_run8[541:800] = 4
    manual_labels_run9 = np.zeros(num_total_steps_sim, dtype=int); manual_labels_run9[0:150] = 4; manual_labels_run9[150:191] = 1; manual_labels_run9[191:306] = 2; manual_labels_run9[306:566] = 3; manual_labels_run9[566:800] = 4
    manual_labels_run10 = np.zeros(num_total_steps_sim, dtype=int); manual_labels_run10[0:150] = 4; manual_labels_run10[150:186] = 1; manual_labels_run10[186:296] = 2; manual_labels_run10[296:556] = 3; manual_labels_run10[556:800] = 4
    
    all_manual_labels_dict_10_runs = {
        "run_1": manual_labels_run1, "run_2": manual_labels_run2, "run_3": manual_labels_run3,
        "run_4": manual_labels_run4, "run_5": manual_labels_run5, "run_6": manual_labels_run6,
        "run_7": manual_labels_run7, "run_8": manual_labels_run8, "run_9": manual_labels_run9,
        "run_10": manual_labels_run10,
    }
    
    loaded_histories_dict = {}
    history_dir = "results_aif_phoenix" 
    num_labeled_runs = 10 
    run_keys_to_load = [f"run_{i+1}" for i in range(num_labeled_runs)]

    for run_key in run_keys_to_load:
        history_file = os.path.join(history_dir, f"history_aif_phoenix_{run_key}.pkl")
        try:
            with open(history_file, 'rb') as f:
                loaded_histories_dict[run_key] = pickle.load(f)
            print(f"Successfully loaded history for {run_key} from {history_file}")
        except FileNotFoundError:
            print(f"ERROR: History file not found for {run_key} at {history_file}. Please ensure histories are saved correctly.")
            return
    
    if len(loaded_histories_dict) != num_labeled_runs:
        print(f"ERROR: Loaded {len(loaded_histories_dict)} histories, but expected {num_labeled_runs} based on manual labels. Exiting.")
        return

    # --- 2. Generate Features for All Labeled Runs ---
    all_X_list = []
    all_y_list = []
    all_groups_list = []

    for run_key, manual_labels in all_manual_labels_dict_10_runs.items():
        if run_key in loaded_histories_dict:
            history_data = loaded_histories_dict[run_key]
            X_df_run, y_run, groups_run = generate_features_from_aif_history(run_key, history_data, manual_labels)
            
            if not X_df_run.empty:
                all_X_list.append(X_df_run)
                all_y_list.append(y_run)
                all_groups_list.append(groups_run)
            else:
                print(f"Warning: No features generated for {run_key}")
        else:
            print(f"Warning: History data not found for labeled run {run_key}")

    if not all_X_list:
        print("ERROR: No features generated from any run. Exiting.")
        return

    X_aif_df = pd.concat(all_X_list, ignore_index=True)
    y_aif_array = np.concatenate(all_y_list)
    groups_aif_array = np.concatenate(all_groups_list)

    nan_rows_mask = X_aif_df.isnull().any(axis=1)
    if nan_rows_mask.any():
        print(f"Warning: Found {nan_rows_mask.sum()} rows with NaNs in features. Dropping them.")
        X_aif_df = X_aif_df[~nan_rows_mask].reset_index(drop=True)
        y_aif_array = y_aif_array[~nan_rows_mask]
        groups_aif_array = groups_aif_array[~nan_rows_mask]

    print(f"Total samples generated for AIF-specific training after NaN handling: {len(X_aif_df)}")
    if len(X_aif_df) == 0:
        print("ERROR: No samples after concatenation or NaN handling. Exiting.")
        return

    # --- 3. Train/Test Split (Run-Wise) ---
    unique_run_ids = np.unique(groups_aif_array)
    if len(unique_run_ids) < 2 :
        print("ERROR: Need at least 2 unique runs for train/test split. Exiting.")
        return
    
    test_fraction = 0.3 
    if int(len(unique_run_ids) * test_fraction) < 1: 
        test_fraction = 1 / len(unique_run_ids) if len(unique_run_ids) > 0 else 0.3 # ensure at least one test run
    
    if len(unique_run_ids) == 1: # Cannot split if only one run ID (group)
        print("Warning: Only one unique run ID. Using all data for training and testing (not ideal).")
        train_run_ids = unique_run_ids
        test_run_ids = unique_run_ids
    else:
        train_run_ids, test_run_ids = train_test_split(unique_run_ids, test_size=test_fraction, random_state=42, shuffle=True)

    train_mask = np.isin(groups_aif_array, train_run_ids)
    test_mask = np.isin(groups_aif_array, test_run_ids)

    X_train_df, X_test_df = X_aif_df[train_mask], X_aif_df[test_mask]
    y_train, y_test = y_aif_array[train_mask], y_aif_array[test_mask]
    
    train_groups_for_cv = groups_aif_array[train_mask]

    print(f"Training runs: {train_run_ids}, Test runs: {test_run_ids}")
    print(f"Training samples: {len(X_train_df)}, Test samples: {len(X_test_df)}")

    if len(X_train_df) == 0: # Test for empty training set specifically
        print("ERROR: Training set is empty after splitting. Check run splitting logic or number of unique runs. Exiting.")
        return
    if len(X_test_df) == 0 and len(unique_run_ids) > 1 : # Only error if we expected a test set
        print("ERROR: Test set is empty after splitting, but multiple unique runs were available. Check splitting logic. Exiting.")
        return


    # --- 4. Feature Scaling ---
    aif_scaler = StandardScaler()
    X_train_scaled = aif_scaler.fit_transform(X_train_df)
    if len(X_test_df) > 0: # Only scale test set if it's not empty
        X_test_scaled = aif_scaler.transform(X_test_df)
    else: # Handle empty test set case for predictions later
        X_test_scaled = np.array([]) # Empty array


    # --- 5. Model Training (Random Forest with GridSearchCV and LeaveOneGroupOut) ---
    print("Training AIF-Specific Random Forest Classifier...")
    param_grid_aif = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    best_aif_rf_model = None # Initialize to None

    # Determine CV strategy
    unique_train_groups = np.unique(train_groups_for_cv)
    if len(unique_train_groups) >= 2: # Need at least 2 groups for LOGO or k-fold
        logo = LeaveOneGroupOut()
        # Check if LOGO is feasible (number of splits will be number of unique groups)
        if logo.get_n_splits(groups=train_groups_for_cv) >= 2: # GridSearchCV typically needs at least 2 splits for CV
            cv_to_use_for_grid = logo
            print(f"Using LeaveOneGroupOut for GridSearchCV with {logo.get_n_splits(groups=train_groups_for_cv)} splits.")
        else: # Fallback to k-fold, e.g., 3-fold if enough samples, or less
            cv_to_use_for_grid = min(3, len(unique_train_groups)) # Ensure k <= n_groups
            if cv_to_use_for_grid < 2: cv_to_use_for_grid = None # No CV if still not possible
            print(f"Falling back to {cv_to_use_for_grid}-fold CV for GridSearchCV.")
    else:
        cv_to_use_for_grid = None # Not enough groups for any meaningful CV
        print("Warning: Not enough unique runs in training set for cross-validation during GridSearchCV. Fitting on whole training set.")

    if cv_to_use_for_grid is not None:
        grid_search_aif = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=8), # class_weight moved to param_grid
            param_grid_aif, 
            cv=cv_to_use_for_grid, 
            scoring='f1_macro', 
            verbose=1,
            error_score='raise'
        )
        if isinstance(cv_to_use_for_grid, LeaveOneGroupOut):
            grid_search_aif.fit(X_train_scaled, y_train, groups=train_groups_for_cv)
        else: # Integer k-fold
            grid_search_aif.fit(X_train_scaled, y_train)
        best_aif_rf_model = grid_search_aif.best_estimator_
        print("Best AIF-Specific RF hyperparameters from GridSearchCV:", grid_search_aif.best_params_)
    else: # No CV performed, fit a default model
        print("Fitting a default RandomForestClassifier as no CV was performed.")
        best_aif_rf_model = RandomForestClassifier(
            random_state=42, class_weight='balanced', n_jobs=8,
            n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=2 # Example good defaults
        )
        best_aif_rf_model.fit(X_train_scaled, y_train)
        print("Best AIF-Specific RF hyperparameters: Used default parameters.")


    # --- 6. Evaluation ---
    if len(X_test_df) > 0 and best_aif_rf_model is not None: # Ensure there's a test set and model
        y_pred_aif = best_aif_rf_model.predict(X_test_scaled)
        
        unique_labels_in_test_and_pred = np.unique(np.concatenate((y_test, y_pred_aif)))
        phase_names_map = {1: "P1", 2: "P2", 3: "P3", 4: "P4"}
        report_labels = sorted([l for l in unique_labels_in_test_and_pred if l in phase_names_map])
        
        if not report_labels:
            print("Warning: No valid phase labels (1-4) found in test/predictions for report.")
            report_dict = classification_report(y_test, y_pred_aif, output_dict=True, zero_division=0)
        else:
            report_target_names = [phase_names_map[l] for l in report_labels]
            print("\n--- AIF-Specific Classification Report (Test Set) ---")
            print(classification_report(y_test, y_pred_aif, labels=report_labels, target_names=report_target_names, zero_division=0))
            report_dict = classification_report(y_test, y_pred_aif, labels=report_labels, target_names=report_target_names, zero_division=0, output_dict=True)

        accuracy_val = accuracy_score(y_test, y_pred_aif)
        print(f"Overall Accuracy: {accuracy_val:.4f}")
        
        cm_display_labels_numeric = sorted(phase_names_map.keys()) 
        cm_display_labels_text = [phase_names_map[l] for l in cm_display_labels_numeric]

        cm_aif = confusion_matrix(y_test, y_pred_aif, labels=cm_display_labels_numeric)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_aif, display_labels=cm_display_labels_text)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix (AIF-Specific Model - Test Set)")
        
        results_dir = "results_aif_phoenix"
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, "aif_specific_classifier_CM.png"), dpi=350)
        print(f"Saved Confusion Matrix to {os.path.join(results_dir, 'aif_specific_classifier_CM.png')}")
        plt.close() # Close the plot

        # --- Save classifier metrics for later analysis ---
        classifier_summary = {
            "accuracy": float(accuracy_val),
            "macro_f1": float(report_dict.get("macro avg", {}).get("f1-score", float('nan'))),
            "per_class": {
                phase_names_map[l]: {
                    "P": report_dict.get(phase_names_map[l], {}).get("precision", float('nan')),
                    "R": report_dict.get(phase_names_map[l], {}).get("recall", float('nan')),
                    "F1": report_dict.get(phase_names_map[l], {}).get("f1-score", float('nan')),
                } for l in cm_display_labels_numeric
            },
            "confusion": cm_aif.tolist()
        }

        with open(os.path.join(results_dir, "classifier_metrics.json"), "w") as f:
            json.dump(classifier_summary, f, indent=2)
        print(f"Saved classifier metrics to {os.path.join(results_dir, 'classifier_metrics.json')}")
    elif best_aif_rf_model is None:
        print("ERROR: best_aif_rf_model was not trained. Skipping evaluation.")
    else:
        print("Test set is empty. Skipping evaluation on test set.")


    # --- 7. Save Model and Scaler (if model was trained) ---
    if best_aif_rf_model is not None:
        model_save_path = "aif_phoenix_classifier_specific.joblib"
        scaler_save_path = "aif_phoenix_scaler_specific.joblib"
        joblib.dump(best_aif_rf_model, model_save_path)
        joblib.dump(aif_scaler, scaler_save_path)
        print(f"Trained AIF-specific model saved to: {model_save_path}")
        print(f"AIF-specific scaler saved to: {scaler_save_path}")
    else:
        print("Model was not trained, so not saving.")


if __name__ == "__main__":
    train_aif_classifier()
