# phoenix_loop_classifier_accuracy_ML.py

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from collections import deque
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib # For saving the model

# --- Configuration Parameters ---
# These serve as local defaults if no external config is imported,
# or they might be overridden by an explicitly passed env_config.

_G_INITIAL_DEFAULT = 1.0
_BETA_INITIAL_DEFAULT = 0.8
_FCRIT_INITIAL_DEFAULT = 100.0
_W1_DEFAULT, _W2_DEFAULT, _W3_DEFAULT = 0.33, 0.33, 0.34
_C_TOLERANCE_DEFAULT = 1.0
_FCRIT_FLOOR_DEFAULT = 10.0
_FCRIT_REPLENISH_RATE_DEFAULT = 0.1
_BETA_DECAY_FLARING_DEFAULT = 0.05
_BETA_REBUILD_PRUNING_DEFAULT = 0.02
_BETA_PHASE_IV_NOISE_STD_DEFAULT = 0.002
_BASELINE_FCRIT_OPERATIONAL_COST_DEFAULT = 0.2
_FCRIT_HIGH_LEVEL_DECAY_RATE_DEFAULT = 0.0

_STRAIN_BASELINE_DEFAULT = 3.0
_STRAIN_SHOCK_MAGNITUDE_DEFAULT = 5.0
_STRAIN_SHOCK_DURATION_DEFAULT = 20
_STRAIN_SHOCK_INTERVAL_DEFAULT = 150
_POST_COLLAPSE_STRAIN_REDUCTION_DEFAULT = 0.7

_SIM_STEPS_DEFAULT_INTERNAL = 200 # Default length for standalone ML training

SMOOTHING_WINDOW_DEFAULT = 15
SMOOTHING_POLY_DEFAULT = 2
ROLLING_WINDOW_COUPLE = 30
ROLLING_WINDOW_ENTROPY = 20

_SHOCK_START_TIME_DEFAULT = 50
_N_SIMULATION_RUNS_DEFAULT = 50 # Used for standalone ML training's baseline calculation
BURN_IN_PERIOD = 30
FEATURE_WINDOW_SIZE = 10


# --- Helper Functions (Unchanged) ---
def smooth_series(series, window=SMOOTHING_WINDOW_DEFAULT, poly=SMOOTHING_POLY_DEFAULT):
    series = np.asarray(series)
    if len(series) < window:
        if len(series) >=3 :
            current_poly = min(poly, len(series)-1)
            current_window = len(series) if len(series) % 2 != 0 else len(series) -1
            if current_window <= current_poly:
                current_window = current_poly + 1 if (current_poly+1) % 2 != 0 else current_poly + 2
            if current_window > len(series): current_window = len(series) if len(series)%2!=0 else len(series)-1
            if current_window > current_poly and current_window >=3 and current_window <= len(series):
                 return savgol_filter(series, current_window, current_poly)
        return series
    return savgol_filter(series, window, poly)

def calculate_derivative(series, window=SMOOTHING_WINDOW_DEFAULT, poly=SMOOTHING_POLY_DEFAULT, dt=1):
    series = np.asarray(series)
    smoothed = smooth_series(series, window, poly)
    if len(smoothed) < 2:
        return np.zeros_like(smoothed)
    return np.gradient(smoothed, dt)

def add_true_phase_background(ax, time_history, true_phase_history):
    colors = ['white', 'lightcoral', 'lightsalmon', 'lightgreen', 'lightblue']
    min_len = min(len(time_history), len(true_phase_history))
    for i in range(min_len -1):
        phase_val = true_phase_history[i]
        color_idx = 0
        if isinstance(phase_val, (int, np.integer)):
            if 0 <= phase_val < len(colors): color_idx = phase_val
            elif phase_val >= len(colors): color_idx = phase_val % len(colors)
        ax.axvspan(time_history[i], time_history[i+1],
                   facecolor=colors[color_idx], alpha=0.3, edgecolor=None, zorder=-10)


class TDSystem:
    def __init__(self, run_id=0, sim_steps_override=None, env_config=None):
        self.run_id = run_id

        if env_config:
            self.config = env_config
            # print(f"DEBUG TDSystem (ML Script): Using EXPLICITLY PASSED env_config: {self.config.level if hasattr(self.config, 'level') else 'N/A'}")
        else:
            # Fallback to local defaults if no env_config is passed
            # This path is primarily for when TDSystem is instantiated directly
            # from within this ML script's __main__ block for training.
            print("WARNING (ML Script - TDSystem): env_config not provided. Using local defaults. Expected for standalone ML training.")
            class LocalConfig:
                fcrit_initial = _FCRIT_INITIAL_DEFAULT
                fcrit_floor = _FCRIT_FLOOR_DEFAULT
                fcrit_replenish_rate = _FCRIT_REPLENISH_RATE_DEFAULT
                baseline_fcrit_operational_cost = _BASELINE_FCRIT_OPERATIONAL_COST_DEFAULT
                fcrit_high_level_decay_rate = _FCRIT_HIGH_LEVEL_DECAY_RATE_DEFAULT
                beta_decay_flaring = _BETA_DECAY_FLARING_DEFAULT
                beta_rebuild_pruning = _BETA_REBUILD_PRUNING_DEFAULT
                beta_phase_iv_noise_std = _BETA_PHASE_IV_NOISE_STD_DEFAULT
                strain_baseline = _STRAIN_BASELINE_DEFAULT
                strain_shock_magnitude = _STRAIN_SHOCK_MAGNITUDE_DEFAULT
                strain_shock_duration = _STRAIN_SHOCK_DURATION_DEFAULT
                strain_shock_interval = _STRAIN_SHOCK_INTERVAL_DEFAULT
                post_collapse_strain_reduction = _POST_COLLAPSE_STRAIN_REDUCTION_DEFAULT
                sim_steps = _SIM_STEPS_DEFAULT_INTERNAL
                shock_start_time = _SHOCK_START_TIME_DEFAULT
                n_intervention_runs = _N_SIMULATION_RUNS_DEFAULT # For standalone baseline calc
                level = "local_ml_default"
            self.config = LocalConfig()

        self.g_initial = _G_INITIAL_DEFAULT
        self.beta_initial = _BETA_INITIAL_DEFAULT
        self.w1, self.w2, self.w3 = _W1_DEFAULT, _W2_DEFAULT, _W3_DEFAULT
        self.c_tolerance = _C_TOLERANCE_DEFAULT

        self.g_lever = self.g_initial
        self.beta_lever = self.beta_initial
        self.fcrit = self.config.fcrit_initial
        self.strain = self.config.strain_baseline
        
        self.theta_t = self._calculate_theta_t()
        self.safety_margin = self.theta_t - self.strain
        self.time = 0
        self.collapsed = False
        self.true_phase = 4
        self.time_in_phase = 0
        
        self._shock_active_until = -1
        
        self._next_shock_time = self.config.shock_start_time + np.random.randint(
            -int(self.config.shock_start_time*0.1), 
            int(self.config.shock_start_time*0.1)+1 if self.config.shock_start_time > 0 else 0
        )
        
        if sim_steps_override is not None:
            self.sim_steps_limit = sim_steps_override
        else:
             self.sim_steps_limit = self.config.sim_steps

        self.history = {
            'run_id': [], 'time': [], 'g_lever': [], 'beta_lever': [], 'fcrit': [],
            'strain': [], 'theta_t': [], 'safety_margin': [],
            'true_phase': [], 'collapsed_flag': []
        }
        self._record_history()

    def _calculate_theta_t(self):
        return self.c_tolerance * (self.g_lever**self.w1) * \
               (self.beta_lever**self.w2) * (np.clip(self.fcrit, 1e-9, None)**self.w3)

    def _update_strain(self):
        base_strain_val = self.config.strain_baseline
        if self.collapsed:
            base_strain_val *= self.config.post_collapse_strain_reduction
        current_strain_val = base_strain_val
        if self.time >= self._next_shock_time and \
           self.time < self._next_shock_time + self.config.strain_shock_duration:
            current_strain_val += self.config.strain_shock_magnitude
            if self.time == self._next_shock_time:
                 self._shock_active_until = self._next_shock_time + self.config.strain_shock_duration
        elif self._shock_active_until != -1 and self.time == self._shock_active_until:
             self._next_shock_time = self.time + self.config.strain_shock_interval + \
                                     np.random.randint(-int(self.config.strain_shock_interval*0.1), 
                                                       int(self.config.strain_shock_interval*0.1)+1 if self.config.strain_shock_interval > 0 else 0)
             self._shock_active_until = -1
        self.strain = max(0.1, current_strain_val + np.random.normal(0, 0.05 * base_strain_val))

    def _update_levers_and_fcrit(self):
        cost_g = 0.05 * (self.g_lever**0.5)
        cost_beta = 0.02 * (self.beta_lever**1.2)
        self.fcrit -= (self.config.baseline_fcrit_operational_cost + cost_g + cost_beta)
        if self.fcrit > self.config.fcrit_initial * 1.2 and self.config.fcrit_high_level_decay_rate > 0:
            self.fcrit -= self.fcrit * self.config.fcrit_high_level_decay_rate
        if not self.collapsed or (self.collapsed and self.true_phase >= 3):
            replenish_target = self.config.fcrit_initial
            self.fcrit += self.config.fcrit_replenish_rate * (replenish_target - self.fcrit)
            self.fcrit = min(self.fcrit, self.config.fcrit_initial * 1.5)
        prev_phase = self.true_phase
        if self.collapsed:
            if self.true_phase == 1:
                self.g_lever = max(0.5, self.g_lever - 0.05 * np.random.uniform(0.7, 1.3))
                self.beta_lever = max(0.1, self.beta_lever - 0.02 * np.random.uniform(0.7, 1.3))
                if self.time_in_phase > np.random.randint(15, 30) and self.fcrit > self.config.fcrit_floor * 1.1:
                    self.true_phase = 2
                    self.g_lever = self.g_initial * np.random.uniform(0.5, 0.9)
                    self.beta_lever = self.beta_initial * np.random.uniform(0.1, 0.4)
            elif self.true_phase == 2:
                self.beta_lever = max(0.05, self.beta_lever - self.config.beta_decay_flaring * self.beta_lever * np.random.uniform(0.8, 1.2))
                self.g_lever = np.clip(self.g_lever + np.random.normal(0, 0.1) - 0.01, 0.2, self.g_initial * 1.5)
                self.fcrit += self.config.fcrit_replenish_rate * 0.1 
                if self.time_in_phase > np.random.randint(35, 55) and self.beta_lever < self.beta_initial * 0.35:
                    self.true_phase = 3
            elif self.true_phase == 3:
                self.beta_lever = min(self.beta_initial * 1.3, self.beta_lever + self.config.beta_rebuild_pruning * (self.beta_initial - self.beta_lever) * np.random.uniform(0.8, 1.2))
                self.g_lever = min(self.g_initial * 1.3, self.g_lever + 0.01 * (self.g_initial - self.g_lever) * np.random.uniform(0.8, 1.2))
                self.fcrit += self.config.fcrit_replenish_rate * 0.85 
                if self.time_in_phase > np.random.randint(45, 65) and self.fcrit > self.config.fcrit_initial * 0.8 and self.beta_lever > self.beta_initial * 0.8:
                    self.true_phase = 4
                    self.collapsed = False
                    self._next_shock_time = self.time + self.config.strain_shock_interval + \
                                            np.random.randint(-int(self.config.strain_shock_interval*0.1), 
                                                              int(self.config.strain_shock_interval*0.1)+1 if self.config.strain_shock_interval >0 else 0)
        else: 
            self.true_phase = 4
            if self.safety_margin < self.config.strain_baseline * 0.25:
                self.g_lever = min(self.g_initial * 1.5, self.g_lever + 0.025 * np.random.uniform(0.8,1.2))
                self.beta_lever = min(self.beta_initial * 1.5, self.beta_lever + 0.015 * np.random.uniform(0.8,1.2))
            else:
                self.g_lever = max(self.g_initial * 0.8, self.g_lever - 0.01 * (self.g_lever - self.g_initial) * np.random.uniform(0.8,1.2))
                self.beta_lever = max(self.beta_initial * 0.8, self.beta_lever - 0.005 * (self.beta_lever - self.beta_initial) * np.random.uniform(0.8,1.2))
            self.beta_lever = np.clip(self.beta_lever + np.random.normal(0, self.config.beta_phase_iv_noise_std), 0.05, self.beta_initial * 2.0)
            self.g_lever = np.clip(self.g_lever, 0.3, self.g_initial * 2.0)
        if self.true_phase != prev_phase: self.time_in_phase = 0
        else: self.time_in_phase += 1
        self.fcrit = max(self.config.fcrit_floor * 0.5, self.fcrit)

    def _check_collapse(self):
        if not self.collapsed:
            if self.fcrit <= self.config.fcrit_floor:
                self.collapsed = True; self.true_phase = 1; self.time_in_phase = 0
            elif self.strain > self.theta_t * 1.05 and self.safety_margin < -(self.c_tolerance * 0.1):
                self.collapsed = True; self.true_phase = 1; self.time_in_phase = 0

    def _record_history(self):
        self.history['run_id'].append(self.run_id)
        self.history['time'].append(self.time)
        self.history['g_lever'].append(self.g_lever)
        self.history['beta_lever'].append(self.beta_lever)
        self.history['fcrit'].append(self.fcrit)
        self.history['strain'].append(self.strain)
        self.history['theta_t'].append(self.theta_t)
        self.history['safety_margin'].append(self.safety_margin)
        self.history['true_phase'].append(self.true_phase)
        self.history['collapsed_flag'].append(1 if self.collapsed else 0)

    def step(self):
        if self.time >= self.sim_steps_limit: return False
        self._update_strain()
        self._update_levers_and_fcrit()
        self.theta_t = self._calculate_theta_t()
        self.safety_margin = self.theta_t - self.strain
        self._check_collapse()
        self.time += 1
        self._record_history()
        return True

class DiagnosticsCalculator:
    def __init__(self, env_config=None):
        self.entropy_baseline = 0.01
        self.diagnostics_list = []
        self.config = env_config # Store the explicitly passed config, or it will be None

        if self.config is None:
            # This block is for when the ML script runs standalone (__main__)
            # and DiagnosticsCalculator is instantiated without an env_config.
            print("WARNING (ML Script - DiagnosticsCalculator): env_config not provided to constructor. "
                  "Using local defaults for baseline calculation if run standalone.")
            class LocalConfig: # Define LocalConfig with all necessary attributes
                n_intervention_runs = _N_SIMULATION_RUNS_DEFAULT
                level = "local_ml_diag_default"
            self.config = LocalConfig()


    def _calculate_entropy_exp_proxy_rolling_std(self, beta_hist_single_run):
        if len(beta_hist_single_run) < ROLLING_WINDOW_ENTROPY:
            return np.zeros(len(beta_hist_single_run))
        rolling_std_beta = np.zeros_like(beta_hist_single_run)
        temp_deque = deque(maxlen=ROLLING_WINDOW_ENTROPY)
        for i in range(len(beta_hist_single_run)):
            temp_deque.append(beta_hist_single_run[i])
            if i >= ROLLING_WINDOW_ENTROPY - 1:
                rolling_std_beta[i] = np.std(list(temp_deque))
        return rolling_std_beta

    def calculate_diagnostics_for_run(self, system_run_history):
        current_diagnostics = {}
        h = {k: np.array(v) for k,v in system_run_history.items()}
        current_diagnostics['dot_beta'] = calculate_derivative(h['beta_lever'])
        current_diagnostics['dot_fcrit'] = calculate_derivative(h['fcrit'])
        speed_index_raw = np.sqrt(current_diagnostics['dot_beta']**2 + current_diagnostics['dot_fcrit']**2)
        current_diagnostics['SpeedIndex'] = smooth_series(speed_index_raw)
        couple_index = np.zeros_like(speed_index_raw)
        if len(current_diagnostics['dot_beta']) >= ROLLING_WINDOW_COUPLE:
            d_beta_window  = deque(maxlen=ROLLING_WINDOW_COUPLE)
            d_fcrit_window = deque(maxlen=ROLLING_WINDOW_COUPLE)
            for i, (db, df) in enumerate(zip(current_diagnostics['dot_beta'], current_diagnostics['dot_fcrit'])): 
                d_beta_window.append(db)
                d_fcrit_window.append(df)
                if len(d_beta_window) == ROLLING_WINDOW_COUPLE: 
                    if np.std(d_beta_window) > 1e-6 and np.std(d_fcrit_window) > 1e-6:
                        corr, _ = pearsonr(list(d_beta_window), list(d_fcrit_window)) 
                        couple_index[i] = corr if not np.isnan(corr) else 0.0 
                    else:
                        couple_index[i] = 0.0
        current_diagnostics['CoupleIndex'] = smooth_series(couple_index)
        current_diagnostics['dot_S'] = calculate_derivative(current_diagnostics['SpeedIndex'])
        entropy_exp_raw = self._calculate_entropy_exp_proxy_rolling_std(h['beta_lever'])
        current_diagnostics['EntropyExp'] = smooth_series(entropy_exp_raw)
        current_diagnostics['g_lever_raw'] = h['g_lever']
        current_diagnostics['beta_lever_raw'] = h['beta_lever']
        current_diagnostics['fcrit_raw'] = h['fcrit']
        current_diagnostics['strain_raw'] = h['strain']
        current_diagnostics['theta_t_raw'] = h['theta_t']
        current_diagnostics['safety_margin_raw'] = h['safety_margin']
        current_diagnostics['true_phase'] = h['true_phase']
        current_diagnostics['run_id'] = h['run_id']
        self.diagnostics_list.append(current_diagnostics)
        return current_diagnostics

    def finalize_rhoE_and_baseline(self): # Does not take recompute_baseline
        all_stable_phase_iv_entropy_exp = []
        start_idx_for_valid_entropy = ROLLING_WINDOW_ENTROPY - 1
        
        # Use n_intervention_runs from the config stored in this instance
        n_sim_runs_for_baseline = self.config.n_intervention_runs
        # print(f"DEBUG (ML DiagCalc): Using n_intervention_runs from self.config (level: {self.config.level if hasattr(self.config, 'level') else 'N/A'}): {n_sim_runs_for_baseline}")

        for run_diag in self.diagnostics_list:
            true_phases_arr = run_diag['true_phase']
            entropy_exp_arr = run_diag['EntropyExp']
            phase_iv_indices = np.where(true_phases_arr == 4)[0]
            count = 0
            for idx in phase_iv_indices:
                if idx >= start_idx_for_valid_entropy and idx < len(entropy_exp_arr):
                    if count < 50 :
                        all_stable_phase_iv_entropy_exp.append(entropy_exp_arr[idx])
                        count +=1
                    else: break
        if len(all_stable_phase_iv_entropy_exp) >= max(30, n_sim_runs_for_baseline * (ROLLING_WINDOW_ENTROPY // 4) ):
            calculated_baseline = np.median(all_stable_phase_iv_entropy_exp)
            self.entropy_baseline = max(0.01, calculated_baseline)
            source_msg = f" (config level: {self.config.level if hasattr(self.config, 'level') else 'N/A'})"
            print(f"INFO (ML Script): Global EntropyBaseline calculated from {len(all_stable_phase_iv_entropy_exp)} stable Phase IV points{source_msg}: {self.entropy_baseline:.4f}")
        else:
            print(f"WARNING (ML Script): Global EntropyBaseline set to fallback {self.entropy_baseline:.4f} due to insufficient data ({len(all_stable_phase_iv_entropy_exp)} points). Needed {max(30, n_sim_runs_for_baseline * (ROLLING_WINDOW_ENTROPY // 4))}.")
        
        for run_diag in self.diagnostics_list:
            run_diag['rhoE'] = run_diag['EntropyExp'] / (self.entropy_baseline + 1e-9)
            run_diag['dot_rhoE'] = calculate_derivative(run_diag['rhoE'])

# --- Feature Engineering for ML (Unchanged) ---
def create_ml_features(diagnostics_list, feature_window_size):
    all_features = []; all_labels = []; all_run_ids = []
    feature_names = [
        'g_lever_raw', 'beta_lever_raw', 'fcrit_raw', 'strain_raw', 'theta_t_raw', 'safety_margin_raw',
        'SpeedIndex', 'CoupleIndex', 'EntropyExp', 'rhoE', 'dot_S', 'dot_rhoE'
    ]
    for run_idx, run_diag in enumerate(diagnostics_list):
        num_steps = len(run_diag['SpeedIndex']) # Assuming SpeedIndex always present and representative length
        for t in range(BURN_IN_PERIOD + feature_window_size -1, num_steps):
            current_features = []
            for feature_name in feature_names:
                if feature_name in run_diag and len(run_diag[feature_name]) > t: # Check if feature exists and is long enough
                    current_features.append(run_diag[feature_name][t])
                    window_data = run_diag[feature_name][t - feature_window_size + 1 : t + 1]
                    if len(window_data) == feature_window_size:
                        current_features.append(np.mean(window_data))
                        current_features.append(np.std(window_data))
                    else: current_features.extend([np.nan,np.nan]) # Use NaN for missing window stats
                else: current_features.extend([np.nan,np.nan,np.nan]) # Use NaN for missing feature data
            all_features.append(current_features)
            # Ensure true_phase and run_id are also long enough
            if 'true_phase' in run_diag and len(run_diag['true_phase']) > t:
                all_labels.append(run_diag['true_phase'][t])
            else:
                all_labels.append(np.nan) # Or a suitable placeholder

            if 'run_id' in run_diag and len(run_diag['run_id']) > t: # run_id is often scalar per run, adapt if it's per step
                 all_run_ids.append(run_diag['run_id'][t] if isinstance(run_diag['run_id'], list) or isinstance(run_diag['run_id'], np.ndarray) else run_diag['run_id'][0])
            else:
                 all_run_ids.append(np.nan)

    df_feature_names = []
    for name in feature_names:
        df_feature_names.append(name + "_t")
        df_feature_names.append(name + "_win_mean")
        df_feature_names.append(name + "_win_std")
    
    # Create DataFrame and then handle potential NaNs
    features_df = pd.DataFrame(all_features, columns=df_feature_names)
    labels_array = np.array(all_labels)
    run_ids_array = np.array(all_run_ids)

    # Optional: Drop rows with any NaNs if that's desired, or impute them.
    # For now, let's keep them and let downstream processes handle.
    # if features_df.isnull().any().any():
    #     print(f"Warning: NaNs found in feature data. {features_df.isnull().sum().sum()} NaNs.")
        # features_df = features_df.dropna()
        # corresponding_indices = features_df.index
        # labels_array = labels_array[corresponding_indices]
        # run_ids_array = run_ids_array[corresponding_indices]

    return features_df, labels_array, run_ids_array


# --- Main Simulation and ML Training ---
if __name__ == "__main__":
    n_simulation_runs_local = _N_SIMULATION_RUNS_DEFAULT
    sim_steps_for_ml_training = _SIM_STEPS_DEFAULT_INTERNAL 

    print(f"Starting TD Phoenix Loop ML Study with {n_simulation_runs_local} runs of {sim_steps_for_ml_training} steps each...")
    all_run_histories = []
    # For standalone, TDSystem will use its internal LocalConfig if env_config is None
    # The LocalConfig in TDSystem should define n_intervention_runs for baseline calc
    temp_system_for_config_access = TDSystem(env_config=None) # To get its default local config for plotting fcrit_floor
    
    for i in range(n_simulation_runs_local):
        system = TDSystem(run_id=i, sim_steps_override=sim_steps_for_ml_training, env_config=None) 
        while system.step(): pass
        all_run_histories.append(system.history)
        if (i+1) % 10 == 0: print(f"  Completed simulation run {i+1}/{n_simulation_runs_local}")

    print("Calculating diagnostics for all runs...")
    # DiagnosticsCalculator instantiated without env_config. It will use its own LocalConfig.
    diag_calculator = DiagnosticsCalculator(env_config=None) 
    for history_data in all_run_histories: diag_calculator.calculate_diagnostics_for_run(history_data)
    diag_calculator.finalize_rhoE_and_baseline()
    
    print(f"Creating ML features with window size {FEATURE_WINDOW_SIZE}...")
    X_df, y, run_ids_for_split = create_ml_features(diag_calculator.diagnostics_list, FEATURE_WINDOW_SIZE)

    if X_df.empty:
        print("ERROR: No features were generated. Check BURN_IN_PERIOD and FEATURE_WINDOW_SIZE.")
        exit()
    # Remove rows where y is NaN (if any were created due to length mismatch)
    nan_label_indices = np.where(pd.isnull(y))[0]
    if len(nan_label_indices) > 0:
        print(f"Warning: Removing {len(nan_label_indices)} samples due to NaN labels.")
        X_df = X_df.drop(index=nan_label_indices).reset_index(drop=True)
        y = np.delete(y, nan_label_indices)
        run_ids_for_split = np.delete(run_ids_for_split, nan_label_indices)
    
    # Remove rows where X_df has NaNs (if any were created due to length mismatch)
    if X_df.isnull().any().any():
        print(f"Warning: X_df contains NaNs. Dropping rows with NaNs.")
        nan_feature_indices = X_df[X_df.isnull().any(axis=1)].index
        X_df = X_df.dropna().reset_index(drop=True)
        y = np.delete(y, nan_feature_indices) # y should correspond to original X_df rows
        run_ids_for_split = np.delete(run_ids_for_split, nan_feature_indices) # Same for run_ids
        print(f"Number of samples after NaN removal: {len(X_df)}")


    if len(X_df) == 0:
        print("ERROR: No valid samples left after NaN removal for ML.")
        exit()
    print(f"Generated {len(X_df)} samples for ML.")


    unique_runs = np.unique(run_ids_for_split[~pd.isnull(run_ids_for_split)]) # Exclude NaN run_ids for split
    if not unique_runs.size:
        print("ERROR: No valid run IDs for splitting data.")
        exit()

    train_run_ids, test_run_ids = train_test_split(unique_runs, test_size=0.3, random_state=42)
    
    # Ensure we only use indices where run_ids_for_split is not NaN
    valid_run_id_mask = ~pd.isnull(run_ids_for_split)
    train_indices = np.where(valid_run_id_mask & np.isin(run_ids_for_split, train_run_ids))[0]
    test_indices = np.where(valid_run_id_mask & np.isin(run_ids_for_split, test_run_ids))[0]

    if not train_indices.size or not test_indices.size:
        print("ERROR: Could not create train/test split. Check data and run IDs.")
        exit()

    X_train_df, X_test_df = X_df.iloc[train_indices], X_df.iloc[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    print(f"Training samples: {len(X_train_df)}, Test samples: {len(X_test_df)}")

    scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)

    print("Training Random Forest Classifier...")
    param_grid_simple = {'n_estimators': [50, 100], 'max_depth': [10, None], 'min_samples_split': [5]}
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
                               param_grid_simple, cv=3, scoring='f1_macro', verbose=0) 
    grid_search.fit(X_train_scaled, y_train)
    best_rf_model = grid_search.best_estimator_
    print("Best Random Forest hyperparameters:", grid_search.best_params_)
    joblib.dump(best_rf_model, 'phoenix_rf_classifier.joblib')
    joblib.dump(scaler, 'phoenix_feature_scaler.joblib')
    print("Trained model and scaler saved.")

    print("\n--- ML Classifier Performance on Test Set ---")
    y_pred = best_rf_model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, labels=[1, 2, 3, 4],
                                   target_names=['Phase I', 'Phase II', 'Phase III', 'Phase IV'], zero_division=0)
    print("Classification Report (ML Model):\n", report)
    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4])
    print("Confusion Matrix (ML Model):\n", cm)
    accuracy = np.mean(y_test == y_pred)
    print(f"Overall Accuracy (ML Model on test set): {accuracy:.4f}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['I', 'II', 'III', 'IV'])
    disp.plot(cmap=plt.cm.viridis); plt.title("Confusion Matrix (ML Model - Test Set)"); plt.show(block=False) 

    if len(test_run_ids) > 0:
        example_test_run_id = test_run_ids[0]
        print(f"\nPlotting diagnostics and ML predictions for example test run: {example_test_run_id}")
        example_run_history = next((h for h in all_run_histories if h['run_id'][0] == example_test_run_id), None)
        example_run_diagnostics = next((d for d in diag_calculator.diagnostics_list if d['run_id'][0] == example_test_run_id), None)
        
        if example_run_history and example_run_diagnostics:
            plot_config_fcrit_floor = temp_system_for_config_access.config.fcrit_floor

            run_X_df_single, run_y_true_single, _ = create_ml_features([example_run_diagnostics], FEATURE_WINDOW_SIZE)
            
            # Handle NaNs from create_ml_features before scaling/predicting
            if run_X_df_single.isnull().any().any():
                print("Warning: NaNs in example run features. Dropping these rows for prediction.")
                run_X_df_single = run_X_df_single.dropna()
            
            if not run_X_df_single.empty:
                run_X_scaled_single = scaler.transform(run_X_df_single)
                run_y_pred_ml_single = best_rf_model.predict(run_X_scaled_single)
                
                full_run_y_pred_ml = np.full(len(example_run_history['time']), np.nan, dtype=float) # Use NaN for non-predicted steps
                
                # Align predictions with the original timeline indices from run_X_df_single.index
                # These indices correspond to the 't' values in create_ml_features
                # The first prediction corresponds to time step (BURN_IN_PERIOD + FEATURE_WINDOW_SIZE - 1)
                
                # We need to map the indices of run_X_df_single (after potential dropna) 
                # back to the original time steps. create_ml_features starts from t = BURN_IN + WIN -1.
                # So, the first row of run_X_df_single corresponds to time index (BURN_IN + WIN -1).

                prediction_start_time_index = BURN_IN_PERIOD + FEATURE_WINDOW_SIZE - 1
                
                # Iterate through the valid predictions and place them in full_run_y_pred_ml
                # The indices of run_X_df_single (after dropna) tell us *which* of the original possible prediction points were kept.
                original_time_indices_for_predictions = [prediction_start_time_index + i for i in run_X_df_single.index]

                for i, pred_val in enumerate(run_y_pred_ml_single):
                    original_t_idx = original_time_indices_for_predictions[i]
                    if original_t_idx < len(full_run_y_pred_ml):
                         full_run_y_pred_ml[original_t_idx] = pred_val
                
                time_hist_ex = np.array(example_run_history['time'])
                true_phases_hist_ex = np.array(example_run_history['true_phase'])
                fig, axs = plt.subplots(7, 1, figsize=(16, 20), sharex=True) 
                axs[0].plot(time_hist_ex, example_run_history['g_lever'], label='gLever'); axs[0].plot(time_hist_ex, example_run_history['beta_lever'], label='betaLever')
                axs[0].set_title(f'TD Levers (Run {example_test_run_id})'); axs[0].legend(); axs[0].grid(True, alpha=0.5)
                add_true_phase_background(axs[0], time_hist_ex, true_phases_hist_ex)
                axs[1].plot(time_hist_ex, example_run_history['fcrit'], label='Fcrit', c='g'); axs[1].axhline(plot_config_fcrit_floor, c='r', ls=':', label='Fcrit Floor')
                axs[1].set_title('Fcrit'); axs[1].legend(); axs[1].grid(True, alpha=0.5)
                add_true_phase_background(axs[1], time_hist_ex, true_phases_hist_ex)
                axs[2].plot(time_hist_ex, example_run_history['strain'], label='Strain', c='r'); axs[2].plot(time_hist_ex, example_run_history['theta_t'], label='Theta_T', c='b', ls='--')
                axs[2].plot(time_hist_ex, example_run_history['safety_margin'], label='Safety Margin', c='purple', alpha=0.7)
                axs[2].set_title('Strain vs. Tolerance'); axs[2].legend(); axs[2].grid(True, alpha=0.5)
                add_true_phase_background(axs[2], time_hist_ex, true_phases_hist_ex)
                time_diag_ex = time_hist_ex[:len(example_run_diagnostics.get('SpeedIndex',[]))]
                if len(time_diag_ex)>0:
                    axs[3].plot(time_diag_ex, example_run_diagnostics['SpeedIndex'], label='SpeedIndex', c='purple'); axs[3].set_ylabel('SpeedIndex', c='purple'); axs[3].tick_params(axis='y', labelcolor='purple')
                    ax3_twin = axs[3].twinx(); ax3_twin.plot(time_diag_ex, example_run_diagnostics['CoupleIndex'], label='CoupleIndex', c='teal', ls=':'); ax3_twin.set_ylabel('CoupleIndex', c='teal'); ax3_twin.tick_params(axis='y', labelcolor='teal')
                    axs[3].set_title('SpeedIndex & CoupleIndex'); axs[3].legend(loc='upper left'); ax3_twin.legend(loc='upper right'); axs[3].grid(True, alpha=0.5)
                    add_true_phase_background(axs[3], time_diag_ex, true_phases_hist_ex[:len(time_diag_ex)])
                    axs[4].plot(time_diag_ex, example_run_diagnostics['EntropyExp'], label='EntropyExp', c='darkcyan'); axs[4].set_title('EntropyExp'); axs[4].legend(); axs[4].grid(True, alpha=0.5)
                    add_true_phase_background(axs[4], time_diag_ex, true_phases_hist_ex[:len(time_diag_ex)])
                    axs[5].plot(time_diag_ex, example_run_diagnostics['rhoE'], label='rhoE', c='brown'); axs[5].set_ylabel('rhoE', c='brown'); axs[5].tick_params(axis='y', labelcolor='brown')
                    ax5_twin = axs[5].twinx(); ax5_twin.plot(time_diag_ex, example_run_diagnostics['dot_S'], label='dot_S', c='darkorange', ls='--'); ax5_twin.plot(time_diag_ex, example_run_diagnostics['dot_rhoE'], label='dot_rhoE', c='magenta', ls='-.'); ax5_twin.set_ylabel('Derivatives', c='gray'); ax5_twin.tick_params(axis='y', labelcolor='gray')
                    axs[5].set_title('rhoE & Derivatives'); axs[5].legend(loc='upper left'); ax5_twin.legend(loc='upper right'); axs[5].grid(True, alpha=0.5)
                    add_true_phase_background(axs[5], time_diag_ex, true_phases_hist_ex[:len(time_diag_ex)])
                axs[6].plot(time_hist_ex, true_phases_hist_ex, label='Ground Truth Phase', c='k', lw=2, ds='steps-post'); axs[6].plot(time_hist_ex, full_run_y_pred_ml, label='ML Predicted Phase', c='dodgerblue', ls=':', lw=2, ds='steps-post')
                axs[6].set_title(f'Phase Prediction (Run {example_test_run_id})'); axs[6].set_yticks([1,2,3,4]); axs[6].set_yticklabels(['I', 'II', 'III', 'IV']); axs[6].legend(); axs[6].grid(True, alpha=0.5)
                plt.tight_layout(); plt.show()
            else:
                print(f"No valid features to predict for example run {example_test_run_id} after NaN removal.")
        else:
            print(f"Could not find history or diagnostics for example test run {example_test_run_id}")
            