# aif_phoenix_sim.py

import numpy as np
import matplotlib.pyplot as plt
# Set consistent font sizes across all plots
plt.rcParams.update({'font.size': 16,
                     'xtick.labelsize': 12,
                     'ytick.labelsize': 12})
from matplotlib.colors import ListedColormap
from scipy.stats import entropy as shannon_entropy
from scipy.signal import savgol_filter
from collections import Counter
import random
import time
import os

import joblib
import pandas as pd
from scipy.stats import linregress

# --- Constants matching train_aif_specific_classifier.py ---
AIF_SPECIFIC_FEATURE_WINDOW_SIZE = 20
AIF_SPECIFIC_SMOOTHING_WINDOW = 15
AIF_SPECIFIC_SMOOTHING_POLYORDER = 2
AIF_SPECIFIC_BURN_IN_FOR_FEATURES = 30
LONG_SLOPE_WINDOW_AIF = 25

BASE_FEATURE_KEYS_FOR_AIF_MODEL = [
    'avg_PE_system', 'avg_beta_system', 'avg_fcrit_system',
    'active_agents_count', 'SpeedIndex', 'CoupleIndex',
    'model_entropy_system', 'rhoE'
]

DERIVATIVE_FEATURE_KEYS_FOR_AIF_MODEL = [
    'avg_PE_system', 'avg_beta_system', 'avg_fcrit_system',
    'SpeedIndex', 'model_entropy_system', 'rhoE', 'active_agents_count'
]

SLOPE_TARGET_KEYS_FOR_AIF_MODEL = ['avg_beta_system', 'avg_fcrit_system', 'rhoE', 'avg_PE_system']
STABILITY_STD_TARGET_KEYS_FOR_AIF_MODEL = ['avg_beta_system', 'avg_fcrit_system', 'avg_PE_system']

# --- Configuration ---
class EnvironmentConfigAIF:
    def __init__(self, study_id="aif_phoenix_v1"):
        self.study_id = study_id
        # Grid
        self.grid_size = (30, 30) # Smaller for faster initial testing
        self.n_agents = int(0.5 * self.grid_size[0] * self.grid_size[1]) # 50% density

        # Environment States
        self.possible_env_states = [0, 1, 2] # e.g., different resource types or conditions
        self.initial_env_state = 0
        self.novel_env_state = 1 # State after shock
        self.t_shock = 150 # Time step when environment changes
        self.observation_noise_std = 0.1 # Noise added to agent's observation of env state

        # Agent Parameters
        self.agent_initial_fcrit = 10.0
        self.agent_fcrit_floor = 0.0
        self.agent_fcrit_replenish_on_low_pe = 0.1 # Amount to replenish
        self.agent_pe_threshold_for_replenish = 0.2 # If PE is below this
        self.cost_model_update = 0.05      # Cost for any model update
        self.cost_exploration_factor = 0.1 # Extra cost if beta_i is low (exploration)
        self.cost_high_beta_factor = 0.02  # Cost for maintaining high precision
        self.cost_base_existence = 0.01    # Per-step existence cost
        self.initial_beta_agent = 0.9      # Initial confidence/precision
        self.beta_min = 0.05
        self.beta_max = 0.95
        self.beta_adapt_rate_high_pe = 0.15 # Rate at which beta decreases if PE is high
        self.beta_adapt_rate_low_pe = 0.1  # Rate at which beta increases if PE is low
        self.pe_threshold_for_beta_adapt = 0.3 # PE threshold to adapt beta

        # Simulation
        self.num_steps = 800
        self.random_seed = 42

        # Agent Rebirth/Innovation
        self.enable_rebirth = True
        self.rebirth_check_interval = 5 # Check every N steps to repopulate
        self.max_agents = self.n_agents # Maintain roughly constant population
        self.rebirth_model_from_successful_neighbor_prob = 0.6 # For pruning-like behavior
        self.rebirth_random_model_prob = 0.4
        self.min_fcrit_for_successful_copy = self.agent_initial_fcrit * 0.5

        # Diagnostics
        self.diag_smoothing_window = 21
        self.diag_polyorder = 2
        self.diag_couple_window = 30
        self.rhoE_baseline_period_start = 50
        self.rhoE_baseline_period_end = self.t_shock - 10 # Use pre-shock stable period

# --- Agent Class ---
class AIFPhoenixAgent:
    def __init__(self, agent_id, x, y, config, initial_model=None):
        self.id = agent_id
        self.x = x
        self.y = y
        self.config = config
        
        if initial_model is None:
            self.internal_model_M = random.choice(self.config.possible_env_states)
        else:
            self.internal_model_M = initial_model
            
        self.local_precision_beta = self.config.initial_beta_agent
        self.local_fcrit = self.config.agent_initial_fcrit
        self.prediction_error_PE = 0.0
        self.is_active = True

        # History tracking for individual agent (optional, can be heavy)
        # self.history = {'M': [], 'beta': [], 'fcrit': [], 'pe': []}

    def perceive_and_update_PE(self, environment):
        if not self.is_active: return
        true_cell_state = environment.get_true_state(self.x, self.y)
        observation = true_cell_state + np.random.normal(0, self.config.observation_noise_std)
        # Clip observation to be within possible discrete states for PE calculation, or allow continuous PE
        # For simplicity with discrete models, we can round observation for PE calculation
        # observed_state_discrete = round(np.clip(observation, min(self.config.possible_env_states), max(self.config.possible_env_states)))
        # self.prediction_error_PE = abs(observed_state_discrete - self.internal_model_M)
        self.prediction_error_PE = abs(observation - self.internal_model_M) # Continuous PE

    def adapt_beta(self):
        if not self.is_active: return
        # If PE is high, decrease beta (lose confidence, explore more)
        if self.prediction_error_PE > self.config.pe_threshold_for_beta_adapt:
            self.local_precision_beta -= self.config.beta_adapt_rate_high_pe * self.local_precision_beta
        # If PE is low, increase beta (gain confidence, exploit more)
        else:
            self.local_precision_beta += self.config.beta_adapt_rate_low_pe * (1 - self.local_precision_beta)
        self.local_precision_beta = np.clip(self.local_precision_beta, self.config.beta_min, self.config.beta_max)

    def update_model(self, environment):
        if not self.is_active: return
        cost = self.config.cost_model_update

        # Exploitation (high beta) vs. Exploration (low beta)
        if self.local_precision_beta > 0.5:  # Exploitation
            true_cell_state = environment.get_true_state(self.x, self.y)
            observation = true_cell_state + np.random.normal(0, self.config.observation_noise_std)
            potential_models = self.config.possible_env_states
            errors_for_potentials = [abs(observation - pm) for pm in potential_models]
            best_potential_model = potential_models[np.argmin(errors_for_potentials)]

            if random.random() < self.local_precision_beta:
                self.internal_model_M = best_potential_model
            else:
                current_idx = self.config.possible_env_states.index(self.internal_model_M)
                step_size = random.choice([-1, 1])
                new_idx = np.clip(current_idx + step_size, 0, len(self.config.possible_env_states)-1)
                self.internal_model_M = self.config.possible_env_states[new_idx]
            cost += self.config.cost_high_beta_factor * self.local_precision_beta
        else:  # Exploration
            self.internal_model_M = random.choice(self.config.possible_env_states)
            cost += self.config.cost_exploration_factor * (1 - self.local_precision_beta)
        
        self.local_fcrit -= cost

    def update_fcrit(self):
        if not self.is_active:
            return
        self.local_fcrit -= self.config.cost_base_existence
        if self.prediction_error_PE < self.config.agent_pe_threshold_for_replenish:
            self.local_fcrit += self.config.agent_fcrit_replenish_on_low_pe

        self.local_fcrit = np.clip(self.local_fcrit, 0, self.config.agent_initial_fcrit * 1.5)

        if self.local_fcrit <= self.config.agent_fcrit_floor:
            self.is_active = False
            self.local_fcrit = 0

    def can_be_copied(self):
        return (
            self.is_active
            and self.local_fcrit >= self.config.min_fcrit_for_successful_copy
            and self.prediction_error_PE < self.config.agent_pe_threshold_for_replenish
        )

    def step(self, environment):
        if not self.is_active: return
        self.perceive_and_update_PE(environment)
        self.adapt_beta()
        self.update_model(environment) # Pass env if model update depends on observation
        self.update_fcrit()
        
        # self.history['M'].append(self.internal_model_M)
        # self.history['beta'].append(self.local_precision_beta)
        # self.history['fcrit'].append(self.local_fcrit)
        # self.history['pe'].append(self.prediction_error_PE)




import numpy as np

# --- Environment Class ---
class AIFGridEnvironment:
    def __init__(self, config):
        self.config = config
        self.grid_shape = config.grid_size
        self.true_env_states = np.full(self.grid_shape, config.initial_env_state, dtype=int)
        self.current_time = 0

    def step(self):
        self.current_time += 1
        if self.current_time == self.config.t_shock:
            self.apply_shock()

    def apply_shock(self):
        print(f"INFO: Applying environmental shock at t={self.current_time}. State changes to {self.config.novel_env_state}")
        self.true_env_states.fill(self.config.novel_env_state)

    def get_true_state(self, x, y):
        return self.true_env_states[x, y]

# --- Simulation Class ---
class AIFPhoenixSimulation:
    def __init__(self, config):
        self.config = config
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        self.environment = AIFGridEnvironment(config)
        self.agents = []
        self._initialize_agents()

        self.ml_model_path = "aif_phoenix_classifier_specific.joblib"
        self.ml_scaler_path = "aif_phoenix_scaler_specific.joblib"
        try:
            self.ml_classifier = joblib.load(self.ml_model_path)
            self.ml_scaler = joblib.load(self.ml_scaler_path)
            print(f"INFO (AIF SIM): Successfully loaded AIF-SPECIFIC ML model and scaler from {self.ml_model_path}.")
            self.ml_components_loaded = True
        except FileNotFoundError:
            print(
                f"WARNING (AIF SIM): AIF-SPECIFIC ML model or scaler not found at {self.ml_model_path} or {self.ml_scaler_path}. "
                "Will use fallback rule-based classifier."
            )
            self.ml_classifier = None
            self.ml_scaler = None
            self.ml_components_loaded = False

        self.history = {
            'time': [],
            'avg_PE_system': [],
            'avg_beta_system': [],
            'avg_fcrit_system': [],
            'model_entropy_system': [],
            'active_agents_count': [],
            'SpeedIndex': [],
            'CoupleIndex': [],
            'rhoE': [],
            'algorithmic_phase': []
        }
        self.entropy_baseline = None
        self.agent_model_snapshots = {}
        self.snapshot_interval = 25
        self.snapshot_times = set([
            max(0, self.config.t_shock - 10),
            self.config.t_shock + 10,
            self.config.t_shock + 100,
            self.config.t_shock + 250,
            self.config.num_steps - 10,
        ])

    def _initialize_agents(self):
        occupied_cells = set()
        for i in range(self.config.n_agents):
            while True:
                x = random.randrange(self.config.grid_size[0])
                y = random.randrange(self.config.grid_size[1])
                if (x,y) not in occupied_cells:
                    occupied_cells.add((x,y))
                    break
            self.agents.append(AIFPhoenixAgent(i, x, y, self.config, initial_model=self.config.initial_env_state))

    def _attempt_rebirth(self):
        if not self.config.enable_rebirth: return
        
        num_active_agents = sum(1 for agent in self.agents if agent.is_active)
        num_to_rebirth = self.config.max_agents - num_active_agents

        if num_to_rebirth <= 0: return

        # Find all current agent locations to find empty cells
        occupied_cells = set((agent.x, agent.y) for agent in self.agents if agent.is_active)
        all_possible_cells = set((r, c) for r in range(self.config.grid_size[0]) for c in range(self.config.grid_size[1]))
        empty_cells = list(all_possible_cells - occupied_cells)
        random.shuffle(empty_cells)
        
        successful_agents = [agent for agent in self.agents if agent.can_be_copied()]


        for _ in range(min(num_to_rebirth, len(empty_cells))):
            x, y = empty_cells.pop()
            new_agent_id = len(self.agents) # Simple unique ID
            
            initial_model_for_new = self.config.initial_env_state
            if successful_agents and random.random() < self.config.rebirth_model_from_successful_neighbor_prob:
                copied_agent = random.choice(successful_agents)
                initial_model_for_new = copied_agent.internal_model_M
            elif random.random() < self.config.rebirth_random_model_prob:
                initial_model_for_new = random.choice(self.config.possible_env_states)
            
            new_agent = AIFPhoenixAgent(new_agent_id, x, y, self.config, initial_model=initial_model_for_new)
            self.agents.append(new_agent) # Add to main list
            # Note: In a more complex spatial ABM, you'd add this agent to the grid structure too.


    def _collect_macroscopic_data(self, t):
        self.history['time'].append(t)
        
        active_agents = [agent for agent in self.agents if agent.is_active]
        self.history['active_agents_count'].append(len(active_agents))

        if not active_agents:
            self.history['avg_PE_system'].append(np.nan)
            self.history['avg_beta_system'].append(np.nan)
            self.history['avg_fcrit_system'].append(np.nan)
            self.history['model_entropy_system'].append(np.nan)
            return

        pes = [agent.prediction_error_PE for agent in active_agents]
        betas = [agent.local_precision_beta for agent in active_agents]
        fcrits = [agent.local_fcrit for agent in active_agents]
        models = [agent.internal_model_M for agent in active_agents]

        self.history['avg_PE_system'].append(np.mean(pes) if pes else np.nan)
        self.history['avg_beta_system'].append(np.mean(betas) if betas else np.nan)
        self.history['avg_fcrit_system'].append(np.mean(fcrits) if fcrits else np.nan)
        
        if models:
            model_counts = Counter(models)
            all_states_counts = [model_counts.get(s, 0) for s in self.config.possible_env_states]
            probabilities = [count / len(models) for count in all_states_counts if count > 0]
            self.history['model_entropy_system'].append(shannon_entropy(probabilities, base=2) if probabilities else 0.0)
        else:
            self.history['model_entropy_system'].append(np.nan)

    def _history_to_diag_dict(self):
        """Convert stored history into diagnostics dictionary for the ML classifier."""
        h = self.history
        arr = np.array
        num_time_steps = len(h['time'])

        avg_beta_arr = arr(h['avg_beta_system']) if len(h['avg_beta_system']) > 0 else np.full(num_time_steps, np.nan)
        avg_fcrit_arr = arr(h['avg_fcrit_system']) if len(h['avg_fcrit_system']) > 0 else np.full(num_time_steps, np.nan)
        avg_pe_arr = arr(h['avg_PE_system']) if len(h['avg_PE_system']) > 0 else np.full(num_time_steps, np.nan)


        # Proxy features for TD system compatibility
        g_lever_proxy = np.full(num_time_steps, 1.0)

        W1_DEFAULT, W2_DEFAULT, W3_DEFAULT = 0.33, 0.33, 0.34
        C_TOLERANCE_DEFAULT = 1.0

        fcrit_clipped_proxy = np.clip(avg_fcrit_arr, 1e-9, None)
        beta_clipped_proxy = np.clip(avg_beta_arr, 1e-9, None)

        theta_t_proxy = (
            C_TOLERANCE_DEFAULT
            * (g_lever_proxy ** W1_DEFAULT)
            * (beta_clipped_proxy ** W2_DEFAULT)
            * (fcrit_clipped_proxy ** W3_DEFAULT)
        )

        safety_margin_proxy = theta_t_proxy - avg_pe_arr

        diag_dict = {
            'g_lever_raw': g_lever_proxy,
            'beta_lever_raw': avg_beta_arr,
            'fcrit_raw': avg_fcrit_arr,
            'strain_raw': avg_pe_arr,
            'theta_t_raw': theta_t_proxy,
            'safety_margin_raw': safety_margin_proxy,

            'SpeedIndex': arr(h['SpeedIndex']) if len(h['SpeedIndex']) > 0 else np.full(num_time_steps, np.nan),
            'CoupleIndex': arr(h['CoupleIndex']) if len(h['CoupleIndex']) > 0 else np.full(num_time_steps, np.nan),
            'EntropyExp': arr(h['model_entropy_system']) if len(h['model_entropy_system']) > 0 else np.full(num_time_steps, np.nan),
            'rhoE': arr(h['rhoE']) if len(h['rhoE']) > 0 else np.full(num_time_steps, np.nan),
            'dot_S': savgol_filter(np.gradient(arr(h['SpeedIndex'])), self.config.diag_smoothing_window, self.config.diag_polyorder, mode='mirror') if len(h['SpeedIndex']) > self.config.diag_smoothing_window else np.zeros(num_time_steps),
            'dot_rhoE': savgol_filter(np.gradient(arr(h['rhoE'])), self.config.diag_smoothing_window, self.config.diag_polyorder, mode='mirror') if len(h['rhoE']) > self.config.diag_smoothing_window else np.zeros(num_time_steps),

            'true_phase': [0] * num_time_steps,
            'run_id': [0] * num_time_steps,
        }
        return diag_dict

    def _generate_features_for_ml_prediction(self):
        """Generate features for ML prediction mirroring training pipeline."""
        history_data = self.history
        num_steps_total = len(history_data['time'])

        processed_series = {}
        feature_names_generated_base = []

        for key in BASE_FEATURE_KEYS_FOR_AIF_MODEL:
            if key in history_data and len(history_data[key]) == num_steps_total:
                series = np.array(history_data[key])
                processed_series[key] = np.nan_to_num(series)
                feature_names_generated_base.append(key)
            else:
                processed_series[key] = np.full(num_steps_total, np.nan)
                feature_names_generated_base.append(key)

        for key_to_derive in DERIVATIVE_FEATURE_KEYS_FOR_AIF_MODEL:
            if key_to_derive in processed_series:
                series = processed_series[key_to_derive]
                dot_key = f"dot_{key_to_derive}"
                if len(series) > AIF_SPECIFIC_SMOOTHING_WINDOW:
                    smoothed_arr = savgol_filter(series, AIF_SPECIFIC_SMOOTHING_WINDOW, AIF_SPECIFIC_SMOOTHING_POLYORDER, mode='mirror')
                    processed_series[dot_key] = np.gradient(smoothed_arr)
                else:
                    processed_series[dot_key] = np.zeros_like(series)

                if dot_key not in feature_names_generated_base:
                    feature_names_generated_base.append(dot_key)

        for slope_key in SLOPE_TARGET_KEYS_FOR_AIF_MODEL:
            if slope_key in processed_series:
                series = processed_series[slope_key]
                slope_vals = np.full(num_steps_total, np.nan, dtype=float)
                for idx in range(num_steps_total):
                    if idx >= LONG_SLOPE_WINDOW_AIF - 1:
                        window = series[idx - LONG_SLOPE_WINDOW_AIF + 1 : idx + 1]
                        if not np.all(np.isnan(window)) and len(window) == LONG_SLOPE_WINDOW_AIF:
                            try:
                                slope, _, _, _, _ = linregress(np.arange(LONG_SLOPE_WINDOW_AIF), np.nan_to_num(window))
                                slope_vals[idx] = slope if not np.isnan(slope) else 0.0
                            except ValueError:
                                slope_vals[idx] = 0.0
                        else:
                            slope_vals[idx] = 0.0
                new_key = f"slope_{slope_key}"
                processed_series[new_key] = np.nan_to_num(slope_vals)
                if new_key not in feature_names_generated_base:
                    feature_names_generated_base.append(new_key)

        for stab_key in STABILITY_STD_TARGET_KEYS_FOR_AIF_MODEL:
            if stab_key in processed_series:
                series = processed_series[stab_key]
                std_vals = np.full(num_steps_total, np.nan, dtype=float)
                for idx in range(num_steps_total):
                    if idx >= LONG_SLOPE_WINDOW_AIF - 1:
                        window = series[idx - LONG_SLOPE_WINDOW_AIF + 1 : idx + 1]
                        std_vals[idx] = np.nanstd(window)
                new_key = f"stability_std_{stab_key}"
                processed_series[new_key] = np.nan_to_num(std_vals)
                if new_key not in feature_names_generated_base:
                    feature_names_generated_base.append(new_key)

        time_since_shock = np.array([max(0, t - self.config.t_shock) for t in range(num_steps_total)])
        processed_series['time_since_shock'] = time_since_shock
        if 'time_since_shock' not in feature_names_generated_base:
            feature_names_generated_base.append('time_since_shock')

        X_run_features_list = []
        for t in range(num_steps_total):
            feature_vector_t = []
            for base_feature_name in feature_names_generated_base:
                series = processed_series[base_feature_name]
                current_value = series[t]
                if t >= AIF_SPECIFIC_FEATURE_WINDOW_SIZE - 1:
                    window_data = series[t - AIF_SPECIFIC_FEATURE_WINDOW_SIZE + 1 : t + 1]
                    window_mean = np.nanmean(window_data)
                    window_std = np.nanstd(window_data)
                else:
                    window_mean = np.nan
                    window_std = np.nan
                feature_vector_t.extend([
                    np.nan_to_num(current_value),
                    np.nan_to_num(window_mean),
                    np.nan_to_num(window_std),
                ])
            X_run_features_list.append(feature_vector_t)

        df_column_names = []
        for name in feature_names_generated_base:
            df_column_names.append(name + "_t")
            df_column_names.append(name + "_win_mean")
            df_column_names.append(name + "_win_std")

        X_df = pd.DataFrame(X_run_features_list, columns=df_column_names)
        return X_df

    def _calculate_final_diagnostics(self):
        # Simplified Savitzky-Golay filter
        def smooth(y, window_length, polyorder):
            if len(y) < window_length: return y # Not enough data to smooth
            return savgol_filter(np.nan_to_num(y), window_length, polyorder, mode='mirror')

        beta_sys = np.array(self.history['avg_beta_system'])
        fcrit_sys = np.array(self.history['avg_fcrit_system'])
        model_entropy_sys = np.array(self.history['model_entropy_system'])

        # Ensure arrays are not all NaNs and have sufficient length
        if np.all(np.isnan(beta_sys)) or len(beta_sys) < self.config.diag_smoothing_window: beta_sys_smooth = beta_sys
        else: beta_sys_smooth = smooth(beta_sys, self.config.diag_smoothing_window, self.config.diag_polyorder)
        
        if np.all(np.isnan(fcrit_sys)) or len(fcrit_sys) < self.config.diag_smoothing_window: fcrit_sys_smooth = fcrit_sys
        else: fcrit_sys_smooth = smooth(fcrit_sys, self.config.diag_smoothing_window, self.config.diag_polyorder)

        dot_beta = np.gradient(beta_sys_smooth) if len(beta_sys_smooth) > 1 else np.zeros_like(beta_sys_smooth)
        dot_fcrit = np.gradient(fcrit_sys_smooth) if len(fcrit_sys_smooth) > 1 else np.zeros_like(fcrit_sys_smooth)

        self.history['SpeedIndex'] = np.sqrt(dot_beta**2 + dot_fcrit**2)
        
        # CoupleIndex (rolling correlation)
        couple_idx_vals = [np.nan] * len(dot_beta) # Fill with NaNs initially
        if len(dot_beta) >= self.config.diag_couple_window:
            for i in range(self.config.diag_couple_window -1, len(dot_beta)):
                window_dot_beta = dot_beta[i - self.config.diag_couple_window + 1 : i + 1]
                window_dot_fcrit = dot_fcrit[i - self.config.diag_couple_window + 1 : i + 1]
                if np.std(window_dot_beta) > 1e-6 and np.std(window_dot_fcrit) > 1e-6: # Avoid division by zero
                    corr = np.corrcoef(window_dot_beta, window_dot_fcrit)[0, 1]
                    couple_idx_vals[i] = corr
                else:
                    couple_idx_vals[i] = 0.0 # Or NaN if preferred for no variance
        self.history['CoupleIndex'] = np.array(couple_idx_vals)

        # rhoE
        if self.entropy_baseline is None:
            baseline_entropies = model_entropy_sys[
                self.config.rhoE_baseline_period_start : self.config.rhoE_baseline_period_end
            ]
            self.entropy_baseline = (
                np.nanmedian(baseline_entropies)
                if len(baseline_entropies) > 0 and not np.all(np.isnan(baseline_entropies))
                else 1e-6
            )
            if self.entropy_baseline < 1e-6:
                self.entropy_baseline = 1e-6
            print(f"INFO: Calculated Entropy Baseline: {self.entropy_baseline:.4f}")

        self.history['rhoE'] = model_entropy_sys / self.entropy_baseline

        if self.ml_components_loaded:
            print("INFO (AIF SIM): Generating features for AIF-specific ML classification...")
            X_df_for_prediction = self._generate_features_for_ml_prediction()

            num_total_hist_steps = len(self.history['time'])
            prediction_start_index = AIF_SPECIFIC_BURN_IN_FOR_FEATURES + AIF_SPECIFIC_FEATURE_WINDOW_SIZE - 1

            if prediction_start_index >= num_total_hist_steps:
                print("Warning (AIF SIM): Not enough history steps to make any ML predictions.")
                self.history['algorithmic_phase'] = ['P0'] * num_total_hist_steps
            else:
                X_predictable_df = X_df_for_prediction.iloc[prediction_start_index:].copy()
                X_predictable_df_filled = X_predictable_df.fillna(0)

                if not X_predictable_df_filled.empty:
                    X_scaled_for_prediction = self.ml_scaler.transform(X_predictable_df_filled)
                    y_predicted_numeric = self.ml_classifier.predict(X_scaled_for_prediction)

                    algo_phases = ['P0'] * num_total_hist_steps
                    for i, pred_label in enumerate(y_predicted_numeric):
                        actual_hist_idx = prediction_start_index + i
                        if actual_hist_idx < num_total_hist_steps:
                            algo_phases[actual_hist_idx] = f"P{pred_label}"
                    self.history['algorithmic_phase'] = algo_phases
                    print(f"INFO (AIF SIM): ML phase classification complete. Example phases: {Counter(algo_phases)}")
                else:
                    print("Warning (AIF SIM): No valid feature rows for ML prediction after slicing/NaN handling.")
                    self.history['algorithmic_phase'] = ['P0'] * num_total_hist_steps
        else:
            # Fallback simple rule-based classification
            S = np.array(self.history['SpeedIndex'])
            C = np.array(self.history['CoupleIndex'])
            R = np.array(self.history['rhoE'])

            S_I_high_thresh = np.nanpercentile(S, 90)
            C_I_crit_thresh = 0.2
            rhoE_II_crit_thresh = 1.5  # High exploration excess
            S_III_low_thresh = np.nanpercentile(S, 25)
            C_III_low_thresh_abs = 0.1
            rhoE_III_thresh = 1.2

            phase_labels = []
            for i in range(len(S)):
                if i < self.config.t_shock:
                    phase = 'P4'
                elif (
                    S[i] >= S_I_high_thresh
                    and C[i] >= C_I_crit_thresh
                    and R[i] < rhoE_II_crit_thresh
                ):
                    phase = 'P1'
                elif R[i] >= rhoE_II_crit_thresh:
                    phase = 'P2'
                elif (
                    S[i] <= S_III_low_thresh
                    and abs(C[i]) <= C_III_low_thresh_abs
                    and R[i] <= rhoE_III_thresh
                ):
                    phase = 'P3'
                else:
                    phase = 'P4'
                phase_labels.append(phase)

            self.history['algorithmic_phase'] = phase_labels


    def run(self):
        start_time_sim = time.time()
        for t_step in range(self.config.num_steps):
            self.environment.step() # Update environment (e.g., apply shock)
            
            # Agent actions in random order
            current_agent_order = list(range(len(self.agents)))
            random.shuffle(current_agent_order)
            for agent_idx in current_agent_order:
                self.agents[agent_idx].step(self.environment)
            
            if self.config.enable_rebirth and (t_step + 1) % self.config.rebirth_check_interval == 0:
                self._attempt_rebirth()

            self._collect_macroscopic_data(t_step)
            if t_step % self.snapshot_interval == 0 or t_step in self.snapshot_times:
                self.agent_model_snapshots[t_step] = [
                    agent.internal_model_M for agent in self.agents if agent.is_active
                ]
            if (t_step + 1) % 100 == 0:
                num_active = self.history['active_agents_count'][-1]
                print(f"Sim Step: {t_step+1}/{self.config.num_steps}, Active Agents: {num_active}")
                if num_active == 0 and t_step > self.config.t_shock + 50 : # System died
                    print("All agents inactive post-shock. Ending simulation early.")
                    # Fill remaining history with NaNs or last values for consistent array lengths
                    for key in ['avg_PE_system', 'avg_beta_system', 'avg_fcrit_system', 'model_entropy_system', 'active_agents_count']:
                        while len(self.history[key]) < self.config.num_steps:
                            self.history[key].append(self.history[key][-1] if self.history[key] else np.nan)
                    while len(self.history['time']) < self.config.num_steps:
                        self.history['time'].append(self.history['time'][-1] + 1 if self.history['time'] else t_step + 1)
                    break


        self._calculate_final_diagnostics()
        self.history['agent_model_snapshots_data'] = self.agent_model_snapshots 
        end_time_sim = time.time()
        print(f"Simulation finished in {end_time_sim - start_time_sim:.2f} seconds.")

    def save_history(self, path):
        """Serialize the simulation history AND snapshots to a pickle file."""
        import pickle
        # Ensure snapshots are part of what's saved
        data_to_save = self.history 
        # If self.agent_model_snapshots is not already in self.history (it isn't by default)
        # you need to add it.
        # The change in run() above handles this if you call save_history AFTER run().

        # Alternatively, if save_history is called by replication_runner AFTER sim.run():
        # replication_runner.py should save sim.history AND sim.agent_model_snapshots
        # Or, more cleanly, AIFPhoenixSimulation.run() should ensure agent_model_snapshots
        # becomes a key in self.history before it finishes. The line added above in run() does this.
        with open(path, "wb") as f:
            pickle.dump(data_to_save, f) # Now data_to_save (which is self.history) includes snapshots
        print(f"Saved simulation history (including snapshots) to {path}")

    def plot_results(self):
        if not os.path.exists("results_aif_phoenix"): os.makedirs("results_aif_phoenix")
        
        fig, axs = plt.subplots(7, 1, figsize=(15, 25), sharex=True)
        time_ax = self.history['time']

        axs[0].plot(time_ax, self.history['avg_PE_system'], label='Avg System PE')
        axs[0].set_ylabel('Avg PE')
        axs[0].legend()
        axs[0].grid(True, linestyle=':')

        axs[1].plot(time_ax, self.history['avg_beta_system'], label='Avg System Beta (Precision Proxy)')
        axs[1].set_ylabel('Avg Beta')
        axs[1].legend()
        axs[1].grid(True, linestyle=':')

        axs[2].plot(time_ax, self.history['avg_fcrit_system'], label='Avg System Fcrit (Slack Proxy)')
        axs[2].set_ylabel('Avg Fcrit')
        axs[2].legend()
        axs[2].grid(True, linestyle=':')
        
        axs[3].plot(time_ax, self.history['active_agents_count'], label='Active Agents Count')
        axs[3].set_ylabel('Active Agents')
        axs[3].legend()
        axs[3].grid(True, linestyle=':')

        axs[4].plot(time_ax, self.history['SpeedIndex'], label='SpeedIndex')
        axs[4].set_ylabel('SpeedIndex')
        axs[4].legend()
        axs[4].grid(True, linestyle=':')

        axs[5].plot(time_ax, self.history['CoupleIndex'], label='CoupleIndex')
        axs[5].set_ylabel('CoupleIndex')
        axs[5].set_ylim(-1.1, 1.1)
        axs[5].legend()
        axs[5].grid(True, linestyle=':')

        ax_rhoE_twin = axs[6].twinx()
        axs[6].plot(time_ax, self.history['model_entropy_system'], label='Model Entropy (H(M))', color='teal', alpha=0.6)
        axs[6].set_ylabel('Model Entropy H(M)', color='teal')
        axs[6].tick_params(axis='y', labelcolor='teal')
        axs[6].legend(loc='upper left')
        
        ax_rhoE_twin.plot(time_ax, self.history['rhoE'], label='rhoE (Exploration Excess)', color='purple')
        ax_rhoE_twin.set_ylabel('rhoE', color='purple')
        ax_rhoE_twin.tick_params(axis='y', labelcolor='purple')
        ax_rhoE_twin.axhline(1.0, color='gray', linestyle='--', alpha=0.7, label='rhoE Baseline (1.0)')
        ax_rhoE_twin.axhline(2.0, color='red', linestyle=':', alpha=0.7, label='rhoE Flaring Threshold (2.0)')
        ax_rhoE_twin.legend(loc='upper right')
        axs[6].grid(True, linestyle=':')
        axs[6].set_xlabel('Time Steps')
        
        phase_colors = {
            'P1': 'lightcoral',
            'P2': 'lightsalmon',
            'P3': 'lightgreen',
            'P4': 'lightblue'
        }
        phase_labels = self.history.get('algorithmic_phase', [])
        if phase_labels:
            current_phase = phase_labels[0]
            start_idx = 0
            for idx, phase in enumerate(phase_labels[1:], 1):
                if phase != current_phase:
                    for ax_idx in range(len(axs)):
                        axs[ax_idx].axvspan(start_idx, idx, facecolor=phase_colors.get(current_phase, 'white'), alpha=0.2, zorder=-10)
                    start_idx = idx
                    current_phase = phase
            for ax_idx in range(len(axs)):
                axs[ax_idx].axvspan(start_idx, len(phase_labels), facecolor=phase_colors.get(current_phase, 'white'), alpha=0.2, zorder=-10)


        plt.suptitle(f'AIF-Phoenix Simulation Results ({self.config.study_id})', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(f"results_aif_phoenix/timeseries_{self.config.study_id}.png", dpi=350)
        plt.close(fig)
        print(f"Saved timeseries plot to results_aif_phoenix/timeseries_{self.config.study_id}.png")

        # 3D Diagnostic Plot
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        S_plot = np.nan_to_num(self.history['SpeedIndex'])
        C_plot = np.nan_to_num(self.history['CoupleIndex'])
        R_plot = np.nan_to_num(self.history['rhoE'])
        
        # Ensure arrays are not empty for plotting
        if len(S_plot) > 0 and len(C_plot) > 0 and len(R_plot) > 0:
            phase_colors = {
                'P1': 'lightcoral',
                'P2': 'lightsalmon',
                'P3': 'lightgreen',
                'P4': 'lightblue'
            }
            phase_labels = self.history.get('algorithmic_phase', [])
            for i in range(len(S_plot)-1):
                color = phase_colors.get(phase_labels[i], 'gray') if phase_labels else 'gray'
                ax_3d.plot([S_plot[i], S_plot[i+1]], [C_plot[i], C_plot[i+1]], [R_plot[i], R_plot[i+1]], color=color, linewidth=2)

            legend_handles = [plt.Line2D([0], [0], color=c, lw=4) for c in phase_colors.values()]
            ax_3d.legend(legend_handles, ['P1', 'P2', 'P3', 'P4'])


            ax_3d.set_xlabel('SpeedIndex')
            ax_3d.set_ylabel('CoupleIndex')
            ax_3d.set_zlabel('rhoE')
            ax_3d.set_xlim(np.nanmin(S_plot) if not np.all(np.isnan(S_plot)) else 0, np.nanmax(S_plot) if not np.all(np.isnan(S_plot)) else 1)
            ax_3d.set_ylim(-1.1, 1.1)
            ax_3d.set_zlim(np.nanmin(R_plot) if not np.all(np.isnan(R_plot)) else 0, np.nanmax(R_plot) if not np.all(np.isnan(R_plot)) else 2) # Adjust Zlim as needed
            ax_3d.set_title(r'Diagnostic Trajectory ($\mathcal{S}, \mathcal{C}, \rho_E$) - ' + self.config.study_id)
            plt.savefig(f"results_aif_phoenix/diagnostic_trajectory_3d_{self.config.study_id}.png", dpi=350)
        else:
            print("Skipping 3D plot due to empty diagnostic data.")
        plt.close(fig_3d)
        print(f"Saved 3D diagnostic plot to results_aif_phoenix/diagnostic_trajectory_3d_{self.config.study_id}.png")


        # Plot model distributions from snapshots if available
        if self.agent_model_snapshots:
            peak_rho_time = (
                np.argmax(self.history['rhoE'][self.config.t_shock:]) + self.config.t_shock
            )
            mid_p3_time = int((peak_rho_time + self.config.num_steps) / 2)
            times_desired = [
                max(0, self.config.t_shock - 10),
                self.config.t_shock + 10,
                peak_rho_time,
                mid_p3_time,
                self.config.num_steps - 10,
            ]
            times_desired = [min(max(0, t), self.config.num_steps - 1) for t in times_desired]

            available_times = sorted(self.agent_model_snapshots.keys())
            snapshot_times_sorted = []
            for td in times_desired:
                nearest = min(available_times, key=lambda x: abs(x - td))
                if nearest not in snapshot_times_sorted:
                    snapshot_times_sorted.append(nearest)

            fig_dist, axs_dist = plt.subplots(1, len(snapshot_times_sorted), figsize=(4*len(snapshot_times_sorted), 3.5), sharey=True)
            if len(snapshot_times_sorted) == 1:
                axs_dist = [axs_dist]

            for i, t_plot in enumerate(snapshot_times_sorted):
                models_at_t = self.agent_model_snapshots.get(t_plot, [])
                if not models_at_t:
                    axs_dist[i].text(0.5, 0.5, "No active agents", ha="center", va="center")
                else:
                    model_counts = Counter(models_at_t)
                    labels = sorted(self.config.possible_env_states)
                    counts = [model_counts.get(s, 0) for s in labels]
                    axs_dist[i].bar([str(s) for s in labels], counts)
                    axs_dist[i].set_xlabel("Agent Model State")
                    if i == 0:
                        axs_dist[i].set_ylabel("Number of Agents")

                phase_label = "N/A"
                if 'algorithmic_phase' in self.history and t_plot < len(self.history['algorithmic_phase']):
                    phase_label = self.history['algorithmic_phase'][t_plot]
                axs_dist[i].set_title(f"t={t_plot}\n{phase_label}")

            plt.suptitle(f"Agent Model Distributions - {self.config.study_id}")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(f"results_aif_phoenix/model_distributions_{self.config.study_id}.png", dpi=350)
            plt.close(fig_dist)
            print(f"Saved model distributions plot to results_aif_phoenix/model_distributions_{self.config.study_id}.png")

        # Environment state evolution plot as fallback
        if len(self.config.possible_env_states) <= 10:
            env_states_over_time = []
            temp_env_for_plot = AIFGridEnvironment(self.config)
            for t_idx in range(self.config.num_steps):
                env_states_over_time.append(temp_env_for_plot.get_true_state(0, 0))
                temp_env_for_plot.step()

            fig_env, ax_env = plt.subplots(1, 1, figsize=(15, 2))
            ax_env.plot(self.history['time'], env_states_over_time, label="True Env State (example cell)", drawstyle="steps-post")
            ax_env.set_yticks(self.config.possible_env_states)
            ax_env.set_ylabel("Environment State")
            ax_env.set_xlabel("Time Steps")
            ax_env.legend()
            ax_env.set_title(f"Environment State Evolution - {self.config.study_id}")
            plt.savefig(f"results_aif_phoenix/env_state_evolution_{self.config.study_id}.png", dpi=350)
            plt.close(fig_env)
            print(f"Saved environment state plot to results_aif_phoenix/env_state_evolution_{self.config.study_id}.png")


# --- Main Execution ---
if __name__ == "__main__":
    config = EnvironmentConfigAIF(study_id="aif_phoenix_run_1")
    simulation = AIFPhoenixSimulation(config)
    simulation.run()
    simulation.plot_results()

    print("\n--- Falsification/Verification Hypotheses Check (Qualitative based on plots) ---")
    print("Hypothesis 1 (Macroscopic Phoenix Loop Emergence):")
    print("  Examine timeseries plots: Do avg_beta, avg_fcrit, model_entropy/rhoE show patterns")
    print("  consistent with Disintegration (post-shock drop), Flaring (high rhoE, beta low/variable),")
    print("  Pruning (rhoE decreasing, beta/fcrit recovering), Restabilization (stable low PE, rhoE near 1)?")
    print("  Examine 3D trajectory: Does it show a characteristic loop with high rhoE excursion?")
    
    print("\nHypothesis 2 (Agent Adaptation -> Macro Signatures):")
    print("  High rhoE (macro) should correspond to high diversity in agent models (visualize model distributions if possible).")
    print("  Macro avg_beta should reflect dominant agent beta states (e.g., if most agents are in low beta mode).")
    
    print("\nConsider if results align with or contradict these. Quantitative correlation would be next step.")