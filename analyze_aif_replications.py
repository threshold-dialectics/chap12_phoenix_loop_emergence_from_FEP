# analyze_aif_replications.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json
import datetime
import hashlib
from collections import Counter
from scipy.stats import entropy as shannon_entropy # <-- IMPORTED SHANNON ENTROPY
from scipy.stats import shapiro, levene, f_oneway, kruskal, pearsonr
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison # For Tukey's HSD (optional post-hoc)
# If using Dunn's test, you might need:
# import scikit_posthocs as sp # pip install scikit-posthocs
import scikit_posthocs as sp # Add to top imports
# --- Configuration ---
N_RUNS = 10
HISTORY_DIR = "results_aif_phoenix"
BASE_ID = "aif_phoenix" # Used in filenames by replication_runner
T_SHOCK = 150 # Must match the t_shock used in simulations
OUTPUT_DIR = "results_aif_phoenix/quantitative_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Diagnostics to analyze per phase
DIAGNOSTICS_TO_ANALYZE = [
    'SpeedIndex', 'CoupleIndex', 'rhoE',
    'avg_PE_system', 'avg_beta_system', 'avg_fcrit_system'
]
PHASE_ORDER = ['P1', 'P2', 'P3', 'P4'] # For consistent plotting and tables

# --- Helper Functions ---
def load_all_histories(n_runs, history_dir, base_id):
    """Loads all history.pkl files from the replication runs."""
    all_histories_loaded = []
    print(f"Loading {n_runs} simulation histories...")
    for i in range(n_runs):
        run_id_num = i + 1
        history_file = os.path.join(history_dir, f"history_{base_id}_run_{run_id_num}.pkl")
        try:
            with open(history_file, 'rb') as f:
                saved_data = pickle.load(f)
            
            history_data = None
            config_snapshot = None

            if isinstance(saved_data, dict) and 'history_data' in saved_data and 'config_snapshot' in saved_data:
                history_data = saved_data['history_data']
                config_snapshot = saved_data['config_snapshot']
                print(f"  Successfully loaded history and config snapshot for run_{run_id_num}")
            elif isinstance(saved_data, dict) and 'time' in saved_data: # Likely old format (just history dict)
                history_data = saved_data
                config_snapshot = None # Mark as missing
                print(f"  Loaded history for run_{run_id_num} in old format (no config snapshot).")
            else:
                print(f"  ERROR: Unrecognized format in pickle file for run_{run_id_num}.")
                continue

            if history_data:
                history_data['run_identifier'] = f"run_{run_id_num}" # Keep this
                history_data['config_snapshot_for_analysis'] = config_snapshot # Store it (can be None)
                all_histories_loaded.append(history_data)

        except FileNotFoundError:
            print(f"  ERROR: History file not found for run_{run_id_num} at {history_file}.")
        except Exception as e:
            print(f"  ERROR: Could not load or process file for run_{run_id_num}: {e}")
            
    if not all_histories_loaded:
        raise ValueError("No history files successfully loaded. Ensure paths and filenames are correct and files are valid pickles.")
    return all_histories_loaded

def calculate_shannon_entropy_of_snapshot(model_snapshot, possible_states):
    """Calculates Shannon entropy for a single agent model snapshot."""
    if not model_snapshot:
        return 0.0
    model_counts = Counter(model_snapshot)
    all_states_counts = [model_counts.get(s, 0) for s in possible_states]
    total_agents_in_snapshot = sum(all_states_counts)
    if total_agents_in_snapshot == 0:
        return 0.0
    probabilities = [count / total_agents_in_snapshot for count in all_states_counts if count > 0]
    return shannon_entropy(probabilities, base=2) if probabilities else 0.0


# -------- MINIMAL JSON WRITER -------------------------------------------
def np2py(x):
    """Helper to convert NumPy types for json.dump."""
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    return x


def new_summary(all_histories, pooled_phase_stats, clf_stats,
                corr_stats, possible_env_states_for_entropy):
    """Create a compact summary dictionary for the analysis."""
    summary = {
        "meta": {
            "n_runs": len(all_histories),
            "t_shock": T_SHOCK,
            "generated_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + 'Z',
            "code_hash": hashlib.md5(open(__file__, 'rb').read()).hexdigest()[:8]
        },
        "global": {**clf_stats, **pooled_phase_stats, "HM_vs_rhoE": corr_stats},
        "runs": {}
    }

    for h in all_histories:
        rid = h["run_identifier"]
        algo = np.array(h["algorithmic_phase"])
        rhoE = np.array(h["rhoE"])
        S = np.array(h["SpeedIndex"])
        beta = np.array(h["avg_beta_system"])
        C = np.array(h["CoupleIndex"])

        dur = {p: int(np.sum(algo[T_SHOCK:] == p)) for p in ["P1", "P2", "P3", "P4"]}

        extrema = {
            "rhoE": {"max": float(np.nanmax(rhoE)), "t": int(np.nanargmax(rhoE))},
            "SpeedIndex": {"max": float(np.nanmax(S)), "t": int(np.nanargmax(S))},
            "avg_beta_min": {"min": float(np.nanmin(beta)), "t": int(np.nanargmin(beta))}
        }

        centroids = {}
        for p in ["P1", "P2", "P3", "P4"]:
            idx = np.where(algo == p)[0]
            if idx.size:
                centroids[p] = [
                    float(np.nanmean(S[idx])),
                    float(np.nanmean(C[idx])),
                    float(np.nanmean(rhoE[idx]))
                ]

        snaps = h.get("agent_model_snapshots_data", {})
        peak_H = 0.0
        phase_H = {p: [] for p in ["P1", "P2", "P3", "P4"]}
        for t_snap, models in snaps.items():
            H_val = calculate_shannon_entropy_of_snapshot(
                models, possible_env_states_for_entropy
            )
            peak_H = max(peak_H, H_val)
            p_here = algo[int(t_snap)] if int(t_snap) < len(algo) else None
            if p_here in phase_H:
                phase_H[p_here].append(H_val)
        phase_H_mean = {p: float(np.mean(v)) if v else None for p, v in phase_H.items()}

        summary["runs"][rid] = {
            "phase_durations": dur,
            "diagnostic_extrema": extrema,
            "phase_centroids": centroids,
            "diversity": {
                "peak_HM": float(peak_H),
                "H_mean_per_phase": phase_H_mean
            }
        }
    return summary



# --- Main Analysis Logic ---
def perform_quantitative_analysis():
    print("\n--- Starting Quantitative Analysis of AIF Replications ---")
    all_histories = load_all_histories(N_RUNS, HISTORY_DIR, BASE_ID)

    # Compute phase durations per run for later pooled stats
    phase_duration_records = []
    for hist in all_histories:
        algo = hist.get('algorithmic_phase', [])
        durations = {p: 0 for p in PHASE_ORDER}
        for ph in algo[T_SHOCK:]:
            if ph in durations:
                durations[ph] += 1
        durations['run_identifier'] = hist['run_identifier']
        phase_duration_records.append(durations)
    df_phase_durations = pd.DataFrame(phase_duration_records)

    # Load classifier metrics produced during training if available
    metrics_path = os.path.join('results_aif_phoenix', 'classifier_metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics_json = json.load(f)
            accuracy = float(metrics_json.get('accuracy', float('nan')))
            report_dict = {
                'macro avg': {'f1-score': metrics_json.get('macro_f1', float('nan'))}
            }
            per_class_raw = metrics_json.get('per_class', {})
            for phase, stats in per_class_raw.items():
                report_dict[f'Phase {phase[-1]}'] = {
                    'precision': stats.get('P', float('nan')),
                    'recall': stats.get('R', float('nan')),
                    'f1-score': stats.get('F1', float('nan'))
                }
            cm = np.array(metrics_json.get('confusion', [[0]*4]*4))
            print(f"Loaded classifier metrics from {metrics_path}")
        except Exception as e:
            print(f"Warning: Could not load classifier metrics: {e}")
            accuracy = float('nan')
            report_dict = {"macro avg": {"f1-score": float('nan')}}
            cm = np.zeros((4, 4), dtype=int)
    else:
        accuracy = float('nan')
        report_dict = {"macro avg": {"f1-score": float('nan')}}
        cm = np.zeros((4, 4), dtype=int)
    
    # Get possible_env_states more reliably
    config_snapshot_first_run = all_histories[0].get('config_snapshot_for_analysis')
    if config_snapshot_first_run and 'possible_env_states' in config_snapshot_first_run:
        possible_env_states_for_entropy = config_snapshot_first_run['possible_env_states']
        print(f"Using possible_env_states from loaded config: {possible_env_states_for_entropy}")
    else:
        print("Warning: 'possible_env_states' not found in loaded history config. Using default [0,1,2].")
        possible_env_states_for_entropy = [0, 1, 2]


    # --- Part 1: Diagnostic Statistics per ML-Classified Phase ---
    print("\n1. Analyzing Diagnostic Statistics per ML-Classified Phase...")
    phase_data = {phase: {diag: [] for diag in DIAGNOSTICS_TO_ANALYZE} for phase in PHASE_ORDER}

    for history in all_histories:
        ml_phases_raw = history.get('algorithmic_phase', []) # These are like "P0", "P1", "P2" etc.
        if not ml_phases_raw:
            print(f"Warning: No 'algorithmic_phase' found in history for {history.get('run_identifier', 'Unknown Run')}. Skipping this run for phase analysis.")
            continue
        
        # Filter out "P0" or other non-standard phase labels before processing
        # and ensure we only consider post-shock data for phase-specific stats
        valid_phase_indices = []
        valid_phase_labels = []
        for i in range(T_SHOCK, len(ml_phases_raw)):
            if ml_phases_raw[i] in PHASE_ORDER: # Only P1, P2, P3, P4
                valid_phase_indices.append(i)
                valid_phase_labels.append(ml_phases_raw[i])

        if not valid_phase_labels:
            print(f"Warning: No valid P1-P4 phases found post-shock for {history.get('run_identifier', 'Unknown Run')}.")
            continue

        for diag_key in DIAGNOSTICS_TO_ANALYZE:
            diag_series_raw = history.get(diag_key)
            
            if diag_series_raw is None or not isinstance(diag_series_raw, (list, np.ndarray)) or len(diag_series_raw) == 0:
                print(f"Warning: Diagnostic '{diag_key}' not found, not a list/array, or empty in history for {history.get('run_identifier', 'Unknown Run')}.")
                continue 
            diag_series = np.array(diag_series_raw)
            
            for i, phase_label in zip(valid_phase_indices, valid_phase_labels):
                if i < len(diag_series) and not np.isnan(diag_series[i]):
                    phase_data[phase_label][diag_key].append(diag_series[i])

    # Create DataFrames for easier plotting and stats
    plot_data_list = []
    for phase_label_iter in PHASE_ORDER: # Iterate in defined order
        for diag_key_iter in DIAGNOSTICS_TO_ANALYZE:
            for value in phase_data[phase_label_iter][diag_key_iter]:
                plot_data_list.append({'Phase': phase_label_iter, 'Diagnostic': diag_key_iter, 'Value': value})
    
    df_plot_data = pd.DataFrame(plot_data_list)

    # Generate Box Plots
    print("  Generating Box Plots for Diagnostics per Phase...")
    for diag_key_plot in DIAGNOSTICS_TO_ANALYZE:
        plt.figure(figsize=(8, 6))
        # Filter data for the current diagnostic to avoid issues if some phases have no data for it
        current_diag_data = df_plot_data[df_plot_data['Diagnostic'] == diag_key_plot]
        if not current_diag_data.empty:
            sns.boxplot(x='Phase', y='Value', data=current_diag_data, order=PHASE_ORDER)
            plt.title(f'Distribution of {diag_key_plot} by Phoenix Loop Phase (ML Classified)')
            plt.ylabel(diag_key_plot)
            plt.xlabel('Phase')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"boxplot_{diag_key_plot}_per_phase.png"))
        else:
            print(f"    No data to plot for {diag_key_plot}")
        plt.close()
    print("  Box plots saved.")

    # Calculate and Print Descriptive Statistics Table
    print("\n  Calculating Descriptive Statistics per Phase...")
    desc_stats_list = []
    for phase_stat in PHASE_ORDER:
        for diag_key_stat in DIAGNOSTICS_TO_ANALYZE:
            values = np.array(phase_data[phase_stat][diag_key_stat])
            if len(values) > 0:
                desc_stats_list.append({
                    'Phase': phase_stat, 'Diagnostic': diag_key_stat, 'N': len(values),
                    'Mean': np.mean(values), 'Median': np.median(values),
                    'StdDev': np.std(values), '25th_Perc': np.percentile(values, 25),
                    '75th_Perc': np.percentile(values, 75)
                })
            else:
                 desc_stats_list.append({
                    'Phase': phase_stat, 'Diagnostic': diag_key_stat, 'N': 0,
                    'Mean': np.nan, 'Median': np.nan, 'StdDev': np.nan,
                    '25th_Perc': np.nan, '75th_Perc': np.nan
                })

    # Add phase duration descriptive stats
    for phase_stat in PHASE_ORDER:
        vals = df_phase_durations[phase_stat].dropna().to_numpy()
        desc_stats_list.append({
            'Phase': phase_stat,
            'Diagnostic': 'phase_duration',
            'N': len(vals),
            'Mean': np.mean(vals) if len(vals) else np.nan,
            'Median': np.median(vals) if len(vals) else np.nan,
            'StdDev': np.std(vals) if len(vals) else np.nan,
            '25th_Perc': np.percentile(vals, 25) if len(vals) else np.nan,
            '75th_Perc': np.percentile(vals, 75) if len(vals) else np.nan
        })
    df_desc_stats = pd.DataFrame(desc_stats_list)
    print("\nDescriptive Statistics for Diagnostics by ML-Classified Phase:\n", df_desc_stats)
    df_desc_stats.to_csv(os.path.join(OUTPUT_DIR, "diagnostic_stats_per_phase.csv"), index=False)
    print(f"  Descriptive statistics saved to {os.path.join(OUTPUT_DIR, 'diagnostic_stats_per_phase.csv')}")

    # Perform Statistical Tests (ANOVA/Kruskal-Wallis + Post-Hoc)
    # Perform Statistical Tests (ANOVA/Kruskal-Wallis + Post-Hoc)
    print("\n  Performing Statistical Tests for Differences Between Phases...")
    statistical_test_results = []
    for diag_key_test in DIAGNOSTICS_TO_ANALYZE:
        print(f"    Testing for {diag_key_test}:")
        
        data_for_test = []
        phase_labels_for_test_data = [] # To store labels corresponding to data_for_test
        for phase_test_label in PHASE_ORDER:
            phase_specific_data = np.array(phase_data[phase_test_label][diag_key_test])
            if len(phase_specific_data) > 0:
                data_for_test.append(phase_specific_data)
                phase_labels_for_test_data.append(phase_test_label)
        
        if len(data_for_test) < 2:
            print(f"      Not enough phase groups with data for {diag_key_test} to perform test.")
            statistical_test_results.append({'Diagnostic': diag_key_test, 'Test': 'Skipped (Too few groups)', 'P-value': np.nan, 'PostHoc': 'N/A'})
            continue
        
        # Check assumptions for ANOVA (normality, homogeneity of variances)
        # Shapiro-Wilk for normality (requires len >= 3)
        shapiro_ps = [shapiro(group).pvalue for group in data_for_test if len(group) >= 3]
        all_normal = all(p > 0.05 for p in shapiro_ps) if shapiro_ps else False

        homogeneity_p = np.nan
        if len(data_for_test) >=2 and all(len(g) >=2 for g in data_for_test):
            try:
                homogeneity_p = levene(*data_for_test).pvalue
            except ValueError: 
                print(f"      Levene's test failed for {diag_key_test}. Assuming non-homogeneous.")
                homogeneity_p = 0.0

        use_anova = all_normal and (np.isnan(homogeneity_p) or homogeneity_p > 0.05)

        
        posthoc_results_str = "Pending"
        test_name = "" # Initialize test_name
        p_value = np.nan # Initialize p_value

        try:
            if use_anova:
                stat, p_value = f_oneway(*data_for_test)
                test_name = "ANOVA"
                print(f"      {test_name} for {diag_key_test}: F-statistic = {stat:.2f}, p-value = {p_value:.4e}")
                if p_value < 0.05:
                    flat_data = np.concatenate(data_for_test)
                    group_labels_for_tukey = np.concatenate(
                        [ [phase_labels_for_test_data[i]] * len(data_for_test[i]) for i in range(len(data_for_test)) ]
                    )
                    mc = MultiComparison(flat_data, group_labels_for_tukey)
                    tukey_results = mc.tukeyhsd()
                    print(f"      Tukey's HSD Post-Hoc Test Results for {diag_key_test}:\n{tukey_results}\n")
                    posthoc_results_str = str(tukey_results)
                else: posthoc_results_str = "N/A (overall p >= 0.05)"
            else: 
                stat, p_value = kruskal(*data_for_test) # data_for_test is already a list of arrays
                test_name = "Kruskal-Wallis"
                print(f"      {test_name} for {diag_key_test}: statistic = {stat:.2f}, p-value = {p_value:.4e}")
                if p_value < 0.05:
                    # --- CORRECTED DUNN'S TEST CALL ---
                    if len(data_for_test) >= 2 : 
                        try:
                            # sp.posthoc_dunn takes a list of array-like objects for 'a'
                            posthoc_df = sp.posthoc_dunn(data_for_test, p_adjust='bonferroni')
                            # Set column and row names to the actual phase labels used in data_for_test
                            posthoc_df.columns = phase_labels_for_test_data
                            posthoc_df.index = phase_labels_for_test_data
                            print(f"      Dunn's Post-Hoc Test Results for {diag_key_test} (Bonferroni corrected):\n{posthoc_df}\n")
                            posthoc_results_str = posthoc_df.to_string()
                        except Exception as e_ph:
                            print(f"      Error during Dunn's test for {diag_key_test}: {e_ph}")
                            posthoc_results_str = f"Error in Dunn's test: {e_ph}"
                    else:
                        posthoc_results_str = "Skipped (insufficient groups for posthoc after filtering)"
                    # --- END CORRECTED DUNN'S TEST CALL ---
                else: posthoc_results_str = "N/A (overall p >= 0.05)"

            statistical_test_results.append({'Diagnostic': diag_key_test, 'Test': test_name, 'P-value': p_value, 'PostHoc': posthoc_results_str})

        except Exception as e: # Catch errors from f_oneway, kruskal, or general logic
            print(f"      Error during primary statistical test ({test_name if test_name else 'N/A'}) for {diag_key_test}: {e}")
            statistical_test_results.append({'Diagnostic': diag_key_test, 'Test': test_name if test_name else 'Error in main test', 'P-value': p_value, 'PostHoc': str(e)})

    df_stat_tests = pd.DataFrame(statistical_test_results)
    print("\nSummary of Statistical Test Results (Overall difference between phases):\n", df_stat_tests)
    df_stat_tests.to_csv(os.path.join(OUTPUT_DIR, "statistical_tests_summary.csv"), index=False)
    print(f"  Statistical test summary saved to {os.path.join(OUTPUT_DIR, 'statistical_tests_summary.csv')}")


    # --- Part 2: Agent Model Entropy vs. Macroscopic rhoE ---
    print("\n2. Analyzing Agent Model Entropy vs. Macroscopic rhoE...")
    all_snapshot_entropies_H = [] # Store actual H(M) from snapshots
    all_macro_rhoE_at_snapshots = []

    for run_idx, history in enumerate(all_histories):
        agent_snapshots = history.get('agent_model_snapshots_data', {})
        macro_rhoE_series = history.get('rhoE', [])
        # time_series = history.get('time', []) # Not strictly needed here if indexing by t_snapshot

        if not agent_snapshots or not isinstance(macro_rhoE_series, (list, np.ndarray)) or len(macro_rhoE_series) == 0:
            print(f"Warning: Missing snapshots or rhoE data for run {run_idx+1}. Skipping this run for entropy correlation.")
            continue

        for t_snapshot, model_list_at_t in agent_snapshots.items():
            # Ensure t_snapshot is a valid index for macro_rhoE_series
            # The snapshot times should align with the history time steps
            if 0 <= t_snapshot < len(macro_rhoE_series):
                snapshot_H = calculate_shannon_entropy_of_snapshot(model_list_at_t, possible_env_states_for_entropy)
                macro_rhoE_val = macro_rhoE_series[int(t_snapshot)]
                
                if not np.isnan(snapshot_H) and not np.isnan(macro_rhoE_val):
                    all_snapshot_entropies_H.append(snapshot_H)
                    all_macro_rhoE_at_snapshots.append(macro_rhoE_val)
            # else:
                # print(f"Warning: Snapshot time {t_snapshot} out of bounds for rhoE series (len {len(macro_rhoE_series)}) for run {run_idx+1}.")

    if all_snapshot_entropies_H and all_macro_rhoE_at_snapshots:
        plt.figure(figsize=(8, 6))
        plt.scatter(all_snapshot_entropies_H, all_macro_rhoE_at_snapshots, alpha=0.5)
        plt.xlabel("Snapshot Shannon Entropy H(M) of Agent Models")
        plt.ylabel("Macroscopic rhoE at Snapshot Time")
        plt.title("Correlation: Agent Model Entropy vs. Macroscopic rhoE")
        plt.grid(True, linestyle=':')
        
        if len(all_snapshot_entropies_H) > 1 and len(all_macro_rhoE_at_snapshots) > 1:
            corr_coef, p_val_corr = pearsonr(all_snapshot_entropies_H, all_macro_rhoE_at_snapshots)
            plt.text(0.05, 0.95, f'Pearson r = {corr_coef:.3f}\np-value = {p_val_corr:.3e}',
                     transform=plt.gca().transAxes, ha='left', va='top',
                     bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
            print(f"\n  Pearson correlation between Snapshot H(M) and Macroscopic rhoE: r={corr_coef:.4f}, p={p_val_corr:.4e}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "correlation_snapshot_entropy_vs_rhoE.png"))
        plt.close()
        print("  Correlation plot saved.")
    else:
        print("  Not enough data to generate Snapshot H(M) vs. rhoE correlation plot.")

    # --- Derive pooled diagnostic stats ---------------------------------
    GLOBAL_DIAG_KEYS = ['avg_beta_system','avg_PE_system','rhoE',
                        'SpeedIndex','CoupleIndex','avg_fcrit_system']
    pooled_diag = {k:{p:[] for p in PHASE_ORDER} for k in GLOBAL_DIAG_KEYS}

    for hist in all_histories:
        phases = np.array(hist.get('algorithmic_phase', []))
        for k in GLOBAL_DIAG_KEYS:
            series = np.array(hist.get(k, []))
            for p in PHASE_ORDER:
                pooled_diag[k][p].extend(series[phases==p])

    global_diag_summary = {}
    for k in GLOBAL_DIAG_KEYS:
        global_diag_summary[k] = {
            p:{
                "mean": float(np.nanmean(pooled_diag[k][p])) if pooled_diag[k][p] else float('nan'),
                "sd"  : float(np.nanstd (pooled_diag[k][p])) if pooled_diag[k][p] else float('nan')
            } for p in PHASE_ORDER
        }

    # ---- Build minimal summary JSON ----------------------------------
    pooled_phase_stats = {
        "phase_durations": {
            p: {
                "mean": df_desc_stats.query("Diagnostic=='phase_duration' & Phase==@p").Mean.astype(float).iat[0],
                "sd": df_desc_stats.query("Diagnostic=='phase_duration' & Phase==@p").StdDev.astype(float).iat[0],
            }
            for p in PHASE_ORDER
        }
    }

    corr_stats = {"r": float(corr_coef) if 'corr_coef' in locals() else float('nan'),
                  "p": float(p_val_corr) if 'p_val_corr' in locals() else float('nan')}

    clf_stats = {
        "classifier": {
            "accuracy": accuracy,
            "macro_f1": report_dict.get("macro avg", {}).get("f1-score", float('nan')),
            "per_class": {
                k.replace("Phase ", "P"): {
                    "P": v.get("precision", float('nan')),
                    "R": v.get("recall", float('nan')),
                    "F1": v.get("f1-score", float('nan')),
                }
                for k, v in report_dict.items() if k.startswith("Phase")
            },
            "confusion": cm.tolist() if hasattr(cm, 'tolist') else []
        }
    }

    summary_dict = new_summary(
        all_histories,
        pooled_phase_stats,
        clf_stats,
        corr_stats,
        possible_env_states_for_entropy,  # <-- NEW
    )

    summary_dict['global']['diagnostics'] = global_diag_summary
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary_dict, f, indent=2, default=np2py)
    print(f"Wrote minimal summary to {OUTPUT_DIR}/summary.json")

    print("\n--- Quantitative Analysis Complete ---")

if __name__ == "__main__":
    perform_quantitative_analysis()
