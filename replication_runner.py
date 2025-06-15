# replication_runner.py

import os
import numpy as np
from collections import Counter
import pickle

import matplotlib.pyplot as plt
# Set consistent font sizes across all plots
plt.rcParams.update({'font.size': 16,
                     'xtick.labelsize': 12,
                     'ytick.labelsize': 12})

from aif_phoenix_sim import EnvironmentConfigAIF, AIFPhoenixSimulation


def compute_phase_durations(history, t_shock):
    phases = history.get("algorithmic_phase", [])
    durations = {p: 0 for p in ["P1", "P2", "P3", "P4"]}
    for phase in phases[t_shock:]:
        if phase in durations:
            durations[phase] += 1
    return durations


def _mean_sem(arrays):
    arr = np.array(arrays)
    mean = np.nanmean(arr, axis=0)
    sem = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
    return mean, sem


def _modal_phases(phase_lists):
    transposed = list(zip(*phase_lists))
    modal = []
    for phases in transposed:
        counts = Counter(phases)
        modal.append(max(counts.items(), key=lambda x: x[1])[0])
    return modal


def plot_averaged_history(histories, output_prefix="avg"):
    if not histories:
        return

    keys = [
        "avg_PE_system",
        "avg_beta_system",
        "avg_fcrit_system",
        "active_agents_count",
        "SpeedIndex",
        "CoupleIndex",
        "model_entropy_system",
        "rhoE",
    ]

    data = {k: [h[k] for h in histories] for k in keys}
    means = {}
    sems = {}
    for k in keys:
        means[k], sems[k] = _mean_sem(data[k])

    modal_phase = _modal_phases([h["algorithmic_phase"] for h in histories])
    t = histories[0]["time"]

    fig, axs = plt.subplots(7, 1, figsize=(15, 25), sharex=True)

    axs[0].plot(t, means["avg_PE_system"], label="Avg PE")
    axs[0].fill_between(t, means["avg_PE_system"] - sems["avg_PE_system"], means["avg_PE_system"] + sems["avg_PE_system"], color="gray", alpha=0.3)
    axs[0].set_ylabel("Avg PE")
    axs[0].legend()

    axs[1].plot(t, means["avg_beta_system"], label="Avg Beta")
    axs[1].fill_between(t, means["avg_beta_system"] - sems["avg_beta_system"], means["avg_beta_system"] + sems["avg_beta_system"], color="gray", alpha=0.3)
    axs[1].set_ylabel("Avg Beta")
    axs[1].legend()

    axs[2].plot(t, means["avg_fcrit_system"], label="Avg Fcrit")
    axs[2].fill_between(t, means["avg_fcrit_system"] - sems["avg_fcrit_system"], means["avg_fcrit_system"] + sems["avg_fcrit_system"], color="gray", alpha=0.3)
    axs[2].set_ylabel("Avg Fcrit")
    axs[2].legend()

    axs[3].plot(t, means["active_agents_count"], label="Active Agents")
    axs[3].fill_between(t, means["active_agents_count"] - sems["active_agents_count"], means["active_agents_count"] + sems["active_agents_count"], color="gray", alpha=0.3)
    axs[3].set_ylabel("Active Agents")
    axs[3].legend()

    axs[4].plot(t, means["SpeedIndex"], label="SpeedIndex")
    axs[4].fill_between(t, means["SpeedIndex"] - sems["SpeedIndex"], means["SpeedIndex"] + sems["SpeedIndex"], color="gray", alpha=0.3)
    axs[4].set_ylabel("SpeedIndex")
    axs[4].legend()

    axs[5].plot(t, means["CoupleIndex"], label="CoupleIndex")
    axs[5].fill_between(t, means["CoupleIndex"] - sems["CoupleIndex"], means["CoupleIndex"] + sems["CoupleIndex"], color="gray", alpha=0.3)
    axs[5].set_ylabel("CoupleIndex")
    axs[5].set_ylim(-1.1, 1.1)
    axs[5].legend()

    ax_rho = axs[6].twinx()
    axs[6].plot(t, means["model_entropy_system"], label="H(M)", color="teal", alpha=0.6)
    axs[6].fill_between(t, means["model_entropy_system"] - sems["model_entropy_system"], means["model_entropy_system"] + sems["model_entropy_system"], color="teal", alpha=0.2)
    axs[6].set_ylabel("Model Entropy", color="teal")
    axs[6].tick_params(axis="y", labelcolor="teal")
    axs[6].legend(loc="upper left")

    ax_rho.plot(t, means["rhoE"], label="rhoE", color="purple")
    ax_rho.fill_between(t, means["rhoE"] - sems["rhoE"], means["rhoE"] + sems["rhoE"], color="purple", alpha=0.2)
    ax_rho.set_ylabel("rhoE", color="purple")
    ax_rho.tick_params(axis="y", labelcolor="purple")
    ax_rho.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
    ax_rho.legend(loc="upper right")

    phase_colors = {
        "P1": "lightcoral",
        "P2": "lightsalmon",
        "P3": "lightgreen",
        "P4": "lightblue",
    }
    current_phase = modal_phase[0]
    start_idx = 0
    for idx, ph in enumerate(modal_phase[1:], 1):
        if ph != current_phase:
            for ax in axs:
                ax.axvspan(start_idx, idx, facecolor=phase_colors.get(current_phase, "white"), alpha=0.2, zorder=-10)
            start_idx = idx
            current_phase = ph
    for ax in axs:
        ax.axvspan(start_idx, len(modal_phase), facecolor=phase_colors.get(current_phase, "white"), alpha=0.2, zorder=-10)

    axs[6].set_xlabel("Time Steps")
    plt.suptitle("AIF-Phoenix Replication Mean Trajectories")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"results_aif_phoenix/{output_prefix}_timeseries.png", dpi=350)
    plt.close(fig)


def run_replications(n_runs=10, base_seed=42, base_id="aif_phoenix"):
    os.makedirs("results_aif_phoenix", exist_ok=True)
    all_durations = []
    all_histories = []
    for i in range(n_runs):
        run_id = i + 1
        config = EnvironmentConfigAIF(study_id=f"{base_id}_run_{run_id}")
        config.random_seed = base_seed + i
        sim = AIFPhoenixSimulation(config)
        sim.run()
        sim.plot_results()
        history_path = f"results_aif_phoenix/history_{config.study_id}.pkl"
        sim.save_history(history_path)
        all_durations.append(compute_phase_durations(sim.history, config.t_shock))
        all_histories.append(sim.history)

    # Summary statistics of phase durations
    phases = ["P1", "P2", "P3", "P4"]
    summary_lines = []
    for p in phases:
        vals = np.array([d[p] for d in all_durations])
        summary_lines.append(f"{p}: mean={vals.mean():.2f} std={vals.std():.2f}")
    with open("results_aif_phoenix/replication_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print("\nReplication summary:\n" + "\n".join(summary_lines))

    plot_averaged_history(all_histories, output_prefix=base_id)


if __name__ == "__main__":
    run_replications(n_runs=10)
