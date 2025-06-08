# Experiment 12B: Emergence of the Phoenix Loop in an Active Inference Agent-Based Model

This repository contains the complete Python code for Experiment 12B from the book *Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness* by Axel Pond.

The experiment demonstrates the emergence of the Phoenix Loop recovery phases and their characteristic diagnostic signatures from a bottom-up Agent-Based Model (ABM) grounded in Active Inference (AIF) principles. It shows how macroscopic patterns of systemic collapse and recovery can arise from the collective behavior of simple, FEP-driven agents operating under resource constraints.

## Table of Contents
- [Core Concepts](#core-concepts)
- [Repository Structure](#repository-structure)
- [Workflow: How to Reproduce the Experiment](#workflow-how-to-reproduce-the-experiment)
  - [Prerequisites](#prerequisites)
  - [Step 1: Generate Raw Simulation Data for Training](#step-1-generate-raw-simulation-data-for-training)
  - [Step 2: Train the Phoenix Loop Phase Classifier](#step-2-train-the-phoenix-loop-phase-classifier)
  - [Step 3: Run Replications with Algorithmic Classification](#step-3-run-replications-with-algorithmic-classification)
  - [Step 4: Perform Quantitative Analysis of Replications](#step-4-perform-quantitative-analysis-of-replications)
- [Key Results](#key-results)
- [Note on Other Scripts](#note-on-other-scripts)
- [Citation](#citation)

## Core Concepts

This experiment focuses on the **Phoenix Loop**, a conceptual model from *Threshold Dialectics* that describes the stereotyped sequence of phases a system often traverses during post-collapse recovery. The four phases are:

1.  **Phase I (Disintegration):** The initial, rapid breakdown of order following a systemic shock or breach of the system's tolerance capacity.
2.  **Phase II (Flaring):** A period of high uncertainty and broad exploration, where new strategies and structures are trialed. This phase is characteristically marked by a surge in **Exploration Entropy Excess ($\rho_E$)**.
3.  **Phase III (Pruning):** A consolidation phase where successful innovations are selected and reinforced, while failed experiments are abandoned.
4.  **Phase IV (Restabilization):** The system settles into a new, relatively stable operating regime.

The experiment uses TD diagnostics such as the **Speed Index ($\mathcal{S}$)** and **Couple Index ($\mathcal{C}$)** to track the system's structural drift, alongside $\rho_E$ to identify the crucial exploratory phase.

## Repository Structure

This repository contains the following key Python scripts:

-   "aif_phoenix_sim.py": The core Active Inference Agent-Based Model (AIF-ABM) simulation. It can be run for a single simulation and will load a trained ML classifier to identify Phoenix Loop phases in real-time if one is available.
-   "replication_runner.py": A script to run multiple replications of the "aif_phoenix_sim.py" simulation.
-   "train_aif_specific_classifier.py": Trains a Random Forest machine learning model to classify the Phoenix Loop phases using data from the AIF-ABM simulations and pre-defined manual labels.
-   "analyze_aif_replications.py": Performs a deep quantitative analysis of the collected replication data, generating summary statistics and plots.
-   "phoenix_loop_classifier_accuracy_ML.py": A script related to a different experiment (likely 11B) for training a classifier on a simpler System Dynamics model. It is not part of the primary workflow for Experiment 12B.

## Workflow: How to Reproduce the Experiment

Follow these steps to generate the data, train the model, and reproduce the analytical results presented in the book.

### Prerequisites

You will need Python 3.x and the following libraries. You can install them using the provided "requirements.txt" file:

"""bash
pip install -r requirements.txt
"""

**requirements.txt:**
"""
numpy
matplotlib
scipy
pandas
joblib
seaborn
scikit-learn
statsmodels
scikit-posthocs
"""

### Step 1: Generate Raw Simulation Data for Training

First, we need to generate the raw simulation data that will be used to train the machine learning classifier. The training script ("train_aif_specific_classifier.py") is hardcoded to use 10 simulation runs.

"""bash
python replication_runner.py
"""
This command will run 10 simulations by default. It will create a "results_aif_phoenix/" directory and populate it with "history_aif_phoenix_run_N.pkl" files and corresponding timeseries plots. At this stage, the simulations will not have algorithmically classified phases because the classifier has not been trained yet.

### Step 2: Train the Phoenix Loop Phase Classifier

Next, use the data generated in Step 1 to train the specific classifier for the AIF-ABM. This script uses manually defined phase labels (hardcoded within the script) as the ground truth for training.

"""bash
python train_aif_specific_classifier.py
"""

This will:
-   Load the 10 ".pkl" history files from "results_aif_phoenix/".
-   Train a Random Forest classifier and a feature scaler.
-   Save the trained model as "aif_phoenix_classifier_specific.joblib" and the scaler as "aif_phoenix_scaler_specific.joblib" in the root directory.
-   Generate a confusion matrix plot ("aif_specific_classifier_CM.png") and a performance report ("classifier_metrics.json") in the "results_aif_phoenix/" directory.

### Step 3: Run Replications with Algorithmic Classification

Now that a trained classifier exists, you should re-run the replications. This time, each simulation instance will load the trained model and include the algorithmically classified phases in its history log. This step is necessary to provide the analysis script with the required data.

"""bash
python replication_runner.py
"""

This will overwrite the previous ".pkl" files and plots, now with the "algorithmic_phase" data included.

### Step 4: Perform Quantitative Analysis of Replications

Finally, run the analysis script to process the results from the 10 classified simulation runs.

"""bash
python analyze_aif_replications.py
"""

This will:
-   Load the 10 history files.
-   Perform a detailed quantitative analysis of the diagnostics within each ML-classified phase.
-   Generate summary statistics ("diagnostic_stats_per_phase.csv", "summary.json") and analytical plots (e.g., "boxplot_rhoE_per_phase.png", "correlation_snapshot_entropy_vs_rhoE.png") in the "results_aif_phoenix/quantitative_analysis/" directory.

## Key Results

The experiment successfully demonstrates several key findings of *Threshold Dialectics*:

1.  **Emergence of the Phoenix Loop:** The AIF-ABM simulations robustly demonstrate the spontaneous emergence of the macroscopic Phoenix Loop phases from bottom-up, agent-level FEP-driven adaptation. The mean trajectories from the replications show the characteristic progression from Disintegration (P1) through Flaring (P2), Pruning (P3), and Restabilization (P4).

2.  **Diagnostic Signatures:** The TD diagnostics effectively capture the recovery process. The **Flaring (P2)** phase is clearly marked by a significant peak in **Exploration Entropy Excess ($\rho_E$)**, driven by the diversification of agent internal models as they explore the new environmental state.

3.  **High-Accuracy Phase Classification:** The machine learning classifier, trained on TD diagnostic features, achieves very high accuracy (**~97% on the test set**), demonstrating that the Phoenix Loop phases have distinct, learnable signatures. This confirms the validity of the phase distinctions and the diagnostic power of the TD framework.

4.  **Statistical Distinctiveness:** Quantitative analysis confirms that the diagnostics show statistically significant differences across the ML-classified phases. For example, the boxplots generated by "analyze_aif_replications.py" show that $\rho_E$ is significantly elevated during Phase II compared to all other phases.

These results provide strong empirical support for the Phoenix Loop as a general archetype of systemic recovery and validate the use of TD diagnostics for monitoring and understanding post-collapse dynamics.

## Note on Other Scripts

The script "phoenix_loop_classifier_accuracy_ML.py" is included in this repository but is **not part of the Experiment 12B workflow**. It pertains to Experiment 11B, which involves training a classifier on data from a simpler, top-down System Dynamics model rather than the AIF-ABM used here.

## Citation

If you use or refer to this code or the concepts from Threshold Dialectics, please cite the accompanying book:

@book{pond2025threshold,
  author    = {Axel Pond},
  title     = {Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness},
  year      = {2025},
  isbn      = {978-82-693862-2-6},
  publisher = {Amazon Kindle Direct Publishing},
  url       = {https://www.thresholddialectics.com},
  note      = {Code repository: \url{https://github.com/threshold-dialectics}}
}

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
