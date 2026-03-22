# AIF-DDM: Psychosis Spectrum Simulation and Parameter Recovery

This repository contains code to reproduce and extend the results presented in the paper [Cognitive Effort in the Two-Step Task: An Active Inference Drift-Diffusion Model Approach](https://arxiv.org/abs/2508.04435), accepted at [IWAI2025](https://iwaiworkshop.github.io/).

This code is a modified version of the code authored by Dr. Alvaro Garrido Pérez. The original repository can be found [here](https://github.com/decide-ugent/aif-ddm). The original code is itself a modified version of code authored by Dr. Sam Gijsen for the paper [Active inference in the two-step task](https://www.nature.com/articles/s41598-022-21766-4), available [here](https://github.com/SamGijsen/AI2step).

---

## Overview of Extensions

This repository extends the original AIF-DDM framework in the following ways:

- **Psychosis spectrum simulation**: Synthetic behavioral datasets were generated for healthy control (HC) and psychosis spectrum patient populations by systematically varying AIF-DDM parameters in line with theoretical accounts of psychosis (Adams et al., 2013; Schlagenhauf et al., 2014). Specifically, learning rate (lr) was decreased, policy precision (kappa_a) was reduced, reward prior (prior_r) was increased, and epistemic drive (vunsamp) was elevated in the patient group relative to healthy controls.

- **Parameter recovery analysis**: Both the AIF-DDM and RL-DDM frameworks were fitted to the simulated data and parameter recovery was assessed using Pearson correlation, estimation bias, and RMSE for each parameter separately, as well as broken down by group.

- **Group difference analysis**: Linear mixed effects models were used to compare fitted parameter distributions between simulated HC and psychosis spectrum groups, assessing whether theoretically motivated parameter perturbations remained statistically detectable after model fitting.

- **Classification analysis**: A logistic regression classifier was trained on fitted AIF-DDM parameters to evaluate the multivariate discriminative validity of the parameter space for group membership prediction, assessed via 5-fold cross-validation.

- **Neural substrate visualization**: An interactive 3D brain visualization mapping each AIF-DDM parameter to its proposed neurobiological substrate, with predicted direction of change in psychosis, was generated using nilearn.

---

## TL;DR

- Simulate behavioral data from HC and psychosis spectrum populations using theoretically motivated parameter perturbations
- Fit AIF-DDM and RL-DDM models to simulated data
- Compare models based on AIC and BIC scores
- Perform model and parameter recovery analysis
- Analyse group differences in fitted parameters
- Classify group membership from fitted parameters
- Visualise neural substrates of AIF-DDM parameters

---

## Repository Content

```text
.
├─ exploratory_data_analysis/    # Scripts to perform exploratory data analysis
├─ model_comparison/             # Metrics and routines to compare fitted models
├─ model_fitting/                # Scripts/pipelines to fit models to data (uses MLE.py)
├─ model_recovery/               # Scripts for model recovery analysis
├─ parameter_recovery/           # Scripts for parameter recovery analysis
├─ test_model_predictions/       # Generate & plot model predictions vs. empirical behavior
├─ utils/                        # Utilities and helper functions
├─ analysis.ipynb                # Extended analysis: group differences, classification,
│                                #   parameter recovery by group, neural visualization
├─ MLE.py                        # Maximum-likelihood fitting utilities
├─ models.py                     # Model definitions and likelihoods
├─ LICENSE
└─ README.md
```

---

## Setup

1. **Create the environment**

    Before running any code, set up the conda environment using the provided environment file:

    ```bash
    conda env create -f environment.yml
    conda activate hssm_env
    ```

2. **Install additional dependencies for extended analyses**

    The psychosis simulation and visualization analyses require the following additional packages:

    ```bash
    pip install nilearn statsmodels scikit-learn
    ```

3. **Prepare the dataset**

    This repository does not include datasets.

    To use your own dataset, update the relevant file paths accordingly.

    To reproduce the results from the [original paper](https://arxiv.org/abs/2508.04435), download the 'magic carpet' dataset provided by Dr. Feher da Silva from this [repository](https://github.com/carolfs/muddled_models/tree/master/results/magic_carpet).

    Place the dataset in a subdirectory named:

    ```text
    two_step_task_datasets/magic_carpet_2020_dataset/
    ```

4. **Directory structure**

    Ensure the dataset folder is located alongside this repository:

    ```text
    .
    ├─ two_step_task_datasets/
    │   └─ magic_carpet_2020_dataset/
    ├─ aif-ddm/
    ```

---

## Simulation Parameters

The following parameter values were used to generate synthetic HC and psychosis spectrum datasets:

| Parameter | HC | Psychosis Spectrum | Theoretical Basis |
|---|---|---|---|
| Learning rate (lr) | 1.5 | 0.8 | Reduced prediction error signalling (Adams et al., 2013; Schlagenhauf et al., 2014) |
| Policy precision (kappa_a) | 3.0 | 1.2 | Aberrant precision weighting (Adams et al., 2013) |
| Reward prior (prior_r) | 0.5 | 0.7 | Elevated prior beliefs in psychosis (Friesen et al., 2025; Benrimoh et al., 2024) |
| Epistemic drive (vunsamp) | 0.3 | 0.6 | Disrupted directed exploration (Chen et al., 2025; Katthagen et al., 2022) |

---

## Interactive Visualization

An interactive 3D visualization of the proposed neural substrates of AIF-DDM parameters and their predicted direction of change in psychosis is available at: [link to your OSF/GitHub Pages URL here]

---

## Citation

If you use this extended code in your research, please cite both the original work and this extension:

**Original AIF-DDM paper:**
```bibtex
@inproceedings{Perez2025CognitiveEI,
  title={Cognitive Effort in the Two-Step Task: An Active Inference Drift-Diffusion Model Approach},
  author={Alvaro Garrido Perez and Viktor Lemoine and Amrapali Pednekar and Yara Khaluf and Pieter Simoens},
  year={2025},
  url={https://api.semanticscholar.org/CorpusID:280536605}
}
```

**Original AIF two-step task paper:**
```bibtex
@article{gijsen2022active,
  title={Active inference and the two-step task},
  author={Gijsen, Sam and Grundei, Miro and Blankenburg, Felix},
  journal={Scientific Reports},
  volume={12},
  number={1},
  pages={17682},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
```

## Contact

For questions about the original codebase, please contact:
`alvaro.garridoperez (at) ugent.be`

For questions about the psychosis spectrum extensions, please contact:
`timothy.friesen@mail.mcgill.ca`
