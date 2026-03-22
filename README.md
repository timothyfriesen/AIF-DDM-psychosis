# aif-ddm
This repository contains code to reproduce the results presented in the paper [Cognitive Effort in the Two-Step Task: An Active Inference Drift-Diffusion Model Approach](https://arxiv.org/abs/2508.04435), accepted at [IWAI2025](https://iwaiworkshop.github.io/).
This code is a modified version of the code authored by Dr. Sam Gijsen for the paper [Active inference in the two-step task](https://www.nature.com/articles/s41598-022-21766-4). The original code can be found [here](https://github.com/SamGijsen/AI2step). 


### TL;DR

- Fit models to data

- Compare models based on AIC and BIC scores

- Perform model and parameter recovery analysis

- Analyse and visualise results



### Repository content

```text
.
├─ exploratory_data_analysis/    # Scripts to perform exploratory data analysis
├─ model_comparison/             # Metrics and routines to compare fitted models
├─ model_fitting/                # Scripts/pipelines to fit models to data (uses MLE.py)
├─ model_recovery/               # Scripts for model recovery analysis
├─ parameter_recovery/           # Scripts for parameter recovery analysis
├─ test_model_predictions/       # Generate & plot model predictions vs. empirical behavior
├─ utils/                        # Utilities and helper functions
├─ MLE.py                        # Maximum-likelihood fitting utilities
├─ models.py                     # Model definitions and likelihoods
├─ LICENSE
└─ README.md
```

### Setup

1. **Create the environment**  

    Before running any code, set up the conda environment using the provided environment file:

    ```bash
    conda env create -f environment.yml
    conda activate hssm_env
    ```
2. **Prepare the dataset**
  
    This repository does not include datasets.

    To use your own dataset, update the relevant file paths accordingly.

    To reproduce the results from the [paper](https://arxiv.org/abs/2508.04435), download the 'magic carpet' dataset provided by Dr. Feher da Silva from this [repository](https://github.com/carolfs/muddled_models/tree/master/results/magic_carpet).

    Place the dataset in a subdirectory named:

    ```text
    two_step_task_datasets/magic_carpet_2020_dataset/
    ```

3. **Directory structure**
    Ensure the dataset folder is located alongside this repository, such that the structure looks like:
    ```text
    .
    ├─ two_step_task_datasets/   
    │   └─ magic_carpet_2020_dataset/
    ├─ aif-ddm/

    ```


### Citation

If you find this code useful in your research, please consider citing:

```python
@inproceedings{Perez2025CognitiveEI,
  title={Cognitive Effort in the Two-Step Task: An Active Inference Drift-Diffusion Model Approach},
  author={Alvaro Garrido Perez and Viktor Lemoine and Amrapali Pednekar and Yara Khaluf and Pieter Simoens},
  year={2025},
  url={https://api.semanticscholar.org/CorpusID:280536605}
}
```

Alternatively, if you only use the pure AIF/HRL models (without DDMs), you may also cite:

```python
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
### Contact 

If you have any questions, please feel free to reach out at:  
`alvaro.garridoperez (at) ugent.be`
