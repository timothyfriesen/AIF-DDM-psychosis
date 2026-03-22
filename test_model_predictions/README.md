# Scripts to compare observed behaviour with model predictions
### Contents
- ```simulation_analysis.ipynb``` – Jupyter notebook to generate simulations and compare these with observed behaviour.

## How to run

For each of the models you would like to test, run the script ```simulation_analysis.ipynb```. Note that to test each model you must first change the following variables accordingly in the script:
```bash
   
model = "RL_ddm_biased" # RL, RL_ddm,RL_ddm_biased, AI, or AI_ddm
mtype = 3 # 0, 1, 2 or 3 (only relevant if model = AI or AI_ddm)
drmtype = "linear" # Drift rate model: linear, sigmoid, sigmoid_single_v_mod, sigmoid_single_v_max
n_starts = 35 # Number of random starting parameter values in parameter fitting.
dataset = 'magic_carpet_2020' # Name of the dataset
```

After running this script, a subfolder with the dataset name will be created inside a directory called ```simulation_results```. Within this subfolder, each fitted model will generate a ```.csv``` file containing the the results of the simulations. By running this script you also visualise the results of the analysis.
