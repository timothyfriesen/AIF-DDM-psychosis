# Scripts for fit models to behavioural data

### Contents
- ```fit_model.py``` – Python script to fit a given model to a two-step task dataset

## How to run

For each of the models you would like to fit, run the script ```fit_model.py```. Note that to fit each model you must first change the following variables accordingly in the script:
   ```bash
   
"""-----------SELECT MODEL CLASS AND SET FREE PARAMETER VALUE RANGES----------"""

model = "RL_ddm_biased" # RL, RL_ddm,RL_ddm_biased, AI, or AI_ddm
mtype = 3 # 0, 1, 2 or 3 (only relevant if model = AI or AI_ddm)
drmtype = "linear" # Drift rate model: linear, sigmoid, sigmoid_single_v_mod, sigmoid_single_v_max
n_starts = 35 # Number of random starting parameter values in parameter fitting.
dataset = 'magic_carpet_2020' # Dataset name
optimizer = 'L-BFGS-B' # Nelder-Mead,L-BFGS-B or DE
```

After running this script, a subfolder with the dataset name will be created inside the ```fitted_parameters``` directory. Within this subfolder, each fitted model will generate a ```.csv``` file containing the best-fitting parameters for each participant, as well as the corresponding Negative Log-Likelihood (NLL) values. These NLLs can then be used for model comparison with the scripts located in the ```model_comparison``` folder in the root directory.

   

