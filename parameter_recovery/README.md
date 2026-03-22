# Scripts for parameter recovery analysis

### Contents
- ```synthetic_data_generator.ipynb``` – Jupyter notebook to generate a simulated dataset for a given model
- ```fit_synthetic_data.py``` – Python script to fit a given model to the simulated dataset
- ```parameter_recovery_analysis.ipynb``` – Jupyter notebook to visualise parameter recovery results  

## How to run

1. Run the Jupyter notebook ```synthetic_data_generator.ipynb``` to generate a simulated dataset for each of the models you are testing. Each run will generate a subfolder with the corresponding simulated dataset inside a folder called ```param_recovery_data```. Note that you must change the appropriate variables every time you run to use a specific model:

   ```bash
   # Model class. Options: "RL", "RL_ddm", "RL_ddm_biased", "AI", or "AI_ddm".
   model = "RL_ddm_biased"
   
   # Subclass of Active Inference model (only used if model is "AI" or "AI_ddm").
   # Options: 0, 1, 2, or 3.
   mtype = 3
   
   # Drift rate model. Options: "linear", "sigmoid", "sigmoid_single_v_mod", "sigmoid_single_v_max".
   drmtype = "linear"
   
   # Dataset name.
   dataset = "magic_carpet_2020"
   ```

2. Inside this folder, for each of the models you would like to test, run the script ```fit_synthetic_data.py```. Note that you must change the following variables accordingly in the script:
   ```bash
  
    """---------------SELECT MODEL AND SET VARIABLES ---------------"""

    model = "AI_ddm" # RL, RL_ddm, RL_ddm_biased, AI, or AI_ddm
    mtype = 3 # 0, 1, 2 or 3 (only relevant if model = AI or AI_ddm)
    drmtype = "sigmoid_single_v_mod" # Drift rate model: linear, sigmoid, sigmoid_single_v_mod, sigmoid_single_v_max
    n_starts = 35 # Number of random starting parameter values in parameter fitting.
    dataset = 'magic_carpet_2020' # Dataset name 
   ```

3. Once all the simulated data has been fitted, you can visualize the parameter recovery results by running the ```parameter_recovery_analysis.ipynb``` Jupyter notebook. 
   

   