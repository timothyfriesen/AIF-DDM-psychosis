"""
Module for performing parameter fitting for synthetic datasets 
(parameter recovery analysis)

Author: Alvaro Garrido Perez <alvaro.garridoperez@ugent.be>
Date: 09-12-2025

"""




"""-------------IMPORT PACKAGES-------------"""

import numpy as np 
import csv 
import glob
import pandas as pd
import os
import sys
from tqdm import tqdm
from joblib import Parallel, delayed
import ast

# Get the path to the parent directory
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the Python path
sys.path.append(parent_dir)

current_dir = os.getcwd()


# Import utils folder
from utils.twostep_support import *    
from models import *
from MLE import *

"""---------------SELECT MODEL AND SET VARIABLES ---------------"""

model = "AI_ddm" # RL, RL_ddm, RL_ddm_biased, AI, or AI_ddm
mtype = 3 # 0, 1, 2 or 3 (only relevant if model = AI or AI_ddm)
drmtype = "sigmoid_single_v_mod" # Drift rate model: linear, sigmoid, sigmoid_single_v_mod, sigmoid_single_v_max
n_starts = 35 # Number of random starting parameter values in parameter fitting.
dataset = 'magic_carpet_2020' 
optimizer = 'L-BFGS-B' 

if model in ("RL", "RL_ddm", "RL_ddm_biased"):
    learning = "RL"
elif model in ("AI", "AI_ddm"):
    learning = "PSM"
    

"""-------LOAD SYNTHETIC DATA AND PARAMETERS-------"""

# Path to dataset folder
current_dir = os.getcwd()

if model == "RL":
    path_to_folder = (os.path.join(current_dir, 'param_recovery_data/' + dataset + '/model_' + model +'/'))
    file_path_synthetic_data = os.path.join(path_to_folder, "param_recovery_synthetic_data_M" + model + ".csv")
    file_path_synthetic_params = os.path.join(path_to_folder, "param_recovery_synthetic_params_M" + model + ".csv")

elif model in ("RL_ddm", "RL_ddm_biased"):
    path_to_folder = (os.path.join(current_dir, 'param_recovery_data/' + dataset + '/model_' + model + "_DRM" + drmtype + '/'))
    file_path_synthetic_data = os.path.join(path_to_folder, "param_recovery_synthetic_data_M" + model + "_DRM" + drmtype + ".csv")
    file_path_synthetic_params = os.path.join(path_to_folder, "param_recovery_synthetic_params_M" + model + "_DRM" + drmtype +".csv")
    
elif model == "AI":
    path_to_folder = (os.path.join(current_dir, 'param_recovery_data/' + dataset + '/model_' + model + str(mtype) + '/'))
    file_path_synthetic_data = os.path.join(path_to_folder, "param_recovery_synthetic_data_M" + model + str(mtype) + ".csv")
    file_path_synthetic_params = os.path.join(path_to_folder, "param_recovery_synthetic_params_M" + model + str(mtype) + ".csv")

elif model == "AI_ddm":    
    path_to_folder = (os.path.join(current_dir, 'param_recovery_data/' + dataset + '/model_' + model + str(mtype) + "_DRM" + drmtype + '/'))
    file_path_synthetic_data = os.path.join(path_to_folder, "param_recovery_synthetic_data_M" + model + str(mtype) + "_DRM" + drmtype + ".csv")
    file_path_synthetic_params = os.path.join(path_to_folder, "param_recovery_synthetic_params_M" + model + str(mtype) + "_DRM" + drmtype +".csv")

df_synthetic_data = pd.read_csv(file_path_synthetic_data)

#Get number of trials T
df_par = df_synthetic_data[df_synthetic_data['Synthetic_participant_ID'] == 1]
T = len(ast.literal_eval(df_par['choice1'].iloc[0])) #Number of trials

synthetic_par_ID_list = np.unique(df_synthetic_data['Synthetic_participant_ID'])

df_synthetic_params = pd.read_csv(file_path_synthetic_params)

"""-------------------DEFINE MODEL PARAMETERS---------------------"""

# Different model classes and parameter settings
if model == "RL":
    p_names = ["lr1", "lr2", "lam", "b1", "b2", "p", "w"]
    lower_bounds = np.array([0, 0, 0, 0, 0, -1, 0])
    upper_bounds = np.array([1, 1, 1, 20, 20, 1, 1])

elif model == "RL_ddm":
    if drmtype == "linear":
        p_names = ["lr1", "lr2", "lam","w","p" , "a_bs", "ndt", "v_stage_0", "v_stage_1"]
        lower_bounds = np.array([0, 0, 0, 0, -1, 0.3, 0, 0, 0])
        upper_bounds = np.array([1, 1, 1, 1, 1, 4, 1, 10, 10])
    elif drmtype == "sigmoid_single_v_mod":
        p_names = ["lr1", "lr2", "lam", "w","p","a_bs", "ndt", "v_max_stage_0", "v_max_stage_1", "v_mod"]
        lower_bounds = np.array([0, 0, 0, 0, -1, 0.3, 0, 0, 0, 0])
        upper_bounds = np.array([1, 1, 1, 1, 1, 4, 1, 10, 10, 1])
    elif drmtype == "sigmoid":
        p_names = ["lr1", "lr2", "lam", "w","p","a_bs", "ndt", "v_max_stage_0", "v_max_stage_1", "v_mod_stage_0", "v_mod_stage_1"]
        lower_bounds = np.array([0, 0, 0, 0, -1, 0.3, 0, 0, 0, 0, 0])
        upper_bounds = np.array([1, 1, 1, 1, 1, 4, 1, 10, 10, 1, 1])
    elif drmtype == "sigmoid_single_v_max":
        p_names = ["lr1", "lr2", "lam", "w","p","a_bs", "ndt", "v_max", "v_mod_stage_0", "v_mod_stage_1"]
        lower_bounds = np.array([0, 0, 0, 0, -1, 0.3, 0, 0, 0, 0])
        upper_bounds = np.array([1, 1, 1, 1, 1, 4, 1, 10, 1, 1])

elif model == "RL_ddm_biased":
    if drmtype == "linear":
        p_names = ["lr1", "lr2", "lam","w","p" , "a_bs", "ndt", "z_prime", "v_stage_0", "v_stage_1"]
        lower_bounds = np.array([0, 0, 0, 0, -1, 0.3, 0, 0, 0, 0])
        upper_bounds = np.array([1, 1, 1, 1, 1, 4, 1, 0.5, 10, 10])
    elif drmtype == "sigmoid_single_v_mod":
        p_names = ["lr1", "lr2", "lam", "w","p","a_bs", "ndt", "z_prime", "v_max_stage_0", "v_max_stage_1", "v_mod"]
        lower_bounds = np.array([0, 0, 0, 0, -1, 0.3, 0, 0, 0, 0, 0])
        upper_bounds = np.array([1, 1, 1, 1, 1, 4, 1, 0.5, 10, 10, 1])
    elif drmtype == "sigmoid":
        p_names = ["lr1", "lr2", "lam", "w","p","a_bs", "ndt", "z_prime", "v_max_stage_0", "v_max_stage_1", "v_mod_stage_0", "v_mod_stage_1"]
        lower_bounds = np.array([0, 0, 0, 0, -1, 0.3, 0, 0, 0, 0, 0, 0])
        upper_bounds = np.array([1, 1, 1, 1, 1, 4, 1, 0.5, 10, 10, 1, 1])
    elif drmtype == "sigmoid_single_v_max":
        p_names = ["lr1", "lr2", "lam", "w","p","a_bs", "ndt", "z_prime", "v_max", "v_mod_stage_0", "v_mod_stage_1"]
        lower_bounds = np.array([0, 0, 0, 0, -1, 0.3, 0, 0, 0, 0, 0])
        upper_bounds = np.array([1, 1, 1, 1, 1, 4, 1, 0.5, 10, 1, 1])
        
elif model == "AI":
    if mtype == 0:
        p_names = ["lr","vunsamp", "vsamp", "vps", "gamma1", "gamma2", "lam", "kappa_a", "prior_r"]
        lower_bounds = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.2])
        upper_bounds = np.array([4, 0.9, 0.9, 0.9, 20, 20, 10, 5, 0.8])
    elif mtype == 1:
        p_names = ["lr","vunsamp", "vps", "gamma1", "gamma2", "lam", "kappa_a", "prior_r"]
        lower_bounds = np.array([0, 0, 0, 0, 0, 0, 0, 0.2])
        upper_bounds = np.array([4, 0.9, 0.9, 20, 20, 10, 5, 0.8])
    elif mtype == 2:
        p_names = ["lr", "vsamp", "vps", "gamma1", "gamma2", "lam", "kappa_a", "prior_r"]
        lower_bounds = np.array([0, 0, 0, 0, 0, 0, 0, 0.2])
        upper_bounds = np.array([4, 0.9, 0.9, 20, 20, 10, 5, 0.8])
    elif mtype == 3:
        p_names = ["lr","vunsamp", "vsamp", "gamma1", "gamma2", "lam", "kappa_a", "prior_r"]
        lower_bounds = np.array([0, 0, 0, 0, 0, 0, 0, 0.2])
        upper_bounds = np.array([4, 0.9, 0.9, 20, 20, 10, 5, 0.8])     

elif model == "AI_ddm":
    if mtype == 0:
        if drmtype == "linear":
            p_names = ["lr","vunsamp", "vsamp", "vps", "lam", "kappa_a", "prior_r", "a_bs", "ndt", "v_stage_0", "v_stage_1"]
            lower_bounds = np.array([0, 0, 0, 0, 0, 0, 0.2, 0.3, 0, 0, 0])
            upper_bounds = np.array([4, 0.9, 0.9, 0.9, 10, 5, 0.8, 4, 1, 10, 10])
        if drmtype == "sigmoid":
            p_names = ["lr","vunsamp", "vsamp", "vps", "lam", "kappa_a", "prior_r", "a_bs", "ndt", "v_max_stage_0", "v_max_stage_1", "v_mod_stage_0", "v_mod_stage_1"]
            lower_bounds = np.array([0, 0, 0, 0, 0, 0, 0.2, 0.3, 0, 0, 0, 0, 0])
            upper_bounds = np.array([4, 0.9, 0.9, 0.9, 10, 5, 0.8, 4, 1, 10, 10, 1, 1]) 
        if drmtype == "sigmoid_single_v_mod":
            p_names = ["lr","vunsamp", "vsamp", "vps", "lam", "kappa_a", "prior_r", "a_bs", "ndt", "v_max_stage_0", "v_max_stage_1", "v_mod"]
            lower_bounds = np.array([0, 0, 0, 0, 0, 0, 0.2, 0.3, 0, 0, 0, 0])
            upper_bounds = np.array([4, 0.9, 0.9, 0.9, 10, 5, 0.8, 4, 1, 10, 10, 1]) 
        if drmtype == "sigmoid_single_v_max":
            p_names = ["lr","vunsamp", "vsamp", "vps", "lam", "kappa_a", "prior_r", "a_bs", "ndt", "v_max", "v_mod_stage_0", "v_mod_stage_1"]
            lower_bounds = np.array([0, 0, 0, 0, 0, 0, 0.2, 0.3, 0, 0, 0, 0])
            upper_bounds = np.array([4, 0.9, 0.9, 0.9, 10, 5, 0.8, 4, 1, 10, 1, 1])
    elif mtype == 1:
        if drmtype == "linear":
            p_names = ["lr","vunsamp", "vps","lam", "kappa_a", "prior_r", "a_bs", "ndt", "v_stage_0", "v_stage_1"]
            lower_bounds = np.array([0, 0, 0, 0, 0, 0.2, 0.3, 0, 0, 0])
            upper_bounds = np.array([4, 0.9, 0.9, 10, 5, 0.8, 4, 1, 10, 10])
        if drmtype == "sigmoid":
            p_names = ["lr","vunsamp", "vps","lam", "kappa_a", "prior_r", "a_bs", "ndt", "v_max_stage_0", "v_max_stage_1", "v_mod_stage_0", "v_mod_stage_1"]
            lower_bounds = np.array([0, 0, 0, 0, 0, 0.2, 0.3, 0, 0, 0, 0, 0])
            upper_bounds = np.array([4, 0.9, 0.9, 10, 5, 0.8, 4, 1, 10, 10, 1, 1])
        if drmtype == "sigmoid_single_v_mod":
            p_names = ["lr","vunsamp", "vps","lam", "kappa_a", "prior_r", "a_bs", "ndt", "v_max_stage_0", "v_max_stage_1", "v_mod"]
            lower_bounds = np.array([0, 0, 0, 0, 0, 0.2, 0.3, 0, 0, 0, 0, 0])
            upper_bounds = np.array([4, 0.9, 0.9, 10, 5, 0.8, 4, 1, 10, 10, 1])
        if drmtype == "sigmoid_single_v_max":
            p_names = ["lr","vunsamp", "vps","lam", "kappa_a", "prior_r", "a_bs", "ndt", "v_max", "v_mod_stage_0", "v_mod_stage_1"]
            lower_bounds = np.array([0, 0, 0, 0, 0, 0.2, 0.3, 0, 0, 0, 0])
            upper_bounds = np.array([4, 0.9, 0.9, 10, 5, 0.8, 4, 1, 10, 1, 1])
    elif mtype == 2:
        if drmtype == "linear":
            p_names = ["lr", "vsamp", "vps","lam", "kappa_a", "prior_r", "a_bs", "ndt", "v_stage_0", "v_stage_1"]
            lower_bounds = np.array([0, 0, 0, 0, 0, 0.2, 0.3, 0, 0, 0])
            upper_bounds = np.array([4, 0.9, 0.9, 10, 5, 0.8, 4, 1, 10, 10])

        if drmtype == "sigmoid":
            p_names = ["lr", "vsamp", "vps","lam", "kappa_a", "prior_r", "a_bs", "ndt", "v_max_stage_0", "v_max_stage_1", "v_mod_stage_0", "v_mod_stage_1"]
            lower_bounds = np.array([0, 0, 0, 0, 0, 0.2, 0.3, 0, 0, 0, 0, 0])
            upper_bounds = np.array([4, 0.9, 0.9, 10, 5, 0.8, 4, 1, 10, 10, 1, 1])
        if drmtype == "sigmoid_single_v_mod":
            p_names = ["lr", "vsamp", "vps","lam", "kappa_a", "prior_r", "a_bs", "ndt", "v_max_stage_0", "v_max_stage_1", "v_mod"]
            lower_bounds = np.array([0, 0, 0, 0, 0, 0.2, 0.3, 0, 0, 0, 0])
            upper_bounds = np.array([4, 0.9, 0.9, 10, 5, 0.8, 4, 1, 10, 10, 1])
        if drmtype == "sigmoid_single_v_max":
            p_names = ["lr", "vsamp", "vps","lam", "kappa_a", "prior_r", "a_bs", "ndt", "v_max", "v_mod_stage_0", "v_mod_stage_1"]
            lower_bounds = np.array([0, 0, 0, 0, 0, 0.2, 0.3, 0, 0, 0, 0])
            upper_bounds = np.array([4, 0.9, 0.9, 10, 5, 0.8, 4, 1, 10, 1, 1])
        
    elif mtype == 3:
        if drmtype == "linear":
            p_names = ["lr","vunsamp", "vsamp", "lam", "kappa_a", "prior_r", "a_bs", "ndt", "v_stage_0", "v_stage_1"]
            lower_bounds = np.array([0, 0, 0, 0, 0, 0.2, 0.3, 0, 0, 0])
            upper_bounds = np.array([4, 0.9, 0.9, 10, 5, 0.8, 4, 1, 10, 10])

        if drmtype == "sigmoid":
            p_names = ["lr","vunsamp", "vsamp", "lam", "kappa_a", "prior_r", "a_bs", "ndt", "v_max_stage_0", "v_max_stage_1", "v_mod_stage_0", "v_mod_stage_1"]
            lower_bounds = np.array([0, 0, 0, 0, 0, 0.2, 0.3, 0, 0, 0, 0, 0])
            upper_bounds = np.array([4, 0.9, 0.9, 10, 5, 0.8, 4, 1, 10, 10, 1, 1])
        if drmtype == "sigmoid_single_v_mod":
            p_names = ["lr","vunsamp", "vsamp", "lam", "kappa_a", "prior_r", "a_bs", "ndt", "v_max_stage_0", "v_max_stage_1", "v_mod"]
            lower_bounds = np.array([0, 0, 0, 0, 0, 0.2, 0.3, 0, 0, 0, 0])
            upper_bounds = np.array([4, 0.9, 0.9, 10, 5, 0.8, 4, 1, 10, 10, 1])
            
        if drmtype == "sigmoid_single_v_max":
            p_names = ["lr","vunsamp", "vsamp", "lam", "kappa_a", "prior_r", "a_bs", "ndt", "v_max", "v_mod_stage_0", "v_mod_stage_1"]
            lower_bounds = np.array([0, 0, 0, 0, 0, 0.2, 0.3, 0, 0, 0, 0])
            upper_bounds = np.array([4, 0.9, 0.9, 10, 5, 0.8, 4, 1, 10, 1, 1])


nump = len(p_names) # Parameter names

"""----------------DEFINE FITTING FUNCTIONS------------------"""

# Function to process each participant
def process_participant(par):
    
    df_par = df_synthetic_data[df_synthetic_data['Synthetic_participant_ID'] == par]
    df_synthetic_param_set = df_synthetic_params[df_synthetic_params['Synthetic_participant_ID'] == par]

    T = len(ast.literal_eval(df_par["choice1"].iloc[0])) 

    # Prepare data
    actions = np.zeros((T, 2))
    actions_hssm = np.zeros((T,2))
    observations = np.zeros((T, 2))
    rts = np.zeros((T, 2))

    actions[:,0] = ast.literal_eval(df_par["choice1"].iloc[0])
    actions[:,1] = ast.literal_eval(df_par["choice2"].iloc[0])
    observations[:,0] = ast.literal_eval(df_par["final_state"].iloc[0])
    observations[:,1] = ast.literal_eval(df_par["reward"].iloc[0])

    if model in ("RL_ddm","AI_ddm", "RL_ddm_biased"):
        rts[:,0] = ast.literal_eval(df_par["rt1"].iloc[0])
        rts[:,1] = ast.literal_eval(df_par["rt2"].iloc[0])

    for i in range(len(actions[:,0])):
        if actions[i,0] == 0:
            actions_hssm[i,0] = -1
        else:
            actions_hssm[i,0] = actions[i,0]

        if actions[i,1] == 0:
            actions_hssm[i,1] = -1
        else:
            actions_hssm[i,1] = actions[i,1]

    # Perform MLE procedure
    try:
        best_p, minNLL, NLLs = MLE_procedure(params = p_names,
                                           observations = observations.astype(int),
                                           actions = actions.astype(int),
                                           actions_hssm = actions_hssm.astype(int),
                                           rts = rts,
                                           learning = learning,
                                           lower_bounds = lower_bounds,
                                           upper_bounds = upper_bounds,
                                           n_starts = n_starts,
                                           model=model,
                                           mtype=mtype,
                                           drmtype=drmtype,
                                           seed=par,
                                           optimizer = optimizer)

        best_fitted_params = best_p.tolist()

        results_array = {"ParticipantID": par, "NLL": minNLL}

        for p, param in enumerate(p_names):
            results_array["Recovered_" + param] = best_fitted_params[p] 
            results_array["Synthetic_" + param] = df_synthetic_param_set["Synthetic_" + param].iloc[0]

    
    except Exception as e:
        print(f"MLE procedure failed for participant {par}: {e}")
        results_array = {"ParticipantID": par}
        for p, param in enumerate(p_names):
            results_array["Recovered_" + param] = 0 
            results_array["Synthetic_" + param] = df_synthetic_param_set["Synthetic_" + param].iloc[0]

    return results_array


"""------------------------FIT PARAMETERS--------------------------"""

print('Starting parameter fitting of synthetic data (parameter recovery analysis)')
print(f'Model: {model}')
print(f'Dataset : {dataset}')
print(f'N starts: {n_starts}')

if model in ("AI_ddm", "AI"):
    print(f'Mtype : {mtype}')
if model in ("AI_ddm", "RL_ddm", "RL_ddm_biased"):
    print(f'Drmtype : {drmtype}')

# Parallel execution using Joblib
num_cores = os.cpu_count()  # Detect the number of CPU cores

results = Parallel(n_jobs=10, verbose = 1)(delayed(process_participant)(par) for par in synthetic_par_ID_list)

df_param_recovery_results = pd.DataFrame(results)


"""----------------SAVE FITTED PARAMETERS---------------"""
if model == "RL":
    file_path = os.path.join(path_to_folder, "param_recovery_results_M" + model + "_n_starts" + str(n_starts) + ".csv")
elif model in ("RL_ddm", "RL_ddm_biased"):
    file_path = os.path.join(path_to_folder, "param_recovery_results_M" + model + "_DRM" + drmtype + "_n_starts" + str(n_starts) + ".csv")
elif model == "AI":
    file_path = os.path.join(path_to_folder, "param_recovery_results_M" + model + str(mtype) + "_n_starts" + str(n_starts) + ".csv")
elif model == "AI_ddm":    
    file_path = os.path.join(path_to_folder, "param_recovery_results_M" + model + str(mtype) + "_DRM" + drmtype + "_n_starts" + str(n_starts) + ".csv")
    
df_param_recovery_results.to_csv(file_path, index=False)