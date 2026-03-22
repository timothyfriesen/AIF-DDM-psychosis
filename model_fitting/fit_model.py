"""
Module for performing parameter fitting

Author: Alvaro Garrido Perez <alvaro.garridoperez@ugent.be>
Date: 09-12-2025

"""



"""------------------IMPORT PACKAGES--------------------"""
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
#current_dir = os.getcwd()
#parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the Python path
#sys.path.append(parent_dir)

#current_dir = os.getcwd()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

print("PROJECT ROOT:", project_root)
print("SYS PATH:", sys.path[:3])

# Import utils folder
from utils.twostep_support import *    
from models import *
from MLE import *

"""-----------SELECT MODEL CLASS AND SET FREE PARAMETER VALUE RANGES----------"""

model = "AI_ddm" # RL, RL_ddm,RL_ddm_biased, AI, or AI_ddm
mtype = 3 # 0, 1, 2 or 3 (only relevant if model = AI or AI_ddm)
drmtype = "linear" # Drift rate model: linear, sigmoid, sigmoid_single_v_mod, sigmoid_single_v_max
n_starts = 3 # Number of random starting parameter values in parameter fitting.
dataset = 'magic_carpet_2020' # Name of the dataset
optimizer = 'L-BFGS-B' # Nelder-Mead,L-BFGS-B or DE

if model in ("RL", "RL_ddm", "RL_ddm_biased"):
    learning = "RL"
elif model in ("AI", "AI_ddm"):
    learning = "PSM"
    

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


"""------------------LOAD DATASET--------------------"""

# Path to dataset

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

if dataset == 'magic_carpet_2020':
    path_to_data = os.path.join(
        project_root,
        'two_step_task_datasets',
        'magic_carpet_2020_dataset',
        'choices'
    )
    file_paths = glob.glob(os.path.join(path_to_data, "*game.csv"))

elif dataset == 'magic_carpet_2023':
    path_to_data = os.path.join(
        project_root,
        'two_step_task_datasets',
        'magic_carpet_2023_dataset',
        'task_behaviour'
    )
    file_paths = glob.glob(os.path.join(path_to_data, "*story.csv"))

print("DATA PATH:", path_to_data)
print("FILES FOUND:", len(file_paths))

# Create an empty list to hold the dataframes
df_list = []

par_ids = []

# Loop through each file
for file_path in file_paths:
    # Extract the participant ID from the filename (assuming filename format like 001_story.csv)
    participant_id_str = os.path.basename(file_path).split('_')[0]
    
    participant_id = int(participant_id_str)

    if dataset == 'magic_carpet_2020' and participant_id == 4960: # Avoid outlier participant
        continue
        
    par_ids.append(participant_id)
    
    # Read the CSV file into a dataframe
    df = pd.read_csv(file_path)
    
    # Add the ParticipantID column
    df['ParticipantID'] = participant_id
    
    # Append the dataframe to the list
    df_list.append(df)

# Concatenate all dataframes into one
df_task_behaviour_story = pd.concat(df_list, ignore_index=True)


n_par = len(par_ids) # Number of participants

import time

"""----------------DEFINE FITTING FUNCTIONS------------------"""

# Function to process each participant
def process_participant(par):
    
    start_total = time.time()
    print(f"\n[START] Participant {par}")
    sys.stdout.flush()

    df_par = df_task_behaviour_story[df_task_behaviour_story['ParticipantID'] == par]
    T = len(df_par)

    
    # Identify bad trials
    if dataset == 'magic_carpet_2020':
        badtrials_rt1 = np.where((df_par['rt1'] < 0.1) | (df_par['rt1'] == -1) | df_par['rt1'].isna())[0]
        badtrials_rt2 = np.where((df_par['rt2'] < 0.1) | (df_par['rt2'] == -1) | df_par['rt2'].isna())[0]
    if dataset == 'magic_carpet_2023':
        badtrials_rt1 = np.where((df_par['rt1'] < 0.1) | (df_par['rt1'] == "") | df_par['rt1'].isna())[0]
        badtrials_rt2 = np.where((df_par['rt2'] < 0.1) | (df_par['rt2'] == "") | df_par['rt2'].isna())[0]
        
    badtrials = np.concatenate((badtrials_rt1, badtrials_rt2))
    badtrials = np.sort(np.unique(badtrials))

    # Prepare data
    actions_i = df_par["choice1"].values
    actions_f = df_par["choice2"].values
    rts_i = df_par['rt1'].values
    rts_f = df_par['rt2'].values
    transitions = df_par["final_state"].values
    rewards = df_par["reward"].values

    actions = np.zeros((T - len(badtrials), 2))
    actions_hssm = np.zeros((T - len(badtrials), 2))
    observations = np.zeros((T - len(badtrials), 2))
    rts = np.zeros((T - len(badtrials), 2))
    
    actions[:,0] = np.delete(actions_i, badtrials) - 1
    actions[:,1] = np.delete(actions_f, badtrials) - 1
    observations[:,0] = np.delete(transitions, badtrials) - 1
    observations[:,1] = np.delete(rewards, badtrials)
    rts[:,0] = np.delete(rts_i, badtrials) 
    rts[:,1] = np.delete(rts_f, badtrials)

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
        print(f"[FITTING] Participant {par} | Trials: {len(actions)}")
        sys.stdout.flush()

        start_fit = time.time()
        
        if optimizer == "DE":

            best_p, minNLL = MLE_procedure_DE(params = p_names,
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
                                               seed=par)

        else:
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
        
        
        fit_time = time.time() - start_fit
        total_time = time.time() - start_total 
        print(f"[DONE] Participant {par} | Fit time: {fit_time:.1f}s | Total time: {total_time:.1f}s")
        sys.stdout.flush()

        best_fitted_params = best_p.tolist()

        results_array = {"ParticipantID": par, "NLL": minNLL}

        for p, param in enumerate(p_names):
            results_array["Fitted_" + param] = best_fitted_params[p] 

    
    
    except Exception as e:
        print(f"MLE procedure failed for participant {par}: {e}")
        sys.stdout.flush()

        results_array = {"ParticipantID": par, "NLL": 0}
        for p, param in enumerate(p_names):
            results_array["Fitted_" + param] = 0

    return results_array


"""------------------FIT PARAMETERS------------------"""

print('Starting parameter fitting procedure')
print(f'Model: {model}')
print(f'Dataset : {dataset}')
print(f'N starts: {n_starts}')

if model in ("AI_ddm", "AI"):
    print(f'Mtype : {mtype}')
if model in ("AI_ddm", "RL_ddm", "RL_ddm_biased"):
    print(f'Drmtype : {drmtype}')

# Parallel execution using Joblib
num_cores = os.cpu_count()  # Detect the number of CPU cores

results = Parallel(n_jobs=5,verbose = 1)(delayed(process_participant)(par) for par in par_ids)

df_parameter_fitting_results = pd.DataFrame(results)

"""----------------SAVE FITTED PARAMETERS-------------------"""

path_to_folder = os.path.join(project_root, 'fitted_parameters', dataset)

# Create the folder if it doesn't exist
os.makedirs(path_to_folder, exist_ok=True)

if model == "RL":
    file_path = os.path.join(path_to_folder, "fitted_parameters_M" + model + "_n_starts" + str(n_starts) + ".csv")
elif model == "RL_ddm":
    file_path = os.path.join(path_to_folder, "fitted_parameters_M" + model + "_DRM" + drmtype + "_n_starts" + str(n_starts) + ".csv")
elif model == "RL_ddm_biased":
    file_path = os.path.join(path_to_folder, "fitted_parameters_M" + model + "_DRM" + drmtype + ".csv")
elif model == "AI":
    file_path = os.path.join(path_to_folder, "fitted_parameters_M" + model + str(mtype) + "_n_starts" + str(n_starts) + ".csv")
elif model == "AI_ddm":    
    file_path = os.path.join(path_to_folder, "fitted_parameters_M" + model + str(mtype) + "_DRM" + drmtype + "_n_starts" + str(n_starts) + ".csv")

df_parameter_fitting_results.to_csv(file_path, index=False)

