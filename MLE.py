"""
Module to evaluate Negative Log Likelihoods of different models

@author: Alvaro Garrido Perez <alvaro.garridoperez@ugent.be>
Date: 09-12-2025

This module is an extension of the code developed and made public by Dr. Sam Gijsen
for the paper: Active inference in the two-step task. 
The code was extended to include drift-diffusion models.
Credit goes to Dr. Gijsen for developing the original code.

Link to the original code repository: github.com/SamGijsen/AI2step
"""



import numpy as np
import scipy.optimize as op
from scipy.optimize import differential_evolution, basinhopping, Bounds
import os
import sys
import hddm_wfpt
import models
import pandas as pd
from utils.twostep_support import *    
import random


def eval_NLL_AI(params, observations, actions, learning, mtype):
    """
    Evaluate Negative Log Likelihood for a sequence of actions given a sequence of observations (performed trial-wise).

    ~~~~~~
    INPUTS
    ~~~~~~
    params: model parameters
    observations: sequence of transition and outcome observations
    actions: sequence of taken actions
    learning: learning algorithm used (Default is "PSM": Predictive-surprise modulated learning)
    mtype: integer specifying submodel
    """

    if mtype == 0:
        lr = params[0]
        vunsamp = params[1]
        vsamp = params[2]
        vps = params[3]
        gam1 = params[4]
        gam2 = params[5]
        lam = params[6]
        kappa_a = params[7]
        prior_nu = 2
        prior_r = params[8]

    #If mtype > 0:
    else: # We set parameters 0, 1, and 2 later, for now the identical parameters for all mtype>0:
        lr = params[0] 
        gam1 = params[3]
        gam2 = params[4]
        lam = params[5]
        kappa_a = params[6]
        prior_nu = 2
        prior_r = params[7]

        if mtype == 1: # No Decay for Sampled Actions
            vunsamp = params[1]
            vsamp = 0
            vps = params[2]
            
        if mtype == 2: # No Decay for Unsampled Actions
            vunsamp = 0
            vsamp = params[1]
            vps = params[2]

        if mtype == 3: # No Surprise Learning
            vunsamp = params[1]
            vsamp = params[2]
            vps = 0

    #### ----------
    # Specify task and generate (potential) observations
    T = observations.shape[0]
    task = {  
        "type": "drift",
        "T": T,
        "x": False,
        "r": True,
        "delta": 0.025,
        "bounds": [0.25 ,0.75]
    }
    
    model = { # Model specification
        "act": "AI",
        "learn": "PSM",
        "learn_transitions": False,
        "lr": lr,
        "vunsamp": vunsamp,
        "vsamp": vsamp,
        "vps": vps, 
        "gamma1": gam1,
        "gamma2": gam2,
        "lam": lam,
        "kappa_a": kappa_a,
        "prior_r": prior_r
        }

    temp = models.learn_and_act(task, model)
    La = np.ones((T,2))

    po = np.zeros(2)
    pa = np.zeros(2)
    
    # Check which actions were taken and which outcomes observed
    for t in range(T):
        po = observations[t,:].astype(int)
        pa = actions[t,:].astype(int)

        a, o, pi, p_trans, p_r, GQ = temp.perform_trial(t, pa, po)

        La[t,0] = GQ[t, 0, pa[0]]
        La[t,1] = GQ[t, po[0]+1, pa[1]]
    
    epsilon = 1e-10  # Avoid Log of zero
    return -np.sum(np.log(np.maximum(La, epsilon)))
    #return -np.sum(np.log(La))#, GQ


def sigmoid_func(z,v_max):
    return ((2*v_max)/(1+np.exp(-z))) - v_max

def eval_NLL_AI_ddm(params, observations, actions, actions_hssm, rts, learning, mtype,drmtype):
    """
    Evaluate likelihood for a sequence of actions, RTs and observations (performed trial-wise).

    ~~~~~~
    INPUTS
    ~~~~~~
    params: model parameters
    observations: sequence of transition and outcome observations
    actions: sequence of taken actions
    actions_hssm: sequence of taken actions in format compatible with HSSM package
    rts: sequence of reaction times
    learning: learning algorithm used (Default is "PSM": Predictive-surprise modulated learning)
    mtype: integer specifying submodel
    drmtype: drift rate model type (linear, sigmoid or sigmoid_single_v_mod
    """

    #---------Get parameters-------#

    if mtype == 0:
        lr = params[0]
        vunsamp = params[1]
        vsamp = params[2]
        vps = params[3]
        lam = params[4]
        kappa_a = params[5]
        prior_nu = 2
        prior_r = params[6]
        a_bs = params[7]
        z = 0.5
        ndt = params[8]
        
        if drmtype == 'linear':
            v_stage_0 = params[9]
            v_stage_1 = params[10]
            
        if drmtype == 'sigmoid':
            v_max_stage_0 = params[9]
            v_max_stage_1 = params[10]
            v_mod_stage_0 = params[11]
            v_mod_stage_1 = params[12]
            
        if drmtype == 'sigmoid_single_v_mod':
            v_max_stage_0 = params[9]
            v_max_stage_1 = params[10]
            v_mod = params[11]
        if drmtype == 'sigmoid_single_v_max':
            v_max = params[9]
            v_mod_stage_0 = params[10]
            v_mod_stage_1 = params[11]
   
    else:
        lr = params[0]
        lam = params[3]
        kappa_a = params[4]
        prior_nu = 2
        prior_r = params[5]
        a_bs = params[6]
        z = 0.5
        ndt = params[7]
        
        if drmtype == 'linear':
            v_stage_0 = params[8]
            v_stage_1 = params[9]
            
        if drmtype == 'sigmoid':
            v_max_stage_0 = params[8]
            v_max_stage_1 = params[9]
            v_mod_stage_0 = params[10]
            v_mod_stage_1 = params[11]
        
        if drmtype == 'sigmoid_single_v_mod':
            v_max_stage_0 = params[8]
            v_max_stage_1 = params[9]
            v_mod = params[10]
            
        if drmtype == 'sigmoid_single_v_max':
            v_max = params[8]
            v_mod_stage_0 = params[9]
            v_mod_stage_1 = params[10]

        if mtype == 1: # No Decay for Sampled Actions
            vunsamp = params[1]
            vsamp = 0
            vps = params[2]
            
        if mtype == 2: # No Decay for UNsampled Actions
            vunsamp = 0
            vsamp = params[1]
            vps = params[2]

        if mtype == 3: # No Surprise Learning
            vunsamp = params[1]
            vsamp = params[2]
            vps = 0  

    #### ----------
    # Specify task and generate (potential) observations
    T = observations.shape[0]
    
    task = {  
        "type": "drift",
        "T": T,
        "x": False,
        "r": True,
        "delta": 0.025,
        "bounds": [0.25 ,0.75]
    }
    

    model = { # Model specification
        "act": "AI",
        "learn": "PSM",
        "learn_transitions": False,
        "lr": lr,
        "vunsamp": vunsamp,
        "vsamp": vsamp,
        "vps": vps, 
        "gamma1": 0,
        "gamma2": 0,
        "lam": lam,
        "kappa_a": kappa_a,
        "prior_r": prior_r
        }


    temp = models.learn_and_act(task, model)
    
    EFEs = np.ones((T,6))

    po = np.zeros(2)
    pa = np.zeros(2)

    v0_array = np.zeros(T) # Array of ddm drift rate values for stage 0
    v1_array = np.zeros(T) # Array of ddm drift rate values for stage 1
    
    # Check which actions were taken and which outcomes observed
    for n in range(T):
        po = observations[n,:].astype(int)
        pa = actions[n,:].astype(int)

        # Get Expected Free Energies (EFEs) of all actions
        Gs = temp.perform_trial_return_EFE_or_Q(n, pa, po)

        EFEs[n,0] = Gs[n, 0, 0] # EFE of stage 0, action 0
        EFEs[n,1] = Gs[n, 0, 1] # EFE of stage 0, action 1
        EFEs[n,2] = Gs[n, 1, 0] # EFE of stage 1, action 0
        EFEs[n,3] = Gs[n, 1, 1] # EFE of stage 1, action 1
        EFEs[n,4] = Gs[n, 2, 0] # EFE of stage 2, action 0
        EFEs[n,5] = Gs[n, 2, 1] # EFE of stage 2, action 1

        # Compute difference in EFEs for stage 0
        # Add a small random number so that the difference is never exactly 0
        #rand_num = np.random.uniform(-0.000000001, 0.000000001) 
        delta_EFE_stage_0 = EFEs[n,0]-EFEs[n,1] #+ rand_num

        # Calculate drift rate for stage 0 given delta EFEs
        if drmtype == 'linear':
            v0_array[n] = v_stage_0*delta_EFE_stage_0
        if drmtype == 'sigmoid':
            v0_array[n] = sigmoid_func(v_mod_stage_0*delta_EFE_stage_0,v_max_stage_0)
        if drmtype == 'sigmoid_single_v_mod':
            v0_array[n] = sigmoid_func(v_mod*delta_EFE_stage_0,v_max_stage_0)
        if drmtype == 'sigmoid_single_v_max':
            v0_array[n] = sigmoid_func(v_mod_stage_0*delta_EFE_stage_0,v_max)
        
        observed_tran = observations[n,0]

        # Calculate drift rate for stage 1 given delta EFEs
        
        #rand_num = np.random.uniform(-0.000000001,  0.000000001)

        if observed_tran == 0:
            delta_EFE_stage_1 = EFEs[n, 2] - EFEs[n, 3] #+ rand_num
        if observed_tran == 1:
            delta_EFE_stage_1 = EFEs[n, 4] - EFEs[n, 5] #+ rand_num

        if drmtype == 'linear':
            v1_array[n] = v_stage_1*delta_EFE_stage_1
        if drmtype == 'sigmoid':
            v1_array[n] = sigmoid_func(v_mod_stage_1*delta_EFE_stage_1,v_max_stage_1)
        if drmtype == 'sigmoid_single_v_mod':
            v1_array[n] = sigmoid_func(v_mod*delta_EFE_stage_1,v_max_stage_1)
        if drmtype == 'sigmoid_single_v_max':
            v1_array[n] = sigmoid_func(v_mod_stage_1*delta_EFE_stage_1,v_max)
        
        
    # Format data of stage 0 adequately for HSSM/HDDM package
    data_stage_0 = pd.Series(rts[:,0] * actions_hssm[:,0])

    err=1e-8

    # Compute logLikelihood of making a given a choice with a given RT for stage 0
    loglik_stage_0 =  hddm_wfpt.wfpt.wiener_logp_array(
        np.float64(data_stage_0),
        (v0_array).astype(np.float64),
        np.ones(T) * 0,
        (np.ones(T) * 2 * a_bs).astype(np.float64),
        (np.ones(T) * z).astype(np.float64),
        np.ones(T) * 0,
        (np.ones(T) * ndt).astype(np.float64),
        np.ones(T) * 0,
        err,
        1,
    )

    # Format data of stage 1 adequately for HSSM/HDDM package
    data_stage_1 = pd.Series(rts[:,1] * actions_hssm[:,1])

    # Compute logLikelihood of making a given a choice with a given RT for stage 1
    loglik_stage_1 =  hddm_wfpt.wfpt.wiener_logp_array(
        np.float64(data_stage_1),
        (v1_array).astype(np.float64),
        np.ones(T) * 0,
        (np.ones(T) * 2 * a_bs).astype(np.float64),
        (np.ones(T) * z).astype(np.float64),
        np.ones(T) * 0,
        (np.ones(T) * ndt).astype(np.float64),
        np.ones(T) * 0,
        err,
        1,
    )

    # The overall loglikelihood is equal to :
    # loglikelihoods of stage 0 + loglikelihood of stage 1
    # Since log(a*B) = log(a) + log(b)
    sum_logs = loglik_stage_0 + loglik_stage_1

    epsilon = -999

    #sum overall loglikelihood array and normalize by total number of trials
    return -np.sum(np.maximum(sum_logs,epsilon))/(2*T)

def eval_NLL_RL(params, observations, actions):
    """
    Evaluate likelihood for a sequence of actions given a sequence of observations (performed trial-wise).

    ~~~~~~
    INPUTS
    ~~~~~~
    params: model parameters
    observations: sequence of transition and outcome observations
    actions: sequence of taken actions
    learning: learning algorithm used (Default is "RL": Currently no other options for RL-based modeling)
    """

    # Specify task and generate (potential) observations
    T = observations.shape[0]
    task = {  
        "type": "drift",
        "T": T,
        "x": False,
        "r": True,
        "delta": 0.025,
        "bounds": [0.25, 0.75]
    }
    
    model = { # Model specification
        "act": "RL",
        "learn": "RL",
        "learn_transitions": False,
        "lr1": params[0],
        "lr2": params[1],
        "lam": params[2],
        "b1": params[3], 
        "b2": params[4],
        "p": params[5],
        "w": params[6],
        }

    temp = models.learn_and_act(task, model)
    La = np.ones((T,2))

    po = np.zeros(2)
    pa = np.zeros(2)
    
    # Check which actions were taken and which outcomes observed
    for t in range(T):
        po = observations[t,:].astype(int)
        pa = actions[t,:].astype(int)

        a, o, pi, p_trans, p_r, GQ = temp.perform_trial(t, pa, po)

        La[t,0] = GQ[t, 0, pa[0]]
        La[t,1] = GQ[t, po[0]+1, pa[1]]

    epsilon = 1e-10  # Avoid Log of zero
    return -np.sum(np.log(np.maximum(La, epsilon)))
    #return -np.sum(np.log(La))#, GQ

def eval_NLL_RL_ddm(params, observations, actions, actions_hssm, rts, drmtype):
    """
    Evaluate likelihood for a sequence of actions and reaction times (RTs) using an RL-DDM approach.

    ~~~~~~
    INPUTS
    ~~~~~~
    params: model parameters
    observations: sequence of transition and outcome observations
    actions: sequence of taken actions
    actions_hssm: sequence of taken actions in format compatible with HSSM package
    rts: sequence of reaction times
    drmtype: drift rate model type (linear, sigmoid, sigmoid_single_v_mod, or sigmoid_single_v_max)
    """
    
    # Extract parameters for the RL-DDM model
    lr1 = params[0]
    lr2 = params[1]
    lam = params[2]
    w = params[3]
    p = params[4]
    a_bs = params[5]  # Boundary separation
    z = 0.5           # Starting point (fixed at 0.5 for symmetric decision-making)
    ndt = params[6]   # Non-decision time

    # Extract drift rate parameters based on the specified drmtype
    if drmtype == 'linear':
        v_stage_0 = params[7]
        v_stage_1 = params[8]
    elif drmtype == 'sigmoid_single_v_mod':
        v_max_stage_0 = params[7]
        v_max_stage_1 = params[8]
        v_mod = params[9]
    elif drmtype == 'sigmoid':
        v_max_stage_0 = params[7]
        v_max_stage_1 = params[8]
        v_mod_stage_0 = params[9]
        v_mod_stage_1 = params[10]
    else:
        v_max = params[7]
        v_mod_stage_0 = params[8]
        v_mod_stage_1 = params[9]
        
    # Specify task and generate (potential) observations
    T = observations.shape[0]
    task = {  
        "type": "drift",
        "T": T,
        "x": False,
        "r": True,
        "delta": 0.025,
        "bounds": [0.25, 0.75]
    }
    
    model = { # Model specification
        "act": "RL",
        "learn": "RL",
        "learn_transitions": False,
        "lr1": lr1,
        "lr2": lr2,
        "lam": lam,
        "w": w,
        "p": p,
        #"a_bs": a_bs,
        #"ndt": ndt,
        }

    temp = models.learn_and_act(task, model)
    
    v0_array = np.zeros(T)
    v1_array = np.zeros(T)
    
    loglik_stage_0 = np.zeros(T)
    loglik_stage_1 = np.zeros(T)

    Qs = np.ones((T,6))
    
    # Evaluate likelihood for each trial
    for t in range(T):
        po = observations[t, :].astype(int)
        pa = actions[t, :].astype(int)

        # Simulate trial
        #a, o, pi, p_trans, p_r, Q = temp.perform_trial(t, pa, po)
        # Get Q-values of all actions
        Q = temp.perform_trial_return_EFE_or_Q(t, pa, po)
        
        # Compute Q-value differences for drift rate calculation
        Qs[t,0] = Q[t, 0, 0] # Qs of stage 0, action 0
        Qs[t,1] = Q[t, 0, 1] # Qs of stage 0, action 1
        Qs[t,2] = Q[t, 1, 0] # Qs of stage 1, action 0
        Qs[t,3] = Q[t, 1, 1] # Qs of stage 1, action 1
        Qs[t,4] = Q[t, 2, 0] # Qs of stage 2, action 0
        Qs[t,5] = Q[t, 2, 1] # Qs of stage 2, action 1

        #rand_num = np.random.uniform(-0.000000001, 0.000000001) 
        #delta_Q_stage_0 = -(Qs[t,0]-Qs[t,1] + rand_num)
        delta_Q_stage_0 = -(Qs[t,0]-Qs[t,1])

        # Compute drift rates based on drmtype
        if drmtype == 'linear':
            v0_array[t] = v_stage_0 * delta_Q_stage_0
        elif drmtype == 'sigmoid':
            v0_array[t] = sigmoid_func(v_mod_stage_0*delta_Q_stage_0,v_max_stage_0)
        elif drmtype == 'sigmoid_single_v_mod':
            v0_array[t] = sigmoid_func(v_mod * delta_Q_stage_0, v_max_stage_0)
        elif drmtype == 'sigmoid_single_v_max':
            v0_array[t] = sigmoid_func(v_mod_stage_0*delta_Q_stage_0,v_max)
        
        
        observed_tran = observations[t,0]

        # Calculate drift rate for stage 1 given delta EFEs
        
        #rand_num = np.random.uniform(-0.000000001,  0.000000001)

        if observed_tran == 0:
            delta_Q_stage_1 = -(Qs[t, 2] - Qs[t, 3])
        if observed_tran == 1:
            delta_Q_stage_1 = -(Qs[t, 4] - Qs[t, 5])

        if drmtype == 'linear':
            v1_array[t] = v_stage_1*delta_Q_stage_1
        if drmtype == 'sigmoid':
            v1_array[t] = sigmoid_func(v_mod_stage_1*delta_Q_stage_1,v_max_stage_1)
        if drmtype == 'sigmoid_single_v_mod':
            v1_array[t] = sigmoid_func(v_mod*delta_Q_stage_1,v_max_stage_1)
        if drmtype == 'sigmoid_single_v_max':
            v1_array[t] = sigmoid_func(v_mod_stage_1*delta_Q_stage_1,v_max)
        

    # Format data of stage 0 adequately for HSSM/HDDM package
    data_stage_0 = pd.Series(rts[:, 0] * actions_hssm[:, 0])

    err = 1e-8

    # Compute log-likelihood of making a given choice with a given RT for stage 0
    loglik_stage_0 = hddm_wfpt.wfpt.wiener_logp_array(
        np.float64(data_stage_0),
        v0_array.astype(np.float64),
        np.zeros(T),
        np.ones(T) * 2 * a_bs,
        np.ones(T) * z,
        np.zeros(T),
        np.ones(T) * ndt,
        np.zeros(T),
        err,
        1,
    )

    # Format data of stage 1 adequately for HSSM/HDDM package
    data_stage_1 = pd.Series(rts[:, 1] * actions_hssm[:, 1])

    # Compute log-likelihood of making a given choice with a given RT for stage 1
    loglik_stage_1 = hddm_wfpt.wfpt.wiener_logp_array(
        np.float64(data_stage_1),
        v1_array.astype(np.float64),
        np.zeros(T),
        np.ones(T) * 2 * a_bs,
        np.ones(T) * z,
        np.zeros(T),
        np.ones(T) * ndt,
        np.zeros(T),
        err,
        1,
    )

    # The overall log-likelihood is the sum of log-likelihoods for stage 0 and stage 1
    sum_logs = loglik_stage_0 + loglik_stage_1

    epsilon = -999  # To prevent invalid log probabilities
    total_loglik = np.maximum(sum_logs, epsilon)

    return -np.sum(total_loglik) / (2 * T)

def eval_NLL_RL_ddm_biased(params, observations, actions, actions_hssm, rts, drmtype):
    """
    Evaluate likelihood for a sequence of actions and reaction times (RTs) using an RL-DDM approach.

    ~~~~~~
    INPUTS
    ~~~~~~
    params: model parameters
    observations: sequence of transition and outcome observations
    actions: sequence of taken actions
    actions_hssm: sequence of taken actions in format compatible with HSSM package
    rts: sequence of reaction times
    drmtype: drift rate model type (linear, sigmoid, sigmoid_single_v_mod, or sigmoid_single_v_max)
    common_trans: sequence of common or uncommon transitions
    """
    
    # Extract parameters for the RL-DDM model
    lr1 = params[0]
    lr2 = params[1]
    lam = params[2]
    w = params[3]
    p = params[4]
    a_bs = params[5]  # Boundary separation
    ndt = params[6]   # Non-decision time
    z_prime = params[7]
    

    # Extract drift rate parameters based on the specified drmtype
    if drmtype == 'linear':
        v_stage_0 = params[8]
        v_stage_1 = params[9]
    elif drmtype == 'sigmoid_single_v_mod':
        v_max_stage_0 = params[8]
        v_max_stage_1 = params[9]
        v_mod = params[10]
    elif drmtype == 'sigmoid':
        v_max_stage_0 = params[8]
        v_max_stage_1 = params[9]
        v_mod_stage_0 = params[10]
        v_mod_stage_1 = params[11]
    else:
        v_max = params[8]
        v_mod_stage_0 = params[9]
        v_mod_stage_1 = params[10]
        
    # Specify task and generate (potential) observations
    T = observations.shape[0]
    task = {  
        "type": "drift",
        "T": T,
        "x": False,
        "r": True,
        "delta": 0.025,
        "bounds": [0.25, 0.75]
    }
    
    model = { # Model specification
        "act": "RL",
        "learn": "RL",
        "learn_transitions": False,
        "lr1": lr1,
        "lr2": lr2,
        "lam": lam,
        "w": w,
        "p": p,
        }

    temp = models.learn_and_act(task, model)
    
    v0_array = np.zeros(T)
    v1_array = np.zeros(T)
    z_stage_0 = np.zeros(T)
    z_stage_1 = np.zeros(T)
    
    loglik_stage_0 = np.zeros(T)
    loglik_stage_1 = np.zeros(T)

    Qs = np.ones((T,6))
    
    # Evaluate likelihood for each trial
    for t in range(T):
        po = observations[t, :].astype(int)
        pa = actions[t, :].astype(int)

        # Simulate trial
        #a, o, pi, p_trans, p_r, Q = temp.perform_trial(t, pa, po)
        # Get Q-values of all actions
        Q = temp.perform_trial_return_EFE_or_Q(t, pa, po)
        
        # Compute Q-value differences for drift rate calculation
        Qs[t,0] = Q[t, 0, 0] # Qs of stage 0, action 0
        Qs[t,1] = Q[t, 0, 1] # Qs of stage 0, action 1
        Qs[t,2] = Q[t, 1, 0] # Qs of stage 1, action 0
        Qs[t,3] = Q[t, 1, 1] # Qs of stage 1, action 1
        Qs[t,4] = Q[t, 2, 0] # Qs of stage 2, action 0
        Qs[t,5] = Q[t, 2, 1] # Qs of stage 2, action 1

        delta_Q_stage_0 = -(Qs[t,0]-Qs[t,1])

        # Compute drift rates based on drmtype
        if drmtype == 'linear':
            v0_array[t] = v_stage_0 * delta_Q_stage_0
        elif drmtype == 'sigmoid':
            v0_array[t] = sigmoid_func(v_mod_stage_0*delta_Q_stage_0,v_max_stage_0)
        elif drmtype == 'sigmoid_single_v_mod':
            v0_array[t] = sigmoid_func(v_mod * delta_Q_stage_0, v_max_stage_0)
        elif drmtype == 'sigmoid_single_v_max':
            v0_array[t] = sigmoid_func(v_mod_stage_0*delta_Q_stage_0,v_max)

        # Compute bias stage 0
        z_stage_0[t] = z_prime * np.sign(delta_Q_stage_0) + 0.5
        
        observed_tran = observations[t,0]

        # Calculate drift rate for stage 1 given delta EFEs
        
        if observed_tran == 0:
            delta_Q_stage_1 = -(Qs[t, 2] - Qs[t, 3])
        if observed_tran == 1:
            delta_Q_stage_1 = -(Qs[t, 4] - Qs[t, 5])

        if drmtype == 'linear':
            v1_array[t] = v_stage_1*delta_Q_stage_1
        if drmtype == 'sigmoid':
            v1_array[t] = sigmoid_func(v_mod_stage_1*delta_Q_stage_1,v_max_stage_1)
        if drmtype == 'sigmoid_single_v_mod':
            v1_array[t] = sigmoid_func(v_mod*delta_Q_stage_1,v_max_stage_1)
        if drmtype == 'sigmoid_single_v_max':
            v1_array[t] = sigmoid_func(v_mod_stage_1*delta_Q_stage_1,v_max)

        # Compute bias stage 1
        if pa[0] == observed_tran:
            # If common transition
            z_stage_1[t] = z_prime * np.sign(delta_Q_stage_1) + 0.5
        else:
            # If uncommon transition
            z_stage_1[t] = 0.5
        

    # Format data of stage 0 adequately for HSSM/HDDM package
    data_stage_0 = pd.Series(rts[:, 0] * actions_hssm[:, 0])

    err = 1e-8

    # Compute log-likelihood of making a given choice with a given RT for stage 0
    loglik_stage_0 = hddm_wfpt.wfpt.wiener_logp_array(
        np.float64(data_stage_0),
        v0_array.astype(np.float64),
        np.zeros(T),
        np.ones(T) * 2 * a_bs,
        z_stage_0,
        np.zeros(T),
        np.ones(T) * ndt,
        np.zeros(T),
        err,
        1,
    )

    # Format data of stage 1 adequately for HSSM/HDDM package
    data_stage_1 = pd.Series(rts[:, 1] * actions_hssm[:, 1])

    # Compute log-likelihood of making a given choice with a given RT for stage 1
    loglik_stage_1 = hddm_wfpt.wfpt.wiener_logp_array(
        np.float64(data_stage_1),
        v1_array.astype(np.float64),
        np.zeros(T),
        np.ones(T) * 2 * a_bs,
        z_stage_1,
        np.zeros(T),
        np.ones(T) * ndt,
        np.zeros(T),
        err,
        1,
    )

    # The overall log-likelihood is the sum of log-likelihoods for stage 0 and stage 1
    sum_logs = loglik_stage_0 + loglik_stage_1

    epsilon = -999  # To prevent invalid log probabilities
    total_loglik = np.maximum(sum_logs, epsilon)

    return -np.sum(total_loglik) / (2 * T)



def MLE_procedure(params, 
                  observations, 
                  actions, 
                  actions_hssm, 
                  rts, 
                  learning, 
                  lower_bounds, 
                  upper_bounds, 
                  n_starts, 
                  model, 
                  mtype, 
                  drmtype,
                  seed=1,
                  optimizer = 'L-BFGS-B'):
    """
    This function calls scipy.op.minimize() repeatedly to perform maximum likelihood estimation.

    ~~~~~~
    INPUTS
    ~~~~~~
    params: model parameters
    obs: sequence of transition and outcome observations
    actions: sequence of taken actions
    actions_hssm: sequence of actions formated for hssm and hddm packages
    rts: sequence of reaction times
    learning: learning algorithm
    lower_bounds, upper_bounds: each parameter needs a min and max bound between which the minimizer functions
    n_starts: amount of iterations. be careful of local minima in case n_starts < 10
    model: active inference (AI) or reinforcement learning (RL)
    mtype: submodel type for active inference
    """

    np.random.seed(seed)

    nump = len(params)
    NLL = np.zeros(n_starts)
    init_params = np.zeros((nump, n_starts))
    params = np.zeros((nump, n_starts))

    # Create bounds and parameter initializations
    bounds = []
    for i in range(nump):
        bounds.append((lower_bounds[i], upper_bounds[i]))
        for j in range(n_starts):
            init_params[i,j] = np.random.uniform(low=lower_bounds[i], high=upper_bounds[i])

    options = {"disp":True}

    for j in range(n_starts):
        try:
            if model == "RL":
                res = op.minimize(
                eval_NLL_RL,
                init_params[:,j],
                args=(observations,actions),
                method=optimizer,
                bounds=bounds,
                options=options)
            elif model == "AI":
                res = op.minimize(
                eval_NLL_AI,
                init_params[:,j],
                args=(observations,actions,learning,mtype),
                method=optimizer,
                bounds=bounds,
                options=options) 
        
            elif model == "AI_ddm":
                res = op.minimize(
                eval_NLL_AI_ddm,
                init_params[:,j],
                args=(observations,actions,actions_hssm, rts, learning, mtype,drmtype),
                method=optimizer,
                bounds=bounds,
                options=options)
                
            elif model == "RL_ddm":
                res = op.minimize(
                eval_NLL_RL_ddm,
                init_params[:,j],
                args=(observations, actions, actions_hssm, rts, drmtype),
                method=optimizer,
                bounds=bounds,
                options=options)  
            
            elif model == "RL_ddm_biased":
                res = op.minimize(
                eval_NLL_RL_ddm_biased,
                init_params[:,j],
                args=(observations, actions, actions_hssm, rts, drmtype),
                method=optimizer,
                bounds=bounds,
                options=options) 
            
            else:
                raise ValueError(f"Unsupported model type: {model}")
  
    
            for i in range(nump):
                params[i,j] = res.x[i]
            NLL[j] = res.fun
        except Exception as e:
            print(f"MLE procedure failed for random start {j}: {e}")
            # Optionally, handle failure by setting default values, e.g.
            NLL[j] = np.nan  # or some other fallback value
            for i in range(nump):
                params[i, j] = np.nan

        #print(res.nit, res.success, res.status, res.message)

    best_iter = np.nanargmin(NLL)
    print("allparams=",params,"NLLs",NLL,"chosen",best_iter)
    return params[:,best_iter], NLL[best_iter], NLL

def MLE_procedure_DE(params, 
                  observations, 
                  actions, 
                  actions_hssm, 
                  rts, 
                  learning, 
                  lower_bounds, 
                  upper_bounds, 
                  n_starts, 
                  model, 
                  mtype, 
                  drmtype,
                  seed=1):
    """
    This function calls scipy.op.minimize() repeatedly to perform maximum likelihood estimation.

    ~~~~~~
    INPUTS
    ~~~~~~
    params: model parameters
    obs: sequence of transition and outcome observations
    actions: sequence of taken actions
    actions_hssm: sequence of actions formated for hssm and hddm packages
    rts: sequence of reaction times
    learning: learning algorithm
    lower_bounds, upper_bounds: each parameter needs a min and max bound between which the minimizer functions
    n_starts: amount of iterations. be careful of local minima in case n_starts < 10
    model: active inference (AI) or reinforcement learning (RL)
    mtype: submodel type for active inference
    """

    np.random.seed(seed)

    nump = len(params)

    # Create bounds and parameter initializations
    bounds = []
    for i in range(nump):
        bounds.append((lower_bounds[i], upper_bounds[i]))

    options = {"disp":True}

    try:
        if model == "RL":
            res = op.differential_evolution(
                    eval_NLL_RL,
                    args=(observations, actions, actions_hssm, rts, drmtype),
                    disp=True,
                    bounds=bounds) 
        elif model == "AI":
            res = op.differential_evolution(
                    eval_NLL_AI,
                    args=(observations, actions, actions_hssm, rts, drmtype),
                    disp=True,
                    bounds=bounds) 
    
        elif model == "AI_ddm":
            res = op.differential_evolution(
                    eval_NLL_AI_ddm,
                    args=(observations, actions, actions_hssm, rts, drmtype),
                    disp=True,
                    bounds=bounds) 
        
            
        elif model == "RL_ddm":
            res = op.differential_evolution(
                    eval_NLL_RL_ddm,
                    args=(observations, actions, actions_hssm, rts, drmtype),
                    disp=True,
                    bounds=bounds)
            
        elif model == "RL_ddm_biased":
            res = op.differential_evolution(
                    eval_NLL_RL_ddm_biased,
                    args=(observations, actions, actions_hssm, rts, drmtype),
                    disp=True,
                    bounds=bounds,
                    popsize = 50, # Or some reasonable number like 40-100
                    #maxiter=max(500, 100 * nump), # Give it enough iterations
                    polish=True,
                    #tol=0.01
            )
        
        else:
            raise ValueError(f"Unsupported model type: {model}")
          
    except Exception as e:
        print(f"MLE procedure DE failed WITH ERROR: {e}")
        print(traceback.format_exc()) # This will print the full traceback
        # To prevent errors later if `res` is not defined:
        return np.full(nump, np.nan), np.nan # Or some other indicator of failure


    return res.x, res.fun