"""
Classes to simulate an agent performing the two-step task

Original code author: Dr. Sam Gijsen
Code extension author: Alvaro Garrido Perez <alvaro.garridoperez@ugent.be> 
Date: 09-12-2025

This module is an extension of the code developed and made public by Dr. Sam Gijsen
for the paper: Active inference in the two-step task. 
The code was extended to include drift-diffusion models.
Credit goes to Dr. Gijsen for developing the original code.

Link to the original code repository: github.com/SamGijsen/AI2step

"""

import numpy as np
from scipy.special import gamma, digamma, gammaln, betaincinv
import scipy.io as sio
import hssm
import random
from utils.twostep_environment import *
from utils.twostep_support import *   

class learn_and_act():

    def __init__(self, task, model, seed=1):
        """
        DESCRIPTION: RL and Active inference agent 
            * Learns from two-step task observations
            * Acts on each stage to produce behaviour
        INPUT:  Task:
                    * type: str; drift, changepoint
                    * T: int; number of trials
                    * x: Boolean; Whether transition probabilities are resampled
                    * r: Boolean; Whether outcome probabilities are resampled
                    * delta: float; The volatility of task statistics (variance of Gaussian for drift-version)
                    * bounds: list of 2 floats; lower and upper bounds of (final-stage) outcome probabilities
                Model:
                    * act: RL, RL_ddm, AI or AI_ddm
                    if RL, then required arguments:
                        * learn: "RL"
                        * learn_transitions: False
                        * lr1: learning rate for first stage
                        * lr2: learning rate for second stage
                        * lam: lambda model parameter
                        * b1: temperature parameter for first stage softmax
                        * b2: temperature parameter for second stage softmax
                        * p: response stickiness parameter
                        * w: model-based weight
                    if AI, then:
                        * learn: "PSM"
                        * learn_transitions: False
                        * lr: learning rate
                        * lam: Precision over prior preferences
                        * vunsamp: volatility/decay rate for beliefs of unsampled actions
                        * vsamp: volatility/decay rate for beliefs of sampled actions
                        * vps: rate of predictive surprise influence on beliefs
                        * gamma1, gamma2: temperature parameter for first and second stage softmax, respectively
                        * kappa_a: precision of action-repetition habit
                        * prior_r: prior outcome probability
                    if AI_ddm, then:
                        * learn: "PSM"
                        * drmtype: Drift rate function. Can be "linear", "sigmoid", sigmoid_single_v_mod, sigmoid_single_v_max
                        * learn_transitions: False
                        * lr: learning rate
                        * lam: Precision over prior preferences
                        * vunsamp: volatility/decay rate for beliefs of unsampled actions
                        * vsamp: volatility/decay rate for beliefs of sampled actions
                        * vps: rate of predictive surprise influence on beliefs
                        * kappa_a: precision of action-repetition habit
                        * prior_r: prior outcome probability
                        * ndt: non-decision time
                        * a_bs: decision boundary
                    if RL_ddm or RL_ddm_biased, then required arguments:
                        * learn: "RL"
                        * learn_transitions: False
                        * lr1: learning rate for first stage
                        * lr2: learning rate for second stage
                        * lam: lambda model parameter
                        * p: response stickiness parameter
                        * w: model-based weight
                        * ndt: non-decision time
                        * a_bs: decision boundary
                        if RL_ddm_biased:
                            *z_prime: Bias parameter
                    if AI_ddm or RL_ddm, include also:
                        if drmtype is "linear":
                            * v_stage_0, v_stage_1: drift rates parameters for stage 0 and stage 1
                        if drmtype is "sigmoid":
                            * v_mod_stage_0, v_mod_stage_1: softmax curvature parameters for stage 0 and stage 1
                            * v_max_stage_0, v_max_stage_1: max drift rate stage 0 and stage 1
                        if drmtype is "sigmoid_single_v_mod":
                            * v_mod: softmax curvature parameters for stage 0 and stage 1
                            * v_max_stage_0, v_max_stage_1: max drift rate stage 0 and stage 1
                        if drmtype is "sigmoid_single_v_max":
                            * v_mod_stage_0, v_mod_stage_1: softmax curvature parameters for stage 0 and stage 1
                            * v_max: max drift rate for stage 0 and stage 1


        OUTCOME:
                * A sequence of agent actions
                * A sequence of agent observations
                * A sequence of agent reaction times (if model includes ddm)
                * A sequence of agent beliefs
        """

        self.task = task
        self.model = model
        self.seed = seed

        self.T = task["T"]
        self.Steps = 2

        # intialize
        self.pi = np.ones((self.T+1,2,2,6)) # T, 2 steps, Alpha/Beta, 6 rv

        if model["learn_transitions"] == False: # Set to correct Transition Probabilities
            self.pi[:,:,0,0] *= 7
            self.pi[:,:,1,0] *= 3
            self.pi[:,:,0,1] *= 3
            self.pi[:,:,1,1] *= 7

        # Integrate prior parameters
        if model["act"] == "AI" or model["act"] == "AI_ddm":
            prior_nu = 2
            self.prior = np.array([(1-self.model["prior_r"])*prior_nu, self.model["prior_r"]*prior_nu])
            for step in range(self.Steps):
                for a in range(2,6):
                    self.pi[:,step,:,a] = self.prior

        # Recordings
        self.actions = np.zeros((self.T,self.Steps)).astype(int)
        self.observations = np.zeros((self.T,self.Steps)).astype(int)
        self.common_trans = np.zeros(self.T).astype(int)
        self.rts = np.zeros((self.T,self.Steps))
        self.GQ = np.zeros((self.T,3,2))
        self.Gs = np.zeros((self.T,3,2))
        self.Qs = np.zeros((self.T,3,2))
        self.prev_a = 999
        self.o = 999

        self.Qb = np.zeros((3,2))
        self.Qf = np.zeros((3,2))

        self.counts = np.zeros((2,2)) # 2 actions by 2 final states 
        self.tm = np.array([0.5, 0.5])

        np.random.seed(seed)

        self.obs, self.p_trans, self.p_r = generate_observations_twostep(
             type=task["type"], T=self.T, delta=task["delta"],bounds=task["bounds"],change_transitions=task["x"],seed=seed)

    def sigmoid_func(self,z,v_max):
        """Function to calculate ddm drift rate with sigmoid mapping
        """
        return ((2*v_max)/(1+np.exp(-z))) - v_max

    def perform_task(self):

        for t in range(self.T):
            for step in range(self.Steps):
                if step == 0:
                    state = 0
                else:
                    state = o + 1
                    self.counts[a_t, o] += 1

                # Action selection --------------------------------------
                if self.model["act"] == "RL":
                    a_t, self.GQ[t,state,:] = self.action_selection_RL(state)

                elif self.model["act"] == "AI":
                    if step == 0:
                        gamma = self.model["gamma1"]
                    else:
                        gamma = self.model["gamma2"]

                    a_t, self.GQ[t,state,:] = self.action_selection_AI(t, state, gamma, self.model["learn"])
                
                elif self.model["act"] == "AI_ddm":
                    a_t, rt, self.GQ[t,state,:] = self.action_selection_AI_ddm(t, state, self.model["learn"], self.model["drmtype"])
                
                elif self.model["act"] == "RL_ddm":
                    a_t, rt, self.GQ[t, state, :] = self.action_selection_RL_ddm(state, self.model["drmtype"])
                
                elif self.model["act"] == "RL_ddm_biased":
                    a_t, rt, self.GQ[t, state, :] = self.action_selection_RL_ddm_biased(t, state, self.model["drmtype"])

                if step == 0: 
                    self.prev_a = np.copy(a_t)

                # Interact ------------------------------------------------
                o = self.obs[state,a_t,t] 
                
                # Determine if it was a common or uncommon transition------
                if step == 0:
                    if o == a_t:
                        self.common_trans[t] = 1
                    else:
                        self.common_trans[t] = 0
                        

                # Update -------------------------------------------------
                if self.model["learn"] in ["RL", "RL_ddm", "RL_ddm_biased"]: 
                    # Update Q-values
                    if step == 1:
                        self.Qf = self.update_SARSA(a_t, state, o)
                        self.Qb[1:,:] = np.copy(self.Qf[1:,:]) # MB equals MF for the final stage
                        self.Qb = self.update_MB()

                elif self.model["act"] in ["AI", "AI_ddm"]:
                    if step == 1:

                        ao = state*2
                        self.pi = self.PSM_learning(t, step, a_t+ao, o, self.pi, self.model["lr"], self.model["vunsamp"], self.model["vsamp"],
                                                    self.model["vps"], self.model["prior_r"], self.model["learn_transitions"])

                # Determine most likely transition matrix
                if (self.counts[0,0] + self.counts[1,1]) > (self.counts[0,1] + self.counts[1,0]):
                    self.tm = np.array([0.3, 0.7])
                if (self.counts[0,0] + self.counts[1,1]) < (self.counts[0,1] + self.counts[1,0]):
                    self.tm = np.array([0.7, 0.3])
                if (self.counts[0,0] + self.counts[1,1]) == (self.counts[0,1] + self.counts[1,0]):
                    self.tm = np.array([0.5, 0.5])

                self.actions[t,step] = a_t
                self.observations[t,step] = o 

                if self.model["act"] in ["AI_ddm", "RL_ddm", "RL_ddm_biased"]:
                    self.rts[t, step] = rt 

        if self.model["act"] in ["AI_ddm", "RL_ddm", "RL_ddm_biased"]:
            return self.actions, self.observations, self.rts, self.pi, self.p_trans, self.p_r, self.GQ
        else:
            return self.actions, self.observations, self.pi, self.p_trans, self.p_r, self.GQ


    def perform_trial(self, t, pa, po):
        """
        Advances task by one trial by advancing through by steps.
        Differences to running a full task:
        - actions are provided (pa: [1x2])
        - observations are provided (po: [1x2])
        - particularly interesting are the distributions over actions/policies, rather than actions themselves
        """

        for step in range(self.Steps):
            if step == 0:
                state = 0
            else:
                state = o + 1
                self.counts[a_t, o] += 1

            # Action selection --------------------------------------
            if self.model["act"] == "RL":
                a_t, self.GQ[t, state, :] = self.action_selection_RL(state)

            elif self.model["act"] == "AI":
                gamma = self.model["gamma1"] if step == 0 else self.model["gamma2"]
                a_t, self.GQ[t, state, :] = self.action_selection_AI(t, state, gamma, self.model["learn"])
            elif self.model["act"] == "AI_ddm":
                a_t, rt, self.GQ[t,state,:] = self.action_selection_AI_ddm(t, state, self.model["learn"], self.model["drmtype"])
            elif self.model["act"] == "RL_ddm":
                a_t, rt, self.GQ[t, state, :] = self.action_selection_RL_ddm(state, self.model["drmtype"])

            a_t = pa[step]

            if step == 0: 
                self.prev_a = np.copy(a_t)

            # Interact (Fixed) --------------------------------------
            o = po[step]

            # Update ------------------------------------------------
            if self.model["learn"] in ["RL", "RL_ddm"]:
                if step == 1:
                    self.Qf = self.update_SARSA(a_t, state, o)
                    self.Qb[1:, :] = np.copy(self.Qf[1:, :])  # MB equals MF for the final stage
                    self.Qb = self.update_MB()

            elif self.model["act"] in ["AI", "AI_ddm"] and step == 1:
                ao = state * 2
                self.pi = self.PSM_learning(
                    t, step, a_t + ao, o, self.pi, self.model["lr"], self.model["vunsamp"], self.model["vsamp"],
                    self.model["vps"], self.model["prior_r"], self.model["learn_transitions"]
                )

            # Determine most likely transition matrix
            if (self.counts[0,0] + self.counts[1,1]) > (self.counts[0,1] + self.counts[1,0]):
                self.tm = np.array([0.3, 0.7])
            if (self.counts[0,0] + self.counts[1,1]) < (self.counts[0,1] + self.counts[1,0]):
                self.tm = np.array([0.7, 0.3])
            if (self.counts[0,0] + self.counts[1,1]) == (self.counts[0,1] + self.counts[1,0]):
                self.tm = np.array([0.5, 0.5])

            self.actions[t,step] = a_t
            self.observations[t,step] = o


        return self.actions, self.observations, self.pi, self.p_trans, self.p_r, self.GQ

    def perform_trial_return_EFE_or_Q(self, t, pa, po):
        """
        Advances task by one trial by advancing through by steps.
        Differences to running a full task:
        - actions are provided (pa: [1x2])
        - observations are provided (po: [1x2])
    
        """

        for step in range(self.Steps):
            if step == 0:
                state = 0
            else:
                state = o + 1
                self.counts[a_t, o] += 1

            # Calculate trial G or Q values --------------------------------------
            if self.model["act"] == "RL":
                #a_t, self.GQ[t,state,:] = self.action_selection_RL(state)
                self.Qs[t,state,:] = self.calculate_trial_Q_values(state)

            elif self.model["act"] == "AI":
                self.Gs[t,state,:] = self.calculate_trial_EFE_values(t, state, self.model["learn"])


            a_t = pa[step]

            if step == 0: 
                self.prev_a = np.copy(a_t)

            # Interact (Fixed) --------------------------------------
            o = po[step]

            # Update ------------------------------------------------
            if self.model["learn"] == "RL": 
                # Update Q-values
                if step == 1:
                    self.Qf = self.update_SARSA(a_t, state, o)
                    self.Qb[1:,:] = np.copy(self.Qf[1:,:]) # MB equals MF for the final stage
                    self.Qb = self.update_MB()

            elif self.model["act"] == "AI" and step == 1:

                ao = state*2
                self.pi = self.PSM_learning(t, step, a_t+ao, o, self.pi, self.model["lr"], self.model["vunsamp"], self.model["vsamp"], self.model["vps"], 
                                        self.model["prior_r"], self.model["learn_transitions"])

            # Determine most likely transition matrix
            if (self.counts[0,0] + self.counts[1,1]) > (self.counts[0,1] + self.counts[1,0]):
                self.tm = np.array([0.3, 0.7])
            if (self.counts[0,0] + self.counts[1,1]) < (self.counts[0,1] + self.counts[1,0]):
                self.tm = np.array([0.7, 0.3])
            if (self.counts[0,0] + self.counts[1,1]) == (self.counts[0,1] + self.counts[1,0]):
                self.tm = np.array([0.5, 0.5])

            self.actions[t,step] = a_t
            self.observations[t,step] = o


        if self.model["act"] == "RL":
            return self.Qs
        else:
            return self.Gs

    

    def PSM_learning(self, t, step, a, o, pi, lr, vunsamp, vsamp, vps, prior_r=0.5, learn_transitions=False):

        # Predictive-Surprise Modulated learning
        prior_nu = 2

        prior = np.array([(1-prior_r)*prior_nu, prior_r*prior_nu])

        copy = np.array([0,1,2,3,4,5])
        decay = np.array([2,3,4,5])

        PS = -np.log(pi[t,1,o,a]/np.sum(pi[t,1,:,a]))
        m = vps/(1-vps) # uses vps for PS modulation
        gamma = (m*PS)/(1+m*PS)

        # Decay unsampled arms by vunsamp
        pi[t+1,1,:,copy] = np.copy(pi[t,1,:,copy])
        pi[t+1,1,:,decay] = (1-vunsamp)*pi[t,1,:,decay] + vunsamp*prior
        # Sampled arm
        # first, decay by vasmp
        pi[t+1,1,:,a] = (1-vsamp)*pi[t,1,:,a] + vsamp*prior
        # second, decay by gamma=f(vps)
        pi[t+1,1,:,a] = (1-gamma)*pi[t+1,1,:,a]
        # third, increment sampled action by lr
        pi[t+1,1,o,a] += lr

        return pi
    

    def update_SARSA(self, a2, state, o):
        # SARSA(\lambda): temporal difference learning
        # Q contains our Q-values: Q_TD(s,a)

        lr1 = self.model["lr1"]
        lr2 = self.model["lr2"]
        lam = self.model["lam"]

        PE_i = self.Qf[state,a2] - self.Qf[0,self.prev_a]
        PE_f = o - self.Qf[state,a2]
        self.Qf[0,self.prev_a] = self.Qf[0,self.prev_a] + lr1*PE_i + lr1*lam*PE_f

        self.Qf[state,a2] = self.Qf[state, a2] + lr2*PE_f

        return self.Qf


    def update_MB(self):

        self.Qb[0,:] = (1-self.tm) * np.max(self.Qb[1,:]) + self.tm*np.max(self.Qb[2,:])

        return self.Qb


    def compute_drift_EFE(self, t, step, state, lr, vunsamp, vsamp, vps, ao, lam, prior_r=0.5, learn_transitions=False):
        # Empirically compute EFE for a state

        G = np.zeros(2)
        for a in range(2):

            Gi = np.zeros(2)
            for o in range(2):
                pi_temp = np.copy(self.pi)
                Q_pi = self.PSM_learning(t, step, a+ao, o, pi_temp, lr, vunsamp, vsamp, vps, prior_r, learn_transitions)

                G[a] -= KL_dir(self.pi[t,step,:,a+ao], Q_pi[t+1,1,:,a+ao]) * (self.pi[t,step,o,a+ao]/np.sum(self.pi[t,step,:,a+ao])) # Intrinsic term


            G[a] -= 2*lam*np.log(self.pi[t,step,1,a+ao]/np.sum(self.pi[t,step,:,a+ao])) # Extrinsic term

        return G


    def action_selection_AI(self, t, state, gamma, learning, learn_transitions=False):
        """
        ~~~~~~
        INPUTS
        ~~~~~~
        t: current timepoint
        state: current state
        pi: belief distributions
        lr: learning rate (model parameter)
        vunsamp: rate of decay for beliefs on unsampled actions (model parameter)
        vsamp: rate of decay for beliefs of sampled actions (model parameter)
        vps: rate of influence of predictive surprise on beliefs of sampled actions (model parameter)
        lam: precision of prior preferences (model parameter)
        kappa_a: precision of 'action-stickiness' habit (model parameter)
        prev_a: previous first-stage action taken by the agent
        learning": type of learning algorithm
        gamma: softmax inverse temperature parameter controlling for decision noise (model parameter)
        prior_r: \alpha / (\alpha + \beta) of prior Beta-distribution, i.e. the prior reward probability(model parameter)
        learn_transitions: whether state-transition probabilities are known to be 0.3 and 0.7 
        """

        lr = self.model["lr"]
        vunsamp = self.model["vunsamp"]
        vsamp = self.model["vsamp"]
        vps = self.model["vps"]
        lam = self.model["lam"]
        kappa_a = self.model["kappa_a"]
        prior_r = self.model["prior_r"]

        if state == 0:
            step = 0
            deep = 1 # Flag deep-policy
        else:   
            step = 1
            deep = 0

        G_s0, G_s1, G_s2 = np.zeros(2), np.zeros(2), np.zeros(2)

        if state == 1 or deep:
            ao = 2
            G_s1 = self.compute_drift_EFE(t, 1, 1, lr, vunsamp, vsamp, vps, ao, lam, prior_r, learn_transitions)

        if state == 2 or deep:
            ao = 4
            G_s2 = self.compute_drift_EFE(t, 1, 2, lr, vunsamp, vsamp, vps, ao, lam, prior_r, learn_transitions)

        if state == 0:
            G = np.zeros(2)

            # Habits
            E = np.zeros(2)
            if t > 0:
                E[self.prev_a] += -np.exp(kappa_a)
                E[1-self.prev_a] += -np.exp(-kappa_a)

            G_s0 = np.concatenate((G_s1, G_s2))
            G[0] = np.dot(G_s0, np.array([ # Action 0
            1-self.tm[0], 1-self.tm[0], self.tm[0], self.tm[0]]))
            G[1] = np.dot(G_s0, np.array([ # Action 1
            1-self.tm[1], 1-self.tm[1], self.tm[1], self.tm[1]]))

            G = G + E

        elif state == 1:
            G = G_s1
        elif state == 2:
            G = G_s2

        Gg = np.clip(-G * gamma,-500,500)
        probs = np.exp(Gg)/np.sum(np.exp(Gg))

        return np.random.choice(np.arange(2),p=probs), probs   

    def action_selection_AI_ddm(self, t, state, learning, drmtype, learn_transitions=False):
        """
        ~~~~~~
        INPUTS
        ~~~~~~
        t: current timepoint
        state: current state
        pi: belief distributions
        lr: learning rate (model parameter)
        vunsamp: rate of decay for beliefs on unsampled actions (model parameter)
        vsamp: rate of decay for beliefs of sampled actions (model parameter)
        vps: rate of influence of predictive surprise on beliefs of sampled actions (model parameter)
        lam: precision of prior preferences (model parameter)
        kappa_a: precision of 'action-stickiness' habit (model parameter)
        prev_a: previous first-stage action taken by the agent
        learning: type of learning algorithm
        drmtype: Type of drift rate function (linear, sigmoid, sigmoid_single_v_mod, sigmoid_single_v_max) 
        ndt: Non-decision time
        prior_r: \alpha / (\alpha + \beta) of prior Beta-distribution, i.e. the prior reward probability(model parameter)
        learn_transitions: whether state-transition probabilities are known to be 0.3 and 0.7 
        """

        # Get model parameters
        lr = self.model["lr"]
        vunsamp = self.model["vunsamp"]
        vsamp = self.model["vsamp"]
        vps = self.model["vps"]
        lam = self.model["lam"]
        kappa_a = self.model["kappa_a"]
        prior_r = self.model["prior_r"]

        # Drift-diffusion specific parameters
        
        a_bs = self.model["a_bs"]
        ndt = self.model["ndt"]

        if state == 0:
            step = 0
            deep = 1 # Flag deep-policy    
            if drmtype == "linear":
                v_dr = self.model["v_stage_0"]
            if drmtype == "sigmoid":
                v_max = self.model["v_max_stage_0"]
                v_mod = self.model["v_mod_stage_0"]
            if drmtype == "sigmoid_single_v_mod":
                v_max = self.model["v_max_stage_0"]
                v_mod = self.model["v_mod"]
            if drmtype == "sigmoid_single_v_max":
                v_max = self.model["v_max"]
                v_mod = self.model["v_mod_stage_0"]
        else:   
            step = 1
            deep = 0
            if drmtype == "linear":
                v_dr = self.model["v_stage_1"]
            if drmtype == "sigmoid":
                v_max = self.model["v_max_stage_1"]
                v_mod = self.model["v_mod_stage_1"]
            if drmtype == "sigmoid_single_v_mod":
                v_max = self.model["v_max_stage_1"]
                v_mod = self.model["v_mod"]
            if drmtype == "sigmoid_single_v_max":
                v_max = self.model["v_max"]
                v_mod = self.model["v_mod_stage_1"]

        G_s0, G_s1, G_s2 = np.zeros(2), np.zeros(2), np.zeros(2)

        if state == 1 or deep:
            ao = 2
            G_s1 = self.compute_drift_EFE(t, 1, 1, lr, vunsamp, vsamp, vps, ao, lam, prior_r, learn_transitions)

        if state == 2 or deep:
            ao = 4
            G_s2 = self.compute_drift_EFE(t, 1, 2, lr, vunsamp, vsamp, vps, ao, lam, prior_r, learn_transitions)

        if state == 0:
            G = np.zeros(2)

            # Habits
            E = np.zeros(2)
            if t > 0:
                E[self.prev_a] += -np.exp(kappa_a)
                E[1-self.prev_a] += -np.exp(-kappa_a)

            G_s0 = np.concatenate((G_s1, G_s2))
            G[0] = np.dot(G_s0, np.array([ # Action 0
            1-self.tm[0], 1-self.tm[0], self.tm[0], self.tm[0]]))
            G[1] = np.dot(G_s0, np.array([ # Action 1
            1-self.tm[1], 1-self.tm[1], self.tm[1], self.tm[1]]))

            G = G + E

        elif state == 1:
            G = G_s1
        elif state == 2:
            G = G_s2

        delta_G = G[0]-G[1]
            
        if drmtype == "linear":
            param_dict = dict(v=v_dr*delta_G, a=a_bs, z=0.5, t=ndt)
        else:
            v_sig = self.sigmoid_func(v_mod*delta_G,v_max)
            param_dict = dict(v=v_sig, a=a_bs, z=0.5, t=ndt)
        

        # Simulate trial
        ddm_simulation = hssm.simulate_data(model="ddm", theta=param_dict, size=1,random_state= random.randint(1, 9999999))

        rt = ddm_simulation['rt'].iloc[0]

        response = ddm_simulation['response'].iloc[0] 

        if response == -1:
            a_chosen = 0
        else:
            a_chosen = 1

        return a_chosen, rt, G  

    def calculate_trial_EFE_values(self, t, state, learning, learn_transitions=False):
        """
        ~~~~~~
        INPUTS
        ~~~~~~
        t: current timepoint
        state: current state
        pi: belief distributions
        lr: learning rate (model parameter)
        vunsamp: rate of decay for beliefs on unsampled actions (model parameter)
        vsamp: rate of decay for beliefs of sampled actions (model parameter)
        vps: rate of influence of predictive surprise on beliefs of sampled actions (model parameter)
        lam: precision of prior preferences (model parameter)
        kappa_a: precision of 'action-stickiness' habit (model parameter)
        prev_a: previous first-stage action taken by the agent
        learning": type of learning algorithm
        prior_r: \alpha / (\alpha + \beta) of prior Beta-distribution, i.e. the prior reward probability(model parameter)
        learn_transitions: whether state-transition probabilities are known to be 0.3 and 0.7 
        """

        lr = self.model["lr"]
        vunsamp = self.model["vunsamp"]
        vsamp = self.model["vsamp"]
        vps = self.model["vps"]
        lam = self.model["lam"]
        kappa_a = self.model["kappa_a"]
        prior_r = self.model["prior_r"]

        if state == 0:
            step = 0
            deep = 1 # Flag deep-policy
        else:   
            step = 1
            deep = 0

        G_s0, G_s1, G_s2 = np.zeros(2), np.zeros(2), np.zeros(2)

        if state == 1 or deep:
            ao = 2
            G_s1 = self.compute_drift_EFE(t, 1, 1, lr, vunsamp, vsamp, vps, ao, lam, prior_r, learn_transitions)

        if state == 2 or deep:
            ao = 4
            G_s2 = self.compute_drift_EFE(t, 1, 2, lr, vunsamp, vsamp, vps, ao, lam, prior_r, learn_transitions)


        if state == 0:
            G = np.zeros(2)

            # Habits
            E = np.zeros(2)
            if t > 0:
                E[self.prev_a] += -np.exp(kappa_a)
                E[1-self.prev_a] += -np.exp(-kappa_a)

            G_s0 = np.concatenate((G_s1, G_s2))
            G[0] = np.dot(G_s0, np.array([ # Action 0
            1-self.tm[0], 1-self.tm[0], self.tm[0], self.tm[0]]))
            G[1] = np.dot(G_s0, np.array([ # Action 1
            1-self.tm[1], 1-self.tm[1], self.tm[1], self.tm[1]]))

            G = G + E

        elif state == 1:
            G = G_s1
        elif state == 2:
            G = G_s2

        return G 


    def action_selection_RL(self, state):
        # Softmax with step-dependent Beta (inverse temperature) parameters

        b1 = self.model["b1"]
        b2 = self.model["b2"]
        w = self.model["w"]
        p = self.model["p"]

        rep = np.zeros(2)
        if self.prev_a<2:
            rep[self.prev_a] = 1

        probs = np.zeros(2)

        if state == 0:
            for a in range(2):
                probs[a] = np.exp(b1 * (w*self.Qb[state,a] + (1-w)*self.Qf[state,a] + p*rep[a])) \
                / np.sum(np.exp(b1* (w*self.Qb[state,:] + (1-w)*self.Qf[state,:] + p*rep[:])))
        else:
            for a in range(2):
                probs[a] = np.exp(b2*self.Qf[state,a]) / np.sum(np.exp(b2*(self.Qf[state,:])))

        return int(np.random.choice(np.arange(2), p=probs)), probs

    def action_selection_RL_ddm(self, state, drmtype):
        """
        Combines RL-based softmax action selection with drift-diffusion modeling (DDM).

        ~~~~~~
        INPUTS
        ~~~~~~
        t: current timepoint
        state: current state
        drmtype: Type of drift rate function (linear, sigmoid, sigmoid_single_v_mod, sigmoid_single_v_max)
        learn_transitions: whether state-transition probabilities are known to be 0.3 and 0.7
        """
        # Model parameters
        w = self.model["w"]
        p = self.model["p"]
        a_bs = self.model["a_bs"]
        ndt = self.model["ndt"]

        # Drift-diffusion specific parameters
        if state == 0:
            step = 0
            deep = 1 # Flag deep-policy    
            if drmtype == "linear":
                v_dr = self.model["v_stage_0"]
            if drmtype == "sigmoid":
                v_max = self.model["v_max_stage_0"]
                v_mod = self.model["v_mod_stage_0"]
            if drmtype == "sigmoid_single_v_mod":
                v_max = self.model["v_max_stage_0"]
                v_mod = self.model["v_mod"]
            if drmtype == "sigmoid_single_v_max":
                v_max = self.model["v_max"]
                v_mod = self.model["v_mod_stage_0"]
        else:   
            step = 1
            deep = 0
            if drmtype == "linear":
                v_dr = self.model["v_stage_1"]
            if drmtype == "sigmoid":
                v_max = self.model["v_max_stage_1"]
                v_mod = self.model["v_mod_stage_1"]
            if drmtype == "sigmoid_single_v_mod":
                v_max = self.model["v_max_stage_1"]
                v_mod = self.model["v_mod"]
            if drmtype == "sigmoid_single_v_max":
                v_max = self.model["v_max"]
                v_mod = self.model["v_mod_stage_1"]

        # RL probabilities
        rep = np.zeros(2)
        if self.prev_a < 2:
            rep[self.prev_a] = 1

        
        Qprime = np.zeros(2)
        
        if state == 0:
            for a in range(2):
                Qprime[a] = w * self.Qb[state, a] + (1 - w) * self.Qf[state, a] + p * rep[a]
        else:
            for a in range(2):
                Qprime[a] = self.Qf[state, a]
                
        delta_Qprime = -(Qprime[0] - Qprime[1])

        if drmtype == "linear":
            drift_rate = v_dr * delta_Qprime            
        else:
            drift_rate = self.sigmoid_func(v_mod * delta_Qprime, v_max)

        param_dict = dict(v = drift_rate, a=a_bs, z=0.5, t=ndt)

        # Simulate DDM trial
        ddm_simulation = hssm.simulate_data(model="ddm", theta=param_dict, size=1, random_state=random.randint(1, 9999999))
        rt = ddm_simulation['rt'].iloc[0]
        response = ddm_simulation['response'].iloc[0]

        # Determine action based on DDM response
        if response == -1:
            action_chosen = 0
        else:
            action_chosen = 1

        #return action_chosen, rt, delta_Qprime
        return action_chosen, rt, Qprime

    def action_selection_RL_ddm_biased(self, t, state, drmtype):
        """
        Combines RL-based softmax action selection with drift-diffusion modeling (DDM).

        ~~~~~~
        INPUTS
        ~~~~~~
        t: current timepoint
        state: current state
        drmtype: Type of drift rate function (linear, sigmoid, sigmoid_single_v_mod, sigmoid_single_v_max)
        learn_transitions: whether state-transition probabilities are known to be 0.3 and 0.7
        """
        # Model parameters
        w = self.model["w"]
        p = self.model["p"]
        a_bs = self.model["a_bs"]
        ndt = self.model["ndt"]
        z_prime = self.model["z_prime"]

        # Drift-diffusion specific parameters
        if state == 0:
            step = 0
            deep = 1 # Flag deep-policy    
            if drmtype == "linear":
                v_dr = self.model["v_stage_0"]
            if drmtype == "sigmoid":
                v_max = self.model["v_max_stage_0"]
                v_mod = self.model["v_mod_stage_0"]
            if drmtype == "sigmoid_single_v_mod":
                v_max = self.model["v_max_stage_0"]
                v_mod = self.model["v_mod"]
            if drmtype == "sigmoid_single_v_max":
                v_max = self.model["v_max"]
                v_mod = self.model["v_mod_stage_0"]
        else:   
            step = 1
            deep = 0
            if drmtype == "linear":
                v_dr = self.model["v_stage_1"]
            if drmtype == "sigmoid":
                v_max = self.model["v_max_stage_1"]
                v_mod = self.model["v_mod_stage_1"]
            if drmtype == "sigmoid_single_v_mod":
                v_max = self.model["v_max_stage_1"]
                v_mod = self.model["v_mod"]
            if drmtype == "sigmoid_single_v_max":
                v_max = self.model["v_max"]
                v_mod = self.model["v_mod_stage_1"]

        # RL probabilities
        rep = np.zeros(2)
        if self.prev_a < 2:
            rep[self.prev_a] = 1

        
        Qprime = np.zeros(2)
        
        if state == 0:
            for a in range(2):
                Qprime[a] = w * self.Qb[state, a] + (1 - w) * self.Qf[state, a] + p * rep[a]
            delta_Qprime = -(Qprime[0] - Qprime[1])
            z_bias  = z_prime * np.sign(delta_Qprime) + 0.5
        
        else:
            for a in range(2):
                Qprime[a] = self.Qf[state, a]
                
            delta_Qprime = -(Qprime[0] - Qprime[1])
            if self.common_trans[t] == 1:
                z_bias = z_prime * np.sign(delta_Qprime) + 0.5
            else:
                z_bias = 0.5

        if drmtype == "linear":
            drift_rate = v_dr * delta_Qprime            
        else:
            drift_rate = self.sigmoid_func(v_mod * delta_Qprime, v_max)

        param_dict = dict(v = drift_rate, a=a_bs, z=z_bias, t=ndt)

        # Simulate DDM trial
        ddm_simulation = hssm.simulate_data(model="ddm", theta=param_dict, size=1, random_state=random.randint(1, 9999999))
        rt = ddm_simulation['rt'].iloc[0]
        response = ddm_simulation['response'].iloc[0]

        # Determine action based on DDM response
        if response == -1:
            action_chosen = 0
        else:
            action_chosen = 1

        #return action_chosen, rt, delta_Qprime
        return action_chosen, rt, Qprime

    def calculate_trial_Q_values(self, state):
        """
        Combines RL-based softmax action selection with drift-diffusion modeling (DDM).

        ~~~~~~
        INPUTS
        ~~~~~~
        t: current timepoint
        state: current state
        drmtype: Type of drift rate function (linear, sigmoid, sigmoid_single_v_mod, sigmoid_single_v_max)
        learn_transitions: whether state-transition probabilities are known to be 0.3 and 0.7
        """
        
        # Model parameters
        w = self.model["w"]
        p = self.model["p"]

        # RL probabilities
        rep = np.zeros(2)
        if self.prev_a < 2:
            rep[self.prev_a] = 1
        
        Qprime = np.zeros(2)
        
        if state == 0:
            for a in range(2):
                Qprime[a] = w * self.Qb[state, a] + (1 - w) * self.Qf[state, a] + p * rep[a]
        else:
            for a in range(2):
                Qprime[a] = self.Qf[state, a]
                
        return Qprime

    

    def update_transitions(t, pi, a, o, learn_transitions=False):
        if learn_transitions:
            if t>0:
                pi[t,0,:,:] = np.copy(pi[t-1,0,:,:])
            pi[t,0,o,a] += 1
        else:
            pi[t,0,0,0:2] = [7,3]
            pi[t,0,1,0:2] = [3,7]

        return pi