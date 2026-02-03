# This file contains code adapted from:
#
# 1. MyoSuite (https://github.com/MyoHub/myosuite/blob/main/myosuite/envs/myo/fatigue.py)

import gymnasium as gym
import mujoco
import numpy as np

class CumulativeFatigue():
    def __init__(self, mj_model, frameskip, seed=None):
        self._dt = mj_model.opt.timestep * frameskip

        self._r = 20
        # self._F = 0.1
        self._F = 0.0
        self._R = 0.02
        # self._F_l = 0.6 * 6e-4 * 5
        self._F_l = 0.0
        self._R_l = 0.6e-5 * 5

        muscle_act_ind = mj_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self.na = sum(muscle_act_ind)

        self._tauact = np.array([mj_model.actuator_dynprm[i][0] for i in range(len(muscle_act_ind)) if muscle_act_ind[i]])
        self._taudeact = np.array([mj_model.actuator_dynprm[i][1] for i in range(len(muscle_act_ind)) if muscle_act_ind[i]])

        self._MA = np.zeros((self.na,))  # Muscle Active
        self._MR = np.ones((self.na,))   # Muscle Resting
        self._MF = np.zeros((self.na,))  # Muscle Fatigue
        self._MF_L= np.zeros((self.na,))
        self.TL  = np.zeros((self.na,))  # Target Load

        self.seed(seed)

    def set_FatigueCoefficient(self, F):
        # Set Fatigue coefficients
        self._F = F
    
    def set_RecoveryCoefficient(self, R):
        # Set Recovery coefficients
        self._R = R
    
    def set_RecoveryMultiplier(self, r):
        # Set Recovery time multiplier
        self._r = r

    def compute_transfer_rate(self, activation):
        self.TL = activation.copy()

        self._LD = 1 / (self._tauact * (0.5 + 1.5 * self._MA))
        self._LR = 1 / (self._taudeact / (0.5 + 1.5 * self._MA))

        C = np.where(self.TL < self._MA, 
                     self._LR * (self.TL - self._MA), 
                     self._LD * np.minimum((self.TL - self._MA), self._MR))

        return C
    
    def compute_act(self, act):
        C = self.compute_transfer_rate(act)

        # Calculate rR
        rR = np.zeros_like(self._MA)
        idxs = self._MA >= self.TL
        rR[idxs] = self._r*self._R
        idxs = self._MA < self.TL
        rR[idxs] = self._R

        C = np.clip(C,np.maximum((-self._MA / self._dt + (self._F + self._F_l) * self._MA),  (self._MR - 1) / self._dt + rR * self._MF + self._R_l * self._MF_L    ),
                         np.minimum( ((1 - self._MA) / self._dt + (self._F + self._F_l) * self._MA),  self._MR / self._dt + rR * self._MF + self._R_l * self._MF_L  ))
            
        dMA = (C - (self._F + self._F_l)*self._MA)*self._dt
        dMR = (-C + rR*self._MF+self._R_l*self._MF_L)*self._dt
        dMF = (self._F*self._MA - rR*self._MF)*self._dt
        dMF_L = (self._F_l*self._MA - self._R_l*self._MF_L)*self._dt

        MA = np.array(self._MA) + np.array(dMA)
        MR = np.array(self._MR) + np.array(dMR)
        MF = np.array(self._MF) + np.array(dMF)
        MF_L = np.array(self._MF_L) + np.array(dMF_L)

        return MA, MR, MF, MF_L
    
    def step(self, act):
        self._MA, self._MR, self._MF, self._MF_L = self.compute_act(act)
        return self._MA, self._MR, self._MF, self._MF_L

    def get_effort(self):
        # Calculate effort
        return np.linalg.norm(self._MA - self.TL)
    
    def reset(self, fatigue_reset_vec=None):
        if fatigue_reset_vec is not None:
            self.set_fatigue_vector(fatigue_reset_vec)
        else:
            self._MA = np.zeros((self.na,))  # Muscle Active
            self._MR = np.ones((self.na,))   # Muscle Resting
            self._MF = np.zeros((self.na,))  # Muscle Fatigue
            self._MF_L = np.zeros((self.na,))

    def set_fatigue_vector(self, fatigue_vector):
        self._MF = fatigue_vector[0]
        self._MF_L = fatigue_vector[1]
        self._MR = 1 - self._MF - self._MF_L
        self._MA = np.zeros((self.na,))

    def seed(self, seed=None):
        if seed is not None:
            self.input_seed = seed
            self.np_random, seed = gym.utils.seeding.np_random(seed)

    @property
    def MF(self):
        return self._MF
    
    @property
    def MF_L(self):
        return self._MF_L
    
    @property
    def MR(self):
        return self._MR
    
    @property
    def MA(self):
        return self._MA
    
    @property
    def F(self):
        return self._F
    
    @property
    def R(self):
        return self._R
    
    @property
    def r(self):
        return self._r