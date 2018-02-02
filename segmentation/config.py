# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:52:56 2015

@author: jr
"""
import numpy as np


# Data Container is comprised of all necessary info related to the input data:
# - points: input data itself
# - assignments: final assignments (points -> super states)
# - gt: ground-truth (if it is known)
class DataContainer:
    def __init__(self, points, assignments, gt):
        self.points = points
        self.assignments = assignments
        self.gt = gt


# Model Container consists of all information related to the model:
# - beta: distribution over the super-states
# - psi: distribution over the sub-states of each super-state
# - theta: markov chain representation of each super-state
#       * row <-> from
#       * column <-> to
#       * last element of both: artificial boundary-state
# - key_reg: dictionary to translate internal sub-state representation to external one
# - L: maximum # super-states
# - L2: # unique sub-states
# - boundary: representation of the artificial boundary-state
class ModelContainer:
    def __init__(self, beta, pi, psi, theta, key_reg, l2, cfg, dmodel=None, didx=None, dtheta=None):
        self.beta = beta
        self.pi = pi
        # self.super_states = super_states
        self.psi = psi
        self.theta = theta
        self.key_reg = key_reg
        self.L2 = l2
        self.cfg = cfg
        self.dmodel = dmodel
        self.didx = didx
        self.dtheta = dtheta


class Config:
    def __init__(self):

        self.gamma = 1.0  # for beta
        self.alpha = 1.0  # for pi
        self.sigma = 1.0  # for psi
        self.lam = 1.0  # for theta

        self.d_gamma = 1.0
        self.d_lam = 1.0
        self.d_base_prob = 1E-5

        self.rho = 2.0  # 80.0  # 1000 intra-transition bias

        self.L = 100  # 75
        self.num_it = 81
        self.boundary = -1
        self.recognition_threshold = 5e-3

        self.collapsed = False
        self.it_gmm = int(1e7)
        self.sub_data_size = 200000
        self.k_gmm = 80
        self.raster_thresh = 1E-2
        self.d_importance_thresh = 200

        # path where to store final model
        self.base_path = 'data/'
        self.log_path = 'data/'
        self.eval_path = ''
        self.eval_name = 'final_model_all_2'
        self.assignments_name = 'final_assignments2_'

        self.extension = '.pckl'

        # for evaluation
        self.gt = []
        self.eval_intervals = np.array([(20, 40), (50, 80)])
        self.eval_ends = self.eval_intervals[:, 1]
        self.eval_beginnings = self.eval_intervals[:, 0]
        self.assignment_counter = dict()
        self.assignments_store = []
        # self.params_store = dict()
        self.active_counters = []
        self.elapsed_time = []
        self.base_prob = 1E-3

    def set_attr(self, L=100, num_it=100, gamma=1.0, alpha=1.0, sigma=1.0, lam=1.0, d_gamma=1.0, d_lam=1.0,
                 d_base_prob=1E-5, rho=2.0, boundary=-1, num_it_gmm=1E7, sub_sample_gmm=2E6, k_gmm=50, raster_thresh=1E-2, d_importance_trash=200,
                 model_save_dir='data/', model_save_path='model', eval_intervals=[(40, 100)]):
        self.gamma = gamma
        self.alpha = alpha
        self.sigma = sigma
        self.lam = lam

        self.d_gamma = d_gamma
        self.d_lam = d_lam
        self.d_base_prob = d_base_prob

        self.rho = rho

        self.L = L
        self.num_it = num_it
        self.boundary = boundary
        self.recognition_threshold = 5e-4

        self.collapsed = False
        self.it_gmm = num_it_gmm
        self.sub_data_size = sub_sample_gmm
        self.k_gmm = k_gmm
        self.raster_thresh = raster_thresh
        self.d_importance_thresh = d_importance_trash

        # path where to store final model
        self.base_path = model_save_dir
        self.log_path = model_save_dir
        self.eval_name = model_save_path
        self.eval_intervals = np.array(eval_intervals)
        self.eval_ends = self.eval_intervals[:, 1]
        self.eval_beginnings = self.eval_intervals[:, 0]

    def set_name(self, dname):
        self.log_path = self.base_path + dname + "/"

    def set_path(self, dname, aname, fname):
        self.set_name(dname)
        self.eval_path = self.log_path + '/' + aname + '_logs' + fname + '.csv'
        open(self.eval_path, 'w').close()
