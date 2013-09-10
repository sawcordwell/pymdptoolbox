# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 14:44:07 2013

@author: steve
"""

import numpy as np
import scipy as sp

import mdptoolbox.example

STATES = 10
ACTIONS = 3
SMALLNUM = 10e-12

# np.arrays
P_small = np.array([[[0.5, 0.5],[0.8, 0.2]],[[0, 1],[0.1, 0.9]]])
R_small = np.array([[5, 10], [-1, 2]])

P_sparse = np.empty(2, dtype=object)
P_sparse[0] = sp.sparse.csr_matrix([[0.5, 0.5],[0.8, 0.2]])
P_sparse[1] = sp.sparse.csr_matrix([[0, 1],[0.1, 0.9]])

P_forest, R_forest = mdptoolbox.example.forest()

P_forest_sparse, R_forest_sparse = mdptoolbox.example.forest(S=STATES,
                                                             is_sparse=True)

np.random.seed(0)
P_rand, R_rand = mdptoolbox.example.rand(STATES, ACTIONS)

np.random.seed(0)
P_rand_sparse, R_rand_sparse = mdptoolbox.example.rand(STATES, ACTIONS,
                                                       is_sparse=True)
