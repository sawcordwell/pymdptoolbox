# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 14:44:07 2013

@author: steve
"""

from nose.tools import assert_true
import numpy as np
import scipy as sp

import mdptoolbox.example

class BaseTestMDP(object):
    small_P, small_R = mdptoolbox.example.small()

def assert_sequence_almost_equal(a, b, spacing=10e-12):
    assert_true(all(abs(a[k] - b[k]) < spacing for k in range(len(a))))

STATES = 10
ACTIONS = 3
SMALLNUM = 10e-12

# np.arrays
P_small, R_small = mdptoolbox.example.small()

P_sparse = np.empty(2, dtype=object)
P_sparse[0] = sp.sparse.csr_matrix(P_small[0])
P_sparse[1] = sp.sparse.csr_matrix(P_small[1])

P_forest, R_forest = mdptoolbox.example.forest()

P_forest_sparse, R_forest_sparse = mdptoolbox.example.forest(S=STATES,
                                                             is_sparse=True)

np.random.seed(0)
P_rand, R_rand = mdptoolbox.example.rand(STATES, ACTIONS)

np.random.seed(0)
P_rand_sparse, R_rand_sparse = mdptoolbox.example.rand(STATES, ACTIONS,
                                                       is_sparse=True)
