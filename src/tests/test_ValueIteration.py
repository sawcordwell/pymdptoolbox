# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 15:07:01 2013

@author: steve
"""

import numpy as np

import mdptoolbox

from utils import SMALLNUM, STATES, P_forest, R_forest, P_forest_sparse
from utils import R_forest_sparse, P_rand, R_rand, P_rand_sparse, R_rand_sparse
from utils import P_small, R_small   

def test_ValueIteration_small():
    sdp = mdptoolbox.mdp.ValueIteration(P_small, R_small, 0.9, 0.01)
    v = np.array((40.048625392716822,  33.65371175967546))
    assert (sdp.max_iter == 28)
    assert (np.absolute(np.array(sdp.V) - v) < SMALLNUM).all()
    assert (sdp.policy == (1, 0))
    assert (sdp.iter == 26)

def test_ValueIteration_forest():
    sdp = mdptoolbox.mdp.ValueIteration(P_forest, R_forest, 0.96)
    assert (np.array(sdp.policy) == np.array([0, 0, 0])).all()
    assert sdp.iter == 4

def test_ValueIteration_forest_sparse():
    sdp = mdptoolbox.mdp.ValueIteration(P_forest_sparse, R_forest_sparse, 0.96)
    assert (np.array(sdp.policy) == np.array([0] * STATES)).all()
    assert sdp.iter == 14

def test_ValueIteration_rand():
    sdp = mdptoolbox.mdp.ValueIteration(P_rand, R_rand, 0.9)
    assert sdp.policy

def test_ValueIteration_rand_sparse():
    sdp = mdptoolbox.mdp.ValueIteration(P_rand_sparse, R_rand_sparse, 0.9)
    assert sdp.policy
