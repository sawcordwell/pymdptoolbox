# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 15:04:06 2013

@author: steve
"""

import numpy as np

import mdptoolbox

from .utils import SMALLNUM, P_forest, R_forest, P_forest_sparse
from .utils import R_forest_sparse, P_small, R_small, P_sparse

def test_QLearning_small():
    np.random.seed(0)
    sdp = mdptoolbox.mdp.QLearning(P_small, R_small, 0.9)
    sdp.run()
    q = np.matrix("33.330108655211646, 40.82109564847122; "
                  "34.37431040682546, 29.672368452303164")
    v = np.matrix("40.82109564847122, 34.37431040682546")
    p = (1, 0)
    assert (np.absolute(sdp.Q - q) < SMALLNUM).all()
    assert (np.absolute(np.array(sdp.V) - v) < SMALLNUM).all()
    assert sdp.policy == p

def test_QLearning_small_sparse():
    np.random.seed(0)
    sdp = mdptoolbox.mdp.QLearning(P_sparse, R_small, 0.9)
    sdp.run()
    q = np.matrix("33.330108655211646, 40.82109564847122; "
                  "34.37431040682546, 29.672368452303164")
    v = np.matrix("40.82109564847122, 34.37431040682546")
    p = (1, 0)
    assert (np.absolute(sdp.Q - q) < SMALLNUM).all()
    assert (np.absolute(np.array(sdp.V) - v) < SMALLNUM).all()
    assert sdp.policy == p

def test_QLearning_forest():
    np.random.seed(0)
    sdp = mdptoolbox.mdp.QLearning(P_forest, R_forest, 0.96)
    sdp.run()
    q = np.matrix("11.198908998901134, 10.34652034142302; "
                  "10.74229967143465, 11.741057920409865; "
                  "2.8698000059458546, 12.259732864170232")
    v = np.matrix("11.198908998901134, 11.741057920409865, 12.259732864170232")
    p = (0, 1, 1)
    assert (np.absolute(sdp.Q - q) < SMALLNUM).all()
    assert (np.absolute(np.array(sdp.V) - v) < SMALLNUM).all()
    assert sdp.policy == p

def test_QLearning_forest_sparse():
    np.random.seed(0)
    sdp = mdptoolbox.mdp.QLearning(P_forest_sparse, R_forest_sparse, 0.96)
    sdp.run()
    p = (0, 1, 1, 1, 1, 1, 0, 0, 0, 0)
    assert sdp.policy == p
