# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 15:05:47 2013

@author: steve
"""

import numpy as np

import mdptoolbox

from utils import SMALLNUM, P_forest, R_forest, P_small, R_small, P_sparse

def test_RelativeValueIteration_dense():
    a = mdptoolbox.mdp.RelativeValueIteration(P_small, R_small)
    p= np.matrix('1 0')
    ar = 3.88523524641183
    itr = 29
    assert (np.array(a.policy) == p).all()
    assert a.iter == itr
    assert np.absolute(a.average_reward - ar) < SMALLNUM

def test_RelativeValueIteration_sparse():
    a = mdptoolbox.mdp.RelativeValueIteration(P_sparse, R_small)
    p= np.matrix('1 0')
    ar = 3.88523524641183
    itr = 29
    assert (np.array(a.policy) == p).all()
    assert a.iter == itr
    assert np.absolute(a.average_reward - ar) < SMALLNUM

def test_RelativeValueIteration_exampleForest():
    a = mdptoolbox.mdp.RelativeValueIteration(P_forest, R_forest)
    itr = 4
    p = np.matrix('0 0 0')
    #v = np.matrix('-4.360000000000000 -0.760000000000000 3.240000000000000')
    ar = 2.43000000000000
    assert (np.array(a.policy) == p).all()
    assert a.iter == itr
    #assert (np.absolute(np.array(a.V) - v) < SMALLNUM).all()
    assert np.absolute(a.average_reward - ar) < SMALLNUM
