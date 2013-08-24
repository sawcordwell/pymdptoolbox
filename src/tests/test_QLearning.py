# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 15:04:06 2013

@author: steve
"""

from random import seed as randseed

import numpy as np

import mdptoolbox

from utils import SMALLNUM, P_forest, R_forest, P_small, R_small

def test_QLearning_small():
    randseed(0)
    np.random.seed(0)
    a = mdptoolbox.mdp.QLearning(P_small, R_small, 0.9)
    q = np.matrix("39.9336909966907 43.175433380901488; "
               "36.943942243204454 35.42568055796341")
    v = np.matrix("43.17543338090149, 36.943942243204454")
    p = np.matrix("1 0")
    assert (np.absolute(a.Q - q) < SMALLNUM).all()
    assert (np.absolute(np.array(a.V) - v) < SMALLNUM).all()
    assert (np.array(a.policy) == p).all()

def test_QLearning_exampleForest():
    randseed(0)
    np.random.seed(0)
    a = mdptoolbox.mdp.QLearning(P_forest, R_forest, 0.9)
    q = np.matrix("26.209597296761608, 18.108253687076136; "
               "29.54356354184715, 18.116618509050486; "
               "33.61440797109655, 25.1820819845856")
    v = np.matrix("26.209597296761608, 29.54356354184715, 33.61440797109655")
    p = np.matrix("0 0 0")
    assert (np.absolute(a.Q - q) < SMALLNUM).all()
    assert (np.absolute(np.array(a.V) - v) < SMALLNUM).all()
    assert (np.array(a.policy) == p).all()