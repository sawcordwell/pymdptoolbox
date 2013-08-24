# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 15:07:01 2013

@author: steve
"""

import numpy as np

import mdptoolbox

from utils import SMALLNUM, P_forest, R_forest, P_small, R_small

def test_ValueIteration_boundIter():
    inst = mdptoolbox.mdp.ValueIteration(P_small, R_small, 0.9, 0.01)
    assert (inst.max_iter == 28)

def test_ValueIteration_iterate():
    inst = mdptoolbox.mdp.ValueIteration(P_small, R_small, 0.9, 0.01)
    v = np.array((40.048625392716822,  33.65371175967546))
    assert (np.absolute(np.array(inst.V) - v) < SMALLNUM).all()
    assert (inst.policy == (1, 0))
    assert (inst.iter == 26)

def test_ValueIteration_exampleForest():
    a = mdptoolbox.mdp.ValueIteration(P_forest, R_forest, 0.96)
    assert (a.policy == np.array([0, 0, 0])).all()
    assert a.iter == 4
