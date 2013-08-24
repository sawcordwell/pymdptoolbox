# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 15:08:01 2013

@author: steve
"""

import numpy as np

import mdptoolbox

from utils import SMALLNUM, P_forest, R_forest

def test_ValueIterationGS_boundIter_exampleForest():
    a = mdptoolbox.mdp.ValueIterationGS(P_forest, R_forest, 0.9)
    itr = 39
    assert (a.max_iter == itr)

def test_ValueIterationGS_exampleForest():
    a = mdptoolbox.mdp.ValueIterationGS(P_forest, R_forest, 0.9)
    p = np.matrix('0 0 0')
    v = np.matrix('25.5833879767579 28.8306546355469 32.8306546355469')
    itr = 33
    assert (np.array(a.policy) == p).all()
    assert a.iter == itr
    assert (np.absolute(np.array(a.V) - v) < SMALLNUM).all()
