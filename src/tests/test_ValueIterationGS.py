# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 15:08:01 2013

@author: steve
"""

import numpy as np

import mdptoolbox

from .utils import SMALLNUM, P_forest, R_forest, P_small, R_small, P_sparse
from .utils import P_forest_sparse, R_forest_sparse

def test_ValueIterationGS_small():
    sdp = mdptoolbox.mdp.ValueIterationGS(P_small, R_small, 0.9)
    sdp.run()
    p = (1, 0)
    itr = 28 # from Octave MDPtoolbox
    v = np.matrix('42.27744026138212, 35.89524504047155')
    assert sdp.iter == itr
    assert sdp.policy == p
    assert (np.absolute(np.array(sdp.V) - v) < SMALLNUM).all()

def test_ValueIterationGS_small_sparse():
    sdp = mdptoolbox.mdp.ValueIterationGS(P_sparse, R_small, 0.9)
    sdp.run()
    p = (1, 0)
    itr = 28 # from Octave MDPtoolbox
    v = np.matrix('42.27744026138212, 35.89524504047155')
    assert sdp.iter == itr
    assert sdp.policy == p
    assert (np.absolute(np.array(sdp.V) - v) < SMALLNUM).all()

def test_ValueIterationGS_forest():
    sdp = mdptoolbox.mdp.ValueIterationGS(P_forest, R_forest, 0.96)
    sdp.run()
    p = (0, 0, 0)
    v = np.matrix('69.98910821400665, 73.46560194552877, 77.46560194552877')
    itr = 63 # from Octave MDPtoolbox
    assert sdp.max_iter == 63
    assert sdp.policy == p
    assert sdp.iter == itr
    assert (np.absolute(np.array(sdp.V) - v) < SMALLNUM).all()

def test_ValueIterationGS_forest_sparse():
    sdp = mdptoolbox.mdp.ValueIterationGS(P_forest_sparse, R_forest_sparse,
                                          0.96)
    sdp.run()
    p = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    itr = 16 # from Octave MDPtoolbox
    assert sdp.policy == p
    assert sdp.iter == itr
