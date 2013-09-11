# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 15:05:47 2013

@author: steve
"""

import numpy as np

import mdptoolbox

from .utils import SMALLNUM, P_forest, R_forest, P_forest_sparse
from .utils import R_forest_sparse, P_small, R_small, P_sparse

def test_RelativeValueIteration_small():
    sdp = mdptoolbox.mdp.RelativeValueIteration(P_small, R_small)
    sdp.run()
    p = np.matrix('1 0')
    ar = 3.88523524641183 # from Octave MDPtoolbox
    assert (np.array(sdp.policy) == p).all()
    assert np.absolute(sdp.average_reward - ar) < SMALLNUM

def test_RelativeValueIteration_small_sparse():
    sdp = mdptoolbox.mdp.RelativeValueIteration(P_sparse, R_small)
    sdp.run()
    p = np.matrix('1 0')
    ar = 3.88523524641183 # from Octave MDPtoolbox
    assert (np.array(sdp.policy) == p).all()
    assert np.absolute(sdp.average_reward - ar) < SMALLNUM

def test_RelativeValueIteration_forest():
    sdp = mdptoolbox.mdp.RelativeValueIteration(P_forest, R_forest)
    sdp.run()
    p = np.matrix('0 0 0')
    ar = 3.24000000000000 # from Octave MDPtoolbox
    assert (np.array(sdp.policy) == p).all()
    assert np.absolute(sdp.average_reward - ar) < SMALLNUM

def test_RelativeValueIteration_forest_sparse():
    sdp = mdptoolbox.mdp.RelativeValueIteration(P_forest_sparse,
                                                R_forest_sparse)
    sdp.run()
    p = np.matrix('0 0 0 0 0 0 0 0 0 0')
    ar = 1.54968195600000 # from Octave MDPtoolbox
    assert (np.array(sdp.policy) == p).all()
    assert np.absolute(sdp.average_reward - ar) < SMALLNUM
