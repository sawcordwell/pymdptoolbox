# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 15:00:29 2013

@author: steve
"""

import numpy as np
import scipy.sparse as sp

import mdptoolbox

from .utils import SMALLNUM, P_forest, R_forest, P_small, R_small, P_sparse, \
                   P_forest_sparse, R_forest_sparse, \
                   assert_sequence_almost_equal

def test_PolicyIteration_init_policy0():
    sdp = mdptoolbox.mdp.PolicyIteration(P_small, R_small, 0.9)
    p = np.array([1, 1])
    assert (sdp.policy == p).all()

def test_PolicyIteration_init_policy0_forest():
    sdp = mdptoolbox.mdp.PolicyIteration(P_forest, R_forest, 0.9)
    p = np.array([0, 1, 0])
    assert (sdp.policy == p).all()

def test_PolicyIteration_computePpolicyPRpolicy_forest():
    sdp = mdptoolbox.mdp.PolicyIteration(P_forest, R_forest, 0.9)
    P1 = np.matrix('0.1, 0.9, 0; 1, 0, 0; 0.1, 0, 0.9').A
    R1 = np.array([0, 1, 4])
    Ppolicy, Rpolicy = sdp._computePpolicyPRpolicy()
    assert (np.absolute(Ppolicy - P1) < SMALLNUM).all()
    assert (np.absolute(Rpolicy - R1) < SMALLNUM).all()

def test_PolicyIteration_evalPolicyIterative_forest():
    v0 = np.array([0, 0, 0])
    v1 = np.array([4.47504640074458, 5.02753258879703, 23.17234211944304])
    p = np.array([0, 1, 0])
    sdp = mdptoolbox.mdp.PolicyIteration(P_forest, R_forest, 0.9)
    assert (np.absolute(sdp.V - v0) < SMALLNUM).all()
    sdp._evalPolicyIterative()
    assert (np.absolute(sdp.V - v1) < SMALLNUM).all()
    assert (sdp.policy == p).all()

def test_PolicyIteration_evalPolicyIterative_bellmanOperator_forest():
    v = np.array([4.47504640074458, 5.02753258879703, 23.17234211944304])
    p = np.array([0, 0, 0])
    sdp = mdptoolbox.mdp.PolicyIteration(P_forest, R_forest, 0.9)
    sdp._evalPolicyIterative()
    policy, value = sdp._bellmanOperator()
    assert (policy == p).all()
    assert (np.absolute(sdp.V - v) < SMALLNUM).all()

def test_PolicyIteration_iterative_forest():
    sdp = mdptoolbox.mdp.PolicyIteration(P_forest, R_forest, 0.9, eval_type=1)
    sdp.run()
    v = np.array([26.2439058351861, 29.4839058351861, 33.4839058351861])
    p = (0, 0, 0)
    itr = 2
    assert (np.absolute(np.array(sdp.V) - v) < SMALLNUM).all()
    assert sdp.policy == p
    assert sdp.iter == itr

def test_PolicyIteration_evalPolicyMatrix_forest():
    v_pol = np.matrix([4.47513812154696, 5.02762430939227, 23.17243384704857])
    sdp = mdptoolbox.mdp.PolicyIteration(P_forest, R_forest, 0.9)
    sdp._evalPolicyMatrix()
    assert (np.absolute(sdp.V - v_pol) < SMALLNUM).all()

def test_PolicyIteration_matrix_forest():
    sdp = mdptoolbox.mdp.PolicyIteration(P_forest, R_forest, 0.9)
    sdp.run()
    v = np.matrix([26.2440000000000, 29.4840000000000, 33.4840000000000])
    p = (0, 0, 0)
    itr = 2
    assert (np.absolute(np.array(sdp.V) - v) < SMALLNUM).all()
    assert sdp.policy == p
    assert sdp.iter == itr

def test_PolicyIteration_small():
    sdp = mdptoolbox.mdp.PolicyIteration(P_small, R_small, 0.9)
    sdp.run()
    v = np.array([42.4418604651163, 36.0465116279070]) # from Octave MDPtoolbox
    p = (1, 0) # from Octave MDPtoolbox
    itr = 2 # from Octave MDPtoolbox
    assert sdp.policy == p
    assert (np.absolute(np.array(sdp.V) - v) < SMALLNUM).all()
    assert sdp.iter == itr

def test_PolicyIteration_small_sparse():
    sdp = mdptoolbox.mdp.PolicyIteration(P_sparse, R_small, 0.9)
    sdp.run()
    v = np.array([42.4418604651163, 36.0465116279070]) # from Octave MDPtoolbox
    p = (1, 0) # from Octave MDPtoolbox
    itr = 2 # from Octave MDPtoolbox
    assert sdp.policy == p
    assert (np.absolute(np.array(sdp.V) - v) < SMALLNUM).all()
    assert sdp.iter == itr

def test_PolicyIterative_forest():
    sdp = mdptoolbox.mdp.PolicyIteration(P_forest, R_forest, 0.96)
    sdp.run()
    # v, p and itr from Octave MDPtoolbox
    v = np.array([74.6496000000000, 78.1056000000000, 82.1056000000000])
    p = (0, 0, 0)
    itr = 2
    assert sdp.policy == p
    assert (np.absolute(np.array(sdp.V) - v) < SMALLNUM).all()
    assert sdp.iter == itr

def test_PolicyIterative_forest_sparse():
    sdp = mdptoolbox.mdp.PolicyIteration(P_forest_sparse, R_forest_sparse,
                                         0.96)
    sdp.run()
    # v, p and itr from Octave MDPtoolbox
    v = np.array([26.8301859311444, 28.0723241686974, 29.5099841658652,
                  31.1739424959205, 33.0998201927438, 35.3288453048078,
                  37.9087354808078, 40.8947194808078, 44.3507194808078,
                  48.3507194808078])
    p = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    itr = 9
    assert sdp.policy == p
    assert (np.absolute(np.array(sdp.V) - v) < SMALLNUM).all()
    assert sdp.iter == itr

def test_goggle_code_issue_5():
    P = [sp.csr_matrix([[0.5, 0.5], [0.8, 0.2]]),
         sp.csr_matrix([[0.0, 1.0], [0.1, 0.9]])]
    P = np.array(P)
    R = np.array([[5, 10], [-1, 2]])
    pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.96)
    pi.run()
    expected = (100.67873303167413, 94.45701357466055)
    assert_sequence_almost_equal(pi.V, expected)
