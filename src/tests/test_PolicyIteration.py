# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 15:00:29 2013

@author: steve
"""

import numpy as np

import mdptoolbox

from utils import SMALLNUM, P_forest, R_forest, P_small, R_small

def test_PolicyIteration_init_policy0():
    a = mdptoolbox.mdp.PolicyIteration(P_small, R_small, 0.9)
    p = np.matrix('1; 1')
    assert (a.policy == p).all()

def test_PolicyIteration_init_policy0_exampleForest():
    a = mdptoolbox.mdp.PolicyIteration(P_forest, R_forest, 0.9)
    p = np.matrix('0, 1, 0')
    assert (a.policy == p).all()

def test_PolicyIteration_computePpolicyPRpolicy_exampleForest():
    a = mdptoolbox.mdp.PolicyIteration(P_forest, R_forest, 0.9)
    P1 = np.matrix('0.1 0.9 0; 1 0 0; 0.1 0 0.9')
    R1 = np.matrix('0, 1, 4')
    Ppolicy, Rpolicy = a._computePpolicyPRpolicy()
    assert (np.absolute(Ppolicy - P1) < SMALLNUM).all()
    assert (np.absolute(Rpolicy - R1) < SMALLNUM).all()

def test_PolicyIteration_evalPolicyIterative_exampleForest():
    v0 = np.matrix('0, 0, 0')
    v1 = np.matrix('4.47504640074458, 5.02753258879703, 23.17234211944304')
    p = np.matrix('0, 1, 0')
    a = mdptoolbox.mdp.PolicyIteration(P_forest, R_forest, 0.9)
    assert (np.absolute(a.V - v0) < SMALLNUM).all()
    a._evalPolicyIterative()
    assert (np.absolute(a.V - v1) < SMALLNUM).all()
    assert (a.policy == p).all()

def test_PolicyIteration_evalPolicyIterative_bellmanOperator_exampleForest():
    v = np.matrix('4.47504640074458, 5.02753258879703, 23.17234211944304')
    p = np.matrix('0, 0, 0')
    a = mdptoolbox.mdp.PolicyIteration(P_forest, R_forest, 0.9)
    a._evalPolicyIterative()
    policy, value = a._bellmanOperator()
    assert (policy == p).all()
    assert (np.absolute(a.V - v) < SMALLNUM).all()

def test_PolicyIteration_iterative_exampleForest():
    a = mdptoolbox.mdp.PolicyIteration(P_forest, R_forest, 0.9, eval_type=1)
    v = np.matrix('26.2439058351861, 29.4839058351861, 33.4839058351861')
    p = np.matrix('0 0 0')
    itr = 2
    assert (np.absolute(np.array(a.V) - v) < SMALLNUM).all()
    assert (np.array(a.policy) == p).all()
    assert a.iter == itr

def test_PolicyIteration_evalPolicyMatrix_exampleForest():
    v_pol = np.matrix('4.47513812154696, 5.02762430939227, 23.17243384704857')
    a = mdptoolbox.mdp.PolicyIteration(P_forest, R_forest, 0.9)
    a._evalPolicynp.matrix()
    assert (np.absolute(a.V - v_pol) < SMALLNUM).all()

def test_PolicyIteration_matrix_exampleForest():
    a = mdptoolbox.mdp.PolicyIteration(P_forest, R_forest, 0.9)
    v = np.matrix('26.2440000000000, 29.4840000000000, 33.4840000000000')
    p = np.matrix('0 0 0')
    itr = 2
    assert (np.absolute(np.array(a.V) - v) < SMALLNUM).all()
    assert (np.array(a.policy) == p).all()
    assert a.iter == itr
