# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 14:18:51 2013

@author: steve
"""

import numpy as np

import mdptoolbox, mdptoolbox.example

from utils import SMALLNUM, P_small, R_small

def test_MDP_P_R_1():
    P1 = []
    P1.append(np.array(np.matrix('0.5 0.5; 0.8 0.2')))
    P1.append(np.array(np.matrix('0 1; 0.1 0.9')))
    P1 = tuple(P1)
    R1 = []
    R1.append(np.array(np.matrix('5, -1')))
    R1.append(np.array(np.matrix('10, 2')))
    R1 = tuple(R1)
    a = mdptoolbox.mdp.MDP(P_small, R_small, 0.9, 0.01, 1)
    assert type(a.P) == type(P1)
    assert type(a.R) == type(R1)
    for kk in range(2):
        assert (a.P[kk] == P1[kk]).all()
        assert (a.R[kk] == R1[kk]).all()

def test_MDP_P_R_2():
    R = np.array([[[5, 10], [-1, 2]], [[1, 2], [3, 4]]])
    P1 = []
    P1.append(np.array(np.matrix('0.5 0.5; 0.8 0.2')))
    P1.append(np.array(np.matrix('0 1; 0.1 0.9')))
    P1 = tuple(P1)
    R1 = []
    R1.append(np.array(np.matrix('7.5, -0.4')))
    R1.append(np.array(np.matrix('2, 3.9')))
    R1 = tuple(R1)
    a = mdptoolbox.mdp.MDP(P_small, R, 0.9, 0.01, 1)
    assert type(a.P) == type(P1)
    assert type(a.R) == type(R1)
    for kk in range(2):
        assert (a.P[kk] == P1[kk]).all()
        assert (np.absolute(a.R[kk] - R1[kk]) < SMALLNUM).all()

def test_MDP_P_R_3():
    P = np.array([[[0.6116, 0.3884],[0, 1]],[[0.6674, 0.3326],[0, 1]]])
    R = np.array([[[-0.2433, 0.7073],[0, 0.1871]],[[-0.0069, 0.6433],[0, 0.2898]]])
    PR = []
    PR.append(np.array(np.matrix('0.12591304, 0.1871')))
    PR.append(np.array(np.matrix('0.20935652,0.2898')))
    PR = tuple(PR)
    a = mdptoolbox.mdp.MDP(P, R, 0.9, 0.01, 1)
    for kk in range(2):
        assert (np.absolute(a.R[kk] - PR[kk]) < SMALLNUM).all()
