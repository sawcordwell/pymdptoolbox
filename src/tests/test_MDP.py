# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 14:18:51 2013

@author: steve
"""

import numpy as np
import scipy.sparse as sp

import mdptoolbox.example

from .utils import SMALLNUM, P_small, R_small


class TestUnitsMDP(object):

    def test_MDP_has_startRun_method(self):
        P, R = mdptoolbox.example.small()
        sdp = mdptoolbox.mdp.MDP(P, R, None, None, None)
        sdp._startRun()
        assert sdp.time is not None

    def test_MDP_has_endRun_method(self):
        P, R = mdptoolbox.example.small()
        sdp = mdptoolbox.mdp.MDP(P, R, None, None, None)
        sdp._startRun()
        sdp.V = np.zeros(1)  # to prevent AttributeError in sdp._endRun()
        sdp.policy = np.zeros(1)  # to prevent AttributeError in sdp._endRun()
        time = sdp.time
        sdp._endRun()
        assert time != sdp.time


class TestMDP(object):
    P = (((0.0, 0.0, 0.6, 0.4, 0.0),
          (0.0, 0.0, 0.0, 0.0, 1.0),
          (0.0, 0.0, 1.0, 0.0, 0.0),
          (0.0, 0.0, 0.0, 1.0, 0.0),
          (0.0, 0.0, 0.0, 0.0, 1.0)),
         ((0.0, 0.4, 0.0, 0.0, 0.6),
          (0.0, 1.0, 0.0, 0.0, 0.0),
          (0.0, 0.0, 0.0, 0.0, 1.0),
          (0.0, 0.0, 0.0, 0.0, 1.0),
          (0.0, 0.0, 0.0, 0.0, 1.0)))
    R = (((0, 0, 0, 0, 0),
          (0, 0, 0, 0, 0),
          (0, 0, 0, 0, 0),
          (0, 0, 0, 0, 0),
          (0, 0, 0, 0, 0)),
         ((0, 0, 0, 0, 1),
          (0, 0, 0, 0, 0),
          (0, 0, 0, 0, 0),
          (0, 0, 0, 0, 0),
          (0, 0, 0, 0, 0)))
    computed_R = ((0.0, 0.0, 0.0, 0.0, 0.0),
                  (0.6, 0.0, 0.0, 0.0, 0.0))

    def test_a(self):
        P = np.array(self.P)
        R = np.array(self.R)
        sdp = mdptoolbox.mdp.MDP(P, R, None, None, None)
        assert sdp.S == 5
        assert sdp.A == 2
        assert (sdp.R[0] == np.array(self.computed_R[0])).all()
        assert (sdp.R[1] == np.array(self.computed_R[1])).all()

    def test_b(self):
        P = np.array(self.P)
        R = np.array(self.computed_R).T
        sdp = mdptoolbox.mdp.MDP(P, R, None, None, None)
        assert sdp.S == 5
        assert sdp.A == 2
        assert (sdp.R[0] == R[:, 0]).all()
        assert (sdp.R[1] == R[:, 1]).all()

    def test_c(self):
        P = np.array(self.P)
        R = (0, 1, 0, 1, 0)
        sdp = mdptoolbox.mdp.MDP(P, R, None, None, None)
        assert sdp.S == 5
        assert sdp.A == 2
        assert (sdp.R[0] == np.array(R)).all()
        assert (sdp.R[1] == np.array(R)).all()
        assert id(sdp.R[0]) == id(sdp.R[1])

    def test_d(self):
        P = [None] * 2
        P[0] = sp.csr_matrix(np.array(self.P[0]))
        P[1] = sp.csr_matrix(np.array(self.P[1]))
        R = [None] * 2
        R[0] = sp.csr_matrix(np.array(self.R[0]))
        R[1] = sp.csr_matrix(np.array(self.R[1]))
        sdp = mdptoolbox.mdp.MDP(P, R, None, None, None)
        assert sdp.S == 5
        assert sdp.A == 2
        assert (sdp.R[0] == np.array(self.computed_R[0])).all()
        assert (sdp.R[1] == np.array(self.computed_R[1])).all()

    def test_e(self):
        P = np.empty(shape=2, dtype=object)
        P[0] = np.array(self.P[0])
        P[1] = np.array(self.P[1])
        R = np.empty(shape=2, dtype=object)
        R[0] = np.array(self.R[0])
        R[1] = np.array(self.R[1])
        sdp = mdptoolbox.mdp.MDP(P, R, None, None, None)
        assert sdp.S == 5
        assert sdp.A == 2
        assert (sdp.R[0] == np.array(self.computed_R[0])).all()
        assert (sdp.R[1] == np.array(self.computed_R[1])).all()


def test_MDP_P_R_1():
    P1 = []
    P1.append(np.array(np.matrix('0.5 0.5; 0.8 0.2')))
    P1.append(np.array(np.matrix('0 1; 0.1 0.9')))
    P1 = tuple(P1)
    R1 = []
    R1.append(np.array(np.matrix('5, -1')))
    R1.append(np.array(np.matrix('10, 2')))
    R1 = tuple(R1)
    a = mdptoolbox.mdp.MDP(P_small, R_small, None, None, None)
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
    a = mdptoolbox.mdp.MDP(P_small, R, None, None, None)
    assert type(a.P) == type(P1)
    assert type(a.R) == type(R1)
    for kk in range(2):
        assert (a.P[kk] == P1[kk]).all()
        assert (np.absolute(a.R[kk] - R1[kk]) < SMALLNUM).all()


def test_MDP_P_R_3():
    P = np.array([[[0.6116, 0.3884], [0, 1]], [[0.6674, 0.3326], [0, 1]]])
    R = np.array([[[-0.2433, 0.7073],[0, 0.1871]],[[-0.0069, 0.6433],[0, 0.2898]]])
    PR = []
    PR.append(np.array(np.matrix('0.12591304, 0.1871')))
    PR.append(np.array(np.matrix('0.20935652,0.2898')))
    PR = tuple(PR)
    a = mdptoolbox.mdp.MDP(P, R, None, None, None)
    for kk in range(2):
        assert (np.absolute(a.R[kk] - PR[kk]) < SMALLNUM).all()
