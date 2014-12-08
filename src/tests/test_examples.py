# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 14:55:05 2013

@author: steve
"""

import numpy as np

import mdptoolbox.example

from .utils import ACTIONS, STATES, P_forest, R_forest, P_rand, R_rand
from .utils import P_rand_sparse, R_rand_sparse

def test_example_forest_P_shape():
    assert (P_forest == np.array([[[0.1, 0.9, 0.0],
                         [0.1, 0.0, 0.9],
                         [0.1, 0.0, 0.9]],
                        [[1, 0, 0],
                         [1, 0, 0],
                         [1, 0, 0]]])).all()

def test_example_forest_R_shape():
    assert (R_forest == np.array([[0, 0],
                        [0, 1],
                        [4, 2]])).all()

def test_example_forest_check():
    P, R = mdptoolbox.example.forest(10, 5, 3, 0.2)
    assert mdptoolbox.util.check(P, R) == None

# exampleRand

def test_example_rand_dense_P_shape():
    assert (P_rand.shape == (ACTIONS, STATES, STATES))

def test_example_rand_dense_R_shape():
    assert (R_rand.shape == (ACTIONS, STATES, STATES))

def test_example_rand_dense_check():
    assert mdptoolbox.util.check(P_rand, R_rand) == None

def test_example_rand_sparse_P_shape():
    assert (len(P_rand_sparse) == ACTIONS)
    for a in range(ACTIONS):
        assert (P_rand_sparse[a].shape == (STATES, STATES))

def test_example_rand_sparse_R_shape():
    assert (len(R_rand_sparse) == ACTIONS)
    for a in range(ACTIONS):
        assert (R_rand_sparse[a].shape == (STATES, STATES))

def test_example_rand_sparse_check():
    assert mdptoolbox.util.check(P_rand_sparse, R_rand_sparse) == None
