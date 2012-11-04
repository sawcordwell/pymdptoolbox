# -*- coding: utf-8 -*-
"""
Created on Sun May 27 23:16:57 2012

@author: -
"""

from math import factorial
from mdp import exampleForest, exampleRand, MDP, PolicyIteration, ValueIteration
from numpy import array, eye, matrix, zeros
from numpy.random import rand
from scipy.sparse import eye as speye
from scipy.sparse import csr_matrix as sparse

inst = MDP()

STATES = 10
ACTIONS = 3

# check: square, stochastic and non-negative

def test_check_square_stochastic_nonnegative_array():
    P = zeros((ACTIONS, STATES, STATES))
    R = zeros((STATES, ACTIONS))
    for a in range(ACTIONS):
        P[a, :, :] = eye(STATES)
        R[:, a] = rand(STATES)
    inst.check(P, R)

# check: square, stochastic and non-negative object arrays

def test_check_square_stochastic_nonnegative_object_array():
    P = zeros((ACTIONS, ), dtype=object)
    R = zeros((STATES, ACTIONS))
    for a in range(ACTIONS):
        P[a] = eye(STATES)
        R[:, a] = rand(STATES)
    inst.check(P, R)

def test_check_square_stochastic_nonnegative_object_matrix():
    P = zeros((ACTIONS, ), dtype=object)
    R = zeros((STATES, ACTIONS))
    for a in range(ACTIONS):
        P[a] = matrix(eye(STATES))
        R[:, a] = rand(STATES)
    inst.check(P, R)

def test_check_square_stochastic_nonnegative_object_sparse():
    P = zeros((ACTIONS, ), dtype=object)
    R = zeros((STATES, ACTIONS))
    for a in range(ACTIONS):
        P[a] = speye(STATES, STATES).tocsr()
        R[:, a] = rand(STATES)
    inst.check(P, R)

# checkSquareStochastic: square, stochastic and non-negative

def test_checkSquareStochastic_square_stochastic_nonnegative_array():
    P = rand(STATES, STATES)
    for s in range(STATES):
        P[s, :] = P[s, :] / P[s, :].sum()
    assert inst.checkSquareStochastic(P) == None

def test_checkSquareStochastic_square_stochastic_nonnegative_matrix():
    P = rand(STATES, STATES)
    for s in range(STATES):
        P[s, :] = P[s, :] / P[s, :].sum()
    P = matrix(P)
    assert inst.checkSquareStochastic(P) == None

def test_checkSquareStochastic_square_stochastic_nonnegative_sparse():
    P = rand(STATES, STATES)
    for s in range(STATES):
        P[s, :] = P[s, :] / P[s, :].sum()
    P = sparse(P)
    assert inst.checkSquareStochastic(P) == None

# checkSquareStochastic: eye

def test_checkSquareStochastic_eye_array():
    P = eye(STATES)
    assert inst.checkSquareStochastic(P) == None

def test_checkSquareStochastic_eye_matrix():
    P = matrix(eye(STATES))
    assert inst.checkSquareStochastic(P) == None

def test_checkSquareStochastic_eye_sparse():
    P = speye(STATES, STATES).tocsr()
    assert inst.checkSquareStochastic(P) == None

# exampleForest

def test_exampleForest_shape():
    P, R = exampleForest()
    assert (P == array([[[0.1, 0.9, 0.0],
                         [0.1, 0.0, 0.9],
                         [0.1, 0.0, 0.9]],
                        [[1, 0, 0],
                         [1, 0, 0],
                         [1, 0, 0]]])).all()
    assert (R == array([[0, 0],
                        [0, 1],
                        [4, 2]])).all()

def test_exampleForest_check():
    P, R = exampleForest(10, 5, 3, 0.2)
    inst.check(P, R)

# exampleRand

def test_exampleRand_dense_shape():
    P, R = exampleRand(STATES, ACTIONS)
    assert (P.shape == (ACTIONS, STATES, STATES))
    assert (R.shape == (ACTIONS, STATES, STATES))

def test_exampleRand_dense_check():
    P, R = exampleRand(STATES, ACTIONS)
    inst.check(P, R)

def test_exampleRand_sparse_shape():
    P, R = exampleRand(STATES, ACTIONS, is_sparse=True)
    assert (P.shape == (ACTIONS, ))
    assert (R.shape == (ACTIONS, ))

def test_exampleRand_sparse_check():
    P, R = exampleRand(STATES, ACTIONS, is_sparse=True)
    inst.check(P, R)

# ValueIteration

def test_ValueIteration():
    P = array([[[0.5, 0.5],[0.8, 0.2]],[[0, 1],[0.1, 0.9]]])
    R = array([[5, 10], [-1, 2]])
    inst = ValueIteration(P, R, 0.9)
    inst.iterate()
    assert (inst.value == (40.048625392716822,  33.65371175967546))
    assert (inst.policy == (1, 0))
    assert (inst.iter == 26)

def test_JacksCarRental():
    S = 21 ** 2
    A = 11
    P = zeros((A, S, S))
    R = zeros((A, S, S))
    for a in range(A):
        for s in range(S):
            for s1 in range(S):
                ncL1s = int(s / 21)
                ncL2s = s - ncL1s * 21
                ncL1s1 = int(s1 / 21)
                ncL2s1 = s - ncL1s * 21
                ncs = ncL1s + ncL2s
                ncs1 = ncL1s1 + ncL2s1
                netmove = 5 - a
                pL1 = 
                pL2 = 
                p = ((3 ** n)/factorial(3)) * exp(-3)
                P[a, s, s1] = 
                R[a, s, s1] = 10 * (ncs - ncs1) - 2 * abs(a)
    
    inst = ValueIteration(P, R, 0.9)
    inst.iterate()
    assert (inst.policy == )

# checkSquareStochastic: not square, stochastic and non-negative

#@raises(ValueError(mdperr["mat_square"]))
#def test_checkSquareStochastic_notsquare_stochastic_nonnegative_array():
#    P = eye(STATES, STATES + 1)
#    inst.checkSquareStochastic(P)
#
#@raises(ValueError(mdperr["mat_square"]))
#def test_checkSquareStochastic_notsquare_stochastic_nonnegative_matrix():
#    P = matrix(eye(STATES, STATES + 1))
#    inst.checkSquareStochastic(P)
#
#@raises(ValueError(mdperr["mat_square"]))
#def test_checkSquareStochastic_notsquare_stochastic_nonnegative_sparse():
#    P = speye(STATES, STATES + 1).tocsr()
#    inst.checkSquareStochastic(P)

# checkSquareStochastic: square, not stochastic and non-negative
    
#def test_checkSquareStochastic_square_notstochastic_nonnegative_array():
#    P = eye(STATES)
#    i = randint(STATES)
#    j = randint(STATES)
#    P[i, j] = P[i, j] + 1
#    try:
#        inst.checkSquareStochastic(P)
#    except ValueError(mdperr["mat_stoch"]):
#        pass
#
#def test_checkSquareStochastic_square_notstochastic_nonnegative_matrix():
#    P = matrix(eye(STATES))
#    i = randint(STATES)
#    j = randint(STATES)
#    P[i, j] = P[i, j] + 1
#    try:
#        inst.checkSquareStochastic(P)
#    except ValueError(mdperr["mat_stoch"]):
#        pass
#
#def test_checkSquareStochastic_square_notstochastic_nonnegative_sparse():
#    P = speye(STATES, STATES).tolil()
#    i = randint(STATES)
#    j = randint(STATES)
#    P[i, j] = P[i, j] + 1
#    P = P.tocsr()
#    try:
#        inst.checkSquareStochastic(P)
#    except ValueError(mdperr["mat_stoch"]):
#        pass

# checkSquareStochastic: square, stochastic and negative

#def test_checkSquareStochastic_square_stochastic_negative_array():
#    P = eye(STATES, STATES)
#    i = randint(STATES)
#    j = randint(STATES)
#    while j == i:
#        j = randint(STATES)
#    P[i, i] = -1
#    P[i, j] = 1
#    try:
#        inst.checkSquareStochastic(P)
#    except ValueError(mdperr["mat_nonneg"]):
#        pass
#
#def test_checkSquareStochastic_square_stochastic_negative_matrix():
#    P = matrix(eye(STATES, STATES))
#    i = randint(STATES)
#    j = randint(STATES)
#    while j == i:
#        j = randint(STATES)
#    P[i, i] = -1
#    P[i, j] = 1
#    try:
#        inst.checkSquareStochastic(P)
#    except ValueError(mdperr["mat_nonneg"]):
#        pass
#
#def test_checkSquareStochastic_square_stochastic_negative_sparse():
#    P = speye(STATES, STATES)
#    i = randint(STATES)
#    j = randint(STATES)
#    while j == i:
#        j = randint(STATES)
#    P[i, i] = -1
#    P[i, j] = 1
#    try:
#        inst.checkSquareStochastic(P)
#    except ValueError(mdperr["mat_nonneg"]):
#        pass

#def test_check_square_stochastic_array_Rtranspose():
#    P = array([eye(STATES), eye(STATES)])
#    R = array([ones(STATES), ones(STATES)])
#    assert inst.check(P, R) == (True, "R is wrong way")