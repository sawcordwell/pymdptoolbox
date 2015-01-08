# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

from nose.tools import assert_raises

import mdptoolbox

from .utils import ACTIONS, STATES

def test_check_square_stochastic_nonnegative_array_1():
    P = np.zeros((ACTIONS, STATES, STATES))
    R = np.zeros((STATES, ACTIONS))
    for a in range(ACTIONS):
        P[a, :, :] = np.eye(STATES)
        R[:, a] = np.random.rand(STATES)
    assert mdptoolbox.util.check(P, R) is None

def test_check_square_stochastic_nonnegative_array_2():
    P = np.zeros((ACTIONS, STATES, STATES))
    R = np.random.rand(ACTIONS, STATES, STATES)
    for a in range(ACTIONS):
        P[a, :, :] = np.eye(STATES)
    assert mdptoolbox.util.check(P, R) is None

# check: P - square, stochastic and non-negative object np.arrays

def test_check_P_square_stochastic_nonnegative_object_array():
    P = np.empty(ACTIONS, dtype=object)
    R = np.random.rand(STATES, ACTIONS)
    for a in range(ACTIONS):
        P[a] = np.eye(STATES)
    assert mdptoolbox.util.check(P, R) is None

def test_check_P_square_stochastic_nonnegative_object_matrix():
    P = np.empty(ACTIONS, dtype=object)
    R = np.random.rand(STATES, ACTIONS)
    for a in range(ACTIONS):
        P[a] = np.matrix(np.eye(STATES))
    assert mdptoolbox.util.check(P, R) is None

def test_check_P_square_stochastic_nonnegative_object_sparse():
    P = np.empty(ACTIONS, dtype=object)
    R = np.random.rand(STATES, ACTIONS)
    for a in range(ACTIONS):
        P[a] = sp.sparse.eye(STATES, STATES).tocsr()
    assert mdptoolbox.util.check(P, R) is None

# check: P - square, stochastic and non-negative lists

def test_check_P_square_stochastic_nonnegative_list_array():
    P = []
    R = np.random.rand(STATES, ACTIONS)
    for a in range(ACTIONS):
        P.append(np.eye(STATES))
    assert mdptoolbox.util.check(P, R) is None

def test_check_P_square_stochastic_nonnegative_list_matrix():
    P = []
    R = np.random.rand(STATES, ACTIONS)
    for a in range(ACTIONS):
        P.append(np.matrix(np.eye(STATES)))
    assert mdptoolbox.util.check(P, R) is None

def test_check_P_square_stochastic_nonnegative_list_sparse():
    P = []
    R = np.random.rand(STATES, ACTIONS)
    for a in range(ACTIONS):
        P.append(sp.sparse.eye(STATES, STATES).tocsr())
    assert mdptoolbox.util.check(P, R) is None

# check: P - square, stochastic and non-negative dicts

def test_check_P_square_stochastic_nonnegative_dict_array():
    P = {}
    R = np.random.rand(STATES, ACTIONS)
    for a in range(ACTIONS):
        P[a] = np.eye(STATES)
    assert mdptoolbox.util.check(P, R) is None

def test_check_P_square_stochastic_nonnegative_dict_matrix():
    P = {}
    R = np.random.rand(STATES, ACTIONS)
    for a in range(ACTIONS):
        P[a] = np.matrix(np.eye(STATES))
    assert mdptoolbox.util.check(P, R) is None

def test_check_P_square_stochastic_nonnegative_dict_sparse():
    P = {}
    R = np.random.rand(STATES, ACTIONS)
    for a in range(ACTIONS):
        P[a] = sp.sparse.eye(STATES, STATES).tocsr()
    assert mdptoolbox.util.check(P, R) is None

# check: R - square stochastic and non-negative sparse

def test_check_R_square_stochastic_nonnegative_sparse():
    P = np.zeros((ACTIONS, STATES, STATES))
    R = sp.sparse.csr_matrix(np.random.rand(STATES, ACTIONS))
    for a in range(ACTIONS):
        P[a, :, :] = np.eye(STATES)
    assert mdptoolbox.util.check(P, R) is None

# check: R - square, stochastic and non-negative object np.arrays

def test_check_R_square_stochastic_nonnegative_object_array():
    P = np.zeros((ACTIONS, STATES, STATES))
    R = np.empty(ACTIONS, dtype=object)
    for a in range(ACTIONS):
        P[a, :, :] = np.eye(STATES)
        R[a] = np.random.rand(STATES, STATES)
    assert mdptoolbox.util.check(P, R) is None

def test_check_R_square_stochastic_nonnegative_object_matrix():
    P = np.zeros((ACTIONS, STATES, STATES))
    R = np.empty(ACTIONS, dtype=object)
    for a in range(ACTIONS):
        P[a, :, :] = np.eye(STATES)
        R[a] = np.matrix(np.random.rand(STATES, STATES))
    assert mdptoolbox.util.check(P, R) is None

def test_check_R_square_stochastic_nonnegative_object_sparse():
    P = np.zeros((ACTIONS, STATES, STATES))
    R = np.empty(ACTIONS, dtype=object)
    for a in range(ACTIONS):
        P[a, :, :] = np.eye(STATES)
        R[a] = sp.sparse.csr_matrix(np.random.rand(STATES, STATES))
    assert mdptoolbox.util.check(P, R) is None

# checkSquareStochastic: square, stochastic and non-negative

def test_checkSquareStochastic_square_stochastic_nonnegative_array():
    P = np.random.rand(STATES, STATES)
    for s in range(STATES):
        P[s, :] = P[s, :] / P[s, :].sum()
    assert mdptoolbox.util.checkSquareStochastic(P) is None

def test_checkSquareStochastic_square_stochastic_nonnegative_matrix():
    P = np.random.rand(STATES, STATES)
    for s in range(STATES):
        P[s, :] = P[s, :] / P[s, :].sum()
    P = np.matrix(P)
    assert mdptoolbox.util.checkSquareStochastic(P) is None

def test_checkSquareStochastic_square_stochastic_nonnegative_sparse():
    P = np.random.rand(STATES, STATES)
    for s in range(STATES):
        P[s, :] = P[s, :] / P[s, :].sum()
    P = sp.sparse.csr_matrix(P)
    assert mdptoolbox.util.checkSquareStochastic(P) is None

# checkSquareStochastic: eye

def test_checkSquareStochastic_eye_array():
    P = np.eye(STATES)
    assert mdptoolbox.util.checkSquareStochastic(P) is None

def test_checkSquareStochastic_eye_matrix():
    P = np.matrix(np.eye(STATES))
    assert mdptoolbox.util.checkSquareStochastic(P) is None

def test_checkSquareStochastic_eye_sparse():
    P = sp.sparse.eye(STATES, STATES).tocsr()
    assert mdptoolbox.util.checkSquareStochastic(P) is None

def test_check_vector_R():
    R = np.random.rand(STATES)
    P = [np.matrix(np.eye(STATES))] * 3
    assert mdptoolbox.util.check(P, R) is None

def test_check_vector_R_error():
    R = np.random.rand(STATES+1)
    P = [np.matrix(np.eye(STATES))] * 3
    assert_raises(mdptoolbox.error.InvalidError,
                  mdptoolbox.util.check, P=P, R=R)

# Exception tests
def test_check_P_shape_error_1():
    P = np.eye(STATES)[:STATES - 1, :STATES]
    assert_raises(mdptoolbox.error.InvalidError, mdptoolbox.util.check,
                  P=P, R=np.random.rand(STATES, ACTIONS))

def test_check_P_shape_error_2():
    P = (np.random.rand(9, 9), np.random.rand(9, 9), np.random.rand(9, 5))
    assert_raises(mdptoolbox.error.InvalidError, mdptoolbox.util.check,
                  P=P, R=np.random.rand(9))

def test_check_R_shape_error_1():
    R = (np.random.rand(9, 9), np.random.rand(9, 9), np.random.rand(9, 5))
    P = np.random.rand(3, 10, 10)
    assert_raises(mdptoolbox.error.InvalidError, mdptoolbox.util.check,
                  P=P, R=R)

def test_isSqaure_tuple():
    P = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert mdptoolbox.util.isSquare(P)

def test_isSqaure_string():
    P = "a string, the wrong type"
    assert not mdptoolbox.util.isSquare(P)

def test_isStochastic_tuple():
    P = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert mdptoolbox.util.isStochastic(P)

def test_isStochastic_string():
    P = "a string, the wrong type"
    assert_raises(TypeError, mdptoolbox.util.isStochastic, matrix=P)

def test_isNonNegative_tuple():
    P = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert mdptoolbox.util.isStochastic(P)

def test_isNonNegative_string():
    P = "a string, the wrong type"
    assert_raises(TypeError, mdptoolbox.util.isStochastic, matrix=P)

def test_checkSquareStochastic_SquareError():
    P = np.eye(STATES)[:STATES - 1, :STATES]
    assert_raises(mdptoolbox.error.SquareError,
                  mdptoolbox.util.checkSquareStochastic, matrix=P)

def test_checkSquareStochastic_StochasticError():
    P = np.random.rand(STATES, STATES)
    assert_raises(mdptoolbox.error.StochasticError,
                  mdptoolbox.util.checkSquareStochastic, matrix=P)

def test_checkSquareStochastic_NonNegativeError():
    P = np.eye(STATES)
    P[0, 0] = -0.5
    P[0, 1] = 1.5
    assert_raises(mdptoolbox.error.NonNegativeError,
                  mdptoolbox.util.checkSquareStochastic, matrix=P)
