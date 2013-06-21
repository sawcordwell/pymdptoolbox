# -*- coding: utf-8 -*-
"""The Python Markov Decision Process (MDP) Toolbox Test Suite
===========================================================

These unit tests are written for the nosetests framwork. You will need to have
nosetests installed, and then run from the command line.

    $ cd /path/to/pymdptoolbox
    $ nostests

"""

#from random import seed as randseed

from numpy import absolute, array, empty, eye, matrix, zeros
from numpy.random import rand
from scipy.sparse import eye as speye
from scipy.sparse import csr_matrix as sparse
#from scipy.stats.distributions import poisson

import mdp

STATES = 10
ACTIONS = 3
SMALLNUM = 10e-12

# Arrays
P = array([[[0.5, 0.5],[0.8, 0.2]],[[0, 1],[0.1, 0.9]]])
R = array([[5, 10], [-1, 2]])
Ps = empty(2, dtype=object)
Ps[0] = sparse([[0.5, 0.5],[0.8, 0.2]])
Ps[1] = sparse([[0, 1],[0.1, 0.9]])
Pf, Rf = mdp.exampleForest()
Pr, Rr = mdp.exampleRand(STATES, ACTIONS)
Prs, Rrs = mdp.exampleRand(STATES, ACTIONS, is_sparse=True)

# check: square, stochastic and non-negative ndarrays

def test_check_square_stochastic_nonnegative_array_1():
    P = zeros((ACTIONS, STATES, STATES))
    R = zeros((STATES, ACTIONS))
    for a in range(ACTIONS):
        P[a, :, :] = eye(STATES)
        R[:, a] = rand(STATES)
    assert (mdp.check(P, R) == None)

def test_check_square_stochastic_nonnegative_array_2():
    P = zeros((ACTIONS, STATES, STATES))
    R = rand(ACTIONS, STATES, STATES)
    for a in range(ACTIONS):
        P[a, :, :] = eye(STATES)
    assert (mdp.check(P, R) == None)

# check: P - square, stochastic and non-negative object arrays

def test_check_P_square_stochastic_nonnegative_object_array():
    P = empty(ACTIONS, dtype=object)
    R = rand(STATES, ACTIONS)
    for a in range(ACTIONS):
        P[a] = eye(STATES)
    assert (mdp.check(P, R) == None)

def test_check_P_square_stochastic_nonnegative_object_matrix():
    P = empty(ACTIONS, dtype=object)
    R = rand(STATES, ACTIONS)
    for a in range(ACTIONS):
        P[a] = matrix(eye(STATES))
    assert (mdp.check(P, R) == None)

def test_check_P_square_stochastic_nonnegative_object_sparse():
    P = empty(ACTIONS, dtype=object)
    R = rand(STATES, ACTIONS)
    for a in range(ACTIONS):
        P[a] = speye(STATES, STATES).tocsr()
    assert (mdp.check(P, R) == None)

# check: P - square, stochastic and non-negative lists

def test_check_P_square_stochastic_nonnegative_list_array():
    P = []
    R = rand(STATES, ACTIONS)
    for a in xrange(ACTIONS):
        P.append(eye(STATES))
    assert (mdp.check(P, R) == None)

def test_check_P_square_stochastic_nonnegative_list_matrix():
    P = []
    R = rand(STATES, ACTIONS)
    for a in xrange(ACTIONS):
        P.append(matrix(eye(STATES)))
    assert (mdp.check(P, R) == None)

def test_check_P_square_stochastic_nonnegative_list_sparse():
    P = []
    R = rand(STATES, ACTIONS)
    for a in xrange(ACTIONS):
        P.append(speye(STATES, STATES).tocsr())
    assert (mdp.check(P, R) == None)

# check: P - square, stochastic and non-negative dicts

def test_check_P_square_stochastic_nonnegative_dict_array():
    P = {}
    R = rand(STATES, ACTIONS)
    for a in xrange(ACTIONS):
        P[a] = eye(STATES)
    assert (mdp.check(P, R) == None)

def test_check_P_square_stochastic_nonnegative_dict_matrix():
    P = {}
    R = rand(STATES, ACTIONS)
    for a in xrange(ACTIONS):
        P[a] = matrix(eye(STATES))
    assert (mdp.check(P, R) == None)

def test_check_P_square_stochastic_nonnegative_dict_sparse():
    P = {}
    R = rand(STATES, ACTIONS)
    for a in xrange(ACTIONS):
        P[a] = speye(STATES, STATES).tocsr()
    assert (mdp.check(P, R) == None)

# check: R - square stochastic and non-negative sparse

def test_check_R_square_stochastic_nonnegative_sparse():
    P = zeros((ACTIONS, STATES, STATES))
    R = sparse(rand(STATES, ACTIONS))
    for a in range(ACTIONS):
        P[a, :, :] = eye(STATES)
    assert (mdp.check(P, R) == None)

# check: R - square, stochastic and non-negative object arrays

def test_check_R_square_stochastic_nonnegative_object_array():
    P = zeros((ACTIONS, STATES, STATES))
    R = empty(ACTIONS, dtype=object)
    for a in range(ACTIONS):
        P[a, :, :] = eye(STATES)
        R[a] = rand(STATES, STATES)
    assert (mdp.check(P, R) == None)

def test_check_R_square_stochastic_nonnegative_object_matrix():
    P = zeros((ACTIONS, STATES, STATES))
    R = empty(ACTIONS, dtype=object)
    for a in range(ACTIONS):
        P[a, :, :] = eye(STATES)
        R[a] = matrix(rand(STATES, STATES))
    assert (mdp.check(P, R) == None)

def test_check_R_square_stochastic_nonnegative_object_sparse():
    P = zeros((ACTIONS, STATES, STATES))
    R = empty(ACTIONS, dtype=object)
    for a in range(ACTIONS):
        P[a, :, :] = eye(STATES)
        R[a] = sparse(rand(STATES, STATES))
    assert (mdp.check(P, R) == None)

# checkSquareStochastic: square, stochastic and non-negative

def test_checkSquareStochastic_square_stochastic_nonnegative_array():
    P = rand(STATES, STATES)
    for s in range(STATES):
        P[s, :] = P[s, :] / P[s, :].sum()
    assert mdp.checkSquareStochastic(P) == None

def test_checkSquareStochastic_square_stochastic_nonnegative_matrix():
    P = rand(STATES, STATES)
    for s in range(STATES):
        P[s, :] = P[s, :] / P[s, :].sum()
    P = matrix(P)
    assert mdp.checkSquareStochastic(P) == None

def test_checkSquareStochastic_square_stochastic_nonnegative_sparse():
    P = rand(STATES, STATES)
    for s in range(STATES):
        P[s, :] = P[s, :] / P[s, :].sum()
    P = sparse(P)
    assert mdp.checkSquareStochastic(P) == None

# checkSquareStochastic: eye

def test_checkSquareStochastic_eye_array():
    P = eye(STATES)
    assert mdp.checkSquareStochastic(P) == None

def test_checkSquareStochastic_eye_matrix():
    P = matrix(eye(STATES))
    assert mdp.checkSquareStochastic(P) == None

def test_checkSquareStochastic_eye_sparse():
    P = speye(STATES, STATES).tocsr()
    assert mdp.checkSquareStochastic(P) == None

# exampleForest

def test_exampleForest_P_shape():
    assert (Pf == array([[[0.1, 0.9, 0.0],
                         [0.1, 0.0, 0.9],
                         [0.1, 0.0, 0.9]],
                        [[1, 0, 0],
                         [1, 0, 0],
                         [1, 0, 0]]])).all()

def test_exampleForest_R_shape():
    assert (Rf == array([[0, 0],
                        [0, 1],
                        [4, 2]])).all()

def test_exampleForest_check():
    P, R = mdp.exampleForest(10, 5, 3, 0.2)
    assert mdp.check(P, R) == None

# exampleRand

def test_exampleRand_dense_P_shape():
    assert (Pr.shape == (ACTIONS, STATES, STATES))

def test_exampleRand_dense_R_shape():
    assert (Rr.shape == (ACTIONS, STATES, STATES))

def test_exampleRand_dense_check():
    assert mdp.check(Pr, Rr) == None

def test_exampleRand_sparse_P_shape():
    assert (len(Prs) == ACTIONS)
    for a in range(ACTIONS):
        assert (Prs[a].shape == (STATES, STATES))

def test_exampleRand_sparse_R_shape():
    assert (len(Rrs) == ACTIONS)
    for a in range(ACTIONS):
        assert (Rrs[a].shape == (STATES, STATES))

def test_exampleRand_sparse_check():
    assert mdp.check(Prs, Rrs) == None

# MDP

def test_MDP_P_R_1():
    P1 = []
    P1.append(array(matrix('0.5 0.5; 0.8 0.2')))
    P1.append(array(matrix('0 1; 0.1 0.9')))
    P1 = tuple(P1)
    R1 = []
    R1.append(array(matrix('5, -1')))
    R1.append(array(matrix('10, 2')))
    R1 = tuple(R1)
    a = mdp.MDP(P, R, 0.9, 0.01, 1)
    assert type(a.P) == type(P1)
    assert type(a.R) == type(R1)
    for kk in range(2):
        assert (a.P[kk] == P1[kk]).all()
        assert (a.R[kk] == R1[kk]).all()

def test_MDP_P_R_2():
    R = array([[[5, 10], [-1, 2]], [[1, 2], [3, 4]]])
    P1 = []
    P1.append(array(matrix('0.5 0.5; 0.8 0.2')))
    P1.append(array(matrix('0 1; 0.1 0.9')))
    P1 = tuple(P1)
    R1 = []
    R1.append(array(matrix('7.5, -0.4')))
    R1.append(array(matrix('2, 3.9')))
    R1 = tuple(R1)
    a = mdp.MDP(P, R, 0.9, 0.01, 1)
    assert type(a.P) == type(P1)
    assert type(a.R) == type(R1)
    for kk in range(2):
        assert (a.P[kk] == P1[kk]).all()
        assert (absolute(a.R[kk] - R1[kk]) < SMALLNUM).all()

def test_MDP_P_R_3():
    P = array([[[0.6116, 0.3884],[0, 1]],[[0.6674, 0.3326],[0, 1]]])
    R = array([[[-0.2433, 0.7073],[0, 0.1871]],[[-0.0069, 0.6433],[0, 0.2898]]])
    PR = []
    PR.append(array(matrix('0.12591304, 0.1871')))
    PR.append(array(matrix('0.20935652,0.2898')))
    PR = tuple(PR)
    a = mdp.MDP(P, R, 0.9, 0.01, 1)
    for kk in range(2):
        assert (absolute(a.R[kk] - PR[kk]) < SMALLNUM).all()

# LP

#def test_LP():
#    a = LP(P, R, 0.9)
#    v = matrix('42.4418604651163 36.0465116279070')
#    p = matrix('1 0')
#    a.iterate()
#    assert (array(a.policy) == p).all()
#    assert (absolute(array(a.V) - v) < SMALLNUM).all()

# PolicyIteration

def test_PolicyIteration_init_policy0():
    a = mdp.PolicyIteration(P, R, 0.9)
    p = matrix('1; 1')
    assert (a.policy == p).all()

def test_PolicyIteration_init_policy0_exampleForest():
    a = mdp.PolicyIteration(Pf, Rf, 0.9)
    p = matrix('0, 1, 0')
    assert (a.policy == p).all()

def test_PolicyIteration_computePpolicyPRpolicy_exampleForest():
    a = mdp.PolicyIteration(Pf, Rf, 0.9)
    P1 = matrix('0.1 0.9 0; 1 0 0; 0.1 0 0.9')
    R1 = matrix('0, 1, 4')
    Ppolicy, Rpolicy = a._computePpolicyPRpolicy()
    assert (absolute(Ppolicy - P1) < SMALLNUM).all()
    assert (absolute(Rpolicy - R1) < SMALLNUM).all()

def test_PolicyIteration_evalPolicyIterative_exampleForest():
    v0 = matrix('0, 0, 0')
    v1 = matrix('4.47504640074458, 5.02753258879703, 23.17234211944304')
    p = matrix('0, 1, 0')
    a = mdp.PolicyIteration(Pf, Rf, 0.9)
    assert (absolute(a.V - v0) < SMALLNUM).all()
    a._evalPolicyIterative()
    assert (absolute(a.V - v1) < SMALLNUM).all()
    assert (a.policy == p).all()

def test_PolicyIteration_evalPolicyIterative_bellmanOperator_exampleForest():
    v = matrix('4.47504640074458, 5.02753258879703, 23.17234211944304')
    p = matrix('0, 0, 0')
    a = mdp.PolicyIteration(Pf, Rf, 0.9)
    a._evalPolicyIterative()
    policy, value = a._bellmanOperator()
    assert (policy == p).all()
    assert (absolute(a.V - v) < SMALLNUM).all()

def test_PolicyIteration_iterative_exampleForest():
    a = mdp.PolicyIteration(Pf, Rf, 0.9, eval_type=1)
    v = matrix('26.2439058351861, 29.4839058351861, 33.4839058351861')
    p = matrix('0 0 0')
    itr = 2
    a.iterate()
    assert (absolute(array(a.V) - v) < SMALLNUM).all()
    assert (array(a.policy) == p).all()
    assert a.iter == itr

def test_PolicyIteration_evalPolicyMatrix_exampleForest():
    v_pol = matrix('4.47513812154696, 5.02762430939227, 23.17243384704857')
    a = mdp.PolicyIteration(Pf, Rf, 0.9)
    a._evalPolicyMatrix()
    assert (absolute(a.V - v_pol) < SMALLNUM).all()

def test_PolicyIteration_matrix_exampleForest():
    a = mdp.PolicyIteration(Pf, Rf, 0.9)
    v = matrix('26.2440000000000, 29.4840000000000, 33.4840000000000')
    p = matrix('0 0 0')
    itr = 2
    a.iterate()
    assert (absolute(array(a.V) - v) < SMALLNUM).all()
    assert (array(a.policy) == p).all()
    assert a.iter == itr

# QLearning

def test_QLearning():
    #randseed(0)
    a = mdp.QLearning(P, R, 0.9)
    #q = matrix('36.63245946346517 42.24434307022128; ' \
    #           '35.96582807367007 32.70456417451635')
    #v = matrix('42.24434307022128 35.96582807367007')
    p = matrix('1 0')
    a.iterate()
    #assert (absolute(a.Q - q) < SMALLNUM).all()
    #assert (absolute(array(a.V) - v) < SMALLNUM).all()
    assert (array(a.policy) == p).all()

def test_QLearning_exampleForest():
    a = mdp.QLearning(Pf, Rf, 0.9)
    #q = matrix('26.1841860892231 18.6273657021260; ' \
    #           '29.5880960371007 18.5901207622881; '\
    #           '33.3526406657418 25.2621054631519')
    #v = matrix('26.1841860892231 29.5880960371007 33.3526406657418')
    p = matrix('0 0 0')
    a.iterate()
    #assert (absolute(a.Q - q) < SMALLNUM).all()
    #assert (absolute(array(a.V) - v) < SMALLNUM).all()
    assert (array(a.policy) == p).all()

# RelativeValueIteration

def test_RelativeValueIteration_dense():
    a = mdp.RelativeValueIteration(P, R)
    p= matrix('1 0')
    ar = 3.88523524641183
    itr = 29
    a.iterate()
    assert (array(a.policy) == p).all()
    assert a.iter == itr
    assert absolute(a.average_reward - ar) < SMALLNUM

def test_RelativeValueIteration_sparse():
    a = mdp.RelativeValueIteration(Ps, R)
    p= matrix('1 0')
    ar = 3.88523524641183
    itr = 29
    a.iterate()
    assert (array(a.policy) == p).all()
    assert a.iter == itr
    assert absolute(a.average_reward - ar) < SMALLNUM

def test_RelativeValueIteration_exampleForest():
    a = mdp.RelativeValueIteration(Pf, Rf)
    itr = 4
    p = matrix('0 0 0')
    #v = matrix('-4.360000000000000 -0.760000000000000 3.240000000000000')
    ar = 2.43000000000000
    a.iterate()
    assert (array(a.policy) == p).all()
    assert a.iter == itr
    #assert (absolute(array(a.V) - v) < SMALLNUM).all()
    assert absolute(a.average_reward - ar) < SMALLNUM

# ValueIteration

def test_ValueIteration_boundIter():
    inst = mdp.ValueIteration(P, R, 0.9, 0.01)
    assert (inst.max_iter == 28)

def test_ValueIteration_iterate():
    inst = mdp.ValueIteration(P, R, 0.9, 0.01)
    inst.iterate()
    v = array((40.048625392716822,  33.65371175967546))
    assert (absolute(array(inst.V) - v) < SMALLNUM).all()
    assert (inst.policy == (1, 0))
    assert (inst.iter == 26)

def test_ValueIteration_exampleForest():
    a = mdp.ValueIteration(Pf, Rf, 0.96)
    a.iterate()
    assert (a.policy == array([0, 0, 0])).all()
    assert a.iter == 4

# ValueIterationGS

def test_ValueIterationGS_boundIter_exampleForest():
    a = mdp.ValueIterationGS(Pf, Rf, 0.9)
    itr = 39
    assert (a.max_iter == itr)

def test_ValueIterationGS_exampleForest():
    a = mdp.ValueIterationGS(Pf, Rf, 0.9)
    p = matrix('0 0 0')
    v = matrix('25.5833879767579 28.8306546355469 32.8306546355469')
    itr = 33
    a.iterate()
    assert (array(a.policy) == p).all()
    assert a.iter == itr
    assert (absolute(array(a.V) - v) < SMALLNUM).all()

#def test_JacksCarRental():
#    S = 21 ** 2
#    A = 11
#    P = zeros((A, S, S))
#    R = zeros((A, S, S))
#    for a in range(A):
#        for s in range(21):
#            for s1 in range(21):
#                c1s = int(s / 21)
#                c2s = s - c1s * 21
#                c1s1 = int(s1 / 21)
#                c2s1 = s - c1s * 21
#                cs = c1s + c2s
#                cs1 = c1s1 + c2s1
#                netmove = 5 - a
#                if (s1 < s):
#                    pass
#                else:
#                    pass
#                P[a, s, s1] = 1
#                R[a, s, s1] = 10 * (cs - cs1) - 2 * abs(a)
#    
#    inst = PolicyIteration(P, R, 0.9)
#    inst.iterate()
#    #assert (inst.policy == )
#
#def test_JacksCarRental2():
#    pass
#
#def test_GamblersProblem():
#    inst = ValueIteration()
#    inst.iterate()
#    #assert (inst.policy == )

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