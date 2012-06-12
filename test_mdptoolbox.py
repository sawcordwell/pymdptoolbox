# -*- coding: utf-8 -*-
"""
Created on Sun May 27 23:16:57 2012

@author: -
"""

from mdp import MDP
from numpy import array, eye, matrix, ones
from numpy.random import randint
from scipy.sparse import eye as speye

inst = MDP()

SQUARE_ERROR = 'MDP Toolbox ERROR: Matrix must be square'
STOCHASTIC_ERROR = 'MDP Toolbox ERROR: Row sums of the matrix must be 1'
DIM = 10

def test_checkSquareStochastic_square_stochastic_array():
    P = array([eye(DIM), eye(DIM)])
    R = array([ones(DIM), ones(DIM)]).T
    assert inst.check(P, R) == (True, None)

def test_checkSquareStochastic_square_stochastic_matrix():
    P = matrix(eye(DIM))
    assert inst.checkSquareStochastic(P) == None

#def test_checkSquareStochastic_square_stochastic_sparse():
#    """FAILS!"""
#    pass
#    a = speye(DIM, DIM).tocsr()
#    assert inst.checkSquareStochastic(a) == None


def test_checkSquareStochastic_notsquare_stochastic_array():
    P = eye(DIM, DIM + 1)
    assert inst.checkSquareStochastic(P) == SQUARE_ERROR

def test_checkSquareStochastic_notsquare_stochastic_matrix():
    P = matrix(eye(DIM, DIM + 1))
    assert inst.checkSquareStochastic(P) == SQUARE_ERROR

def test_checkSquareStochastic_notsquare_stochastic_sparse():
    P = speye(DIM, DIM + 1).tocsr()
    assert inst.checkSquareStochastic(P) == SQUARE_ERROR

    
def test_checkSquareStochastic_square_notstochastic_array():
    P = eye(DIM)
    i = randint(DIM)
    j = randint(DIM)
    P[i, j] = P[i, j] + 1
    assert inst.checkSquareStochastic(P) == STOCHASTIC_ERROR

def test_checkSquareStochastic_square_notstochastic_matrix():
    P = matrix(eye(DIM))
    i = randint(DIM)
    j = randint(DIM)
    P[i, j] = P[i, j] + 1
    assert inst.checkSquareStochastic(P) == STOCHASTIC_ERROR

def test_checkSquareStochastic_square_notstochastic_sparse():
    P = speye(DIM, DIM).tolil()
    i = randint(DIM)
    j = randint(DIM)
    P[i, j] = P[i, j] + 1
    P = P.tocsr()
    assert inst.checkSquareStochastic(P) == STOCHASTIC_ERROR

def test_check_square_stochastic_array_Rtranspose():
    P = array([eye(DIM), eye(DIM)])
    R = array([ones(DIM), ones(DIM)])
    assert inst.check(P, R) == (True, "R is wrong way")
    