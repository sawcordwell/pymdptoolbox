# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 14:32:25 2013

@author: steve
"""

from numpy import diag, ones, where, zeros
from numpy.random import randint, random
from scipy.sparse import coo_matrix, dok_matrix

def forest(S=3, r1=4, r2=2, p=0.1, is_sparse=False):
    """Generate a MDP example based on a simple forest management scenario.
    
    This function is used to generate a transition probability
    (``A`` × ``S`` × ``S``) array ``P`` and a reward (``S`` × ``A``) matrix
    ``R`` that model the following problem. A forest is managed by two actions:
    'Wait' and 'Cut'. An action is decided each year with first the objective
    to maintain an old forest for wildlife and second to make money selling cut
    wood. Each year there is a probability ``p`` that a fire burns the forest.
    
    Here is how the problem is modelled.
    Let {1, 2 . . . ``S`` } be the states of the forest, with ``S`` being the 
    oldest. Let 'Wait' be action 1 and 'Cut' action 2.
    After a fire, the forest is in the youngest state, that is state 1.
    The transition matrix P of the problem can then be defined as follows::
        
                   | p 1-p 0.......0  |
                   | .  0 1-p 0....0  |
        P[1,:,:] = | .  .  0  .       |
                   | .  .        .    |
                   | .  .         1-p |
                   | p  0  0....0 1-p |
        
                   | 1 0..........0 |
                   | . .          . |
        P[2,:,:] = | . .          . |
                   | . .          . |
                   | . .          . |
                   | 1 0..........0 |
    
    The reward matrix R is defined as follows::
        
                 |  0  |
                 |  .  |
        R[:,1] = |  .  |
                 |  .  |
                 |  0  |
                 |  r1 |
        
                 |  0  |
                 |  1  |
        R[:,2] = |  .  |
                 |  .  |
                 |  1  |
                 |  r2 |
    
    Parameters
    ---------
    S : int, optional
        The number of states, which should be an integer greater than 0. By
        default it is 3.
    r1 : float, optional
        The reward when the forest is in its oldest state and action 'Wait' is
        performed. By default it is 4.
    r2 : float, optional
        The reward when the forest is in its oldest state and action 'Cut' is
        performed. By default it is 2.
    p : float, optional
        The probability of wild fire occurence, in the range ]0, 1[. By default
        it is 0.1.
    
    Returns
    -------
    out : tuple
        ``out[1]`` contains the transition probability matrix P with a shape of
        (A, S, S). ``out[2]`` contains the reward matrix R with a shape of
        (S, A).
    
    Examples
    --------
    >>> import mdp
    >>> P, R = mdp.exampleForest()
    >>> P
    array([[[ 0.1,  0.9,  0. ],
            [ 0.1,  0. ,  0.9],
            [ 0.1,  0. ,  0.9]],
    <BLANKLINE>
           [[ 1. ,  0. ,  0. ],
            [ 1. ,  0. ,  0. ],
            [ 1. ,  0. ,  0. ]]])
    >>> R
    array([[ 0.,  0.],
           [ 0.,  1.],
           [ 4.,  2.]])
    
    """
    assert S > 1, "The number of states S must be greater than 1."
    assert (r1 > 0) and (r2 > 0), "The rewards must be non-negative."
    assert 0 <= p <= 1, "The probability p must be in [0; 1]."
    # Definition of Transition matrix P(:,:,1) associated to action Wait
    # (action 1) and P(:,:,2) associated to action Cut (action 2)
    #             | p 1-p 0.......0  |                  | 1 0..........0 |
    #             | .  0 1-p 0....0  |                  | . .          . |
    #  P(:,:,1) = | .  .  0  .       |  and P(:,:,2) =  | . .          . |
    #             | .  .        .    |                  | . .          . |
    #             | .  .         1-p |                  | . .          . |
    #             | p  0  0....0 1-p |                  | 1 0..........0 |
    if is_sparse:
        P = []
        rows = range(S) * 2
        cols = [0] * S + range(1, S) + [S - 1]
        vals = [p] * S + [1-p] * S
        P.append(coo_matrix((vals, (rows, cols)), shape=(S,S)).tocsr())
        rows = range(S)
        cols = [0] * S
        vals = [1] * S
        P.append(coo_matrix((vals, (rows, cols)), shape=(S,S)).tocsr())
    else:
        P = zeros((2, S, S))
        P[0, :, :] = (1 - p) * diag(ones(S - 1), 1)
        P[0, :, 0] = p
        P[0, S - 1, S - 1] = (1 - p)
        P[1, :, :] = zeros((S, S))
        P[1, :, 0] = 1
    # Definition of Reward matrix R1 associated to action Wait and 
    # R2 associated to action Cut
    #           | 0  |                   | 0  |
    #           | .  |                   | 1  |
    #  R(:,1) = | .  |  and     R(:,2) = | .  |	
    #           | .  |                   | .  |
    #           | 0  |                   | 1  |                   
    #           | r1 |                   | r2 |
    R = zeros((S, 2))
    R[S - 1, 0] = r1
    R[:, 1] = ones(S)
    R[0, 1] = 0
    R[S - 1, 1] = r2
    # we want to return the generated transition and reward matrices
    return (P, R)

def rand(S, A, is_sparse=False, mask=None):
    """Generate a random Markov Decision Process.
    
    Parameters
    ----------
    S : int
        number of states (> 0)
    A : int
        number of actions (> 0)
    is_sparse : logical, optional
        false to have matrices in dense format, true to have sparse
        matrices (default false).
    mask : array or None, optional
        matrix with 0 and 1 (0 indicates a place for a zero
        probability), (SxS) (default, random)
    
    Returns
    -------
    out : tuple
        ``out[1]`` contains the transition probability matrix P with a shape of
        (A, S, S). ``out[2]`` contains the reward matrix R with a shape of
        (S, A).

    Examples
    --------
    >>> import mdp
    >>> P, R = mdp.exampleRand(5, 3)
    
    """
    # making sure the states and actions are more than one
    assert S > 1, "The number of states S must be greater than 1."
    assert A > 1, "The number of actions A must be greater than 1."
    # if the user hasn't specified a mask, then we will make a random one now
    if mask is not None:
        # the mask needs to be SxS or AxSxS
        try:
            assert mask.shape in ((S, S), (A, S, S)), "'mask' must have " \
            "dimensions S×S or A×S×S."
        except AttributeError:
            raise TypeError("'mask' must be a numpy array or matrix.")
    # generate the transition and reward matrices based on S, A and mask
    if is_sparse:
        # definition of transition matrix : square stochastic matrix
        P = [None] * A
        # definition of reward matrix (values between -1 and +1)
        R = [None] * A
        for a in xrange(A):
            # it may be more efficient to implement this by constructing lists
            # of rows, columns and values then creating a coo_matrix, but this
            # works for now
            PP = dok_matrix((S, S))
            RR = dok_matrix((S, S))
            for s in xrange(S):
                if mask is None:
                    m = random(S)
                    m[m <= 2/3.0] = 0
                    m[m > 2/3.0] = 1
                elif mask.shape == (A, S, S):
                    m = mask[a][s] # mask[a, s, :]
                else:
                    m = mask[s]
                n = int(m.sum()) # m[s, :]
                if n == 0:
                    m[randint(0, S)] = 1
                    n = 1
                cols = where(m)[0] # m[s, :]
                vals = random(n)
                vals = vals / vals.sum()
                reward = 2*random(n) - ones(n)
                PP[s, cols] = vals
                RR[s, cols] = reward
            # PP.tocsr() takes the same amount of time as PP.tocoo().tocsr()
            # so constructing PP and RR as coo_matrix in the first place is
            # probably "better"
            P[a] = PP.tocsr()
            R[a] = RR.tocsr()
    else:
        # definition of transition matrix : square stochastic matrix
        P = zeros((A, S, S))
        # definition of reward matrix (values between -1 and +1)
        R = zeros((A, S, S))
        for a in range(A):
            for s in range(S):
                # create our own random mask if there is no user supplied one
                if mask is None:
                    m = random(S)
                    r = random()
                    m[m <= r] = 0
                    m[m > r] = 1
                elif mask.shape == (A, S, S):
                    m = mask[a][s] # mask[a, s, :]
                else:
                    m = mask[s]
                # Make sure that there is atleast one transition in each state
                if m.sum() == 0:
                    m[randint(0, S)] = 1
                    n = 1
                P[a][s] = m * random(S)
                P[a][s] = P[a][s] / P[a][s].sum()
                R[a][s] = (m * (2*random(S) - ones(S, dtype=int)))
    # we want to return the generated transition and reward matrices
    return (P, R)
