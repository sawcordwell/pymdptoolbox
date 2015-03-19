# -*- coding: utf-8 -*-
"""Markov Decision Process (MDP) Toolbox: ``example`` module
=========================================================

The ``example`` module provides functions to generate valid MDP transition and
reward matrices.

Available functions
-------------------

:func:`~mdptoolbox.example.forest`
    A simple forest management example
:func:`~mdptoolbox.example.rand`
    A random example
:func:`~mdptoolbox.example.small`
    A very small example

"""

# Copyright (c) 2011-2014 Steven A. W. Cordwell
# Copyright (c) 2009 INRA
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   * Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#   * Neither the name of the <ORGANIZATION> nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as _np
import scipy.sparse as _sp


def forest(S=3, r1=4, r2=2, p=0.1, is_sparse=False):
    """Generate a MDP example based on a simple forest management scenario.

    This function is used to generate a transition probability
    (``A`` × ``S`` × ``S``) array ``P`` and a reward (``S`` × ``A``) matrix
    ``R`` that model the following problem. A forest is managed by two actions:
    'Wait' and 'Cut'. An action is decided each year with first the objective
    to maintain an old forest for wildlife and second to make money selling cut
    wood. Each year there is a probability ``p`` that a fire burns the forest.

    Here is how the problem is modelled.
    Let {0, 1 . . . ``S``-1 } be the states of the forest, with ``S``-1 being
    the oldest. Let 'Wait' be action 0 and 'Cut' be action 1.
    After a fire, the forest is in the youngest state, that is state 0.
    The transition matrix ``P`` of the problem can then be defined as follows::

                   | p 1-p 0.......0  |
                   | .  0 1-p 0....0  |
        P[0,:,:] = | .  .  0  .       |
                   | .  .        .    |
                   | .  .         1-p |
                   | p  0  0....0 1-p |

                   | 1 0..........0 |
                   | . .          . |
        P[1,:,:] = | . .          . |
                   | . .          . |
                   | . .          . |
                   | 1 0..........0 |

    The reward matrix R is defined as follows::

                 |  0  |
                 |  .  |
        R[:,0] = |  .  |
                 |  .  |
                 |  0  |
                 |  r1 |

                 |  0  |
                 |  1  |
        R[:,1] = |  .  |
                 |  .  |
                 |  1  |
                 |  r2 |

    Parameters
    ---------
    S : int, optional
        The number of states, which should be an integer greater than 1.
        Default: 3.
    r1 : float, optional
        The reward when the forest is in its oldest state and action 'Wait' is
        performed. Default: 4.
    r2 : float, optional
        The reward when the forest is in its oldest state and action 'Cut' is
        performed. Default: 2.
    p : float, optional
        The probability of wild fire occurence, in the range ]0, 1[. Default:
        0.1.
    is_sparse : bool, optional
        If True, then the probability transition matrices will be returned in
        sparse format, otherwise they will be in dense format. Default: False.

    Returns
    -------
    out : tuple
        ``out[0]`` contains the transition probability matrix P  and ``out[1]``
        contains the reward matrix R. If ``is_sparse=False`` then P is a numpy
        array with a shape of ``(A, S, S)`` and R is a numpy array with a shape
        of ``(S, A)``. If ``is_sparse=True`` then P is a tuple of length ``A``
        where each ``P[a]`` is a scipy sparse CSR format matrix of shape
        ``(S, S)``; R remains the same as in the case of ``is_sparse=False``.

    Examples
    --------
    >>> import mdptoolbox.example
    >>> P, R = mdptoolbox.example.forest()
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
    >>> Psp, Rsp = mdptoolbox.example.forest(is_sparse=True)
    >>> len(Psp)
    2
    >>> Psp[0]
    <3x3 sparse matrix of type '<... 'numpy.float64'>'
        with 6 stored elements in Compressed Sparse Row format>
    >>> Psp[1]
    <3x3 sparse matrix of type '<... 'numpy.int64'>'
        with 3 stored elements in Compressed Sparse Row format>
    >>> Rsp
    array([[ 0.,  0.],
           [ 0.,  1.],
           [ 4.,  2.]])
    >>> (Psp[0].todense() == P[0]).all()
    True
    >>> (Rsp == R).all()
    True

    """
    assert S > 1, "The number of states S must be greater than 1."
    assert (r1 > 0) and (r2 > 0), "The rewards must be non-negative."
    assert 0 <= p <= 1, "The probability p must be in [0; 1]."
    # Definition of Transition matrix
    if is_sparse:
        P = []
        rows = list(range(S)) * 2
        cols = [0] * S + list(range(1, S)) + [S - 1]
        vals = [p] * S + [1-p] * S
        P.append(_sp.coo_matrix((vals, (rows, cols)), shape=(S, S)).tocsr())
        rows = list(range(S))
        cols = [0] * S
        vals = [1] * S
        P.append(_sp.coo_matrix((vals, (rows, cols)), shape=(S, S)).tocsr())
    else:
        P = _np.zeros((2, S, S))
        P[0, :, :] = (1 - p) * _np.diag(_np.ones(S - 1), 1)
        P[0, :, 0] = p
        P[0, S - 1, S - 1] = (1 - p)
        P[1, :, :] = _np.zeros((S, S))
        P[1, :, 0] = 1
    # Definition of Reward matrix
    R = _np.zeros((S, 2))
    R[S - 1, 0] = r1
    R[:, 1] = _np.ones(S)
    R[0, 1] = 0
    R[S - 1, 1] = r2
    return(P, R)


def _randDense(states, actions, mask):
    """Generate random dense ``P`` and ``R``. See ``rand`` for details.

    """
    # definition of transition matrix : square stochastic matrix
    P = _np.zeros((actions, states, states))
    # definition of reward matrix (values between -1 and +1)
    R = _np.zeros((actions, states, states))
    for action in range(actions):
        for state in range(states):
            # create our own random mask if there is no user supplied one
            if mask is None:
                m = _np.random.random(states)
                r = _np.random.random()
                m[m <= r] = 0
                m[m > r] = 1
            elif mask.shape == (actions, states, states):
                m = mask[action][state]  # mask[action, state, :]
            else:
                m = mask[state]
            # Make sure that there is atleast one transition in each state
            if m.sum() == 0:
                m[_np.random.randint(0, states)] = 1
            P[action][state] = m * _np.random.random(states)
            P[action][state] = P[action][state] / P[action][state].sum()
            R[action][state] = (m * (2 * _np.random.random(states) -
                                _np.ones(states, dtype=int)))
    return(P, R)


def _randSparse(states, actions, mask):
    """Generate random sparse ``P`` and ``R``. See ``rand`` for details.

    """
    # definition of transition matrix : square stochastic matrix
    P = [None] * actions
    # definition of reward matrix (values between -1 and +1)
    R = [None] * actions
    for action in range(actions):
        # it may be more efficient to implement this by constructing lists
        # of rows, columns and values then creating a coo_matrix, but this
        # works for now
        PP = _sp.dok_matrix((states, states))
        RR = _sp.dok_matrix((states, states))
        for state in range(states):
            if mask is None:
                m = _np.random.random(states)
                m[m <= 2/3.0] = 0
                m[m > 2/3.0] = 1
            elif mask.shape == (actions, states, states):
                m = mask[action][state]  # mask[action, state, :]
            else:
                m = mask[state]
            n = int(m.sum())  # m[state, :]
            if n == 0:
                m[_np.random.randint(0, states)] = 1
                n = 1
            # find the columns of the vector that have non-zero elements
            nz = m.nonzero()
            if len(nz) == 1:
                cols = nz[0]
            else:
                cols = nz[1]
            vals = _np.random.random(n)
            vals = vals / vals.sum()
            reward = 2*_np.random.random(n) - _np.ones(n)
            PP[state, cols] = vals
            RR[state, cols] = reward
        # PP.tocsr() takes the same amount of time as PP.tocoo().tocsr()
        # so constructing PP and RR as coo_matrix in the first place is
        # probably "better"
        P[action] = PP.tocsr()
        R[action] = RR.tocsr()
    return(P, R)


def rand(S, A, is_sparse=False, mask=None):
    """Generate a random Markov Decision Process.

    Parameters
    ----------
    S : int
        Number of states (> 1)
    A : int
        Number of actions (> 1)
    is_sparse : bool, optional
        False to have matrices in dense format, True to have sparse matrices.
        Default: False.
    mask : array, optional
        Array with 0 and 1 (0 indicates a place for a zero probability), shape
        can be ``(S, S)`` or ``(A, S, S)``. Default: random.

    Returns
    -------
    out : tuple
        ``out[0]`` contains the transition probability matrix P  and ``out[1]``
        contains the reward matrix R. If ``is_sparse=False`` then P is a numpy
        array with a shape of ``(A, S, S)`` and R is a numpy array with a shape
        of ``(S, A)``. If ``is_sparse=True`` then P and R are tuples of length
        ``A``, where each ``P[a]`` is a scipy sparse CSR format matrix of shape
        ``(S, S)`` and each ``R[a]`` is a scipy sparse csr format matrix of
        shape ``(S, 1)``.

    Examples
    --------
    >>> import numpy, mdptoolbox.example
    >>> numpy.random.seed(0) # Needed to get the output below
    >>> P, R = mdptoolbox.example.rand(4, 3)
    >>> P
    array([[[ 0.21977283,  0.14889403,  0.30343592,  0.32789723],
            [ 1.        ,  0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.43718772,  0.54480359,  0.01800869],
            [ 0.39766289,  0.39997167,  0.12547318,  0.07689227]],
    <BLANKLINE>
           [[ 1.        ,  0.        ,  0.        ,  0.        ],
            [ 0.32261337,  0.15483812,  0.32271303,  0.19983549],
            [ 0.33816885,  0.2766999 ,  0.12960299,  0.25552826],
            [ 0.41299411,  0.        ,  0.58369957,  0.00330633]],
    <BLANKLINE>
           [[ 0.32343037,  0.15178596,  0.28733094,  0.23745272],
            [ 0.36348538,  0.24483321,  0.16114188,  0.23053953],
            [ 1.        ,  0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  1.        ,  0.        ]]])
    >>> R
    array([[[-0.23311696,  0.58345008,  0.05778984,  0.13608912],
            [-0.07704128,  0.        , -0.        ,  0.        ],
            [ 0.        ,  0.22419145,  0.23386799,  0.88749616],
            [-0.3691433 , -0.27257846,  0.14039354, -0.12279697]],
    <BLANKLINE>
           [[-0.77924972,  0.        , -0.        , -0.        ],
            [ 0.47852716, -0.92162442, -0.43438607, -0.75960688],
            [-0.81211898,  0.15189299,  0.8585924 , -0.3628621 ],
            [ 0.35563307, -0.        ,  0.47038804,  0.92437709]],
    <BLANKLINE>
           [[-0.4051261 ,  0.62759564, -0.20698852,  0.76220639],
            [-0.9616136 , -0.39685037,  0.32034707, -0.41984479],
            [-0.13716313,  0.        , -0.        , -0.        ],
            [ 0.        , -0.        ,  0.55810204,  0.        ]]])
    >>> numpy.random.seed(0) # Needed to get the output below
    >>> Psp, Rsp = mdptoolbox.example.rand(100, 5, is_sparse=True)
    >>> len(Psp), len(Rsp)
    (5, 5)
    >>> Psp[0]
    <100x100 sparse matrix of type '<... 'numpy.float64'>'
        with 3296 stored elements in Compressed Sparse Row format>
    >>> Rsp[0]
    <100x100 sparse matrix of type '<... 'numpy.float64'>'
        with 3296 stored elements in Compressed Sparse Row format>
    >>> # The number of non-zero elements (nnz) in P and R are equal
    >>> Psp[1].nnz == Rsp[1].nnz
    True

    """
    # making sure the states and actions are more than one
    assert S > 1, "The number of states S must be greater than 1."
    assert A > 1, "The number of actions A must be greater than 1."
    # if the user hasn't specified a mask, then we will make a random one now
    if mask is not None:
        # the mask needs to be SxS or AxSxS
        try:
            assert mask.shape in ((S, S), (A, S, S)), (
                "'mask' must have dimensions S×S or A×S×S."
            )
        except AttributeError:
            raise TypeError("'mask' must be a numpy array or matrix.")
    # generate the transition and reward matrices based on S, A and mask
    if is_sparse:
        P, R = _randSparse(S, A, mask)
    else:
        P, R = _randDense(S, A, mask)
    return(P, R)


def small():
    """A very small Markov decision process.

    The probability transition matrices are::

            | | 0.5 0.5 | |
            | | 0.8 0.2 | |
        P = |             |
            | | 0.0 1.0 | |
            | | 0.1 0.9 | |

    The reward matrix is::

        R = |  5 10 |
            | -1  2 |

    Returns
    =======
    out : tuple
        ``out[0]`` is a numpy array of the probability transition matriices.
        ``out[1]`` is a numpy arrray of the reward matrix.

    Examples
    ========
    >>> import mdptoolbox.example
    >>> P, R = mdptoolbox.example.small()
    >>> P
    array([[[ 0.5,  0.5],
            [ 0.8,  0.2]],
    <BLANKLINE>
           [[ 0. ,  1. ],
            [ 0.1,  0.9]]])
    >>> R
    array([[ 5, 10],
           [-1,  2]])

    """
    P = _np.array([[[0.5, 0.5], [0.8, 0.2]], [[0, 1], [0.1, 0.9]]])
    R = _np.array([[5, 10], [-1, 2]])
    return(P, R)
