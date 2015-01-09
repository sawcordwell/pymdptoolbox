# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 13:41:56 2015

@author: steve
"""

import time as _time

import numpy as _np

from mdptoolbox.mdp_base import MDP

class FiniteHorizon(MDP):

    """A MDP solved using the finite-horizon backwards induction algorithm.

    Parameters
    ----------
    transitions : array
        Transition probability matrices. See the documentation for the ``MDP``
        class for details.
    reward : array
        Reward matrices or vectors. See the documentation for the ``MDP`` class
        for details.
    discount : float
        Discount factor. See the documentation for the ``MDP`` class for
        details.
    N : int
        Number of periods. Must be greater than 0.
    h : array, optional
        Terminal reward. Default: a vector of zeros.

    Data Attributes
    ---------------
    V : array
        Optimal value function. Shape = (S, N+1). ``V[:, n]`` = optimal value
        function at stage ``n`` with stage in {0, 1...N-1}. ``V[:, N]`` value
        function for terminal stage.
    policy : array
        Optimal policy. ``policy[:, n]`` = optimal policy at stage ``n`` with
        stage in {0, 1...N}. ``policy[:, N]`` = policy for stage ``N``.
    time : float
        used CPU time

    Notes
    -----
    In verbose mode, displays the current stage and policy transpose.

    Examples
    --------
    >>> import mdptoolbox, mdptoolbox.example
    >>> P, R = mdptoolbox.example.forest()
    >>> fh = mdptoolbox.mdp.FiniteHorizon(P, R, 0.9, 3)
    >>> fh.run()
    >>> fh.V
    array([[ 2.6973,  0.81  ,  0.    ,  0.    ],
           [ 5.9373,  3.24  ,  1.    ,  0.    ],
           [ 9.9373,  7.24  ,  4.    ,  0.    ]])
    >>> fh.policy
    array([[0, 0, 0],
           [0, 0, 1],
           [0, 0, 0]])

    """

    def __init__(self, transitions, reward, discount, N, h=None):
        # Initialise a finite horizon MDP.
        self.N = int(N)
        assert self.N > 0, "N must be greater than 0."
        # Initialise the base class
        MDP.__init__(self, transitions, reward, discount, None, None)
        # remove the iteration counter, it is not meaningful for backwards
        # induction
        del self.iter
        # There are value vectors for each time step up to the horizon
        self.V = _np.zeros((self.S, N + 1))
        # There are policy vectors for each time step before the horizon, when
        # we reach the horizon we don't need to make decisions anymore.
        self.policy = _np.empty((self.S, N), dtype=int)
        # Set the reward for the final transition to h, if specified.
        if h is not None:
            self.V[:, N] = h

    def run(self):
        # Run the finite horizon algorithm.
        self.time = _time.time()
        # loop through each time period
        for n in range(self.N):
            W, X = self._bellmanOperator(self.V[:, self.N - n])
            stage = self.N - n - 1
            self.V[:, stage] = X
            self.policy[:, stage] = W
            if self.verbose:
                print(("stage: %s, policy: %s") % (
                    stage, self.policy[:, stage].tolist()))
        # update time spent running
        self.time = _time.time() - self.time
        # After this we could create a tuple of tuples for the values and
        # policies.
        #self.V = tuple(tuple(self.V[:, n].tolist()) for n in range(self.N))
        #self.policy = tuple(tuple(self.policy[:, n].tolist())
        #                    for n in range(self.N))
