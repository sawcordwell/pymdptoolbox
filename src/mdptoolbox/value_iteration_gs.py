# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 13:51:08 2015

@author: steve
"""

import time as _time

import numpy as _np

import mdptoolbox.util as _util
from mdptoolbox.value_iteration import MDP, ValueIteration 

class ValueIterationGS(ValueIteration):

    """
    A discounted MDP solved using the value iteration Gauss-Seidel algorithm.

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
    epsilon : float, optional
        Stopping criterion. See the documentation for the ``MDP`` class for
        details. Default: 0.01.
    max_iter : int, optional
        Maximum number of iterations. See the documentation for the ``MDP``
        and ``ValueIteration`` classes for details. Default: computed.
    initial_value : array, optional
        The starting value function. Default: a vector of zeros.

    Data Attribues
    --------------
    policy : tuple
        epsilon-optimal policy
    iter : int
        number of done iterations
    time : float
        used CPU time

    Notes
    -----
    In verbose mode, at each iteration, displays the variation of V
    and the condition which stopped iterations: epsilon-optimum policy found
    or maximum number of iterations reached.

    Examples
    --------
    >>> import mdptoolbox.example, numpy as np
    >>> P, R = mdptoolbox.example.forest()
    >>> vigs = mdptoolbox.mdp.ValueIterationGS(P, R, 0.9)
    >>> vigs.run()
    >>> expected = (25.5833879767579, 28.830654635546928, 32.83065463554693)
    >>> all(expected[k] - vigs.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> vigs.policy
    (0, 0, 0)

    """

    def __init__(self, transitions, reward, discount, epsilon=0.01,
                 max_iter=10, initial_value=0):
        # Initialise a value iteration Gauss-Seidel MDP.

        MDP.__init__(self, transitions, reward, discount, epsilon, max_iter)

        # initialization of optional arguments
        if initial_value == 0:
            self.V = _np.zeros(self.S)
        else:
            if len(initial_value) != self.S:
                raise ValueError("The initial value must be a vector of "
                                 "length S.")
            else:
                try:
                    self.V = initial_value.reshape(self.S)
                except AttributeError:
                    self.V = _np.array(initial_value)
                except:
                    raise
        if self.discount < 1:
            # compute a bound for the number of iterations and update the
            # stored value of self.max_iter
            self._boundIter(epsilon)
            # computation of threshold of variation for V for an epsilon-
            # optimal policy
            self.thresh = epsilon * (1 - self.discount) / self.discount
        else: # discount == 1
            # threshold of variation for V for an epsilon-optimal policy
            self.thresh = epsilon

    def run(self):
        # Run the value iteration Gauss-Seidel algorithm.

        done = False

        if self.verbose:
            print('  Iteration\t\tV-variation')

        self.time = _time.time()

        while not done:
            self.iter += 1

            Vprev = self.V.copy()

            for s in range(self.S):
                Q = [float(self.R[a][s]+
                           self.discount * self.P[a][s, :].dot(self.V))
                     for a in range(self.A)]

                self.V[s] = max(Q)

            variation = _util.getSpan(self.V - Vprev)

            if self.verbose:
                print(("    %s\t\t  %s" % (self.iter, variation)))

            if variation < self.thresh:
                done = True
                if self.verbose:
                    print(_util._MSG_STOP_EPSILON_OPTIMAL_POLICY)
            elif self.iter == self.max_iter:
                done = True
                if self.verbose:
                    print(_util._MSG_STOP_MAX_ITER)

        self.policy = []
        for s in range(self.S):
            Q = _np.zeros(self.A)
            for a in range(self.A):
                Q[a] = self.R[a][s] + self.discount * self.P[a][s, :].dot(self.V)

            self.V[s] = Q.max()
            self.policy.append(int(Q.argmax()))

        self.time = _time.time() - self.time

        self.V = tuple(self.V.tolist())
        self.policy = tuple(self.policy)
