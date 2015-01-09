# -*- coding: utf-8 -*-
"""

"""

import time as _time

import numpy as _np

import mdptoolbox.util as _util
from mdptoolbox.mdp_base import MDP

class RelativeValueIteration(MDP):

    """A MDP solved using the relative value iteration algorithm.

    Arguments
    ---------
    transitions : array
        Transition probability matrices. See the documentation for the ``MDP``
        class for details.
    reward : array
        Reward matrices or vectors. See the documentation for the ``MDP`` class
        for details.
    epsilon : float, optional
        Stopping criterion. See the documentation for the ``MDP`` class for
        details. Default: 0.01.
    max_iter : int, optional
        Maximum number of iterations. See the documentation for the ``MDP``
        class for details. Default: 1000.

    Data Attributes
    ---------------
    policy : tuple
        epsilon-optimal policy
    average_reward  : tuple
        average reward of the optimal policy
    cpu_time : float
        used CPU time

    Notes
    -----
    In verbose mode, at each iteration, displays the span of U variation
    and the condition which stopped iterations : epsilon-optimum policy found
    or maximum number of iterations reached.

    Examples
    --------
    >>> import mdptoolbox, mdptoolbox.example
    >>> P, R = mdptoolbox.example.forest()
    >>> rvi = mdptoolbox.mdp.RelativeValueIteration(P, R)
    >>> rvi.run()
    >>> rvi.average_reward
    3.2399999999999993
    >>> rvi.policy
    (0, 0, 0)
    >>> rvi.iter
    4

    >>> import mdptoolbox
    >>> import numpy as np
    >>> P = np.array([[[0.5, 0.5],[0.8, 0.2]],[[0, 1],[0.1, 0.9]]])
    >>> R = np.array([[5, 10], [-1, 2]])
    >>> rvi = mdptoolbox.mdp.RelativeValueIteration(P, R)
    >>> rvi.run()
    >>> expected = (10.0, 3.885235246411831)
    >>> all(expected[k] - rvi.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> rvi.average_reward
    3.8852352464118312
    >>> rvi.policy
    (1, 0)
    >>> rvi.iter
    29

    """

    def __init__(self, transitions, reward, epsilon=0.01, max_iter=1000):
        # Initialise a relative value iteration MDP.

        MDP.__init__(self,  transitions, reward, None, epsilon, max_iter)

        self.epsilon = epsilon
        self.discount = 1

        self.V = _np.zeros(self.S)
        self.gain = 0 # self.U[self.S]

        self.average_reward = None

    def run(self):
        # Run the relative value iteration algorithm.

        done = False
        if self.verbose:
            print('  Iteration\t\tU variation')

        self.time = _time.time()

        while not done:

            self.iter += 1

            self.policy, Vnext = self._bellmanOperator()
            Vnext = Vnext - self.gain

            variation = _util.getSpan(Vnext - self.V)

            if self.verbose:
                print(("    %s\t\t  %s" % (self.iter, variation)))

            if variation < self.epsilon:
                done = True
                self.average_reward = self.gain + (Vnext - self.V).min()
                if self.verbose:
                    print(_util._MSG_STOP_EPSILON_OPTIMAL_POLICY)
            elif self.iter == self.max_iter:
                done = True
                self.average_reward = self.gain + (Vnext - self.V).min()
                if self.verbose:
                    print(_util._MSG_STOP_MAX_ITER)

            self.V = Vnext
            self.gain = float(self.V[self.S - 1])

        self.time = _time.time() - self.time

        # store value and policy as tuples
        self.V = tuple(self.V.tolist())
        self.policy = tuple(self.policy.tolist())
