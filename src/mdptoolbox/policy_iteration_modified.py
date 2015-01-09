# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 13:47:35 2015

@author: steve
"""

import time as _time

import numpy as _np

import mdptoolbox.util as _util
from mdptoolbox.policy_iteration import PolicyIteration

class PolicyIterationModified(PolicyIteration):

    """A discounted MDP  solved using a modifified policy iteration algorithm.

    Arguments
    ---------
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
        class for details. Default is 10.

    Data Attributes
    ---------------
    V : tuple
        value function
    policy : tuple
        optimal policy
    iter : int
        number of done iterations
    time : float
        used CPU time

    Examples
    --------
    >>> import mdptoolbox, mdptoolbox.example
    >>> P, R = mdptoolbox.example.forest()
    >>> pim = mdptoolbox.mdp.PolicyIterationModified(P, R, 0.9)
    >>> pim.run()
    >>> pim.policy
    (0, 0, 0)
    >>> expected = (21.81408652334702, 25.054086523347017, 29.054086523347017)
    >>> all(expected[k] - pim.V[k] < 1e-12 for k in range(len(expected)))
    True

    """

    def __init__(self, transitions, reward, discount, epsilon=0.01,
                 max_iter=10):
        # Initialise a (modified) policy iteration MDP.

        # Maybe its better not to subclass from PolicyIteration, because the
        # initialisation of the two are quite different. eg there is policy0
        # being calculated here which doesn't need to be. The only thing that
        # is needed from the PolicyIteration class is the _evalPolicyIterative
        # function. Perhaps there is a better way to do it?
        PolicyIteration.__init__(self, transitions, reward, discount, None,
                                 max_iter, 1)

        # PolicyIteration doesn't pass epsilon to MDP.__init__() so we will
        # check it here
        self.epsilon = float(epsilon)
        assert epsilon > 0, "'epsilon' must be greater than 0."

        # computation of threshold of variation for V for an epsilon-optimal
        # policy
        if self.discount != 1:
            self.thresh = self.epsilon * (1 - self.discount) / self.discount
        else:
            self.thresh = self.epsilon

        if self.discount == 1:
            self.V = _np.zeros(self.S)
        else:
            Rmin = min(R.min() for R in self.R)
            self.V = 1 / (1 - self.discount) * Rmin * _np.ones((self.S,))

    def run(self):
        # Run the modified policy iteration algorithm.

        if self.verbose:
            print('  \tIteration\t\tV-variation')

        self.time = _time.time()

        done = False
        while not done:
            self.iter += 1

            self.policy, Vnext = self._bellmanOperator()
            #[Ppolicy, PRpolicy] = mdp_computePpolicyPRpolicy(P, PR, policy);

            variation = _util.getSpan(Vnext - self.V)
            if self.verbose:
                print(("    %s\t\t  %s" % (self.iter, variation)))

            self.V = Vnext
            if variation < self.thresh:
                done = True
            else:
                is_verbose = False
                if self.verbose:
                    self.setSilent()
                    is_verbose = True

                self._evalPolicyIterative(self.V, self.epsilon, self.max_iter)

                if is_verbose:
                    self.setVerbose()

        self.time = _time.time() - self.time

        # store value and policy as tuples
        self.V = tuple(self.V.tolist())
        self.policy = tuple(self.policy.tolist())
