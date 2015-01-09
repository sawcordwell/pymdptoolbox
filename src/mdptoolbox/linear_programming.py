# -*- coding: utf-8 -*-
"""

"""

import time as _time

import numpy as _np
import scipy.sparse as _sp

from mdptoolbox.mdp_base import MDP

class LP(MDP):

    """A discounted MDP soloved using linear programming.

    This class requires the Python ``cvxopt`` module to be installed.

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
    h : array, optional
        Terminal reward. Default: a vector of zeros.

    Data Attributes
    ---------------
    V : tuple
        optimal values
    policy : tuple
        optimal policy
    time : float
        used CPU time

    Examples
    --------
    >>> import mdptoolbox.example
    >>> P, R = mdptoolbox.example.forest()
    >>> lp = mdptoolbox.mdp.LP(P, R, 0.9)
    >>> lp.run()

    >>> import numpy, mdptoolbox
    >>> P = numpy.array((((0.5, 0.5), (0.8, 0.2)), ((0, 1), (0.1, 0.9))))
    >>> R = numpy.array(((5, 10), (-1, 2)))
    >>> lp = mdptoolbox.mdp.LP(P, R, 0.9)
    >>> lp.run()
    >>> #lp.policy #FIXME: gives (1, 1), should be (1, 0)

    """

    def __init__(self, transitions, reward, discount):
        # Initialise a linear programming MDP.
        # import some functions from cvxopt and set them as object methods
        try:
            from cvxopt import matrix, solvers
            self._linprog = solvers.lp
            self._cvxmat = matrix
        except ImportError:
            raise ImportError("The python module cvxopt is required to use "
                              "linear programming functionality.")
        # initialise the MDP. epsilon and max_iter are not needed
        MDP.__init__(self, transitions, reward, discount, None, None)
        # Set the cvxopt solver to be quiet by default, but ...
        # this doesn't do what I want it to do c.f. issue #3
        if not self.verbose:
            solvers.options['show_progress'] = False

    def run(self):
        #Run the linear programming algorithm.
        self.time = _time.time()
        # The objective is to resolve : min V / V >= PR + discount*P*V
        # The function linprog of the optimisation Toolbox of Mathworks
        # resolves :
        # min f'* x / M * x <= b
        # So the objective could be expressed as :
        # min V / (discount*P-I) * V <= - PR
        # To avoid loop on states, the matrix M is structured following actions
        # M(A*S,S)
        f = self._cvxmat(_np.ones((self.S, 1)))
        h = _np.array(self.R).reshape(self.S * self.A, 1, order="F")
        h = self._cvxmat(h, tc='d')
        M = _np.zeros((self.A * self.S, self.S))
        for aa in range(self.A):
            pos = (aa + 1) * self.S
            M[(pos - self.S):pos, :] = (
                self.discount * self.P[aa] - _sp.eye(self.S, self.S))
        M = self._cvxmat(M)
        # Using the glpk option will make this behave more like Octave
        # (Octave uses glpk) and perhaps Matlab. If solver=None (ie using the
        # default cvxopt solver) then V agrees with the Octave equivalent
        # only to 10e-8 places. This assumes glpk is installed of course.
        self.V = _np.array(self._linprog(f, M, -h)['x']).reshape(self.S)
        # apply the Bellman operator
        self.policy, self.V = self._bellmanOperator()
        # update the time spent solving
        self.time = _time.time() - self.time
        # store value and policy as tuples
        self.V = tuple(self.V.tolist())
        self.policy = tuple(self.policy.tolist())
