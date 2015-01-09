# -*- coding: utf-8 -*-
"""

"""

import numpy as _np

import mdptoolbox.util as _util

class MDP(object):

    """A Markov Decision Problem.

    Let ``S`` = the number of states, and ``A`` = the number of acions.

    Parameters
    ----------
    transitions : array
        Transition probability matrices. These can be defined in a variety of
        ways. The simplest is a numpy array that has the shape ``(A, S, S)``,
        though there are other possibilities. It can be a tuple or list or
        numpy object array of length ``A``, where each element contains a numpy
        array or matrix that has the shape ``(S, S)``. This "list of matrices"
        form is useful when the transition matrices are sparse as
        ``scipy.sparse.csr_matrix`` matrices can be used. In summary, each
        action's transition matrix must be indexable like ``transitions[a]``
        where ``a`` ∈ {0, 1...A-1}, and ``transitions[a]`` returns an ``S`` ×
        ``S`` array-like object.
    reward : array
        Reward matrices or vectors. Like the transition matrices, these can
        also be defined in a variety of ways. Again the simplest is a numpy
        array that has the shape ``(S, A)``, ``(S,)`` or ``(A, S, S)``. A list
        of lists can be used, where each inner list has length ``S`` and the
        outer list has length ``A``. A list of numpy arrays is possible where
        each inner array can be of the shape ``(S,)``, ``(S, 1)``, ``(1, S)``
        or ``(S, S)``. Also ``scipy.sparse.csr_matrix`` can be used instead of
        numpy arrays. In addition, the outer list can be replaced by any object
        that can be indexed like ``reward[a]`` such as a tuple or numpy object
        array of length ``A``.
    discount : float
        Discount factor. The per time-step discount factor on future rewards.
        Valid values are greater than 0 upto and including 1. If the discount
        factor is 1, then convergence is cannot be assumed and a warning will
        be displayed. Subclasses of ``MDP`` may pass ``None`` in the case where
        the algorithm does not use a discount factor.
    epsilon : float
        Stopping criterion. The maximum change in the value function at each
        iteration is compared against ``epsilon``. Once the change falls below
        this value, then the value function is considered to have converged to
        the optimal value function. Subclasses of ``MDP`` may pass ``None`` in
        the case where the algorithm does not use an epsilon-optimal stopping
        criterion.
    max_iter : int
        Maximum number of iterations. The algorithm will be terminated once
        this many iterations have elapsed. This must be greater than 0 if
        specified. Subclasses of ``MDP`` may pass ``None`` in the case where
        the algorithm does not use a maximum number of iterations.

    Attributes
    ----------
    P : array
        Transition probability matrices.
    R : array
        Reward vectors.
    V : tuple
        The optimal value function. Each element is a float corresponding to
        the expected value of being in that state assuming the optimal policy
        is followed.
    discount : float
        The discount rate on future rewards.
    max_iter : int
        The maximum number of iterations.
    policy : tuple
        The optimal policy.
    time : float
        The time used to converge to the optimal policy.
    verbose : boolean
        Whether verbose output should be displayed or not.

    Methods
    -------
    run
        Implemented in child classes as the main algorithm loop. Raises an
        exception if it has not been overridden.
    setSilent
        Turn the verbosity off
    setVerbose
        Turn the verbosity on

    """

    def __init__(self, transitions, reward, discount, epsilon, max_iter):
        # Initialise a MDP based on the input parameters.

        # if the discount is None then the algorithm is assumed to not use it
        # in its computations
        if discount is not None:
            self.discount = float(discount)
            assert 0.0 < self.discount <= 1.0, "Discount rate must be in ]0; 1]"
            if self.discount == 1:
                print("WARNING: check conditions of convergence. With no "
                      "discount, convergence can not be assumed.")
        # if the max_iter is None then the algorithm is assumed to not use it
        # in its computations
        if max_iter is not None:
            self.max_iter = int(max_iter)
            assert self.max_iter > 0, "The maximum number of iterations " \
                                      "must be greater than 0."
        # check that epsilon is something sane
        if epsilon is not None:
            self.epsilon = float(epsilon)
            assert self.epsilon > 0, "Epsilon must be greater than 0."
        # we run a check on P and R to make sure they are describing an MDP. If
        # an exception isn't raised then they are assumed to be correct.
        _util.check(transitions, reward)
        # computePR will assign the variables self.S, self.A, self.P and self.R
        self._computePR(transitions, reward)
        # the verbosity is by default turned off
        self.verbose = False
        # Initially the time taken to perform the computations is set to None
        self.time = None
        # set the initial iteration count to zero
        self.iter = 0
        # V should be stored as a vector ie shape of (S,) or (1, S)
        self.V = None
        # policy can also be stored as a vector
        self.policy = None

    def __repr__(self):
        P_repr = "P: \n"
        R_repr = "R: \n"
        for aa in range(self.A):
            P_repr += repr(self.P[aa]) + "\n"
            R_repr += repr(self.R[aa]) + "\n"
        return(P_repr + "\n" + R_repr)

    def _bellmanOperator(self, V=None):
        # Apply the Bellman operator on the value function.
        #
        # Updates the value function and the Vprev-improving policy.
        #
        # Returns: (policy, value), tuple of new policy and its value
        #
        # If V hasn't been sent into the method, then we assume to be working
        # on the objects V attribute
        if V is None:
            # this V should be a reference to the data rather than a copy
            V = self.V
        else:
            # make sure the user supplied V is of the right shape
            try:
                assert V.shape in ((self.S,), (1, self.S)), "V is not the " \
                    "right shape (Bellman operator)."
            except AttributeError:
                raise TypeError("V must be a numpy array or matrix.")
        # Looping through each action the the Q-value matrix is calculated.
        # P and V can be any object that supports indexing, so it is important
        # that you know they define a valid MDP before calling the
        # _bellmanOperator method. Otherwise the results will be meaningless.
        Q = _np.empty((self.A, self.S))
        for aa in range(self.A):
            Q[aa] = self.R[aa] + self.discount * self.P[aa].dot(V)
        # Get the policy and value, for now it is being returned but...
        # Which way is better?
        # 1. Return, (policy, value)
        return (Q.argmax(axis=0), Q.max(axis=0))
        # 2. update self.policy and self.V directly
        # self.V = Q.max(axis=1)
        # self.policy = Q.argmax(axis=1)

    def _computeP(self, P):
        # Set self.P as a tuple of length A, with each element storing an S×S
        # matrix.
        self.A = len(P)
        try:
            if P.ndim == 3:
                self.S = P.shape[1]
            else:
                self.S = P[0].shape[0]
        except AttributeError:
            self.S = P[0].shape[0]
        # convert P to a tuple of numpy arrays
        self.P = tuple(P[aa] for aa in range(self.A))

    def _computePR(self, P, R):
        # Compute the reward for the system in one state chosing an action.
        # Arguments
        # ---------
        # Let S = number of states, A = number of actions
        #    P(SxSxA)  = transition matrix
        #        P could be an array with 3 dimensions or  a cell array (1xA),
        #        each cell containing a matrix (SxS) possibly sparse
        #    R(SxSxA) or (SxA) = reward matrix
        #        R could be an array with 3 dimensions (SxSxA) or  a cell array
        #        (1xA), each cell containing a sparse matrix (SxS) or a 2D
        #        array(SxA) possibly sparse
        # Evaluation
        # ----------
        #    PR(SxA)   = reward matrix
        #
        # We assume that P and R define a MDP i,e. assumption is that
        # _util.check(P, R) has already been run and doesn't fail.
        #
        # First compute store P, S, and A
        self._computeP(P)
        # Set self.R as a tuple of length A, with each element storing an 1×S
        # vector.
        try:
            if R.ndim == 1:
                r = _np.array(R).reshape(self.S)
                self.R = tuple(r for aa in range(self.A))
            elif R.ndim == 2:
                self.R = tuple(_np.array(R[:, aa]).reshape(self.S)
                               for aa in range(self.A))
            else:
                self.R = tuple(_np.multiply(P[aa], R[aa]).sum(1).reshape(self.S)
                               for aa in range(self.A))
        except AttributeError:
            if len(R) == self.A:
                self.R = tuple(_np.multiply(P[aa], R[aa]).sum(1).reshape(self.S)
                               for aa in range(self.A))
            else:
                r = _np.array(R).reshape(self.S)
                self.R = tuple(r for aa in range(self.A))

    def run(self):
        # Raise error because child classes should implement this function.
        raise NotImplementedError("You should create a run() method.")

    def setSilent(self):
        """Set the MDP algorithm to silent mode."""
        self.verbose = False

    def setVerbose(self):
        """Set the MDP algorithm to verbose mode."""
        self.verbose = True
