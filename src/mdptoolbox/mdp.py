# -*- coding: utf-8 -*-
"""Markov Decision Process (MDP) Toolbox: ``mdp`` module
=====================================================

The ``mdp`` module provides classes for the resolution of descrete-time Markov
Decision Processes.

Available classes
-----------------
MDP
    Base Markov decision process class
FiniteHorizon
    Backwards induction finite horizon MDP
LP
    Linear programming MDP
PolicyIteration
    Policy iteration MDP
PolicyIterationModified
    Modified policy iteration MDP
QLearning
    Q-learning MDP
RelativeValueIteration
    Relative value iteration MDP
ValueIteration
    Value iteration MDP
ValueIterationGS
    Gauss-Seidel value iteration MDP

"""

# Copyright (c) 2011-2013 Steven A. W. Cordwell
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

from math import ceil, log, sqrt
from time import time

from numpy import absolute, array, empty, mean, mod, multiply, ones, zeros
from numpy.random import randint, random
from scipy.sparse import csr_matrix as sparse

from .utils import check, getSpan

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
                      "discount, convergence is can not be assumed.")
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
        check(transitions, reward)
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
        Q = empty((self.A, self.S))
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
        # check(P, R) has already been run and doesn't fail.
        #
        # First compute store P, S, and A
        self._computeP(P)
        # Set self.R as a tuple of length A, with each element storing an 1×S
        # vector.
        try:
            if R.ndim == 1:
                r = array(R).reshape(self.S)
                self.R = tuple(r for aa in range(self.A))
            elif R.ndim == 2:
                self.R = tuple(array(R[:, aa]).reshape(self.S)
                                for aa in range(self.A))
            else:
                self.R = tuple(multiply(P[aa], R[aa]).sum(1).reshape(self.S)
                                for aa in range(self.A))
        except AttributeError:
            if len(R) == self.A:
                self.R = tuple(multiply(P[aa], R[aa]).sum(1).reshape(self.S)
                                for aa in range(self.A))
            else:
                r = array(R).reshape(self.S)
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
        assert self.N > 0, 'PyMDPtoolbox: N must be greater than 0.'
        # Initialise the base class
        MDP.__init__(self, transitions, reward, discount, None, None)
        # remove the iteration counter, it is not meaningful for backwards
        # induction
        del self.iter
        # There are value vectors for each time step up to the horizon
        self.V = zeros((self.S, N + 1))
        # There are policy vectors for each time step before the horizon, when
        # we reach the horizon we don't need to make decisions anymore.
        self.policy = empty((self.S, N), dtype=int)
        # Set the reward for the final transition to h, if specified.
        if h is not None:
            self.V[:, N] = h
        # Call the iteration method
        #self.run()
        
    def run(self):
        # Run the finite horizon algorithm.
        self.time = time()
        # loop through each time period
        for n in range(self.N):
            W, X = self._bellmanOperator(self.V[:, self.N - n])
            self.V[:, self.N - n - 1] = X
            self.policy[:, self.N - n - 1] = W
            if self.verbose:
                print(("stage: %s ... policy transpose : %s") % (
                    self.N - n, self.policy[:, self.N - n -1].tolist()))
        # update time spent running
        self.time = time() - self.time
        # After this we could create a tuple of tuples for the values and 
        # policies.
        #V = []
        #p = []
        #for n in xrange(self.N):
        #    V.append()
        #    p.append()
        #V.append()
        #self.V = tuple(V)
        #self.policy = tuple(p)

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
    
    Notes    
    -----
    In verbose mode, displays the current stage and policy transpose.
    
    Examples
    --------
    >>> import mdptoolbox, mdptoolbox.example
    >>> P, R = mdptoolbox.example.forest()
    >>> lp = mdptoolbox.mdp.LP(P, R, 0.9)
    >>> lp.run()
    
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
        # we also need diagonal matrices, and using a sparse one may be more
        # memory efficient
        from scipy.sparse import eye as speye
        self._speye = speye
        # initialise the MDP. epsilon and max_iter are not needed
        MDP.__init__(self, transitions, reward, discount, None, None)
        # Set the cvxopt solver to be quiet by default, but ...
        # this doesn't do what I want it to do c.f. issue #3
        if not self.verbose:
            solvers.options['show_progress'] = False
        # Call the iteration method
        #self.run()
    
    def run(self):
        #Run the linear programming algorithm.
        self.time = time()
        # The objective is to resolve : min V / V >= PR + discount*P*V
        # The function linprog of the optimisation Toolbox of Mathworks
        # resolves :
        # min f'* x / M * x <= b
        # So the objective could be expressed as :
        # min V / (discount*P-I) * V <= - PR
        # To avoid loop on states, the matrix M is structured following actions
        # M(A*S,S)
        f = self._cvxmat(ones((self.S, 1)))
        h = self._cvxmat(self.R.reshape(self.S * self.A, 1, order="F"), tc='d')
        M = zeros((self.A * self.S, self.S))
        for aa in range(self.A):
            pos = (aa + 1) * self.S
            M[(pos - self.S):pos, :] = (
                self.discount * self.P[aa] - self._speye(self.S, self.S))
        M = self._cvxmat(M)
        # Using the glpk option will make this behave more like Octave
        # (Octave uses glpk) and perhaps Matlab. If solver=None (ie using the 
        # default cvxopt solver) then V agrees with the Octave equivalent
        # only to 10e-8 places. This assumes glpk is installed of course.
        self.V = array(self._linprog(f, M, -h, solver='glpk')['x'])
        # apply the Bellman operator
        self.policy, self.V =  self._bellmanOperator()
        # update the time spent solving
        self.time = time() - self.time
        # store value and policy as tuples
        self.V = tuple(self.V.tolist())
        self.policy = tuple(self.policy.tolist())

class PolicyIteration(MDP):
    
    """A discounted MDP solved using the policy iteration algorithm.
    
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
    policy0 : array, optional
        Starting policy.
    max_iter : int, optional
        Maximum number of iterations. See the documentation for the ``MDP``
        class for details. Default is 1000.
    eval_type : int or string, optional
        Type of function used to evaluate policy. 0 or "matrix" to solve as a
        set of linear equations. 1 or "iterative" to solve iteratively.
        Default: 0.
             
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
    
    Notes
    -----
    In verbose mode, at each iteration, displays the number 
    of differents actions between policy n-1 and n
    
    Examples
    --------
    >>> import mdptoolbox, mdptoolbox.example
    >>> P, R = mdptoolbox.example.rand()
    >>> pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.9)
    >>> pi.run()
    
    >>> P, R = mdptoolbox.example.forest()
    >>> pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.9)
    >>> pi.run()
    >>> pi.V
    (26.244000000000018, 29.48400000000002, 33.484000000000016)
    >>> pi.policy
    (0, 0, 0)
    """
    
    def __init__(self, transitions, reward, discount, policy0=None,
                 max_iter=1000, eval_type=0):
        # Initialise a policy iteration MDP.
        #
        # Set up the MDP, but don't need to worry about epsilon values
        MDP.__init__(self, transitions, reward, discount, None, max_iter)
        # Check if the user has supplied an initial policy. If not make one.
        if policy0 == None:
            # Initialise the policy to the one which maximises the expected
            # immediate reward
            null = zeros(self.S)
            self.policy, null = self._bellmanOperator(null)
            del null
        else:
            # Use the policy that the user supplied
            # Make sure it is a numpy array
            policy0 = array(policy0)
            # Make sure the policy is the right size and shape
            assert policy0.shape in ((self.S, ), (self.S, 1), (1, self.S)), \
                "'policy0' must a vector with length S."
            # reshape the policy to be a vector
            policy0 = policy0.reshape(self.S)
            # The policy can only contain integers between 0 and S-1
            msg = "'policy0' must be a vector of integers between 0 and S-1."
            assert not mod(policy0, 1).any(), msg
            assert (policy0 >= 0).all(), msg
            assert (policy0 < self.S).all(), msg
            self.policy = policy0
        # set the initial values to zero
        self.V = zeros(self.S)
        # Do some setup depending on the evaluation type
        if eval_type in (0, "matrix"):
            from numpy.linalg import solve
            from scipy.sparse import eye
            self._speye = eye
            self._lin_eq = solve
            self.eval_type = "matrix"
        elif eval_type in (1, "iterative"):
            self.eval_type = "iterative"
        else:
            raise ValueError("PyMDPtoolbox: eval_type should be 0 for matrix "
                             "evaluation or 1 for iterative evaluation. "
                             "The strings 'matrix' and 'iterative' can also "
                             "be used.")
        # Call the iteration method
        #self.run()
    
    def _computePpolicyPRpolicy(self):
        # Compute the transition matrix and the reward matrix for a policy.
        #
        # Arguments
        # ---------
        # Let S = number of states, A = number of actions
        # P(SxSxA)  = transition matrix 
        #     P could be an array with 3 dimensions or a cell array (1xA),
        #     each cell containing a matrix (SxS) possibly sparse
        # R(SxSxA) or (SxA) = reward matrix
        #     R could be an array with 3 dimensions (SxSxA) or 
        #     a cell array (1xA), each cell containing a sparse matrix (SxS) or
        #     a 2D array(SxA) possibly sparse  
        # policy(S) = a policy
        #
        # Evaluation
        # ----------
        # Ppolicy(SxS)  = transition matrix for policy
        # PRpolicy(S)   = reward matrix for policy
        #
        Ppolicy = empty((self.S, self.S))
        Rpolicy = zeros(self.S)
        for aa in range(self.A): # avoid looping over S
            # the rows that use action a.
            ind = (self.policy == aa).nonzero()[0]
            # if no rows use action a, then no need to assign this
            if ind.size > 0:
                try:
                    Ppolicy[ind, :] = self.P[aa][ind, :]
                except ValueError:
                    Ppolicy[ind, :] = self.P[aa][ind, :].todense()
                #PR = self._computePR() # an apparently uneeded line, and
                # perhaps harmful in this implementation c.f.
                # mdp_computePpolicyPRpolicy.m
                Rpolicy[ind] = self.R[aa][ind]
        # self.R cannot be sparse with the code in its current condition, but
        # it should be possible in the future. Also, if R is so big that its
        # a good idea to use a sparse matrix for it, then converting PRpolicy
        # from a dense to sparse matrix doesn't seem very memory efficient
        if type(self.R) is sparse:
            Rpolicy = sparse(Rpolicy)
        #self.Ppolicy = Ppolicy
        #self.Rpolicy = Rpolicy
        return (Ppolicy, Rpolicy)
    
    def _evalPolicyIterative(self, V0=0, epsilon=0.0001, max_iter=10000):
        # Evaluate a policy using iteration.
        #
        # Arguments
        # ---------
        # Let S = number of states, A = number of actions
        # P(SxSxA)  = transition matrix 
        #    P could be an array with 3 dimensions or 
        #    a cell array (1xS), each cell containing a matrix possibly sparse
        # R(SxSxA) or (SxA) = reward matrix
        #    R could be an array with 3 dimensions (SxSxA) or 
        #    a cell array (1xA), each cell containing a sparse matrix (SxS) or
        #    a 2D array(SxA) possibly sparse  
        # discount  = discount rate in ]0; 1[
        # policy(S) = a policy
        # V0(S)     = starting value function, optional (default : zeros(S,1))
        # epsilon   = epsilon-optimal policy search, upper than 0,
        #    optional (default : 0.0001)
        # max_iter  = maximum number of iteration to be done, upper than 0, 
        #    optional (default : 10000)
        #    
        # Evaluation
        # ----------
        # Vpolicy(S) = value function, associated to a specific policy
        #
        # Notes
        # -----
        # In verbose mode, at each iteration, displays the condition which
        # stopped iterations: epsilon-optimum value function found or maximum
        # number of iterations reached.
        #
        try:
            assert V0.shape in ((self.S, ), (self.S, 1), (1, self.S)), \
                "'V0' must be a vector of length S."
            policy_V = array(V0).reshape(self.S)
        except AttributeError:
            if len(V0) == self.S:
                policy_V = array(V0).reshape(self.S)
            else:
                policy_V = zeros(self.S)
        
        policy_P, policy_R = self._computePpolicyPRpolicy()
        
        if self.verbose:
            print('  Iteration    V_variation')
        
        itr = 0
        done = False
        while not done:
            itr += 1
            
            Vprev = policy_V
            policy_V = policy_R + self.discount * policy_P.dot(Vprev)
            
            variation = absolute(policy_V - Vprev).max()
            if self.verbose:
                print(('      %s         %s') % (itr, variation))
            
            # ensure |Vn - Vpolicy| < epsilon
            if variation < ((1 - self.discount) / self.discount) * epsilon:
                done = True
                if self.verbose:
                    print("PyMDPtoolbox: iterations stopped, epsilon-optimal "
                          "value function.")
            elif itr == max_iter:
                done = True
                if self.verbose:
                    print("PyMDPtoolbox: iterations stopped by maximum number "
                          "of iteration condition.")
        
        self.V = policy_V
    
    def _evalPolicyMatrix(self):
        # Evaluate the value function of the policy using linear equations.
        #
        # Arguments 
        # ---------
        # Let S = number of states, A = number of actions
        # P(SxSxA) = transition matrix 
        #      P could be an array with 3 dimensions or a cell array (1xA),
        #      each cell containing a matrix (SxS) possibly sparse
        # R(SxSxA) or (SxA) = reward matrix
        #      R could be an array with 3 dimensions (SxSxA) or 
        #      a cell array (1xA), each cell containing a sparse matrix (SxS) or
        #      a 2D array(SxA) possibly sparse  
        # discount = discount rate in ]0; 1[
        # policy(S) = a policy
        #
        # Evaluation
        # ----------
        # Vpolicy(S) = value function of the policy
        #
        Ppolicy, Rpolicy = self._computePpolicyPRpolicy()
        # V = PR + gPV  => (I-gP)V = PR  => V = inv(I-gP)* PR
        self.V = self._lin_eq(
            (self._speye(self.S, self.S) - self.discount * Ppolicy), Rpolicy)
    
    def run(self):
        # Run the policy iteration algorithm.
        # If verbose the print a header
        if self.verbose:
            print('  Iteration  Number_of_different_actions')
        # Set up the while stopping condition and the current time
        done = False
        self.time = time()
        # loop until a stopping condition is reached
        while not done:
            self.iter += 1
            # these _evalPolicy* functions will update the classes value
            # attribute
            if self.eval_type == "matrix":
                self._evalPolicyMatrix()
            elif self.eval_type == "iterative":
                self._evalPolicyIterative()
            # This should update the classes policy attribute but leave the
            # value alone
            policy_next, null = self._bellmanOperator()
            del null
            # calculate in how many places does the old policy disagree with
            # the new policy
            n_different = (policy_next != self.policy).sum()
            # if verbose then continue printing a table
            if self.verbose:
                print(('       %s                 %s') % (self.iter,
                                                         n_different))
            # Once the policy is unchanging of the maximum number of 
            # of iterations has been reached then stop
            if n_different == 0:
                done = True
                if self.verbose:
                    print("PyMDPtoolbox: iterations stopped, unchanging "
                          "policy found.")
            elif (self.iter == self.max_iter):
                done = True 
                if self.verbose:
                    print("PyMDPtoolbox: iterations stopped by maximum number "
                          "of iteration condition.")
            else:
                self.policy = policy_next
        # update the time to return th computation time
        self.time = time() - self.time
        # store value and policy as tuples
        self.V = tuple(self.V.tolist())
        self.policy = tuple(self.policy.tolist())

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
    >>> pim.V
    (21.81408652334702, 25.054086523347017, 29.054086523347017)
    
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
            self.V = zeros((self.S, 1))
        else:
            Rmin = min(R.min() for R in self.R)
            self.V = 1 / (1 - self.discount) * Rmin * ones((self.S,))
        
        # Call the iteration method
        #self.run()
    
    def run(self):
        # Run the modified policy iteration algorithm.
        
        if self.verbose:
            print('\tIteration\tV-variation')
        
        self.time = time()
        
        done = False
        while not done:
            self.iter += 1
            
            self.policy, Vnext = self._bellmanOperator()
            #[Ppolicy, PRpolicy] = mdp_computePpolicyPRpolicy(P, PR, policy);
            
            variation = getSpan(Vnext - self.V)
            if self.verbose:
                print(("\t%s\t%s" % (self.iter, variation)))
            
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
        
        self.time = time() - self.time
        
        # store value and policy as tuples
        self.V = tuple(self.V.tolist())
        self.policy = tuple(self.policy.tolist())

class QLearning(MDP):
    
    """A discounted MDP solved using the Q learning algorithm.
    
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
    n_iter : int, optional
        Number of iterations to execute. This is ignored unless it is an 
        integer greater than the default value. Defaut: 10,000.
    
    Data Attributes
    ---------------
    Q : array
        learned Q matrix (SxA)     
    V : tuple
        learned value function (S).    
    policy : tuple
        learned optimal policy (S).    
    mean_discrepancy : array
        Vector of V discrepancy mean over 100 iterations. Then the length of
        this vector for the default value of N is 100 (N/100).

    Examples
    ---------
    >>> # These examples are reproducible only if random seed is set to 0 in
    >>> # both the random and numpy.random modules.
    >>> import numpy as np
    >>> import mdptoolbox, mdptoolbox.example
    >>> np.random.seed(0)
    >>> P, R = mdptoolbox.example.forest()
    >>> ql = mdptoolbox.mdp.QLearning(P, R, 0.96)
    >>> ql.run()
    >>> ql.Q
    array([[ 68.38037354,  43.24888454],
           [ 72.37777922,  42.75549145],
           [ 77.02892702,  64.68712932]])
    >>> ql.V
    (68.38037354422798, 72.37777921607258, 77.02892701616531)
    >>> ql.policy
    (0, 0, 0)
    
    >>> import mdptoolbox
    >>> import numpy as np
    >>> P = np.array([[[0.5, 0.5],[0.8, 0.2]],[[0, 1],[0.1, 0.9]]])
    >>> R = np.array([[5, 10], [-1, 2]])
    >>> np.random.seed(0)
    >>> ql = mdptoolbox.mdp.QLearning(P, R, 0.9)
    >>> ql.run()
    >>> ql.Q
    array([[ 39.933691  ,  43.17543338],
           [ 36.94394224,  35.42568056]])
    >>> ql.V
    (43.17543338090149, 36.943942243204454)
    >>> ql.policy
    (1, 0)
    
    """
    
    def __init__(self, transitions, reward, discount, n_iter=10000):
        # Initialise a Q-learning MDP.
        
        # The following check won't be done in MDP()'s initialisation, so let's
        # do it here
        self.max_iter = int(n_iter)
        assert self.max_iter >= 10000, "PyMDPtoolbox: n_iter should be " \
                                        "greater than 10000."
        
        # We don't want to send this to MDP because _computePR should not be
        # run on it, so check that it defines an MDP
        check(transitions, reward)
        
        # Store P, S, and A
        self._computeP(transitions)
        
        self.R = reward
        
        self.discount = discount
        
        # Initialisations
        self.Q = zeros((self.S, self.A))
        self.mean_discrepancy = []
        
        # Call the iteration method
        #self.run()
        
    def run(self):
        # Run the Q-learning algoritm.
        discrepancy = []
        
        self.time = time()
        
        # initial state choice
        s = randint(0, self.S)
        
        for n in range(1, self.max_iter + 1):
            
            # Reinitialisation of trajectories every 100 transitions
            if ((n % 100) == 0):
                s = randint(0, self.S)
            
            # Action choice : greedy with increasing probability
            # probability 1-(1/log(n+2)) can be changed
            pn = random()
            if (pn < (1 - (1 / log(n + 2)))):
                # optimal_action = self.Q[s, :].max()
                a = self.Q[s, :].argmax()
            else:
                a = randint(0, self.A)
            
            # Simulating next state s_new and reward associated to <s,s_new,a>
            p_s_new = random()
            p = 0
            s_new = -1
            while ((p < p_s_new) and (s_new < (self.S - 1))):
                s_new = s_new + 1
                p = p + self.P[a][s, s_new]
            
            try:
                r = self.R[a][s, s_new]
            except IndexError:
                r = self.R[s, a]
            
            # Updating the value of Q
            # Decaying update coefficient (1/sqrt(n+2)) can be changed
            delta = r + self.discount * self.Q[s_new, :].max() - self.Q[s, a]
            dQ = (1 / sqrt(n + 2)) * delta
            self.Q[s, a] = self.Q[s, a] + dQ
            
            # current state is updated
            s = s_new
            
            # Computing and saving maximal values of the Q variation
            discrepancy.append(absolute(dQ))
            
            # Computing means all over maximal Q variations values
            if len(discrepancy) == 100:
                self.mean_discrepancy.append(mean(discrepancy))
                discrepancy = []
            
            # compute the value function and the policy
            self.V = self.Q.max(axis=1)
            self.policy = self.Q.argmax(axis=1)
            
        self.time = time() - self.time
        
        # convert V and policy to tuples
        self.V = tuple(self.V.tolist())
        self.policy = tuple(self.policy.tolist())

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
    >>> rvi.V
    (10.0, 3.885235246411831)
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
        
        self.V = zeros(self.S)
        self.gain = 0 # self.U[self.S]
        
        self.average_reward = None
        
        # Call the iteration method
        #self.run()
    
    def run(self):
        # Run the relative value iteration algorithm.
        
        done = False
        if self.verbose:
            print('  Iteration  U_variation')
        
        self.time = time()
        
        while not done:
            
            self.iter += 1;
            
            self.policy, Vnext = self._bellmanOperator()
            Vnext = Vnext - self.gain
            
            variation = getSpan(Vnext - self.V)
            
            if self.verbose:
                print(("      %s         %s" % (self.iter, variation)))
            
            if variation < self.epsilon:
                 done = True
                 self.average_reward = self.gain + (Vnext - self.V).min()
                 if self.verbose:
                     print("MDP Toolbox : iterations stopped, epsilon-optimal "
                           "policy found.")
            elif self.iter == self.max_iter:
                 done = True 
                 self.average_reward = self.gain + (Vnext - self.V).min()
                 if self.verbose:
                     print("MDP Toolbox : iterations stopped by maximum "
                           "number of iteration condition.")
            
            self.V = Vnext
            self.gain = float(self.V[self.S - 1])
        
        self.time = time() - self.time
        
        # store value and policy as tuples
        self.V = tuple(self.V.tolist())
        self.policy = tuple(self.policy.tolist())

class ValueIteration(MDP):
    
    """A discounted MDP solved using the value iteration algorithm.
    
    Description
    -----------
    ValueIteration applies the value iteration algorithm to solve a
    discounted MDP. The algorithm consists of solving Bellman's equation
    iteratively.
    Iteration is stopped when an epsilon-optimal policy is found or after a
    specified number (``max_iter``) of iterations. 
    This function uses verbose and silent modes. In verbose mode, the function
    displays the variation of ``V`` (the value function) for each iteration and
    the condition which stopped the iteration: epsilon-policy found or maximum
    number of iterations reached.
    
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
        details.  Default: 0.01.
    max_iter : int, optional
        Maximum number of iterations. If the value given is greater than a
        computed bound, a warning informs that the computed bound will be used
        instead. By default, if ``discount`` is not equal to 1, a bound for 
        ``max_iter`` is computed, otherwise ``max_iter`` = 1000. See the 
        documentation for the ``MDP`` class for further details.
    initial_value : array, optional
        The starting value function. Default: a vector of zeros.
    
    Data Attributes
    ---------------
    V : tuple
        The optimal value function.
    policy : tuple
        The optimal policy function. Each element is an integer corresponding
        to an action which maximises the value function in that state.
    iter : int
        The number of iterations taken to complete the computation.
    time : float
        The amount of CPU time used to run the algorithm.
    
    Methods
    -------
    run()
        Do the algorithm iteration.
    setSilent()
        Sets the instance to silent mode.
    setVerbose()
        Sets the instance to verbose mode.
    
    Notes
    -----
    In verbose mode, at each iteration, displays the variation of V
    and the condition which stopped iterations: epsilon-optimum policy found
    or maximum number of iterations reached.
    
    Examples
    --------
    >>> import mdptoolbox, mdptoolbox.example
    >>> P, R = mdptoolbox.example.forest()
    >>> vi = mdptoolbox.mdp.ValueIteration(P, R, 0.96)
    >>> vi.verbose
    False
    >>> vi.run()
    >>> vi.V
    (5.93215488, 9.38815488, 13.38815488)
    >>> vi.policy
    (0, 0, 0)
    >>> vi.iter
    4
    
    >>> import mdptoolbox
    >>> import numpy as np
    >>> P = np.array([[[0.5, 0.5],[0.8, 0.2]],[[0, 1],[0.1, 0.9]]])
    >>> R = np.array([[5, 10], [-1, 2]])
    >>> vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    >>> vi.setVerbose()
    >>> vi.run()
        Iteration       V-variation
        1       8.0
        2       2.76
        3       1.9872
        4       1.430784
        5       1.03016448
        6       0.7417184256
        7       0.534037266432
        8       0.384506831831
        9       0.276844918918
        10      0.199328341621
        11      0.143516405967
        12      0.103331812296
        13      0.0743989048534
        14      0.0535672114945
        15      0.038568392276
        16      0.0277692424387
        17      0.0199938545559
        18      0.0143955752802
        19      0.0103648142018
        20      0.00746266622526
        21      0.00537311968218
        22      0.00386864617116
        23      0.00278542524322
        24      0.00200550617512
        25      0.00144396444609
        26      0.0010396544012
    PyMDPToolbox: iteration stopped, epsilon-optimal policy found.
    >>> vi.V
    (40.048625392716815, 33.65371175967546)
    >>> vi.policy
    (1, 0)
    >>> vi.iter
    26
    
    >>> import mdptoolbox
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix as sparse
    >>> P = [None] * 2
    >>> P[0] = sparse([[0.5, 0.5],[0.8, 0.2]])
    >>> P[1] = sparse([[0, 1],[0.1, 0.9]])
    >>> R = np.array([[5, 10], [-1, 2]])
    >>> vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    >>> vi.run()
    >>> vi.V
    (40.048625392716815, 33.65371175967546)
    >>> vi.policy
    (1, 0)
    
    """
    
    def __init__(self, transitions, reward, discount, epsilon=0.01,
                 max_iter=1000, initial_value=0):
        # Initialise a value iteration MDP.
        
        MDP.__init__(self, transitions, reward, discount, epsilon, max_iter)
        
        # initialization of optional arguments
        if initial_value == 0:
            self.V = zeros(self.S)
        else:
            assert len(initial_value) == self.S, "The initial value must be " \
                "a vector of length S."
            self.V = array(initial_value).reshape(self.S)
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
        
        # Call the iteration method
        #self.run()
    
    def _boundIter(self, epsilon):
        # Compute a bound for the number of iterations.
        #
        # for the value iteration
        # algorithm to find an epsilon-optimal policy with use of span for the 
        # stopping criterion
        #
        # Arguments -----------------------------------------------------------
        # Let S = number of states, A = number of actions
        #    epsilon   = |V - V*| < epsilon,  upper than 0,
        #        optional (default : 0.01)
        # Evaluation ----------------------------------------------------------
        #    max_iter  = bound of the number of iterations for the value 
        #    iteration algorithm to find an epsilon-optimal policy with use of
        #    span for the stopping criterion
        #    cpu_time  = used CPU time
        #
        # See Markov Decision Processes, M. L. Puterman, 
        # Wiley-Interscience Publication, 1994 
        # p 202, Theorem 6.6.6
        # k =    max     [1 - S min[ P(j|s,a), p(j|s',a')] ]
        #     s,a,s',a'       j
        k = 0
        h = zeros(self.S)
        
        for ss in range(self.S):
            PP = zeros((self.A, self.S))
            for aa in range(self.A):
                try:
                    PP[aa] = self.P[aa][:, ss]
                except ValueError:
                    PP[aa] = self.P[aa][:, ss].todense().A1
            # minimum of the entire array.
            h[ss] = PP.min()
        
        k = 1 - h.sum()
        Vprev = self.V
        null, value = self._bellmanOperator()
        # p 201, Proposition 6.6.5
        max_iter = (log((epsilon * (1 - self.discount) / self.discount) /
                    getSpan(value - Vprev) ) / log(self.discount * k))
        #self.V = Vprev
        
        self.max_iter = int(ceil(max_iter))
    
    def run(self):
        # Run the value iteration algorithm.
        
        if self.verbose:
            print('\tIteration\tV-variation')
        
        self.time = time()
        while True:
            self.iter += 1
            
            Vprev = self.V.copy()
            
            # Bellman Operator: compute policy and value functions
            self.policy, self.V = self._bellmanOperator()
            
            # The values, based on Q. For the function "max()": the option
            # "axis" means the axis along which to operate. In this case it
            # finds the maximum of the the rows. (Operates along the columns?)
            variation = getSpan(self.V - Vprev)
            
            if self.verbose:
                print(("\t%s\t%s" % (self.iter, variation)))
            
            if variation < self.thresh:
                if self.verbose:
                    print("Iteration stopped, epsilon-optimal policy found.")
                break
            elif (self.iter == self.max_iter):
                if self.verbose:
                    print("Iteration stopped by maximum number of iterations "
                          "condition.")
                break
        
        # store value and policy as tuples
        self.V = tuple(self.V.tolist())
        self.policy = tuple(self.policy.tolist())
        
        self.time = time() - self.time

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
    >>> import mdptoolbox, mdptoolbox.example
    >>> P, R = mdptoolbox.example.forest()
    >>> vigs = mdptoolbox.mdp.ValueIterationGS(P, R, 0.9)
    >>> vigs.run()
    >>> vigs.V
    (25.5833879767579, 28.830654635546928, 32.83065463554693)
    >>> vigs.policy
    (0, 0, 0)
    
    """
    
    def __init__(self, transitions, reward, discount, epsilon=0.01,
                 max_iter=10, initial_value=0):
        # Initialise a value iteration Gauss-Seidel MDP.
        
        MDP.__init__(self, transitions, reward, discount, epsilon, max_iter)
        
        # initialization of optional arguments
        if initial_value == 0:
            self.V = zeros(self.S)
        else:
            if len(initial_value) != self.S:
                raise ValueError("The initial value must be a vector of "
                                 "length S.")
            else:
                try:
                    self.V = initial_value.reshape(self.S)
                except AttributeError:
                    self.V = array(initial_value)
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
        
        # Call the iteration method
        #self.run()
    
    def run(self):
        # Run the value iteration Gauss-Seidel algorithm.
        
        done = False
        
        if self.verbose:
            print('  Iteration  V_variation')
        
        self.time = time()
        
        while not done:
            self.iter += 1
            
            Vprev = self.V.copy()
            
            for s in range(self.S):
                Q = [float(self.R[a][s]+
                           self.discount * self.P[a][s, :].dot(self.V))
                     for a in range(self.A)]
                
                self.V[s] = max(Q)
            
            variation = getSpan(self.V - Vprev)
            
            if self.verbose:
                print(("      %s         %s" % (self.iter, variation)))
            
            if variation < self.thresh: 
                done = True
                if self.verbose:
                    print("Iterations stopped, epsilon-optimal policy found.")
             
            elif self.iter == self.max_iter:
                done = True 
                if self.verbose:
                    print("Iterations stopped by maximum number of iteration "
                          "condition.")
        
        self.policy = []
        for s in range(self.S):
            Q = zeros(self.A)
            for a in range(self.A):
                Q[a] =  self.R[a][s] + self.discount * self.P[a][s,:].dot(self.V)
            
            self.V[s] = Q.max()
            self.policy.append(int(Q.argmax()))

        self.time = time() - self.time
        
        self.V = tuple(self.V.tolist())
        self.policy = tuple(self.policy)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
