# -*- coding: utf-8 -*-
"""Markov Decision Process (MDP) Toolbox
=====================================

The MDP toolbox provides classes and functions for the resolution of
descrete-time Markov Decision Processes.

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

Available functions
-------------------
check
    Check that an MDP is properly defined
checkSquareStochastic
    Check that a matrix is square and stochastic
exampleForest
    A simple forest management example
exampleRand
    A random example

How to use the documentation
----------------------------
Documentation is available both as docstrings provided with the code and
in html or pdf format from 
`The MDP toolbox homepage <http://www.somewhere.com>`_. The docstring
examples assume that the `mdp` module has been imported imported like so::

  >>> import mdptoolbox.mdp as mdp

Code snippets are indicated by three greater-than signs::

  >>> x = 17
  >>> x = x + 1
  >>> x
  18

The documentation can be displayed with
`IPython <http://ipython.scipy.org>`_. For example, to view the docstring of
the ValueIteration class use ``mdp.ValueIteration?<ENTER>``, and to view its
source code use ``mdp.ValueIteration??<ENTER>``.

Acknowledgments
---------------
This module is modified from the MDPtoolbox (c) 2009 INRA available at 
http://www.inra.fr/mia/T/MDPtoolbox/.

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

from numpy import absolute, array, empty, mean, mod, multiply
from numpy import ndarray, ones, zeros
from numpy.random import randint, random
from scipy.sparse import csr_matrix as sparse

from utils import check, getSpan

# These need to be fixed so that we use classes derived from Error.
mdperr = {
"mat_nonneg" :
    "PyMDPtoolbox: Probabilities must be non-negative.",
"mat_square" :
    "PyMDPtoolbox: The matrix must be square.",
"mat_stoch" :
    "PyMDPtoolbox: Rows of the matrix must sum to one (1).",
"mask_numpy" :
    "PyMDPtoolbox: mask must be a numpy array or matrix; i.e. type(mask) is "
    "ndarray or type(mask) is matrix.", 
"mask_SbyS" : 
    "PyMDPtoolbox: The mask must have shape SxS; i.e. mask.shape = (S, S).",
"obj_shape" :
    "PyMDPtoolbox: Object arrays for transition probabilities and rewards "
    "must have only 1 dimension: the number of actions A. Each element of "
    "the object array contains an SxS ndarray or matrix.",
"obj_square" :
    "PyMDPtoolbox: Each element of an object array for transition "
    "probabilities and rewards must contain an SxS ndarray or matrix; i.e. "
    "P[a].shape = (S, S) or R[a].shape = (S, S).",
"P_type" :
    "PyMDPtoolbox: The transition probabilities must be in a numpy array; "
    "i.e. type(P) is ndarray.",
"P_shape" :
    "PyMDPtoolbox: The transition probability array must have the shape "
    "(A, S, S)  with S : number of states greater than 0 and A : number of "
    "actions greater than 0. i.e. R.shape = (A, S, S)",
"PR_incompat" :
    "PyMDPtoolbox: Incompatibility between P and R dimensions.",
"prob_in01" :
    "PyMDPtoolbox: Probability p must be in [0; 1].",
"R_type" :
    "PyMDPtoolbox: The rewards must be in a numpy array; i.e. type(R) is "
    "ndarray, or numpy matrix; i.e. type(R) is matrix.",
"R_shape" :
    "PyMDPtoolbox: The reward matrix R must be an array of shape (A, S, S) or "
    "(S, A) with S : number of states greater than 0 and A : number of "
    "actions greater than 0. i.e. R.shape = (S, A) or (A, S, S).",
"R_gt_0" :
    "PyMDPtoolbox: The rewards must be greater than 0.",
"S_gt_1" :
    "PyMDPtoolbox: Number of states S must be greater than 1.",
"SA_gt_1" : 
    "PyMDPtoolbox: The number of states S and the number of actions A must be "
    "greater than 1.",
"discount_rng" : 
    "PyMDPtoolbox: Discount rate must be in ]0; 1]",
"maxi_min" :
    "PyMDPtoolbox: The maximum number of iterations must be greater than 0"
}

class MDP(object):
    
    """A Markov Decision Problem.
    
    Parameters
    ----------
    transitions : array
            transition probability matrices
    reward : array
            reward matrices
    discount : float or None
            discount factor
    epsilon : float or None
            stopping criteria
    max_iter : int or None
            maximum number of iterations
    
    Attributes
    ----------
    P : array
        Transition probability matrices
    R : array
        Reward matrices
    V : list
        Value function
    discount : float
        b
    max_iter : int
        a
    policy : list
        a
    time : float
        a
    verbose : logical
        a
    
    Methods
    -------
    iterate
        To be implemented in child classes, raises exception
    setSilent
        Turn the verbosity off
    setVerbose
        Turn the verbosity on
    
    """
    
    def __init__(self, transitions, reward, discount, epsilon, max_iter):
        # Initialise a MDP based on the input parameters.
        
        # if the discount is None then the algorithm is assumed to not use it
        # in its computations
        if type(discount) in (int, float):
            if (discount <= 0) or (discount > 1):
                raise ValueError(mdperr["discount_rng"])
            else:
                if discount == 1:
                    print("PyMDPtoolbox WARNING: check conditions of "
                          "convergence. With no discount, convergence is not "
                          "always assumed.")
                self.discount = discount
        elif discount is not None:
            raise ValueError("PyMDPtoolbox: the discount must be a positive "
                             "real number less than or equal to one.")
        # if the max_iter is None then the algorithm is assumed to not use it
        # in its computations
        if type(max_iter) in (int, float):
            if max_iter <= 0:
                raise ValueError(mdperr["maxi_min"])
            else:
                self.max_iter = max_iter
        elif max_iter is not None:
            raise ValueError("PyMDPtoolbox: max_iter must be a positive real "
                             "number greater than zero.")
        # check that epsilon is something sane
        if type(epsilon) in (int, float):
            if epsilon <= 0:
                raise ValueError("PyMDPtoolbox: epsilon must be greater than "
                                 "0.")
        elif epsilon is not None:
            raise ValueError("PyMDPtoolbox: epsilon must be a positive real "
                             "number greater than zero.")
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
    
    def _bellmanOperator(self, V=None):
        # Apply the Bellman operator on the value function.
        # Updates the value function and the Vprev-improving policy.
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
                if V.shape not in ((self.S,), (1, self.S)):
                    raise ValueError("bellman: V is not the right shape.")
            except AttributeError:
                raise TypeError("bellman: V must be a numpy array or matrix.")
        # Looping through each action the the Q-value matrix is calculated
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
        except:
            raise
        # convert Ps to matrices
        self.P = []
        for aa in xrange(self.A):
            self.P.append(P[aa])
        self.P = tuple(self.P)
        # Set self.R as a tuple of length A, with each element storing an 1×S
        # vector.
        try:
            if R.ndim == 2:
                self.R = []
                for aa in xrange(self.A):
                    self.R.append(array(R[:, aa]).reshape(self.S))
            else:
                raise AttributeError
        except AttributeError:
            self.R = []
            for aa in xrange(self.A):
                try:
                    self.R.append(P[aa].multiply(R[aa]).sum(1).reshape(self.S))
                except AttributeError:
                    self.R.append(multiply(P[aa],R[aa]).sum(1).reshape(self.S))
                except:
                    raise
        except:
            raise
        self.R = tuple(self.R)
    
    def _iterate(self):
        # Raise error because child classes should implement this function.
        raise NotImplementedError("You should create an _iterate() method.")
    
    def setSilent(self):
        """Set the MDP algorithm to silent mode."""
        self.verbose = False
    
    def setVerbose(self):
        """Set the MDP algorithm to verbose mode."""
        self.verbose = True

class FiniteHorizon(MDP):
    
    """A MDP solved using the finite-horizon backwards induction algorithm.
    
    Let S = number of states, A = number of actions
    
    Parameters
    ----------
    P(SxSxA) = transition matrix 
             P could be an array with 3 dimensions ora cell array (1xA),
             each cell containing a matrix (SxS) possibly sparse
    R(SxSxA) or (SxA) = reward matrix
             R could be an array with 3 dimensions (SxSxA) or 
             a cell array (1xA), each cell containing a sparse matrix (SxS) or
             a 2D array(SxA) possibly sparse  
    discount = discount factor, in ]0, 1]
    N        = number of periods, upper than 0
    h(S)     = terminal reward, optional (default [0; 0; ... 0] )
    
    Attributes
    ----------
    
    Methods
    -------
    V(S,N+1)     = optimal value function
                 V(:,n) = optimal value function at stage n
                        with stage in 1, ..., N
                        V(:,N+1) = value function for terminal stage 
    policy(S,N)  = optimal policy
                 policy(:,n) = optimal policy at stage n
                        with stage in 1, ...,N
                        policy(:,N) = policy for stage N
    cpu_time = used CPU time
  
    Notes
    -----
    In verbose mode, displays the current stage and policy transpose.
    
    Examples
    --------
    >>> import mdptoolbox, mdptoolbox.example
    >>> P, R = mdptoolbox.example.forest()
    >>> fh = mdptoolbox.mdp.FiniteHorizon(P, R, 0.9, 3)
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
        if N < 1:
            raise ValueError('PyMDPtoolbox: N must be greater than 0')
        else:
            self.N = N
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
        self._iterate()
        
    def _iterate(self):
        # Run the finite horizon algorithm.
        self.time = time()
        # loop through each time period
        for n in range(self.N):
            W, X = self._bellmanOperator(self.V[:, self.N - n])
            self.V[:, self.N - n - 1] = X
            self.policy[:, self.N - n - 1] = W
            if self.verbose:
                print("stage: %s ... policy transpose : %s") % (
                    self.N - n, self.policy[:, self.N - n -1].tolist())
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

    Arguments
    ---------
    Let S = number of states, A = number of actions
    P(SxSxA) = transition matrix 
             P could be an array with 3 dimensions or a cell array (1xA),
             each cell containing a matrix (SxS) possibly sparse
    R(SxSxA) or (SxA) = reward matrix
             R could be an array with 3 dimensions (SxSxA) or 
             a cell array (1xA), each cell containing a sparse matrix (SxS) or
             a 2D array(SxA) possibly sparse  
    discount = discount rate, in ]0; 1[
    h(S)     = terminal reward, optional (default [0; 0; ... 0] )
    
    Evaluation
    ----------
    V(S)   = optimal values
    policy(S) = optimal policy
    cpu_time = used CPU time
    
    Notes    
    -----
    In verbose mode, displays the current stage and policy transpose.
    
    Examples
    --------
    >>> import mdptoolbox, mdptoolbox.example
    >>> P, R = mdptoolbox.example.forest()
    >>> lp = mdptoolbox.mdp.LP(P, R, 0.9)
    
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
        self._iterate()
    
    def _iterate(self):
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
    Let S = number of states, A = number of actions
    P(SxSxA) = transition matrix 
             P could be an array with 3 dimensions or a cell array (1xA),
             each cell containing a matrix (SxS) possibly sparse
    R(SxSxA) or (SxA) = reward matrix
             R could be an array with 3 dimensions (SxSxA) or 
             a cell array (1xA), each cell containing a sparse matrix (SxS) or
             a 2D array(SxA) possibly sparse  
    discount = discount rate, in ]0, 1[
    policy0(S) = starting policy, optional 
    max_iter = maximum number of iteration to be done, upper than 0, 
             optional (default 1000)
    eval_type = type of function used to evaluate policy: 
             0 for mdp_eval_policy_matrix, else mdp_eval_policy_iterative
             optional (default 0)
             
    Evaluation
    ----------
    V(S)   = value function 
    policy(S) = optimal policy
    iter     = number of done iterations
    cpu_time = used CPU time
    
    Notes
    -----
    In verbose mode, at each iteration, displays the number 
    of differents actions between policy n-1 and n
    
    Examples
    --------
    >>> import mdptoolbox, mdptoolbox.example
    >>> P, R = mdptoolbox.example.rand()
    >>> pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.9)
    
    >>> P, R = mdptoolbox.example.forest()
    >>> pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.9)
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
            if not policy0.shape in ((self.S, ), (self.S, 1), (1, self.S)):
                raise ValueError("PyMDPtolbox: policy0 must a vector with "
                                 "length S.")
            # reshape the policy to be a vector
            policy0 = policy0.reshape(self.S)
            # The policy can only contain integers between 1 and S
            if (mod(policy0, 1).any() or (policy0 < 0).any() or
                    (policy0 >= self.S).any()):
                raise ValueError("PyMDPtoolbox: policy0 must be a vector of "
                                 "integers between 1 and S.")
            else:
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
        self._iterate()
    
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
                Ppolicy[ind, :] = self.P[aa][ind, :]
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
        if (type(V0) in (int, float)) and (V0 == 0):
            policy_V = zeros(self.S)
        else:
            if (type(V0) in (ndarray)) and (V0.shape == (self.S, 1)):
                policy_V = V0
            else:
                raise ValueError("PyMDPtoolbox: V0 vector/array type not "
                                 "supported. Use ndarray of matrix column "
                                 "vector length S.")
        
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
                print('      %s         %s') % (itr, variation)
            
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
    
    def _iterate(self):
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
                print('       %s                 %s') % (self.iter,
                                                         n_different)
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
    Let S = number of states, A = number of actions
    P(SxSxA) = transition matrix 
             P could be an array with 3 dimensions or a cell array (1xA),
             each cell containing a matrix (SxS) possibly sparse
    R(SxSxA) or (SxA) = reward matrix
             R could be an array with 3 dimensions (SxSxA) or 
             a cell array (1xA), each cell containing a sparse matrix (SxS) or
             a 2D array(SxA) possibly sparse  
    discount = discount rate, in ]0, 1[
    policy0(S) = starting policy, optional 
    max_iter = maximum number of iteration to be done, upper than 0, 
             optional (default 1000)
    eval_type = type of function used to evaluate policy: 
             0 for mdp_eval_policy_matrix, else mdp_eval_policy_iterative
             optional (default 0)
    
    Data Attributes
    ---------------
    V(S)   = value function 
    policy(S) = optimal policy
    iter     = number of done iterations
    cpu_time = used CPU time
    
    Notes
    -----
    In verbose mode, at each iteration, displays the number 
    of differents actions between policy n-1 and n
    
    Examples
    --------
    >>> import mdptoolbox, mdptoolbox.example
    >>> P, R = mdptoolbox.example.forest()
    >>> pim = mdptoolbox.mdp.PolicyIterationModified(P, R, 0.9)
    >>> pim.policy
    FIXME
    >>> pim.V
    FIXME
    
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
        if type(epsilon) in (int, float):
            if epsilon <= 0:
                raise ValueError("PyMDPtoolbox: epsilon must be greater than "
                                 "0.")
        else:
            raise ValueError("PyMDPtoolbox: epsilon must be a positive real "
                             "number greater than zero.")
        
        # computation of threshold of variation for V for an epsilon-optimal
        # policy
        if self.discount != 1:
            self.thresh = epsilon * (1 - self.discount) / self.discount
        else:
            self.thresh = epsilon
        
        self.epsilon = epsilon
        
        if discount == 1:
            self.V = zeros((self.S, 1))
        else:
            # min(min()) is not right
            self.V = 1 / (1 - discount) * self.R.min() * ones((self.S, 1))
        
        # Call the iteration method
        self._iterate()
    
    def _iterate(self):
        # Run the modified policy iteration algorithm.
        
        if self.verbose:
            print('  Iteration  V_variation')
        
        self.time = time()
        
        done = False
        while not done:
            self.iter += 1
            
            self.policy, Vnext = self._bellmanOperator()
            #[Ppolicy, PRpolicy] = mdp_computePpolicyPRpolicy(P, PR, policy);
            
            variation = getSpan(Vnext - self.V)
            if self.verbose:
                print("      %s         %s" % (self.iter, variation))
            
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
    
    Let S = number of states, A = number of actions
    
    Parameters
    ----------
    P : transition matrix (SxSxA)
        P could be an array with 3 dimensions or a cell array (1xA), each
        cell containing a sparse matrix (SxS)
    R : reward matrix(SxSxA) or (SxA)
        R could be an array with 3 dimensions (SxSxA) or a cell array
        (1xA), each cell containing a sparse matrix (SxS) or a 2D
        array(SxA) possibly sparse
    discount : discount rate
        in ]0; 1[    
    n_iter : number of iterations to execute (optional).
        Default value = 10000; it is an integer greater than the default
        value.
    
    Results
    -------
    Q : learned Q matrix (SxA) 
    
    V : learned value function (S).
    
    policy : learned optimal policy (S).
    
    mean_discrepancy : vector of V discrepancy mean over 100 iterations
        Then the length of this vector for the default value of N is 100 
        (N/100).

    Examples
    ---------
    >>> # These examples are reproducible only if random seed is set to 0 in
    >>> # both the random and numpy.random modules.
    >>> import numpy as np
    >>> import mdptoolbox, mdptoolbox.example
    >>> np.random.seed(0)
    >>> P, R = mdptoolbox.example.forest()
    >>> ql = mdptoolbox.mdp.QLearning(P, R, 0.96)
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
    >>> pim = mdptoolbox.mdp.QLearning(P, R, 0.9)
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
        if (n_iter < 10000):
            raise ValueError("PyMDPtoolbox: n_iter should be greater than "
                             "10000.")
        
        # We don't want to send this to MDP because _computePR should not be
        # run on it
        # MDP.__init__(self, transitions, reward, discount, None, n_iter)
        check(transitions, reward)
        
        if (transitions.dtype is object):
            self.P = transitions
            self.A = self.P.shape[0]
            self.S = self.P[0].shape[0]
        else: # convert to an object array
            self.A = transitions.shape[0]
            self.S = transitions.shape[1]
            self.P = zeros(self.A, dtype=object)
            for aa in range(self.A):
                self.P[aa] = transitions[aa, :, :]
        
        self.R = reward
        
        self.discount = discount
        
        self.max_iter = n_iter
        
        # Initialisations
        self.Q = zeros((self.S, self.A))
        self.mean_discrepancy = []
        
        # Call the iteration method
        self._iterate()
        
    def _iterate(self):
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
            
            if (self.R.dtype == object):
                r = self.R[a][s, s_new]
            elif (self.R.ndim == 3):
                r = self.R[a, s, s_new]
            else:
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
    Let S = number of states, A = number of actions
    P(SxSxA) = transition matrix 
             P could be an array with 3 dimensions or a cell array (1xA), 
             each cell containing a matrix (SxS) possibly sparse
    R(SxSxA) or (SxA) = reward matrix
             R could be an array with 3 dimensions (SxSxA) or 
             a cell array (1xA), each cell containing a sparse matrix (SxS) or
             a 2D array(SxA) possibly sparse  
    epsilon  = epsilon-optimal policy search, upper than 0, 
             optional (default: 0.01)
    max_iter = maximum number of iteration to be done, upper than 0,
             optional (default 1000)
    
    Evaluation
    ----------
    policy(S)       = epsilon-optimal policy
    average_reward  = average reward of the optimal policy
    cpu_time = used CPU time
    
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
    >>> rvi.average_reward
    2.4300000000000002
    >>> rvi.policy
    (0, 0, 0)
    >>> rvi.iter
    4
    
    >>> import mdptoolbox
    >>> import numpy as np
    >>> P = np.array([[[0.5, 0.5],[0.8, 0.2]],[[0, 1],[0.1, 0.9]]])
    >>> R = np.array([[5, 10], [-1, 2]])
    >>> rvi = mdptoolbox.mdp.RelativeValueIteration(P, R)
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
        self._iterate()
    
    def _iterate(self):
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
                print("      %s         %s" % (self.iter, variation))
            
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
    mdp.ValueIteration applies the value iteration algorithm to solve
    discounted MDP. The algorithm consists in solving Bellman's equation
    iteratively.
    Iterating is stopped when an epsilon-optimal policy is found or after a
    specified number (max_iter) of iterations. 
    This function uses verbose and silent modes. In verbose mode, the function
    displays the variation of V (value function) for each iteration and the
    condition which stopped iterations: epsilon-policy found or maximum number
    of iterations reached.
    
    Let S = number of states, A = number of actions.
    
    Parameters
    ----------
    P : array
        transition matrix 
        P could be a numpy ndarray with 3 dimensions (AxSxS) or a 
        numpy ndarray of dytpe=object with 1 dimenion (1xA), each 
        element containing a numpy ndarray (SxS) or scipy sparse matrix. 
    R : array
        reward matrix
        R could be a numpy ndarray with 3 dimensions (AxSxS) or numpy
        ndarray of dtype=object with 1 dimension (1xA), each element
        containing a sparse matrix (SxS). R also could be a numpy 
        ndarray with 2 dimensions (SxA) possibly sparse.
    discount : float
        discount rate
        Greater than 0, less than or equal to 1. Beware to check conditions of
        convergence for discount = 1.
    epsilon : float, optional
        epsilon-optimal policy search
        Greater than 0, optional (default: 0.01).
    max_iter : int, optional
        maximum number of iterations to be done
        Greater than 0, optional (default: computed)
    initial_value : array, optional
        starting value function
        optional (default: zeros(S,)).
    
    Data Attributes
    ---------------
    V : value function
        A vector which stores the optimal value function. Prior to calling the
        _iterate() method it has a value of None. Shape is (S, ).
    policy : epsilon-optimal policy
        A vector which stores the optimal policy. Prior to calling the
        _iterate() method it has a value of None. Shape is (S, ).
    iter : number of iterations taken to complete the computation
        An integer
    time : used CPU time
        A float
    
    Methods
    -------
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
    >>> vi.V
    (5.93215488, 9.38815488, 13.38815488)
    >>> vi.policy
    (0, 0, 0)
    >>> vi.iter
    4
    >>> vi.time
    0.0009911060333251953
    
    >>> import mdptoolbox
    >>> import numpy as np
    >>> P = np.array([[[0.5, 0.5],[0.8, 0.2]],[[0, 1],[0.1, 0.9]]])
    >>> R = np.array([[5, 10], [-1, 2]])
    >>> vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    >>> vi.V
    (40.048625392716815, 33.65371175967546)
    >>> vi.policy
    (1, 0)
    >>> vi.iter
    26
    >>> vi.time
    0.0066509246826171875
    
    >>> import mdptoolbox
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix as sparse
    >>> P = [None] * 2
    >>> P[0] = sparse([[0.5, 0.5],[0.8, 0.2]])
    >>> P[1] = sparse([[0, 1],[0.1, 0.9]])
    >>> R = np.array([[5, 10], [-1, 2]])
    >>> vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
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
            if len(initial_value) != self.S:
                raise ValueError("PyMDPtoolbox: The initial value must be "
                                 "a vector of length S.")
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
        self._iterate()
    
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
                except:
                    raise
            # the function "min()" without any arguments finds the
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
    
    def _iterate(self):
        # Run the value iteration algorithm.
        
        if self.verbose:
            print('  Iteration  V_variation')
        
        self.time = time()
        done = False
        while not done:
            self.iter += 1
            
            Vprev = self.V.copy()
            
            # Bellman Operator: compute policy and value functions
            self.policy, self.V = self._bellmanOperator()
            
            # The values, based on Q. For the function "max()": the option
            # "axis" means the axis along which to operate. In this case it
            # finds the maximum of the the rows. (Operates along the columns?)
            variation = getSpan(self.V - Vprev)
            
            if self.verbose:
                print("      %s         %s" % (self.iter, variation))
            
            if variation < self.thresh:
                done = True
                if self.verbose:
                    print("...iterations stopped, epsilon-optimal policy "
                          "found.")
            elif (self.iter == self.max_iter):
                done = True 
                if self.verbose:
                    print("...iterations stopped by maximum number of "
                          "iteration condition.")
        
        # store value and policy as tuples
        self.V = tuple(self.V.tolist())
        self.policy = tuple(self.policy.tolist())
        
        self.time = time() - self.time

class ValueIterationGS(ValueIteration):
    
    """
    A discounted MDP solved using the value iteration Gauss-Seidel algorithm.
    
    Arguments
    ---------
    Let S = number of states, A = number of actions
    P(SxSxA)  = transition matrix 
             P could be an array with 3 dimensions or a cell array (1xA),
             each cell containing a matrix (SxS) possibly sparse
    R(SxSxA) or (SxA) = reward matrix
             R could be an array with 3 dimensions (SxSxA) or 
             a cell array (1xA), each cell containing a sparse matrix (SxS) or
             a 2D array(SxA) possibly sparse  
    discount  = discount rate in ]0; 1]
              beware to check conditions of convergence for discount = 1.
    epsilon   = epsilon-optimal policy search, upper than 0,
              optional (default : 0.01)
    max_iter  = maximum number of iteration to be done, upper than 0, 
              optional (default : computed)
    V0(S)     = starting value function, optional (default : zeros(S,1))
    
    Evaluation
    ----------
    policy(S) = epsilon-optimal policy
    iter      = number of done iterations
    cpu_time  = used CPU time
    
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
    
    """
    
    def __init__(self, transitions, reward, discount, epsilon=0.01,
                 max_iter=10, initial_value=0):
        # Initialise a value iteration Gauss-Seidel MDP.
        
        ValueIteration.__init__(self, transitions, reward, discount, epsilon,
                                max_iter, initial_value)
        # Call the iteration method
        self._iterate()
    
    def _iterate(self):
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
                print("      %s         %s" % (self.iter, variation))
            
            if variation < self.thresh: 
                done = True
                if self.verbose:
                    print("MDP Toolbox : iterations stopped, epsilon-optimal "
                          "policy found.")
             
            elif self.iter == self.max_iter:
                done = True 
                if self.verbose:
                    print("MDP Toolbox : iterations stopped by maximum number "
                          "of iteration condition.")
        
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
