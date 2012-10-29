# -*- coding: utf-8 -*-
"""
Copyright (c) 2011, 2012 Steven Cordwell
Copyright (c) 2009 Iadine Chadès
Copyright (c) 2009 Marie-Josée Cros
Copyright (c) 2009 Frédérick Garcia
Copyright (c) 2009 Régis Sabbadin

All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
  * Neither the name of the <ORGANIZATION> nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON 
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from numpy import abs, array, matrix, ndarray, ones, zeros
from random import randint, random
from math import ceil, log, sqrt
from time import time

class MDP():
    """The Markov Decision Problem Toolbox."""
    def bellmanOperator(self):
        """Apply the Bellman operator on the value function.
        
        Returns a new value function and a Vprev-improving policy
        Parameters
        --------------------------------------------------------------
        Let S = number of states, A = number of actions
            P(SxSxA) = transition matrix
                     P could be an array with 3 dimensions ora cell array (1xA), 
                     each cell containing a matrix (SxS) possibly sparse
            PR(SxA) = reward matrix
                     PR could be an array with 2 dimensions or a sparse matrix
            discount = discount rate, in ]0, 1]
            Vprev(S) = value function
        Returns
        -------------------------------------------------------------
            V(S)   = new value function
            policy(S) = Vprev-improving policy
        """
        Q = matrix(zeros((self.S, self.A)))
        for aa in range(self.A):
            Q[:, aa] = self.R[:, aa] + (self.discount * self.P[aa] * self.value)
        
        # update the value and policy
        self.value = Q.max(axis=1)
        self.policy = Q.argmax(axis=1)
    
    def check(self, P, R):
        """Check if the matrices P and R define a Markov Decision Process
        
        Let S = number of states, A = number of actions
            The transition matrix P must be on the form P(AxSxS) and P[a,:,:]
            must be stochastic
            The reward matrix R must be on the form (SxSxA) or (SxA)
        Arguments
        --------------------------------------------------------------
            P(SxSxA) = transition matrix
                P could be an array with 3 dimensions or a cell array (1xA),
                each cell containing a matrix (SxS) possibly sparse
            R(SxSxA) or (SxA) = reward matrix
                 R could be an array with 3 dimensions (SxSxA) or a cell array
                 (1xA), each cell containing a sparse matrix (SxS) or a 2D
                 array(SxA) possibly sparse  
        Evaluation
        -------------------------------------------------------------
            is_mdp = True if P and R define a Markov Decision Process, False
            otherwise
            err_msg = error message or None if correct
        """
        is_error_detected = False
        err_msg = None
        
        # Check of P
        # tranitions must be a numpy array either an AxSxS ndarray (with any 
        # dtype other than "object"); or, a 1xA ndarray with a "object" dtype, 
        # and each element containing an SxS array. An AxSxS array will be
        # be converted to an object array. A numpy object array is similar to a
        # MATLAB cell array.
        if (not type(P) is ndarray):
            return(False, "The transition probabilities must be a numpy array.")
        elif ((type(P) is ndarray) and (not P.dtype is object) and (P.ndim != 3)):
            return(False, "The transition probability array must have 3 dimensions: AxSxS.")
        elif ((type(P) is ndarray) and (P.dtype is object) and (P.ndim > 1)):
            return(False, "You are using an object array for the transition probability: The array must have only 1 dimension: A. Each element of the contains a SxS array.")

        if (not type(R) is ndarray):
            return(False, "The reward must be a numpy array.")
        elif ((type(R) is ndarray) and (not R.dtype is object) and (not R.ndim in (2, 3))):
            return(False, "The reward array must have 2 or 3 dimensions: AxSxS or SxA.")
        elif ((type(R) is ndarray) and (R.dtype is object) and (R.ndim > 1)):
            return(False, "You are using an object array for the reward: The array must have only 1 dimension: A. Each element of the contains a SxS array.")

        if (P.dtype is object):
            P_is_object = True
        else:
            P_is_object = False
            
        if (R.dtype is object):
            R_is_object = True
        else:
            R_is_object = False
        
        if P_is_object:
            aP = P.shape[0]
            sP0 = P[0].shape[0]
            sP1 = P[0].shape[1]
            # check to see that each object array element is the same size
            for aa in range(1, aP):
                sP0aa = P[aa].shape[0]
                sP1aa = P[aa].shape[1]
                if ((sP0aa != sP0) or (sP1aa != sP1)):
                    is_error_detected = True
                    err_msg = "You are using and object array for the transition probability: The dimensions of each array within the object array must be equal to each other."
                    break
        else:
            aP, sP0, sP1 = P.shape
        
        if ((sP0 < 1) or (aP < 1) or (sP0 != sP1)):
            is_error_detected = True
            err_msg = "The transition probability array must have the shape (A, S, S)  with S : number of states greater than 0 and A : number of actions greater than 0."
        
        if (not is_error_detected):
            aa = 1
            while aa <= aP:
                if P_is_object:
                    err_msg = self.checkSquareStochastic(P[aa])
                else:
                    err_msg = self.checkSquareStochastic(P[aa, :, :])
                
                if (err_msg == None):
                    aa = aa + 1
                else:
                    is_error_detected = True
                    aa = aP + 1
        
        if (not is_error_detected):
            if R_is_object:
                sR0 = R[0].shape[0]
                sR1 = R[0].shape[1]
                aR = R.shape[0]
            elif R.ndim == 3:
                aR, sR0, sR1 = R.shape
            else:
                sR0, aR = R.shape
                sR1 = sR0
            
            if ((sR0 < 1) or (aR < 1) or (sR0 != sR1)):
                is_error_detected = True
                err_msg = "MDP Toolbox ERROR: The reward matrix R must be an array (S,S,A) or (SxA) with S : number of states greater than 0 and A : number of actions greater than 0"
                is_error_detected = True
                
        if (not is_error_detected):
            if (sP0 != sR0) or (aP != aR):
                err_msg = "MDP Toolbox ERROR: Incompatibility between P and R dimensions"
                is_error_detected = True
        
        return(((not is_error_detected), err_msg))
    
    def checkSquareStochastic(self, Z):
        """Check if Z is a square stochastic matrix
        
        Arguments
        --------------------------------------------------------------
            Z = a numpy ndarray SxS
        Evaluation
        -------------------------------------------------------------
            error_msg = error message or None if correct
        """
        s1, s2 = Z.shape
        if (s1 != s2):
           return('MDP Toolbox ERROR: Matrix must be square')
        elif (abs(Z.sum(axis=1) - ones(s2))).max() > 10**(-12):
           return('MDP Toolbox ERROR: Row sums of the matrix must be 1')
        elif (Z < 0).any():
           return('MDP Toolbox ERROR: Probabilities must be non-negative')
        else:
            return(None)

    def computePpolicyPRpolicy(self):
        """Computes the transition matrix and the reward matrix for a policy.
        """
        pass    
    
    def computePR(self, P, R):
        """Computes the reward for the system in one state chosing an action
        
        Arguments
        --------------------------------------------------------------
        Let S = number of states, A = number of actions
            P(SxSxA)  = transition matrix 
                P could be an array with 3 dimensions or  a cell array (1xA), 
                each cell containing a matrix (SxS) possibly sparse
            R(SxSxA) or (SxA) = reward matrix
                R could be an array with 3 dimensions (SxSxA) or  a cell array 
                (1xA), each cell containing a sparse matrix (SxS) or a 2D 
                array(SxA) possibly sparse  
        Evaluation
        -------------------------------------------------------------
            PR(SxA)   = reward matrix
        """
        # make P be an object array with (S, S) shaped array elements
        if (P.dtype == object):
            self.P = P
            self.A = self.P.shape[0]
            self.S = self.P[0].shape[0]
        else: # convert to an object array
            self.A = P.shape[0]
            self.S = P.shape[1]
            self.P = zeros(self.A, dtype=object)
            for aa in range(self.A):
                self.P[aa] = P[aa, :, :]
        
        # make R have the shape (S, A)
        if ((R.ndim == 2) and (not R.dtype is object)):
            # R already has shape (S, A)
            self.R = R
        else: 
            # R has shape (A, S, S) or object shaped (A,) with each element
            # shaped (S, S)
            self.R = zeros((self.S, self.A))
            if (R.dtype is object):
                for aa in range(self.A):
                    self.R[:, aa] = sum(P[aa] * R[aa], 2)
            else:
                for aa in range(self.A):
                    self.R[:, aa] = sum(P[aa] * R[aa, :, :], 2)
        
        # convert the arrays to numpy matrices
        for aa in range(self.A):
            if (type(self.P[aa]) is ndarray):
                self.P[aa] = matrix(self.P[aa])
        if (type(self.R) is ndarray):
            self.R = matrix(self.R)
    
    def silent(self):
        """Ask for running resolution functions of the MDP Toolbox in silent
        mode.
        """
        # self.verbose = False
        pass
    
    def span(self, W):
        """Returns the span of W
        
        sp(W) = max W(s) - min W(s)
        """
        return (W.max() - W.min())
    
    def verbose(self):
        """Ask for running resolution functions of the MDP Toolbox in verbose
        mode.
        """
        # self.verbose = True
        pass

class ExampleForest(MDP):
    """Generate a Markov Decision Process example based on a simple forest
    management.
    """
    pass

class ExampleRand(MDP):
    """Generate a random Markov Decision Process.
    """
    pass

class FiniteHorizon(MDP):
    """Resolution of finite-horizon MDP with backwards induction.
    """
    pass

class LP(MDP):
    """Resolution of discounted MDP with linear programming.
    """
    pass

class PolicyIteration(MDP):
    """Resolution of discounted MDP with policy iteration algorithm.
    """
    pass

class PolicyIterationModified(MDP):
    """Resolution of discounted MDP with modified policy iteration algorithm.
    """
    pass

class QLearning(MDP):
    """Evaluation of the matrix Q, using the Q learning algorithm.
    """
    
    def __init__(self, transitions, reward, discount, n_iter=10000):
        """Evaluation of the matrix Q, using the Q learning algorithm
        
        Arguments
        -----------------------------------------------------------------------
        Let S = number of states, A = number of actions
        transitions(SxSxA) = transition matrix 
            P could be an array with 3 dimensions or a cell array (1xA), each
            cell containing a sparse matrix (SxS)
        
        reward(SxSxA) or (SxA) = reward matrix
            R could be an array with 3 dimensions (SxSxA) or a cell array
            (1xA), each cell containing a sparse matrix (SxS) or a 2D
            array(SxA) possibly sparse
        
        discount = discount rate in ]0; 1[
        
        n_iter(optional) = number of iterations to execute.
            Default value = 10000; it is an integer greater than the default
            value.
        Evaluation
        -----------------------------------------------------------------------
        Q(SxA) = learned Q matrix 
        value(S)   = learned value function.
        policy(S) = learned optimal policy.
        mean_discrepancy(N/100) = vector of V discrepancy mean over 100 iterations
            Then the length of this vector for the default value of N is 100.
        """
        
        # Check of arguments
        if (discount <= 0) or (discount >= 1):
            raise ValueError("MDP Toolbox Error: Discount rate must be in ]0,1[")
        elif (n_iter < 10000):
            raise ValueError("MDP Toolbox Error: n_iter must be greater than 10000")
        
        is_mdp, err_msg = self.check(transitions, reward)
        if (not is_mdp):
            raise TypeError(err_msg)
        
        self.computePR(transitions, reward)
        
        self.discount = discount
        
        self.n_iter = n_iter
        
        # Initialisations
        self.Q = zeros((self.S, self.A))
        #self.dQ = zeros(self.S, self.A)
        self.mean_discrepancy = []
        self.discrepancy = zeros((self.S, 100))
        
        self.time = None
        
    def iterate(self):
        """
        """
        self.time = time()
        
        # initial state choice
        s = randint(0, self.S - 1)
        
        for n in range(self.n_iter):
            
            # Reinitialisation of trajectories every 100 transitions
            if ((n % 100) == 0):
                s = randint(1, self.S)
            
            # Action choice : greedy with increasing probability
            # probability 1-(1/log(n+2)) can be changed
            pn = random()
            if (pn < (1 - (1 / log(n + 2)))):
                optimal_action, a = self.Q[s, ].max()
            else:
                a = randint(0, self.A - 1)
              
            # Simulating next state s_new and reward associated to <s,s_new,a>
            p_s_new = random()
            p = 0
            s_new = 0
            while ((p < p_s_new) and (s_new < s)):
                s_new = s_new + 1
                if (type(self.P) is object):
                    p = p + self.P[a][s, s_new]
                else:
                    p = p + self.P[a, s, s_new]
                
            if (type(self.R) is object):
                r = self.R[a][s, s_new]
            elif (self.R.ndim == 3):
                r = self.R(a, s, s_new)
            else:
                r = self.R(s, a)
                
            # Updating the value of Q   
            # Decaying update coefficient (1/sqrt(n+2)) can be changed
            delta = r + self.discount * self.Q[s_new, ].max() - self.Q[s, a]
            dQ = (1 / sqrt(n + 2)) * delta
            self.Q[s, a] = self.Q[s, a] + dQ
            
            # current state is updated
            s = s_new
            
            # Computing and saving maximal values of the Q variation
            self.discrepancy[(n % 100) + 1, ] = abs(dQ)
            
            # Computing means all over maximal Q variations values
            if ((n % 100) == 99):
                self.mean_discrepancy.append(self.discrepancy.mean(1))
                self.discrepancy = zeros((self.S, 100))
            
            # compute the value function and the policy
            self.value = self.Q.max(axis=1)
            self.policy = self.Q.argmax(axis=1)
            
            self.time = time() - self.time

class RelativeValueIteration(MDP):
    """Resolution of MDP with average reward with relative value iteration
    algorithm.
    """
    pass

class ValueIteration(MDP):
    """
    Resolve a discounted Markov Decision Problem with value iteration.
    """
    
    def __init__(self, transitions, reward, discount, epsilon=0.01, max_iter=1000, initial_value=0, verbose=False):
        """Resolution of discounted MDP with value iteration algorithm.
        
        Arguments
        ---------
        Let S = number of states, A = number of actions.
        
        transitions = transition matrix 
            P could be a numpy ndarray with 3 dimensions (AxSxS) or a 
            numpy ndarray of dytpe=object with 1 dimenion (1xA), each 
            element containing a numpy ndarray (SxS) or scipy sparse matrix.
        
        reward = reward matrix
            R could be a numpy ndarray with 3 dimensions (AxSxS) or numpy
            ndarray of dtype=object with 1 dimension (1xA), each element
            containing a sparse matrix (SxS). R also could be a numpy 
            ndarray with 2 dimensions (SxA) possibly sparse.
        
        discount = discount rate in ]0; 1]
            Beware to check conditions of convergence for discount = 1.
        
        epsilon = epsilon-optimal policy search
            Greater than 0, optional (default: 0.01).
        
        max_iter = maximum number of iterations to be done.
            greater than 0, optional (default: computed)
        
        initial_value = starting value function.
            optional (default: zeros(S,1)).
        
        Evaluation
        ----------
        value(S) = value function.
        
        policy(S) = epsilon-optimal policy.
        
        iter = number of done iterations.
        
        time = used CPU time.
        
        Notes
        -----
        In verbose mode, at each iteration, displays the variation of V
        and the condition which stopped iterations: epsilon-optimum policy found
        or maximum number of iterations reached.
        """
        
        self.verbose = verbose
        
        is_mdp, err_msg = self.check(transitions, reward)
        if (not is_mdp):
            raise TypeError(err_msg)

        self.computePR(transitions, reward)
        
        # initialization of optional arguments
        if (initial_value == 0):
            self.value = matrix(zeros((self.S, 1)))
        else:
            if (initial_value.size != self.S):
                raise TypeError("The initial value must be length S")
            
            self.value = matrix(initial_value)
        
        self.discount = discount
        if (discount < 1):
            # compute a bound for the number of iterations
            #self.max_iter = self.boundIter(epsilon)
            self.max_iter = 5000
            # computation of threshold of variation for V for an epsilon-optimal policy
            self.thresh = epsilon * (1 - self.discount) / self.discount
        else: # discount == 1
            # bound for the number of iterations
            self.max_iter = max_iter
            # threshold of variation for V for an epsilon-optimal policy
            self.thresh = epsilon 
        
        self.itr = 0
        
        self.time = None
    
    def boundIter(self, epsilon):
        """Computes a bound for the number of iterations for the value iteration
        algorithm to find an epsilon-optimal policy with use of span for the 
        stopping criterion
        
        Arguments --------------------------------------------------------------
        Let S = number of states, A = number of actions
            epsilon   = |V - V*| < epsilon,  upper than 0,
                optional (default : 0.01)
        Evaluation -------------------------------------------------------------
            max_iter  = bound of the number of iterations for the value 
            iteration algorithm to find an epsilon-optimal policy with use of
            span for the stopping criterion
            cpu_time  = used CPU time
        """
        # See Markov Decision Processes, M. L. Puterman, 
        # Wiley-Interscience Publication, 1994 
        # p 202, Theorem 6.6.6
        # k =    max     [1 - S min[ P(j|s,a), p(j|s',a')] ]
        #     s,a,s',a'       j
        k = 0
        h = zeros(self.S)
        
        for ss in range(self.S):
            PP = zeros((self.S, self.A))
            for aa in range(self.A):
                PP[:, aa] = self.P[aa][:, ss]
            # the function "min()" without any arguments finds the
            # minimum of the entire array.
            h[ss] = PP.min()
        
        k = 1 - h.sum()
        V1 = self.bellmanOperator(self.value)
        # p 201, Proposition 6.6.5
        max_iter = log( (epsilon * (1 - self.discount) / self.discount) / self.span(V1 - self.value) ) / log(self.discount * k)
        return ceil(max_iter)
    
    def iterate(self):
        """
        """
        self.time = time()
        done = False
        while not done:
            self.itr = self.itr + 1
            
            Vprev = self.value
            
            # Bellman Operator: updates "self.value" and "self.policy"
            self.bellmanOperator()
            
            # The values, based on Q. For the function "max()": the option
            # "axis" means the axis along which to operate. In this case it
            # finds the maximum of the the rows. (Operates along the columns?)
            variation = self.span(self.value - Vprev)
            
            if self.verbose:
                print("      %s         %s" % (self.itr, variation))
            
            if variation < self.thresh:
                done = True
                if self.verbose:
                    print("...iterations stopped, epsilon-optimal policy found")
            elif (self.itr == self.max_iter):
                done = True 
                if self.verbose:
                    print("...iterations stopped by maximum number of iteration condition")

        self.value = array(self.value).reshape(self.S)
        self.policy = array(self.policy).reshape(self.S)

        self.time = time() - self.time

class ValueIterationGS(MDP):
    """Resolution of discounted MDP with value iteration Gauss-Seidel algorithm.
    """
    pass