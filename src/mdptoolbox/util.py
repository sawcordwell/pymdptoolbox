# -*- coding: utf-8 -*-
"""Markov Decision Process (MDP) Toolbox: ``utils`` module
=======================================================

The ``utils`` module provides functions to check that an MDP is validly
described. There are also functions for working with MDPs while they are being
solved.

Available functions
-------------------
check
    Check that an MDP is properly defined
checkSquareStochastic
    Check that a matrix is square and stochastic
getSpan
    Calculate the span of an array

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

import numpy as _np

# These need to be fixed so that we use classes derived from Error.
mdperr = {
"mat_nonneg" :
    "Transition probabilities must be non-negative.",
"mat_square" :
    "A transition probability matrix must be square, with dimensions S×S.",
"mat_stoch" :
    "Each row of a transition probability matrix must sum to one (1).",
"obj_shape" :
    "Object arrays for transition probabilities and rewards "
    "must have only 1 dimension: the number of actions A. Each element of "
    "the object array contains an SxS ndarray or matrix.",
"obj_square" :
    "Each element of an object array for transition "
    "probabilities and rewards must contain an SxS ndarray or matrix; i.e. "
    "P[a].shape = (S, S) or R[a].shape = (S, S).",
"P_type" :
    "The transition probabilities must be in a numpy array; "
    "i.e. type(P) is ndarray.",
"P_shape" :
    "The transition probability array must have the shape "
    "(A, S, S)  with S : number of states greater than 0 and A : number of "
    "actions greater than 0. i.e. R.shape = (A, S, S)",
"PR_incompat" :
    "Incompatibility between P and R dimensions.",
"R_type" :
    "The rewards must be in a numpy array; i.e. type(R) is "
    "ndarray, or numpy matrix; i.e. type(R) is matrix.",
"R_shape" :
    "The reward matrix R must be an array of shape (A, S, S) or "
    "(S, A) with S : number of states greater than 0 and A : number of "
    "actions greater than 0. i.e. R.shape = (S, A) or (A, S, S)."
}

def check(P, R):
    """Check if ``P`` and ``R`` define a valid Markov Decision Process (MDP).
    
    Let ``S`` = number of states, ``A`` = number of actions.
    
    Parameters
    ---------
    P : array
        The transition matrices. It can be a three dimensional array with
        a shape of (A, S, S). It can also be a one dimensional arraye with
        a shape of (A, ), where each element contains a matrix of shape (S, S)
        which can possibly be sparse.
    R : array
        The reward matrix. It can be a three dimensional array with a
        shape of (S, A, A). It can also be a one dimensional array with a
        shape of (A, ), where each element contains matrix with a shape of
        (S, S) which can possibly be sparse. It can also be an array with
        a shape of (S, A) which can possibly be sparse.  
    
    Notes
    -----
    Raises an error if ``P`` and ``R`` do not define a MDP.
    
    Examples
    --------
    >>> import mdptoolbox, mdptoolbox.example
    >>> P_valid, R_valid = mdptoolbox.example.rand(100, 5)
    >>> mdptoolbox.util.check(P_valid, R_valid) # Nothing should happen
    >>> 
    >>> import numpy as np
    >>> P_invalid = np.random.rand(5, 100, 100)
    >>> mdptoolbox.util.check(P_invalid, R_valid) # Raises an exception
    Traceback (most recent call last):
    ...
    AssertionError: Each row of a transition probability matrix must sum to one (1).

    """
    # Checking P
    try:
        if P.ndim == 3:
            aP, sP0, sP1 = P.shape
        elif P.ndim == 1:
            # A hack so that we can go into the next try-except statement and
            # continue checking from there
            raise AttributeError
        else:
            raise InvalidMDPError(mdperr["P_shape"])
    except AttributeError:
        try:
            aP = len(P)
            sP0, sP1 = P[0].shape
            for aa in range(1, aP):
                sP0aa, sP1aa = P[aa].shape
                if (sP0aa != sP0) or (sP1aa != sP1):
                    raise InvalidMDPError(mdperr["obj_square"])
        except AttributeError:
            raise InvalidMDPError(mdperr["P_shape"])
    # Checking R
    try:
        ndimR = R.ndim
        if ndimR == 1:
            # A hack so that we can go into the next try-except statement
            raise AttributeError
        elif ndimR == 2:
            sR0, aR = R.shape
            sR1 = sR0
        elif ndimR == 3:
            aR, sR0, sR1 = R.shape
        else:
            raise InvalidMDPError(mdperr["R_shape"])
    except AttributeError:
        try:
            lenR = len(R)
            if lenR == aP:
                aR = lenR
                sR0, sR1 = R[0].shape
                for aa in range(1, aR):
                    sR0aa, sR1aa = R[aa].shape
                    if ((sR0aa != sR0) or (sR1aa != sR1)):
                        raise InvalidMDPError(mdperr["obj_square"])
            elif lenR == sP0:
                aR = aP
                sR0 = sR1 = lenR
            else:
                raise InvalidMDPError(mdperr["R_shape"])
        except AttributeError:
            raise InvalidMDPError(mdperr["R_shape"])
    # Checking dimensions
    assert sP0 > 0, "The number of states in P must be greater than 0."
    assert aP > 0, "The number of actions in P must be greater than 0."
    assert sP0 == sP1, "The matrix P must be square with respect to states."
    assert sR0 > 0, "The number of states in R must be greater than 0."
    assert aR > 0, "The number of actions in R must be greater than 0."
    assert sR0 == sR1, "The matrix R must be square with respect to states."
    assert sP0 == sR0, "The number of states must agree in P and R."
    assert aP == aR, "The number of actions must agree in P and R."
    # Check that the P's are square and stochastic
    for aa in range(aP):
        checkSquareStochastic(P[aa])
    # We are at the end of the checks, so if no exceptions have been raised
    # then that means there are (hopefullly) no errors and we return None
    return None
    
    # These are the old code comments, which need to be converted to
    # information in the docstring:
    #
    # tranitions must be a numpy array either an AxSxS ndarray (with any 
    # dtype other than "object"); or, a 1xA ndarray with a "object" dtype, 
    # and each element containing an SxS array. An AxSxS array will be
    # be converted to an object array. A numpy object array is similar to a
    # MATLAB cell array.
    #
    # NumPy has an array type of 'object', which is roughly equivalent to
    # the MATLAB cell array. These are most useful for storing sparse
    # matrices as these can only have two dimensions whereas we want to be
    # able to store a transition matrix for each action. If the dytpe of
    # the transition probability array is object then we store this as
    # P_is_object = True.
    # If it is an object array, then it should only have one dimension
    # otherwise fail with a message expalining why.
    # If it is a normal array then the number of dimensions must be exactly
    # three, otherwise fail with a message explaining why.
    #
    # As above but for the reward array. A difference is that the reward
    # array can have either two or 3 dimensions.
    #
    # We want to make sure that the transition probability array and the 
    # reward array are in agreement. This means that both should show that
    # there are the same number of actions and the same number of states.
    # Furthermore the probability of transition matrices must be SxS in
    # shape, so we check for that also.
    #
        # If the user has put their transition matrices into a numpy array
        # with dtype of 'object', then it is possible that they have made a
        # mistake and not all of the matrices are of the same shape. So,
        # here we record the number of actions and states that the first
        # matrix in element zero of the object array says it has. After
        # that we check that every other matrix also reports the same
        # number of actions and states, otherwise fail with an error.
        # aP: the number of actions in the transition array. This
        # corresponds to the number of elements in the object array.
        #
        # sP0: the number of states as reported by the number of rows of
        # the transition matrix
        # sP1: the number of states as reported by the number of columns of
        # the transition matrix
        #
        # Now we check to see that every element of the object array holds
        # a matrix of the same shape, otherwise fail.
        #
            # sp0aa and sp1aa represents the number of states in each
            # subsequent element of the object array. If it doesn't match
            # what was found in the first element, then we need to fail
            # telling the user what needs to be fixed.
            #
        # if we are using a normal array for this, then the first
        # dimension should be the number of actions, and the second and 
        # third should be the number of states
        #
    # the first dimension of the transition matrix must report the same
    # number of states as the second dimension. If not then we are not
    # dealing with a square matrix and it is not a valid transition
    # probability. Also, if the number of actions is less than one, or the
    # number of states is less than one, then it also is not a valid
    # transition probability.
    #
    # now we check that each transition matrix is square-stochastic. For
    # object arrays this is the matrix held in each element, but for
    # normal arrays this is a matrix formed by taking a slice of the array
    #
        # if the rewarad array has an object dtype, then we check that
        # each element contains a matrix of the same shape as we did 
        # above with the transition array.
        #
        # This indicates that the reward matrices are constructed per 
        # transition, so that the first dimension is the actions and
        # the second two dimensions are the states.
        #
        # then the reward matrix is per state, so the first dimension is 
        # the states and the second dimension is the actions.
        #
        # this is added just so that the next check doesn't error out
        # saying that sR1 doesn't exist
        #
    # the number of actions must be more than zero, the number of states
    # must also be more than 0, and the states must agree
    #
    # now we check to see that what the transition array is reporting and
    # what the reward arrar is reporting agree as to the number of actions
    # and states. If not then fail explaining the situation

def rowsSumToOne(Z, n):
    return((_np.abs(Z.sum(axis=1) - _np.ones(n))).max() <=
           10 * _np.spacing(_np.float64(1)))

def checkSquareStochastic(Z):
    """Check if Z is a square stochastic matrix.
    
    Let S = number of states.
    
    Parameters
    ----------
    Z : matrix
        This should be a two dimensional array with a shape of (S, S). It can
        possibly be sparse.
    
    Notes 
    ----------
    Returns None if no error has been detected, else it raises an error.
    
    """
    # try to get the shape of the matrix
    try:
        s1, s2 = Z.shape
    except AttributeError:
        raise TypeError("Matrix should be a numpy type.")
    except ValueError:
        raise InvalidMDPError(mdperr["mat_square"])
    # check that the matrix is square, and that each row sums to one
    assert s1 == s2, mdperr["mat_square"]
    assert rowsSumToOne(Z, s2), mdperr["mat_stoch"]
    # make sure that there are no values less than zero
    try:
        assert (Z >= 0).all(), mdperr["mat_nonneg"]
    except (NotImplementedError, AttributeError, TypeError):
        try:
            assert (Z.data >= 0).all(), mdperr["mat_nonneg"]
        except AttributeError:
            raise TypeError("Matrix should be a numpy type.")
    
    return(None)

def getSpan(W):
    """Return the span of W
    
    sp(W) = max W(s) - min W(s)
    
    """
    return (W.max() - W.min())

class Error(Exception):
    """Base class for exceptions in this module."""
    
    def __init__(self):
        Exception.__init__(self)
        self.message = "PyMDPToolbox: "
    
    def __str__(self):
        return repr(self.message)

class InvalidMDPError(Error):
    """Class for invalid definitions of a MDP."""
    
    def __init__(self, msg):
        Error.__init__(self)
        self.message += msg
        self.args = tuple(msg)
