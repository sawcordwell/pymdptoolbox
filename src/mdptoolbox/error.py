# -*- coding: utf-8 -*-
"""Markov Decision Process (MDP) Toolbox: ``error`` module
=======================================================

The ``error`` module provides exception classes that can be raised by
the toolbox.

Available classes
-----------------
Error
    Base exception class derived from ``Exception``
InvalidError
    Exception for invalid definitions of an MDP
NonNegativeError
    Exception for transition matrices that have negative elements
SquareError
    Exception for transition matrices that are not square
StochasticError
    Exception for transition matrices that are not stochastic

"""

# Copyright (c) 2015 Steven A. W. Cordwell
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

class Error(Exception):
    """Base class for exceptions in this module."""

    def __init__(self):
        Exception.__init__(self)
        self.message = "PyMDPToolbox - "

    def __str__(self):
        return repr(self.message)

class InvalidError(Error):
    """Class for invalid definitions of a MDP."""

    def __init__(self, msg):
        Error.__init__(self)
        self.message += msg
        self.args = tuple(msg)

class NonNegativeError(Error):
    """Class for transition matrix stochastic errors"""

    default_msg = "The transition probability matrix is negative."

    def __init__(self, msg=None):
        if msg is None:
            msg = self.default_msg
        Error.__init__(self)
        self.message += msg
        self.args = tuple(msg)

class SquareError(Error):
    """Class for transition matrix square errors"""

    default_msg = "The transition probability matrix is not square."

    def __init__(self, msg=None):
        if msg is None:
            msg = self.default_msg
        Error.__init__(self)
        self.message += msg
        self.args = tuple(msg)

class StochasticError(Error):
    """Class for transition matrix stochastic errors"""

    default_msg = "The transition probability matrix is not stochastic."

    def __init__(self, msg=None):
        if msg is None:
            msg = self.default_msg
        Error.__init__(self)
        self.message += msg
        self.args = tuple(msg)
