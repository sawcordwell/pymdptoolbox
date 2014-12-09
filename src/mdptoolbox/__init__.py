# -*- coding: utf-8 -*-
"""Markov Decision Process (MDP) Toolbox
=====================================

The MDP toolbox provides classes and functions for the resolution of
descrete-time Markov Decision Processes.

Available modules
-----------------
example
    Examples of transition and reward matrices that form valid MDPs
mdp
    Makov decision process algorithms
util
    Functions for validating and working with an MDP

How to use the documentation
----------------------------
Documentation is available both as docstrings provided with the code and
in html or pdf format from 
`The MDP toolbox homepage <http://www.somewhere.com>`_. The docstring
examples assume that the ``mdptoolbox`` package is imported like so::

  >>> import mdptoolbox

To use the built-in examples, then the ``example`` module must be imported::

  >>> import mdptoolbox.example

Once the ``example`` module has been imported, then it is no longer neccesary
to issue ``import mdptoolbox``.

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

from . import mdp
