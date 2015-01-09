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

# Copyright (c) 2011-2015 Steven A. W. Cordwell
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

from mdptoolbox.finite_horizon import FiniteHorizon
from mdptoolbox.linear_programming import LP
from mdptoolbox.mdp_base import MDP
from mdptoolbox.policy_iteration import PolicyIteration 
from mdptoolbox.policy_iteration_modified import PolicyIterationModified
from mdptoolbox.q_learning import QLearning
from mdptoolbox.relative_value_iteration import RelativeValueIteration
from mdptoolbox.value_iteration import ValueIteration
from mdptoolbox.value_iteration_gs import ValueIterationGS
