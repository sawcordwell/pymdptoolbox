# -*- coding: utf-8 -*-
"""Optimal fire management of a spatially structured threatened species
====================================================================

This PyMDPtoolbox example is based on a paper [Possingham1997]_ preseneted by
Hugh Possingham and Geoff Tuck at the 1997 MODSIM conference. The paper is 
freely available to read from the link provided, so minimal details are given
here.

.. [Possingham1997] Possingham H & Tuck G, 1997, ‘Application of stochastic
   dynamic programming to optimal fire management of a spatially structured
   threatened species’, *MODSIM 1997*, vol. 2, pp. 813–817. `Available online
   <http://www.mssanz.org.au/MODSIM97/Vol%202/Possingham.pdf>`_.

"""

# Copyright (c) 2014 Steven A. W. Cordwell
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

from mdptoolbox import mdp

import random

import numpy as np

# The number of population abundance classes
POPULATION_CLASSES = 7
# The number of years since a fire classes
FIRE_CLASSES = 13
# The number of states
STATES = POPULATION_CLASSES * FIRE_CLASSES
# The number of actions
ACTIONS = 4

def convertStateToIndex(population, fire):
    """Convert state parameters to transition probability matrix index.
    
    Parameters
    ----------
    population : int
        The population abundance class of the threatened species.
    fire : int
        The time in years since last fire.
    
    Returns
    -------
    index : int
        The index into the transition probability matrix that corresponds to
        the state parameters.
    
    """
    assert 0 <= population < POPULATION_CLASSES, "'population' must be in " \
        "(0, 1...%s)" % str(POPULATION_CLASSES - 1)
    assert 0 <= fire < FIRE_CLASSES, "'fire' must be in " \
        "(0, 1...%s) " % str(FIRE_CLASSES - 1)
    return(population * FIRE_CLASSES + fire)

def convertIndexToState(index):
    """Convert transition probability matrix index to state parameters.
    
    Parameters
    ----------
    index : int
        The index into the transition probability matrix that corresponds to
        the state parameters.
    
    Returns
    -------
    population, fire : tuple of int
        ``population``, the population abundance class of the threatened
        species. ``fire``, the time in years since last fire.
    
    """
    assert index < STATES
    population = index // FIRE_CLASSES
    fire = index % FIRE_CLASSES
    return(population, fire)

def getHabitatSuitability(years):
    """The habitat suitability of a patch relatve to the time since last fire.
    
    The habitat quality is low immediately after a fire, rises rapidly until
    five years after a fire, and declines once the habitat is mature. See
    Figure 2 in Possingham and Tuck (1997) for more details.
    
    Parameters
    ----------
    years : int
        Years since last fire.
    
    Returns
    -------
    r : float
        The habitat suitability.
    
    """
    assert years >= 0, "'years' must be a positive number"
    if years <= 5:
        return(0.2 * years)
    elif 5 <= years <= 10:
        return(-0.1 * years + 1.5)
    else:
        return(0.5)

def getTransitionProbabilities(s, x, F, a):
    """Calculate the transition probabilities for the given state and action.
    
    Parameters
    ----------
    s : float
        The probability of a population remaining in its current abundance
        class
    x : int
        The population abundance class
    F : int
        The number of years since a fire
    a : int
        The action to be performed
    
    Returns
    -------
    prob : array
        The transition probabilities as a vector from state (x, F) to every
        other state given action ``a`` is performed.
    
    """
    assert 0 <= x < POPULATION_CLASSES
    assert 0 <= F < FIRE_CLASSES
    assert 0 <= s <= 1
    assert 0 <= a < ACTIONS 
    prob = np.zeros((STATES,))
    r = getHabitatSuitability(F)
    # Efect of action on time in years since fire.
    if a == 0:
        # Increase the time since the patch has been burned by one year.
        # The years since fire in patch is absorbed into the last class
        if F < FIRE_CLASSES - 1:
            F += 1
    elif a == 1:
        # When the patch is burned set the years since fire to 0.
        F = 0
    elif a == 2:
        pass
    elif a == 3:
        pass
    # Population transitions
    if x == 0:
        # Demographic model probabilities
        # population abundance class stays at 0 (extinct)
        new_state = convertStateToIndex(0, F)
        prob[new_state] = 1
    elif x == POPULATION_CLASSES - 1:
        # Population abundance class either stays at maximum or transitions
        # down
        x_1 = x
        x_2 = x - 1
        # Effect of action on the state
        # If action 1 is taken, then the patch is burned so the population
        # abundance moves down a class.
        if a == 1:
            x_1 -= 1
            x_2 -= 1
        elif a == 2:
            pass
        elif a == 3:
            pass
        # Demographic model probabilities
        new_state = convertStateToIndex(x_1, F)
        prob[new_state] = 1 - (1 - s) * (1 - r) # abundance stays the same
        new_state = convertStateToIndex(x_2, F)
        prob[new_state] = (1 - s) * (1 - r) # abundance goes down
    else:
        # Population abundance class can stay the same, transition up, or
        # transition down.
        x_1 = x
        x_2 = x + 1
        x_3 = x - 1
        # Effect of action on the state
        # If action 1 is taken, then the patch is burned so the population
        # abundance moves down a class.
        if a == 1:
            x_1 -= 1
            x_2 -= 1
            # Ensure that the abundance class doesn't go to -1
            if x_3 > 0:
                x_3 -= 1
        elif a == 2:
            pass
        elif a == 3:
            pass
        # Demographic model probabilities
        new_state = convertStateToIndex(x_1, F)
        prob[new_state] = s # abundance stays the same
        new_state = convertStateToIndex(x_2, F)
        prob[new_state] = (1 - s) * r # abundance goes up
        new_state = convertStateToIndex(x_3, F)
        # In the case when x_3 = 0 before the effect of an action is applied,
        # then the final state is going to be the same as that for x_1, so we
        # need to add the probabilities together.
        prob[new_state] += (1 - s) * (1 - r) # abundance goes down
    return(prob)

def getTransitionAndRewardArrays(s):
    """Generate the fire management transition and reward matrices.
    
    The output arrays from this function are valid input to the mdptoolbox.mdp
    classes.
    
    Let ``S`` = number of states, and ``A`` = number of actions.
    
    Parameters
    ----------
    s : float
        The class-independent probability of the population staying in its
        current population abundance class.
    
    Returns
    -------
    out : tuple
        ``out[0]`` contains the transition probability matrices P and
        ``out[1]`` contains the reward vector R. P is an  ``A`` × ``S`` × ``S``
        numpy array and R is a numpy vector of length ``S``.
    
    """
    assert 0 <= s <= 1, "'s' must be between 0 and 1"
    # The transition probability array
    P = np.zeros((ACTIONS, STATES, STATES))
    # The reward vector
    R = np.zeros(STATES)
    # Loop over all states
    for idx in range(STATES):
        # Get the state index as inputs to our functions
        x, F = convertIndexToState(idx)
        # The reward for being in this state is 1 if the population is extant
        if x != 0:
            R[idx] = 1
        # Loop over all actions
        for a in range(ACTIONS):
            # Assign the transition probabilities for this state, action pair
            P[a][idx] = getTransitionProbabilities(s, x, F, a)
    return(P, R)

def solveMDP():
    """Solve the problem as a finite horizon Markov decision process.
    
    The optimal policy at each stage is found using backwards induction.
    Possingham and Tuck report strategies for a 50 year time horizon, so the
    number of stages for the finite horizon algorithm is set to 50. There is no
    discount factor reported, so we set it to 0.96 rather arbitrarily.
    
    Returns
    -------
    mdp : mdptoolbox.mdp.FiniteHorizon
        The PyMDPtoolbox object that represents a finite horizon MDP. The
        optimal policy for each stage is accessed with mdp.policy, which is a
        numpy array with 50 columns (one for each stage).
    
    """
    P, R = getTransitionAndRewardArrays(0.5)
    sdp = mdp.FiniteHorizon(P, R, 0.96, 50)
    sdp.run()
    return(sdp)

def printPolicy(policy):
    """Print out a policy vector as a table to console
    
    Let ``S`` = number of states.
    
    The output is a table that has the population class as rows, and the years
    since a fire as the columns. The items in the table are the optimal action
    for that population class and years since fire combination.
    
    Parameters
    ----------
    p : array
        ``p`` is a numpy array of length ``S``.
    
    """
    p = np.array(policy).reshape(POPULATION_CLASSES, FIRE_CLASSES)
    range_F = range(FIRE_CLASSES)
    print("    " + " ".join("%2d" % f for f in range_F))
    print("    " + "---" * FIRE_CLASSES)
    for x in range(POPULATION_CLASSES):
        print(" %2d|" % x + " ".join("%2d" % p[x, f] for f in range_F))

def simulateTransition(x, s, r, fire):
    """Simulate a state transition.
    
    Parameters
    ----------
    x : int
        The current abundance class of the threatened species.
    s : float
        The state-independent probability of the population staying in its
        current abundance class.
    r : float
        The probability the population moves up one abundance class, assuming
        it is not staying in its current state. ``r`` depends on ``F``, the
        time in years since the last fire.
    fire : bool
        True if there has been a fire in the current year, otherwise False.
    
    Returns
    -------
    x : int
        The new abundance class of the threatened species.
    
    """
    assert 0 <= x < POPULATION_CLASSES, "'x' must be in " \
        "{0, 1...%s}" % POPULATION_CLASSES - 1
    assert 0 <= s <= 1, "'s' must be in [0; 1]"
    assert 0 <= r <= 1, "'r' must be in [0; 1]"
    assert fire in (True, False), "'fire' must be a boolean value"
    x = int(x)
    if x == 0:
        pass
    elif x == POPULATION_CLASSES - 1:
        if random.random() <= 1 - (1 - s) * (1 - r):
            pass
        else: # with probability (1 - s)(1 - r)
            x -= 1
    else:
        if random.random() <= s:
            pass
        else:
            if random.random() <= r: # with probability (1 - s)r
                x += 1
            else: # with probability (1 - s)(1 - r)
                x -= 1
    # Add the effect of a fire, making sure x doesn't go to -1
    if fire and (x > 0):
        x -= 1
    return(x)

def _runTests():
    #Run tests on the modules functions.
    assert getHabitatSuitability(0) == 0
    assert getHabitatSuitability(2) == 0.4
    assert getHabitatSuitability(5) == 1
    assert getHabitatSuitability(8) == 0.7
    assert getHabitatSuitability(10) == 0.5
    assert getHabitatSuitability(15) == 0.5
    assert convertIndexToState(STATES-1) == (POPULATION_CLASSES - 1, 
                                       FIRE_CLASSES - 1)
    assert convertIndexToState(STATES-2) == (POPULATION_CLASSES -1, 
                                       FIRE_CLASSES - 2)
    assert convertIndexToState(0) == (0, 0)
    for idx in range(STATES):
        s1, s2 = convertIndexToState(idx)
        assert convertStateToIndex(s1, s2) == idx
    print("Tests complete.")

if __name__ == "__main__":
    import sys
    try:
        argv = sys.argv[1]
    except IndexError:
        argv = None
    if argv == "test":
        _runTests()
    else:
        sdp = solveMDP()
        printPolicy(sdp.policy[:, 0])
