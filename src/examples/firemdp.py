# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 17:20:30 2014

@author: Steven A W Cordwell
"""

from mdptoolbox import mdp

import random

import numpy as np

NUMBER_OF_POPULATION_CLASSES = 7
NUMBER_OF_FIRE_CLASSES = 13
STATES = NUMBER_OF_POPULATION_CLASSES * NUMBER_OF_FIRE_CLASSES
ACTIONS = 2

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
    assert 0 <= population < NUMBER_OF_POPULATION_CLASSES, "'population' " \
        "must be in (0, 1...%s)" % str(NUMBER_OF_POPULATION_CLASSES - 1)
    assert 0 <= fire < NUMBER_OF_FIRE_CLASSES, "'fire' must be in " \
        "(0, 1...%s) " % str(NUMBER_OF_FIRE_CLASSES - 1)
    return(population * NUMBER_OF_FIRE_CLASSES + fire)

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
    assert index < NUMBER_OF_POPULATION_CLASSES * NUMBER_OF_FIRE_CLASSES
    population = index // NUMBER_OF_FIRE_CLASSES
    fire = index % NUMBER_OF_FIRE_CLASSES
    return(population, fire)

def getHabitatSuitability(time):
    """The habitat suitability of a patch relatve to the time since last fire.
    
    The habitat quality is low immediately after a fire, rises rapidly until
    five years after a fire, and declines once the habitat is mature. See
    Figure 2 in Possingham and Tuck (1997) for more details.
    
    Parameters
    ----------
    time : int
        Time in years since last fire.
    
    Returns
    -------
    r : float
        The habitat suitability.
    
    """
    assert time >= 0, "'time' must be a positive number"
    if time <= 5:
        return(0.2 * time)
    elif 5 <= time <= 10:
        return(-0.1 * time + 1.5)
    else:
        return(0.5)

def getTransitionAndRewardMatrices(s=0.5):
    P = np.zeros((ACTIONS, STATES, STATES))
    R = np.zeros((STATES,))
    for idx in range(STATES):
        x, F = convertIndexToState(idx)
        if x != 0:
            R[idx] = 1
        for a in range(ACTIONS):
            P[a][idx] = getTransitionProbabilities(x, F, s, a)
    return(P, R)

def getTransitionProbabilities(x, F, s, action):
    tp = np.zeros((STATES,))
    r = getHabitatSuitability(F)
    # Efect of action on time in years since fire.
    if action == 0:
        # Increase the time since the patch has been burned by one year.
        # The years since fire in patch is absorbed into the last class
        if F < NUMBER_OF_FIRE_CLASSES - 1:
            F += 1
    elif action == 1:
        # When the patch is burned set the time since fire to 0.
        F = 0
    # Population transitions
    if x == 0:
        # population abundance class stays at 0 (extinct)
        # Demographic model probabilities
        idx = convertStateToIndex(0, F)
        tp[idx] = 1
    elif x == NUMBER_OF_POPULATION_CLASSES - 1:
        # Population abundance class either stays at maximum or transitions
        # down
        transition_a = x
        transition_b = x - 1
        # Effect of action on the state
        # If action 1 is taken, then the patch is burned so the population
        # abundance moves down a class.
        if action == 1:
            transition_a -= 1
            transition_b -= 1
        # Demographic model probabilities
        idx = convertStateToIndex(transition_a, F)
        tp[idx] = 1 - (1 - s) * (1 - r)
        idx = convertStateToIndex(transition_b, F)
        tp[idx] = (1 - s) * (1 - r)
    else:
        # Population abundance class can stay the same, transition up, or
        # transition down.
        transition_a = x
        transition_b = x + 1
        transition_c = x - 1
        # Effect of action on the state
        # If action 1 is taken, then the patch is burned so the population
        # abundance moves down a class.
        if action == 1:
            transition_a -= 1
            transition_b -= 1 
            if transition_c > 0:
                transition_c -= 1
        # Demographic model probabilities
        idx = convertStateToIndex(transition_a, F)
        tp[idx] = s
        idx = convertStateToIndex(transition_b, F)
        tp[idx] = (1 - s) * r
        idx = convertStateToIndex(transition_c, F)
        tp[idx] += (1 - s) * (1 - r)
    return(tp)

def simulateTransition(state, s, r, fire):
    """Simulate a state transition.
    
    Parameters
    ----------
    state : int
        The current abundance class of the threatened species.
    s : float
        The state-independent probability of the population staying in its
        current state.
    r : float
        The probability the population moves up one state, assuming it is not
        staying in its current state. ``r`` depends on ``F``, the time in years
        since the last fire.
    fire : boolean
        True if there has been a fire in the current year, otherwise False.
    
    Returns
    -------
    state : int
        The new abundance class of the threatened species.
    
    """
    assert 0 <= state <= 6, "'state' must be in (0, 1...6)"
    assert 0 <= s <= 1, "'s' must be in [0; 1]"
    assert 0 <= r <= 1, "'r' must be in [0; 1]"
    assert fire in (True, False), "'fire' must be a boolean value"
    state = int(state)
    if state == 0:
        pass
    elif state == 6:
        if random.random() <= 1 - (1 - s) * (1 - r):
            pass
        else: # with probability (1 - s)(1 - r)
            state = 5
    else:
        if random.random() <= s:
            pass
        else: 
            if random.random() <= r: # with probability (1 - s)r
                state += 1
            else: # with probability (1 - s)(1 - r)
                state -= 1
    # Add the effect of a fire
    if fire:
        state -= 1
    # Make sure state didn't transition to -1
    if state < 0:
        state = 0
    return(state)

def solveMDP():
    P, R = getTransitionAndRewardMatrices()
    sdp = mdp.FiniteHorizon(P, R, 0.9, 50)
    sdp.run()
    return(sdp)

def tests():
    assert getHabitatSuitability(0) == 0
    assert getHabitatSuitability(2) == 0.4
    assert getHabitatSuitability(5) == 1
    assert getHabitatSuitability(8) == 0.7
    assert getHabitatSuitability(10) == 0.5
    assert getHabitatSuitability(15) == 0.5
    assert convertIndexToState(90) == (NUMBER_OF_POPULATION_CLASSES - 1, 
                                       NUMBER_OF_FIRE_CLASSES - 1)
    assert convertIndexToState(89) == (NUMBER_OF_POPULATION_CLASSES -1, 
                                       NUMBER_OF_FIRE_CLASSES - 2)
    assert convertIndexToState(0) == (0, 0)

if __name__ == "__main__":
    tests()
