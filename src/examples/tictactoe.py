# -*- coding: utf-8 -*-

import cPickle as pickle

import numpy as np
from scipy.sparse import dok_matrix

from mdptoolbox import mdp

ACTIONS = 9
STATES = 3**ACTIONS
PLAYER = 1
OPPONENT = 2
WINS = ([1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 1, 0, 0])

# The valid number of cells belonging to either the player or the opponent:
# (player, opponent)
OWNED_CELLS = ((0, 0),
               (1, 1),
               (2, 2),
               (3, 3),
               (4, 4),
               (0, 1),
               (1, 2),
               (2, 3),
               (3, 4))

def convertIndexToTuple(state):
    """"""
    return(tuple(int(x) for x in np.base_repr(state, 3, 9)[-9::]))

def convertTupleToIndex(state):
    """"""
    return(int("".join(str(x) for x in state), 3))

def getLegalActions(state):
    """"""
    return(tuple(x for x in range(ACTIONS) if state[x] == 0))

def getTransitionAndRewardArrays():
    """"""
    P = [dok_matrix((STATES, STATES)) for a in range(ACTIONS)]
    #R = spdok((STATES, ACTIONS))
    R = np.zeros((STATES, ACTIONS))
    # Naive approach, iterate through all possible combinations
    for a in range(ACTIONS):
        for s in range(STATES):
            state = convertIndexToTuple(s)
            if not isValid(state):
                # There are no defined moves from an invalid state, so
                # transition probabilities cannot be calculated. However,
                # P must be a square stochastic matrix, so assign a
                # probability of one to the invalid state transitioning
                # back to itself.
                P[a][s, s] = 1
                # Reward is 0
            else:
                s1, p, r = getTransitionProbabilities(state, a)
                P[a][s, s1] = p
                R[s, a] = r
        P[a] = P[a].tocsr()
    #R = R.tolil()
    return(P, R)

def getTransitionProbabilities(state, action):
    """
    Parameters
    ----------
    state : tuple
        The state
    action : int
        The action
    
    Returns
    -------
    s1, p, r : tuple of two lists and an int
        s1 are the next states, p are the probabilities, and r is the reward
    
    """
    #assert isValid(state)
    assert 0 <= action < ACTIONS
    if not isLegal(state, action):
        # If the action is illegal, then transition back to the same state but
        # incur a high negative reward
        s1 = [convertTupleToIndex(state)]
        return(s1, [1], -10)
    # Update the state with the action
    state = list(state)
    state[action] = PLAYER
    if isWon(state, PLAYER):
        # If the player's action is a winning move then transition to the
        # winning state and receive a reward of 1.
        s1 = [convertTupleToIndex(state)]
        return(s1, [1], 1)
    elif isDraw(state):
        s1 = [convertTupleToIndex(state)]
        return(s1, [1], 0)
    # Now we search through the opponents moves, and calculate transition
    # probabilities based on maximising the opponents chance of winning..
    s1 = []
    p = []
    legal_a = getLegalActions(state)
    for a in legal_a:
        state[a] = OPPONENT
        # If the opponent is going to win, we assume that the winning move will
        # be chosen:
        if isWon(state, OPPONENT):
            s1 = [convertTupleToIndex(state)]
            return(s1, [1], -1)
        elif isDraw(state):
            s1 = [convertTupleToIndex(state)]
            return(s1, [1], 0)
        # Otherwise we assume the opponent will select a move with uniform
        # probability across potential moves:
        s1.append(convertTupleToIndex(state))
        p.append(1.0 / len(legal_a))
        state[a] = 0
    # During non-terminal play states the reward is 0.
    return(s1, p, 0)

def getReward(state, action):
    """"""
    if not isLegal(state, action):
        return -100
    state = list(state)
    state[action] = PLAYER
    if isWon(state, PLAYER):
        return 1
    elif isWon(state, OPPONENT):
        return -1
    else:
        return 0

def isDraw(state):
    """"""
    try:
        state.index(0)
        return False
    except ValueError:
        return True

def isLegal(state, action):
    """"""
    if state[action] == 0:
        return True
    else:
        return False

def isWon(state, who):
    """Test if a tic-tac-toe game has been won.
    
    Assumes that the board is in a legal state.
    Will test if the value 1 is in any winning combination.
    
    """
    for w in WINS:
        S = sum(1 if (w[k] == 1 and state[k] == who) else 0
                for k in range(ACTIONS))
        if S == 3:
            # We have a win
            return True
    # There were no wins so return False
    return False

def isValid(state):
    """"""
    # S1 is the sum of the player's cells
    S1 = sum(1 if x == PLAYER else 0 for x in state)
    # S2 is the sum of the opponent's cells
    S2 = sum(1 if x == OPPONENT else 0 for x in state)
    if (S1, S2) in OWNED_CELLS:
        return True
    else:
        return False

if __name__ == "__main__":
    P, R = getTransitionAndRewardArrays()
    ttt = mdp.ValueIteration(P, R, 1)
    ttt.setVerbose()
    ttt.run()
    f = "tictactoe.pkl"
    pickle.dump(ttt.policy, open(f, "wb"))
    print("Optimal policy pickled as '%s' in current directory." % f)
