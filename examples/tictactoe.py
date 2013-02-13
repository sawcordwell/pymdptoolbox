# -*- coding: utf-8 -*-

import mdp

class TicTacToeMDP(object):
    """"""
         
    def __init__(self):
        """"""
        pass
    
    def isLegal(state, action):
        """"""
        if state[action] == 0:
            return True
        else:
            return False
    
    def isWon(state):
        """"""
        wins = ([1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 1, 0, 1, 0, 1, 0, 0])
        # Check to see if there are any wins
        for w in wins:
            S = sum([1 if (w[k] == 1 and state[k] == 1) else 0 for k in xrange(9)])
            if S == 3:
                # We have a win
                return True
        # There were no wins so return False
        return False
    
    def isDraw(state):
        """"""
        try:
            state.index(0)
            return False
        except ValueError:
            return True
        except:
            raise
    
    def run(self):
        """"""
        pass
     

if __name__ == "__main__":
    P, R = TicTacToeMDP().run()
    ttt = mdp.ValueIteration(P, R, 1)