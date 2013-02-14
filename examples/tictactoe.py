# -*- coding: utf-8 -*-

#import mdp

class TicTacToeMDP(object):
    """"""
         
    def __init__(self):
        """"""
        self.P = {}
        self.R = {}
        # some board states are equal, just rotations of other states
        self.rotorder = []
        #self.rotorder.append([0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.rotorder.append([6, 3, 0, 7, 4, 1, 8, 5, 2])
        self.rotorder.append([8, 7, 6, 5, 4, 3, 2, 1, 0])
        self.rotorder.append([2, 5, 8, 1, 4, 7, 0, 3, 6])
    
    def rotate(self, state):
        rotations = []
        identity = []
        rotations.append(state)
        identity.append(int("".join(str(x) for x in state), 3))
        for k in range(3):
            rotations.append(tuple([state[self.rotorder[k][kk]]
                                      for kk in xrange(9)]))
            # Convert the state from base 3 number to integer.
            identity.append(int("".join(str(x) for x in rotations[k + 1]), 3))
        # return the rotation with the smallest identity number
        idx = identity.index(min(identity))
        return (identity[idx], rotations[idx])
    
    def unrotate(self, move, rotation):
        rotation -= 1
        # return the move
        return self.rotorder[rotation][move]
    
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
            S = sum([1 if (w[k] == 1 and state[k] == 1) else 0
                     for k in xrange(9)])
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
        nXO = ((0, 0),
               (1, 1),
               (2, 2),
               (3, 3),
               (4, 4),
               (0, 1),
               (1, 2),
               (2, 3),
               (3, 4))
        #
        # return (P, R)
    
    def transition(self, s):
        """"""
        idn_s = int("".join(str(x) for x in s), 3)
        legal_a = [a for a in xrange(9) if s[a] == 0]
        for a1 in legal_a:
            s[a1] = 1
            legal_m = [a for a in xrange(9) if s[a] == 0]
            for m1 in legal_m:
                s_new = s
                s_new[m1] = 2
                idn_s_new, s_new = self.rotate(s_new)
                if self.P.has_key((idn_s, idn_s_new)):
                    raise Exception("unexpected, P already has Pr(s,s')")
                else:
                    self.P[(idn_s, idn_s_new)] = 1 / len(legal_m)

if __name__ == "__main__":
    P, R = TicTacToeMDP().run()
    ttt = mdp.ValueIteration(P, R, 1)