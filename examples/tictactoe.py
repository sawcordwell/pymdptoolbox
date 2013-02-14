# -*- coding: utf-8 -*-

#import mdp

def str_base(num, base, numerals = '0123456789abcdefghijklmnopqrstuvwxyz'):
    if base < 2 or base > len(numerals):
        raise ValueError("str_base: base must be between 2 and %i" % 
                         len(numerals))
    
    if num == 0:
        return '0'
    
    if num < 0:
        sign = '-'
        num = -num
    else:
        sign = ''
    
    result = ''
    while num:
        result = numerals[num % (base)] + result
        num //= base
    
    return sign + result


class TicTacToeMDP(object):
    """"""
         
    def __init__(self):
        """"""
        self.P = {}
        for a in xrange(9):
            self.P[a] = {}
        self.R = {}
        # some board states are equal, just rotations of other states
        self.rotorder = []
        #self.rotorder.append([0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.rotorder.append([6, 3, 0, 7, 4, 1, 8, 5, 2])
        self.rotorder.append([8, 7, 6, 5, 4, 3, 2, 1, 0])
        self.rotorder.append([2, 5, 8, 1, 4, 7, 0, 3, 6])
        # The valid number of cells belonging to either the player or the
        # opponent: (player, opponent)
        self.nXO = ((0, 0),
                    (1, 1),
                    (2, 2),
                    (3, 3),
                    (4, 4),
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4))
        # The winning positions
        self.wins = ([1, 1, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 1],
                     [1, 0, 0, 1, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 1, 0, 0, 1, 0],
                     [0, 0, 1, 0, 0, 1, 0, 0, 1],
                     [1, 0, 0, 0, 1, 0, 0, 0, 1],
                     [0, 0, 1, 0, 1, 0, 1, 0, 0])
    
    def rotate(self, state):
        #rotations = []
        identity = []
        #rotations.append(state)
        identity.append(int("".join(str(x) for x in state), 3))
        for k in range(3):
            #rotations.append(tuple(state[self.rotorder[k][kk]]
            #                          for kk in xrange(9)))
            # Convert the state from base 3 number to integer.
            #identity.append(int("".join(str(x) for x in rotations[k + 1]), 3))
            identity.append(int("".join(str(state[self.rotorder[k][kk]])
                                        for kk in xrange(9)), 3))
        # return the rotation with the smallest identity number
        #idx = identity.index(min(identity))
        #return (identity[idx], rotations[idx])
        return min(identity)
    
    def unrotate(self, move, rotation):
        rotation -= 1
        # return the move
        return self.rotorder[rotation][move]
    
    def isLegal(self, state, action):
        """"""
        if state[action] == 0:
            return True
        else:
            return False
    
    def isWon(self, state, who):
        """"""
        # Check to see if there are any wins
        for w in self.wins:
            S = sum(1 if (w[k] == 1 and state[k] == who) else 0
                    for k in xrange(9))
            if S == 3:
                # We have a win
                return True
        # There were no wins so return False
        return False
    
    def isDraw(self, state):
        """"""
        try:
            state.index(0)
            return False
        except ValueError:
            return True
        except:
            raise
    
    def isValid(self, state):
        """"""
        # S1 is the sum of the player's cells
        S1 = sum(1 if x == 1 else 0 for x in state)
        # S2 is the sum of the opponent's cells
        S2 = sum(1 if x == 2 else 0 for x in state)
        if (S1, S2) in self.nXO:
            return True
        else:
            return False
        
    def run(self):
        """"""
        l = (0,1,2)
        # Iterate through a generator of all the combinations
        for s in ([a0,a1,a2,a3,a4,a5,a6,a7,a8] for a0 in l for a1 in l
                   for a2 in l for a3 in l for a4 in l for a5 in l
                   for a6 in l for a7 in l for a8 in l):
            if self.isValid(s):
                self.transition(s)
        # Convert P and R to ijv lists
        # Iterate through up to the theorectically maxmimum value of s
        for s in xrange(int('222211110',3)):
            pass
        # return (P, R)
    
    def toTuple(self, state):
        """"""
        state = str_base(state, 3)
        state = ''.join('0' for x in range(9 - len(state))) + state
        return tuple(int(x) for x in state)
        
    def transition(self, state):
        """"""
        #TODO: the state needs to be rotated before anything else is done!!!
        idn_s = int("".join(str(x) for x in state), 3)
        legal_a = [x for x in xrange(9) if state[x] == 0]
        for a in legal_a:
            s = [x for x in state]
            s[a] = 1
            is_won = self.isWon(s, 1)
            legal_m = [x for x in xrange(9) if s[x] == 0]
            for m in legal_m:
                s_new = [x for x in s]
                s_new[m] = 2
                idn_s_new = self.rotate(s_new)
                if not self.P[a].has_key((idn_s, idn_s_new)):
                    self.P[a][(idn_s, idn_s_new)] = len(legal_m)
                if not self.R.has_key((idn_s, idn_s_new)):
                    if is_won:
                        self.R[(idn_s, idn_s_new)] = 1
                    elif self.isWon(s_new, 2):
                        self.R[(idn_s, idn_s_new)] = -1
                    else:
                        self.R[(idn_s, idn_s_new)] = 0

if __name__ == "__main__":
    P, R = TicTacToeMDP().run()
    #ttt = mdp.ValueIteration(P, R, 1)