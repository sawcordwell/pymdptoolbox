# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:35:08 2013

@author: steve
"""
import mdptoolbox.example
P, R = mdptoolbox.example.forest()
vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
vi.run()
print "Value Iteration:\nV: {}\npolicy: {}".format(vi.V, vi.policy)
lp = mdptoolbox.mdp._LP(P, R, 0.9)
lp.run()
print "Linear Programming\nV: {}\npolicy: {}".format(lp.V, lp.policy)

