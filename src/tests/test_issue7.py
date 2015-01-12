# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp

import mdptoolbox

class BaseTestIssue7(object):
    discount = 0.9
    P = [None] * 2
    P[0] = np.array([
        [ 0.  ,  0.  ,  0.  ,  0.64,  0.  ,  0.  ,  0.36,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.93,  0.  ,  0.  ,  0.07,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.2 ,  0.  ,  0.  ,  0.8 ],
        [ 0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  1.  ]
    ])
    P[1] = np.array([
        [ 0.  ,  0.  ,  0.4 ,  0.  ,  0.6 ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  1.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.87,  0.13,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.11,  0.89],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  1.  ]
    ])

    R = [None] * 2
    R[0] = np.zeros((9, 9))
    R[1] = np.array([
        [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]
    ])
    
    computed_R = (np.array((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                  np.array((0.6, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))

    policy = (1, 1, 0, 0, 0, 0, 0, 0, 0)

    def dense_P_dense_R(self, algorithm):
        sdp = algorithm(self.P, self.R, self.discount)
        if algorithm != mdptoolbox.mdp.QLearning:
            assert (sdp.R[0] == self.computed_R[0]).all()
            assert (sdp.R[1] == self.computed_R[1]).all()
        assert not sp.issparse(sdp.P[0])
        assert not sp.issparse(sdp.P[1])
        assert not sp.issparse(sdp.R[0])
        assert not sp.issparse(sdp.R[1])
        sdp.run()
        if algorithm != mdptoolbox.mdp.QLearning:
            assert sdp.policy == self.policy, sdp.policy

    def sparse_P_dense_R(self, algorithm):
        P = list(map(sp.csr_matrix, self.P))
        sdp = algorithm(P, self.R, self.discount)
        if algorithm != mdptoolbox.mdp.QLearning:
            assert (sdp.R[0] == self.computed_R[0]).all()
            assert (sdp.R[1] == self.computed_R[1]).all()
        assert sp.issparse(sdp.P[0])
        assert sp.issparse(sdp.P[1])
        assert not sp.issparse(sdp.R[0])
        assert not sp.issparse(sdp.R[1])
        sdp.run()
        if algorithm != mdptoolbox.mdp.QLearning:
            assert sdp.policy == self.policy, sdp.policy

    def dense_P_sparse_R(self, algorithm):
        R = list(map(sp.csr_matrix, self.R))
        sdp = algorithm(self.P, R, self.discount)
        if algorithm != mdptoolbox.mdp.QLearning:
            assert (sdp.R[0] == self.computed_R[0]).all()
            assert (sdp.R[1] == self.computed_R[1]).all()
        assert not sp.issparse(sdp.P[0])
        assert not sp.issparse(sdp.P[1])
        #assert sp.issparse(sdp.R[0])
        #assert sp.issparse(sdp.R[1])
        sdp.run()
        if algorithm != mdptoolbox.mdp.QLearning:
            assert sdp.policy == self.policy, sdp.policy

    def sparse_P_sparse_R(self, algorithm):
        P = list(map(sp.csr_matrix, self.P))
        R = list(map(sp.csr_matrix, self.R))
        sdp = algorithm(P, R, self.discount)
        if algorithm != mdptoolbox.mdp.QLearning:
            assert (sdp.R[0] == self.computed_R[0]).all()
            assert (sdp.R[1] == self.computed_R[1]).all()
        assert sp.issparse(sdp.P[0])
        assert sp.issparse(sdp.P[1])
        #assert sp.issparse(sdp.R[0])
        #assert sp.issparse(sdp.R[1])
        sdp.run()
        if algorithm != mdptoolbox.mdp.QLearning:
            assert sdp.policy == self.policy, sdp.policy

# Needs some work before can use, need to pass horizon
#class TestFiniteHorizon(BaseTestIssue7):
#
#    def test_dense_P_dense_R(self):
#        self.dense_P_dense_R(mdptoolbox.mdp.FiniteHorizon)
#
#    def test_sparse_P_dense_R(self):
#        self.sparse_P_dense_R(mdptoolbox.mdp.FiniteHorizon)
#
#    def test_dense_P_sparse_R(self):
#        self.dense_P_sparse_R(mdptoolbox.mdp.FiniteHorizon)
#
#    def test_sparse_P_sparse_R(self):
#       self.sparse_P_sparse_R(mdptoolbox.mdp.FiniteHorizon)

#class TestLP(BaseTestIssue7):
#
#    def test_dense_P_dense_R(self):
#        self.dense_P_dense_R(mdptoolbox.mdp.LP)
#
#    def test_sparse_P_dense_R(self):
#        self.sparse_P_dense_R(mdptoolbox.mdp.LP)
#
#    def test_dense_P_sparse_R(self):
#        self.dense_P_sparse_R(mdptoolbox.mdp.LP)
#
#    def test_sparse_P_sparse_R(self):
#       self.sparse_P_sparse_R(mdptoolbox.mdp.LP)

class TestPolicyIteration(BaseTestIssue7):

    def test_dense_P_dense_R(self):
        self.dense_P_dense_R(mdptoolbox.mdp.PolicyIteration)

    def test_sparse_P_dense_R(self):
        self.sparse_P_dense_R(mdptoolbox.mdp.PolicyIteration)

    def test_dense_P_sparse_R(self):
        self.dense_P_sparse_R(mdptoolbox.mdp.PolicyIteration)

    def test_sparse_P_sparse_R(self):
       self.sparse_P_sparse_R(mdptoolbox.mdp.PolicyIteration)

class TestPolicyIterationModified(BaseTestIssue7):

    def test_dense_P_dense_R(self):
        self.dense_P_dense_R(mdptoolbox.mdp.PolicyIterationModified)

    def test_sparse_P_dense_R(self):
        self.sparse_P_dense_R(mdptoolbox.mdp.PolicyIterationModified)

    def test_dense_P_sparse_R(self):
        self.dense_P_sparse_R(mdptoolbox.mdp.PolicyIterationModified)

    def test_sparse_P_sparse_R(self):
       self.sparse_P_sparse_R(mdptoolbox.mdp.PolicyIterationModified)

class TestQLearning(BaseTestIssue7):

    def test_dense_P_dense_R(self):
        self.dense_P_dense_R(mdptoolbox.mdp.QLearning)

    def test_sparse_P_dense_R(self):
        self.sparse_P_dense_R(mdptoolbox.mdp.QLearning)

    def test_dense_P_sparse_R(self):
        self.dense_P_sparse_R(mdptoolbox.mdp.QLearning)

    def test_sparse_P_sparse_R(self):
       self.sparse_P_sparse_R(mdptoolbox.mdp.QLearning)

class TestValueIteration(BaseTestIssue7):

    def test_dense_P_dense_R(self):
        self.dense_P_dense_R(mdptoolbox.mdp.ValueIteration)

    def test_sparse_P_dense_R(self):
        self.sparse_P_dense_R(mdptoolbox.mdp.ValueIteration)

    def test_dense_P_sparse_R(self):
        self.dense_P_sparse_R(mdptoolbox.mdp.ValueIteration)

    def test_sparse_P_sparse_R(self):
       self.sparse_P_sparse_R(mdptoolbox.mdp.ValueIteration)

class TestRelativeValueIteration(BaseTestIssue7):

    def test_dense_P_dense_R(self):
        self.dense_P_dense_R(mdptoolbox.mdp.RelativeValueIteration)

    def test_sparse_P_dense_R(self):
        self.sparse_P_dense_R(mdptoolbox.mdp.RelativeValueIteration)

    def test_dense_P_sparse_R(self):
        self.dense_P_sparse_R(mdptoolbox.mdp.RelativeValueIteration)

    def test_sparse_P_sparse_R(self):
       self.sparse_P_sparse_R(mdptoolbox.mdp.RelativeValueIteration)

class TestValueIterationGS(BaseTestIssue7):

    def test_dense_P_dense_R(self):
        self.dense_P_dense_R(mdptoolbox.mdp.ValueIterationGS)

    def test_sparse_P_dense_R(self):
        self.sparse_P_dense_R(mdptoolbox.mdp.ValueIterationGS)

    def test_dense_P_sparse_R(self):
        self.dense_P_sparse_R(mdptoolbox.mdp.ValueIterationGS)

    def test_sparse_P_sparse_R(self):
       self.sparse_P_sparse_R(mdptoolbox.mdp.ValueIterationGS)
