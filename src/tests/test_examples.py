# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 14:55:05 2013

@author: steve
"""

from nose.tools import assert_equal, assert_is_none, assert_true, \
    assert_raises
import numpy as np
import scipy.sparse as sp

import mdptoolbox.example

def assert_equal_numpy_spacing(A, B):
    A = np.array(A)
    B = np.array(B)
    x = np.amax(np.maximum(np.abs(A), np.abs(B)))
    assert_true((np.abs(A - B) <= np.spacing(x)).all())

## example.forest

class TestExampleForest(object):
    P = np.array(
        [[[0.1, 0.9, 0.0],
          [0.1, 0.0, 0.9],
          [0.1, 0.0, 0.9]],
         [[1, 0, 0],
          [1, 0, 0],
          [1, 0, 0]]])
    R = np.array(
        [[0, 0],
         [0, 1],
         [4, 2]])

    def test_dense_PR(self):
        P, R = mdptoolbox.example.forest()
        assert_equal(P.shape, self.P.shape)
        assert_equal(R.shape, self.R.shape)
        assert_equal_numpy_spacing(P, self.P)
        assert_equal_numpy_spacing(R, self.R)

    def test_sparse_PR(self):
        P, R = mdptoolbox.example.forest(is_sparse=True)
        assert_equal(len(P), len(self.P))
        for a in range(len(self.P)):
            assert_equal(P[a].shape, self.P[a].shape)
            try:
                assert_equal((P[a] != sp.csr_matrix(self.P[a])).nnz, 0)
            except AttributeError:
                assert_true((P[a].todense() == self.P[a]).all())
        assert_true((R == self.R).all())
        assert_equal(R.shape, self.R.shape)

def test_example_forest_dense_check():
    P, R = mdptoolbox.example.forest(10, 5, 3, 0.2)
    assert_is_none(mdptoolbox.util.check(P, R))

def test_example_forest_sparse_check():
    P, R = mdptoolbox.example.forest(S=30, is_sparse=True)
    assert_is_none(mdptoolbox.util.check(P, R))

def test_example_forest_S_raise():
    assert_raises(AssertionError, mdptoolbox.example.forest, S=0)

def test_example_forest_r1_raise():
    assert_raises(AssertionError, mdptoolbox.example.forest, r1=0)

def test_example_forest_r1_raise():
    assert_raises(AssertionError, mdptoolbox.example.forest, r2=0)

def test_example_forest_p_low_raise():
    assert_raises(AssertionError, mdptoolbox.example.forest, p=-1)

def test_example_forest_p_high_raise():
    assert_raises(AssertionError, mdptoolbox.example.forest, p=1.1)

## example.rand

class TestExampleRand(object):
    S = 3
    A = 2
    P = np.array(
        [[[0.28109922699468015, 0.4285572503528079, 0.2903435226525119],
          [0.0, 1.0, 0.0],
          [0.0, 1.0, 0.0]],
        [[0.4656280088928742, 0.21500769329533384, 0.31936429781179193],
         [0.0, 0.0, 1.0],
         [0.19806726878845474, 0.8019327312115453, 0.0]]])
    R = np.array(
        [[[0.7835460015641595, 0.9273255210020586, -0.2331169623484446],
          [0.0, -0.7192984391747097, 0.0],
          [0.0, -0.7881847856244157, -0.0]],
         [[-0.22702203774827612, 0.8051969510588093, -0.10010002017754482],
          [-0.0, -0.0, 0.14039354083575928],
          [-0.06737845428738742, -0.5111488159967945, -0.0]]])

    def test_dense_PR(self):
        np.random.seed(0)
        P, R = mdptoolbox.example.rand(self.S, self.A)
        assert_equal(P.shape, self.P.shape)
        assert_equal(R.shape, self.R.shape)
        assert_equal_numpy_spacing(P, self.P)
        assert_equal_numpy_spacing(R, self.R)

    def test_sparse_PR(self):
        P, R = mdptoolbox.example.rand(self.S, self.A, is_sparse=True)
        for a in range(self.A):
            assert_equal(P[a].shape, self.P[a].shape)
            assert_equal(R[a].shape, self.R[a].shape)

    def test_dense_PR_check(self):
        np.random.seed(0)
        P, R = mdptoolbox.example.rand(self.S, self.A)
        assert_is_none(mdptoolbox.util.check(P, R))

    def test_sparse_PR_check(self):
        np.random.seed(0)
        P, R = mdptoolbox.example.rand(self.S, self.A, is_sparse=True)
        assert_is_none(mdptoolbox.util.check(P, R))

    def test_S_raise(self):
        assert_raises(AssertionError, mdptoolbox.example.rand, S=0, A=self.A)

    def test_A_raise(self):
        assert_raises(AssertionError, mdptoolbox.example.rand, S=self.S, A=0)

    def test_mask_raise_1(self):
        mask = np.random.randint(2, size=(3, 6, 9))
        assert_raises(AssertionError, mdptoolbox.example.rand, S=self.S,
                      A=self.A, mask=mask)

    def test_mask_raise_2(self):
        assert_raises(TypeError, mdptoolbox.example.rand, S=self.S, A=self.A,
                      mask=True)

    def test_mask_dense_1(self):
        mask = np.array(
            [[1, 0, 0],
             [0, 1, 1],
             [1, 0, 1]])
        P, R = mdptoolbox.example.rand(S=self.S, A=self.A, mask=mask)
        assert_is_none(mdptoolbox.util.check(P, R))
        P, R = mdptoolbox.example.rand(S=self.S, A=self.A, mask=mask,
                                       is_sparse=True)
        assert_is_none(mdptoolbox.util.check(P, R))

    def test_mask_dense_2(self):
        mask = np.array(
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]])
        P, R = mdptoolbox.example.rand(S=self.S, A=self.A, mask=mask)
        assert_is_none(mdptoolbox.util.check(P, R))
        P, R = mdptoolbox.example.rand(S=self.S, A=self.A, mask=mask,
                                       is_sparse=True)
        assert_is_none(mdptoolbox.util.check(P, R))

    def test_mask_dense_3(self):
        mask = np.array(
            [[[1, 1, 0],
              [0, 1, 1],
              [1, 0, 1]],
             [[0, 0, 1],
              [0, 1, 0],
              [1, 0, 0]]])
        P, R = mdptoolbox.example.rand(S=self.S, A=self.A, mask=mask)
        assert_is_none(mdptoolbox.util.check(P, R))
        P, R = mdptoolbox.example.rand(S=self.S, A=self.A, mask=mask,
                                       is_sparse=True)
        assert_is_none(mdptoolbox.util.check(P, R))

    def test_mask_sparse_1(self):
        mask = sp.csr_matrix(
            [[1, 0, 0],
             [0, 1, 1],
             [1, 0, 1]])
        P, R = mdptoolbox.example.rand(S=self.S, A=self.A, mask=mask)
        assert_is_none(mdptoolbox.util.check(P, R))
        P, R = mdptoolbox.example.rand(S=self.S, A=self.A, mask=mask,
                                       is_sparse=True)
        assert_is_none(mdptoolbox.util.check(P, R))

    def test_mask_sparse_2(self):
        mask = np.array(
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]])
        P, R = mdptoolbox.example.rand(S=self.S, A=self.A, mask=mask)
        assert_is_none(mdptoolbox.util.check(P, R))
        P, R = mdptoolbox.example.rand(S=self.S, A=self.A, mask=mask,
                                       is_sparse=True)
        assert_is_none(mdptoolbox.util.check(P, R))
