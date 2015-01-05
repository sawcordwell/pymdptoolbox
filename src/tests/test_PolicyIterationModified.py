# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:33:16 2013

@author: steve
"""

from nose.tools import assert_equal
import numpy as np
import scipy.sparse as sp

from mdptoolbox import mdp

from .utils import BaseTestMDP, assert_sequence_almost_equal

class TestPolicyIterationModified(BaseTestMDP):
    def test_small(self):
        pim = mdp.PolicyIterationModified(self.small_P, self.small_R, 0.9)
        pim.run()
        assert_sequence_almost_equal(pim.V,
                                    (41.8656419239403, 35.4702797722819))
        assert_equal(pim.policy, (1, 0))

    def test_small_undiscounted(self):
        pim = mdp.PolicyIterationModified(self.small_P, self.small_R, 1)
        pim.run()
        assert_equal(pim.policy, (1, 0))
