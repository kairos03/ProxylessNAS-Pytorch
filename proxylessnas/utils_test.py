import unittest

import torch
import numpy as np

import utils


class test_utils(unittest.TestCase):

    def test_binarize(self):
        p = torch.tensor([0.1, 0.4, 0.4, 0.1])
        b = utils.binarize(p)
        assert len(b) == len(p)
        assert sum(b) == 1

        p = torch.tensor([1., 0., 0., 0.])
        for _ in range(100):
            b = utils.binarize(p)
            assert torch.equal(b, p)
