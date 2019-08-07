import unittest

import torch
import numpy as np

from proxylessnas import utils


class test_utils(unittest.TestCase):

    def test_binarize(self):
        p = torch.tensor([0.1, 0.4, 0.4, 0.1])
        b = utils.binarize(p, 1)
        self.assertEqual(len(b), len(p))
        self.assertEqual(sum(b), 1)

        p = torch.tensor([1., 0., 0., 0.])
        for _ in range(100):
            b = utils.binarize(p)
            self.assertTrue(torch.equal(b, p))

        p = torch.tensor([0.1, 0.4, 0.4, 0.1])
        b = utils.binarize(p, 2)
        print(b)
        self.assertEqual(len(b), len(p))
        self.assertEqual(sum(b), 2)
