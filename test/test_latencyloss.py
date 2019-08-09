import unittest

import torch
from torch.autograd import Variable

from proxylessnas.latencyloss import *
from proxylessnas.model_search import Network


class test_latencyloss(unittest.TestCase):

    def setUp(self):
        # 14x14x80-14x14x80-expand:3-kernel:5
        self.channels = [32, 16, 24, 40, 80, 96, 192, 320, 1280]
        self.steps =    [1,  1,  2,  3,  4,  3,  3,   1,   1]
        self.strides =  [2,  1,  2,  2,  2,  1,  2,   1,   1]
        self.loss = LatnecyLoss(self.channels[2:8], self.steps[2:8], self.strides[2:8])

    def test_find_latency(self):
        self.assertEqual(self.loss._predictor('identity_3_5_80_80_14_1'), 0)
        self.assertEqual(self.loss._predictor('mbconv_3_5_80_80_14_1'), 1.9960465116279071)

    def test_calculate_feature_map_size(self):
        self.loss._calculate_feature_map_size(112)
        self.assertEqual(self.loss.feature_maps, [112, 56, 28, 14, 14, 7])

    def test_forward(self):
        # run test
        num_ops = len(PRIMITIVES)
        # init alpha param for each mixed op
        self._alphas_parameters = list()
        for k in self.steps[2:8]:
            self._alphas_parameters.append(Variable(1e-3*torch.randn(k, num_ops), requires_grad=True))
        
        self.loss.forward(self._alphas_parameters)
