import torch
import torch.nn as nn
from genotypes import PRIMITIVES


class LatnecyLoss(nn.Module):
    def __init__(self, channels, steps, strides, input_size=224):
        super(LatnecyLoss, self).__init__()

        self.channels = channels
        self.steps = steps
        self.strides = strides

        self._calculate_feature_map_size(input_size)
        self._build_predictor()

    def _calculate_feature_map_size(self, input_size):
        self.feature_maps = [input_size]
        for s in self.strides:
            x = input_size // s
            self.feature_maps.append(x)

    def _build_predictor(self):
        self.predictor = None        

    def forward(self, model):
        latency = 0
        alpha = model.arch_parameters()

        for i, a_cell in enumerate(alpha):
            c_in = self.channels[i]
            c_out = self.channels[i+1]
            fm = self.feature_maps[i]
            strides = self.strides[i]

            for j, weights in enumerate(a_cell):
                op_names = PRIMITIVES
                strides = 1 if j != 0 else strides
                latency += sum(w * self.predictor('{}_{}_{}'.format()) for w, op in zip(weights, op_names))
