import csv
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

import proxylessnas

import torch
import torch.nn as nn
from proxylessnas.genotypes import PRIMITIVES


class LatnecyLoss(nn.Module):
    def __init__(self, channels, steps, strides, input_size=56):
        super(LatnecyLoss, self).__init__()

        self.channels = channels
        self.steps = steps
        self.strides = strides

        self._calculate_feature_map_size(input_size)
        self._load_latency()

    def _load_latency(self):
        # load predicted latency file
        f = pkg_resources.open_text(proxylessnas, "latency.csv")
        rdr = csv.reader(f)

        self._latency = {}
        for line in rdr:
            self._latency[line[0]] = line[1]
        f.close()

    def _calculate_feature_map_size(self, input_size):
        self.feature_maps = [input_size]
        for s in self.strides[:-1]:
            input_size = input_size // s
            self.feature_maps.append(input_size)

    def _predictor(self, inputs):
        """predict latency
        input example: mbconv_6_3_80_80_14_1
        """
        div = inputs.split('_', maxsplit=-1)
        if div[0] == 'identity' or div[0] == 'none':
            div.insert(1, 0)  # insert fake exp_rate
            div.insert(2, 0)  # insert fake ksize
        op, exp_rate, ksize, C_in, C_out, size, stride = div
        print(op)
        if op == 'identity' or op == 'none':
            return 0
        out_size = int(size) // int(stride)
        findstr = f'{size}x{size}x{C_in}-{out_size}x{out_size}x{C_out}-expand:{exp_rate}-kernel:{ksize}-stride:{stride}' 
        print(findstr)
        return float(self._latency.get(findstr))

    def forward(self, alpha):
        latency = 0

        for i, a_cell in enumerate(alpha):
            c_in = self.channels[i]
            c_out = self.channels[i+1]
            fm = self.feature_maps[i]
            strides = self.strides[i]

            for j, weights in enumerate(a_cell):
                op_names = PRIMITIVES
                strides = 1 if j != 0 else strides
                latency += sum(w * self._predictor(f'{op}_{c_in}_{c_out}_{fm}_{strides}') for w, op in zip(weights, op_names))
