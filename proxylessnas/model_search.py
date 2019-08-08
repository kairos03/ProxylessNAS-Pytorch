import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES, Genotype
import utils


class MixedOp(nn.Module):
  """mixed operation
  """
  def __init__(self, C_in, C_out, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      if primitive == 'identity' and C_in != C_out:
        continue
      op = OPS[primitive](C_in, C_out, stride, False)
      self._ops.append(op)

  def forward(self, x, weights):
    # weighted sum for all operations weights is architacture weight(alpha)
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):
  """Cell"""
  def __init__(self, C_in, C_out, steps, stride):
    super(Cell, self).__init__()

    self._steps = steps

    self._ops = nn.ModuleList()

    # stack layer
    self._ops.append(MixedOp(C_in, C_out, stride))
    for i in range(1, self._steps):
      op = MixedOp(C_out, C_out, 1)
      self._ops.append(op)

  def forward(self, x, weights):
    for i, op in enumerate(self._ops):
      x = op(x, weights[i])

    return x


class Network(nn.Module):

  def __init__(self, C_list, steps_list, strides_list, num_classes, criterion):
    super(Network, self).__init__()
    self._C_list = C_list               # [32, 16, 24, 32, 64, 96, 160, 320, 1280]  ## TODO how to select Channel dynamically??
    self._steps_list = steps_list       # [1,  1,  2,  3,  4,  3,  3,   1,   1]
    self._strides_list = strides_list   # [2,  1,  2,  2,  2,  1,  2,   1,   1]
    self._num_classes = num_classes     # 1000 for Imagenet
    self._criterion = criterion
            
    # stem layer
    self.stem = nn.Sequential(
      nn.Conv2d(3, self._C_list[0], 3, stride=self._strides_list[0], padding=1, bias=False),
      nn.BatchNorm2d(self._C_list[0]),
      MBConv(self._C_list[0], self._C_list[1], 1, self._strides_list[1], 0, 1)
    )

    # cells
    self.cells = list()
    for i in range(2, 8):
      cell = Cell(self._C_list[i-1], self._C_list[i], self._steps_list[i], self._strides_list[i])
      self.cells.append(cell)
    self.cells = nn.Sequential(*self.cells)

    # postprocess
    self.post = ConvBNReLU(self._C_list[7], self._C_list[8], 1, 1, 0)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(self._C_list[8], num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C_list, self._steps_list, self._strides_list, self._num_classes, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, x):
    x = self.stem(x)
    for i, cell in enumerate(self.cells):
      alpha = F.softmax(self._alphas_parameters[i], dim=-1)
      x = cell(x, alpha)
    x = self.post(x)
    x = self.global_pooling(x)
    logits = self.classifier(x.view(x.size(0), -1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = self._steps_list[2:8]
    num_ops = len(PRIMITIVES)

    # init alpha param for each mixed op
    self._alphas_parameters = list()
    for k in self._steps_list[2:8]:
      self._alphas_parameters.append(Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True))

  def arch_parameters(self):
    return self._alphas_parameters

  def genotype(self):
    def _parse(weights):
      gene = []
      for i in range(len(weights)):
        idx = torch.argmax(weights[i][1:]) + 1 # except zero operation
        best = PRIMITIVES[idx]
        gene.append(best)
      return gene

    genotype = list()
    for i in range(len(self.cells)):
      genotype.append(_parse(F.softmax(self._alphas_parameters[i], dim=-1).data.cpu()))

    return genotype
