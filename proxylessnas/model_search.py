import torch
import torch.nn as nn
import torch.nn.functional as F
from proxylessnas.operations import *
from torch.autograd import Variable
from proxylessnas.genotypes import PRIMITIVES, Genotype
from proxylessnas import utils


class MixedOp(nn.Module):
  """mixed operation
  """
  def __init__(self, C_in, C_out, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C_in, C_out, stride)
      self._ops.append(op)

  def forward(self, x, weights):
    # weighted sum for all operations weights is architacture weight(alpha)
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):
  def __init__(self, C_in, C_out, steps, stride):
    super(Cell, self).__init__()

    self._steps = steps

    self._ops = nn.ModuleList()
    self.alpha = torch.autograd.Variable(1e-3*torch.randn(self._steps, len(PRIMITIVES)))

    # stack layer
    self._ops.append(MixedOp(C_in, C_out, stride))
    for i in range(1, self._steps):
      op = MixedOp(C_out, C_out, 1)
      self._ops.append(op)

  def forward(self, x):
    for i, op in enumerate(self._ops):
      bin = utils.binarize(F.softmax(self.alpha[i]))
      x = op(x, bin)

    return x


class Network(nn.Module):

  def __init__(self, C_list, steps_list, strides_list, num_classes, criterion):
    super(Network, self).__init__()
    self._C_list = C_list             # [32, 16, 24, 32, 64, 96, 160, 320, 1280]
    self._steps_list = steps_list     # [1,  1,  2,  3,  4,  3,  3,   1,   1]
    self._strides_list = strides_list # [2,  1,  2,  2,  2,  1,  2,   1,   1]
    self._num_classes = num_classes   # 1000 for Imagenet
    self._criterion = criterion

    self.stem = nn.Sequential(
      nn.Conv2d(3, self._C_list[0], 3, stride=self._strides_list[0], padding=1, bias=False),
      nn.BatchNorm2d(self._C_list[0]),
      MBConv(self._C_list[0], self._C_list[1], 1, self._strides_list[1], 0, 1)
    )

    self.cells = nn.Sequential()
    for i in range(2, 6):
      cell = Cell(self._C_list[i-1], self._C_list[i], self._steps_list[i], self._strides_list[i])
      self.cells.add_module

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self.criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      # caclulate alpha softmax
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      # binarize
      binarize = utils.binarize(weights, 1)
      s0, s1 = s1, cell(s0, s1, binarize)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):
    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

