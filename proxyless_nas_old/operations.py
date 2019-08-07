from collections import OrderedDict

import torch
import torch.nn as nn

OPS = {
  'mbconv-3-3' : lambda C, stride, affine: MBConv(C, C, 3, stride, 3),
  'mbconv-6-3' : lambda C, stride, affine: MBConv(C, C, 3, stride, 6),
  'mbconv-3-5' : lambda C, stride, affine: MBConv(C, C, 5, stride, 3),
  'mbconv-6-5' : lambda C, stride, affine: MBConv(C, C, 5, stride, 6),
  'mbconv-3-7' : lambda C, stride, affine: MBConv(C, C, 7, stride, 3),
  'mbconv-6-7' : lambda C, stride, affine: MBConv(C, C, 7, stride, 6),
  'identitiy' : lambda C, stride, affine: nn.Identity(),
}

class MBConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size=3, stride=1, expand_ratio=6):
    super(MBConv, self).__init__()

    feature_dim = round(C_in * self.expand_ratio)
    self.inverted_bottleneck = nn.Sequential(
      nn.Conv2d(C_in, feature_dim, 1, 1, 0, bias=False),
      nn.BatchNorm2d(feature_dim),
      nn.ReLU6(inplace=True),
    )
    self.depth_conv = nn.Sequential(
      nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, kernel_size//2, groups=feature_dim, bias=False),
      nn.BatchNorm2d(feature_dim),
      nn.ReLU6(inplace=True)
    )
    self.point_linear = nn.Sequential(
      nn.Conv2d(feature_dim, C_out, 1, 1, 0, bias=False),
      nn.BatchNorm2d(C_out),
    )

  def forward(self, x):
    x = self.inverted_bottleneck(x)
    x = self.depth_conv(x)
    x = self.point_linear(x)
    return x
