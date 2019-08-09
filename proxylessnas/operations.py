import torch
import torch.nn as nn

OPS = {
  'none': lambda C_in, C_out, stride, affine: Zero(C_out, stride),
  'identity': lambda C_in, C_out, stride, affine: Identity() if stride == 1 else FactorizedReduce(C_in, C_out, affine),
  'mbconv_3_3': lambda C_in, C_out, stride, affine: MBConv(C_in, C_out, 3, stride, 1, 3, affine),
  'mbconv_3_5': lambda C_in, C_out, stride, affine: MBConv(C_in, C_out, 5, stride, 2, 3, affine),
  'mbconv_3_7': lambda C_in, C_out, stride, affine: MBConv(C_in, C_out, 7, stride, 3, 3, affine),
  'mbconv_6_3': lambda C_in, C_out, stride, affine: MBConv(C_in, C_out, 3, stride, 1, 6, affine),
  'mbconv_6_5': lambda C_in, C_out, stride, affine: MBConv(C_in, C_out, 5, stride, 2, 6, affine),
  'mbconv_6_7': lambda C_in, C_out, stride, affine: MBConv(C_in, C_out, 7, stride, 3, 6, affine),
}

def depthwise_conv(in_channels, kernel_size, stride, groups, affine):
    padding = kernel_size // 2
    return ConvBNReLU(in_channels, in_channels, kernel_size, stride, padding, groups, affine)


class ConvBNReLU(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, groups=1, affine=True, activation=True):
    super(ConvBNReLU, self).__init__()

    self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride, padding, groups=groups, bias=False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)
    if activation:
      self.act = nn.ReLU6()
    
  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    if hasattr(self, 'act'):
      x = self.act(x)
    return x


class MBConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, expansion_factor, affine=True):
    super(MBConv, self).__init__()

    C_exp = C_in * expansion_factor
    self.res_connect = C_in == C_out and stride == 1

    self.op = nn.Sequential(
      ConvBNReLU(C_in, C_exp, 1, 1, 0, affine=affine),
      depthwise_conv(C_exp, kernel_size, stride, C_exp, affine=affine),
      ConvBNReLU(C_exp, C_out, 1, 1, 0, activation=False, affine=affine)
    )

  def forward(self, x):
    if self.res_connect:
      return self.op(x) + x
    else: 
      return self.op(x)


class ReLUConvBN(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)


class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, C_out, stride):
    super(Zero, self).__init__()
    self.stride = stride
    self.C_out = C_out

  def forward(self, x):
    n, _, h, w = x.size()
    c = self.C_out
    h //= self.stride
    w //= self.stride
    if x.is_cuda:
        with torch.cuda.device(x.get_device()):
            padding = torch.cuda.FloatTensor(n, c, h, w).fill_(0)
    else:
        padding = torch.zeros(n, c, h, w)
    padding = torch.autograd.Variable(padding, requires_grad=False)
    return padding

class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

