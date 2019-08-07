import unittest
import torch

from proxylessnas import operations as op


class test_operations(unittest.TestCase):

  def test_ConvBNReLU(self):
    x = torch.randn([1, 10, 8, 8])

    conv = op.ConvBNReLU(10, 20, 1, 1, 0)
    out = conv(x)
    self.assertEqual(out.shape, (1, 20, 8, 8))
    print(out.shape)

    conv = op.ConvBNReLU(10, 10, 1, 1, 0)
    out = conv(x)
    print(out.shape)
    self.assertEqual(out.shape, (1, 10, 8, 8))

    conv = op.ConvBNReLU(10, 5, 3, 2, 1)
    out = conv(x)
    print(out.shape)
    self.assertEqual(out.shape, (1, 5, 4, 4))

    conv = op.ConvBNReLU(10, 5, 3, 2, 1)
    out = conv(x)
    print(out.shape)
    self.assertEqual(out.shape, (1, 5, 4, 4))

  def test_depthwise_conv(self):
    x = torch.randn([1, 10, 8, 8])

    conv = op.depthwise_conv(10, 1, 1, 10, True)
    out = conv(x)
    self.assertEqual(out.shape, (1, 10, 8, 8))

  def test_MBConv(self):
    x = torch.randn([1, 10, 8, 8])

    conv = op.MBConv(10, 10, 3, 1, 1, 3)
    out = conv(x)
    self.assertEqual(out.shape, (1, 10, 8, 8))

    conv = op.MBConv(10, 20, 3, 1, 1, 3)
    out = conv(x)
    self.assertEqual(out.shape, (1, 20, 8, 8))

    conv = op.MBConv(10, 20, 3, 2, 1, 6)
    out = conv(x)
    self.assertEqual(out.shape, (1, 20, 4, 4))
