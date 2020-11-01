
import unittest

import numpy as np

from yaonet.tensor import Tensor
from yaonet.layers import *


class TestConv(unittest.TestCase):
    def test_conv2d(self):
        image = np.array(
             [[[[ 0,  1,  2,  3,  4],
                [ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19]]]]
        ).reshape(1, 1, 4, 5)  # batchsize, channel, height, width
        image = Tensor(image, requires_grad=True)

        ksize = 3
        stride = 1

        s = im2col(image, ksize, stride)
        s.backward(np.ones(54).reshape(1, 6, 9))
        
        assert image.grad.shape == image.shape

        image.zero_grad()    

        cnn = Conv2d(in_channels=1, out_channels=1, kernel_size=ksize, stride=stride)

        y = cnn(image)

        y.backward(np.ones(6).reshape(1, 1, 2, 3))