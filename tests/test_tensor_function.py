
import unittest
import pytest

import numpy as np

from yaonet.tensor import Tensor
from yaonet.functional import *


class TestTensorFunstion(unittest.TestCase):
    def test_sigmoid(self):
        # 90, 30, 150
        t1 = Tensor([1, 2, 6], requires_grad=True)

        t2 = sigmoid(t1)

        # assert
        np.testing.assert_array_almost_equal(t2.data, 1 / (1+ np.exp([-1, -2, -6])))

        t2.backward(Tensor([-1., -2., -3.]))

        def sig(t):
            return 1 / (1 + np.exp(-t))
        
        truth = [-1,-2,-3]*sig(t1.data)*(1-sig(t1.data))
        np.testing.assert_array_almost_equal(t1.grad, truth)

    def test_tanh(self):
        t1 = Tensor([1, 2, 6], requires_grad=True)

        t2 = tanh(t1)

        np.testing.assert_array_almost_equal(t2.data, np.tanh(t1.data))

        t2.backward(Tensor([-1., -2., -3.]))

        np.testing.assert_array_almost_equal(t1.grad, [-1,-2,-3]*(1-np.power(t2.data,2 )))

    def test_relu(self):
        t1 = Tensor([-1, 0, 2], requires_grad=True)

        t2 = relu(t1)

        assert t2.data.tolist() == [0,0,2]

        t2.backward(Tensor([-1., -2., -3.]))

        assert t1.grad.tolist() == [0, 0, -6]