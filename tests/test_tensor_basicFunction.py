
import unittest
import pytest

import numpy as np

from yaonet.tensor import Tensor
from yaonet.basic_functions import *


class TestTensorBscFunstion(unittest.TestCase):
    def test_sin(self):
        # 90, 30, 150
        t1 = Tensor([np.pi/2., np.pi/6., np.pi*(5/6)], requires_grad=True)

        t2 = sin(t1)

        # assert
        np.testing.assert_array_almost_equal(t2.data, [1, 0.5, 0.5])

        t2.backward(Tensor([-1., -2., -3.]))

        assert t1.grad.tolist() == (np.array([-1., -2., -3]) * np.cos(t1.data)).tolist()

    def test_log(self):

        t1 = Tensor([1, 2, 4], requires_grad=True)

        t2 = log(2, t1)

        np.testing.assert_array_almost_equal(t2.data, [0, 1, 2])

        t2.backward(Tensor([-1., -2., -3.]))
        
        assert t1.grad.tolist() == (np.array([-1., -2., -3]) * (1 / (t1.data * np.log(2)))).tolist()