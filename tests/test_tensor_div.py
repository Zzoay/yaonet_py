import unittest
import pytest

import numpy as np

from yaonet.tensor import Tensor

class TestTensorDiv(unittest.TestCase):
    def test_simple_div(self):
        t1 = Tensor([2, 4, 8], requires_grad=True)
        t2 = Tensor([1, 2, 8], requires_grad=True)

        t3 = t1 / t2

        assert t3.data.tolist() == [2, 2, 1]

        t3.backward(Tensor([-1., -2., -3.]))

        assert t1.grad.tolist() == [-1, -1, -3/8]
        assert t2.grad.tolist() == [2, 2, 3/8]

        t1 /= -2
        assert t1.grad is None

        np.testing.assert_array_almost_equal(t1.data, [-1, -2, -4])