import unittest
import pytest

import numpy as np

from autograd.tensor import BasicTensor, Tensor


class TestTensorSlice(unittest.TestCase):
    def test_slice(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)

        t3 = t1[0] + t2

        assert t1[:2].tolist() == [1, 2]
        assert t3.tolist() == [5, 6, 7]

        t3.backward(np.array([-1., -2., -3.]))

        assert t1.grad.tolist() == [-6, 0, 0]  # broadcasted, sum over all grad
        assert t2.grad.tolist() == [-1, -2, -3]