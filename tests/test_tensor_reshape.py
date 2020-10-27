
import unittest
import pytest

import numpy as np

from yaonet.tensor import Tensor
from yaonet.operations import matmul

class TestTensorAdd(unittest.TestCase):
    def test_tensor_reshape(self):
        # t1 is (1, 2)
        t1 = Tensor([[1, 2]], requires_grad=True)

        # t2 is a (2, 2)
        t2 = Tensor([[10, 10], [20, 10]], requires_grad=True)

        # (1, 2)
        t3 = t1 @ t2
        assert t3.data.tolist() == [[50, 30]]

        grad = Tensor([[-1, -2]])
        t3.backward(grad)

        assert t1.grad.tolist() == [[-30, -40]]

        # t1 is (1, 2)
        t1 = Tensor([[1, 2]], requires_grad=True)

        # t2 is a (2, 2)
        t2 = Tensor([[10, 10], [20, 10]], requires_grad=True)

        # (2, 1)
        t4 = matmul(t2, t1.reshape(2, 1))

        assert t4.data.tolist() == [[30], [40]]

        grad = Tensor([[-1], [-2]])
        t4.backward(grad)

        assert t1.grad.reshape(2 ,1).tolist() == [[-50], [-30]]