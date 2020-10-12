
import numpy as np

from autograd.tensor import Dependency, Tensor, t_add, t_sub, t_mul, t_matmul, ensure_tensor


def add(t1: Tensor, t2: Tensor) -> Tensor:
    return t_add(ensure_tensor(t1), ensure_tensor(t2))


def mul(t1: Tensor, t2: Tensor) -> Tensor:
    return t_mul(ensure_tensor(t1), ensure_tensor(t2))


def sub(t1: Tensor, t2: Tensor) -> Tensor:
    return t_sub(ensure_tensor(t1), ensure_tensor(t2))


def matmul(t1: Tensor, t2: Tensor) -> Tensor:
    return t_matmul(ensure_tensor(t1), ensure_tensor(t2))
 