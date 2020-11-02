
import numpy as np

from yaonet.tensor import Tensor


class Parameter(Tensor):
    def __init__(self, shape) -> None:
        data = np.random.standard_normal(shape)
        super().__init__(data, requires_grad=True)
