
import numpy as np

from yaonet.tensor import *

a = Tensor(np.arange(120.).reshape(3, 4, 5, 2), requires_grad = True)
b = Tensor(np.arange(72.).reshape(4, 3, 2, 3), requires_grad = True)

c = t_tensordot(a, b, dims=([1, 0], [0, 1]))
sc = (c*2).sum()

sc.backward()

print()