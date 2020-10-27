
import numpy as np

from yaonet.tensor import Tensor
from yaonet.layers import *


image = np.array(
    [[[[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19]]]]
).reshape(1, 1, 4, 5)
ksize = 3
stride = 1

image = Tensor(image, requires_grad=True)
# s = im2col(image, ksize, stride)

# s.backward(np.array([1, 1, 1, 1, 1, 1]).reshape(6,1))

# assert image.grad.shape == image.shape

cnn = Conv2d(in_channels=1, out_channels=1, kernel_size=ksize, stride=stride)

y = cnn(image)

y.backward(np.array([[1], [1], [1], [1], [1], [1]]).reshape(1,1,2,3))
print