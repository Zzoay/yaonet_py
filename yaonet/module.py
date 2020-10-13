from typing import Iterator

from yaonet.tensor import Tensor
from yaonet.parameter import Parameter


class Module():
    def __init__(self,) -> None:
        return None
    
    def parameters(self) -> Iterator[Parameter]:
        for _, value in vars(self).items():
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()
