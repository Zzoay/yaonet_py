from typing import Iterator

from yaonet.tensor import Tensor
from yaonet.parameter import Parameter


class Module():
    def __init__(self,) -> None:
        return None
    
    def parameters(self, for_optim=False):
        # if for_optim, means that we want to send it to a optimizer, it better be a generator but had not call
        if for_optim:
            return self._parameters
        else:
            return self._parameters()

    def _parameters(self) -> Iterator[Parameter]:
        for _, value in vars(self).items():
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()
