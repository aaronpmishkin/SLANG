r"""
Base class for optimizer depending on a closure the returns gradients
and other information instead of appending it to ``parameter.grad``.
"""

import torch

class Optimizer(torch.optim.Optimizer):
    r"""Base class for optimizer depending on a closure the returns gradients
    and other information instead of appending it to ``parameter.grad``.
    """

    def step(self, closure):
        r"""
        """
        raise NotImplementedError
