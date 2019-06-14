# TorchUtils: `Low rank + diagonal operations` and some helper functions

This package provides some utility functions based on PyTorch.

Install with `pip install -e .` (in this folder)

## Modules

* `torchutils.low_rank` - Fast Operations for Low-Rank + Diagonal Matrices
* `torchutils.distributions` - `Low-Rank + Diagonal` Multivariate Gaussian
* `torchutils.fastpca` - Fast Randomized PCA, ported from [FB PCA](https://github.com/facebook/fbpca)
* `torchutils.curvfuncs` - Implementation of [Goodfellow's trick](https://arxiv.org/abs/1510.01799) for computing individual gradients
* `torchutils.models` - Simple models instantiation helpers with support for Goodfellow's trick and parallel forward/backward passes with MC samples of parameters
* `torchutils.params` - Some helpers to move between `matrix` and `list-of-parameters` representations

