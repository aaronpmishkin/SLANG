**Status**: We are actively updating this repository with code to reproduce our experimental results.

## SLANG: Fast Structured Covariance Approximations for Bayesian Deep Learning with Natural Gradient

---

Code for reproducing experimental results in the paper
[SLANG: Fast Structured Covariance Approximations for Bayesian Deep Learning with Natural Gradient](https://arxiv.org/abs/1811.04504) by Aaron Mishkin, Frederik Kunstner, Didrik Nielsen, Mark Schmidt, and Mohammad Emtiyaz Khan.


### Requirements:

Our code requires that you have installed
* Python (version 3.6 or higher),
* PyTorch (version 4.0 or higher),
* SciPy + NumPy

### Basic Usage
1. Clone the repository with
```
git clone https://github.com/aaronpmishkin/SLANG
```
2. Install the library of low-rank matrix operations using
```
cd SLANG/code/python/low_rank_ops
pip install .
```
3. Run the demo example with
```
cd ..
python demo.py
```

### Currently Available Code
We are preparing code to replicate all of our experiments. Currently, we have added the core of our library, which includes:
* Fast linear algebra computations of `low-rank + diagonal` matrices,
* An implementation of Goodfellow's trick to compute individual gradients from a minibatch,
* An implementation of a MLP matching the requirements for Goodfellow's trick (getting the output for each layer) along with support for parrallel computation for different sets of weights (e.g., additive noise),
* The SLANG optimizer.

We expect to update this repository shortly.
