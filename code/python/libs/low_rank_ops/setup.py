from setuptools import setup

setup(
    name='slang_torchutils', 
    version='0.1.0',
    description='Some extensions/helpers for common functionality of PyTorch',  
    long_description=(""
    "* `torchutils.low_rank` - Fast Operations for Low-Rank + Diagonal Matrices\n"
    "* `torchutils.distributions` - `Low-Rank + Diagonal` Multivariate Gaussian\n"
    "* `torchutils.fastpca` - Fast Randomized PCA, ported from [FB PCA](https://github.com/facebook/fbpca)\n"
    "* `torchutils.curvfuncs` - Implementation of [Goodfellow's trick](https://arxiv.org/abs/1510.01799) for computing individual gradients\n"
    "* `torchutils.models` - Simple models instantiation helpers with support for Goodfellow's trick and parallel forward/backward passes with MC samples of parameters\n"
    "* `torchutils.params` - Some helpers to move between `matrix` and `list-of-parameters` representations"),
    long_description_content_type='text/markdown',
    url='https://github.com/aaronpmishkin/SLANG',
    author='Frederik Kunstner, Aaron Mishkin, Didrik Nielsen',
    author_email='frederik.kunstner@gmail.com, amishkin@cs.ubc.ca, didrik.nielsen@riken.jp',
    packages=['torchutils'],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='torch pytorch torchutils',
    install_requires=['torch'],
)
