# Code overview

We use Matlab (R2018b) and Python 3.

Our Python dependencies are 
[`Artemis-ML`](https://github.com/QUVA-Lab/artemis), 
[`scikit-learn`](https://scikit-learn.org/stable/index.html), 
[`PyTorch`](https://github.com/pytorch/pytorch/), 
[`Pandas`](https://pandas.pydata.org), and
[`BayesianOptimization`](https://github.com/fmfn/BayesianOptimization).
You will also need to install two custom packages, `Torchutils` and `vi-lib`, bundled in the repo.

## Installation

We recommend to create an environment for running this code.
The following snippet creates an Anaconda environment, 
installs the dependencies (except Pytorch) 
and our custom packages (make sure to change the paths!)
```
conda create --name SLANG python=3
conda activate SLANG

# See installation instructions for Pytorch (: https://pytorch.org/get-started/locally/

pip install scikit-learn pandas artemis-ml
pip install bayesian-optimization==0.6.0

cd /path/to/SLANG/code/python/libs/vi_lib 
pip install .
cd /path/to/SLANG/code/python/libs/low_rank_ops
pip install .
```

## Downloading Datasets and pre-trained models

The [last release](https://github.com/aaronpmishkin/SLANG/releases) has `.zip` attachements;
* the content of `data.zip` should be set in `~/data`
* the content of `artemis.zip` should be put in `~/.artemis/experiments`
* the content of `paper_experiment_data.zip` should be put in `path_to_slang/code/paper_experiment_data`

## Running instruction 

The `main.py` script serves as an entry point for running any experiments.
* To reproduce the tables and plots from the paper using the pre-trained models, run `python main.py -expN`, where `N` is the experiment to run, see list below.
* To start from scratch run the optimizers, add the `-run` option (this will take a while).
* The script has additional warnings to remind you to put the data where it expects it and when the optimization process will take a long time. 
  Those can be disabled with the `--force` flag. 

Running `python main.py --help` outputs the following help.
```
usage: main.py [-h] (-exp1 | -exp2 | -exp3 | -exp4 | -exp5 | -exp6) [-run]
               [-f] [-n]

Main Runner

optional arguments:
  -h, --help            show this help message and exit
  -exp1, --logreg-table
  -exp2, --logreg-convergence
  -exp3, --logreg-vizualization
  -exp4, --bnn-uci-table
  -exp5, --bnn-mnist-table
  -exp6, --bnn-convergence
  -run, --run-optimizer
                        Runs the optimizer for the experiment. If not set,
                        will use pretrained data.
  -f, --force           Disable warnings.
  -n, --no-exec         Does not execute code, just print the commands to run.
```
