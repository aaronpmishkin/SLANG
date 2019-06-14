# @Author: amishkin
# @Date:   18-08-17
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-09-13

import torch
from torch.optim import Adam
from artemis.experiments import experiment_function, ExperimentFunction

from lib.data.datasets import DatasetCV, Dataset, DEFAULT_DATA_FOLDER
from lib.experiments.experiment import run_experiment
from lib.experiments.cv_experiment import run_cv_experiment
from lib.models.bayesian_mlp import BayesianMLP
import lib.metrics.metric_factory as metric_factory
import lib.optimizers.closure_factories as closure_factories
from lib.plots.basics import plot_objective
from lib.utilities.general_utilities import set_seeds

import experiments.default_parameters as defaults

###############################
### BBB Core Initialization ###
###############################

def init_bbb_experiment(data, model_params, optimizer_params, train_set_size=None, use_cuda=torch.cuda.is_available()):

    output_size = data.num_classes
    if output_size == 2:
        output_size = None  # Binary Classification

    # Initialize the model
    model = BayesianMLP(input_size=data.num_features,
                        hidden_sizes=model_params['hidden_sizes'],
                        output_size=output_size,
                        act_func=model_params['activation_function'],
                        prior_prec=model_params['prior_precision'],
                        bias_prior_prec=model_params['bias_prior_precision'])

    if use_cuda:
        model = model.cuda()

    def predict_fn(x, mc_samples):
        preds = [model(x) for _ in range(mc_samples)]
        return preds

    def kl_fn():
        return model.kl_divergence()

    # Use the closure that returns aggregated gradients
    closure_factory = closure_factories.vi_closure_factory

    # Initialize optimizer
    optimizer = Adam(model.parameters(),
                     lr=optimizer_params['learning_rate'],
                     betas=(optimizer_params['momentum_rate'], optimizer_params['beta']),
                     eps=1e-8)

    return model, predict_fn, kl_fn, closure_factory, optimizer


#######################
### BBB Experiments ###
#######################

@ExperimentFunction(show=plot_objective)
def bbb_base(data_set='australian_presplit',
                    model_params=defaults.MLP_CLASS_PARAMETERS,
                    optimizer_params=defaults.VI_PARAMETERS,
                    objective='avneg_elbo_bernoulli',
                    metrics=metric_factory.BAYES_BINCLASS,
                    normalize={'x': False, 'y': False},
                    save_params=defaults.SAVE_PARAMETERS,
                    seed=123,
                    use_cuda=torch.cuda.is_available(),
                    init_hook=None):

    set_seeds(seed)

    data = Dataset(data_set=data_set, data_folder=DEFAULT_DATA_FOLDER)

    model, predict_fn, kl_fn, closure_factory, optimizer = init_bbb_experiment(data, model_params, optimizer_params, use_cuda=use_cuda)

    if init_hook is not None:
        init_hook(model, optimizer)

    results_dict = run_experiment(data, model, model_params, predict_fn, kl_fn, optimizer, optimizer_params, objective, metrics, closure_factory, normalize, save_params, seed, use_cuda)

    return results_dict

@ExperimentFunction()
def bbb_cv(data_set='australian_presplit',
                  n_splits=10,
                  model_params=defaults.MLP_CLASS_PARAMETERS,
                  optimizer_params=defaults.VI_PARAMETERS,
                  objective='avneg_elbo_bernoulli',
                  metrics=metric_factory.BAYES_BINCLASS,
                  normalize={'x': False, 'y': False},
                  save_params=defaults.SAVE_PARAMETERS,
                  seed=123,
                  use_cuda=torch.cuda.is_available()):

    set_seeds(seed)
    data = DatasetCV(data_set=data_set, n_splits=n_splits, seed=seed, data_folder=DEFAULT_DATA_FOLDER)

    results_dict = run_cv_experiment(data, n_splits, init_bbb_experiment, model_params, optimizer_params, objective, metrics, normalize, save_params, seed, use_cuda)

    return results_dict

####################################
### Register Experiment Variants ###
####################################

def bbb_binclass():
    # set the model parameters
    model_params = defaults.MLP_CLASS_PARAMETERS.copy()
    model_params['hidden_sizes'] = [] # No hidden layers
    model_params['prior_prec'] = 1e-5
    model_params['bias_prior_precision'] = 1e-6
    model_params['noise_precision'] = None # Classification Problem

    # Create the experiment and register it.
    bbb_base.add_variant("logistic regression", model_params=model_params, data_set="australian")


def bbb_regression():
    # set the model parameters
    model_params = defaults.MLP_REG_PARAMETERS

    opt_params = { 'num_epochs': 100,
                   'batch_size': 32,
                   'learning_rate': 0.01,
                   'momentum_rate': 0.9,
                   'train_mc_samples': 20,
                   'eval_mc_samples': 20,
                   'beta': 0.999 }

    metrics = metric_factory.BAYES_REG
    objective = 'avneg_elbo_gaussian'

    # Create the experiment and register it.
    bbb_base.add_variant("bnn regression", data_set="wine1", model_params=model_params, optimizer_params=opt_params, metrics=metrics, objective=objective)
    bbb_cv.add_variant("bnn regression cv", data_set="wine1", n_splits=5, model_params=model_params, optimizer_params=opt_params, metrics=metrics, objective=objective)


####################################
### All Load Experiment Variants ###
####################################

def load_experiments():
    bbb_binclass()
    bbb_regression()

load_experiments()
