# @Author: amishkin
# @Date:   18-08-17
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-09-13

import torch
import torch.nn.functional as F
from artemis.experiments import experiment_function, ExperimentFunction

from lib.optimizers.slang import SLANG
from lib.data.datasets import DatasetCV, Dataset, DEFAULT_DATA_FOLDER
from lib.experiments.experiment import run_experiment
from lib.experiments.cv_experiment import run_cv_experiment
import lib.metrics.metric_factory as metric_factory
import lib.optimizers.closure_factories as closure_factories
from lib.plots.basics import plot_objective
from lib.utilities.general_utilities import construct_prior_vector
from lib.utilities.general_utilities import set_seeds

from torchutils.models import MLP as MultiSampleMLP

import experiments.default_parameters as defaults

#################################
### SLANG Core Initialization ###
#################################

def init_slang_experiment(data, model_params, optimizer_params, train_set_size=None, use_cuda=torch.cuda.is_available()):
    output_size = data.num_classes
    if output_size == 2 or output_size is None:
        output_size = 1  # Binary Classification works differently on Fred's MLP...

    act_func = F.tanh if model_params['activation_function'] == "tanh" else F.relu

    # Initialize the model
    model = MultiSampleMLP(input_size=data.num_features,
                           hidden_sizes=model_params['hidden_sizes'],
                           output_size=output_size,
                           act_func=act_func)

    if use_cuda:
        model = model.cuda()

    # Use the closure that returns separate minibatch gradients.
    closure_factory = closure_factories.individual_gradients_closure_factory


    prior_vector = construct_prior_vector(weight_prior=model_params['prior_precision'],
                                          bias_prior=model_params['bias_prior_precision'],
                                          named_parameters=model.named_parameters())
    # Initialize optimizer
    optimizer = SLANG(model.parameters(),
                      lr=optimizer_params['learning_rate'],
                      betas=(optimizer_params['momentum_rate'], optimizer_params['beta']),
                      prior_prec=prior_vector,
                      s_init=optimizer_params['s_init'],
                      decay_params=optimizer_params['decay_params'],
                      num_samples=optimizer_params['train_mc_samples'],
                      L=optimizer_params['L'],
                      train_set_size=train_set_size)

    def predict_fn(x, mc_samples):
        noise = optimizer.distribution().rsample(mc_samples).t()
        if use_cuda:
            noise = noise.cuda()
        preds = model(x, noise, False)
        if output_size == 1:
            preds = preds.reshape(mc_samples, -1)
        return preds

    def kl_fn():
        return optimizer.kl_divergence()

    return model, predict_fn, kl_fn, closure_factory, optimizer

#########################
### SLANG Experiments ###
#########################


@ExperimentFunction(show=plot_objective)
def slang_base(data_set='australian_presplit',
               model_params=defaults.MLP_CLASS_PARAMETERS,
               optimizer_params=defaults.SLANG_PARAMETERS,
               objective='avneg_loglik_bernoulli',
               metrics=['avneg_elbo_bernoulli', 'pred_avneg_loglik_bernoulli', 'pred_avneg_loglik_bernoulli', 'sigmoid_predictive_accuracy'],
               normalize={'x': False, 'y': False},
               save_params=defaults.SAVE_PARAMETERS,
               seed=123,
               use_cuda=torch.cuda.is_available(),
               init_hook=None,
               iter_hook=None,
               end_hook=None):

    set_seeds(seed)
    data = Dataset(data_set=data_set, data_folder=DEFAULT_DATA_FOLDER)
    train_set_size=data.get_train_size()

    model, predict_fn, kl_fn, closure_factory, optimizer = init_slang_experiment(data, model_params, optimizer_params, train_set_size=train_set_size, use_cuda=use_cuda)

    if init_hook is not None:
        init_hook(model, optimizer)

    results_dict = run_experiment(data, model, model_params, predict_fn, kl_fn, optimizer, optimizer_params, objective, metrics, closure_factory, normalize, save_params, seed, use_cuda, iter_hook)

    if end_hook is not None:
        end_hook(results_dict, model, optimizer)

    return results_dict

@ExperimentFunction()
def slang_cv(data_set='australian_presplit',
             n_splits=10,
             model_params=defaults.MLP_CLASS_PARAMETERS,
             optimizer_params=defaults.SLANG_PARAMETERS,
             objective='avneg_loglik_bernoulli',
             metrics=['avneg_elbo_bernoulli', 'pred_avneg_loglik_bernoulli', 'sigmoid_predictive_accuracy'],
             normalize={'x': False, 'y': False},
             save_params=defaults.SAVE_PARAMETERS,
             seed=123,
             use_cuda=torch.cuda.is_available()):

    set_seeds(seed)
    data = DatasetCV(data_set=data_set, n_splits=n_splits, seed=seed, data_folder=DEFAULT_DATA_FOLDER)

    results_dict = run_cv_experiment(data, n_splits, init_slang_experiment, model_params, optimizer_params, objective, metrics, normalize, save_params, seed, use_cuda)

    return results_dict


####################################
### Register Experiment Variants ###
####################################

def slang_binclass():
    # set the model parameters
    model_params = defaults.MLP_CLASS_PARAMETERS.copy()
    model_params['hidden_sizes'] = [] # No hidden layers
    model_params['prior_prec'] = 1e-5
    model_params['bias_prior_precision'] = 1e-6
    model_params['noise_precision'] = None # Classification Problem

    optimizer_params = {
        'num_epochs': 500,
        'batch_size': 32,
        'learning_rate': 0.05,
        'momentum_rate': 0.9,
        'train_mc_samples': 20,
        'eval_mc_samples': 20,
        'beta': 0.95,
        'decay_params': { 'learning_rate': 0.51, 'beta': 0.51 },
        's_init': 1,
        'L': 1 }

    # Create the experiment and register it.
    slang_base.add_variant("logistic regression", model_params=model_params, optimizer_params=optimizer_params)
    optimizer_params["L"] = 5
    slang_base.add_variant("logistic regression, L = 5", model_params=model_params, optimizer_params=optimizer_params)

def slang_regression():
    # set the model parameters
    model_params = defaults.MLP_REG_PARAMETERS

    opt_params = { 'num_epochs': 500,
                   'batch_size': 32,
                   'learning_rate': 0.05,
                   'momentum_rate': 0.9,
                   'train_mc_samples': 10,
                   'eval_mc_samples': 10,
                   's_init': 1,
                   'decay_params': { 'learning_rate': 0, 'beta': 0 },
                   'L': 1,
                   'beta': 0.95 }

    metrics = metric_factory.BAYES_REG
    objective = 'avneg_loglik_gaussian'

    # Create the experiment and register it.
    slang_base.add_variant("bnn regression", data_set="wine1", model_params=model_params, optimizer_params=opt_params, metrics=metrics, objective=objective)
    slang_cv.add_variant("bnn regression cv", data_set="wine1", n_splits=5, model_params=model_params, optimizer_params=opt_params, metrics=metrics, objective=objective)


####################################
### All Load Experiment Variants ###
####################################

def load_experiments():
    slang_binclass()
    slang_regression()

load_experiments()
