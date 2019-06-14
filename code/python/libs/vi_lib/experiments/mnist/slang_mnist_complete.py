import torch
from artemis.experiments import ExperimentFunction

from lib.data.datasets import Dataset, DEFAULT_DATA_FOLDER
from lib.experiments.experiment import run_experiment
from lib.plots.basics import plot_objective
from lib.utilities.general_utilities import set_seeds

import pdb

import experiments.default_parameters as defaults
from experiments.base.slang_experiments import init_slang_experiment

MNIST_OPTIM_PARAM = {
    'num_epochs': 100,
    'batch_size': 200,
    'learning_rate': 0.1,
    'momentum_rate': 0.9,
    'train_mc_samples': 4,
    'eval_mc_samples': 10,
    'beta': 0.9,
    'decay_params': { 'learning_rate': 0.51, 'beta': 0.51 },
    's_init': 1,
    'L': 1 }

MNIST_MODEL_PARAM = {
    'hidden_sizes': [400,400],
    'activation_function': 'relu',
    'prior_precision': 5.0,
    'bias_prior_precision': 1e-6,
    'noise_precision': None } # Classification Problem


def load_dicts(model, optimizer, dict, new_train_set_size, use_cuda):
    model.load_state_dict(dict['model'])
    if use_cuda:
        model = model.cuda()
    optimizer.load_state_dict(dict['optimizer'])
    if use_cuda:
        optimizer.defaults['train_set_size'] = new_train_set_size
        optimizer.defaults['prior_prec'] = optimizer.defaults['prior_prec'].cuda()
        optimizer.state['mean'] = optimizer.state['mean'].cuda()
        optimizer.state['momentum_grad'] = optimizer.state['momentum_grad'].cuda()
        optimizer.state['prec_diag'] = optimizer.state['prec_diag'].cuda()
        optimizer.state['prec_factor'] = optimizer.state['prec_factor'].cuda()

    return model, optimizer

#########################
### SLANG Experiments ###
#########################

@ExperimentFunction(show=plot_objective)
def slang_complete(val_data_set='mnist_val',
                   continue_data_set='mnist',
                   model_params=MNIST_MODEL_PARAM,
                   optimizer_params=MNIST_OPTIM_PARAM,
                   continue_train_set_size=60000,
                   num_continues = 3,
                   objective='avneg_loglik_categorical',
                   metrics=['pred_avneg_loglik_categorical', 'softmax_predictive_accuracy', 'avneg_elbo_categorical'],
                   normalize={'x': False, 'y': False},
                   save_params=defaults.SAVE_PARAMETERS,
                   val_seed=123,
                   continue_seeds=[123, 123, 123],
                   use_cuda=torch.cuda.is_available(),
                   init_hook=None,
                   iter_hook=None,
                   end_hook=None):


    ######################################
    ### Run Validation-set Experiment: ###
    ######################################

    set_seeds(val_seed)
    data = Dataset(data_set=val_data_set, data_folder=DEFAULT_DATA_FOLDER)
    train_set_size=data.get_train_size()
    model, predict_fn, kl_fn, closure_factory, optimizer = init_slang_experiment(data, model_params, optimizer_params, train_set_size=train_set_size, use_cuda=use_cuda)

    if init_hook is not None:
        init_hook(model, optimizer)
    results_dict = run_experiment(data, model, model_params, predict_fn, kl_fn, optimizer, optimizer_params, objective, metrics, closure_factory, normalize, save_params, val_seed, use_cuda, iter_hook)

    if end_hook is not None:
        end_hook(results_dict, model, optimizer)

    #####################################
    ### Run Continuation Experiments: ###
    #####################################

    for continuation in range(num_continues):

        set_seeds(continue_seeds[continuation])

        data = Dataset(data_set=continue_data_set, data_folder=DEFAULT_DATA_FOLDER)
        train_set_size=data.get_train_size()

        model, predict_fn, kl_fn, closure_factory, optimizer = init_slang_experiment(data, model_params, optimizer_params, train_set_size=train_set_size, use_cuda=use_cuda)
        
        results_dict['optimizer']['state']['mean'] = results_dict['optimizer']['state']['mean'].detach()
        model, optimizer = load_dicts(model, optimizer, results_dict, continue_train_set_size, use_cuda)

        if init_hook is not None:
            init_hook(model, optimizer)

        results_dict = run_experiment(data, model, model_params, predict_fn, kl_fn, optimizer, optimizer_params, objective, metrics, closure_factory, normalize, save_params, continue_seeds[continuation], use_cuda, iter_hook)

        if end_hook is not None:
            end_hook(results_dict, model, optimizer)


    return results_dict
