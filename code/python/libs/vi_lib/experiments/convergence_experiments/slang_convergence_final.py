# @Author: aaronmishkin
# @Date:   18-09-19
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-09-19

import copy
import numpy as np

import experiments.default_parameters as defaults
import experiments.base.slang_experiments as slang

# set the model parameters
model_params = {
    'hidden_sizes': [50],
    'activation_function': 'relu',
    'prior_precision': 1e-5,
    'bias_prior_precision': 1e-6,
    'noise_precision': None } # Classification Problem

optimizer_params = {
    'num_epochs': 500,
    'batch_size': 0,
    'learning_rate': 0.05,
    'momentum_rate': 0.9,
    'train_mc_samples': 20,
    'eval_mc_samples': 20,
    'beta': 0.95,
    'decay_params': { 'learning_rate': 0, 'beta': 0 },
    's_init': 1,
    'L': 1 }

save_params = {
    'metric_history': True,
    'objective_history': True,
    'model': True,
    'optimizer': True,
    'every_iter': True }

slang_objective = 'avneg_loglik_bernoulli'
metrics = ['pred_avneg_loglik_bernoulli', 'sigmoid_predictive_accuracy', 'avneg_elbo_bernoulli']

experiment_name = "slang_convergence_final"
slang_convergence_base = slang.slang_base.add_variant(experiment_name,
                                                      data_set='',
                                                      model_params=model_params,
                                                      optimizer_params=optimizer_params,
                                                      objective=slang_objective,
                                                      metrics=metrics,
                                                      save_params=save_params,
                                                      use_cuda=False)

# Define the parameters of the experiment:
Ls = [1, 8, 16, 32, 64]

data_sets = ['australian_presplit', 'breastcancer_presplit', 'usps_3vs5']

# Picked by previous experiment.
prior_precisions = [8, 8, 32]

batch_sizes = [32, 32, 64]
variants = []
random_seeds = np.arange(1,11)


for seed in random_seeds:
    for L in Ls:
        for i, data_set in enumerate(data_sets):
            # Set the grids for each dataset.
            lr = 0.02154435

            params = copy.deepcopy(slang_convergence_base.get_args())
            params['optimizer_params']['learning_rate'] = lr
            params['model_params']['prior_precision'] = prior_precisions[i]
            params['optimizer_params']['batch_size'] = batch_sizes[i]
            params['optimizer_params']['beta'] = 1-lr
            params['optimizer_params']['L'] = L

            name = data_set + '_L_{}_lr_{}_seed_{}'.format(L, lr, seed)
            variants.append(name)
            slang_convergence_base.add_variant(name, data_set=data_set, model_params=params['model_params'], optimizer_params=params['optimizer_params'], seed=seed)
