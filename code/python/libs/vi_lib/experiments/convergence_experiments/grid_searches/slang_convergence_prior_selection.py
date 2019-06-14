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
    'num_epochs': 5000,
    'batch_size': 0,
    'learning_rate': 0.05,
    'momentum_rate': 0.9,
    'train_mc_samples': 12,
    'eval_mc_samples': 20,
    'beta': 0.95,
    'decay_params': { 'learning_rate': 0, 'beta': 0 },
    's_init': 1,
    'L': 1 }

save_params = {
    'metric_history': True,
    'objective_history': True,
    'model': True,
    'optimizer': True }

slang_objective = 'avneg_loglik_bernoulli'
metrics = ['pred_avneg_loglik_bernoulli', 'sigmoid_predictive_accuracy', 'avneg_elbo_bernoulli']

experiment_name = "convergence_prior_grid"
slang_convergence_base = slang.slang_cv.add_variant(experiment_name,
                                                      n_splits=5,
                                                      data_set='',
                                                      model_params=model_params,
                                                      optimizer_params=optimizer_params,
                                                      objective=slang_objective,
                                                      metrics=metrics,
                                                      save_params=save_params,
                                                      use_cuda=False)

# Define the parameters of the experiment:
Ls = [1, 10]

data_sets = ['australian_presplit', 'breastcancer_presplit', 'usps_3vs5']

# Consider a grid of prior precisions
prior_precisions = [0.001, 0.01, 0.1, 1, 8, 32, 64, 128, 512]

batch_sizes = [32, 32, 64]
variants = []
random_seeds = [1]

# Learning Rates: Use these to select prior precisions.
# Obtained by inspecting convergence of sample runs.
# Australian
# L = 1  : 0.0045
# L = 10 : 0.00215
# Breast Cancer
# L = 1  : 0.01
# L = 10 : 0.003
# usps_3vs5
# L = 1  : 0.01
# L = 10 : 0.01


for seed in random_seeds:
    for L in Ls:
        for i, data_set in enumerate(data_sets):
            if data_set == "australian_presplit":
                # We consider an order of magnitude smaller learnign rates.
                lr = 0.0045 if L == 1 else 0.00215
            elif data_set == "breastcancer_presplit":
                lr = 0.01 if L == 1 else 0.003
            else:
                lr = 0.01

            for prior_prec in prior_precisions:
                params = copy.deepcopy(slang_convergence_base.get_args())
                params['optimizer_params']['learning_rate'] = lr
                params['model_params']['prior_precision'] = prior_prec
                params['optimizer_params']['batch_size'] = batch_sizes[i]
                params['optimizer_params']['beta'] = 1-lr
                params['optimizer_params']['L'] = L

                name = data_set + '_prior_prec_{}_L_{}_lr_{}_seed_{}'.format(prior_prec, L, lr, seed)
                variants.append(name)
                slang_convergence_base.add_variant(name, data_set=data_set, model_params=params['model_params'], optimizer_params=params['optimizer_params'], seed=seed)
