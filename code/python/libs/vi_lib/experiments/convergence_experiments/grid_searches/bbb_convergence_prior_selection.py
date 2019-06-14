# @Author: aaronmishkin
# @Date:   18-09-19
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-09-19

import copy
import numpy as np

import experiments.default_parameters as defaults
import experiments.base.bayes_by_backprop as bbb

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
    'beta': 0.95}

save_params = {
    'metric_history': True,
    'objective_history': True,
    'model': True,
    'optimizer': True }

bbb_objective = 'avneg_elbo_bernoulli'
metrics = ['pred_avneg_loglik_bernoulli', 'sigmoid_predictive_accuracy', 'avneg_elbo_bernoulli']

bbb_experiment_name = "bbb_convergence_prior_grid"
bbb_convergence_base = bbb.bbb_cv.add_variant(bbb_experiment_name,
                                                n_splits=5,
                                                data_set='',
                                                model_params=model_params,
                                                optimizer_params=optimizer_params,
                                                objective=bbb_objective,
                                                metrics=metrics,
                                                save_params=save_params,
                                                use_cuda=False)

# Define the parameters of the experiment:
decay_rates =  [0]
lrs = np.logspace(-4, -0.6, 10)

data_sets = ['australian_presplit', 'breastcancer_presplit', 'usps_3vs5']

# These might need to be adjusted.
# Currently using the same parameters as in the logistic regression experiments.
prior_precisions = [0.001, 0.01, 0.1, 1, 8, 32, 64, 128, 512]
batch_sizes = [32, 32, 64]
bbb_variants = []
random_seeds = [1]


# BBB needs more epochs for the ELBO to converge on every dataset. Note that
# LogLoss converges much faster.
# Previous run used 2000 epochs.

# Learning Rates: Use these to select prior precisions.
# Obtained by inspecting convergence of sample runs.
# Australian Scale  : 0.003
# Breast Cancer     : 0.0032
# USPS 3vs5         : 0.0015



# Trying the Default Adam Learning Rates
for seed in random_seeds:
    for i, data_set in enumerate(data_sets):
        if data_set == "australian_presplit":
            lr = 0.0032
        elif data_set == "breastcancer_presplit":
            lr = 0.0032
        else:
            lr = 0.0015

        for prior_prec in prior_precisions:
            params = copy.deepcopy(bbb_convergence_base.get_args())
            params['optimizer_params']['learning_rate'] = lr
            params['model_params']['prior_precision'] = prior_prec
            params['optimizer_params']['batch_size'] = batch_sizes[i]
            # Always use the default Adam beta.
            params['optimizer_params']['beta'] = 0.999
            name = data_set + '_prior_prec_{}_lr_{}_seed_{}'.format(prior_prec, lr, seed)

            bbb_variants.append(name)
            bbb_convergence_base.add_variant(name, data_set=data_set, model_params= params['model_params'], optimizer_params=params['optimizer_params'], seed=seed)
