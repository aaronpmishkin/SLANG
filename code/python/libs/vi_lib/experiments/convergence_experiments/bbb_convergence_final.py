# @Author: aaronmishkin
# @Date:   18-09-19
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-09-19

import copy
import numpy as np

import experiments.default_parameters as defaults
import experiments.base.bbb_copy_slang as bbb

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
    's_init': 1}

save_params = {
    'metric_history': True,
    'objective_history': True,
    'model': True,
    'optimizer': True,
    'every_iter': True }

bbb_objective = 'avneg_elbo_bernoulli'
metrics = ['pred_avneg_loglik_bernoulli', 'sigmoid_predictive_accuracy', 'avneg_elbo_bernoulli']

experiment_name = "bbb_convergence_final"
bbb_convergence_base = bbb.bbb_copy_slang.add_variant(experiment_name,
                                                data_set='',
                                                model_params=model_params,
                                                optimizer_params=optimizer_params,
                                                objective=bbb_objective,
                                                metrics=metrics,
                                                save_params=save_params,
                                                use_cuda=False)

# Define the parameters of the experiment:
decay_rates =  [0]


data_sets = ['australian_presplit', 'breastcancer_presplit', 'usps_3vs5']

# Picked by previous experiment.
prior_precisions = [8, 8, 32]

batch_sizes = [32, 32, 64]
variants = []
random_seeds = np.arange(1,11)

# Trying the Default Adam Learning Rates
for seed in random_seeds:
    for i, data_set in enumerate(data_sets):

        lr = 0.01

        params = copy.deepcopy(bbb_convergence_base.get_args())
        params['optimizer_params']['learning_rate'] = lr
        params['model_params']['prior_precision'] = prior_precisions[i]
        params['optimizer_params']['batch_size'] = batch_sizes[i]
        # Always use the default Adam beta.
        params['optimizer_params']['beta'] = 0.999
        params['seed'] = seed
        name = data_set + '_lr_{}_seed_{}'.format(lr, seed)
        variants.append(name)
        bbb_convergence_base.add_variant(name, data_set=data_set, model_params= params['model_params'], optimizer_params=params['optimizer_params'], seed=seed)
