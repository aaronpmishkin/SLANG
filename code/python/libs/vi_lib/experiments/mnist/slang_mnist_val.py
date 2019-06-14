import copy
import numpy as np
import experiments.base.slang_experiments as slang

experiment_name = "slang_mnist_val"

data_set = 'mnist_val'

# set the model parameters
model_params = {
    'hidden_sizes': [400,400],
    'activation_function': 'relu',
    'prior_precision': None, # Will be tuned
    'bias_prior_precision': 1e-6,
    'noise_precision': None } # Classification Problem

optimizer_params = {
    'num_epochs': 100,
    'batch_size': 200,
    'learning_rate': 0.1,
    'momentum_rate': 0.9,
    'train_mc_samples': 4,
    'eval_mc_samples': 10,
    'beta': 0.9,
    'decay_params': { 'learning_rate': None, 'beta': None }, # Will be tuned
    's_init': 1,
    'L': None } # Multiple values will be run

save_params = {
    'metric_history': True,
    'objective_history': True,
    'model': True,
    'optimizer': True }

objective = 'avneg_loglik_categorical'
metrics = ['pred_avneg_loglik_categorical', 'softmax_predictive_accuracy', 'avneg_elbo_categorical']

# Create the base experiment and register it.
mnist_base = slang.slang_base.add_variant(experiment_name,
                             data_set=data_set,
                             model_params=model_params,
                             optimizer_params=optimizer_params,
                             objective=objective,
                             metrics=metrics,
                             save_params=save_params)



# Define the parameters of the experiment:
Ls = [1, 2, 4, 8, 16, 32]

decays = [0.52, 0.54, 0.56, 0.58, 0.60]

log_sigmas_np = - np.array([0,1,2])
prior_precs_np = 1/np.exp(log_sigmas_np)**2
prior_precs = list(prior_precs_np)

variants = []

for L in Ls:
    for decay in decays:
        for prior_prec in prior_precs:
            params = copy.deepcopy(mnist_base.get_args())
            params['optimizer_params']['decay_params'] = { 'learning_rate': decay, 'beta': decay }
            params['optimizer_params']['L'] = L
            params['model_params']['prior_precision'] = prior_prec
            name = 'L_{}_pp_{:.2f}_dec_{:.2f}'.format(L, prior_prec, decay)
            variants.append(name)
            mnist_base.add_variant(name, model_params=params['model_params'], optimizer_params=params['optimizer_params'])
