import experiments.default_parameters as defaults
import experiments.base.slang_experiments as slang

experiment_name = "mnist_shallownet"

data_set = 'mnist'

# set the model parameters
model_params = {
    'hidden_sizes': [100],
    'activation_function': 'relu',
    'prior_precision': 1e-5,
    'bias_prior_precision': 1e-6,
    'noise_precision': None } # Classification Problem

# set the optimizer parameters
optimizer_params = {
    'num_epochs': 1,
    'batch_size': 128,
    'learning_rate': 0.05,
    'momentum_rate': 0.9,
    'train_mc_samples': 1,
    'eval_mc_samples': 1,
    'beta': 0.95,
    'decay_params': { 'learning_rate': 0.51, 'beta': 0.51 },
    's_init': 1,
    'L': 1 }

objective = 'avneg_loglik_categorical'
metrics = ['pred_avneg_loglik_categorical', 'softmax_predictive_accuracy']

save_params = defaults.SAVE_PARAMETERS
# Create the base experiment and register it.
experiment_shallow = slang.slang_base.add_variant(experiment_name,
                             data_set=data_set,
                             model_params=model_params,
                             optimizer_params=optimizer_params,
                             objective=objective,
                             metrics=metrics,
                             save_params=save_params)



# Try running experiment
record = experiment_shallow.run()