import copy
import math
import experiments.mnist.slang_mnist_complete as slang_mnist_complete

experiment_name = "slang_mnist_experiment"
val_data_set = 'mnist_val'
continue_data_set = 'mnist'
prior_prec = 1/math.exp(-2)**2

# set the model parameters
model_params = {
    'hidden_sizes': [400,400],
    'activation_function': 'relu',
    'prior_precision': prior_prec, # All values of L preferred this value
    'bias_prior_precision': 1e-6,
    'noise_precision': None } # Classification Problem

optimizer_params = {
    'num_epochs': 100,            # should be 100
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
mnist_complete = slang_mnist_complete.slang_complete.add_variant(experiment_name,
                             val_data_set=val_data_set,
                             continue_data_set=continue_data_set,
                             model_params=model_params,
                             optimizer_params=optimizer_params,
                             objective=objective,
                             metrics=metrics,
                             save_params=save_params)



# Define the parameters of the experiment:
Ls = [1, 2, 4, 8, 16, 32]


variants = []

for L in Ls:
    if L==1 or L==2 or L==8 or L==32:
        decay = 0.6
    elif L==4:
        decay = 0.58
    elif L==16:
        decay = 0.56
    params = copy.deepcopy(mnist_complete.get_args())
    params['optimizer_params']['decay_params'] = { 'learning_rate': decay, 'beta': decay }
    params['optimizer_params']['L'] = L
    name = 'L_{}'.format(L)
    variants.append(name)
    mnist_complete.add_variant(name, optimizer_params=params['optimizer_params'])
