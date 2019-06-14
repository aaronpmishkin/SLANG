import copy
import math
from experiments.mnist.slang_mnist_continue1 import *
import experiments.mnist.slang_mnist_continue as slang_mnist_continue

experiment_name = "slang_mnist_continue2"

data_set = 'mnist'


prior_prec = 1/math.exp(-2)**2

# set the model parameters
model_params = {
    'hidden_sizes': [400,400],
    'activation_function': 'relu',
    'prior_precision': prior_prec, # All values of L preferred this value
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
mnist_continue2 = slang_mnist_continue.slang_continue.add_variant(experiment_name,
                             data_set=data_set,
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

    best_variant = 'L_{}'.format(L)

    # Load state dicts
    try:
        exp = slang_mnist_continue.slang_continue.get_variant("slang_mnist_continue1").get_variant(best_variant)
        record = exp.get_latest_record()
        result = record.get_result()
        model_state_dict = result['model']
        optimizer_state_dict = result['optimizer']

    except:
        model_state_dict = None
        optimizer_state_dict = None
    finally:
        params = copy.deepcopy(mnist_continue2.get_args())
        params['optimizer_params']['decay_params'] = { 'learning_rate': decay, 'beta': decay }
        params['optimizer_params']['L'] = L
        name = 'L_{}'.format(L)
        variants.append(name)
        mnist_continue2.add_variant(name, optimizer_params=params['optimizer_params'], model_state_dict=model_state_dict, optimizer_state_dict=optimizer_state_dict)
