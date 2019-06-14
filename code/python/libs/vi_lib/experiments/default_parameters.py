# @Author: amishkin
# @Date:   18-08-17
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-08-30

# =================================
# ======== Save Parameters ========
# =================================

SAVE_PARAMETERS = {
    'metric_history': True,
    'objective_history': True,
    'model': True,
    'optimizer': True }

# ==================================
# ====== Optimizer Parameters ======
# ==================================

OPTIMIZER_PARAMETERS = {
    'num_epochs': 500,
    'batch_size': 32,
    'learning_rate': 0.1,
    'momentum_rate': 0.9,
    'decay_params': {'learning_rate': 0.51 } }

VI_PARAMETERS = {
    'num_epochs': 500,
    'batch_size': 32,
    'learning_rate': 0.1,
    'momentum_rate': 0.9,
    'train_mc_samples': 10,
    'eval_mc_samples': 10,
    'beta': 0.99,
    'decay_params': {'learning_rate': 0.51, 'beta': 0.51 } }

SLANG_PARAMETERS = {
    'num_epochs': 500,
    'batch_size': 32,
    'learning_rate': 0.05,
    'momentum_rate': 0.9,
    'train_mc_samples': 1,
    'eval_mc_samples': 10,
    'beta': 0.95,
    'decay_params': {'learning_rate': 0.51, 'beta': 0.51 },
    's_init': 1,
    'L': 1 }

BBB_COPYSLANG_PARAMETERS = {
    'num_epochs': 500,
    'batch_size': 32,
    'learning_rate': 0.1,
    'momentum_rate': 0.9,
    'train_mc_samples': 10,
    'eval_mc_samples': 10,
    'beta': 0.99,
    's_init': 1,
    'decay_params': {'learning_rate': 0} }

# ==================================
# ======== Model Parameters ========
# ==================================

MLP_REG_PARAMETERS = {
    'hidden_sizes': [64],
    'activation_function': 'relu',
    'prior_precision': 1.0,
    'bias_prior_precision': 1e-6,
    'noise_precision': 1.0 }

MLP_CLASS_PARAMETERS = {
    'hidden_sizes': [64],
    'activation_function': 'relu',
    'prior_precision': 1.0,
    'bias_prior_precision': 1e-6 }
