UCI_SMALL_BBB_PARAMETERS = {
    'num_epochs': 120,
    'batch_size': 10,
    'learning_rate': 0.01,
    'momentum_rate': 0.9,
    'train_mc_samples': 4,
    'eval_mc_samples': 20,
    'beta': 0.99 }

UCI_LARGE_BBB_PARAMETERS = {
    'num_epochs': 120,
    'batch_size': 100,
    'learning_rate': 0.01,
    'momentum_rate': 0.9,
    'train_mc_samples': 2,
    'eval_mc_samples': 20,
    'beta': 0.99 }

UCI_SMALL_SLANG_PARAMETERS = {
    'num_epochs': 120,
    'batch_size': 10,
    'learning_rate': 0.01,
    'momentum_rate': 0.9,
    'train_mc_samples': 4,
    'eval_mc_samples': 20,
    'beta': 0.99,
    'decay_params': {'learning_rate': 0.0, 'beta': 0.0 },
    's_init': 1,
    'L': 1 }

UCI_LARGE_SLANG_PARAMETERS = {
    'num_epochs': 120,
    'batch_size': 10,
    'learning_rate': 0.01,
    'momentum_rate': 0.9,
    'train_mc_samples': 2,
    'eval_mc_samples': 20,
    'beta': 0.99,
    'decay_params': {'learning_rate': 0.0, 'beta': 0.0 },
    's_init': 1,
    'L': 1 }