import copy
import experiments.uci.uci_experiments as uci
import experiments.uci.uci_default_parameters as defaults

experiment_name = "uci_slang_bo_powerplant"

lr = 0.005

optimizer_params = defaults.UCI_LARGE_SLANG_PARAMETERS

optimizer_params["learning_rate"] = lr
optimizer_params["beta"] = 1-lr

save_params = {
    'metric_history': False,
    'objective_history': True,
    'model': False,
    'optimizer': False }

param_bounds = {'log_noise_prec': (-3.1, -2.5), 'log_prior_prec': (-2, 2)}

from sklearn.gaussian_process.kernels import Matern
gp_params = {'kernel': Matern(nu=2.5, length_scale=[0.12, 1.25]), 'alpha': 1e-2}

# Create the base experiment and register it.
uci_slang_run = uci.uci_slang_bo.add_variant(experiment_name, optimizer_params=optimizer_params, param_bounds=param_bounds, gp_params=gp_params)



# Define the parameters of the experiment:
data_sets = ["powerplant" + str(i) for i in range(20)]

variants = []

for data_set in data_sets:
    params = copy.deepcopy(uci_slang_run.get_args())
    name = '{}'.format(data_set)
    variants.append(name)
    uci_slang_run.add_variant(name, data_set=data_set)
