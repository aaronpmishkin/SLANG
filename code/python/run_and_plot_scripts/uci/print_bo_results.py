# Print to file
import sys
sys.stdout = open('results.txt','wt')

import experiments.uci.uci_experiments as uci
import numpy as np

def get_result(experiment_base, experiment_name, variants):
    rmse = []
    loglik = []

    for i, var in enumerate(variants):
        # Retrieve record
        exp = experiment_base.get_variant(experiment_name).get_variant(var)
        record = exp.get_latest_record()

        # Extract metrics
        if record.has_result():
            result = record.get_result()
            rmse.append(result['final_run']['final_metrics']['test_pred_rmse'][0])
            loglik.append(-result['final_run']['final_metrics']['test_pred_logloss'][0])
        else:
            RuntimeError("Missing result")

    return dict(loglik = loglik, rmse = rmse)

def print_mean_result(data_set, result_slang, result_bbb):
    format_str = '{:.2f} $\pm$ {:.2f}'
    print("Dataset: ", data_set)
    print("\nRMSE:", "\nSLANG:\t", format_str.format(np.mean(result_slang['rmse']), np.std(result_slang['rmse'])/np.sqrt(20)), "\nBBB:\t", format_str.format(np.mean(result_bbb['rmse']), np.std(result_bbb['rmse'])/np.sqrt(20)))
    print("\nLoglik:", "\nSLANG:\t", format_str.format(np.mean(result_slang['loglik']), np.std(result_slang['loglik'])/np.sqrt(20)), "\nBBB:\t", format_str.format(np.mean(result_bbb['loglik']), np.std(result_bbb['loglik'])/np.sqrt(20)))



# boston

from experiments.uci.uci_slang_experiments_bo_boston import experiment_name, variants
experiment_base = uci.uci_slang_bo
result_slang = get_result(experiment_base, experiment_name, variants)

from experiments.uci.uci_bbb_experiments_bo_boston import experiment_name, variants
experiment_base = uci.uci_bbb_bo
result_bbb = get_result(experiment_base, experiment_name, variants)

print_mean_result(data_set=variants[0][:-1], result_slang=result_slang, result_bbb=result_bbb)
print("\n\n\n")

# concrete

from experiments.uci.uci_slang_experiments_bo_concrete import experiment_name, variants
experiment_base = uci.uci_slang_bo
result_slang = get_result(experiment_base, experiment_name, variants)

from experiments.uci.uci_bbb_experiments_bo_concrete import experiment_name, variants
experiment_base = uci.uci_bbb_bo
result_bbb = get_result(experiment_base, experiment_name, variants)

print_mean_result(data_set=variants[0][:-1], result_slang=result_slang, result_bbb=result_bbb)
print("\n\n\n")

# energy

from experiments.uci.uci_slang_experiments_bo_energy import experiment_name, variants
experiment_base = uci.uci_slang_bo
result_slang = get_result(experiment_base, experiment_name, variants)

from experiments.uci.uci_bbb_experiments_bo_energy import experiment_name, variants
experiment_base = uci.uci_bbb_bo
result_bbb = get_result(experiment_base, experiment_name, variants)

print_mean_result(data_set=variants[0][:-1], result_slang=result_slang, result_bbb=result_bbb)
print("\n\n\n")

# kin8nm

from experiments.uci.uci_slang_experiments_bo_kin8nm import experiment_name, variants
experiment_base = uci.uci_slang_bo
result_slang = get_result(experiment_base, experiment_name, variants)

from experiments.uci.uci_bbb_experiments_bo_kin8nm import experiment_name, variants
experiment_base = uci.uci_bbb_bo
result_bbb = get_result(experiment_base, experiment_name, variants)

print_mean_result(data_set=variants[0][:-1], result_slang=result_slang, result_bbb=result_bbb)
print("\n\n\n")

# naval

from experiments.uci.uci_slang_experiments_bo_naval import experiment_name, variants
experiment_base = uci.uci_slang_bo
result_slang = get_result(experiment_base, experiment_name, variants)

from experiments.uci.uci_bbb_experiments_bo_naval import experiment_name, variants
experiment_base = uci.uci_bbb_bo
result_bbb = get_result(experiment_base, experiment_name, variants)

print_mean_result(data_set=variants[0][:-1], result_slang=result_slang, result_bbb=result_bbb)
print("\n\n\n")

# powerplant

from experiments.uci.uci_slang_experiments_bo_powerplant import experiment_name, variants
experiment_base = uci.uci_slang_bo
result_slang = get_result(experiment_base, experiment_name, variants)

from experiments.uci.uci_bbb_experiments_bo_powerplant import experiment_name, variants
experiment_base = uci.uci_bbb_bo
result_bbb = get_result(experiment_base, experiment_name, variants)

print_mean_result(data_set=variants[0][:-1], result_slang=result_slang, result_bbb=result_bbb)
print("\n\n\n")

# wine

from experiments.uci.uci_slang_experiments_bo_wine import experiment_name, variants
experiment_base = uci.uci_slang_bo
result_slang = get_result(experiment_base, experiment_name, variants)

from experiments.uci.uci_bbb_experiments_bo_wine import experiment_name, variants
experiment_base = uci.uci_bbb_bo
result_bbb = get_result(experiment_base, experiment_name, variants)

print_mean_result(data_set=variants[0][:-1], result_slang=result_slang, result_bbb=result_bbb)
print("\n\n\n")

# yacht

from experiments.uci.uci_slang_experiments_bo_yacht import experiment_name, variants
experiment_base = uci.uci_slang_bo
result_slang = get_result(experiment_base, experiment_name, variants)

from experiments.uci.uci_bbb_experiments_bo_yacht import experiment_name, variants
experiment_base = uci.uci_bbb_bo
result_bbb = get_result(experiment_base, experiment_name, variants)

print_mean_result(data_set=variants[0][:-1], result_slang=result_slang, result_bbb=result_bbb)
print("\n\n\n")
