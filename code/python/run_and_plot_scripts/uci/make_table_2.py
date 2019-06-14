# Print to file

import numpy as np
import pandas as pd

import experiments.uci.uci_experiments as uci

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

def format_result(datasets, slang_loglik, slang_rmse, bbb_loglik, bbb_rmse, data_set, result_slang, result_bbb):
    format_str = '{:.2f} $\pm$ {:.2f}'
    datasets.append(data_set)
    slang_rmse.append(format_str.format(np.mean(result_slang['rmse']), np.std(result_slang['rmse'])/np.sqrt(20)))
    bbb_rmse.append(format_str.format(np.mean(result_bbb['rmse']), np.std(result_bbb['rmse'])/np.sqrt(20)))

    slang_loglik.append(format_str.format(np.mean(result_slang['loglik']), np.std(result_slang['loglik'])/np.sqrt(20)))
    bbb_loglik.append(format_str.format(np.mean(result_bbb['loglik']), np.std(result_bbb['loglik'])/np.sqrt(20)))

    return datasets, slang_rmse, bbb_rmse, slang_loglik, bbb_loglik


datasets = []
slang_rmse = []
bbb_rmse = []
slang_loglik = []
bbb_loglik = []

# boston

from experiments.uci.uci_slang_experiments_bo_boston import experiment_name, variants
experiment_base = uci.uci_slang_bo
result_slang = get_result(experiment_base, experiment_name, variants)

from experiments.uci.uci_bbb_experiments_bo_boston import experiment_name, variants
experiment_base = uci.uci_bbb_bo
result_bbb = get_result(experiment_base, experiment_name, variants)

datasets, slang_loglik, slang_rmse, bbb_loglik, bbb_rmse = format_result(datasets, slang_loglik, slang_rmse, bbb_loglik, bbb_rmse, data_set=variants[0][:-1], result_slang=result_slang, result_bbb=result_bbb)

# concrete

from experiments.uci.uci_slang_experiments_bo_concrete import experiment_name, variants
experiment_base = uci.uci_slang_bo
result_slang = get_result(experiment_base, experiment_name, variants)

from experiments.uci.uci_bbb_experiments_bo_concrete import experiment_name, variants
experiment_base = uci.uci_bbb_bo
result_bbb = get_result(experiment_base, experiment_name, variants)

datasets, slang_loglik, slang_rmse, bbb_loglik, bbb_rmse = format_result(datasets, slang_loglik, slang_rmse, bbb_loglik, bbb_rmse, data_set=variants[0][:-1], result_slang=result_slang, result_bbb=result_bbb)

# energy

from experiments.uci.uci_slang_experiments_bo_energy import experiment_name, variants
experiment_base = uci.uci_slang_bo
result_slang = get_result(experiment_base, experiment_name, variants)

from experiments.uci.uci_bbb_experiments_bo_energy import experiment_name, variants
experiment_base = uci.uci_bbb_bo
result_bbb = get_result(experiment_base, experiment_name, variants)

datasets, slang_loglik, slang_rmse, bbb_loglik, bbb_rmse = format_result(datasets, slang_loglik, slang_rmse, bbb_loglik, bbb_rmse, data_set=variants[0][:-1], result_slang=result_slang, result_bbb=result_bbb)

# kin8nm

from experiments.uci.uci_slang_experiments_bo_kin8nm import experiment_name, variants
experiment_base = uci.uci_slang_bo
result_slang = get_result(experiment_base, experiment_name, variants)

from experiments.uci.uci_bbb_experiments_bo_kin8nm import experiment_name, variants
experiment_base = uci.uci_bbb_bo
result_bbb = get_result(experiment_base, experiment_name, variants)

datasets, slang_loglik, slang_rmse, bbb_loglik, bbb_rmse = format_result(datasets, slang_loglik, slang_rmse, bbb_loglik, bbb_rmse, data_set=variants[0][:-1], result_slang=result_slang, result_bbb=result_bbb)

# naval

from experiments.uci.uci_slang_experiments_bo_naval import experiment_name, variants
experiment_base = uci.uci_slang_bo
result_slang = get_result(experiment_base, experiment_name, variants)

from experiments.uci.uci_bbb_experiments_bo_naval import experiment_name, variants
experiment_base = uci.uci_bbb_bo
result_bbb = get_result(experiment_base, experiment_name, variants)

datasets, slang_loglik, slang_rmse, bbb_loglik, bbb_rmse = format_result(datasets, slang_loglik, slang_rmse, bbb_loglik, bbb_rmse, data_set=variants[0][:-1], result_slang=result_slang, result_bbb=result_bbb)

# powerplant

from experiments.uci.uci_slang_experiments_bo_powerplant import experiment_name, variants
experiment_base = uci.uci_slang_bo
result_slang = get_result(experiment_base, experiment_name, variants)

from experiments.uci.uci_bbb_experiments_bo_powerplant import experiment_name, variants
experiment_base = uci.uci_bbb_bo
result_bbb = get_result(experiment_base, experiment_name, variants)

datasets, slang_loglik, slang_rmse, bbb_loglik, bbb_rmse = format_result(datasets, slang_loglik, slang_rmse, bbb_loglik, bbb_rmse, data_set=variants[0][:-1], result_slang=result_slang, result_bbb=result_bbb)

# wine

from experiments.uci.uci_slang_experiments_bo_wine import experiment_name, variants
experiment_base = uci.uci_slang_bo
result_slang = get_result(experiment_base, experiment_name, variants)

from experiments.uci.uci_bbb_experiments_bo_wine import experiment_name, variants
experiment_base = uci.uci_bbb_bo
result_bbb = get_result(experiment_base, experiment_name, variants)

datasets, slang_loglik, slang_rmse, bbb_loglik, bbb_rmse = format_result(datasets, slang_loglik, slang_rmse, bbb_loglik, bbb_rmse, data_set=variants[0][:-1], result_slang=result_slang, result_bbb=result_bbb)

# yacht

from experiments.uci.uci_slang_experiments_bo_yacht import experiment_name, variants
experiment_base = uci.uci_slang_bo
result_slang = get_result(experiment_base, experiment_name, variants)

from experiments.uci.uci_bbb_experiments_bo_yacht import experiment_name, variants
experiment_base = uci.uci_bbb_bo
result_bbb = get_result(experiment_base, experiment_name, variants)

datasets, slang_loglik, slang_rmse, bbb_loglik, bbb_rmse = format_result(datasets, slang_loglik, slang_rmse, bbb_loglik, bbb_rmse, data_set=variants[0][:-1], result_slang=result_slang, result_bbb=result_bbb)

# make table

results_dict = dict(data_set=datasets, slang_test_rmse=slang_rmse, bbb_test_rmse=bbb_rmse, slang_test_loglik=slang_loglik, bbb_test_loglik=bbb_loglik)
df_final = pd.DataFrame(results_dict)
df_final.to_csv("table_2.csv", index=False)
