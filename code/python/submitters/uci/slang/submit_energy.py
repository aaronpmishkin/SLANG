from lib.experiments.submitters import submit_python_jobs

# Energy
from experiments.uci.uci_slang_experiments_bo_energy import experiment_name, variants
submit_python_jobs(experiment_name, variants, method='UCI_SLANG_BO', cv=0)
