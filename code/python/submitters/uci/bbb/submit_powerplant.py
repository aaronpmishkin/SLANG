from lib.experiments.submitters import submit_python_jobs

# Powerplant
from experiments.uci.uci_bbb_experiments_bo_powerplant import experiment_name, variants
submit_python_jobs(experiment_name, variants, method='UCI_BBB_BO', cv=0)
