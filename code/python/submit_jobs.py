# @Author: amishkin
# @Date:   18-09-07
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-09-07
from lib.experiments.submitters import profiled_submit_raiden_jobs, profiled_submit_python_jobs, submit_raiden_jobs, submit_python_jobs


# Import your experiment definitions here:
from experiments.convergence_experiments.slang_convergence import experiment_name, variants

# Write your call to the job submitter here:
# submit_python_jobs(experiment_name, variants, method='SLANG', cv=0)
