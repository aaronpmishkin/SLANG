from lib.experiments.submitters import submit_python_jobs

# Import your experiment definitions here:
from experiments.mnist.slang_mnist_experiment import experiment_name, variants

# Write your call to the job submitter here:
submit_python_jobs(experiment_name, variants, method='SLANG_COMPLETE', cv=0)
