from lib.experiments.submitters import submit_python_jobs

# Import your experiment definitions here:
from experiments.mnist.slang_mnist_continue2 import experiment_name, variants

# Write your call to the job submitter here:
submit_python_jobs(experiment_name, variants, method='SLANG_CONTINUE', cv=0)
