# Experiment Bases
from experiments.base.bayes_by_backprop import *
from experiments.base.bayes_by_backprop_decay import *
from experiments.base.bbb_copy_slang import *
from experiments.base.slang_experiments import *

# MNIST Classification Experiment
from experiments.mnist.slang_mnist_val import *
from experiments.mnist.slang_mnist_continue import *
from experiments.mnist.slang_mnist_complete import *
from experiments.mnist.slang_mnist_continue1 import *
from experiments.mnist.slang_mnist_continue2 import *
from experiments.mnist.slang_mnist_continue3 import *
from experiments.mnist.slang_mnist_experiment import *

# UCI Regression Experiment
from experiments.uci.uci_experiments import *
from experiments.uci.uci_slang_experiments_bo_boston import *
from experiments.uci.uci_slang_experiments_bo_concrete import *
from experiments.uci.uci_slang_experiments_bo_energy import *
from experiments.uci.uci_slang_experiments_bo_kin8nm import *
from experiments.uci.uci_slang_experiments_bo_naval import *
from experiments.uci.uci_slang_experiments_bo_powerplant import *
from experiments.uci.uci_slang_experiments_bo_wine import *
from experiments.uci.uci_slang_experiments_bo_yacht import *
from experiments.uci.uci_bbb_experiments_bo_boston import *
from experiments.uci.uci_bbb_experiments_bo_concrete import *
from experiments.uci.uci_bbb_experiments_bo_energy import *
from experiments.uci.uci_bbb_experiments_bo_kin8nm import *
from experiments.uci.uci_bbb_experiments_bo_naval import *
from experiments.uci.uci_bbb_experiments_bo_powerplant import *
from experiments.uci.uci_bbb_experiments_bo_wine import *
from experiments.uci.uci_bbb_experiments_bo_yacht import *

# Convergence Experiments:
    # Grid Searches:
from experiments.convergence_experiments.grid_searches.slang_convergence_lr_selection import *
from experiments.convergence_experiments.grid_searches.bbb_convergence_lr_selection import *
from experiments.convergence_experiments.grid_searches.slang_convergence_prior_selection import *
from experiments.convergence_experiments.grid_searches.bbb_convergence_prior_selection import *
    # Final Experiments
from experiments.convergence_experiments.slang_convergence_final import *
from experiments.convergence_experiments.bbb_convergence_final import *




# from experiments.qualitative_1d_covariance_experiment import *
