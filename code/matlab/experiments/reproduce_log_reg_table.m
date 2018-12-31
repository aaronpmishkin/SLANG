% @Author: aaronmishkin
% @Date:   18-09-30
% @Email:  amishkin@cs.ubc.ca
% @Last modified by:   aaronmishkin
% @Last modified time: 18-09-30

% =========================================================================================
% Reproduces the logistic regression results in Tables 1, 6, and 7. The final distributions
% found for the first split of each dataset are also visualized in Figures 1, and 7.
% =========================================================================================

addpath(genpath('.'))
mkdir('./experiments/data')
output_dir = './experiments/data/final_log_reg_table';
mkdir(output_dir)

epoch_lists = {[10000]};

L = 0;   % N/A for EF and mf-EF
K = 0; % N/A for EF and mf-EF

num_samples = 12;
beta = 0.05;
alpha = 0.05;
decay_rate = 0.51;   % controls the rate at which alpha and beta decay during training.

random_split = 1;
num_restarts = 20;

% =================================
% ========== CVI Methods ==========
% =================================

datasets = {'australian_scale'};
M_lists = {[32]};

methods = {'EF', 'mf-EF'}
run_experiment(datasets, methods, M_lists, epoch_lists, L, K, alpha, beta, decay_rate, num_samples, num_restarts, random_split, 0, output_dir)

methods = {'Hessian', 'mf-Hessian'}
run_experiment(datasets, methods, M_lists, epoch_lists, L, K, alpha, beta, decay_rate, num_samples, num_restarts, random_split, 0, output_dir)


datasets = {'breast_cancer_scale'};
M_lists = {[32]};

methods = {'EF', 'mf-EF'}
run_experiment(datasets, methods, M_lists, epoch_lists, L, K, alpha, beta, decay_rate, num_samples, num_restarts, random_split, 0, output_dir)

methods = {'Hessian', 'mf-Hessian'}
run_experiment(datasets, methods, M_lists, epoch_lists, L, K, alpha, beta, decay_rate, num_samples, num_restarts, random_split, 0, output_dir)


datasets = {'usps_3vs5'};
M_lists = {[64]};

methods = {'EF', 'mf-EF'}
run_experiment(datasets, methods, M_lists, epoch_lists, L, K, alpha, beta, decay_rate, num_samples, num_restarts, random_split, 0, output_dir)

methods = {'Hessian', 'mf-Hessian'}
run_experiment(datasets, methods, M_lists, epoch_lists, L, K, alpha, beta, decay_rate, num_samples, num_restarts, random_split, 0, output_dir)

datasets = {'a1a'};
M_lists = {[128]};
num_restarts = 1;

methods = {'EF', 'mf-EF'}
run_experiment(datasets, methods, M_lists, epoch_lists, L, K, alpha, beta, decay_rate, num_samples, num_restarts, random_split, 0, output_dir)

methods = {'Hessian', 'mf-Hessian'}
run_experiment(datasets, methods, M_lists, epoch_lists, L, K, alpha, beta, decay_rate, num_samples, num_restarts, random_split, 0, output_dir)

% =================================
% ============= SLANG =============
% =================================

num_restarts = 20;
datasets = {'australian_scale'};
M_lists = {[32]};

run_experiment(datasets, methods, M_lists, epoch_lists, 1, 0, alpha, beta, decay_rate, num_samples, num_restarts, random_split, output_dir)

run_experiment(datasets, methods, M_lists, epoch_lists, 2, 0, alpha, beta, decay_rate, num_samples, num_restarts, random_split, output_dir)

run_experiment(datasets, methods, M_lists, epoch_lists, 5, 0, alpha, beta, decay_rate, num_samples, num_restarts, random_split, output_dir)

run_experiment(datasets, methods, M_lists, epoch_lists, 10, 0, alpha, beta, decay_rate, num_samples, num_restarts, random_split, output_dir)

datasets = {'breast_cancer_scale'};
M_lists = {[32]};

run_experiment(datasets, methods, M_lists, epoch_lists, 1, 0, alpha, beta, decay_rate, num_samples, num_restarts, random_split, output_dir)

run_experiment(datasets, methods, M_lists, epoch_lists, 2, 0, alpha, beta, decay_rate, num_samples, num_restarts, random_split, output_dir)

run_experiment(datasets, methods, M_lists, epoch_lists, 5, 0, alpha, beta, decay_rate, num_samples, num_restarts, random_split, output_dir)

run_experiment(datasets, methods, M_lists, epoch_lists, 10, 0, alpha, beta, decay_rate, num_samples, num_restarts, random_split, output_dir)

datasets = {'usps_3vs5'};
M_lists = {[64]};

run_experiment(datasets, methods, M_lists, epoch_lists, 1, 0, alpha, beta, decay_rate, num_samples, num_restarts, random_split, output_dir)

run_experiment(datasets, methods, M_lists, epoch_lists, 2, 0, alpha, beta, decay_rate, num_samples, num_restarts, random_split, output_dir)

run_experiment(datasets, methods, M_lists, epoch_lists, 5, 0, alpha, beta, decay_rate, num_samples, num_restarts, random_split, output_dir)

run_experiment(datasets, methods, M_lists, epoch_lists, 10, 0, alpha, beta, decay_rate, num_samples, num_restarts, random_split, output_dir)

datasets = {'a1a'};
M_lists = {[128]};
num_restarts = 1;

run_experiment(datasets, methods, M_lists, epoch_lists, 1, 0, alpha, beta, decay_rate, num_samples, num_restarts, random_split, output_dir)

run_experiment(datasets, methods, M_lists, epoch_lists, 2, 0, alpha, beta, decay_rate, num_samples, num_restarts, random_split, output_dir)

run_experiment(datasets, methods, M_lists, epoch_lists, 5, 0, alpha, beta, decay_rate, num_samples, num_restarts, random_split, output_dir)

run_experiment(datasets, methods, M_lists, epoch_lists, 10, 0, alpha, beta, decay_rate, num_samples, num_restarts, random_split, output_dir)


% =================================
% ========= Exact Methods =========
% =================================

M_lists = {[1], [1], [1]};  % N/A for exact.
epoch_lists = {[500], [500], [500]};
L = 0; % N/A for exact
K = 0; % N/A for exact.
num_restarts = 20;

methods = {'exact', 'mf-exact'};
datasets = {'australian_scale', 'breast_cancer_scale', 'usps_3vs5'};
run_experiment(datasets, methods, M_lists, epoch_lists, L, K, 0, 0, 0, 0, num_restarts, random_split, 0, output_dir)

datasets = {'a1a'};
num_restarts = 1;
run_experiment(datasets, methods, M_lists, epoch_lists, L, K, 0, 0, 0, 0, num_restarts, random_split, 0, output_dir)
