% @Author: amishkin
% @Date:   18-09-12
% @Email:  amishkin@cs.ubc.ca
% @Last modified by:   amishkin
% @Last modified time: 18-09-12

% =======================================================================================
% Reproduces the grid search used to select learning rates for the Hessian method in
% the logistic regression convergence experiments (Figures 4 and 6 (Left Columns)). See
% section E.3 of the appendix for details on the experimental procedure.
% =======================================================================================

% We want to retain all training paths for plotting.
PLOT = 1;

addpath(genpath('../../'))
mkdir('../data')
output_dir = '../data/convergence-comparison-grid-search/'
mkdir(output_dir)

methods = {'Hessian'}

L = 0;   % N/A for EF and mf-EF
K = 0; % N/A for EF and mf-EF
num_samples = 12;

% Try no decay for now.
decay_rate = 0   % controls the rate at which alpha and beta decay during training.

random_split = 0;
num_restarts = 3;
datasets = {'australian_scale', 'breast_cancer_scale', 'usps_3vs5'};

M_lists = {[32], [32], [64]};
epoch_lists = {[5000], [5000], [5000]};

% define a grid of learning rates to search over.
learning_rates = logspace(-3, -0.6, 10)
decay_rates = []

decay_rates = [0, 0.55]

for decay_rate = decay_rates
    for lr = learning_rates
        alpha = lr
        beta = lr
        run_experiment(datasets, methods, M_lists, epoch_lists, L, K, alpha, beta, decay_rate, num_samples, num_restarts, random_split, PLOT, output_dir)
    end
end
