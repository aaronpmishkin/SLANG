% @Author: amishkin
% @Date:   18-09-12
% @Email:  amishkin@cs.ubc.ca
% @Last modified by:   amishkin
% @Last modified time: 18-09-12

% =======================================================================================
% Reproduces the convergence experiments for Figures 4 (left column) and 6 (left column).
% =======================================================================================

% We want to retain all training information for plotting.
PLOT = 1;

addpath(genpath('.'))
mkdir('./experiments/data')
output_dir = './experiments/data/final-convergence-comparison/';
mkdir(output_dir)

L = 0;   % N/A for EF, Hessian and mf-EF, mf-Hessian
K = 0; % N/A for EF, Hessian and mf-EF, mf-Hessian
num_samples = 12; % number of MC samples used to estimate expectations.

decay_rate = 0;   % controls the rate at which alpha and beta decay during training. 0 means no decay rate.

random_split = 0;   % use the same split of the dataset every restart.
num_restarts = 10;

datasets = {'australian_scale', 'breast_cancer_scale', 'usps_3vs5'};

aust_breast_batch_size = {[32]};
usps_batch_size = {[64]};
a1a_batch_size = {[128]};

num_epochs = {[2000]};


%============ MF Hessian ============

learning_rates = [0.0215, 0.0215, 0.0063];
run_experiment({'australian_scale'}, {'mf-Hessian'}, aust_breast_batch_size, num_epochs, L, K, learning_rates(1), learning_rates(1), decay_rate, num_samples, num_restarts, random_split, PLOT, output_dir)
run_experiment({'breast_cancer_scale'}, {'mf-Hessian'}, aust_breast_batch_size, num_epochs, L, K, learning_rates(2), learning_rates(2), decay_rate, num_samples, num_restarts, random_split, PLOT, output_dir)
run_experiment({'usps_3vs5'}, {'mf-Hessian'}, usps_batch_size, num_epochs, L, K, learning_rates(3), learning_rates(3), decay_rate, num_samples, num_restarts, random_split, PLOT, output_dir)


%============ Hessian ============

learning_rates = [0.0117, 0.0398, 0.0398];
run_experiment({'australian_scale'}, {'Hessian'}, aust_breast_batch_size, num_epochs, L, K, learning_rates(1), learning_rates(1), decay_rate, num_samples, num_restarts, random_split, PLOT, output_dir)
run_experiment({'breast_cancer_scale'}, {'Hessian'}, aust_breast_batch_size, num_epochs, L, K, learning_rates(2), learning_rates(2), decay_rate, num_samples, num_restarts, random_split, PLOT, output_dir)
run_experiment({'usps_3vs5'}, {'Hessian'}, usps_batch_size, num_epochs, L, K, learning_rates(3), learning_rates(3), decay_rate, num_samples, num_restarts, random_split, PLOT, output_dir)


%============ SLANG ============

%      ===== L = 1 =====
L = 1;
learning_rates = [0.0117, 0.0398, 0.0117];
run_experiment({'australian_scale'}, {'SLANG'}, aust_breast_batch_size, num_epochs, L, K, learning_rates(1), learning_rates(1), decay_rate, num_samples, num_restarts, random_split, PLOT, output_dir)
run_experiment({'breast_cancer_scale'}, {'SLANG'}, aust_breast_batch_size, num_epochs, L, K, learning_rates(2), learning_rates(2), decay_rate, num_samples, num_restarts, random_split, PLOT, output_dir)
run_experiment({'usps_3vs5'}, {'SLANG'}, usps_batch_size, num_epochs, L, K, learning_rates(3), learning_rates(3), decay_rate, num_samples, num_restarts, random_split, PLOT, output_dir)


%      ===== L = 5 =====
L = 5;
learning_rates = [0.0117, 0.0398, 0.0215];
run_experiment({'australian_scale'}, {'SLANG'}, aust_breast_batch_size, num_epochs, L, K, learning_rates(1), learning_rates(1), decay_rate, num_samples, num_restarts, random_split, PLOT, output_dir)
run_experiment({'breast_cancer_scale'}, {'SLANG'}, aust_breast_batch_size, num_epochs, L, K, learning_rates(2), learning_rates(2), decay_rate, num_samples, num_restarts, random_split, PLOT, output_dir)
run_experiment({'usps_3vs5'}, {'SLANG'}, usps_batch_size, num_epochs, L, K, learning_rates(3), learning_rates(3), decay_rate, num_samples, num_restarts, random_split, PLOT, output_dir)


%      ===== L = 10 =====
L = 10;
learning_rates = [0.0117, 0.0398, 0.0398];
run_experiment({'australian_scale'}, {'SLANG'}, aust_breast_batch_size, num_epochs, L, K, learning_rates(1), learning_rates(1), decay_rate, num_samples, num_restarts, random_split, PLOT, output_dir)
run_experiment({'breast_cancer_scale'}, {'SLANG'}, aust_breast_batch_size, num_epochs, L, K, learning_rates(2), learning_rates(2), decay_rate, num_samples, num_restarts, random_split, PLOT, output_dir)
run_experiment({'usps_3vs5'}, {'SLANG'}, usps_batch_size, num_epochs, L, K, learning_rates(3), learning_rates(3), decay_rate, num_samples, num_restarts, random_split, PLOT, output_dir)
