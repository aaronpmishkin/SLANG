% @Author: aaronmishkin
% @Date:   2018-07-30T11:30:27-07:00
% @Email:  amishkin@cs.ubc.ca
% @Last modified by:   aaronmishkin
% @Last modified time: 18-08-24

clear all
addpath(genpath('..'))

results_directory = '../experiments/paper_experiment_data/final_log_reg_table/';
methods = {'mf-EF', 'mf-Hessian', 'mf-exact', 'SLANG', 'EF', 'Hessian', 'exact'};

datasets = {'australian_scale', 'breast_cancer_scale', 'a1a', 'usps_3vs5'};
mini_batch_sizes = [32, 32, 128, 64];
num_restarts = 20;

slang_L_values = [1,2,5,10];
alpha_beta = 0.05;

table_header =   {'Datasets', 'Metrics', '', '',    'Mean-Field', '',  '', '',      '', '',      '', '',      'SLANG', '', '', '',      '', '',   'Full-Gaussian', '', '' '',; ...
                  '',        '',        'EF (mean)', '(se)',  'Hessian (mean)', '(se)',      'Exact (mean)', '(se)', 'L = 1 (mean)', '(se)', 'L = 2 (mean)', '(se)', 'L = 5 (mean)', '(se)', 'L = 10 (mean)', '(se)', 'EF (mean)', '(se)', 'Hessian (mean)', '(se)',       'Exact (mean)' '(se)',};

table_left_margin = {'', 'ELBO'; 'Australian', 'NLL'; '', 'KL'; '', 'ELBO'; 'Breast Cancer', 'NLL'; '', 'KL'; '', 'ELBO'; 'a1a', 'NLL'; '', 'KL'; '', 'ELBO'; 'USPS 3vs5', 'NLL'; '', 'KL'; };


table = {};
for i  = 1:length(methods)
    method_name = methods{i};

    L_values = [0];
    alpha_beta = 0.05;
    decay_rate = 0.51;

    switch method_name
        case 'SLANG'
            L_values = slang_L_values;
        case {'exact', 'mf-exact'}
            alpha_beta = 0;
            decay_rate = 0;
        otherwise
        % do nothing
    end

    for l = 1:length(L_values)
        L = L_values(l);
        new_columns = {};

        for j = 1:length(datasets)
            switch method_name
                case {'exact', 'mf-exact'}
                    mini_batch_size = 1;
                otherwise
                    mini_batch_size = mini_batch_sizes(j);
            end

            dataset_name = datasets{j};
            method_path = strcat(results_directory, dataset_name, '/', method_name, '/');
            [y, X, y_te, X_te] = get_data_log_reg(dataset_name, 1);
            [N, D] = size(X);
            exact_vi_path = strcat(results_directory, dataset_name, '/exact/');

            lls = [];
            nlZs = [];
            KLs = [];

            if strcmp(dataset_name, 'a1a')
                num_restarts = 1;
            else
                num_restarts = 20;
            end

            for s = 1:num_restarts
                exact_vi_file_name = strcat(dataset_name, '_exact_M_1_L_0_K_0_beta_0_alpha_0_decay_0_restart_', num2str(s), '.mat');
                exact = load(strcat(exact_vi_path, exact_vi_file_name));
                vi_exact.sigma = exact.Sigma;
                vi_exact.mu = exact.mu;

                % load trial data.
                method_file_name = strcat(dataset_name, '_', method_name, '_M_', num2str(mini_batch_size), '_L_', num2str(L), '_K_0_beta_', num2str(alpha_beta), '_alpha_', num2str(alpha_beta), '_decay_', num2str(decay_rate), '_restart_', num2str(s), '.mat');

                method = load(strcat(method_path, method_file_name));
                lls(s) = method.log_loss(end);
                nlZs(s) = method.nlZ(end);

                method.sigma = method.Sigma;
                method.mu = method.mu;

                KLs(s) = (KL(vi_exact, method) + KL(method, vi_exact)) / 2;
            end

            % compute means and standard deviations:
            mean_nlZ = mean(nlZs / N);
            se_nlZ = std(nlZs / N) / sqrt(num_restarts);
            mean_ll = mean(lls / log2(exp(1)));
            se_ll = std(lls ./ log2(exp(1))) / sqrt(num_restarts);
            mean_KL = mean(KLs);
            se_KL = std(KLs) / sqrt(num_restarts);

            % add the new columns to the table.
            new_columns = [new_columns; mean_nlZ se_nlZ; mean_ll, se_ll; mean_KL, se_KL];
        end

    table = [table, new_columns];
    end
end

mkdir('../tables')

fid = fopen('../tables/table_1_7.csv','wt');
if fid>0
    for k=1:size(table_header, 1)
        for i=1:size(table_header,2)
            fprintf(fid,'%s, ',table_header{k,i});
        end
        fprintf(fid, '\n');
    end

    for k=1:size(table,1)
        fprintf(fid, '%s, ', table_left_margin{k, 1});
        fprintf(fid, '%s, ', table_left_margin{k, 2});
        for i=1:size(table,2)
            fprintf(fid,'%f, ',table{k,i});
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
end
