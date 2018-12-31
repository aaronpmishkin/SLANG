% @Author: amishkin
% @Date:   2018-07-10T13:52:49-07:00
% @Email:  amishkin@cs.ubc.ca
% @Last modified by:   amishkin
% @Last modified time: 18-08-15

% ##########################################################
% ############# Helper for Running Experiments #############
% ##########################################################

function [] = run_experiment(datasets, methods, M_lists, epoch_lists, L, K, alpha, beta, decay_rate, num_samples, num_restarts, random_split, plot, output_dir)
    for j = 1:length(datasets)
        dataset_name = datasets{j};
        M_list = M_lists{j};
        epoch_list = epoch_lists{j};
        for k = 1: length(methods)
            method_name = methods{k};
            % Create the necessary directory structure if it doesn't exist.
            mkdir(strcat('./', output_dir, '/', dataset_name));
            mkdir(strcat('./', output_dir, '/', dataset_name, '/', method_name));
            for s = 1:num_restarts
                for m = 1:length(M_list)
                    mini_batch_size = M_list(m);
                    num_epochs = epoch_list(m);

                    % Set the random seed to be the restart number.
                    if random_split % use a different split for each restart
                        [y, X, y_te, X_te] = get_data_log_reg(dataset_name, s);
                    else % use the same split for each restart
                        [y, X, y_te, X_te] = get_data_log_reg(dataset_name, 1);
                        % The seed is set in get_data_log_reg, so we must reset it after the function call.
                        setSeed(s);
                    end
                    [N, D] = size(X);
                    % Set default starting parameters.
                    mu_start = zeros(D,1);
                    sigma_start = diag(ones(D,1));

                    [N,D] = size(X);

                    % Define the file name for this run.
                    save_name = strcat(dataset_name, '_', method_name, '_M_', num2str(mini_batch_size), '_L_', num2str(L), '_K_', num2str(K), '_beta_', num2str(beta), '_alpha_', num2str(alpha), '_decay_', num2str(decay_rate),  '_restart_', num2str(s), '.mat')
                    % Print the parameters for this run.
                    trial_params = [num_samples, mini_batch_size, num_epochs, beta, alpha, decay_rate, L, K];
                    parameter_string = sprintf('S: %0.5g, M: %0.5g, Epochs: %0.5g, beta: %0.5g, alpha: %0.5g, decay rate: %0.5g, L: %0.5g, K: %0.5g', trial_params)

                    [delta, parameter_object] =  get_expt_params(dataset_name, method_name, trial_params);
                    deltas = [1e-5; delta.*ones(D-1,1)]; % prior variance

                    % Run the correct method:
                    switch method_name
                    case {'mf-exact', 'exact'}
                        [nlZ, log_loss, Sigma, mu] = exact_vi(method_name, y, X, deltas, y_te, X_te, parameter_object, mu_start, sigma_start, plot);
                    case {'SLANG'}
                        [nlZ, log_loss, Sigma, mu] = slang(method_name, y, X, deltas, y_te, X_te, parameter_object, mu_start, sigma_start, plot);
                    case {'EF', 'mf-EF', 'Hessian', 'mf-Hessian'}
                        [nlZ, log_loss, Sigma, mu] = cvi(method_name, y, X, deltas, y_te, X_te, parameter_object, mu_start, sigma_start, plot);
                    otherwise
                        error('do not support');
                    end

                    if isfield(parameter_object,'mini_batch_size')
                        mini_bsz = parameter_object.mini_batch_size;
                    else
                        mini_bsz = N;
                    end
                    ipp = floor(N / mini_bsz);
                    % Print information about the most recent run:
                    if plot
                        fprintf('%s Restart Number: %.4f, L: %.4f, K: %.4f, ELBO: %.4f, LogLoss: %.4f\n', method_name, s, L, K, nlZ(end), log_loss(end));
                    else
                        fprintf('%s Restart Number: %.4f, L: %.4f, K: %.4f, ELBO: %.4f, LogLoss: %.4f\n', method_name, s, L, K, nlZ, log_loss);
                    end
                    file_name = strcat('./', output_dir, '/', dataset_name, '/', method_name, '/', save_name);
                    save(file_name, 'method_name', 'dataset_name', 'trial_params', 'Sigma', 'mu', 'ipp', 'log_loss', 'nlZ');
                end
            end
        end
    end
end
