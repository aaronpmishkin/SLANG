% @Author: amishkin
% @Date:   18-08-10
% @Email:  amishkin@cs.ubc.ca
% @Last modified by:   amishkin
% @Last modified time: 18-08-15

% ##############################################################
% ########## Implements Hessian, mf-Hessian, EF, mf-EF #########
% ##############################################################

function [nlz, log_loss, Sigma, mu] = my_methods(method_name, y, X, gamma, y_te, X_te, options, mu_start, sigma_start, plot)

    fprintf('%s\n',method_name);
    [N,D] = size(X);

    % set default options
    [max_iters, lowerBoundTol, display, num_samples, beta_start, alpha_start, decay_rate, mini_batch_size, L, K] = myProcessOptions(options, ...
    'max_iters', 2000, 'lowerBoundTol',1e-4, 'display', 1, 'num_samples', 1, 'beta', 0.1, 'alpha', 0.8, 'decay_rate', 0.55, 'mini_batch_size', N, 'L', 1, 'K', 0);

    % because we use LogisticLoss
    y_recoded = 2*y-1;

    mu = mu_start;
    Sigma = sigma_start;
    S = inv(Sigma);
    s = diag(S);

    % plot:
    Sigma_all = zeros(D,D,0);
    mu_all = zeros(D,0);
    nlZ_all = [];
    log_loss_all = [];

    data_seen = N;

    if (plot == 1)
        % Sigma_all(:, :, iter) = Sigma;
        % mu_all(:, iter) = mu;

        post_dist.mean = mu(:);
        post_dist.covMat = Sigma;
        [pred, log_lik] = get_loss(1, post_dist, X, y, gamma, X_te, y_te);
        log_loss_all = [log_loss_all, pred.log_loss];
        nlZ_all = [nlZ_all, -log_lik];
    end

    % iterate
    for iter = 1:max_iters

        % Decay the learning rates.
        if decay_rate > 0
            alpha = alpha_start / (1 + iter^(decay_rate));
            beta = beta_start / (1 + iter^(decay_rate));
        else
            alpha = alpha_start;
            beta = beta_start;
        end

        % select a minibatch
        if mini_batch_size < N
            idx = unidrnd(N,[mini_batch_size,1]);
            Xi = X(idx, :);
            yi = y_recoded(idx, :); % use recoded to -1,1
        else  % batch mode, no minibatch exist
            Xi = X;
            yi = y_recoded; % use recoded to -1,1
        end

        weight = N/mini_batch_size;

        switch method_name
        case {'mf-Hessian', 'mf-EF'}
            sig = 1./sqrt(s);

            g = 0; h = 0;
            for i = 1:num_samples
                w = mu + sig.*randn(D,1);
            switch method_name
            case 'mf-Hessian'
                [~, gi, Hi] = LogisticLoss(w, Xi, yi);
            case 'mf-EF'
                [~, gi, ~] = LogisticLoss(w, Xi, yi);
                [~, ~, Hi] = LogisticLossGN(w, Xi, yi);
            end
                g = g + gi;
                h = h + diag(Hi);
            end
            g = (g./num_samples)*weight;
            h = (h./num_samples)*weight;

            s = (1-beta)*s + beta*(h + gamma);
            mu = mu - alpha* ((g + gamma.*mu) ./s);
            Sigma = diag(1./s);
        case {'Hessian', 'EF'}
            U = chol(Sigma);
            % compute gradients
            g = 0; H = 0;
            for i = 1:num_samples
                w = mu + U'*randn(D,1);

                switch method_name
                case 'Hessian'
                    [~, gi, Hi] = LogisticLoss(w, Xi, yi);
                case 'EF'
                    [~, gi, ~] = LogisticLoss(w, Xi, yi);
                    [~, ~, Hi] = LogisticLossGN(w, Xi, yi);

                end
                g = g + gi;
                H = H + Hi; % full Hessian
            end
            g = (g./num_samples)*weight;
            H = (H./num_samples)*weight;

            S = (1-beta)*S + beta*(H + diag(gamma));
            Sigma = inv(S);
            mu = mu - alpha*(Sigma*(g + gamma.*mu));
        end

        if (plot == 1)
            % Sigma_all(:, :, iter) = Sigma;
            % mu_all(:, iter) = mu;

            post_dist.mean = mu(:);
            post_dist.covMat = Sigma;
            [pred, log_lik] = get_loss(iter+1, post_dist, X, y, gamma, X_te, y_te);
            log_loss_all = [log_loss_all, pred.log_loss];
            nlZ_all = [nlZ_all, -log_lik];
        end
    end

    % Save the training path for plotting.
    if plot == 1
        Sigma = Sigma;
        mu = mu;
        log_loss = log_loss_all;
        nlz = nlZ_all;
    else
        post_dist.mean = mu(:);
        post_dist.covMat = Sigma;
        [pred, log_lik] = get_loss(iter, post_dist, X, y, gamma, X_te, y_te);
        log_loss = pred.log_loss;
        nlz = -log_lik;
    end
end
