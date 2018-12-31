% @Author: amishkin
% @Date:   18-08-15
% @Email:  amishkin@cs.ubc.ca
% @Last modified by:   aaronmishkin
% @Last modified time: 18-08-16


% ############################################
% ############# Implements SLANG #############
% ############################################

function [nlz, log_loss, Sigma, mu] = slang(method_name, y, X, gamma, y_te, X_te, options, mu_start, sigma_start, plot)

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

    % Memory for SLANG:
    Dprev = zeros(D, 0);
    Delprev = zeros(0);
    Rprev = zeros(D, 0);

    % SLANG's Parameterization of Sigma
    diagonal = diag(S);
    V = zeros(D,L);

    % Momentum variables.
    beta_momentum = 0.1;
    grad_average = zeros(D,1);

    ipp = floor(N / mini_batch_size);

    % Records for use when plotting:
    Sigma_all = zeros(D,D,0);
    mu_all = zeros(D,0);
    nlZ_all = [];
    log_loss_all = [];

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

        % compute gradients
        g = 0; H = 0; g_prev = 0;
        for i = 1:num_samples
            sample = randn(D,1);

            w = sherman_morrison_sr(diagonal, Rprev, Delprev, sample) + mu;

            [~, gi, ~] = LogisticLoss(w, Xi, yi);
            [~, ~, Hi] = LogisticLossGN(w, Xi, yi);
            g = g + gi;
            H = H + Hi;
        end

        g = (g./num_samples)*weight;
        g_prev = (g_prev./num_samples)*weight;
        H = (H./num_samples)*weight;

        % ###############################################
        % ############# Main Algorithm Code #############
        % ###############################################

        % Create the expanded matrix. This is not efficient, but it is correct.
        A = (1-beta) .* (V * V') + beta .* H;

        [a,b] = eigs(A, L);
        b = diag(b);
        [~, ind] = sort(b);
        % Get the weighted eigenvectors
        b = sqrt(b(ind(end-L+1:end)));
        a = a(:,ind(end-L+1:end));

        % Update the parameterization of Sigma.
        V_corrected = a .* b';
        diag_corrected = ((1-beta) .* diagonal) + (diag(A) - (sum(V.^2, 2))) + beta .* gamma;

        % This implementation uses the sherman-morrison matrix inversion lemma to invert the precision
        % and to compute square-root of this inverse. The main difference between this and the Woodbury
        % implementation used in our Python experiments is that the sherman-morrison procedure is iterative,
        % and so parallelizes poorly on GPUs. However, it is sufficient for these experiments.
        [Dprev, Delprev, Rprev] = sherman_morrison_memory(V_corrected, diag_corrected);

        % momentum
        grad_average = ((1-beta_momentum) .* grad_average) + (g .* beta_momentum);
        grad_average_corrected = grad_average ./ (1 - (1 - beta_momentum)^iter);
        % use the sherman-morrison lemma to compute: Sigma * (grad_average + gamma * mu)
        prod = sherman_morrison_inv(V_corrected, diag_corrected, Dprev, Delprev, (grad_average_corrected + gamma.*mu));
        %
        mu = mu - alpha.*prod;
        % compute Sigma for the purpose of recording the loss.
        % This can be done using the sherman_morrison_inv procedure and the identity matrix.
        Sigma = sherman_morrison_inv(V_corrected, diag_corrected, Dprev, Delprev, eye(D));

        if (plot == 1) % record the loss for plotting purposes.
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
