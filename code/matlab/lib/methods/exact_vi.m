% @Author: amishkin
% @Date:   2018-07-10T14:56:18-07:00
% @Email:  amishkin@cs.ubc.ca
% @Last modified by:   amishkin
% @Last modified time: 18-08-10

% Adapted from code written by Emtiyaz Khan.

% ###############################################
% ############# Implements mf-exact #############
% ###############################################

function [nlz, log_loss, Sigma, mu] = exact_vi(method_name, y, X, gamma, y_te, X_te, options, mu_start, sigma_start, plot)

    fprintf('%s\n',method_name);
    [N,D] = size(X);

    % set default options
    [max_iters, lowerBoundTol, display, num_samples, beta_start, alpha_start, decay_rate, mini_batch_size, L, K] = myProcessOptions(options, ...
    'max_iters', 2000, 'lowerBoundTol',1e-4, 'display', 1, 'num_samples', 1, 'beta', 0.1, 'alpha', 0.8, 'decay_rate', 0.55, 'mini_batch_size', N, 'L', 1, 'K', 0);


    % minfunc options
    optMinFunc = struct('display', display,...
        'Method', 'lbfgs',...
        'DerivativeCheck', 'off',...
        'LS', 2,...
        'recordPath', 1, ...
        'recordPathIters', 1, ...
        'MaxIter', max_iters+1,...
        'MaxFunEvals', max_iters+1,...
        'TolFun', lowerBoundTol,......
        'TolX', lowerBoundTol);

    V = sigma_start;
    m = mu_start;

    % plot:
    Sigma_all = zeros(D,D,0);
    mu_all = zeros(D,0);
    nlZ_all = [];
    log_loss_all = [];

    switch method_name
    case 'exact'
        v0 = [m; packcovariance( triu(chol(V)) )];
        funObj = @funObj_vi_exact;
    case 'mf-exact'
        v0 = [m; sqrt(diag(V))];
        funObj = @funObj_mfvi_exact;
    end

    % compute loss at iter =0
    post_dist.mean = m;
    post_dist.covMat = V;

    [v, f, exitflag, inform] = minFunc(funObj, v0, optMinFunc, y, X, gamma);
    v_all = inform.trace.x;

    % compute loss for iter>0
    for ii=1:size(v_all,2)
        vi = v_all(:,ii);
        post_dist.mean = vi(1:D);
        switch method_name
        case 'exact'
            U = triu(unpackcovariance(vi(D+1:end),D));
        case 'mf-exact'
            U = diag(vi(D+1:end));
        end
        post_dist.covMat = U'*U;
        if sum(eig(post_dist.covMat) <= 1e-8) > 0
            ALERT = 'NOT PD'
            nlz(ii) = NaN;
            log_loss(ii) = NaN;
            continue
        end
        [pred, log_lik]=get_loss(ii-1, post_dist, X, y, gamma, X_te, y_te);
        nlz(ii)=-log_lik;
        log_loss(ii)=pred.log_loss;
        Sigma_all(:,:,ii) = post_dist.covMat;
        mu_all(:,ii) = post_dist.mean;
    end


    % Save the training path for plotting.
    if plot == 1
        Sigma = post_dist.covMat;
        mu = post_dist.mean;
        log_loss = log_loss;
        nlz = nlz;
    else
        [pred, log_lik] = get_loss(ii-1, post_dist, X, y, gamma, X_te, y_te);
        nlz = -log_lik;
        log_loss = pred.log_loss;

        Sigma = post_dist.covMat;
        mu = post_dist.mean;
    end
end
