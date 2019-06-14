# @Author: aaronmishkin
# @Date:   18-08-24
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-09-13

import torch
import numpy as np

import lib.metrics.metric_factory as metric_factory
from lib.experiments.printers import print_cv_progress, print_cv_objective
from lib.experiments.evaluate_model import evaluate_model
from lib.experiments.create_results_dictionary import create_results_dictionary

def run_cv_experiment(data,
                      n_splits,
                      init_fn,
                      model_params,
                      optimizer_params,
                      objective_name,
                      metric_names,
                      normalize,
                      save_params,
                      seed,
                      use_cuda):

    ##############################################
    ########### Random Seeds and Setup ###########
    ##############################################

    num_epochs = optimizer_params['num_epochs']
    batch_size = optimizer_params['batch_size']
    train_mc_samples = optimizer_params['train_mc_samples']

    # Set random seed
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

    split_results = {}
    for split in range(n_splits):

        ###############################################
        ########### Load and Normalize Data ###########
        ###############################################

        # Set current split
        data.set_current_split(split)
        # Prepare data loader for current split for training
        train_loader = data.get_current_train_loader(batch_size=batch_size)

        # Load full data set for evaluation
        x_train, y_train = data.load_current_train_set(use_cuda=use_cuda)
        x_val, y_val = data.load_current_val_set(use_cuda=use_cuda)
        train_set_size = data.get_current_train_size()

        # Compute normalization of x
        if normalize['x']:
            normalize['x_means'] = torch.mean(x_train, dim=0)
            normalize['x_stds'] = torch.std(x_train, dim=0)
            normalize['x_stds'][normalize['x_stds'] == 0] = 1

        # Compute normalization of y
        if normalize['y']:
            normalize['y_mean'] = torch.mean(y_train)
            normalize['y_std'] = torch.std(y_train)
            if normalize['y_std'] == 0:
                normalize['y_std'] = 1

        ##########################################
        ### Initialize the Model and Optimizer ###
        ##########################################

        model, predict_fn, kl_fn, closure_factory, optimizer = init_fn(data, model_params, optimizer_params, train_set_size=train_set_size, use_cuda=use_cuda)

        # Need to handle the normalization around tau carefully.
        # Normalize noise precision

        if 'noise_precision' in model_params:
            tau = model_params['noise_precision']   # Regression Problem

            # Transform tau into the normalized space.
            if normalize['y']:
                tau_normalized = tau * (normalize['y_std']**2)
            else:
                tau_normalized = tau
        else:                                        # Classification Problem
            tau_normalized = None
            tau = None

        #########################################################
        ########### Instantiate Objective and Metrics ###########
        #########################################################

        train_metrics, test_metrics = metric_factory.make_metric_closures(metric_names, kl_fn, train_set_size, tau=tau)
        objective = metric_factory.make_objective_closure(objective_name, kl_fn, train_set_size, tau=tau_normalized)

        # Create metric history
        metric_history = {}
        final_metrics = {}
        for name in train_metrics.keys():
            metric_history[name] = []
            final_metrics[name] = []
        for name in test_metrics.keys():
            metric_history[name] = []
            final_metrics[name] = []
        objective_history = []

        # Evaluate Model Before Training Begins
        if save_params['metric_history']:
            # Set model in test mode
            model.train(False)
            # Evaluate model
            with torch.no_grad():
                metric_history = evaluate_model(predict_fn, train_metrics, test_metrics, metric_history, x_train, y_train, x_val, y_val, optimizer_params['eval_mc_samples'], normalize)
            # Print progress


        #####################################
        ########### Training Loop ###########
        #####################################

        for epoch in range(num_epochs):

            # Set model in training mode
            model.train(True)

            # Initialize batch objective accumulator
            batch_objective = []

            for _, (x, y) in enumerate(train_loader):

                # Prepare minibatch
                if use_cuda:
                    x, y = x.cuda(), y.cuda()

                # Normalize x and y
                if normalize['x']:
                    x = (x-normalize['x_means'])/normalize['x_stds']
                if normalize['y']:
                    y = (y-normalize['y_mean'])/normalize['y_std']

                closure = closure_factory(x, y, objective, model, predict_fn, optimizer, train_mc_samples)
                loss = optimizer.step(closure)

                # Store batch objective
                batch_objective.append(loss.detach().cpu().item())

            # Compute and store average objective from last epoch
            objective_history.append(np.mean(batch_objective))

            if save_params['metric_history']:
                # Set model in test mode
                model.train(False)
                # Evaluate model
                with torch.no_grad():
                    metric_history = evaluate_model(predict_fn, train_metrics, test_metrics, metric_history, x_train, y_train, x_val, y_val, optimizer_params['eval_mc_samples'], normalize)
                # Print progress
                print_cv_progress(split, n_splits, epoch, num_epochs, metric_history)
                print_cv_objective(split, n_splits, epoch, num_epochs, objective_history[-1])
            else:
                # Print average objective from last epoch
                print_cv_objective(split, n_splits, epoch, num_epochs, objective_history[-1])

        # Set model in test mode
        model.train(False)

        # Evaluate model
        with torch.no_grad():
            final_metrics = evaluate_model(predict_fn, train_metrics, test_metrics, final_metrics, x_train, y_train, x_val, y_val, optimizer_params['eval_mc_samples'], normalize)

        # create the dictionary of results that will be saved
        split_results[split] = create_results_dictionary(save_params, final_metrics, metric_history, objective_history, model, optimizer)

    return split_results
