# @Author: aaronmishkin
# @Date:   18-08-17
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-08-30

import torch
import numpy as np

import lib.metrics.metric_factory as metric_factory
from lib.experiments.printers import print_progress, print_objective
from lib.experiments.evaluate_model import evaluate_model
from lib.experiments.create_results_dictionary import create_results_dictionary

def run_experiment(data,
                   model,
                   model_params,
                   predict_fn,
                   kl_fn,
                   optimizer,
                   optimizer_params,
                   objective_name,
                   metric_names,
                   closure_factory,
                   normalize,
                   save_params,
                   seed,
                   use_cuda,
                   iter_hook=None):

    ##############################################
    ########### Random Seeds and Setup ###########
    ##############################################

    num_epochs = optimizer_params['num_epochs']
    batch_size = optimizer_params['batch_size']
    train_mc_samples = optimizer_params['train_mc_samples']

    ###############################################
    ########### Load and Normalize Data ###########
    ###############################################

    # Prepare data loader for training
    train_loader = data.get_train_loader(batch_size=batch_size)

    # Load full data set for evaluation
    x_train, y_train = data.load_full_train_set(use_cuda=use_cuda)
    x_test, y_test = data.load_full_test_set(use_cuda=use_cuda)
    train_set_size = data.get_train_size()

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


    # Need to handle the normalization around tau carefully.
    # Normalize noise precision
    if 'noise_precision' in model_params:
        tau = model_params['noise_precision']   # Regression Problem

        # Transform tau out of the normalized space.
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

    # the unormalized tau is used for metrics in order to report loss on the scale of the outputs.
    train_metrics, test_metrics = metric_factory.make_metric_closures(metric_names, kl_fn, train_set_size, tau=tau)
    # the unormalized tau is used for metrics in order to report loss on the scale of the outputs.
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
            metric_history = evaluate_model(predict_fn, train_metrics, test_metrics, metric_history, x_train, y_train, x_test, y_test, optimizer_params['eval_mc_samples'], normalize)


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

            if ('every_iter' in save_params) and save_params['every_iter']:
                model.train(False)
                # Evaluate model
                with torch.no_grad():
                    metric_history = evaluate_model(predict_fn, train_metrics, test_metrics, metric_history, x_train, y_train, x_test, y_test, optimizer_params['eval_mc_samples'], normalize)
                model.train(True)


        # Compute and store average objective from last epoch
        objective_history.append(np.mean(batch_objective))

        if save_params['metric_history']:
            # Set model in test mode
            model.train(False)
            # Evaluate model
            with torch.no_grad():
                metric_history = evaluate_model(predict_fn, train_metrics, test_metrics, metric_history, x_train, y_train, x_test, y_test, optimizer_params['eval_mc_samples'], normalize)
            # Print progress
            print_progress(epoch, num_epochs, metric_history)
            print_objective(epoch, num_epochs, objective_history[-1])

            if iter_hook is not None:
                iter_hook(model, optimizer, metric_history)

        else:
            # Print average objective from last epoch
            print_objective(epoch, num_epochs, objective_history[-1])

    # Set model in test mode
    model.train(False)

    # Evaluate model
    with torch.no_grad():
        final_metrics = evaluate_model(predict_fn, train_metrics, test_metrics, final_metrics, x_train, y_train, x_test, y_test, optimizer_params['eval_mc_samples'], normalize)

    # create the dictionary of results that will be saved
    results_dict = create_results_dictionary(save_params, final_metrics, metric_history, objective_history, model, optimizer)

    return results_dict
