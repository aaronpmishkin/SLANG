# @Author: aaronmishkin
# @Date:   18-10-07
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-10-10

import os

import numpy as np
import matplotlib.pyplot as plt

from experiments.convergence_experiments.grid_searches.slang_convergence_lr_selection import experiment_name, variants
import experiments.base.slang_experiments as slang
import lib.utilities.record_utils as record_utils

def plot_slang_convergence(data_set, L, metrics, lr_indices):

    if not (data_set == 'australian_presplit' or data_set == 'breastcancer_presplit' or data_set == 'usps_3vs5'):
        raise ValueError('"' + data_set + '" is not a valid value for the data_set argument.')

    # possible datasets are: 'australian_presplit', 'breastcancer_presplit', or 'usps_3vs5'.
    # valid L values are: 1, 8, 16, 32, 64
    # possible metrics are:  'test_pred_logloss', 'train_pred_logloss', 'elbo_neg_ave', 'accuracy' or 'all'.
    # possible learning rates are:
    #       [0.0001, 0.00021544, 0.00046416, 0.001, 0.00215443, 0.00464159, 0.01, 0.02154435, 0.04641589, 0.1]
    # lr_indices is used to select these values and should be list containing integers between
    # 0 and 9.

    if metrics == 'all':
        metrics = ['test_pred_logloss', 'train_pred_logloss', 'elbo_neg_ave', 'accuracy']

    seeds = [1,2,3]

    variants_to_plot = list(filter( lambda z: (data_set in z) and (('_' + str(L) + '_') in z), variants))
    lrs = np.logspace(-4, -1, 10)

    lrs = lrs[lr_indices]

    experiment_base = slang.slang_cv
    lr_variants = []

    for lr in lrs:
        lr_string = ('lr_' + str(lr))
        v = list(filter(lambda z: lr_string in z, variants_to_plot))
        lr_variants.append(v)


    experiment_results = {}
    for i, list_of_restarts in enumerate(lr_variants):
        crashed = False
        for restart in list_of_restarts:
            record = slang.slang_cv.get_variant(experiment_name).get_variant(restart).get_latest_record()
            crashed = crashed or (not record.has_result())

        if not crashed:
            results = record_utils.get_experiment_results(slang.slang_cv, experiment_name, list_of_restarts)
            results = list(results[0].values())
            experiment_results[lrs[i]] = record_utils.summarize_metric_histories(results)
        else:
            experiment_results[lrs[i]] = None      # One of these restarts crashed.



    if not os.path.exists('plots/lr_selection/'):
        os.makedirs('plots/lr_selection/')

    if not os.path.exists('plots/lr_selection/' + data_set + '/'):
        os.makedirs('plots/lr_selection/' + data_set + '/')


    ######################
    #### Plot Results ####
    ######################

    plt.ioff()

    if 'test_pred_logloss' in metrics:
        # Plot test logloss
        plt.figure()
        labels = []
        for key, item in experiment_results.items():
            if not item is None:
                plt.plot(1+np.arange(1001), item['test_pred_logloss']['mean'])
                plt.fill_between(1+np.arange(1001), item['test_pred_logloss']['mean'] - item['test_pred_logloss']['se'], item['test_pred_logloss']['mean'] + item['test_pred_logloss']['se'], alpha=0.6)
                labels.append(str(key))

        plt.legend(labels)
        plt.grid()
        plt.xlabel("Epoch")
        plt.ylabel("Test logloss")
        plt.title("L = " + str(L) + ", Dataset = " + data_set)
        plt.savefig("plots/lr_selection/" + data_set + "/slang_L_" + str(L) + "_Dataset_" + data_set + "_test_logloss.pdf")


    if 'train_pred_logloss' in metrics:
        # Plot train logloss
        plt.figure()
        labels = []
        for key, item in experiment_results.items():
            if not item is None:
                plt.plot(1+np.arange(1001), item['train_pred_logloss']['mean'])
                labels.append(str(key))

        plt.legend(labels)
        plt.grid()
        plt.xlabel("Epoch")
        plt.ylabel("Train logloss")
        plt.title("L = " + str(L) + ", Dataset = " + data_set)
        plt.savefig("plots/lr_selection/" + data_set + "/slang_L_" + str(L) + "_Dataset_" + data_set + "_train_logloss.pdf")
        plt.close()


    if 'elbo_neg_ave' in metrics:
        # Plot ELBO
        plt.figure()
        labels = []
        for key, item in experiment_results.items():
            if not item is None:
                plt.loglog(1+np.arange(1001), item['elbo_neg_ave']['mean'])
                labels.append(str(key))

        plt.legend(labels)
        plt.grid()
        plt.xlabel("Epoch")
        plt.ylabel("Neg. Average ELBO")
        plt.title("L = " + str(L) + ", Dataset = " + data_set)
        plt.savefig("plots/lr_selection/" + data_set + "/slang_L_" + str(L) + "_Dataset_" + data_set + "_elbo.pdf")
        plt.close()
