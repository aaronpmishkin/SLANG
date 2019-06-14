# @Author: aaronmishkin
# @Date:   18-10-07
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-10-10

import os

import numpy as np
import matplotlib.pyplot as plt

from experiments.convergence_experiments.grid_searches.slang_convergence_prior_selection import experiment_name, variants
import experiments.base.slang_experiments as slang
import lib.utilities.record_utils as record_utils

def plot_slang_convergence(data_set, L, metrics, prior_prec_indices):

    if not (data_set == 'australian_presplit' or data_set == 'breastcancer_presplit' or data_set == 'usps_3vs5'):
        raise ValueError('"' + data_set + '" is not a valid value for the data_set argument.')

    # possible datasets are: 'australian_presplit', 'breastcancer_presplit', or 'usps_3vs5'.
    # valid values of L are: 1 and 10.
    # possible metrics are:  'test_pred_logloss', 'train_pred_logloss', 'elbo_neg_ave', 'accuracy' or 'all'.
    # possible prior precisions are:
    #       prior_precisions = [0.001, 0.01, 0.1, 1, 8, 32, 64, 128, 512]
    # prior_prec_indices is used to select these values and should be list containing integers between
    # 0 and 8.


    if metrics == 'all':
        metrics = ['test_pred_logloss', 'train_pred_logloss', 'elbo_neg_ave']

    prior_precisions = [0.001, 0.01, 0.1, 1, 8, 32, 64, 128, 512]
    seeds = [1,2,3]

    variants_to_plot = list(filter( lambda z: (data_set in z) and (('_' + str(L) + '_') in z), variants))

    experiment_base = slang.slang_cv
    prior_prec_variants = []

    for i in prior_prec_indices:
        prec = prior_precisions[i]
        prior_string = ('prior_prec_' + str(prec))
        v = list(filter(lambda z: prior_string in z, variants_to_plot))
        prior_prec_variants.append(v)

    experiment_results = {}
    for i, list_of_restarts in enumerate(prior_prec_variants):
        crashed = False
        for restart in list_of_restarts:
            record = slang.slang_cv.get_variant(experiment_name).get_variant(restart).get_latest_record()
            crashed = crashed or (not record.has_result())

        if not crashed:
            results = record_utils.get_experiment_results(slang.slang_cv, experiment_name, list_of_restarts)
            results = list(results[0].values())
            experiment_results[prior_precisions[i]] = record_utils.summarize_metric_histories(results)
        else:
            experiment_results[prior_precisions[i]] = None      # One of these restarts crashed.

    if not os.path.exists('plots/prior_selection/'):
        os.makedirs('plots/prior_selection/')

    if not os.path.exists('plots/prior_selection/' + data_set + '/'):
        os.makedirs('plots/prior_selection/' + data_set + '/')

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
                plt.plot(1+np.arange(5001), item['test_pred_logloss']['mean'])
                labels.append(str(key))

        plt.legend(labels)
        plt.grid()
        plt.xlabel("Epoch")
        plt.ylabel("Test logloss")
        plt.title("L = " + str(L) + ", Dataset = " + data_set)
        plt.savefig("plots/prior_selection/" + data_set + "/slang_L_" + str(L) + "_Dataset_" + data_set + "_test_logloss.pdf")
        plt.close()


    if 'train_pred_logloss' in metrics:
        # Plot train logloss
        plt.figure()
        labels = []
        for key, item in experiment_results.items():
            if not item is None:
                plt.plot(1+np.arange(5001), item['train_pred_logloss']['mean'])
                labels.append(str(key))

        plt.legend(labels)
        plt.grid()
        plt.xlabel("Epoch")
        plt.ylabel("Train logloss")
        plt.title("L = " + str(L) + ", Dataset = " + data_set)
        plt.savefig("plots/prior_selection/" + data_set + "/slang_L_" + str(L) + "_Dataset_" + data_set + "_train_logloss.pdf")
        plt.close()


    if 'elbo_neg_ave' in metrics:
        # Plot ELBO
        plt.figure()
        labels = []
        for key, item in experiment_results.items():
            if not item is None:
                plt.loglog(1+np.arange(5001), item['elbo_neg_ave']['mean'])
                labels.append(str(key))

        plt.legend(labels)
        plt.grid()
        plt.xlabel("Epoch")
        plt.ylabel("Neg. Average ELBO")
        plt.title("L = " + str(L) + ", Dataset = " + data_set)
        plt.savefig("plots/prior_selection/" + data_set + "/slang_L_" + str(L) + "_Dataset_" + data_set + "_elbo.pdf")
        plt.close()
