# @Author: aaronmishkin
# @Date:   18-10-07
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-10-10

import os

import numpy as np
import matplotlib.pyplot as plt

import experiments.convergence_experiments.slang_convergence_final as slang_final
import experiments.convergence_experiments.bbb_convergence_final as bbb_final
import experiments.base.slang_experiments as slang
import experiments.base.bbb_copy_slang as bbb
import lib.utilities.record_utils as record_utils

def plot_convergence(data_set, metrics):

    if not (data_set == 'australian_presplit' or data_set == 'breastcancer_presplit' or data_set == 'usps_3vs5'):
        raise ValueError('"' + data_set + '" is not a valid value for the data_set argument.')

    # possible datasets are: 'australian_presplit', 'breastcancer_presplit', or 'usps_3vs5'.
    # possible metrics are:  'test_pred_logloss', 'train_pred_logloss', 'elbo_neg_ave', 'accuracy' or 'all'.

    if metrics == 'all':
        metrics = ['test_pred_logloss', 'train_pred_logloss', 'elbo_neg_ave', 'accuracy']


    random_seeds = np.arange(1,11)

    slang_variants_to_plot = list(filter(lambda z: (data_set in z), slang_final.variants))
    bbb_variants_to_plot = list(filter(lambda z: (data_set in z), bbb_final.variants))


    experiment_base = slang.slang_base
    L_variants = []
    Ls = [1,8,32,64]
    for l in Ls:
        l_string = ('L_' + str(l))
        v = list(filter(lambda z: l_string in z, slang_variants_to_plot))
        L_variants.append(v)


    experiment_results = {}
    for i, list_of_restarts in enumerate(L_variants):
        crashed = False
        for restart in list_of_restarts:
            record = slang.slang_base.get_variant(slang_final.experiment_name).get_variant(restart).get_latest_record()
            crashed = crashed or (not record.has_result())

        if not crashed:
            results = record_utils.get_experiment_results(slang.slang_base, slang_final.experiment_name, list_of_restarts)
            experiment_results[Ls[i]] = record_utils.summarize_metric_histories(results)
        else:
            experiment_results[Ls[i]] = None      # One of these restarts crashed.


    results = record_utils.get_experiment_results(bbb.bbb_copy_slang, bbb_final.experiment_name, bbb_variants_to_plot)
    experiment_results["BBB"] = record_utils.summarize_metric_histories(results)


    # create the necessary directories if they don't exist.
    if not os.path.exists('plots/final_plots/'):
        os.makedirs('plots/final_plots/')

    if not os.path.exists('plots/final_plots/' + data_set + '/'):
        os.makedirs('plots/final_plots/' + data_set + '/')


    if data_set == 'usps_3vs5':
        iters = 6501
    else:
        iters = 6001


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
                plt.plot(np.arange(iters), item['test_pred_logloss']['mean'])
                labels.append(str(key))

        plt.legend(labels)
        plt.grid()
        plt.xlabel("Epoch")
        plt.ylabel("Test logloss")
        plt.title("Dataset = " + data_set)
        plt.savefig("plots/final_plots/" + data_set +  "/test_logloss.pdf")
        plt.close()

    if 'accuracy' in metrics:
        # Plot test logloss
        plt.figure()
        labels = []
        for key, item in experiment_results.items():
            if not item is None:
                plt.plot(np.arange(iters), item['test_pred_accuracy']['mean'])
                labels.append(str(key))

        plt.legend(labels)
        plt.grid()
        plt.xlabel("Epoch")
        plt.ylabel("Test Accuracy")
        plt.title("Dataset = " + data_set)
        plt.savefig("plots/final_plots/" + data_set +  "/test_accuracy.pdf")
        plt.close()


    if 'train_pred_logloss' in metrics:
        # Plot train logloss
        plt.figure()
        labels = []
        for key, item in experiment_results.items():
            if not item is None:
                plt.plot(np.arange(iters), item['train_pred_logloss']['mean'])
                labels.append(str(key))

        plt.legend(labels)
        plt.grid()
        plt.xlabel("Epoch")
        plt.ylabel("Train logloss")
        plt.title("Dataset = " + data_set)
        plt.savefig("plots/final_plots/" + data_set + "/train_logloss.pdf")
        plt.close()


    if 'elbo_neg_ave' in metrics:
        # Plot ELBO
        plt.figure()
        labels = []
        for key, item in experiment_results.items():
            if not item is None:
                plt.loglog(np.arange(iters), item['elbo_neg_ave']['mean'])
                labels.append(str(key))

        plt.legend(labels)
        plt.grid()
        plt.xlabel("Epoch")
        plt.ylabel("Neg. Average ELBO")
        plt.title("Dataset = " + data_set)
        plt.savefig("plots/final_plots/" + data_set + "/elbo.pdf")
        plt.close()
