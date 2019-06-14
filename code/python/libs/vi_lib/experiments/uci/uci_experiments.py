import copy
import math

from artemis.experiments import ExperimentFunction

from lib.data.datasets import DatasetCV, Dataset, DEFAULT_DATA_FOLDER
from lib.experiments.experiment import run_experiment
from lib.experiments.cv_experiment import run_cv_experiment
from lib.utilities.general_utilities import set_seeds

import experiments.default_parameters as defaults

from experiments.base.slang_experiments import init_slang_experiment
from experiments.base.bayes_by_backprop import init_bbb_experiment

from bayes_opt import BayesianOptimization
from sklearn.gaussian_process.kernels import Matern


MIN_SAVE_PARAMETERS = {
    'metric_history': True,
    'objective_history': False,
    'model': False,
    'optimizer': False }

UCI_BO_PARAMETERS = {
    'hidden_sizes': [50],
    'activation_function': 'relu',
    'prior_precision': None, # Will be tuned by BO
    'bias_prior_precision': 1e-6,
    'noise_precision': None } # Will be tuned by BO

UCI_SLANG_PARAMETERS = {
    'num_epochs': 120,
    'batch_size': 10,
    'learning_rate': 0.01,
    'momentum_rate': 0.9,
    'train_mc_samples': 4,
    'eval_mc_samples': 10,
    'beta': 0.99,
    'decay_params': {'learning_rate': 0.0, 'beta': 0.0 },
    's_init': 1,
    'L': 1 }

UCI_BBB_PARAMETERS = {
    'num_epochs': 120,
    'batch_size': 10,
    'learning_rate': 0.01,
    'momentum_rate': 0.9,
    'train_mc_samples': 4,
    'eval_mc_samples': 10,
    'beta': 0.99 }

@ExperimentFunction()
def uci_slang_bo(data_set='boston0',
             n_splits=5,
             model_params=UCI_BO_PARAMETERS,
             optimizer_params=UCI_SLANG_PARAMETERS,
               objective='avneg_loglik_gaussian',
               metrics=['avneg_elbo_gaussian', 'pred_avneg_loglik_gaussian', 'predictive_rmse'],
             normalize={'x': True, 'y': True},
             save_params=MIN_SAVE_PARAMETERS,
             param_bounds = {'log_noise_prec': (0, 5), 'log_prior_prec': (-4, 4)},
             gp_params = {'kernel': Matern(nu=2.5, length_scale=[1, 2]), 'alpha': 1e-2},
             bo_params = {'acq': 'ei', 'init_points': 5, 'n_iter': 25},
             seed=123,
             use_cuda=False):

    set_seeds(seed)
    data_cv = DatasetCV(data_set=data_set, n_splits=n_splits, seed=seed, data_folder=DEFAULT_DATA_FOLDER)

    def get_cv_average(dict_of_dicts, key):
        value = 0
        for split, result_dict in dict_of_dicts.items():
            value += result_dict["final_metrics"][key][0]
        value = value/len(dict_of_dicts)
        return value

    def cv_exp(log_noise_prec, log_prior_prec):

        model_params_cv = copy.deepcopy(model_params)
        model_params_cv['prior_precision'] = math.exp(log_prior_prec)
        model_params_cv['noise_precision'] = math.exp(log_noise_prec)

        try:
            # Run experiment
            results_dicts = run_cv_experiment(data_cv, n_splits, init_slang_experiment, model_params_cv, optimizer_params, objective, metrics, normalize, save_params, seed, use_cuda)

            # Return Avg. Test LL
            logloss = get_cv_average(results_dicts, key='test_pred_logloss')
            return -logloss
        except:
            logloss = 5.0
        return -logloss

    # Run BO
    bo = BayesianOptimization(cv_exp, param_bounds, random_state=seed)
    bo.maximize(init_points=bo_params['init_points'], n_iter=bo_params['n_iter'], acq=bo_params['acq'], **gp_params)

    # Run final experiment
    model_params_final = copy.deepcopy(model_params)
    model_params_final['prior_precision'] = math.exp(bo.res['max']['max_params']['log_prior_prec'])
    model_params_final['noise_precision'] = math.exp(bo.res['max']['max_params']['log_noise_prec'])

    data = Dataset(data_set=data_set, data_folder=DEFAULT_DATA_FOLDER)
    train_set_size=data.get_train_size()
    model, predict_fn, kl_fn, closure_factory, optimizer = init_slang_experiment(data, model_params_final, optimizer_params, train_set_size=train_set_size, use_cuda=use_cuda)
    save_params = defaults.SAVE_PARAMETERS
    results_dict = run_experiment(data, model, model_params_final, predict_fn, kl_fn, optimizer, optimizer_params, objective, metrics, closure_factory, normalize, save_params, seed, use_cuda)

    results = dict(final_run=results_dict, bo_results=bo.res)

    return results

@ExperimentFunction()
def uci_bbb_bo(data_set='boston0',
             n_splits=5,
             model_params=UCI_BO_PARAMETERS,
             optimizer_params=UCI_BBB_PARAMETERS,
               objective='avneg_elbo_gaussian',
               metrics=['avneg_elbo_gaussian', 'pred_avneg_loglik_gaussian', 'predictive_rmse'],
             normalize={'x': True, 'y': True},
             save_params=MIN_SAVE_PARAMETERS,
             param_bounds = {'log_noise_prec': (0, 5), 'log_prior_prec': (-4, 4)},
             gp_params = {'kernel': Matern(nu=2.5, length_scale=[1, 2]), 'alpha': 1e-2},
             bo_params = {'acq': 'ei', 'init_points': 5, 'n_iter': 25},
             seed=123,
             use_cuda=False):

    set_seeds(seed)
    data_cv = DatasetCV(data_set=data_set, n_splits=n_splits, seed=seed, data_folder=DEFAULT_DATA_FOLDER)

    def get_cv_average(dict_of_dicts, key):
        value = 0
        for split, result_dict in dict_of_dicts.items():
            value += result_dict["final_metrics"][key][0]
        value = value/len(dict_of_dicts)
        return value

    def cv_exp(log_noise_prec, log_prior_prec):

        model_params_cv = copy.deepcopy(model_params)
        model_params_cv['prior_precision'] = math.exp(log_prior_prec)
        model_params_cv['noise_precision'] = math.exp(log_noise_prec)

        try:
            # Run experiment
            results_dicts = run_cv_experiment(data_cv, n_splits, init_bbb_experiment, model_params_cv, optimizer_params, objective, metrics, normalize, save_params, seed, use_cuda)

            # Return Avg. Test LL
            logloss = get_cv_average(results_dicts, key='test_pred_logloss')
        except:
            logloss = 5.0
        return -logloss

    # Run BO
    bo = BayesianOptimization(cv_exp, param_bounds, random_state=seed)
    bo.maximize(init_points=bo_params['init_points'], n_iter=bo_params['n_iter'], acq=bo_params['acq'], **gp_params)

    # Run final experiment
    model_params_final = copy.deepcopy(model_params)
    model_params_final['prior_precision'] = math.exp(bo.res['max']['max_params']['log_prior_prec'])
    model_params_final['noise_precision'] = math.exp(bo.res['max']['max_params']['log_noise_prec'])

    data = Dataset(data_set=data_set, data_folder=DEFAULT_DATA_FOLDER)
    train_set_size=data.get_train_size()
    model, predict_fn, kl_fn, closure_factory, optimizer = init_bbb_experiment(data, model_params_final, optimizer_params, train_set_size=train_set_size, use_cuda=use_cuda)
    save_params = defaults.SAVE_PARAMETERS
    results_dict = run_experiment(data, model, model_params_final, predict_fn, kl_fn, optimizer, optimizer_params, objective, metrics, closure_factory, normalize, save_params, seed, use_cuda)

    results = dict(final_run=results_dict, bo_results=bo.res)

    return results
