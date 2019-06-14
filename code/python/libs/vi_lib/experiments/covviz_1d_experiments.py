import copy 
import pdb 

import lib.metrics.metric_factory as metric_factory

import lib.utilities.general_utilities as utils

import torch 

import experiments 
from experiments.base.bayes_by_backprop import bbb_base
from experiments.base.bayes_by_backprop_decay import bbb_decay_base
from experiments.base.slang_experiments import slang_base

__all__ = ["slang_covviz", "bbb_covviz"]

################################################################################
# Main optim params

EPOCHS = 10000

GAMMA = 10
BETA = 0.5
DECAY_GAMMA = utils.compute_decay_rate(GAMMA, 0.1, EPOCHS)
DECAY_BETA = utils.compute_decay_rate(1-BETA, 1-0.95, EPOCHS)

GAMMA_BBB = .1
DECAY_GAMMA_BBB = utils.compute_decay_rate(GAMMA_BBB, .01, EPOCHS)

FINE_TUNING = False

################################################################################
# Problem Def

MODEL_PARAMS = copy.deepcopy(experiments.default_parameters.MLP_REG_PARAMETERS)
MODEL_PARAMS["hidden_sizes"] = [10]
MODEL_PARAMS['prior_precision'] = .01
MODEL_PARAMS['noise_precision'] = .1
METRICS = metric_factory.BAYES_REG
OBJECTIVE = 'avneg_elbo_gaussian'
DATASETS = ['cov_viz_1d', 'cov_viz_1d_outlier', 'cov_viz_1d_gap']


def add_experiments_for_dataset(dataset):
    def slang_covviz():
        opt_params = { 'num_epochs': EPOCHS,
                       'batch_size': 30,
                       'learning_rate': GAMMA,
                       'momentum_rate': 0,
                       'train_mc_samples': 100,
                       'eval_mc_samples': 100,
                       's_init': 1,
                       'decay_params': { 
                            'learning_rate': DECAY_GAMMA, 
                            'beta': DECAY_BETA
                        },
                       'L': 1,
                       'beta': BETA }

        L1_params = copy.deepcopy(opt_params)
        L5_params = copy.deepcopy(opt_params)
        L10_params = copy.deepcopy(opt_params)
        L20_params = copy.deepcopy(opt_params)
        L5_params["L"] = 5
        L10_params["L"] = 10
        L20_params["L"] = 20

        if FINE_TUNING:
            def init_hook_for(exp_name):
                def hook(model, optimizer):
                    start_experiment = slang_base.get_variant(exp_name)
                    start_results = start_experiment.get_records()[-2].get_result()
                    model.load_state_dict(start_results['model'])
                    optimizer.state = start_results['optimizer']['state']
                    print(optimizer)
                return hook
        else:
            def init_hook_for(exp_name):
                return lambda x,y: None
                
        def iter_hook(model, optimizer, metric_hist):
            firstIter = not hasattr(optimizer, 'best_loss') 
            currLoss = metric_hist['elbo_neg_ave'][-1]
            
            if firstIter:
                optimizer.best_loss = currLoss
                optimizer.best_state = {}
                
            if currLoss <= optimizer.best_loss:
                optimizer.best_loss = currLoss
                for k in optimizer.state.keys():
                    optimizer.best_state[k] = torch.tensor(optimizer.state[k])
                    
        def end_hook(result_dict, model, optimizer):
            result_dict['optim_best_state'] = utils.cast_to_cpu(optimizer.best_state)

        exps = [
            "SLANG covviz L=1 "+dataset, 
            "SLANG covviz L=5 "+dataset, 
            "SLANG covviz L=10 "+dataset, 
            "SLANG covviz L=20 "+dataset
        ]
        
        params = [L1_params, L5_params, L10_params, L20_params]
        for (exp, param) in zip(exps, params):
            slang_base.add_variant(
                exp, data_set=dataset, 
                model_params=MODEL_PARAMS, optimizer_params=param, 
                metrics=METRICS, objective=OBJECTIVE, 
                init_hook=init_hook_for(exp),
                iter_hook=iter_hook,
                end_hook=end_hook
            )

    def bbb_covviz():
        opt_params = { 'num_epochs': EPOCHS,
                       'batch_size': 30,
                       'learning_rate': GAMMA_BBB,
                       'momentum_rate': 0.9,
                       'train_mc_samples': 25,
                       'eval_mc_samples': 25,
                       'decay_params': { 
                            'learning_rate': DECAY_GAMMA_BBB, 
                        },
                       's_init': 1,
                       'beta': .99 }

        if FINE_TUNING:
            def hook(model, optimizer):
                start_experiment = bbb_decay_base.get_variant("BBB covviz "+dataset)
                start_results = start_experiment.get_records()[-2].get_result()
                model.load_state_dict(start_results['model'])
        else:
            hook = lambda x,y: None 
        
        bbb_decay_base.add_variant(
            "BBB covviz "+dataset, data_set=dataset, 
            model_params=MODEL_PARAMS, optimizer_params=opt_params, 
            metrics=METRICS, objective=OBJECTIVE,
            init_hook=hook,
        )
        bbb_base.add_variant(
            "BBB covviz "+dataset, data_set=dataset, 
            model_params=MODEL_PARAMS, optimizer_params=opt_params, 
            metrics=METRICS, objective=OBJECTIVE,
            init_hook=hook,
        )

    """starting_optimization()"""
    bbb_covviz()
    slang_covviz()

for dataset in DATASETS:
    add_experiments_for_dataset(dataset)
    