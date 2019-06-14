"""
(Local) Runner for the cov. viz. experiments and plotting code.
"""

import itertools
import pdb

import cli
import settings

import numpy as np
import torch
import torch.nn.functional as F

import experiments.base.slang_experiments as slang
import experiments.base.bayes_by_backprop as bbb
import experiments.base.bayes_by_backprop_decay as bbb_decay
import experiments.covviz_1d_experiments as covviz

import lib
from lib.data.datasets import Dataset
from lib.utilities.general_utilities import construct_prior_vector
from lib.optimizers.slang import SLANG
from lib.models.bayesian_mlp import BayesianMLP
from torchutils.models import MLP as MultiSampleMLP
import lib.utilities.plotting as plotutils

import matplotlib
import matplotlib.pyplot as plt

np.random.seed(settings.SEED)
torch.manual_seed(settings.SEED)
matplotlib.rcParams.update({'font.size': settings.GLOBAL_FONT_SIZE})

def load_prediction_function_bbb(model_params, results):
    r"""Load the prediction function for BBB"""
    model = BayesianMLP(input_size=1,
            hidden_sizes=model_params['hidden_sizes'],
            output_size=None,
            act_func=model_params['activation_function'],
            prior_prec=model_params['prior_precision'],
            bias_prior_prec=model_params['bias_prior_precision'])
    model.load_state_dict(results['model'])

    def predict_fn(x, mc_samples):
        preds = [model(x).view(-1,1) for _ in range(mc_samples)]
        return torch.cat(preds, dim=1).detach().numpy()

    return predict_fn

def load_prediction_function_slang(model_params, optimizer_params, results, best=False):
    r"""Load the prediction function for SLANG """
    act_func = F.tanh if model_params['activation_function'] == "tanh" else F.relu

    model = MultiSampleMLP(input_size=1,
                           hidden_sizes=model_params['hidden_sizes'],
                           output_size=1,
                           act_func=act_func)

    prior_vector = construct_prior_vector(weight_prior=model_params['prior_precision'],
                              bias_prior=model_params['bias_prior_precision'],
                              named_parameters=model.named_parameters())

    optimizer = SLANG(model.parameters(),
                      lr=optimizer_params['learning_rate'],
                      betas=(optimizer_params['momentum_rate'], optimizer_params['beta']),
                      prior_prec=prior_vector,
                      s_init=optimizer_params['s_init'],
                      decay_params=optimizer_params['decay_params'],
                      num_samples=optimizer_params['train_mc_samples'],
                      L=optimizer_params['L'],
                      train_set_size=30)

    model.load_state_dict(results['model'])

    if best:
        optimizer.load_state_dict({
            'state': results['optim_best_state'], 
            'param_groups': results['optimizer']['param_groups']
        })
    else:
        optimizer.load_state_dict(results['optimizer'])

    def predict_fn(x, mc_samples):
        noise = optimizer.distribution().rsample(mc_samples).t()
        preds = model(x.view(-1,1), noise, False).squeeze(dim=2).t().detach().numpy()
        return preds

    return predict_fn

def load_prediction_function(name, experiment, best=False):
    r"""Loads the prediction function found at the end of the experiments."""
    model_params = experiment.get_args()['model_params']
    results = experiment.get_records()[settings.RECORD_ID].get_result()

    if name.startswith("BBB"):
        pred_fun = load_prediction_function_bbb(model_params, results)
    else:
        optimizer_params = experiment.get_args()['optimizer_params']
        pred_fun = load_prediction_function_slang(model_params, optimizer_params, results, best)

    return pred_fun

def plt_dataset(dataset, ax=None):
    r"""
    Plots the cov. viz. dataset on the axis, or a new one if None is passed.
    """
    data = Dataset(data_set=dataset)
    x, y = data.load_full_train_set()
    x, y = x.cpu().numpy(), y.cpu().numpy()
    if ax is None:
        plt.plot(x, y, '.')
    else:
        ax.plot(x, y, '.')

def appendix_plot(dataset, experiments, with_gaussian_noise=False):
    FIGSCALE = 4
    fig = plt.figure(figsize=(4*FIGSCALE, 2*FIGSCALE))

    xlim, ylim = 8, 200

    gs =  matplotlib.gridspec.GridSpec(2,3,width_ratios=[2,2,2], height_ratios=[1,1])
    
    axes1, axes2 = [], []
    axes1.append(fig.add_subplot(gs[0,0]))
    axes1.append(fig.add_subplot(gs[0,1]))
    axes1.append(fig.add_subplot(gs[0,2]))
    
    axes2.append(fig.add_subplot(gs[1,0]))
    axes2.append(fig.add_subplot(gs[1,1]))
    axes2.append(fig.add_subplot(gs[1,2]))

    x = np.linspace(-xlim, xlim, settings.LINSPACE_PREC)

    axId = 0
    for k, v in experiments.items():
        predfunc = load_prediction_function(k, v, best=True)
        predfunc2 = load_prediction_function(k, v, best=False)
        
        y = predfunc(torch.Tensor(x), settings.MC_SAMPLES)
        y2 = predfunc2(torch.Tensor(x), settings.MC_SAMPLES)
        if with_gaussian_noise:
            eps = np.random.normal(0, 10, size=y.shape)
            y += eps
            y2 += eps

        plt_dataset(dataset, axes1[axId])
        if axId > 0:
            plt_dataset(dataset, axes2[axId])
        
        axes1[axId].plot(x, np.mean(y2, axis=1))
        if axId > 0:
            axes2[axId].plot(x, np.mean(y, axis=1))
        for std_scaling in [1,2,3]:
            upper = np.mean(y, axis=1)+std_scaling*(np.std(y, axis=1))
            lower = np.mean(y, axis=1)-std_scaling*(np.std(y, axis=1))
            axes1[axId].fill_between(x, lower, upper, alpha=0.2, color=[.5,.5,.5], linewidth=0, edgecolor=[1,1,1])
            if axId > 0:
                upper = np.mean(y2, axis=1)+std_scaling*(np.std(y2, axis=1))
                lower = np.mean(y2, axis=1)-std_scaling*(np.std(y2, axis=1))
                axes2[axId].fill_between(x, lower, upper, alpha=0.2, color=[.5,.5,.5], linewidth=0, edgecolor=[1,1,1])

        axes1[axId].set_title(settings.METHOD_DISPLAY_TITLE[k], fontweight=settings.FONTWEIGHT)
        axes1[axId].set_xlim([-xlim,xlim])
        axes1[axId].set_ylim([-ylim,ylim])
        if axId > 0:
            axes2[axId].set_xlim([-xlim,xlim])
            axes2[axId].set_ylim([-ylim,ylim])

            
        axes1[axId].set_yticks([-200,-100,0,100,200])
        axes1[axId].set_xticks([-8,-4,0,4,8])
        if axId > 0:
            axes2[axId].set_xticks([-8,-4,0,4,8])
            axes2[axId].set_yticks([-200,-100,0,100,200])

        axId += 1
    
    axes1[1].set_title("SLANG (L=1, best model)", fontweight=settings.FONTWEIGHT)
    axes1[2].set_title("SLANG (L=5, best model)", fontweight=settings.FONTWEIGHT)
    axes2[1].set_title("SLANG (L=1, last model)", fontweight=settings.FONTWEIGHT)
    axes2[2].set_title("SLANG (L=5, last model)", fontweight=settings.FONTWEIGHT)

    for ax in axes1[0], axes1[1], axes1[2], axes2[1], axes2[2]:
        ax.set_xlabel("x")
    axes1[0].set_ylabel("y")
    
    for k, v in experiments.items():
        axes2[0].plot(
            v.get_records()[settings.RECORD_ID].get_result()['metric_history']['elbo_neg_ave'],
            label=k,
            alpha=1,
            linewidth=2
        )
    axes2[0].legend(framealpha=1, fancybox=True)
    axes2[0].set_xlim([8000,10000])
    axes2[0].set_ylim([5,7])
    axes2[0].set_yticks([5,6,7])
    axes2[0].set_xticks([8000,9000,10000])
    axes2[0].set_title("Convergence", fontweight=settings.FONTWEIGHT)
    axes2[0].set_xlabel("Iteration")
    axes2[0].set_ylabel("ELBO")
    
    for k, v in experiments.items():
        if "BBB" in k:
            continue 
            
        elbos = np.array(v.get_records()[settings.RECORD_ID].get_result()['metric_history']['elbo_neg_ave'])
        minx = np.argmin(elbos[8000:])
        miny = elbos[8000+minx]
        axes2[0].scatter([8000+minx], [miny], s=100, marker='*', c=[0,0,0,0], linewidths=1, edgecolors=[0,0,0,1], zorder=10)

    fig.tight_layout()
    
def main_plot(dataset, experiments, with_gaussian_noise=True):

    FIGSCALE = 3
    fig = plt.figure(figsize=(6*FIGSCALE, 2*FIGSCALE))

    xlim, xlim2 = 7, 3
    ylim, ylim2 = 200, 20

    gs1 = matplotlib.gridspec.GridSpec(1,3,width_ratios=[2,2,2], height_ratios=[1])
    gs1.update(bottom=0.58, top=0.95, wspace=0.15)

    gs2 = matplotlib.gridspec.GridSpec(1,3,width_ratios=[2,2,2], height_ratios=[1])
    gs2.update(bottom=0.1, top=0.45, wspace=0.15)
    
    axes1, axes2 = [], []
    axes1.append(fig.add_subplot(gs1[0,0]))
    axes1.append(fig.add_subplot(gs1[0,1]))
    axes1.append(fig.add_subplot(gs1[0,2]))
    axes2.append(fig.add_subplot(gs2[0,0]))
    axes2.append(fig.add_subplot(gs2[0,1]))
    axes2.append(fig.add_subplot(gs2[0,2]))

    x = np.linspace(-xlim, xlim, settings.LINSPACE_PREC)

    axId = 0
    for k, v in experiments.items():
        predfunc = load_prediction_function(k, v, best=True)
        
        y = predfunc(torch.Tensor(x), settings.MC_SAMPLES)
        if with_gaussian_noise:
            y += np.random.normal(0, 10, size=y.shape)

        plt_dataset(dataset, axes1[axId])
        plt_dataset(dataset, axes2[axId])
        
        axes1[axId].plot(x, np.mean(y, axis=1))
        axes2[axId].plot(x, np.mean(y, axis=1))
        for std_scaling in [1,2,3]:
            upper = np.mean(y, axis=1)+std_scaling*(np.std(y, axis=1))
            lower = np.mean(y, axis=1)-std_scaling*(np.std(y, axis=1))
            axes1[axId].fill_between(x, lower, upper, alpha=0.2, color=[.5,.5,.5], linewidth=0, edgecolor=[1,1,1])
            axes2[axId].fill_between(x, lower, upper, alpha=0.2, color=[.5,.5,.5], linewidth=0, edgecolor=[1,1,1])

        axes1[axId].set_title(settings.METHOD_DISPLAY_TITLE[k], fontweight=settings.FONTWEIGHT)
        axes1[axId].set_xlim([-xlim,xlim])
        axes1[axId].set_ylim([-ylim,ylim])
        
        axes1[axId].add_patch(matplotlib.patches.Rectangle(
            (-xlim2, -ylim2), 2*xlim2, 2*ylim2,
            fill=False, clip_on=False,
            linestyle='--',
            linewidth=0.5,
            color=[.5,.5,.5]
        ))

        axes2[axId].set_xlim([-xlim2, xlim2])
        axes2[axId].set_ylim([-ylim2, ylim2])
        
        axes1[axId].set_yticks([-200,-100,0,100,200])
        axes2[axId].set_yticks([-20,-10,0,10,20])
        
        if axId > 0:
            axes1[axId].yaxis.set_ticklabels([])
            axes2[axId].yaxis.set_ticklabels([])
            
        axId += 1
    axes1[0].set_ylabel("y")
    axes2[0].set_ylabel("y")
    for ax in axes2:
        ax.set_xlabel("x")

if __name__ == "__main__":
    args = cli.parse_args()

    def plot_and_save_as_needed(name):
        if args.save:
            print("Saving "+name)
            plt.savefig(name, bbox_inches='tight')
        if not args.noshow:
            plt.show()
        plt.close()
    
    dataset = settings.DATASETS[args.dataset_id]
    
    experiments = {
        "BBB": bbb_decay.bbb_decay_base.get_variant("BBB covviz "+dataset),
        "SLANG L=1":slang.slang_base.get_variant("SLANG covviz L=1 "+dataset),
        "SLANG L=5":slang.slang_base.get_variant("SLANG covviz L=5 "+dataset),
    }
    
    if args.run:
        for k, v in experiments.items():
            v.run()

    appendix_plot(dataset, experiments, with_gaussian_noise=False)
    plot_and_save_as_needed('covviz1d_appendix.pdf')
    
    main_plot(dataset, experiments, with_gaussian_noise=False)
    plot_and_save_as_needed('covviz1d_main.pdf')
