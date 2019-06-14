import torch
from torch.optim import Adam

from artemis.experiments import experiment_function, ExperimentFunction

from lib.data.datasets import DatasetCV, Dataset, DEFAULT_DATA_FOLDER
from lib.experiments.experiment import run_experiment
import lib.metrics.metric_factory as metric_factory
import lib.optimizers.closure_factories as closure_factories
from lib.plots.basics import plot_objective
from lib.utilities.mlp_to_bmlp import copy_mlp_to_bmlp, make_bmlp
import experiments.default_parameters as defaults
from lib.utilities.general_utilities import set_seeds


def init_bbb_copy_slang_experiment(data, model_params, optimizer_params, use_cuda=torch.cuda.is_available()):
    """ Initialization function for the bbb_copy_slang experiments.
    """
    output_size = data.num_classes
    if output_size == 2:
        output_size = None # Binary Classification

    # Initialize the model
    model = make_bmlp(data, model_params, optimizer_params, output_size)

    if use_cuda:
        model = model.cuda()

    def predict_fn(x, mc_samples):
        preds = [model(x) for _ in range(mc_samples)]
        return preds

    def kl_fn():
        return model.kl_divergence()

    # Use the closure that returns aggregated gradients
    closure_factory = closure_factories.vi_closure_factory

    # Initialize optimizer
    optimizer = Adam(model.parameters(),
                     lr=optimizer_params['learning_rate'],
                     betas=(optimizer_params['momentum_rate'], optimizer_params['beta']),
                     eps=1e-8)

    return model, predict_fn, kl_fn, closure_factory, optimizer

@ExperimentFunction(show=plot_objective)
def bbb_copy_slang(
        data_set='australian_presplit',
        model_params=defaults.MLP_CLASS_PARAMETERS,
        optimizer_params=defaults.BBB_COPYSLANG_PARAMETERS,
        objective='avneg_elbo_bernoulli',
        metrics=metric_factory.BAYES_BINCLASS,
        normalize={'x': False, 'y': False},
        save_params=defaults.SAVE_PARAMETERS,
        seed=123,
        use_cuda=torch.cuda.is_available(),
        init_hook=None):
    """
    Base class for BBB experiments that are to be initialized with the same model parameters as a specific SLANG experiment.
    BBB_COPYSLANG_PARAMETERS are modified BBB parameters that specify the initialization of the SLANG experiment in question.
    This experiment is useful for comparing the convergence rates of BBB and SLANG from the same initial model.
    """

    set_seeds(seed)

    data = Dataset(data_set=data_set, data_folder=DEFAULT_DATA_FOLDER)

    model, predict_fn, kl_fn, closure_factory, optimizer = init_bbb_copy_slang_experiment(data, model_params, optimizer_params, use_cuda=use_cuda)

    if init_hook is not None:
        init_hook(model, optimizer)

    results_dict = run_experiment(data, model, model_params, predict_fn, kl_fn, optimizer, optimizer_params, objective, metrics, closure_factory, normalize, save_params, seed, use_cuda)

    return results_dict
