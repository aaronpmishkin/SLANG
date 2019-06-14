import torch
import torch.nn.functional as F

import itertools
from lib.models.bayesian_mlp import BayesianMLP
from lib.utilities.general_utilities import construct_prior_vector
from lib.optimizers.slang import SLANG

import torchutils as tu
from torchutils.models import MLP as MultiSampleMLP

def copy_mlp_to_bmlp(mlp, bmlp, slang_opt):
    r"""
    (Inplace) copy of the parameters and variance of a Torchutil MLP
    and SLANG optimizer to a vi_lib BayesianMLP `bmlp`.

    Both MLPs need to have the same architecture.
    """
    plist = [x.clone() for x in mlp.parameters()]

    invert_soft_plus = torch.log(torch.exp(1.0/slang_opt.state['prec_diag'].clone().view(-1)) - 1)

    vlist = tu.params.bv2p(invert_soft_plus, mlp.parameters())

    bmlp_new_plist = itertools.chain.from_iterable(zip(plist, vlist))
    bmlp_new_pvec = tu.params.bp2v(bmlp_new_plist, 0)
    torch.nn.utils.vector_to_parameters(bmlp_new_pvec, bmlp.parameters())

def make_bmlp(data, model_params, optimizer_params, output_size):
    """
    Creates a Bayesian MLP that mimics the initialization procedure of the
    SLANG experiments by actually initializing a SLANG optimizer and MLP
    and copying parameters to a Bayesian MLP.
    """
    act_func = F.tanh if model_params['activation_function'] == "tanh" else F.relu

    if output_size is None:
        slang_output_size = 1  # Binary Classification
    else:
        slang_output_size = output_size


    slang_model = MultiSampleMLP(
        input_size=data.num_features,
        hidden_sizes=model_params['hidden_sizes'],
        output_size=slang_output_size,
        act_func=act_func
    )

    slang_prior_vector = construct_prior_vector(
        weight_prior=model_params['prior_precision'],
        bias_prior=model_params['bias_prior_precision'],
        named_parameters=slang_model.named_parameters()
    )

    slang_optimizer = SLANG(
        slang_model.parameters(),
        prior_prec=slang_prior_vector,
        s_init=optimizer_params['s_init'],
    )

    model = BayesianMLP(
        input_size=data.num_features,
        hidden_sizes=model_params['hidden_sizes'],
        output_size=output_size,
        act_func=model_params['activation_function'],
        prior_prec=model_params['prior_precision'],
        bias_prior_prec=model_params['bias_prior_precision']
    )

    copy_mlp_to_bmlp(slang_model, model, slang_optimizer)

    return model
