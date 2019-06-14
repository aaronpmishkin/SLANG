# @Author: amishkin
# @Date:   18-08-17
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   amishkin
# @Last modified time: 18-08-17

import torch
import numpy as np
import math
from torch.nn.utils import parameters_to_vector

def construct_prior_vector(weight_prior, bias_prior, named_parameters):
    # Obtain **only** the network (i.e mean) parameters
    prior_list = []
    offest = 0
    for i,p in enumerate(named_parameters):
        if 'bias' in p[0]:
            prior_list.append(torch.ones_like(p[1]).mul(bias_prior))
        else:
            prior_list.append(torch.ones_like(p[1]).mul(weight_prior))

    return parameters_to_vector(prior_list)

def decay_rates(init_learning_rates, decay_params, i):
    decayed_rates = {}
    for key, init_rate in init_learning_rates.items():

        if decay_params[key] > 0:
            decayed_rates[key] = init_rate / (1 + i**(decay_params[key]))
        else:
            decayed_rates[key] = init_rate

    return decayed_rates

def set_seeds(seed=123, use_cuda=torch.cuda.is_available()):
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

def cast_to_cpu(p):
    if isinstance(p, dict):
        return cast_dict_to_cpu(p)
    elif isinstance(p, list):
        return cast_list_to_cpu(p)
    elif torch.is_tensor(p) or isinstance(p, (torch.nn.Parameter, torch.nn.ParameterList)):
        return p.cpu()

    return p

def cast_list_to_cpu(p):
    for i, item in enumerate(p):
        p[i] = cast_to_cpu(item)

    return p

def cast_dict_to_cpu(p):
    for _, key in enumerate(p):
        p[key] = cast_to_cpu(p[key])

    return p

def compute_decay_rate(start, end, epochs):
    """
    Computes the decay rate required to go to end, from start, in epochs,
    such that `end = start/epochs**decay_rate`
    """
    return math.log(start/end)/math.log(epochs)
