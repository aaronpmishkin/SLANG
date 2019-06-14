# @Author: amishkin
# @Date:   18-08-17
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   amishkin
# @Last modified time: 18-08-17

import math
import numpy as np
from scipy.stats import truncnorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import Module

from torch.nn.utils import parameters_to_vector, vector_to_parameters

class BayesianDropout(nn.modules.dropout._DropoutNd):

    def forward(self, input):
        return F.dropout(input, self.p, True, self.inplace)

class BayesDropoutMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, act_func, p):
        super(type(self), self).__init__()
        self.input_size = input_size
        if output_size:
            self.output_size = output_size
            self.squeeze_output = False
        else :
            self.output_size = 1
            self.squeeze_output = True
        self.act = F.tanh if act_func == "tanh" else F.relu
        if len(hidden_sizes) == 0:
            self.hidden_layers = []
            self.output_layer = nn.Linear(self.input_size, self.output_size)
        else:
            self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
            self.dropout_layers = nn.ModuleList([BayesianDropout(p, size) for size in hidden_sizes])
            self.output_layer = nn.Linear(hidden_sizes[-1], self.output_size)

    def forward(self, x):
        '''
            x: The input patterns/features.
            individual_grads: Whether or not the activations tensors and linear
                combination tensors from each layer are returned. These tensors
                are necessary for computing the GGN using compute_ggn.goodfellow_ggn
        '''

        x = x.view(-1, self.input_size)
        out = x

        for layer, drop in zip(self.hidden_layers, self.dropout_layers):
            out = self.act(layer(out))
            out = drop(out)
        logits = self.output_layer(out)

        if self.squeeze_output:
            logits = torch.squeeze(logits)
        return logits
