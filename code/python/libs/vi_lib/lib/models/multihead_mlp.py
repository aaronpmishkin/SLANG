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

class MultiheadMLP(nn.Module):
    # Pass number of classes per task
    def __init__(self, input_size, hidden_sizes, output_size, act_func, no_classes, is_multihead):
        super(MultiheadMLP, self).__init__()
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
            self.output_layer = nn.Linear(hidden_sizes[-1], self.output_size)

            for l in self.hidden_layers:
                l.weight.data = torch.from_numpy(truncnorm.rvs(a=-2, b=2, loc=0., scale=.1, size=l.weight.data.shape).astype(np.float32))
                l.bias.data = torch.from_numpy(truncnorm.rvs(a=-2, b=2, loc=0., scale=.1, size=l.bias.data.shape).astype(np.float32))
            self.output_layer.weight.data = torch.from_numpy(
                truncnorm.rvs(a=-2, b=2, loc=0., scale=.1, size=self.output_layer.weight.data.shape).astype(np.float32))
            self.output_layer.bias.data = torch.from_numpy(
                truncnorm.rvs(a=-2, b=2, loc=0., scale=.1, size=self.output_layer.bias.data.shape).astype(np.float32))

            print(self.output_layer.weight.data)
            print(self.output_layer.bias.data)

        self.no_classes = no_classes
        self.is_multihead = is_multihead
        self.hidden_sizes = hidden_sizes

    def forward(self, x, task_id=0, individual_grads=False):
        '''
            x: The input patterns/features.
            individual_grads: Whether or not the activations tensors and linear
                combination tensors from each layer are returned. These tensors
                are necessary for computing the GGN using compute_ggn.goodfellow_ggn
        '''

        # return super(MultiheadMLP, self).forward(x)

        x = x.view(-1, self.input_size)
        out = x
        # Save the model inputs, which are considered the activations of the
        # 0'th layer.
        if individual_grads:
            H_list = [out]
            Z_list = []

        for layer in self.hidden_layers:
            Z = layer(out)
            out = self.act(Z)

            # Save the activations and linear combinations from this layer.
            if individual_grads:
                H_list.append(out)
                Z.retain_grad()
                Z.requires_grad_(True)
                Z_list.append(Z)

        # print('out')
        # print(out)
        # print('output_layer')
        # print(self.output_layer)
        # print('weight')
        # print(self.output_layer.weight)
        # print('bias')
        # print(self.output_layer.bias)
        logits = self.output_layer(out)
        # print('logits')
        # print(logits)
        if self.is_multihead:
            logits = logits[:, task_id * self.no_classes : (task_id + 1) * self.no_classes]

        if self.squeeze_output:
            logits = torch.squeeze(logits)

        # Save the final model outputs, which are the linear combinations
        # from the final layer.
        if individual_grads:
            logits.retain_grad()
            logits.requires_grad_(True)
            Z_list.append(logits)

        if individual_grads:
            return (logits, H_list, Z_list)

        return logits

    # Add units for additional classes in the final layer.
    # Required for continual learning with multihead network
    def add_head(self, no_add_outputs, use_cuda=False):
        self.output_size += no_add_outputs
        if len(self.hidden_sizes) == 0:
            in_size = self.input_size
        else:
            in_size = self.hidden_sizes[-1]

        if use_cuda:
            self.output_layer.weight = torch.nn.Parameter(torch.cat([
                self.output_layer.weight,
                torch.tensor(np.random.normal(
                    0., .1, (no_add_outputs, in_size)).astype(np.float32)).cuda()],
                dim=0))

            self.output_layer.bias = torch.nn.Parameter(torch.cat([
                self.output_layer.bias,
                torch.tensor(np.random.normal(
                    0., .1, (no_add_outputs)).astype(np.float32)).cuda()],
                dim=0))
        else:
            self.output_layer.weight = torch.nn.Parameter(torch.cat([
                self.output_layer.weight,
                torch.tensor(np.random.normal(
                    0., .1, (no_add_outputs, in_size)).astype(np.float32))],
                dim=0))

            self.output_layer.bias = torch.nn.Parameter(torch.cat([
                self.output_layer.bias,
                torch.tensor(np.random.normal(
                    0., .1, (no_add_outputs)).astype(np.float32))],
                dim=0))
