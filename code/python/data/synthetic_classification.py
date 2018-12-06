# @Author: aaronmishkin
# @Date:   18-11-26
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-11-26

import math
import torch

# Data taken from "Machine Learning: A Probabilistic Perspective" by Kevin P. Murphy

def generate_data(N=60, std1=1, std2=1.1):
    ''' Generate a binary classification dataset where the features for the two classes are
        sampled from different 2-dimensional Gaussians.
    '''
    mid_index = math.floor(N/2)

#   define different mean vectors for the two different classes.
    mu1 = torch.stack([torch.ones(mid_index), torch.ones(N-mid_index)*5],dim=1)
    mu2 = torch.stack([torch.ones(mid_index)*-5, torch.ones(N-mid_index)], dim=1)

#   sample features from normal distributions.
    X = torch.cat([
        torch.normal(mean=mu1, std=std1),
        torch.normal(mean=mu2, std=std2)])

#   class labels are in the set {0,1}.
    y = torch.ones(N)
    y[mid_index:N] = y[mid_index:N] - 1

    return X, y
