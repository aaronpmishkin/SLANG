# @Author: amishkin
# @Date:   18-08-17
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-08-30

from functools import partial

import torch
import torchutils as tu

def vi_closure_factory(x, y, objective, model, predict_fn, optimizer, num_samples):

    def closure():
        optimizer.zero_grad()
        logits = predict_fn(x, num_samples)
        loss = objective(logits, y)
        loss.backward()
        return loss

    return closure


### wrappers for curvfuncs closures; this should be expanded as needed ###

# We avoid using the predict_fn here because the SLANG optimizer is currently handling sampling
# by passing noise into the closure.

def individual_gradients_closure_factory(x, y, objective, model, predict_fn, optimizer, num_samples):
    def loss_fn(preds):
        if model.output_size == 1:
            preds = preds.reshape(num_samples, -1)
        return objective(preds, y)

    return tu.curvfuncs.closure_factory(model, x, loss_fn, ['grads'])
