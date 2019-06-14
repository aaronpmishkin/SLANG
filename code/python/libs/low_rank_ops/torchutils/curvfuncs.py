r"""Implementation of Goodfellow's trick 
to compute individual gradients from a minibatch and other quantities

Ref: https://arxiv.org/abs/1510.01799
"""

import torch

__all__ = [
    'closure_factory',
    'AVAILABLE_OPTIONS'
]

AVAILABLE_OPTIONS = [
    'grad', 'grad_pf',
    'grads', 'grads_pf',
    'diag_ggn', 'diag_ggn_pf'
]

def closure_factory(model, x, loss, outputs):
    r"""Creates closues for optimizers based on curvature information.
    
    The closure will take :code:`noise` as an argument, and pass it
    to the model before computing the loss; :code:`loss(model(x, noise))`.

    The closure returns the loss as the first argument.
    The remaining outputs depend on the content of outputs, which should be
    a list of string matching the :code:`AVAILABLE_OPTIONS`:

    .. code-block:: python

        {0}

    .. warning::

        :code:`loss` needs to be able to broadcast through the :code:`noise`
        batch dimension.
        If :code:`noise` is a [S x D] tensor, 
        :code:`x` is a [N x ...] tensor 
        :code:`model(x, noise)` will be a [S x N x ...] tensor.

    Arguments:
        model (Torch Model): A differentiable function 
        x (Tensor): The inputs to the model 
        loss (function): A function that returns the loss.
          will be called using loss(model(x))

    """
    assert all([name in AVAILABLE_OPTIONS for name in outputs])
    closures = [MAPPING[name] for name in outputs]

    def closure(noise=None):
        outputs, activations, linear_combs = model(x, noise, indgrad=True)

        linear_grads = torch.autograd.grad(loss(outputs), linear_combs, retain_graph=True)

        return tuple([loss(outputs)] + [c(activations, linear_grads) for c in closures])

    return closure

closure_factory.__doc__ = closure_factory.__doc__.format(AVAILABLE_OPTIONS)

def grad_pf(activations, linear_grads):
    r"""Return the overall gradient in parameter format"""
    grads = []
    for G, X in zip(linear_grads, activations):

        if len(G.shape) == 3:
            gW = (G.transpose(1, 2) @ X).sum(dim=0)
            gB = G.sum(dim=[0, 1])
        else:
            if len(G.shape) < 2:
                G = G.unsqueeze(1)
            gW = G.t() @ X
            gB = G.sum(dim=0)

        grads.append(gW)
        grads.append(gB)
    return grads

def grads_pf(activations, linear_grads):
    r"""Return individual gradients in parameter format"""
    grads = []

    for G, X in zip(linear_grads, activations):

        if len(G.shape) == 3:
            if len(X.shape) == 2:
                gW = torch.einsum(
                    '...ij,...jk->...ik',
                    (G.unsqueeze(3), X.unsqueeze(1).unsqueeze(0).expand(G.shape[0], -1, -1, -1))
                )
            elif len(X.shape) == 3:
                gW = torch.einsum('...ij,...jk->...ik', (G.unsqueeze(3), X.unsqueeze(2)))
        else:
            if len(G.shape) < 2:
                G = G.unsqueeze(1)
            gW = torch.bmm(G.unsqueeze(2), X.unsqueeze(1))
        gB = G.unsqueeze(len(G.shape))

        grads.append(gW)
        grads.append(gB)
    return grads

def diag_ggn_pf(activations, linear_grads):
    r"""Return the diagonal of the GGN in parameter format"""
    raise NotImplementedError

def flatten_last_dim(params):
    mats = []
    for p in params:
        newshape = list(p.shape)[:-2] + [-1]
        mats.append(p.view(newshape))
    return torch.cat(mats, dim=-1)

def flatten_last_dim_(func):
    return lambda *args: flatten_last_dim(func(*args))

def grad(activations, linear_grads):
    r"""Return the overall gradient as a matrix"""
    return flatten_last_dim_(grad_pf)(activations, linear_grads)

def grads(activations, linear_grads):
    r"""Return individual gradients as a matrix"""
    return flatten_last_dim_(grads_pf)(activations, linear_grads)

def diag_ggn(activations, linear_grads):
    r"""Return the diagonal of the GGN as a matrix"""
    return flatten_last_dim_(diag_ggn_pf)(activations, linear_grads)

MAPPING = {
    'grad_pf': grad_pf,
    'grads_pf': grads_pf,
    'diag_ggn_pf': diag_ggn_pf,
    'grad': grad,
    'grads': grads,
    'diag_ggn': diag_ggn,
}
