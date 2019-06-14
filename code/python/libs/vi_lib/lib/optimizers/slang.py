r"""
"""

import warnings
import math, pdb

import torch
from torch.nn.utils import parameters_to_vector as p2v
from torch.nn.utils import vector_to_parameters as v2p
import torchutils as tu

import lib
from lib.utilities.general_utilities import decay_rates, cast_to_cpu

mm = torch.mm
addmm = torch.addmm

THRESHOLD_EIGENVALUES = True
EPS = 1e-6

class SLANG(torch.optim.Optimizer):
    r"""SLANG
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), prior_prec=1.0,
                 s_init=1.0, decay_params={ 'learning_rate': 0.55, 'beta': 0.55},
                 num_samples=1, L=1, train_set_size=None):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if prior_prec.lt(0).sum() > 0:
            raise ValueError("Invalid prior precision value: {}".format(prior_prec))
        if s_init < 0.0:
            raise ValueError("Invalid initial s value: {}".format(s_init))
        if num_samples < 1:
            raise ValueError("Invalid num_samples parameter: {}".format(num_samples))
        if L < 1:
            raise ValueError("Invalid L parameter: {}".format(L))
        if train_set_size is None:
            warnings.warn("Warning: train_set_size = None, objective will be assumed to be an unbiased log-likelihood estimate, i.e. (N/M)*sum_m=1^M(log p_m). If objective is of the form (1/M)*sum_m=1^M(log p_m), set train_set_size = N")
            train_set_size = 1
        if train_set_size < 1:
            raise ValueError("Invalid number of training data points: {}".format(train_set_size))

        defaults = dict(lr=lr, betas=betas, prior_prec=prior_prec, s_init=s_init, decay_params=decay_params, num_samples=num_samples, L=L, train_set_size=train_set_size)
        super(SLANG, self).__init__(params, defaults)

        if len(self.param_groups) > 1:
            raise ValueError("")

        num_params = tu.params.num_params(self.param_groups[0]['params'])
        dtype = self.param_groups[0]['params'][0].dtype
        device = self.param_groups[0]['params'][0].device

        self.state['mean'] = p2v(self.param_groups[0]['params'])
        self.state['prec_factor'] = torch.zeros(
            num_params, L,
            dtype=dtype, device=device
        )
        self.state['prec_diag'] = s_init*torch.ones(
            num_params, 1,
            dtype=dtype, device=device
        )
        # momentum term
        self.state['momentum_grad'] = torch.zeros(num_params, 1, dtype=dtype, device=device)
        self.state['step'] = 0

    def __pca_update(self, U, V, b):

        def matvecprod(v):
            return addmm(mm(U, mm(U.t(), v)), V, mm(V.t(), v), beta=b, alpha=1-b)

        e, V = tu.fastpca.eigsh_func(
            matvecprod,
            dtype=U.dtype, device=U.device, n=U.shape[0],
            k=self.defaults["L"]
        )

        if THRESHOLD_EIGENVALUES:
            mask = e > EPS
            if (mask == 0).all():
                return torch.zeros([1,1], dtype=V.dtype, device=V.device)
            return V[:,mask] * torch.sqrt(e[mask])
        else:
            return V * torch.sqrt(e)


    def step(self, closure):
        r"""
        Use with ``torchutils.curvfuncs(..., 'grads')``
        """
        noise = self.distribution().rsample(self.defaults['num_samples']).t()

        loss, grads = closure(noise)
        grads = grads.view(-1, grads.shape[-1]).t()

        with torch.no_grad():
            # increment the step counter
            self.state['step'] = self.state['step'] + 1
            grad = grads.sum(dim=1)

            # The correct scaling of both the gradient and EF is N / MS,
            # where N is the training set size, M is the minibatch size, and S
            # is the number of samples used in the MC integral.
            # Gradients of the mean log-likelihood are computed by .backward(),
            # which means that the gradients are computed with scaling (1 / MS).

            # For the mean gradient, we need only multiply by N to obtain the correct scaling.
            scaling = self.defaults['train_set_size']
            # scale the gradient
            grad.mul_(scaling)

            # scale the EF by scaling the individual gradients by sqrt(N / MS).
            # The EF is the outerproduce of the individual gradients, so this yields
            # the correct scaling.
            MS = (grads.size()[1])
            grads.mul_(math.sqrt(MS * scaling))

            # Decay the learning rates
            init_learning_rates = {'beta': (1-self.defaults['betas'][1]), 'learning_rate': self.defaults['lr']}
            decayed_lrs = decay_rates(init_learning_rates, self.defaults['decay_params'], self.state['step'])
            # momentum beta
            mb = self.defaults['betas'][0]
            # slang beta
            b = (1-decayed_lrs['beta'])
            lr = decayed_lrs['learning_rate']
            prior_prec = self.defaults['prior_prec']
            U = self.state['prec_factor']
            D = self.state['prec_diag']
            mu = self.state['mean']

            # update the momentum term
            self.state['momentum_grad'] = self.state['momentum_grad'].mul(mb) + grad.view(-1, 1).mul((1-mb))
            # bias correct the momentum term
            bc_mg = self.state['momentum_grad'] / (1 - mb**self.state['step'])

            diag_should_be = (U**2).sum(dim=1).mul(b) + (grads**2).sum(dim=1).mul((1-b))
            # this operation produces a new object reference
            U = self.__pca_update(U, grads, b)
            # assign the new object reference to the optimizer state.
            self.state['prec_factor'] = U
            diag_corr = diag_should_be - (U**2).sum(dim=1)

            # Diagonal correction is theoretically positive,
            # as all eigendirections are not captured,
            # but it is possible for it to be negative due to noise,
            # which can lead the diagonal to become 0 or negative
            # if `(1-b)*prior_prec` is very small.
            diag_corr[diag_corr < 0] = 0

            D.mul_(b).add_(prior_prec.view(-1,1).mul((1-b)) + diag_corr.view(-1,1))
            # compute the direction using the momentum gradient
            direction = bc_mg.view(-1,1) + prior_prec.view(-1,1)*mu.view(-1,1)

            adapted_direction = tu.low_rank.invMult(U, D, direction)
            mu.add_(adapted_direction.view(-1).mul(-lr))

            v2p(mu, self.param_groups[0]['params'])

        return loss

    def distribution(self):
        r"""
        """
        U = self.state['prec_factor']
        D = self.state['prec_diag']

        if (self.state['prec_factor'] == 0).all():
            return tu.distributions.MeanFieldMultivariateNormal(loc=self.state['mean'].view(-1,1), prec_diag=D)
        return tu.distributions.LowRankMultivariateNormal(
            loc=self.state['mean'].view(-1,1),
            prec_factor=U,
            prec_diag=D
        )

    def kl_divergence(self):
        r""" Computes and returns the KL divergence KL(q||p), where:
            q is the variational distribution; and
            p is the prior distribution
        """

        q = self.distribution()
        p_mu = torch.zeros_like(self.state['mean'])
        p = tu.distributions.MeanFieldMultivariateNormal(loc=p_mu.view(-1,1), prec_diag=self.defaults['prior_prec'].view(-1,1))
        kl = q.kl(p)

        return kl

    def cpu(self):
        r""" Move the optimizer's internal representation to CPU.
        """
        self.state = cast_to_cpu(self.state)
        self.defaults = cast_to_cpu(self.defaults)
        self.param_groups = cast_to_cpu(self.param_groups)

        return self
