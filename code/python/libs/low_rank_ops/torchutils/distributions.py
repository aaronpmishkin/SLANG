r"""Additional Probability Distributions

To avoid too much dependence between torchutils and torch, those distributions 
have a kl method that is only implemented between Torchutils distributions.
To computes :math:`KL(p||q)`, use

.. code-block:: python

    p = Distr(...)
    q = Distr(...)
    p.kl(q)

"""
import pdb

import torch
import torchutils.low_rank as lr

__all__ = [
    "LowRankMultivariateNormal", "MeanFieldMultivariateNormal"
]

def trace_mult(A, B):
    r"""Trace of :math:`AB`"""
    return torch.sum(A * B.t())

def trace_square(A):
    r"""Trace of :math:`A A^\top` (equal to the trace of :math:`A^\top A`)"""
    return torch.sum(A**2)

class TorchUtilsDistribution():
    r"""Base class for additional probability distribution"""

    def rsample(self, n_samples=1):
        r"""Returns samples from the distribution as a :math:`[D, n_\text{samples}]` Tensor
        """
        raise NotImplementedError

    def kl(p, q):
        r""":code:`p.kl(q)` computes the KL-divergence :math:`KL(p||q)`"""
        raise NotImplementedError

    def mean(self):
        r"""Returns the mean of the distribution"""
        raise NotImplementedError

class MultivariateNormal(TorchUtilsDistribution):
    r"""Base class for Multivariate Normal distributions.

    Contains some private methods shared by Multivariate Normal distributions.
    """

    def _prec_diag(self):
        r"""Returns the diagonal of the (computed) precision matrix"""
        raise NotImplementedError

    def _prec_mult(self, v):
        r"""Left-multiplies v by the precision matrix"""
        raise NotImplementedError

    def _logdet_cov(self):
        r"""Log-Determinant of the covariance matrix"""
        raise NotImplementedError

    def _base_kl(p, q):
        diff_mu = q.mean() - p.mean()
        diff_mu_projected = diff_mu.t() @ q._prec_mult(diff_mu)
        diff_det = q._logdet_cov() - p._logdet_cov()

        return diff_mu_projected + diff_det - p.mean().numel()

class LowRankMultivariateNormal(MultivariateNormal):
    r"""Creates a normal distribution a low-rank + diagonal covariance matrix
    parameterized by ``cov_factor`` and ``cov_diag``::

        covariance_matrix = cov_factor @ cov_factor.T + cov_diag

    or ``prec_factor`` and ``prec_diag``::

        covariance_matrix = (prec_factor @ prec_factor.T + prec_diag)^-1

    Example:
        Create a normal distribution with
        `mean=[0,0], cov_factor=[1,0], cov_diag=[1,1]`

        >>> mean = torch.zeros(2, 1)
        >>> U = torch.tensor([[1], [0]])
        >>> diag = torch.tensor([1, 1])
        >>> m = MultivariateNormal(mean, cov_factor=U, cov_diag=diag)

    Args:
        loc (Tensor): mean of the distribution with shape `d, 1`
        cov_factor (Tensor): *factor* part of low-rank form of *covariance*
            matrix with shape :math:`[D, K]`
        cov_diag (Tensor): *diagonal* part of low-rank form of *covariance*
            matrix with shape :math:`[D, 1]`
        prec_factor (Tensor): *factor* part of low-rank form of *precision*
            matrix with shape :math:`[D, K]`
        prec_diag (Tensor): *diagonal* part of low-rank form of *precision*
            matrix with shape :math:`[D, 1]`
    """

    def __init__(self, loc, cov_factor=None, cov_diag=None, prec_factor=None, prec_diag=None):
        self.use_cov = (cov_factor is not None and cov_diag is not None)
        self.loc = loc

        if self.use_cov:
            assert loc.shape[0] == cov_factor.shape[0]
            assert loc.shape[0] == cov_diag.shape[0]
            assert cov_diag.shape[1] == 1
            assert len(cov_factor.shape) == 2
            self.k = cov_factor.shape[1]
        else:
            assert loc.shape[0] == prec_factor.shape[0]
            assert loc.shape[0] == prec_diag.shape[0]
            assert prec_diag.shape[1] == 1
            assert len(prec_factor.shape) == 2
            self.k = prec_factor.shape[1]

        self.d = loc.shape[0]
        self.cov_factor = cov_factor
        self.cov_diag = cov_diag
        self.prec_factor = prec_factor
        self.prec_diag = prec_diag

    def rsample(self, n_samples=1):
        eps = torch.randn((self.d, n_samples), dtype=self.loc.dtype, device=self.loc.device)
        try:
            if self.use_cov:
                eps_k = torch.randn((self.k, n_samples), dtype=self.loc.dtype, device=self.loc.device)
                return self.loc + self.cov_factor @ eps_k + self.cov_diag.sqrt() * eps

            return self.loc.view(-1,1) + lr.invFactMult(self.prec_factor, self.prec_diag, eps)
        except RuntimeError as e:
            pdb.set_trace()

    def mean(self):
        return self.loc

    def kl(p, q):
        base_kl = p._base_kl(q)

        if p.use_cov:
            base_kl += trace_mult(p.cov_factor.t(), q._prec_mult(p.cov_factor))
            base_kl += torch.sum(q._prec_diag() * p.cov_diag)
        if not p.use_cov:
            A, B, v = lr.invFacts(p.prec_factor, p.prec_diag)
            base_kl += torch.sum(q._prec_diag() * v)
            base_kl -= trace_mult(q._prec_mult(A), B)
        return base_kl/2

    # UTILITIES FOR KL

    def _prec_mult(self, v):
        if self.use_cov:
            return lr.invMult(self.cov_factor, self.cov_diag, v)
        else:
            return lr.mult(self.prec_factor, self.prec_diag, v)

    def _prec_diag(self):
        if self.use_cov:
            return lr.invDiag(self.cov_factor, self.cov_diag)
        else:
            return lr.diag(self.prec_factor, self.prec_diag)

    def _logdet_cov(self):
        if self.use_cov:
            return lr.logdet(self.cov_factor, self.cov_diag)
        else:
            return - lr.logdet(self.prec_factor, self.prec_diag)


class MeanFieldMultivariateNormal(MultivariateNormal):
    r"""Creates a normal distribution with a diagonal covariance matrix

    Args:
        loc (Tensor): mean of the distribution with shape `d, 1`
        cov_diag (Tensor): *diagonal* of the *covariance*
            matrix with shape :math:`[D, 1]`
        prec_diag (Tensor): *diagonal* of the *precision*
            matrix with shape :math:`[D, 1]`
    """

    def __init__(self, loc, cov_diag=None, prec_diag=None):
        self.use_cov = cov_diag is not None
        self.loc = loc

        if self.use_cov:
            assert loc.shape[0] == cov_diag.shape[0]
            assert cov_diag.shape[1] == 1
        else:
            assert loc.shape[0] == prec_diag.shape[0]
            assert prec_diag.shape[1] == 1

        self.d = loc.shape[0]
        self.cov_diag = cov_diag
        self.prec_diag = prec_diag

    def rsample(self, n_samples=1):
        eps = torch.randn((self.d, n_samples), dtype=self.loc.dtype, device=self.loc.device)

        if self.use_cov:
            return self.loc + self.cov_diag.sqrt() * eps
        return self.loc.view(-1, 1) + eps * (1/self.prec_diag.sqrt())

    def mean(self):
        return self.loc

    def kl(p, q):
        base_kl = p._base_kl(q)

        if p.use_cov:
            base_kl += torch.sum(q._prec_diag() * p.cov_diag)
        if not p.use_cov:
            base_kl += torch.sum(q._prec_diag() / p.prec_diag)
        return base_kl/2

    # UTILITIES FOR KL

    def _prec_mult(self, v):
        if self.use_cov:
            return v / self.cov_diag
        return v * self.prec_diag

    def _prec_diag(self):
        if self.use_cov:
            return 1/self.cov_diag
        return self.prec_diag

    def _logdet_cov(self):
        if self.use_cov:
            return torch.sum(torch.log(self.cov_diag))
        return - torch.sum(torch.log(self.prec_diag))
