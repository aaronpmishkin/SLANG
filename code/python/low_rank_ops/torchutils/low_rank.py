r"""Linear Algebra routines for Low-Rank + Diagonal Matrix operations

This module implements operations involving a matrix :math:`A` that can be written as
:math:`UU^\top + D`, where :math:`U` is a low rank :math:`[n \times k]` matrix,
:math:`k \ll n`, and :math:`D` is a diagonal :math:`[n \times n]` matrix.

Those can be implemented more efficiently when working with :math:`U, D` directly
instead of computing :math:`A`, especially if :math:`n` is large
and :math:`A` might not fit into memory.
"""

# Disable naming scheme error for math symbols (U, d, k, ...)
# pylint: disable=C0103

import torch
import warnings

__all__ = [
    "mult", "invMult", "factMult", "invFactMult", "logdet", "trace"
]

NOT_FULL_RANK_ERR_MSG = "Lapack Error in potrf : the leading minor of order "

def reduceRank(A):
    r"""Reduces a symmetric factorization to a full rank symmetric factorization.

    Given a matrix :math:`U` of size :math`[n \times k]`, returns a matrix :math:`W`
    of size :math:`[n \times r(U)]`, where :math:`r(U)` is the rank of :math:`U`,
    such that :math:`WW`\top = UU^\top`.
    """
    
    (e, V) = torch.symeig(A.t() @ A, eigenvectors=True)
    
    eps = 10**-6
    
    Sigma = torch.sqrt(e)
    mask = 1-torch.isnan(Sigma)
    return A @ V[:, mask]

def mult(U, d, x):
    r"""Computes :math:`Ax` where :math:`A = UU^\top + diag(d)`

    Arguments:
        U (Tensor): a low-rank matrix of size [n x k]
        d (Tensor): a vector ([n x 1] Tensor) representing the diagonal
        x (Tensor): size [n x d]
    """
    return d*x + U @ (U.t() @ x)

def invMult(U, d, x):
    r"""Computes :math:`A^{-1}x` where :math:`A = UU^\top + diag(d)`

    Arguments:
        U (Tensor): a low-rank matrix of size [n x k]
        d (Tensor): a vector ([n x 1] Tensor) representing the diagonal
        x (Tensor): size [n x d]
    """
    dInv = 1/d
    I_k = torch.eye(U.shape[1], dtype=U.dtype, device=U.device)
    Kinv = I_k + U.t() @ (dInv * U)

    s1 = U.t() @ (dInv * x)
    s2, _ = torch.gesv(s1, Kinv)
    return dInv*(x - (U @ s2))

def factCore(V, reduce_flag=False):
    r"""Computes :math:`K` such that :math:`I_n + VKV^\top`
    is a square-root for :math:`I_n + VV^\top`

    Arguments:
        V (Tensor): a low-rank matrix of size [n x k]
    """
    
    try:
        if reduce_flag:
            V = reduceRank(V)
            
        I_k = torch.eye(V.shape[1], dtype=V.dtype, device=V.device)
        L = torch.potrf(V.t() @ V, upper=False)
        M = torch.potrf(I_k + L.t() @ L, upper=False)
        Linv = torch.inverse(L)
        K = Linv.t() @ (M - I_k) @ Linv
        
    except RuntimeError as err:
        if reduce_flag:
            raise
        if str(err).startswith(NOT_FULL_RANK_ERR_MSG):
            warnings.warn("The factor matrix is not full-rank. Torchutil will attempt to remove unused dimensions. This might impact performance.")
            return factCore(V, reduce_flag=True)
        else:
            raise
            
    return K, V

def factMult(U, d, x):
    r"""Computes :math:`Bx` where :math:`BB^\top = UU^\top + diag(d)`

    Arguments:
        U (Tensor): a low-rank matrix of size [n x k]
        d (Tensor): a vector ([n x 1] Tensor) representing the diagonal
        x (Tensor): size [n x d]
    """
    d_sqrt = torch.sqrt(d)
    V = U/d_sqrt

    K, V = factCore(V)

    return d_sqrt * (x + V @ (K @ (V.t() @ x)))

def invFactMult(U, d, x):
    r"""Computes :math:`Cx` where :math:`CC^\top = (UU^\top + diag(d))^{-1}`

    Arguments:
        U (Tensor): a low-rank matrix of size [n x k]
        d (Tensor): a vector ([n x 1] Tensor) representing the diagonal
        x (Tensor): size [n x d]
    """
    d_sqrt = torch.sqrt(d)
    V = U/d_sqrt

    K, V = factCore(V)

    dirInv = torch.inverse(torch.inverse(K.t()) + V.t() @ V) @ (V.t() @ x)
    return (x - V @ dirInv)/d_sqrt

def logdet(U, d):
    r"""Computes :math:`\det(A)` where :math:`A = UU^\top + diag(d)`

    Arguments:
        U (Tensor): a low-rank matrix of size [n x k]
        d (Tensor): a vector ([n x 1] Tensor) representing the diagonal
    """
    V = U/torch.sqrt(d)
    I_K = torch.eye(U.shape[1], dtype=U.dtype, device=U.device)
    sign, logdet_factor = torch.slogdet(I_K + V.t() @ V)
    assert sign > 0
    return torch.sum(torch.log(d)) + logdet_factor

def trace(U, d):
    r"""Computes :math:`\text{Trace}(A)` where :math:`A = UU^\top + diag(d)`

    Arguments:
        U (Tensor): a low-rank matrix of size [n x k]
        d (Tensor): a vector ([n x 1] Tensor) representing the diagonal
    """
    return torch.sum(d) + torch.sum(U**2)

def invFacts(U, d):
    r"""Returns the inverse of (UU^T + d) in a low-memory cost factorization.
    Returns A, B, v, giving the inverse as :math:`\text{diag}(v) - A @ B`
    """
    dInv = 1/d
    I_k = torch.eye(U.shape[1], dtype=U.dtype, device=U.device)

    Kinv = I_k + U.t() @ (dInv * U)

    s1 = (U * dInv).t()
    s2, _ = torch.gesv(s1, Kinv)
    return dInv*U, s2, dInv

def invDiag(U, d):
    r"""Returns the diagonal of the inverse of (UU^T + d)"""
    A, B, v = invFacts(U, d)
    return v - torch.sum(A * B.t(), dim=1).view(-1, 1)

def diag(U, d):
    r"""Returns the diagonal of (UU^T + d)"""
    return torch.sum(U**2, dim=1).view(-1, 1) + d.view(-1, 1)
