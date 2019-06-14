r"""
Fast computation of eigenvalue decomposition and PCA.

Adapted from Facebook's Fast Randomized PCA/SVD;
* `Github <https://github.com/facebook/fbpca>`_
* `Doc <http://fbpca.readthedocs.io/en/latest/>`_
* `Blog post <https://research.fb.com/fast-randomized-svd/>`_.
"""

import math
import torch

# Disable naming scheme error for math symbols (U, d, k, ...)
# pylint: disable=C0103

__all__ = [
    "eigsh_func", "eigsh"
]

def eigsh_func(f, dtype, device, n, k=6, n_iter=4, L=None):
    r"""Top-k Eigendecomposition of PSD linear operator :math:`f(x) = Ax`.

    Provides a way to use :meth:`fastpca.eigsh` when the matrix
    to eigendecompose is only accessible through matrix-vector products.

    Might be useful if for the eigendecomposition of a matrix :math:`A = UU^\top`,
    where :math:`U` is a known matrix of size :math:`(n,m)`;
    computing :math:`U(U^\top x)` costs :math:`O(nm)` operations whereas
    computing :math:`A` costs :math:`O(n^2m)`.

    See :meth:`fastpca.eigsh` for a description of the other arguments.

    Arguments:
        f (function): Applies the linear operator to eigendecompose,
            :math:`f(x) = Ax`.
        dtype (torch.dtype): The type used in ``f``.
        device (torch.device): The device where ``f`` is allocated.
        m (int): The dimensionality of the domain of ``f``.
            Would be ``A.shape[1]`` if :math:`f(x) = Ax`.
    """
    if L is None:
        L = k+2
    assert k >= 0
    assert k <= n
    assert L >= k
    assert n_iter >= 0

    def orthogonalize(A):
        (Q, _) = torch.qr(A)
        return Q

    def nystrom(Q, anorm):
        r"""
        Use the Nystrom method to obtain approximations to the
        eigenvalues and eigenvectors of A (shifting A on the subspace
        spanned by the columns of Q in order to make the shifted A be
        positive definite).
        """
        
        def svd_thin_matrix(A):
            r"""
            Efficient implementation of SVD on [N x D] matrix, D >> N.
            """
            (e, V) = torch.symeig(A @ A.t(), eigenvectors=True)
            
            Sigma = torch.sqrt(e)
            SigInv = 1/Sigma 
            SigInv[torch.isnan(SigInv)] = 0
            U = A.t() @ (V * SigInv)
            
            return U, Sigma, V
            
        anorm = .1e-6 * anorm * math.sqrt(1. * n)
        E = f(Q) + anorm * Q
        R = Q.t() @ E
        R = (R + R.t()) / 2
        R = torch.potrf(R, upper=False) # Cholesky
        (tmp, _) = torch.gesv(E.t(), R) # Solve
        V, d, _ = svd_thin_matrix(tmp)
        d = d * d - anorm
        return d, V

    Q = 2*(torch.rand((n, L), device=device, dtype=dtype)-0.5)
    for _ in range(max(0, n_iter-1)):
        Q = orthogonalize(f(Q))
    oldQ = Q
    Q = f(Q)
    anorm = torch.max(torch.norm(Q, dim=0)/torch.norm(oldQ, dim=0))
    Q = orthogonalize(Q)

    d, V = nystrom(Q, anorm)

    # Retain only the entries with the k greatest absolute values
    (_, idx) = torch.abs(d).sort()
    idx = idx[(L-k):]
    return abs(d[idx]), V[:, idx]

def eigsh(A=None, k=6, n_iter=4, L=None):
    r"""Top-k Eigendecomposition of a positive semi-definite matrix A.

    Returns a rank-k approximation of the positive definite matrix A.
    Parameters ``n_iter`` and ``L`` control the running time and quality of
    the approximation.
    The quality of the approximation degrades as ``k`` gets close to the size
    of ``A``.
    ``n_iter=1`` should already be sufficient to obtain a good quality
    approximation, especially if ``k`` is small.
    More details in HMT09_.

    Arguments:
        A (Tensor): a positive semi-definite matrix.
        k (int, optional): Number of eigenvalues/eigenvectors to return.
            Default: 6 Valid range: ``0 <= k <= dim(A)``
        n_iter (int, optional): Number of iterations of the power methods.
            Default: 4 Valid range: ``n_iter >= 0``
        L (int, optional): Number of random vector to start the decomposition
            Default: k+2 Valid range: ``L >= k``

    Returns:
        A tuple containing

            * e (Tensor): shape (k,1) containing the largest eigenvalues
            * V (Tensor): shape (m,k) containing the matching eigenvectors

    References:
        .. [HMT09] Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp. 2009.
            Finding structure with randomness: probabilistic algorithms for
            constructing approximate matrix decompositions.
            (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).
    """
    return eigsh_func(lambda x: A @ x, A.dtype, A.device, A.shape[1], k=k, n_iter=n_iter, L=L)
