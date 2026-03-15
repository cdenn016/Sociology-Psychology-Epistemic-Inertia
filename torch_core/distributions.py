# -*- coding: utf-8 -*-
"""
Gaussian Distribution Utilities (PyTorch)
==========================================

KL divergence, entropy, and covariance sanitization for multivariate
Gaussians, fully differentiable via torch.autograd.

All functions accept batched inputs with shape (..., K) for means
and (..., K, K) for covariance matrices.

Mathematical Reference:
    KL(q || p) = 0.5 * [tr(Σ_p⁻¹ Σ_q) + (μ_p - μ_q)ᵀ Σ_p⁻¹ (μ_p - μ_q) - K + ln|Σ_p|/|Σ_q|]
"""

import torch
from torch import Tensor
from typing import Optional, Tuple


def kl_gaussian(
    mu_q: Tensor,
    Sigma_q: Tensor,
    mu_p: Tensor,
    Sigma_p: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """
    KL divergence KL(q || p) between multivariate Gaussians.

    Fully differentiable. Uses Cholesky decomposition for numerical stability.

    Args:
        mu_q: Mean of q, shape (..., K)
        Sigma_q: Covariance of q, shape (..., K, K)
        mu_p: Mean of p, shape (..., K)
        Sigma_p: Covariance of p, shape (..., K, K)
        eps: Regularization for positive-definiteness

    Returns:
        kl: KL divergence, shape (...,), non-negative
    """
    K = mu_q.shape[-1]

    # Regularize for numerical stability
    eye = torch.eye(K, dtype=Sigma_q.dtype, device=Sigma_q.device)
    Sigma_q_reg = Sigma_q + eps * eye
    Sigma_p_reg = Sigma_p + eps * eye

    # Cholesky decomposition
    L_p = torch.linalg.cholesky(Sigma_p_reg)
    L_q = torch.linalg.cholesky(Sigma_q_reg)

    # Log-determinants via Cholesky diagonal
    logdet_p = 2.0 * torch.sum(torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1)), dim=-1)
    logdet_q = 2.0 * torch.sum(torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1)), dim=-1)

    # Term 1: tr(Σ_p⁻¹ Σ_q) via solve
    # Solve L_p X = Sigma_q  =>  X = L_p⁻¹ Sigma_q
    # Then Σ_p⁻¹ Σ_q = L_p⁻ᵀ L_p⁻¹ Σ_q = L_p⁻ᵀ X
    X = torch.linalg.solve_triangular(L_p, Sigma_q_reg, upper=False)
    Y = torch.linalg.solve_triangular(L_p.mT, X, upper=True)
    term_trace = torch.diagonal(Y, dim1=-2, dim2=-1).sum(dim=-1)

    # Term 2: Mahalanobis (μ_p - μ_q)ᵀ Σ_p⁻¹ (μ_p - μ_q)
    delta = mu_p - mu_q  # (..., K)
    v = torch.linalg.solve_triangular(L_p, delta.unsqueeze(-1), upper=False)
    term_quad = (v * v).sum(dim=(-2, -1))

    # Term 3: log-determinant ratio
    term_logdet = logdet_p - logdet_q

    # KL = 0.5 * (trace + quad - K + logdet)
    kl = 0.5 * (term_trace + term_quad - K + term_logdet)

    # Clamp to non-negative (numerical errors can make it slightly negative)
    return kl.clamp(min=0.0)


def kl_gaussian_batch(
    mu_q: Tensor,
    Sigma_q: Tensor,
    mu_p: Tensor,
    Sigma_p: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """
    Batched KL divergence between N pairs of Gaussians.

    Args:
        mu_q: shape (N, K)
        Sigma_q: shape (N, K, K)
        mu_p: shape (N, K)
        Sigma_p: shape (N, K, K)

    Returns:
        kl: shape (N,)
    """
    return kl_gaussian(mu_q, Sigma_q, mu_p, Sigma_p, eps=eps)


def entropy_gaussian(Sigma: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Differential entropy of multivariate Gaussian: H = 0.5 * (K * ln(2πe) + ln|Σ|).

    Args:
        Sigma: Covariance matrix, shape (..., K, K)
        eps: Regularization

    Returns:
        entropy: shape (...,)
    """
    K = Sigma.shape[-1]
    eye = torch.eye(K, dtype=Sigma.dtype, device=Sigma.device)
    Sigma_reg = Sigma + eps * eye

    logdet = torch.logdet(Sigma_reg)
    return 0.5 * (K * (1.0 + torch.log(torch.tensor(2.0 * torch.pi, dtype=Sigma.dtype, device=Sigma.device))) + logdet)


def safe_inv(Sigma: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Numerically stable matrix inverse with regularization.

    Args:
        Sigma: SPD matrices, shape (..., K, K)
        eps: Regularization strength

    Returns:
        Sigma_inv: Inverse matrices, shape (..., K, K)
    """
    K = Sigma.shape[-1]
    eye = torch.eye(K, dtype=Sigma.dtype, device=Sigma.device)

    # Symmetrize
    Sigma_sym = 0.5 * (Sigma + Sigma.mT)

    # Regularize and invert
    return torch.linalg.inv(Sigma_sym + eps * eye)


def sanitize_sigma(
    Sigma: Tensor,
    min_eigenvalue: float = 1e-4,
    max_cond: float = 1e4,
) -> Tensor:
    """
    Project covariance to SPD with eigenvalue floor and condition number cap.

    Differentiable through eigendecomposition.

    Args:
        Sigma: Covariance matrices, shape (..., K, K)
        min_eigenvalue: Absolute eigenvalue floor
        max_cond: Maximum condition number

    Returns:
        Sigma_clean: Sanitized SPD matrices, shape (..., K, K)
    """
    # Symmetrize
    Sigma_sym = 0.5 * (Sigma + Sigma.mT)

    # Eigendecomposition
    w, V = torch.linalg.eigh(Sigma_sym)

    # Floor eigenvalues
    w = w.clamp(min=min_eigenvalue)

    # Condition number cap
    lambda_max = w[..., -1:]
    lambda_min_required = lambda_max / max_cond
    w = torch.max(w, lambda_min_required)

    # Reconstruct
    Sigma_clean = (V * w.unsqueeze(-2)) @ V.mT

    # Final symmetrize
    return 0.5 * (Sigma_clean + Sigma_clean.mT)
