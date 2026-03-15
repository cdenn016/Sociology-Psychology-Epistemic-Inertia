# -*- coding: utf-8 -*-
"""
Fisher-Rao Metric and Natural Gradients (PyTorch)
===================================================

Natural gradient projection on the statistical manifold of Gaussians.

Fisher Information Metric:
    G = [Σ⁻¹              0        ]
        [   0     ½(Σ⁻¹ ⊗ Σ⁻¹) ]

Natural Gradient Formulas:
    δμ = -Σ ∇_μ
    δΣ = -2 Σ sym(∇_Σ) Σ

Note: With autograd, natural gradients are often unnecessary since
we can directly differentiate the free energy. These functions are
provided for compatibility with the existing codebase and for cases
where natural gradient descent is explicitly desired.
"""

import torch
from torch import Tensor
from typing import Tuple

from torch_core.distributions import sanitize_sigma


def natural_gradient_gaussian(
    mu: Tensor,
    Sigma: Tensor,
    grad_mu: Tensor,
    grad_Sigma: Tensor,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """
    Project Euclidean gradients to natural gradients via Fisher-Rao metric.

    Args:
        mu: Current mean, shape (..., K)
        Sigma: Current covariance, shape (..., K, K)
        grad_mu: Euclidean gradient ∂L/∂μ, shape (..., K)
        grad_Sigma: Euclidean gradient ∂L/∂Σ, shape (..., K, K)
        eps: Regularization

    Returns:
        delta_mu: Natural gradient for μ, shape (..., K)
        delta_Sigma: Natural gradient for Σ, shape (..., K, K)

    Formulas:
        δμ = -Σ ∇_μ
        δΣ = -2 Σ sym(∇_Σ) Σ
    """
    # Sanitize Sigma
    Sigma_clean = sanitize_sigma(Sigma, min_eigenvalue=eps)

    # Symmetrize gradient
    grad_Sigma_sym = 0.5 * (grad_Sigma + grad_Sigma.mT)

    # Natural gradient for μ: δμ = -Σ ∇_μ
    delta_mu = -torch.einsum('...ij,...j->...i', Sigma_clean, grad_mu)

    # Natural gradient for Σ: δΣ = -2 Σ sym(∇_Σ) Σ
    tmp = Sigma_clean @ grad_Sigma_sym
    delta_Sigma = -2.0 * tmp @ Sigma_clean

    # Symmetrize result
    delta_Sigma = 0.5 * (delta_Sigma + delta_Sigma.mT)

    return delta_mu, delta_Sigma


def euclidean_from_natural(
    Sigma: Tensor,
    delta_mu: Tensor,
    delta_Sigma: Tensor,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """
    Invert natural gradient projection: natural → Euclidean.

    Args:
        Sigma: Current covariance, shape (..., K, K)
        delta_mu: Natural gradient δμ, shape (..., K)
        delta_Sigma: Natural gradient δΣ, shape (..., K, K)
        eps: Regularization

    Returns:
        grad_mu: Euclidean gradient ∇_μ, shape (..., K)
        grad_Sigma: Euclidean gradient ∇_Σ, shape (..., K, K)

    Formulas:
        ∇_μ = -Σ⁻¹ δμ
        ∇_Σ = -½ Σ⁻¹ δΣ Σ⁻¹
    """
    from torch_core.distributions import safe_inv

    Sigma_inv = safe_inv(Sigma, eps=eps)

    grad_mu = -torch.einsum('...ij,...j->...i', Sigma_inv, delta_mu)

    tmp = Sigma_inv @ delta_Sigma
    grad_Sigma = -0.5 * tmp @ Sigma_inv
    grad_Sigma = 0.5 * (grad_Sigma + grad_Sigma.mT)

    return grad_mu, grad_Sigma
