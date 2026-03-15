# -*- coding: utf-8 -*-
"""
Parallel Transport on SO(3) Principal Bundle (PyTorch)
======================================================

Implements Ω_ij(c) = exp(φ_i) · exp(-φ_j) for gauge-theoretic
active inference, fully differentiable via torch.autograd.

Transport Operator Properties:
    - Ω_ij ∈ SO(K): det(Ω) = +1, Ωᵀ Ω = I
    - Ω_ij · Ω_jk = Ω_ik (transitivity)
    - Ω_ii = I (self-transport is identity)

Push-Forward:
    N(μ, Σ) → N(Ω μ, Ω Σ Ωᵀ)
"""

import torch
from torch import Tensor
from typing import Tuple


def rodrigues(phi: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Compute exp(φ) ∈ SO(3) via Rodrigues' formula.

    Formula: exp(φ) = I + sin(θ)/θ · [φ]_× + (1 - cos(θ))/θ² · [φ]_×²

    Differentiable through all branches (uses smooth interpolation
    instead of hard masking for small angles).

    Args:
        phi: Axis-angle vectors, shape (..., 3)

    Returns:
        R: Rotation matrices, shape (..., 3, 3)
    """
    theta_sq = (phi * phi).sum(dim=-1, keepdim=True)  # (..., 1)
    theta = torch.sqrt(theta_sq.clamp(min=eps * eps))  # (..., 1)

    # Skew-symmetric matrix [φ]_×
    K_mat = _skew_symmetric(phi)  # (..., 3, 3)
    K_sq = K_mat @ K_mat  # (..., 3, 3)

    # Coefficients with Taylor expansion for small angles
    # sin(θ)/θ ≈ 1 - θ²/6 + θ⁴/120 for small θ
    # (1-cos(θ))/θ² ≈ 1/2 - θ²/24 + θ⁴/720 for small θ
    c1 = torch.where(
        theta_sq.unsqueeze(-1) < 1e-6,
        1.0 - theta_sq.unsqueeze(-1) / 6.0,
        (torch.sin(theta) / theta).unsqueeze(-1),
    )
    c2 = torch.where(
        theta_sq.unsqueeze(-1) < 1e-6,
        0.5 - theta_sq.unsqueeze(-1) / 24.0,
        ((1.0 - torch.cos(theta)) / theta_sq).unsqueeze(-1),
    )

    eye = torch.eye(3, dtype=phi.dtype, device=phi.device)
    R = eye + c1 * K_mat + c2 * K_sq

    return R


def _skew_symmetric(v: Tensor) -> Tensor:
    """
    Construct skew-symmetric matrix [v]_× from vector v ∈ ℝ³.

    Args:
        v: Vectors, shape (..., 3)

    Returns:
        v_x: Skew-symmetric matrices, shape (..., 3, 3)
    """
    batch_shape = v.shape[:-1]
    zero = torch.zeros(batch_shape, dtype=v.dtype, device=v.device)

    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]

    # Build rows and stack
    row1 = torch.stack([zero, -vz, vy], dim=-1)
    row2 = torch.stack([vz, zero, -vx], dim=-1)
    row3 = torch.stack([-vy, vx, zero], dim=-1)

    return torch.stack([row1, row2, row3], dim=-2)


def matrix_exp_so(phi: Tensor, generators: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Compute exp(Σ φᵃ Gₐ) for general SO(K) via torch.linalg.matrix_exp.

    Args:
        phi: Lie algebra coordinates, shape (..., d) where d = dim(𝔰𝔬(K))
        generators: Basis matrices, shape (d, K, K)

    Returns:
        exp_phi: Rotation matrices, shape (..., K, K)
    """
    # Construct algebra element: X = Σ_a φ^a G_a
    X = torch.einsum('...a,aij->...ij', phi, generators)

    # Enforce skew-symmetry
    X = 0.5 * (X - X.mT)

    return torch.linalg.matrix_exp(X)


def compute_transport(
    phi_i: Tensor,
    phi_j: Tensor,
    generators: Tensor,
) -> Tensor:
    """
    Compute transport operator Ω_ij = exp(φ_i) · exp(-φ_j).

    Args:
        phi_i: Agent i gauge field, shape (..., d)
        phi_j: Agent j gauge field, shape (..., d)
        generators: SO(K) generators, shape (d, K, K)

    Returns:
        Omega_ij: Transport operator, shape (..., K, K)
    """
    exp_phi_i = matrix_exp_so(phi_i, generators)
    exp_neg_phi_j = matrix_exp_so(-phi_j, generators)
    return exp_phi_i @ exp_neg_phi_j


def push_mean(mu: Tensor, Omega: Tensor) -> Tensor:
    """
    Push mean forward: μ' = Ω μ.

    Args:
        mu: Mean vectors, shape (..., K)
        Omega: Transport operator, shape (..., K, K)

    Returns:
        mu_pushed: Transported mean, shape (..., K)
    """
    return torch.einsum('...ij,...j->...i', Omega, mu)


def push_covariance(Sigma: Tensor, Omega: Tensor) -> Tensor:
    """
    Push covariance forward: Σ' = Ω Σ Ωᵀ.

    Preserves SPD structure (Ω orthogonal).

    Args:
        Sigma: Covariance matrices, shape (..., K, K)
        Omega: Transport operator, shape (..., K, K)

    Returns:
        Sigma_pushed: Transported covariance, shape (..., K, K)
    """
    Sigma_pushed = Omega @ Sigma @ Omega.mT

    # Symmetrize for numerical stability
    return 0.5 * (Sigma_pushed + Sigma_pushed.mT)


def kl_transported(
    mu_i: Tensor,
    Sigma_i: Tensor,
    mu_j: Tensor,
    Sigma_j: Tensor,
    Omega_ij: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """
    Compute KL(q_i || Ω_ij[q_j]) — the alignment energy term.

    Fuses transport and KL computation for efficiency.

    Args:
        mu_i, Sigma_i: Receiver distribution (agent i)
        mu_j, Sigma_j: Sender distribution (agent j)
        Omega_ij: Transport operator i ← j
        eps: Regularization

    Returns:
        kl: Alignment divergence, shape (...,)
    """
    from torch_core.distributions import kl_gaussian

    # Push j → i's frame
    mu_j_pushed = push_mean(mu_j, Omega_ij)
    Sigma_j_pushed = push_covariance(Sigma_j, Omega_ij)

    # KL(q_i || transported q_j)
    return kl_gaussian(mu_i, Sigma_i, mu_j_pushed, Sigma_j_pushed, eps=eps)
