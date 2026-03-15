# -*- coding: utf-8 -*-
"""
Epistemic Mass Matrix (PyTorch)
================================

The mass matrix M_i encodes epistemic inertia: how resistant agent i's
beliefs are to change. Mass = precision = Fisher information.

Complete 4-term formula (Dennis 2025, Eq. 266-268):
    M_i = Λ_p + Λ_o + Σ_k β_ik Ω_ik Λ_k Ω_ikᵀ + (Σ_j β_ji) Λ_qi

where:
    Λ_p = Σ_p⁻¹  (prior precision — anchoring to initial beliefs)
    Λ_o = R⁻¹     (observation precision — sensory anchoring)
    Σ_k β_ik Ω_ik Λ_k Ω_ikᵀ  (outgoing social inertia — transported neighbor precision)
    (Σ_j β_ji) Λ_qi  (incoming social inertia — how much others attend to you)
"""

import torch
from torch import Tensor
from typing import Optional

from torch_core.distributions import safe_inv
from torch_core.transport import push_covariance, compute_transport


def mass_matrix_diagonal(
    mu_q: Tensor,
    Sigma_q: Tensor,
    Sigma_p: Tensor,
    phi: Tensor,
    generators: Tensor,
    beta: Tensor,
    R_obs: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    """
    Compute diagonal mass matrix blocks for all agents.

    Returns per-agent mass matrices (no inter-agent kinetic coupling).

    Args:
        mu_q: Belief means, shape (N, K) [unused, for API consistency]
        Sigma_q: Belief covariances, shape (N, K, K)
        Sigma_p: Prior covariances, shape (N, K, K)
        phi: Gauge fields, shape (N, d)
        generators: SO(K) generators, shape (d, K, K)
        beta: Attention weights, shape (N, N)
        R_obs: Observation noise covariance, shape (D, D) or None
        eps: Regularization

    Returns:
        M: Per-agent mass matrices, shape (N, K, K)
    """
    N, K = Sigma_q.shape[0], Sigma_q.shape[1]

    # TERM 1: Prior precision Λ_p = Σ_p⁻¹
    Lambda_p = safe_inv(Sigma_p, eps=eps)  # (N, K, K)
    M = Lambda_p.clone()

    # TERM 2: Observation precision Λ_o
    if R_obs is not None:
        D = R_obs.shape[0]
        Lambda_o = torch.linalg.inv(
            R_obs + eps * torch.eye(D, dtype=R_obs.dtype, device=R_obs.device)
        )
        if D == K:
            # Direct contribution when observation dimension matches latent
            M = M + Lambda_o.unsqueeze(0).expand(N, K, K)

    # TERM 3: Outgoing social inertia  Σ_k β_ik Ω_ik Λ_k Ω_ikᵀ
    Lambda_q = safe_inv(Sigma_q, eps=eps)  # (N, K, K)

    for i in range(N):
        for k in range(N):
            if i == k:
                continue
            b_ik = beta[i, k]
            if b_ik < 1e-8:
                continue

            # Transport neighbor k's precision to agent i's frame
            Omega_ik = compute_transport(phi[i], phi[k], generators)
            Lambda_k_transported = push_covariance(Lambda_q[k].unsqueeze(0), Omega_ik.unsqueeze(0)).squeeze(0)
            M[i] = M[i] + b_ik * Lambda_k_transported

    # TERM 4: Incoming social inertia  (Σ_j β_ji) Λ_qi
    incoming_beta = beta.sum(dim=0)  # (N,) — sum over senders j for each receiver i
    # Subtract self-attention (should be 0, but safety)
    incoming_beta = incoming_beta - beta.diag()

    for i in range(N):
        if incoming_beta[i] > 1e-8:
            M[i] = M[i] + incoming_beta[i] * Lambda_q[i]

    return M


def mass_matrix_full(
    Sigma_q: Tensor,
    Sigma_p: Tensor,
    phi: Tensor,
    generators: Tensor,
    beta: Tensor,
    R_obs: Optional[Tensor] = None,
    coupling_strength: float = 0.0,
    eps: float = 1e-6,
) -> Tensor:
    """
    Compute full (NK × NK) mass matrix including inter-agent coupling.

    Off-diagonal blocks M_ik create momentum exchange between aligned agents.

    Args:
        Sigma_q: shape (N, K, K)
        Sigma_p: shape (N, K, K)
        phi: shape (N, d)
        generators: shape (d, K, K)
        beta: shape (N, N)
        R_obs: shape (D, D) or None
        coupling_strength: λ for inter-agent kinetic coupling (0 = diagonal only)
        eps: Regularization

    Returns:
        M: Full mass matrix, shape (NK, NK)
    """
    N, K = Sigma_q.shape[0], Sigma_q.shape[1]

    # Start with block-diagonal
    M_diag = mass_matrix_diagonal(
        torch.zeros(N, K, dtype=Sigma_q.dtype, device=Sigma_q.device),
        Sigma_q, Sigma_p, phi, generators, beta, R_obs, eps
    )

    # Build full matrix
    M = torch.zeros(N * K, N * K, dtype=Sigma_q.dtype, device=Sigma_q.device)

    # Place diagonal blocks
    for i in range(N):
        M[i*K:(i+1)*K, i*K:(i+1)*K] = M_diag[i]

    # Off-diagonal coupling blocks
    if coupling_strength > 0:
        Lambda_p = safe_inv(Sigma_p, eps=eps)

        for i in range(N):
            for k in range(N):
                if i == k:
                    continue
                b_ik = beta[i, k]
                if b_ik < 1e-8:
                    continue

                Omega_ik = compute_transport(phi[i], phi[k], generators)
                Lambda_pk_transported = push_covariance(
                    Lambda_p[k].unsqueeze(0), Omega_ik.unsqueeze(0)
                ).squeeze(0)

                M_ik = -coupling_strength * b_ik * Lambda_pk_transported
                M[i*K:(i+1)*K, k*K:(k+1)*K] = M_ik

        # Symmetrize for SPD guarantee
        M = 0.5 * (M + M.mT)

    return M
