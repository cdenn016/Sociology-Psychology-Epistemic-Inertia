# -*- coding: utf-8 -*-
"""
Differentiable Variational Free Energy (PyTorch)
=================================================

Complete VFE functional, fully differentiable via torch.autograd.

F = α · KL(q || p)                                   [Self-coupling]
  + λ_β · Σ_{i,j} β_ij · KL(q_i || Ω_ij[q_j])      [Belief alignment]
  + λ_γ · Σ_{i,j} γ_ij · KL(p_i || Ω_ij[p_j])      [Prior alignment]
  - λ_o · Σ_i E_q[log p(o|x)]                        [Observations]

The key advantage over NumPy: torch.autograd.grad(F, params) replaces
~600 lines of hand-coded gradient computation in gradient_engine.py.

Usage:
    # Define parameters as requires_grad tensors
    mu_q = torch.randn(N, K, requires_grad=True)
    Sigma_q = torch.eye(K).expand(N, K, K).clone().requires_grad_(True)

    # Compute free energy (scalar)
    F = free_energy_total(mu_q, Sigma_q, mu_p, Sigma_p, phi, generators, beta, config)

    # Get gradients automatically
    F.backward()
    grad_mu = mu_q.grad  # ∂F/∂μ for all agents
"""

import torch
from torch import Tensor
from typing import Optional, NamedTuple

from torch_core.distributions import kl_gaussian
from torch_core.transport import compute_transport, kl_transported


class FreeEnergy(NamedTuple):
    """Container for free energy components."""
    self_energy: Tensor
    belief_align: Tensor
    prior_align: Tensor
    observations: Tensor
    total: Tensor


def softmax_attention(
    mu: Tensor,
    Sigma: Tensor,
    phi: Tensor,
    generators: Tensor,
    kappa: float = 1.0,
    eps: float = 1e-6,
) -> Tensor:
    """
    Compute softmax attention weights β_ij = softmax(-KL(q_i || Ω_ij[q_j]) / κ).

    This is the information-geometric attention mechanism: similar beliefs
    (low KL) get high attention weights.

    Args:
        mu: All agent means, shape (N, K)
        Sigma: All agent covariances, shape (N, K, K)
        phi: All agent gauge fields, shape (N, d)
        generators: SO(K) generators, shape (d, K, K)
        kappa: Temperature (higher = softer attention)
        eps: Regularization

    Returns:
        beta: Attention weights, shape (N, N), rows sum to 1
              β_ii = 0 (no self-attention)
    """
    N = mu.shape[0]

    # Compute pairwise KL divergences
    kl_matrix = torch.zeros(N, N, dtype=mu.dtype, device=mu.device)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            Omega_ij = compute_transport(phi[i], phi[j], generators)
            kl_matrix[i, j] = kl_transported(
                mu[i], Sigma[i], mu[j], Sigma[j], Omega_ij, eps=eps
            )

    # Softmax over neighbors (exclude self)
    # Set self-connection to -inf so softmax gives 0
    mask = torch.eye(N, dtype=torch.bool, device=mu.device)
    logits = -kl_matrix / kappa
    logits = logits.masked_fill(mask, float('-inf'))

    beta = torch.softmax(logits, dim=-1)

    # Zero out self-connections (softmax of -inf should be 0, but enforce)
    beta = beta.masked_fill(mask, 0.0)

    return beta


def free_energy_self(
    mu_q: Tensor,
    Sigma_q: Tensor,
    mu_p: Tensor,
    Sigma_p: Tensor,
    alpha: float = 1.0,
    eps: float = 1e-6,
) -> Tensor:
    """
    Self-coupling energy: α · Σ_i KL(q_i || p_i).

    Args:
        mu_q: Belief means, shape (N, K)
        Sigma_q: Belief covariances, shape (N, K, K)
        mu_p: Prior means, shape (N, K)
        Sigma_p: Prior covariances, shape (N, K, K)
        alpha: Self-coupling strength

    Returns:
        energy: Scalar tensor (differentiable)
    """
    kl = kl_gaussian(mu_q, Sigma_q, mu_p, Sigma_p, eps=eps)  # (N,)
    return alpha * kl.sum()


def free_energy_belief_alignment(
    mu_q: Tensor,
    Sigma_q: Tensor,
    phi: Tensor,
    generators: Tensor,
    beta: Tensor,
    lambda_belief: float = 1.0,
    eps: float = 1e-6,
) -> Tensor:
    """
    Belief alignment energy: λ_β · Σ_{i,j} β_ij · KL(q_i || Ω_ij[q_j]).

    Args:
        mu_q: Belief means, shape (N, K)
        Sigma_q: Belief covariances, shape (N, K, K)
        phi: Gauge fields, shape (N, d)
        generators: SO(K) generators, shape (d, K, K)
        beta: Attention weights, shape (N, N)
        lambda_belief: Coupling strength

    Returns:
        energy: Scalar tensor (differentiable)
    """
    N = mu_q.shape[0]
    energy = torch.tensor(0.0, dtype=mu_q.dtype, device=mu_q.device)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            b_ij = beta[i, j]
            if b_ij < 1e-8:
                continue

            Omega_ij = compute_transport(phi[i], phi[j], generators)
            kl_ij = kl_transported(
                mu_q[i], Sigma_q[i], mu_q[j], Sigma_q[j], Omega_ij, eps=eps
            )
            energy = energy + b_ij * kl_ij

    return lambda_belief * energy


def free_energy_prior_alignment(
    mu_p: Tensor,
    Sigma_p: Tensor,
    phi: Tensor,
    generators: Tensor,
    gamma: Tensor,
    lambda_prior: float = 1.0,
    eps: float = 1e-6,
) -> Tensor:
    """
    Prior alignment energy: λ_γ · Σ_{i,j} γ_ij · KL(p_i || Ω_ij[p_j]).

    Same structure as belief alignment but for priors.

    Args:
        mu_p: Prior means, shape (N, K)
        Sigma_p: Prior covariances, shape (N, K, K)
        phi: Gauge fields, shape (N, d)
        generators: SO(K) generators, shape (d, K, K)
        gamma: Prior attention weights, shape (N, N)
        lambda_prior: Coupling strength

    Returns:
        energy: Scalar tensor (differentiable)
    """
    N = mu_p.shape[0]
    energy = torch.tensor(0.0, dtype=mu_p.dtype, device=mu_p.device)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            g_ij = gamma[i, j]
            if g_ij < 1e-8:
                continue

            Omega_ij = compute_transport(phi[i], phi[j], generators)
            kl_ij = kl_transported(
                mu_p[i], Sigma_p[i], mu_p[j], Sigma_p[j], Omega_ij, eps=eps
            )
            energy = energy + g_ij * kl_ij

    return lambda_prior * energy


def free_energy_observation(
    mu_q: Tensor,
    Sigma_q: Tensor,
    observations: Tensor,
    W_obs: Tensor,
    R_obs: Tensor,
    lambda_obs: float = 1.0,
) -> Tensor:
    """
    Observation energy: -λ_o · Σ_i E_q[log p(o_i | x)].

    For Gaussian observation model p(o|x) = N(o | Wx, R):
        E_q[log p(o|x)] = -0.5 [log|2πR| + tr(R⁻¹ W Σ_q Wᵀ) + (o - Wμ)ᵀ R⁻¹ (o - Wμ)]

    Args:
        mu_q: Belief means, shape (N, K)
        Sigma_q: Belief covariances, shape (N, K, K)
        observations: Observed data, shape (N, D)
        W_obs: Observation matrix, shape (D, K)
        R_obs: Observation noise covariance, shape (D, D)
        lambda_obs: Observation coupling strength

    Returns:
        energy: Scalar tensor (differentiable)
    """
    D = observations.shape[-1]

    # Predicted observation mean: Wμ
    o_pred = mu_q @ W_obs.mT  # (N, D)

    # Innovation
    innovation = observations - o_pred  # (N, D)

    # R⁻¹
    R_inv = torch.linalg.inv(R_obs + 1e-8 * torch.eye(D, dtype=R_obs.dtype, device=R_obs.device))

    # Mahalanobis: (o - Wμ)ᵀ R⁻¹ (o - Wμ)
    mahal = (innovation @ R_inv * innovation).sum(dim=-1)  # (N,)

    # Trace: tr(R⁻¹ W Σ_q Wᵀ)
    W_Sigma = W_obs @ Sigma_q  # (N, D, K) via broadcasting
    W_Sigma_Wt = W_Sigma @ W_obs.mT  # (N, D, D)
    trace_term = (R_inv * W_Sigma_Wt).sum(dim=(-2, -1))  # (N,)

    # Log-determinant
    logdet_R = torch.logdet(R_obs)

    # E[log p(o|x)] = -0.5 [log|2πR| + trace + mahal]
    log_lik = -0.5 * (D * torch.log(torch.tensor(2.0 * torch.pi, dtype=mu_q.dtype, device=mu_q.device))
                       + logdet_R + trace_term + mahal)

    # Free energy contribution: -λ * Σ E[log p]
    return -lambda_obs * log_lik.sum()


def free_energy_total(
    mu_q: Tensor,
    Sigma_q: Tensor,
    mu_p: Tensor,
    Sigma_p: Tensor,
    phi: Tensor,
    generators: Tensor,
    kappa: float = 1.0,
    alpha: float = 1.0,
    lambda_belief: float = 1.0,
    lambda_prior: float = 0.0,
    lambda_obs: float = 0.0,
    observations: Optional[Tensor] = None,
    W_obs: Optional[Tensor] = None,
    R_obs: Optional[Tensor] = None,
    beta: Optional[Tensor] = None,
    gamma: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> FreeEnergy:
    """
    Compute complete free energy functional.

    All components are differentiable. Call .backward() on total to get
    gradients w.r.t. all parameters.

    Args:
        mu_q: Belief means, shape (N, K), requires_grad=True
        Sigma_q: Belief covariances, shape (N, K, K), requires_grad=True
        mu_p: Prior means, shape (N, K)
        Sigma_p: Prior covariances, shape (N, K, K)
        phi: Gauge fields, shape (N, d), requires_grad=True
        generators: SO(K) generators, shape (d, K, K)
        kappa: Attention temperature
        alpha: Self-coupling strength
        lambda_belief: Belief alignment strength
        lambda_prior: Prior alignment strength
        lambda_obs: Observation coupling strength
        observations: Observed data, shape (N, D) or None
        W_obs: Observation matrix, shape (D, K) or None
        R_obs: Observation noise, shape (D, D) or None
        beta: Precomputed attention weights or None (computed from beliefs)
        gamma: Precomputed prior attention weights or None
        eps: Regularization

    Returns:
        FreeEnergy: Named tuple with all components and total
    """
    # (1) Self-coupling
    E_self = free_energy_self(mu_q, Sigma_q, mu_p, Sigma_p, alpha=alpha, eps=eps)

    # (2) Belief alignment
    if lambda_belief > 0:
        if beta is None:
            beta = softmax_attention(mu_q, Sigma_q, phi, generators, kappa=kappa, eps=eps)
        E_belief = free_energy_belief_alignment(
            mu_q, Sigma_q, phi, generators, beta, lambda_belief=lambda_belief, eps=eps
        )
    else:
        E_belief = torch.tensor(0.0, dtype=mu_q.dtype, device=mu_q.device)

    # (3) Prior alignment
    if lambda_prior > 0:
        if gamma is None:
            gamma = softmax_attention(mu_p, Sigma_p, phi, generators, kappa=kappa, eps=eps)
        E_prior = free_energy_prior_alignment(
            mu_p, Sigma_p, phi, generators, gamma, lambda_prior=lambda_prior, eps=eps
        )
    else:
        E_prior = torch.tensor(0.0, dtype=mu_q.dtype, device=mu_q.device)

    # (4) Observations
    if lambda_obs > 0 and observations is not None and W_obs is not None and R_obs is not None:
        E_obs = free_energy_observation(
            mu_q, Sigma_q, observations, W_obs, R_obs, lambda_obs=lambda_obs
        )
    else:
        E_obs = torch.tensor(0.0, dtype=mu_q.dtype, device=mu_q.device)

    E_total = E_self + E_belief + E_prior + E_obs

    return FreeEnergy(
        self_energy=E_self,
        belief_align=E_belief,
        prior_align=E_prior,
        observations=E_obs,
        total=E_total,
    )
