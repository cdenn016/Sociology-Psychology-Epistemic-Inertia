# -*- coding: utf-8 -*-
"""
Hamiltonian Belief Dynamics (PyTorch)
======================================

Second-order belief dynamics on the statistical manifold:

    M μ̈ + γ μ̇ + ∇_μ F = 0

Implemented as Hamilton's equations on phase space (μ, p):

    dμ/dt = M⁻¹ p
    dp/dt = -∇_μ F - γ M⁻¹ p

Key advantage: ∇_μ F is computed via torch.autograd, not hand-coded.
This eliminates ~600 lines of gradient code and guarantees correctness.

Integrators:
    - Symplectic Euler (1st order, simplest)
    - Stormer-Verlet / Leapfrog (2nd order, recommended)

The integrators preserve the symplectic structure of phase space,
ensuring long-term stability of Hamiltonian dynamics.
"""

import torch
from torch import Tensor
from typing import Optional, Callable, NamedTuple

from torch_core.mass_matrix import mass_matrix_diagonal
from torch_core.free_energy import free_energy_total


class PhaseState(NamedTuple):
    """Phase space state (position, momentum)."""
    mu_q: Tensor       # Belief means, shape (N, K)
    Sigma_q: Tensor    # Belief covariances, shape (N, K, K)
    p_mu: Tensor       # Mean momentum, shape (N, K)
    phi: Tensor        # Gauge fields, shape (N, d)


class HamiltonianDynamics:
    """
    Hamiltonian dynamics engine with autograd-computed forces.

    The force -∇_μ F is computed by:
        1. Evaluating F(μ, Σ, φ, ...) as a differentiable scalar
        2. Calling torch.autograd.grad(F, μ) to get the gradient
        3. Using the gradient as the force in Hamilton's equations

    This replaces all hand-coded gradient computation.

    Args:
        mu_p: Prior means, shape (N, K)
        Sigma_p: Prior covariances, shape (N, K, K)
        generators: SO(K) generators, shape (d, K, K)
        dt: Integration timestep
        friction: Damping coefficient γ
        mass_scale: Global mass scaling factor
        kappa: Attention temperature
        alpha: Self-coupling strength
        lambda_belief: Belief alignment strength
        lambda_prior: Prior alignment strength
        lambda_obs: Observation coupling strength
        observations: Observed data or None
        W_obs: Observation matrix or None
        R_obs: Observation noise or None
    """

    def __init__(
        self,
        mu_p: Tensor,
        Sigma_p: Tensor,
        generators: Tensor,
        dt: float = 0.01,
        friction: float = 1.0,
        mass_scale: float = 1.0,
        kappa: float = 1.0,
        alpha: float = 1.0,
        lambda_belief: float = 1.0,
        lambda_prior: float = 0.0,
        lambda_obs: float = 0.0,
        observations: Optional[Tensor] = None,
        W_obs: Optional[Tensor] = None,
        R_obs: Optional[Tensor] = None,
    ):
        self.mu_p = mu_p
        self.Sigma_p = Sigma_p
        self.generators = generators
        self.dt = dt
        self.friction = friction
        self.mass_scale = mass_scale
        self.kappa = kappa
        self.alpha = alpha
        self.lambda_belief = lambda_belief
        self.lambda_prior = lambda_prior
        self.lambda_obs = lambda_obs
        self.observations = observations
        self.W_obs = W_obs
        self.R_obs = R_obs

    def compute_force(
        self,
        mu_q: Tensor,
        Sigma_q: Tensor,
        phi: Tensor,
        beta: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute force F_mu = -∂F/∂μ via autograd.

        Args:
            mu_q: Belief means, shape (N, K), requires_grad=True
            Sigma_q: Belief covariances, shape (N, K, K)
            phi: Gauge fields, shape (N, d)
            beta: Precomputed attention weights or None

        Returns:
            force: -∇_μ F, shape (N, K)
        """
        # Ensure mu_q requires grad for autograd
        mu_q_grad = mu_q.detach().requires_grad_(True)

        # Compute free energy
        fe = free_energy_total(
            mu_q_grad, Sigma_q, self.mu_p, self.Sigma_p,
            phi, self.generators,
            kappa=self.kappa,
            alpha=self.alpha,
            lambda_belief=self.lambda_belief,
            lambda_prior=self.lambda_prior,
            lambda_obs=self.lambda_obs,
            observations=self.observations,
            W_obs=self.W_obs,
            R_obs=self.R_obs,
            beta=beta,
        )

        # Autograd: ∂F/∂μ
        grad_mu, = torch.autograd.grad(fe.total, mu_q_grad, create_graph=False)

        # Force = -gradient
        return -grad_mu

    def step_verlet(
        self,
        state: PhaseState,
        beta: Optional[Tensor] = None,
    ) -> PhaseState:
        """
        Stormer-Verlet (leapfrog) integration step.

        The Verlet integrator is symplectic (preserves phase space volume)
        and 2nd-order accurate. It splits the step into:
            1. Half-step momentum update
            2. Full-step position update
            3. Half-step momentum update

        With damping, the update becomes:
            p_{n+1/2} = p_n + (dt/2) * F(q_n) - (dt/2) * γ * v_n
            q_{n+1}   = q_n + dt * M⁻¹ p_{n+1/2}
            p_{n+1}   = p_{n+1/2} + (dt/2) * F(q_{n+1}) - (dt/2) * γ * v_{n+1}

        Args:
            state: Current phase space state
            beta: Precomputed attention weights or None

        Returns:
            new_state: Updated phase space state
        """
        mu_q, Sigma_q, p_mu, phi = state
        dt = self.dt

        # Mass matrix (block diagonal per agent)
        if beta is None:
            from torch_core.free_energy import softmax_attention
            beta = softmax_attention(
                mu_q, Sigma_q, phi, self.generators, kappa=self.kappa
            )

        M = mass_matrix_diagonal(
            mu_q, Sigma_q, self.Sigma_p, phi, self.generators, beta,
            R_obs=self.R_obs
        )  # (N, K, K)
        M_scaled = self.mass_scale * M

        # M⁻¹ for velocity computation
        M_inv = torch.linalg.inv(
            M_scaled + 1e-8 * torch.eye(M.shape[-1], dtype=M.dtype, device=M.device)
        )

        # Current velocity
        v = torch.einsum('...ij,...j->...i', M_inv, p_mu)

        # Force at current position
        force = self.compute_force(mu_q, Sigma_q, phi, beta=beta)

        # Half-step momentum
        p_half = p_mu + 0.5 * dt * force - 0.5 * dt * self.friction * p_mu

        # Full-step position
        v_half = torch.einsum('...ij,...j->...i', M_inv, p_half)
        mu_q_new = mu_q + dt * v_half

        # Force at new position
        force_new = self.compute_force(mu_q_new, Sigma_q, phi, beta=beta)

        # Half-step momentum (second half)
        p_mu_new = p_half + 0.5 * dt * force_new - 0.5 * dt * self.friction * p_half

        return PhaseState(mu_q_new, Sigma_q, p_mu_new, phi)

    def step_euler(
        self,
        state: PhaseState,
        beta: Optional[Tensor] = None,
    ) -> PhaseState:
        """
        Symplectic Euler integration step (1st order).

        Simpler than Verlet, useful for debugging.

        Args:
            state: Current phase space state
            beta: Precomputed attention weights or None

        Returns:
            new_state: Updated phase space state
        """
        mu_q, Sigma_q, p_mu, phi = state
        dt = self.dt

        if beta is None:
            from torch_core.free_energy import softmax_attention
            beta = softmax_attention(
                mu_q, Sigma_q, phi, self.generators, kappa=self.kappa
            )

        M = mass_matrix_diagonal(
            mu_q, Sigma_q, self.Sigma_p, phi, self.generators, beta,
            R_obs=self.R_obs
        )
        M_scaled = self.mass_scale * M
        M_inv = torch.linalg.inv(
            M_scaled + 1e-8 * torch.eye(M.shape[-1], dtype=M.dtype, device=M.device)
        )

        # Force
        force = self.compute_force(mu_q, Sigma_q, phi, beta=beta)

        # Update momentum first (symplectic Euler)
        p_mu_new = p_mu + dt * (force - self.friction * p_mu)

        # Update position with new momentum
        v_new = torch.einsum('...ij,...j->...i', M_inv, p_mu_new)
        mu_q_new = mu_q + dt * v_new

        return PhaseState(mu_q_new, Sigma_q, p_mu_new, phi)

    def kinetic_energy(self, state: PhaseState, beta: Tensor) -> Tensor:
        """Compute T = 0.5 pᵀ M⁻¹ p."""
        mu_q, Sigma_q, p_mu, phi = state
        M = mass_matrix_diagonal(
            mu_q, Sigma_q, self.Sigma_p, phi, self.generators, beta,
            R_obs=self.R_obs
        )
        M_scaled = self.mass_scale * M
        M_inv = torch.linalg.inv(
            M_scaled + 1e-8 * torch.eye(M.shape[-1], dtype=M.dtype, device=M.device)
        )
        v = torch.einsum('...ij,...j->...i', M_inv, p_mu)
        return 0.5 * (p_mu * v).sum()

    def potential_energy(self, state: PhaseState, beta: Optional[Tensor] = None) -> Tensor:
        """Compute V = F(μ, Σ, φ)."""
        mu_q, Sigma_q, _, phi = state
        fe = free_energy_total(
            mu_q, Sigma_q, self.mu_p, self.Sigma_p,
            phi, self.generators,
            kappa=self.kappa,
            alpha=self.alpha,
            lambda_belief=self.lambda_belief,
            lambda_prior=self.lambda_prior,
            lambda_obs=self.lambda_obs,
            observations=self.observations,
            W_obs=self.W_obs,
            R_obs=self.R_obs,
            beta=beta,
        )
        return fe.total
