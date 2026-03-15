# -*- coding: utf-8 -*-
"""
Parity Tests: torch_core vs NumPy implementations
===================================================

Verifies that the PyTorch implementations produce identical results
to the original NumPy code, and that autograd gradients are correct.

Run: python tests/test_torch_core.py
  or: python -m pytest tests/test_torch_core.py -v --override-ini="addopts="
"""

import sys
import os
# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import torch

# Tolerances (NumPy returns float32, PyTorch float64 — small diffs expected)
ATOL = 5e-4
RTOL = 1e-3


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def K():
    """Latent dimension."""
    return 3


@pytest.fixture
def N():
    """Number of agents."""
    return 4


@pytest.fixture
def rng():
    """Seeded random state."""
    return np.random.RandomState(42)


@pytest.fixture
def torch_rng():
    """Seeded torch generator."""
    torch.manual_seed(42)
    return None


def _random_spd(K, rng):
    """Generate random SPD matrix."""
    A = rng.randn(K, K)
    return A @ A.T + 0.1 * np.eye(K)


def _random_spd_batch(N, K, rng):
    """Generate batch of random SPD matrices."""
    return np.array([_random_spd(K, rng) for _ in range(N)])


def _so3_generators():
    """Standard SO(3) generators."""
    G = np.zeros((3, 3, 3))
    G[0, 1, 2] = -1; G[0, 2, 1] = 1
    G[1, 0, 2] = 1;  G[1, 2, 0] = -1
    G[2, 0, 1] = -1; G[2, 1, 0] = 1
    return G


# ============================================================================
# Test 1: KL Divergence Parity
# ============================================================================

class TestKLDivergence:
    def test_kl_single_pair(self, K, rng):
        """KL divergence matches NumPy for single pair."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'numerical_utils',
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'math_utils', 'numerical_utils.py')
        )
        nu = importlib.util.module_from_spec(spec)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec.loader.exec_module(nu)

        mu_q = rng.randn(K)
        mu_p = rng.randn(K)
        Sigma_q = _random_spd(K, rng)
        Sigma_p = _random_spd(K, rng)

        # NumPy (direct import to avoid broken __init__)
        kl_numpy = float(nu.kl_gaussian(mu_q, Sigma_q, mu_p, Sigma_p, eps=1e-6))

        # PyTorch
        from torch_core.distributions import kl_gaussian as kl_torch
        kl_pt = kl_torch(
            torch.tensor(mu_q, dtype=torch.float64),
            torch.tensor(Sigma_q, dtype=torch.float64),
            torch.tensor(mu_p, dtype=torch.float64),
            torch.tensor(Sigma_p, dtype=torch.float64),
            eps=1e-6,
        ).item()

        assert abs(kl_numpy - kl_pt) < ATOL, f"KL mismatch: numpy={kl_numpy}, torch={kl_pt}"

    def test_kl_batch(self, K, N, rng):
        """Batched KL divergence is consistent with single-pair KL."""
        mu_q = rng.randn(N, K)
        mu_p = rng.randn(N, K)
        Sigma_q = _random_spd_batch(N, K, rng)
        Sigma_p = _random_spd_batch(N, K, rng)

        from torch_core.distributions import kl_gaussian, kl_gaussian_batch

        # Single-pair (loop)
        kl_single = np.array([
            kl_gaussian(
                torch.tensor(mu_q[i], dtype=torch.float64),
                torch.tensor(Sigma_q[i], dtype=torch.float64),
                torch.tensor(mu_p[i], dtype=torch.float64),
                torch.tensor(Sigma_p[i], dtype=torch.float64),
                eps=1e-6,
            ).item()
            for i in range(N)
        ])

        # Batched
        kl_batch = kl_gaussian_batch(
            torch.tensor(mu_q, dtype=torch.float64),
            torch.tensor(Sigma_q, dtype=torch.float64),
            torch.tensor(mu_p, dtype=torch.float64),
            torch.tensor(Sigma_p, dtype=torch.float64),
            eps=1e-6,
        ).numpy()

        np.testing.assert_allclose(kl_single, kl_batch, atol=ATOL, rtol=RTOL)

    def test_kl_non_negative(self, K, rng):
        """KL divergence is always non-negative."""
        from torch_core.distributions import kl_gaussian as kl_torch

        for _ in range(20):
            mu_q = torch.randn(K, dtype=torch.float64)
            mu_p = torch.randn(K, dtype=torch.float64)
            A = torch.randn(K, K, dtype=torch.float64)
            B = torch.randn(K, K, dtype=torch.float64)
            Sigma_q = A @ A.mT + 0.1 * torch.eye(K, dtype=torch.float64)
            Sigma_p = B @ B.mT + 0.1 * torch.eye(K, dtype=torch.float64)

            kl = kl_torch(mu_q, Sigma_q, mu_p, Sigma_p)
            assert kl.item() >= 0, f"KL negative: {kl.item()}"

    def test_kl_zero_for_identical(self, K):
        """KL(q || q) = 0."""
        from torch_core.distributions import kl_gaussian as kl_torch

        mu = torch.randn(K, dtype=torch.float64)
        A = torch.randn(K, K, dtype=torch.float64)
        Sigma = A @ A.mT + 0.1 * torch.eye(K, dtype=torch.float64)

        kl = kl_torch(mu, Sigma, mu, Sigma)
        assert kl.item() < 1e-6, f"KL(q||q) not zero: {kl.item()}"

    def test_kl_differentiable(self, K):
        """KL divergence is differentiable w.r.t. mu_q."""
        from torch_core.distributions import kl_gaussian as kl_torch

        mu_q = torch.randn(K, dtype=torch.float64, requires_grad=True)
        mu_p = torch.randn(K, dtype=torch.float64)
        A = torch.randn(K, K, dtype=torch.float64)
        Sigma_q = (A @ A.mT + 0.1 * torch.eye(K, dtype=torch.float64)).detach()
        Sigma_p = (A @ A.mT + 0.2 * torch.eye(K, dtype=torch.float64)).detach()

        kl = kl_torch(mu_q, Sigma_q, mu_p, Sigma_p)
        kl.backward()

        assert mu_q.grad is not None
        assert torch.all(torch.isfinite(mu_q.grad))


# ============================================================================
# Test 2: Rodrigues Formula Parity
# ============================================================================

class TestTransport:
    def test_rodrigues_identity(self):
        """exp(0) = I."""
        from torch_core.transport import rodrigues

        phi = torch.zeros(3, dtype=torch.float64)
        R = rodrigues(phi)
        torch.testing.assert_close(R, torch.eye(3, dtype=torch.float64), atol=1e-6, rtol=1e-6)

    def test_rodrigues_orthogonal(self, rng):
        """exp(φ) is orthogonal: RᵀR = I."""
        from torch_core.transport import rodrigues

        for _ in range(10):
            phi = torch.tensor(rng.randn(3) * 2.0, dtype=torch.float64)
            R = rodrigues(phi)
            RtR = R.mT @ R
            torch.testing.assert_close(RtR, torch.eye(3, dtype=torch.float64), atol=1e-5, rtol=1e-5)

    def test_rodrigues_det_one(self, rng):
        """det(exp(φ)) = +1 (proper rotation)."""
        from torch_core.transport import rodrigues

        for _ in range(10):
            phi = torch.tensor(rng.randn(3) * 2.0, dtype=torch.float64)
            R = rodrigues(phi)
            det = torch.det(R)
            assert abs(det.item() - 1.0) < 1e-5, f"det = {det.item()}"

    def test_transport_self_identity(self, rng):
        """Ω_ii = I (self-transport is identity)."""
        from torch_core.transport import compute_transport

        generators = torch.tensor(_so3_generators(), dtype=torch.float64)
        phi = torch.tensor(rng.randn(3), dtype=torch.float64)

        Omega = compute_transport(phi, phi, generators)
        torch.testing.assert_close(Omega, torch.eye(3, dtype=torch.float64), atol=1e-5, rtol=1e-5)

    def test_transport_transitivity(self, rng):
        """Ω_ij · Ω_jk = Ω_ik."""
        from torch_core.transport import compute_transport

        generators = torch.tensor(_so3_generators(), dtype=torch.float64)
        phi_i = torch.tensor(rng.randn(3), dtype=torch.float64)
        phi_j = torch.tensor(rng.randn(3), dtype=torch.float64)
        phi_k = torch.tensor(rng.randn(3), dtype=torch.float64)

        Omega_ij = compute_transport(phi_i, phi_j, generators)
        Omega_jk = compute_transport(phi_j, phi_k, generators)
        Omega_ik = compute_transport(phi_i, phi_k, generators)

        product = Omega_ij @ Omega_jk
        torch.testing.assert_close(product, Omega_ik, atol=1e-4, rtol=1e-4)

    def test_push_preserves_spd(self, K, rng):
        """Pushed covariance is still SPD."""
        from torch_core.transport import push_covariance, rodrigues

        A = torch.tensor(rng.randn(K, K), dtype=torch.float64)
        Sigma = A @ A.mT + 0.1 * torch.eye(K, dtype=torch.float64)
        phi = torch.tensor(rng.randn(3), dtype=torch.float64)
        Omega = rodrigues(phi)

        Sigma_pushed = push_covariance(Sigma, Omega)
        eigs = torch.linalg.eigvalsh(Sigma_pushed)
        assert torch.all(eigs > 0), f"Non-SPD after push: eigs={eigs}"

    def test_kl_transported_matches_manual(self, K, rng):
        """kl_transported matches manual push + KL."""
        from torch_core.transport import kl_transported, push_mean, push_covariance, rodrigues
        from torch_core.distributions import kl_gaussian as kl_torch

        mu_i = torch.tensor(rng.randn(K), dtype=torch.float64)
        mu_j = torch.tensor(rng.randn(K), dtype=torch.float64)
        Sigma_i = torch.tensor(_random_spd(K, rng), dtype=torch.float64)
        Sigma_j = torch.tensor(_random_spd(K, rng), dtype=torch.float64)
        phi = torch.tensor(rng.randn(3), dtype=torch.float64)
        Omega = rodrigues(phi)

        # Method 1: fused
        kl_fused = kl_transported(mu_i, Sigma_i, mu_j, Sigma_j, Omega)

        # Method 2: manual
        mu_j_pushed = push_mean(mu_j, Omega)
        Sigma_j_pushed = push_covariance(Sigma_j, Omega)
        kl_manual = kl_torch(mu_i, Sigma_i, mu_j_pushed, Sigma_j_pushed)

        torch.testing.assert_close(kl_fused, kl_manual, atol=1e-5, rtol=1e-5)


# ============================================================================
# Test 3: Natural Gradient Parity
# ============================================================================

class TestFisherMetric:
    def test_natural_gradient_formula(self, K, rng):
        """Natural gradient matches direct formula: δμ = -Σ ∇μ, δΣ = -2 Σ sym(∇Σ) Σ."""
        mu = rng.randn(K)
        Sigma = _random_spd(K, rng)
        grad_mu = rng.randn(K)
        grad_Sigma = rng.randn(K, K)
        grad_Sigma = 0.5 * (grad_Sigma + grad_Sigma.T)

        # Direct formula (NumPy)
        delta_mu_expected = -Sigma @ grad_mu
        tmp = Sigma @ grad_Sigma
        delta_Sigma_expected = -2.0 * tmp @ Sigma
        delta_Sigma_expected = 0.5 * (delta_Sigma_expected + delta_Sigma_expected.T)

        # PyTorch
        from torch_core.fisher import natural_gradient_gaussian as ng_torch
        delta_mu_pt, delta_Sigma_pt = ng_torch(
            torch.tensor(mu, dtype=torch.float64),
            torch.tensor(Sigma, dtype=torch.float64),
            torch.tensor(grad_mu, dtype=torch.float64),
            torch.tensor(grad_Sigma, dtype=torch.float64),
            eps=1e-6,  # Small eps to match direct formula
        )

        np.testing.assert_allclose(delta_mu_expected, delta_mu_pt.numpy(), atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(delta_Sigma_expected, delta_Sigma_pt.numpy(), atol=ATOL, rtol=RTOL)


# ============================================================================
# Test 4: Free Energy Autograd
# ============================================================================

class TestFreeEnergy:
    def test_self_energy_non_negative(self, K, N, rng, torch_rng):
        """Self-energy (KL) is non-negative."""
        from torch_core.free_energy import free_energy_self

        mu_q = torch.tensor(rng.randn(N, K), dtype=torch.float64)
        mu_p = torch.tensor(rng.randn(N, K), dtype=torch.float64)
        Sigma_q = torch.tensor(_random_spd_batch(N, K, rng), dtype=torch.float64)
        Sigma_p = torch.tensor(_random_spd_batch(N, K, rng), dtype=torch.float64)

        E = free_energy_self(mu_q, Sigma_q, mu_p, Sigma_p)
        assert E.item() >= 0, f"Self-energy negative: {E.item()}"

    def test_self_energy_zero_at_match(self, K, N, rng):
        """Self-energy = 0 when q = p."""
        from torch_core.free_energy import free_energy_self

        mu = torch.tensor(rng.randn(N, K), dtype=torch.float64)
        Sigma = torch.tensor(_random_spd_batch(N, K, rng), dtype=torch.float64)

        E = free_energy_self(mu, Sigma, mu, Sigma)
        assert E.item() < 1e-5, f"Self-energy not zero at q=p: {E.item()}"

    def test_autograd_produces_gradients(self, K, rng):
        """torch.autograd produces finite gradients for free energy."""
        from torch_core.free_energy import free_energy_total

        N = 3
        generators = torch.tensor(_so3_generators(), dtype=torch.float64)

        mu_q = torch.tensor(rng.randn(N, K), dtype=torch.float64, requires_grad=True)
        Sigma_q = torch.tensor(_random_spd_batch(N, K, rng), dtype=torch.float64)
        mu_p = torch.tensor(rng.randn(N, K), dtype=torch.float64)
        Sigma_p = torch.tensor(_random_spd_batch(N, K, rng), dtype=torch.float64)
        phi = torch.tensor(rng.randn(N, 3) * 0.1, dtype=torch.float64)

        fe = free_energy_total(
            mu_q, Sigma_q, mu_p, Sigma_p, phi, generators,
            kappa=1.0, alpha=1.0, lambda_belief=1.0,
        )

        fe.total.backward()
        assert mu_q.grad is not None
        assert torch.all(torch.isfinite(mu_q.grad))
        assert mu_q.grad.abs().sum() > 0, "Gradient is all zeros"

    def test_softmax_attention_rows_sum_to_one(self, K, rng):
        """Softmax attention weights sum to 1 per row."""
        from torch_core.free_energy import softmax_attention

        N = 4
        generators = torch.tensor(_so3_generators(), dtype=torch.float64)

        mu = torch.tensor(rng.randn(N, K), dtype=torch.float64)
        Sigma = torch.tensor(_random_spd_batch(N, K, rng), dtype=torch.float64)
        phi = torch.tensor(rng.randn(N, 3) * 0.1, dtype=torch.float64)

        beta = softmax_attention(mu, Sigma, phi, generators, kappa=1.0)

        # Rows should sum to 1 (excluding self-connections)
        row_sums = beta.sum(dim=1)
        torch.testing.assert_close(row_sums, torch.ones(N, dtype=torch.float64), atol=1e-5, rtol=1e-5)

        # Diagonal should be 0
        diag = beta.diag()
        torch.testing.assert_close(diag, torch.zeros(N, dtype=torch.float64), atol=1e-8, rtol=1e-8)


# ============================================================================
# Test 5: Mass Matrix
# ============================================================================

class TestMassMatrix:
    def test_mass_matrix_spd(self, K, rng):
        """Mass matrix is symmetric positive definite."""
        from torch_core.mass_matrix import mass_matrix_diagonal
        from torch_core.free_energy import softmax_attention

        N = 3
        generators = torch.tensor(_so3_generators(), dtype=torch.float64)

        Sigma_q = torch.tensor(_random_spd_batch(N, K, rng), dtype=torch.float64)
        Sigma_p = torch.tensor(_random_spd_batch(N, K, rng), dtype=torch.float64)
        mu_q = torch.tensor(rng.randn(N, K), dtype=torch.float64)
        phi = torch.tensor(rng.randn(N, 3) * 0.1, dtype=torch.float64)

        beta = softmax_attention(mu_q, Sigma_q, phi, generators, kappa=1.0)

        M = mass_matrix_diagonal(mu_q, Sigma_q, Sigma_p, phi, generators, beta)

        for i in range(N):
            # Symmetric
            torch.testing.assert_close(M[i], M[i].mT, atol=1e-6, rtol=1e-6)
            # Positive definite
            eigs = torch.linalg.eigvalsh(M[i])
            assert torch.all(eigs > 0), f"Agent {i} mass matrix not SPD: eigs={eigs}"


# ============================================================================
# Test 6: Sanitize Sigma
# ============================================================================

class TestSanitizeSigma:
    def test_output_is_spd(self, K, rng):
        """Sanitized covariance is SPD."""
        from torch_core.distributions import sanitize_sigma

        # Create a poorly conditioned matrix
        A = torch.tensor(rng.randn(K, K), dtype=torch.float64)
        Sigma_bad = A @ A.mT + 1e-8 * torch.eye(K, dtype=torch.float64)

        Sigma_clean = sanitize_sigma(Sigma_bad)
        eigs = torch.linalg.eigvalsh(Sigma_clean)
        assert torch.all(eigs >= 1e-4), f"Eigenvalues below floor: {eigs}"

    def test_preserves_good_matrix(self, K, rng):
        """Sanitization doesn't alter a well-conditioned matrix."""
        from torch_core.distributions import sanitize_sigma

        Sigma = torch.tensor(_random_spd(K, rng), dtype=torch.float64)
        Sigma_clean = sanitize_sigma(Sigma, min_eigenvalue=1e-6)

        torch.testing.assert_close(Sigma, Sigma_clean, atol=1e-3, rtol=1e-3)


# ============================================================================
# Test 7: Dynamics (Smoke Test)
# ============================================================================

class TestDynamics:
    def test_verlet_step_runs(self, K, rng):
        """Verlet integration step completes without error."""
        from torch_core.dynamics import HamiltonianDynamics, PhaseState

        N = 3
        generators = torch.tensor(_so3_generators(), dtype=torch.float64)

        mu_q = torch.tensor(rng.randn(N, K), dtype=torch.float64)
        Sigma_q = torch.tensor(_random_spd_batch(N, K, rng), dtype=torch.float64)
        mu_p = torch.tensor(rng.randn(N, K), dtype=torch.float64)
        Sigma_p = torch.tensor(_random_spd_batch(N, K, rng), dtype=torch.float64)
        phi = torch.tensor(rng.randn(N, 3) * 0.1, dtype=torch.float64)
        p_mu = torch.zeros(N, K, dtype=torch.float64)

        dynamics = HamiltonianDynamics(
            mu_p=mu_p, Sigma_p=Sigma_p, generators=generators,
            dt=0.01, friction=1.0, mass_scale=1.0,
            kappa=1.0, alpha=1.0, lambda_belief=0.5,
        )

        state = PhaseState(mu_q, Sigma_q, p_mu, phi)
        new_state = dynamics.step_verlet(state)

        assert torch.all(torch.isfinite(new_state.mu_q))
        assert torch.all(torch.isfinite(new_state.p_mu))

    def test_gradient_flow_decreases_energy(self, K, rng):
        """In overdamped limit (high friction), energy should decrease."""
        from torch_core.dynamics import HamiltonianDynamics, PhaseState

        N = 3
        generators = torch.tensor(_so3_generators(), dtype=torch.float64)

        mu_q = torch.tensor(rng.randn(N, K), dtype=torch.float64)
        Sigma_q = torch.tensor(_random_spd_batch(N, K, rng), dtype=torch.float64)
        mu_p = torch.zeros(N, K, dtype=torch.float64)
        Sigma_p = torch.tensor(_random_spd_batch(N, K, rng), dtype=torch.float64)
        phi = torch.zeros(N, 3, dtype=torch.float64)
        p_mu = torch.zeros(N, K, dtype=torch.float64)

        dynamics = HamiltonianDynamics(
            mu_p=mu_p, Sigma_p=Sigma_p, generators=generators,
            dt=0.001, friction=100.0, mass_scale=1.0,
            kappa=1.0, alpha=1.0, lambda_belief=0.0,
        )

        state = PhaseState(mu_q, Sigma_q, p_mu, phi)
        E_initial = dynamics.potential_energy(state).item()

        for _ in range(10):
            state = dynamics.step_euler(state)

        E_final = dynamics.potential_energy(state).item()
        assert E_final <= E_initial + 1e-3, f"Energy increased: {E_initial} -> {E_final}"
