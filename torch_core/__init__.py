# -*- coding: utf-8 -*-
"""
PyTorch Core for Variational Free Energy on Statistical Manifolds
=================================================================

Pure PyTorch implementations of the VFE framework, designed for:
- GPU acceleration (CUDA / RTX 5090)
- Automatic differentiation (torch.autograd replaces hand-coded gradients)
- torch.compile optimization

Modules:
    distributions  - Gaussian KL, entropy, log-likelihood
    transport      - SO(3) parallel transport operators
    fisher         - Fisher-Rao metric and natural gradients
    free_energy    - Differentiable VFE functional
    mass_matrix    - Epistemic inertia (mass = precision)
    dynamics       - Hamiltonian belief dynamics with autograd

Author: Chris & Christine
Date: March 2026
"""

from torch_core.distributions import (
    kl_gaussian,
    kl_gaussian_batch,
    entropy_gaussian,
    safe_inv,
    sanitize_sigma,
)
from torch_core.transport import (
    rodrigues,
    compute_transport,
    push_mean,
    push_covariance,
    kl_transported,
)
from torch_core.fisher import (
    natural_gradient_gaussian,
)
from torch_core.free_energy import (
    FreeEnergy,
    free_energy_self,
    free_energy_belief_alignment,
    free_energy_total,
)
from torch_core.mass_matrix import (
    mass_matrix_diagonal,
)

__all__ = [
    "kl_gaussian", "kl_gaussian_batch", "entropy_gaussian",
    "safe_inv", "sanitize_sigma",
    "rodrigues", "compute_transport", "push_mean", "push_covariance",
    "kl_transported",
    "natural_gradient_gaussian",
    "FreeEnergy", "free_energy_self", "free_energy_belief_alignment",
    "free_energy_total",
    "mass_matrix_diagonal",
]
