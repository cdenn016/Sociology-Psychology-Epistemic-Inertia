# -*- coding: utf-8 -*-
"""
Math Utilities Module
=====================

Mathematical utilities for the Variational Free Energy framework.

Core utilities:
    - transport: Gauge transport operators (parallel transport Ω_ij)
    - push_pull: Gaussian push-pull operations
    - sigma: Covariance field initialization
    - generators: SO(3) Lie algebra generators
    - numerical_utils: KL divergence, softmax, entropy
    - fisher_metric: Information-geometric Fisher metric
    - so3_frechet: Fréchet mean on SO(3) manifold

Sociological interpretation:
    - Transport: How beliefs are compared across different cultural frames
    - Push/pull: How agents incorporate information from others
    - Generators: Orientation of agent's reference frame
"""

import numpy as np

# Core utilities (no circular dependencies)
from .transport import compute_transport as transport_operator
from .push_pull import push_gaussian as transport_gaussian, GaussianDistribution
from .generators import generate_so3_generators
from .numerical_utils import (
    safe_inv,
    kl_gaussian,
)
from .so3_frechet import average_gauge_frames_so3

# Deferred imports: these modules have circular dependencies
# (fisher_metric → gradients → math_utils, sigma → agent → math_utils)
def __getattr__(name):
    if name == 'CovarianceFieldInitializer':
        from .sigma import CovarianceFieldInitializer
        return CovarianceFieldInitializer
    if name == 'compute_fisher_matrix':
        from .fisher_metric import natural_gradient_gaussian as compute_fisher_matrix
        return compute_fisher_matrix
    raise AttributeError(f"module 'math_utils' has no attribute {name!r}")


def symmetrize(M: np.ndarray) -> np.ndarray:
    """Symmetrize a matrix: (M + M^T) / 2"""
    return 0.5 * (M + np.swapaxes(M, -1, -2))


def ensure_spd(Sigma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Ensure matrix is symmetric positive definite.

    In sociological terms: ensures belief uncertainty is well-defined
    (you can't have negative uncertainty about anything).

    Args:
        Sigma: Covariance matrix
        eps: Small regularization constant

    Returns:
        SPD covariance matrix
    """
    Sigma = symmetrize(Sigma)
    K = Sigma.shape[-1]
    Sigma = Sigma + eps * np.eye(K, dtype=Sigma.dtype)
    return Sigma


__all__ = [
    # Core transport
    'transport_operator',
    'transport_gaussian',
    'GaussianDistribution',

    # SO(3) operations
    'generate_so3_generators',
    'average_gauge_frames_so3',

    # Numerical utilities
    'safe_inv',
    'kl_gaussian',

    # Covariance utilities
    'CovarianceFieldInitializer',
    'symmetrize',
    'ensure_spd',

    # Information geometry
    'compute_fisher_matrix',
]
