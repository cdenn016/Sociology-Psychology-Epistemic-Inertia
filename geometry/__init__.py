# -*- coding: utf-8 -*-
"""
Geometry Module
===============

Information geometry and differential geometric structures for belief dynamics.

Mathematical Foundation:
------------------------
Beliefs live on a statistical manifold where:
- The natural metric is the Fisher information (KL divergence Hessian)
- Geodesics represent "shortest paths" between belief states
- Curvature affects how beliefs "bend" toward attractors

Key Components:
---------------
1. geometry_base: Base manifold C and support regions χ(c)
2. multi_agent_mass_matrix: Epistemic inertia M_i (the core novel contribution)
3. geodesic_corrections: Curvature forces for Hamiltonian dynamics
4. pullback_metrics: Information metric pullback operations
5. gauge_consensus: SO(3) frame consensus for meta-agents

Sociology/Psychology Interpretation:
------------------------------------
- Base manifold C: The "conceptual space" over which beliefs are defined
- Support region χ_i(c): What topics agent i has beliefs about
- Mass matrix M_i: Epistemic inertia - how hard it is to change beliefs
- Geodesic: The natural path of belief change under social influence
- Curvature: How the "shape" of belief space affects dynamics

The Key Insight (Epistemic Inertia):
------------------------------------
The mass matrix M_i combines:
1. Prior precision (strong priors → hard to change)
2. Observation precision (confident observations → hard to change)
3. Social connections (influential agents → hard to change)

This explains why high-status individuals, experts, and well-connected
agents show greater resistance to belief updating.
"""

from .geometry_base import (
    BaseManifold,
    TopologyType,
    SupportRegion,
    create_full_support,
)

from .multi_agent_mass_matrix import (
    build_full_mass_matrix,
    build_mu_mass_matrix,
    MassMatrixConfig,
)

from .geodesic_corrections import (
    compute_geodesic_force,
    GeodesicConfig,
)

from .pullback_metrics import (
    compute_pullback_metric,
)

__all__ = [
    # Base manifold
    'BaseManifold',
    'TopologyType',
    'SupportRegion',
    'create_full_support',

    # Mass matrix (epistemic inertia)
    'build_full_mass_matrix',
    'build_mu_mass_matrix',
    'MassMatrixConfig',

    # Geodesic corrections
    'compute_geodesic_force',
    'GeodesicConfig',

    # Pullback metrics
    'compute_pullback_metric',
]
