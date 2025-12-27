# -*- coding: utf-8 -*-
"""
Gradients Module
================

Variational Free Energy computation and gradient-based optimization.

Mathematical Foundation:
------------------------
The free energy functional S[q] measures how well an agent's beliefs
balance compression (matching priors) with accuracy (explaining observations).

    S = E_self + E_belief + E_prior + E_obs

Where:
    - E_self: Self-consistency (KL between belief and prior within agent)
    - E_belief: Alignment with other agents' beliefs
    - E_prior: Alignment with other agents' priors
    - E_obs: Fit to observations/evidence

Sociology/Psychology Interpretation:
------------------------------------
- Free energy minimization = reducing cognitive dissonance
- Gradient descent = belief updating through social learning
- The softmax attention ²_ij = who you're listening to
- Mass matrix M = epistemic inertia (how hard it is to change beliefs)

Key Components:
    - compute_total_free_energy: Main VFE functional
    - compute_natural_gradients: Fisher-weighted gradients
    - GradientApplier: Safe parameter updates respecting manifold constraints
"""

from .free_energy_clean import (
    compute_total_free_energy,
    FreeEnergyBreakdown,
    compute_self_energy,
    compute_belief_alignment_energy,
    compute_prior_alignment_energy,
    compute_observation_energy,
)

from .gradient_engine import (
    compute_natural_gradients,
    AgentGradients,
)

from .softmax_grads import (
    compute_softmax_weights,
    compute_kl_matrix,
)

from .update_engine import GradientApplier

from .retraction import (
    retract_spd,
    retract_spd_cholesky,
)

__all__ = [
    # Free energy
    'compute_total_free_energy',
    'FreeEnergyBreakdown',
    'compute_self_energy',
    'compute_belief_alignment_energy',
    'compute_prior_alignment_energy',
    'compute_observation_energy',

    # Gradients
    'compute_natural_gradients',
    'AgentGradients',

    # Softmax attention
    'compute_softmax_weights',
    'compute_kl_matrix',

    # Updates
    'GradientApplier',
    'retract_spd',
    'retract_spd_cholesky',
]
