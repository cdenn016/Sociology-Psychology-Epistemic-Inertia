# -*- coding: utf-8 -*-
"""
Epistemic Inertia Framework
===========================

A Variational Free Energy (VFE) framework for modeling belief dynamics
in social systems, with applications to sociology and psychology.

Quick Start:
------------
```python
from epistemic_inertia import (
    create_agents,
    MultiAgentSystem,
    Trainer,
    SociologyPresets,
)

# Create agents
agents = create_agents(10, preset=SociologyPresets.POLARIZATION)

# Build system and simulate
system = MultiAgentSystem(agents)
trainer = Trainer(system)
history = trainer.train(n_steps=100)
```

Key Modules:
------------
- epistemic_inertia: Simplified API for researchers
- agent: Agent and multi-agent system implementations
- geometry: Information geometry and epistemic inertia
- gradients: VFE computation and belief updates
- meta: Hierarchical emergence (renormalization group)
- analysis: Visualization and analysis tools
- experiments: Empirical validation on real data

Author: Robert C. Dennis
Date: 2025
"""

__version__ = "1.0.0"
__author__ = "Robert C. Dennis"

# Re-export simplified API
from epistemic_inertia import (
    Agent,
    MultiAgentSystem,
    Trainer,
    TrainingHistory,
    SociologyPresets,
    SociologyConfig,
    create_agents,
    compute_epistemic_inertia,
    compute_polarization,
    compute_social_influence_matrix,
)

__all__ = [
    # Version info
    '__version__',
    '__author__',

    # Core classes
    'Agent',
    'MultiAgentSystem',
    'Trainer',
    'TrainingHistory',

    # Presets
    'SociologyPresets',
    'SociologyConfig',

    # Factory functions
    'create_agents',

    # Analysis
    'compute_epistemic_inertia',
    'compute_polarization',
    'compute_social_influence_matrix',
]
