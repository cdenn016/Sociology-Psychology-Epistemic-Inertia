# -*- coding: utf-8 -*-
"""
Analysis Module
===============

Comprehensive analysis and visualization toolkit for belief dynamics simulations.

Modules:
--------
- core: Data loading and trajectory analysis
- plots: Visualization functions (energy landscapes, belief fields, trajectories)

Sociology/Psychology Applications:
----------------------------------
This module provides tools for:
1. Tracking belief convergence/divergence over time
2. Visualizing opinion polarization
3. Analyzing social influence patterns (who affects whom)
4. Measuring epistemic inertia effects
5. Detecting phase transitions (consensus → polarization)

Typical Workflow:
-----------------
```python
from analysis import core, plots

# Load simulation results
history = core.loaders.load_history('simulation_output.pkl')
system = core.loaders.load_system('final_system.pkl')

# Visualize belief evolution
plots.mu_tracking.plot_belief_trajectories(history)

# Analyze influence patterns
plots.softmax.plot_attention_heatmap(system)

# Energy decomposition
plots.fields.plot_energy_components(history)
```
"""

from . import core
from . import plots

__all__ = ['core', 'plots']
