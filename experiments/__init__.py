# -*- coding: utf-8 -*-
"""
Experiments Module
==================

Empirical validation of epistemic inertia theory using real-world data.

Key Experiments:
----------------
1. manifold_epistemic_inertia/
   - Tests epistemic inertia on Manifold Markets prediction data
   - Validates: High-status traders resist belief updates more

2. metaculus_epistemic_inertia/
   - Tests epistemic inertia on Metaculus forecaster data
   - Validates: Expert forecasters show higher epistemic inertia

3. rg_simulation_metrics.py
   - Renormalization group validation
   - Tests emergence of meta-agents (group consensus formation)

Theory Being Tested:
--------------------
Epistemic inertia predicts:
    M_i ∝ (prior precision) + (observation precision) + Σ_j β_ij (social connections)

High inertia agents:
    - Update beliefs more slowly (smaller Δμ per timestep)
    - Require stronger evidence to change
    - Are more influential in group consensus

This matches sociological observations:
    - High-status individuals resist opinion change
    - Experts are "stubborn" relative to novices
    - Social influence creates belief persistence
"""

# Epistemic inertia experiments are in subdirectories
# See manifold_epistemic_inertia/ and metaculus_epistemic_inertia/
