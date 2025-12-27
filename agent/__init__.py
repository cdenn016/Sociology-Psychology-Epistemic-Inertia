# -*- coding: utf-8 -*-
"""
Agent Module
============

Core agent and multi-agent system implementations for belief dynamics.

Sociology/Psychology Interpretation:
------------------------------------
- Agent: An individual with beliefs (q), priors (p), and a reference frame (Æ)
- MultiAgentSystem: A social network where agents influence each other
- Trainer: How beliefs evolve through social learning
- HamiltonianTrainer: Alternative dynamics with "epistemic momentum"

Key Components:
    - Agent: Single agent with Gaussian belief/prior distributions
    - MultiAgentSystem: Collection of interacting agents
    - Trainer: Gradient flow training (overdamped dynamics)
    - HamiltonianTrainer: Hamiltonian dynamics (underdamped, with inertia)
    - MaskConfig: Support region configuration
"""

from .agents import Agent, AgentGeometry
from .system import MultiAgentSystem
from .trainer import Trainer, TrainingHistory
from .hamiltonian_trainer import HamiltonianTrainer, HamiltonianHistory
from .masking import MaskConfig, SupportPatternConfig

__all__ = [
    'Agent',
    'AgentGeometry',
    'MultiAgentSystem',
    'Trainer',
    'TrainingHistory',
    'HamiltonianTrainer',
    'HamiltonianHistory',
    'MaskConfig',
    'SupportPatternConfig',
]
