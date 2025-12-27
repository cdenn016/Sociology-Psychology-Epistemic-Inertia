# -*- coding: utf-8 -*-
"""
Epistemic Inertia Framework
===========================

Simplified entry point for sociology/psychology researchers.

This module provides a clean, high-level API for simulating belief dynamics
in social systems. It abstracts away the mathematical complexity of
information geometry and Hamiltonian mechanics.

Quick Start:
------------
```python
from epistemic_inertia import (
    create_agents,
    MultiAgentSystem,
    Trainer,
    SociologyPresets,
)

# Create agents with a preset
agents = create_agents(n_agents=10, preset=SociologyPresets.POLARIZATION)

# Build the social system
system = MultiAgentSystem(agents)

# Simulate belief evolution
trainer = Trainer(system)
history = trainer.train(n_steps=100)

# Analyze results
print(f"Belief convergence: {history.total_energy[-1]:.3f}")
```

Key Concepts:
-------------
- **Agent**: An individual with beliefs (μ_q) and prior ideology (μ_p)
- **Belief uncertainty**: Σ_q - how confident the agent is
- **Social influence**: β_ij - how much agent i listens to agent j
- **Epistemic inertia**: M_i - how resistant agent i is to belief change
- **Free energy**: S - total "cognitive dissonance" in the system

Author: Chris & Christine Denniston
Date: December 2025
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np

from agent.agents import Agent
from agent.system import MultiAgentSystem
from agent.trainer import Trainer, TrainingHistory
from agent.hamiltonian_trainer import HamiltonianTrainer, HamiltonianHistory
from config import AgentConfig, SystemConfig, TrainingConfig
from geometry.multi_agent_mass_matrix import build_mu_mass_matrix


class SociologyPresets(Enum):
    """
    Preset configurations for common sociological scenarios.

    CONSENSUS: All agents start with similar beliefs, low noise
        - Models: Convergent opinion dynamics
        - Expected outcome: Rapid convergence to shared belief

    POLARIZATION: Two distinct groups with opposing beliefs
        - Models: Political polarization, us-vs-them dynamics
        - Expected outcome: Two stable clusters

    ECHO_CHAMBERS: Multiple isolated groups
        - Models: Social media filter bubbles
        - Expected outcome: Multiple stable clusters

    EXPERT_VS_NOVICE: Mix of high and low certainty agents
        - Models: Expert influence, epistemic authority
        - Expected outcome: Novices converge toward experts

    BACKFIRE: Strong priors that resist evidence
        - Models: Confirmation bias, backfire effect
        - Expected outcome: Beliefs may strengthen despite contrary evidence

    NEUTRAL: Default balanced configuration
        - Models: General belief dynamics
        - Expected outcome: Depends on initial conditions
    """
    CONSENSUS = "consensus"
    POLARIZATION = "polarization"
    ECHO_CHAMBERS = "echo_chambers"
    EXPERT_VS_NOVICE = "expert_vs_novice"
    BACKFIRE = "backfire"
    NEUTRAL = "neutral"


@dataclass
class SociologyConfig:
    """
    User-friendly configuration for sociology/psychology simulations.

    This wraps the internal AgentConfig and SystemConfig with
    intuitive parameter names.

    Attributes:
        belief_dimensions: How many independent belief dimensions (default: 3)
        initial_belief_spread: How diverse initial beliefs are (0-1)
        prior_strength: How strongly agents stick to priors (0-1)
        social_influence_strength: How much agents influence each other (0-1)
        uncertainty_level: How uncertain agents are about beliefs (0-1)
        learning_rate: How fast beliefs update (0-1)
        temperature: Softmax temperature for social influence (higher = more uniform)
    """
    belief_dimensions: int = 3
    initial_belief_spread: float = 0.5
    prior_strength: float = 0.3
    social_influence_strength: float = 0.5
    uncertainty_level: float = 0.3
    learning_rate: float = 0.1
    temperature: float = 1.0

    def to_agent_config(self) -> AgentConfig:
        """Convert to internal AgentConfig."""
        return AgentConfig(
            K=self.belief_dimensions,
            mu_scale=self.initial_belief_spread,
            sigma_scale=self.uncertainty_level,
            lr_mu_q=self.learning_rate,
            lr_sigma_q=self.learning_rate * 0.1,
        )

    def to_system_config(self) -> SystemConfig:
        """Convert to internal SystemConfig."""
        return SystemConfig(
            lambda_self=1.0,
            lambda_belief_align=self.social_influence_strength,
            lambda_prior_align=self.prior_strength,
            kappa_beta=self.temperature,
        )


def get_preset_config(preset: SociologyPresets) -> SociologyConfig:
    """
    Get configuration for a preset scenario.

    Args:
        preset: One of SociologyPresets enum values

    Returns:
        SociologyConfig configured for the scenario
    """
    configs = {
        SociologyPresets.CONSENSUS: SociologyConfig(
            initial_belief_spread=0.1,
            prior_strength=0.1,
            social_influence_strength=0.8,
            uncertainty_level=0.2,
        ),
        SociologyPresets.POLARIZATION: SociologyConfig(
            initial_belief_spread=1.0,
            prior_strength=0.5,
            social_influence_strength=0.3,
            uncertainty_level=0.3,
        ),
        SociologyPresets.ECHO_CHAMBERS: SociologyConfig(
            initial_belief_spread=0.8,
            prior_strength=0.6,
            social_influence_strength=0.2,
            uncertainty_level=0.4,
        ),
        SociologyPresets.EXPERT_VS_NOVICE: SociologyConfig(
            initial_belief_spread=0.3,
            prior_strength=0.4,
            social_influence_strength=0.5,
            uncertainty_level=0.5,  # Will be varied per agent
        ),
        SociologyPresets.BACKFIRE: SociologyConfig(
            initial_belief_spread=0.5,
            prior_strength=0.9,
            social_influence_strength=0.1,
            uncertainty_level=0.1,
        ),
        SociologyPresets.NEUTRAL: SociologyConfig(),
    }
    return configs.get(preset, SociologyConfig())


def create_agents(
    n_agents: int,
    preset: SociologyPresets = SociologyPresets.NEUTRAL,
    config: Optional[SociologyConfig] = None,
    seed: Optional[int] = None,
) -> List[Agent]:
    """
    Create a list of agents for simulation.

    This is the main entry point for setting up a simulation.

    Args:
        n_agents: Number of agents to create
        preset: Scenario preset (overridden by config if provided)
        config: Custom configuration (overrides preset)
        seed: Random seed for reproducibility

    Returns:
        List of Agent objects ready for simulation

    Example:
        >>> agents = create_agents(10, preset=SociologyPresets.POLARIZATION)
        >>> system = MultiAgentSystem(agents)
    """
    if config is None:
        config = get_preset_config(preset)

    agent_config = config.to_agent_config()
    rng = np.random.default_rng(seed)

    agents = []
    for i in range(n_agents):
        # For expert_vs_novice, vary uncertainty
        if preset == SociologyPresets.EXPERT_VS_NOVICE:
            # First 20% are experts (low uncertainty)
            if i < n_agents * 0.2:
                agent_config.sigma_scale = 0.1
            else:
                agent_config.sigma_scale = 0.5

        # For polarization, create two groups
        if preset == SociologyPresets.POLARIZATION:
            agent_rng = np.random.default_rng(seed + i if seed else None)
            # Bias initial beliefs toward +/- based on group
            if i < n_agents // 2:
                agent_config.mu_scale = 0.5  # One group
            else:
                agent_config.mu_scale = -0.5  # Other group (sign doesn't matter, just different)

        agent = Agent(
            agent_id=i,
            config=agent_config,
            rng=np.random.default_rng(seed + i if seed else None),
        )
        agents.append(agent)

    return agents


def compute_epistemic_inertia(system: MultiAgentSystem) -> np.ndarray:
    """
    Compute the epistemic inertia (mass matrix) for all agents.

    Epistemic inertia measures how resistant each agent is to belief change.
    High inertia = agent is hard to persuade.

    Components of inertia:
    1. Prior precision: Strong priors → high inertia
    2. Observation precision: Confident observations → high inertia
    3. Social connections: Many influential connections → high inertia

    Args:
        system: MultiAgentSystem to analyze

    Returns:
        Array of shape (n_agents,) with inertia values

    Example:
        >>> inertia = compute_epistemic_inertia(system)
        >>> print(f"Agent 0 inertia: {inertia[0]:.3f}")
        >>> print(f"Most stubborn agent: {np.argmax(inertia)}")
    """
    M = build_mu_mass_matrix(system)

    # Extract diagonal elements (per-agent inertia)
    # M is block diagonal, so we take trace of each block
    n_agents = system.n_agents
    K = system.agents[0].K

    inertia = np.zeros(n_agents)
    for i in range(n_agents):
        block = M[i*K:(i+1)*K, i*K:(i+1)*K]
        inertia[i] = np.trace(block) / K  # Average eigenvalue

    return inertia


def compute_polarization(system: MultiAgentSystem) -> float:
    """
    Compute a polarization measure for the system.

    Polarization is high when beliefs cluster into distinct groups.
    Uses variance of pairwise belief distances.

    Args:
        system: MultiAgentSystem to analyze

    Returns:
        Polarization score (0 = consensus, higher = more polarized)

    Example:
        >>> polarization = compute_polarization(system)
        >>> print(f"Polarization: {polarization:.3f}")
    """
    beliefs = np.array([agent.mu_q.mean(axis=tuple(range(agent.mu_q.ndim - 1)))
                        for agent in system.agents])

    # Compute pairwise distances
    n = len(beliefs)
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(np.linalg.norm(beliefs[i] - beliefs[j]))

    distances = np.array(distances)

    # Polarization = bimodality of distances
    # High variance relative to mean indicates clustering
    if len(distances) == 0:
        return 0.0

    return float(np.var(distances) / (np.mean(distances) + 1e-8))


def compute_social_influence_matrix(system: MultiAgentSystem) -> np.ndarray:
    """
    Compute the social influence matrix β_ij.

    β_ij represents how much agent i listens to agent j.
    Higher values = more influence.

    Args:
        system: MultiAgentSystem to analyze

    Returns:
        Matrix of shape (n_agents, n_agents) with influence weights.
        Rows sum to 1 (softmax normalized).

    Example:
        >>> beta = compute_social_influence_matrix(system)
        >>> print(f"Agent 0 is most influenced by agent {np.argmax(beta[0])}")
    """
    from gradients.softmax_grads import compute_kl_matrix

    kl_matrix = compute_kl_matrix(system, mode='belief')
    kappa = system.config.kappa_beta

    # Softmax with temperature
    beta = np.exp(-kl_matrix / kappa)
    beta = beta / beta.sum(axis=1, keepdims=True)

    return beta


# Re-export key classes for convenience
__all__ = [
    # Main classes
    'Agent',
    'MultiAgentSystem',
    'Trainer',
    'TrainingHistory',
    'HamiltonianTrainer',
    'HamiltonianHistory',

    # Configuration
    'SociologyPresets',
    'SociologyConfig',
    'AgentConfig',
    'SystemConfig',
    'TrainingConfig',

    # Factory functions
    'create_agents',
    'get_preset_config',

    # Analysis functions
    'compute_epistemic_inertia',
    'compute_polarization',
    'compute_social_influence_matrix',
]
