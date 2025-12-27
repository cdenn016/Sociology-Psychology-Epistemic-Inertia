# Epistemic Inertia: A Variational Free Energy Framework for Social Belief Dynamics

A computational framework for modeling how beliefs evolve in social systems, based on **Variational Free Energy (VFE)** minimization and **information geometry**.

## For Sociology/Psychology Researchers

This framework provides a principled mathematical foundation for studying:
- **Belief updating** under social influence
- **Epistemic inertia**: Why some people resist changing their beliefs
- **Polarization dynamics**: How groups form and persist
- **Expert vs. novice** belief updating patterns
- **Social influence networks**: Who affects whom, and how much

## Quick Start

```python
from epistemic_inertia import (
    create_agents,
    MultiAgentSystem,
    Trainer,
    SociologyPresets,
)

# Create 10 agents with polarization preset
agents = create_agents(
    n_agents=10,
    preset=SociologyPresets.POLARIZATION,
)

# Build the social system
system = MultiAgentSystem(agents)

# Train (simulate belief evolution)
trainer = Trainer(system)
history = trainer.train(n_steps=100)

# Analyze results
print(f"Final polarization: {history.compute_polarization():.3f}")
```

## Conceptual Glossary

| Mathematical Term | Sociological Meaning |
|-------------------|---------------------|
| **Agent** | Individual with beliefs |
| **μ_q (mu_q)** | Current belief state (mean opinion) |
| **Σ_q (Sigma_q)** | Belief uncertainty (confidence) |
| **μ_p (mu_p)** | Prior ideology (background beliefs) |
| **Σ_p (Sigma_p)** | Prior certainty (ideological rigidity) |
| **β_ij (beta)** | Social influence weight (how much i listens to j) |
| **KL divergence** | Belief disagreement measure |
| **Free energy S** | Total cognitive dissonance |
| **Gradient ∇S** | Direction of belief change |
| **Mass matrix M** | **Epistemic inertia** (resistance to change) |

## The Core Insight: Epistemic Inertia

The key theoretical contribution is the **mass matrix** (epistemic inertia), a complete 4-term formula:

```
M_i = Λ_p + Λ_o + Σ_k β_ik Ω_ik Λ_q,k Ω_ik^T + (Σ_j β_ji) Λ_q,i

Where:
    Λ_p = Σ_p^{-1}         Prior precision (ideological commitment)
    Λ_o = R_obs^{-1}       Observation precision (evidential confidence)
    β_ik                   Outgoing attention (who I listen to)
    β_ji                   Incoming attention (who listens to me)
    Ω_ik                   Parallel transport (frame alignment)
    Λ_q,k = Σ_q,k^{-1}     Belief precision of agent k
```

**The four terms represent:**

| Term | Formula | Sociological Meaning |
|------|---------|---------------------|
| **Prior inertia** | Σ_p^{-1} | Strong ideology → hard to change |
| **Observation inertia** | R_obs^{-1} | Confident in evidence → hard to change |
| **Outgoing social** | Σ_k β_ik Ω_ik Λ_q,k Ω_ik^T | Listening to confident others → inherit their inertia |
| **Incoming social** | (Σ_j β_ji) Λ_q,i | Being listened to → my own beliefs stabilized |

**High inertia agents** (large M_i):
- Have strong priors (ideologically committed)
- Have high-confidence observations (experts)
- Listen to confident others (inherit inertia)
- Are listened to by many (social status stabilizes beliefs)

**These agents resist belief change**, matching empirical observations:
- Experts update more slowly than novices
- High-status individuals resist social pressure
- Ideologically committed individuals show confirmation bias
- Well-connected individuals have more stable beliefs

## Module Structure

```
Sociology-Psychology-Epistemic-Inertia/
├── epistemic_inertia.py     # Simplified entry point for researchers
├── agent/                   # Agent and system implementations
│   ├── agents.py            # Individual agent (beliefs + priors)
│   ├── system.py            # Multi-agent social network
│   ├── trainer.py           # Belief evolution (gradient flow)
│   └── hamiltonian_trainer.py # Alternative: momentum-based dynamics
├── geometry/                # Information geometry foundations
│   ├── multi_agent_mass_matrix.py # Epistemic inertia (key!)
│   └── ...
├── gradients/               # Free energy and gradient computation
│   ├── free_energy_clean.py # VFE functional
│   └── gradient_engine.py   # Belief update gradients
├── experiments/             # Empirical validation
│   ├── manifold_epistemic_inertia/  # Prediction market data
│   └── metaculus_epistemic_inertia/ # Forecaster data
├── meta/                    # Group-level emergence
│   └── emergence.py         # Meta-agent (group consensus) formation
└── analysis/                # Visualization and analysis tools
```

## Key Features

### 1. Information-Geometric Attention
Social influence weights emerge from belief similarity:
```
β_ij = softmax(-KL(q_i || q_j) / κ)
```
Agents listen more to those they already agree with.

### 2. Gauge Invariance
Beliefs are compared in a coordinate-independent way, meaning results don't depend on how you parameterize the belief space.

### 3. Hamiltonian Dynamics
Optional momentum-based belief evolution that captures "belief momentum" - the tendency of beliefs to overshoot equilibrium.

### 4. Renormalization Group
Hierarchical emergence of group-level beliefs (meta-agents) from individual beliefs.

## Experiments

### Manifold Markets (Prediction Markets)
Tests epistemic inertia on real trader data:
```bash
cd experiments/manifold_epistemic_inertia
python run_pipeline.py
```

### Metaculus (Forecasting)
Tests epistemic inertia on forecaster data:
```bash
cd experiments/metaculus_epistemic_inertia
python run_pipeline.py
```

## Mathematical Details

For researchers interested in the mathematical foundations, see:
- `classical_models_as_limits.md`: How this reduces to classical models (DeGroot, Friedkin-Johnsen)
- `derivations_sociology_manuscript.tex`: Rigorous proofs and derivations

## Installation

```bash
pip install numpy scipy matplotlib
# Optional: for experiments
pip install requests pandas
```

## Citation

If you use this framework, please cite:
```bibtex
@software{epistemic_inertia2025,
  author = {Denniston, Chris and Denniston, Christine},
  title = {Epistemic Inertia: A Variational Free Energy Framework},
  year = {2025},
  url = {https://github.com/...}
}
```

## License

MIT License
