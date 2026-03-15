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
├── manuscripts/                     # LaTeX manuscripts
│   ├── belief_inertia_unified.tex   # "The Inertia of Belief" (main paper)
│   ├── belief_inertia.tex           # Core framework paper
│   └── GL(K)_attention.tex          # Gauge-theoretic attention
├── agent/                           # Agent and system implementations
│   ├── agents.py                    # Individual agent (beliefs + priors)
│   ├── system.py                    # Multi-agent social network
│   ├── trainer.py                   # Belief evolution (gradient flow)
│   └── hamiltonian_trainer.py       # Momentum-based dynamics
├── geometry/                        # Information geometry foundations
│   ├── multi_agent_mass_matrix.py   # Epistemic inertia (key!)
│   └── ...
├── gradients/                       # Free energy and gradient computation
│   ├── free_energy_clean.py         # VFE functional
│   └── gradient_engine.py           # Belief update gradients
├── experiments/                     # Empirical validation (8 experiments)
│   ├── EXPERIMENTS.md               # Full experiment design document
│   ├── spf_inertia/                 # Fed Survey of Professional Forecasters
│   ├── stackoverflow_inertia/       # Stack Overflow reputation → rigidity
│   ├── manifold_epistemic_inertia/  # Prediction market data
│   ├── metaculus_epistemic_inertia/ # Forecaster belief data
│   ├── wikipedia_inertia/           # Editor influence → rigidity
│   ├── wikipedia_oscillation/       # Edit wars as belief oscillation
│   ├── openalex_inertia/            # Citation network retraction decay
│   ├── anes_inertia/                # Political belief persistence (panel)
│   ├── financial_inertia/           # Analyst forecast revisions
│   └── reddit_echo_chambers/        # Echo chamber threshold test
├── meta/                            # Group-level emergence
│   └── emergence.py                 # Meta-agent (group consensus) formation
├── torch_core/                      # PyTorch GPU-accelerated core
├── docs/                            # Hypotheses, peer review, math verification
│   └── HYPOTHESES.md                # 16 testable hypotheses with prioritization
├── tests/                           # Unit and integration tests
└── analysis/                        # Visualization and analysis tools
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

Eight real-world experiments test the framework's predictions using open public datasets. Each maps the mass formula to concrete, measurable proxies. See [`experiments/EXPERIMENTS.md`](experiments/EXPERIMENTS.md) for full design details.

### Run Immediately (no downloads, no auth)

| Experiment | Dataset | Hypothesis | Command |
|-----------|---------|-----------|---------|
| **SPF Forecasters** | Federal Reserve individual panel data | H2.1 Oscillation, H1.2 Relaxation, H1.1 Overshoot | `cd experiments/spf_inertia && python run_pipeline.py` |
| **Stack Overflow** | Stack Exchange Data Explorer (SQL) | Reputation → edit rigidity | `cd experiments/stackoverflow_inertia && python fetch_data.py --print-queries` |
| **Manifold Markets** | Manifold Markets API | 4-term mass formula | `cd experiments/manifold_epistemic_inertia && python run_pipeline.py` |

### Run with API Calls (no bulk download)

| Experiment | Dataset | Hypothesis | Command |
|-----------|---------|-----------|---------|
| **Wikipedia Inertia** | MediaWiki API | H3.1 Page watchers → revert resistance | `cd experiments/wikipedia_inertia && python fetch_data.py` |
| **Wikipedia Oscillation** | MediaWiki API (edit wars) | H2.1 Edit war ω ∝ 1/√M | `cd experiments/wikipedia_oscillation && python detect_oscillation.py` |
| **OpenAlex Citations** | OpenAlex + CrossRef APIs | H1.2 Retraction citation decay (Cox PH) | `cd experiments/openalex_inertia && python fetch_data.py` |

### Requires Registration or Download

| Experiment | Dataset | Hypothesis | Command |
|-----------|---------|-----------|---------|
| **ANES Panel** | American National Election Studies | H4.1 Precision → persistence, H3.4 context-dependent stubbornness | `cd experiments/anes_inertia && python analyze_panel.py --demo` |
| **Financial Forecasts** | SPF + yfinance | Forecast revision oscillation (runs test) | `cd experiments/financial_inertia && python fetch_and_analyze.py` |
| **Reddit Echo Chambers** | Reddit API | H3.2 Threshold ‖Δμ‖² > 2σ²κ log(N) | `cd experiments/reddit_echo_chambers && python analyze_polarization.py` |

### Mass Formula Proxies Across Datasets

```
M_i = Λ_p          + Λ_o             + Σβ_ik·Λ̃_qk          + Σβ_ji·Λ_qi
      ───            ───               ──────────              ──────────
SPF:  experience     accuracy          consensus proximity     influence on consensus
SO:   reputation     tag expertise     comment network         answer views × score
Wiki: edit count     topic edits       followed editors        page watchers
OAlex: h-index      field pubs        co-author citations     cited_by_count
ANES: certainty     knowledge         discussion network      persuasion attempts
```

## Testable Predictions

The framework generates 16 falsifiable hypotheses (see [`docs/HYPOTHESES.md`](docs/HYPOTHESES.md)):

| Prediction | Equation | Distinguishes From |
|-----------|----------|-------------------|
| Overshoot ∝ √(precision) | d = \|μ̇\| √(M/K) | Linear scaling, no overshoot |
| Belief oscillation | ω = √(K/M - γ²/4M²) | Monotonic convergence (gradient descent) |
| Resonant persuasion | ω_res = √(K/M) | "More exposure = more change" |
| Relaxation time ∝ precision | τ = M/γ | Content-dependent persistence |
| Influence → rigidity | M_out = Σ_j β_ji · Λ_qi | Rigidity as individual trait |
| Echo chamber threshold | ‖Δμ‖² > 2σ²κ log(N) | Topology-dependent polarization |

## Classical Models as Limiting Cases

The unified manuscript derives these established models as special cases:

| Model | Limit | Depends on Ansatz? |
|-------|-------|-------------------|
| **DeGroot** social learning | Overdamped, flat gauge, no priors | No |
| **Friedkin-Johnsen** | Overdamped + prior anchoring | No |
| **Bounded confidence** | Low temperature κ → 0 | No |
| **Echo chambers** | Softmax attention + bimodal beliefs | No |
| **Social Impact Theory** | Interpretive correspondence | No |

## Mathematical Details

For the complete mathematical framework, see:
- [`manuscripts/belief_inertia_unified.tex`](manuscripts/belief_inertia_unified.tex): Full unified manuscript with proofs
- [`manuscripts/belief_inertia.tex`](manuscripts/belief_inertia.tex): Core framework paper
- [`docs/MATH_VERIFICATION.md`](docs/MATH_VERIFICATION.md): Mathematical verification procedures

## Installation

```bash
pip install numpy scipy matplotlib

# For experiments
pip install requests pandas tqdm seaborn

# Optional: for specific experiments
pip install statsmodels              # Logistic regression (SO, ANES)
pip install lifelines                # Cox PH survival analysis (OpenAlex)
pip install yfinance                 # Financial data (financial_inertia)
pip install vaderSentiment textblob  # Sentiment analysis (Reddit)
```

## Citation

If you use this framework, please cite:
```bibtex
@software{epistemic_inertia2025,
  author = {Dennis, Robert C.},
  title = {Epistemic Inertia: A Variational Free Energy Framework},
  year = {2025},
  url = {https://github.com/cdenn016/Sociology-Psychology-Epistemic-Inertia}
}
```

## License

MIT License
