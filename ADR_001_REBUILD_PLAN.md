# ADR-001: Phased Rebuild of Epistemic Inertia Simulation Framework

**Status:** Proposed
**Date:** 2026-03-18
**Author:** Code review by Claude, directed by R.C. Dennis
**Target Hardware:** NVIDIA RTX 5090 (32GB VRAM, ~1.5 PFLOPS FP16)

---

## Context

The Epistemic Inertia simulation codebase supports two research programs:

1. **Belief Inertia Study** — Validating the unified manuscript's predictions: damped oscillator dynamics, multi-agent momentum transfer, classical sociological model recovery (DeGroot, Friedkin-Johnsen, bounded confidence, echo chambers), and 8 empirical experiments against public datasets.

2. **Participatory "It-From-Bit" Study** — Emergent geometry from informational dynamics: Fisher-Rao pullback metrics inducing Riemannian structure on base manifolds, hierarchical meta-agent formation via RG coarse-graining, and cross-scale information flow.

The current codebase is ~77 files across 14 directories, entirely CPU/NumPy with optional Numba JIT. A code review identified:

- **4 critical bugs** (mass matrix accumulation, missing agent methods, emergence shape errors, unused info accumulator)
- **O(N²) agent loops** in 5 key functions that prevent scaling beyond ~8-50 agents
- **Zero test coverage** on all meta/emergence modules
- **A complete `torch_core/` already exists** (6 modules, 23 passing tests) that ports the core math to PyTorch with GPU support — but it's not connected to the agent/system layer

The question is how to get from the current state to a GPU-accelerated, well-tested system that can run both studies on the 5090.

---

## Decision

**Phased rebuild**: Keep the mathematical architecture and `torch_core/` foundation. Rewrite the orchestration layer (agents, system, meta) to operate on batched PyTorch tensors. Fix bugs. Add tests. Connect experiments.

This is NOT a ground-up rewrite. It's a targeted refactor that preserves ~60% of existing code (all math, all geometry theory, all experiment designs) while replacing the ~40% that's structurally incompatible with GPU execution.

---

## Current Codebase Inventory

### KEEP AS-IS (reference / theory)
| Module | Reason |
|--------|--------|
| `geometry/pullback_metrics.py` | Excellent math, "it-from-bit" core — port to torch later |
| `geometry/geometry_base.py` | Clean χ-weight abstraction |
| `geometry/lie_algebra.py` | Correct SO(3)/GL(d) algebra (fix SO(1,3) exp) |
| `geometry/connection.py` | Infrastructure-ready, extend for curvature |
| `gradients/gradient_terms.py` | Verified Euclidean gradients (verify chain rule line 320) |
| `math_utils/generators.py` | Correct SO(3) irreps with caching |
| `experiments/EXPERIMENTS.md` | 8 experiments, well-designed, ready to execute |
| `manuscripts/` | LaTeX sources — sync with Physics_manuscripts/ |
| `figures/` | Publication figures — keep |

### KEEP `torch_core/` — EXTEND AND CONNECT
| Module | Status | What's needed |
|--------|--------|---------------|
| `torch_core/distributions.py` | ✅ Full, tested | Connect to new agent layer |
| `torch_core/transport.py` | ⚠️ Refactor | Simplify: agents now carry `Omega ∈ GL(K)` directly, not `phi ∈ gl(K)`. Transport `Omega_{ij} = Omega_i Omega_j^{-1}` is batched matmul — no `rodrigues()` or `matrix_exp_so()` needed in the hot path. Keep `push_mean()`, `push_covariance()`, `kl_transported()` as-is. |
| `torch_core/fisher.py` | ✅ Full, tested | Add pullback metric port |
| `torch_core/free_energy.py` | ⚠️ Has O(N²) loops | Vectorize pairwise KL using batched `Omega_{ij}` |
| `torch_core/mass_matrix.py` | ⚠️ Has O(N²) loops | Vectorize, fix beta accumulation bug |
| `torch_core/dynamics.py` | ✅ Full, autograd | Extend for meta-agent dynamics; update `PhaseState` to carry `Omega` not `phi` |

### REWRITE (structurally incompatible with GPU)
| Module | Problem | Replacement |
|--------|---------|-------------|
| `agent/agents.py` | Python objects, not tensors | `BatchedAgentState` tensor container |
| `agent/system.py` | Agent-by-agent loops, missing methods | `TensorSystem` with vectorized ops |
| `agent/trainer.py` | NumPy gradient application | Autograd-based training loop |
| `agent/hamiltonian_trainer.py` | Manual Hessian, no SPD validation | Use `torch_core/dynamics.py` |
| `gradients/gradient_engine.py` | 56KB monolith, mixed backends | Replace with `torch.autograd.grad` |
| `gradients/softmax_grads.py` | Hand-coded softmax Jacobian | Autograd handles this |
| `gradients/update_engine.py` | Manual retraction | Riemannian manifold optimizers |

### REWRITE (meta/emergence — buggy, untested)
| Module | Problem | Replacement |
|--------|---------|-------------|
| `meta/emergence.py` | Shape bugs, 2272 lines, untested | Modular: `MetaAgentDetector` + `RenormalizationEngine` |
| `meta/hierarchical_evolution.py` | Unused accumulator, reimplements adapter | Clean hierarchical stepper |
| `meta/consensus.py` | No tests, O(N²) | Batched consensus with spectral methods |

### DISCARD (replaced by torch_core)
| Module | Reason |
|--------|--------|
| `gradients/free_energy_clean.py` | Replaced by `torch_core/free_energy.py` |
| `gradients/gauge_fields.py` | Retraction logic obsolete — GL(K) is open, no retraction needed |
| `gradients/retraction.py` | Same — `Omega ∈ GL(K)` eliminates manifold projection |
| `math_utils/transport.py` | Replaced by `torch_core/transport.py` |
| `math_utils/fisher_metric.py` | Replaced by `torch_core/fisher.py` |
| `math_utils/sigma.py` | Initialization moves to `BatchedAgentState` factory |

---

## New Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EXPERIMENT LAYER                         │
│  experiments/spf_inertia/  wikipedia/  stackoverflow/  ...      │
│  Each experiment: config.yaml + run.py + analysis.py            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                      SIMULATION ENGINE                          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │ TensorSystem │  │ Hamiltonian  │  │ MetaAgent             │ │
│  │              │  │ Integrator   │  │ Emergence             │ │
│  │ N agents as  │  │              │  │                       │ │
│  │ batch dim    │  │ Verlet /     │  │ Consensus detection   │ │
│  │              │  │ Euler with   │  │ Renormalization       │ │
│  │ step()       │  │ autograd     │  │ Hierarchical dynamics │ │
│  │ free_energy()│  │ forces       │  │                       │ │
│  └──────┬───────┘  └──────┬───────┘  └───────────┬───────────┘ │
│         │                 │                       │             │
│  ┌──────▼─────────────────▼───────────────────────▼───────────┐ │
│  │                   torch_core (GPU)                         │ │
│  │                                                            │ │
│  │  distributions.py  — Batched KL, entropy, sanitization     │ │
│  │  transport.py      — GL(K) transport Ω_ij = Ω_i Ω_j⁻¹    │ │
│  │  free_energy.py    — Vectorized F = KL + β·KL + obs       │ │
│  │  mass_matrix.py    — Batched 4-term M_i formula            │ │
│  │  dynamics.py       — Hamiltonian + damped dynamics          │ │
│  │  fisher.py         — Natural gradients                     │ │
│  │  pullback.py [NEW] — Fisher-Rao pullback for it-from-bit  │ │
│  │  curvature.py [NEW]— Gauge holonomy F_γ, frustration      │ │
│  │  attention.py [NEW]— Mixture-of-sources softmax attention  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   ANALYSIS TOOLS                           │ │
│  │  rg_metrics.py     — Modularity, effective rank, flow      │ │
│  │  diagnostics.py    — Energy decomposition, convergence     │ │
│  │  visualization.py  — Phase portraits, attention heatmaps   │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Core Data Structure: `BatchedAgentState`

```python
@dataclass
class BatchedAgentState:
    """All N agents as batched tensors on a single device."""
    mu: Tensor          # (N, K)         beliefs (means)
    sigma: Tensor       # (N, K, K)      beliefs (covariances)
    Omega: Tensor       # (N, K, K)      gauge frames, Omega_i ∈ GL(K)
    mu_p: Tensor        # (N, K)         priors (means)
    sigma_p: Tensor     # (N, K, K)      priors (covariances)
    observations: Tensor # (N, K)        current observations
    obs_precision: Tensor # (N, K, K)    observation noise precision
    device: torch.device

    @staticmethod
    def create(N, K, device='cuda', **init_kwargs) -> 'BatchedAgentState':
        ...
```

#### Gauge Frame Parameterization: `Omega ∈ GL(K)` (not `phi ∈ gl(K)`)

Each agent carries a **frame matrix** `Omega_i ∈ GL(K)` directly, rather than a Lie algebra element `phi_i ∈ gl(K)` with `Omega_i = exp(phi_i)`. This is a deliberate design choice:

- **Transport is trivial**: `Omega_{ij} = Omega_i @ Omega_j^{-1}` — one matmul + one inverse, both trivially batchable. No matrix exponential in the forward pass.
- **GL(K) is open in R^{K×K}**: The constraint `det(Omega) ≠ 0` is generically satisfied. Gradient steps from a non-singular matrix stay non-singular for reasonable learning rates. No retraction needed.
- **Richer than SO(K)**: GL(K) admits scaling and shearing, not just rotations. An agent's frame can encode anisotropic rescaling of the belief space, which is physically meaningful (e.g., one agent weights certain belief dimensions more than others).
- **Previous approach**: The old codebase parameterized `GL+(K)` via `Omega_i = exp(phi_i · G)` where `phi ∈ gl(K)` and `G` are Lie algebra generators. This required SO(K) generators, Rodrigues' formula (K=3), or `torch.linalg.matrix_exp` (general K). All of that complexity is eliminated.

**Covariance transport** remains `Sigma' = Omega @ Sigma @ Omega^T`, which preserves SPD for any invertible `Omega`. For non-orthogonal frames, `Omega^T ≠ Omega^{-1}`, so this is a congruence transform, not a similarity transform — mathematically correct for changing basis of a quadratic form.

**Initialization**: `Omega_i = I + epsilon * randn(K, K)` with `epsilon` small enough to stay in `GL(K)`. Or initialize as random orthogonal matrices via QR decomposition if SO(K) initialization is desired.

### Key Design Principles

1. **Agents are indices, not objects.** Agent $i$ is `state.mu[i]`, not `agents[i].mu`. All operations are batched over the agent dimension.

2. **Autograd replaces hand-coded gradients.** The free energy `F(state)` is a differentiable scalar. Forces come from `torch.autograd.grad(F, [state.mu, state.sigma, state.Omega])`. This eliminates `gradient_engine.py`, `gradient_terms.py`, and `softmax_grads.py` entirely.

3. **Pairwise operations are matmul, not loops.** The O(N²) KL matrix becomes:
   ```python
   # All pairwise transports: (N, N, K, K) — batched matmul, no exp needed
   Omega_inv = torch.linalg.inv(state.Omega)              # (N, K, K)
   Omega_ij = state.Omega[:, None] @ Omega_inv[None, :]   # (N, N, K, K)
   # All pairwise KL: (N, N) — single batched call
   kl_matrix = batched_pairwise_kl(state.mu, state.sigma, Omega_ij)
   # Attention: (N, N)
   beta = torch.softmax(-kl_matrix / tau, dim=-1)
   ```

4. **Meta-agents are a recursive application of the same `BatchedAgentState`.** Scale $\ell+1$ is just another `BatchedAgentState` whose beliefs are coherence-weighted averages of scale $\ell$ clusters.

---

## Phased Plan

### Phase 0: Bug Fixes and Test Baseline (3 days)

**Goal:** Make the existing codebase correct before rebuilding on top of it.

**Deliverables:**
- [ ] Fix `multi_agent_mass_matrix.py:549` — incoming beta accumulation (overwrite → sum)
- [ ] Fix `emergence.py:428` — shape broadcasting in `update_prior_from_global_state()`
- [ ] Fix `lie_algebra.py:341` — replace SO(1,3) exp placeholder with `scipy.linalg.expm`
- [ ] Verify `gradient_terms.py:320` — chain rule for transport gradient (Ω vs Ω^T) — note: this uses the old `phi` parameterization; verify math still holds for reference, but the new engine bypasses this via autograd on `Omega` directly
- [ ] Add `agent.update_beliefs()`, `agent.update_priors()`, `agent.update_gauge()` stubs or fix caller
- [ ] Run existing 23 tests, ensure all pass
- [ ] Add 5 regression tests for the fixed bugs

**Exit criterion:** All 28 tests pass. `torch_core/` verified against NumPy reference for N=4 agents.

---

### Phase 1: Vectorize `torch_core/` Pairwise Operations (1 week)

**Goal:** Eliminate O(N²) Python loops in `free_energy.py`, `mass_matrix.py`, and softmax attention.

**Key simplification:** With agents carrying `Omega_i ∈ GL(K)` directly, the pairwise transport computation becomes trivial:
```python
# All N² transports in one batched call — no matrix exp needed
Omega_inv = torch.linalg.inv(Omega)                  # (N, K, K)
Omega_ij = Omega[:, None, :, :] @ Omega_inv[None, :, :, :]  # (N, N, K, K)
```
This eliminates the need for batched Rodrigues or `matrix_exp_so()` entirely. The bottleneck shifts to batched KL computation (log-det + trace on `(N, N, K, K)` tensors), which is standard batched linear algebra.

**Deliverables:**
- [ ] `torch_core/attention.py` [NEW] — Mixture-of-sources softmax attention
  - `compute_all_transports(Omega)` → `(N, N, K, K)` — batched matmul + inverse
  - `pairwise_kl_matrix(mu, sigma, Omega_ij)` → `(N, N)` — batched KL
  - `softmax_attention(kl_matrix, tau, pi)` → `(N, N)` with prior support
  - Causal masking via `pi_j = 0` for future agents
- [ ] `torch_core/transport.py` — Refactor for GL(K) frames
  - Keep `push_mean()`, `push_covariance()`, `kl_transported()` (already take `Omega` as input)
  - Add `compute_all_transports(Omega)` → `(N, N, K, K)` batched
  - Deprecate `rodrigues()`, `matrix_exp_so()`, `compute_transport(phi_i, phi_j, generators)` — move to `torch_core/lie.py` utility module for reference/testing
- [ ] `torch_core/free_energy.py` — Replace loops with einsum/batched ops
  - `free_energy_alignment(state, beta)` — fully vectorized
  - `free_energy_total(state)` — single forward pass, returns scalar
- [ ] `torch_core/mass_matrix.py` — Vectorized 4-term formula
  - `mass_diagonal(state, beta)` → `(N, K, K)` — all agents simultaneously
  - `mass_full(state, beta)` → `(N*K, N*K)` — full block matrix
- [ ] Tests: Parity with loop-based reference for N=2,4,8 agents, K=3,5

**Performance target:** N=100, K=5 on GPU in <100ms per step (currently impossible).

**Exit criterion:** `torch_core/` handles N=100 agents on GPU. All 23+5 existing tests pass. New vectorized tests pass.

---

### Phase 2: `TensorSystem` + Autograd Training Loop (1 week)

**Goal:** Replace `agent/system.py` + `agent/trainer.py` with GPU-native system.

**Deliverables:**
- [ ] `engine/state.py` — `BatchedAgentState` dataclass with factory methods
  - `create(N, K, device)`, `from_numpy(agents)`, `to_numpy()`
  - `Omega` initialized as `I + eps*randn` or via QR for random orthogonal start
  - SPD validation on sigma mutations
  - GL(K) validation on Omega (det ≠ 0 check, condition number monitoring)
- [ ] `engine/system.py` — `TensorSystem`
  - `free_energy(state) → scalar` (differentiable)
  - `step_gradient(state, lr) → state` (natural gradient descent = overdamped limit)
  - `step_hamiltonian(state, dt, gamma) → state` (Verlet integration with damping)
  - Adjacency/masking support for sparse networks
- [ ] `engine/integrators.py` — Symplectic integrators
  - Störmer-Verlet (leapfrog) with state-dependent mass
  - Euler-Maruyama for stochastic dynamics
  - Energy monitoring and adaptive stepping
- [ ] Autograd replaces all hand-coded gradients:
  ```python
  F = system.free_energy(state)
  grad_mu, grad_sigma, grad_Omega = torch.autograd.grad(F, [state.mu, state.sigma, state.Omega])
  ```
  Note: The gradient `∂F/∂Omega` lives in `gl(K) ≅ R^{K×K}` (tangent space of GL(K) at Omega_i). For gradient descent, update `Omega_i ← Omega_i - lr * grad_Omega_i`. GL(K) is open, so this stays in GL(K) for small enough lr.
- [ ] Tests: Reproduce all 5 belief_inertia simulation figures (damping, momentum transfer, stopping distance, resonance, perseverance) from `torch_core` engine
- [ ] Numerical gradient checks: `autograd` vs finite-difference for each VFE component

**Exit criterion:** All 5 manuscript simulation figures reproduced on GPU. Autograd gradients match finite-difference to 1e-4 relative error.

---

### Phase 3: Meta-Agent Emergence and RG Analysis (1-2 weeks)

**Goal:** Port and fix the hierarchical system for the "it-from-bit" study.

**Deliverables:**
- [ ] `engine/consensus.py` — Batched consensus detection
  - Pairwise KL matrix → spectral clustering or modularity optimization
  - Consensus threshold as function of κ (attention temperature)
  - Returns cluster assignments and coherence scores
- [ ] `engine/renormalization.py` — RG coarse-graining
  - `renormalize(state, clusters) → MetaState` — coherence-weighted averaging
  - Meta-agent beliefs, precisions, gauge frames (`Omega`) from constituents
  - Recursive: `MetaState` is itself a `BatchedAgentState`
- [ ] `engine/rg_metrics.py` — RG observables
  - Modularity Q(β) of attention matrix
  - Effective rank of belief covariance (von Neumann entropy)
  - Within-cluster KL (should decrease with scale)
  - Between-cluster KL (should remain stable)
  - Scale-dependent coupling constants
- [ ] `torch_core/pullback.py` [NEW] — Port `geometry/pullback_metrics.py` to PyTorch
  - Fisher-Rao pullback metric on base manifold
  - Belief-induced and prior-induced metrics
  - Sector decomposition (observable/dark/internal DOF)
- [ ] `engine/diagnostics.py` — Port `meta/participatory_diagnostics.py`
  - Per-agent energy decomposition (self, alignment, model, obs)
  - Per-scale energy aggregates
  - Cross-scale information flow tracking
  - Non-equilibrium indicators (gradient norms, energy flux)
- [ ] Tests:
  - Renormalization preserves total free energy (up to coarse-graining error)
  - Meta-agent beliefs are coherence-weighted averages of constituents
  - RG metrics show expected trends (modularity ↑, rank ↓) on synthetic data
  - Pullback metric matches NumPy reference

**Exit criterion:** Hierarchical system runs 3 scales on GPU. RG metrics computable. Pullback geometry produces emergent metric from informational dynamics.

---

### Phase 4: Gauge Curvature and New Experiments (1 week)

**Goal:** Implement the gauge curvature predictions from manuscript §5.6 and connect to the 8 empirical experiments.

**Deliverables:**
- [ ] `torch_core/curvature.py` [NEW] — Gauge holonomy and frustration
  - `holonomy(Omega_ij, loop_indices) → (n_loops, K, K)` — compute F_γ for all triangles
  - `frustration_energy(state, triangles)` — residual energy from gauge incompatibility
  - `curvature_norm(holonomy)` — scalar measure of ||F_γ - I||

  **Design note on flat connections:** When transport is `Omega_{ij} = Omega_i Omega_j^{-1}` (factored through per-agent frames), the connection is flat by construction — holonomy around any loop is `I`. Curvature requires transport operators that **don't factor** into per-agent frames. Two approaches:
  1. **Data-driven transport**: Fit `Omega_{ij}` to empirical frame differences from data (e.g., different expert calibrations). The residual `F_γ = Omega_{12} Omega_{23} Omega_{31} ≠ I` measures genuine gauge frustration.
  2. **Perturbative curvature**: Add connection terms `A_{ij}` so that `Omega_{ij}^{full} = A_{ij} · Omega_i Omega_j^{-1}`, where `A_{ij} ∈ GL(K)` encodes local connection curvature. Holonomy becomes `Prod_{cycle} A_{ij}`.
  - Flat-connection test: verify F_γ = I when `Omega_{ij} = Omega_i Omega_j^{-1}` exactly
- [ ] `experiments/` — Connect experiment infrastructure to new engine
  - Adapter: load public dataset → `BatchedAgentState`
  - Mass proxy computation from data (reputation → Λ_p, watchers → Σβ_ji, etc.)
  - Overdamped simulation runner for classical model recovery
  - Underdamped simulation runner for oscillation/resonance detection
- [ ] Curvature simulations for §5.6 predictions:
  - Intransitive understanding: 3-agent triangle with incompatible frames
  - Frustrated consensus: spin-glass-like ground state search
  - Holonomy-induced polarization: identical beliefs, misaligned frames
  - Phase diagram: fragmentation in (κ, ||F_γ||) plane
- [ ] Tests:
  - Holonomy vanishes for flat gauge (factored `Omega_{ij} = Omega_i Omega_j^{-1}`)
  - Holonomy is gauge-covariant: F'_γ = g F_γ g^{-1}
  - Frustration energy ≥ 0, equals 0 iff flat

**Exit criterion:** Curvature simulations produce figures for manuscript §5.6. At least 2 of the 8 empirical experiments run end-to-end with the new engine.

---

### Phase 5: Optimization, Scaling, and Publication Readiness (1 week)

**Goal:** Performance optimization for the 5090, full test suite, documentation.

**Deliverables:**
- [ ] Profile and optimize GPU utilization
  - Mixed precision (FP16 for KL computation, FP32 for mass matrix)
  - Memory-efficient pairwise KL (chunked for N > 1000)
  - Sparse attention for large networks (top-k or ε-threshold)
- [ ] Benchmark suite: N × K × steps → wall time, GPU memory, energy drift
  - Target: N=1000, K=10 in <1s per step on 5090
  - Target: N=100, K=5, 10000 steps in <60s total
- [ ] Full test suite: ≥80% line coverage on `torch_core/` and `engine/`
  - Property tests: gauge invariance, SPD preservation, energy conservation (undamped)
  - Regression tests: all 5 manuscript figures reproducible to 1e-3 tolerance
  - Integration tests: 3-scale hierarchical system, full evolution
- [ ] Sync manuscripts: copy updated `belief_inertia_unified.tex` to repo
- [ ] README with installation, quickstart, and experiment runner docs
- [ ] CI: GitHub Actions running test suite on CPU (GPU tests marked optional)

**Exit criterion:** All tests pass. N=1000 on GPU. README complete. Ready for publication supplementary code release.

---

## Timeline Summary

| Phase | Duration | Deliverable | Risk |
|-------|----------|-------------|------|
| 0: Bug fixes | 3 days | Correct baseline | Low |
| 1: Vectorize torch_core | 1 week | GPU pairwise ops | Low — math already done |
| 2: TensorSystem + autograd | 1 week | GPU training loop | Medium — integrator stability |
| 3: Meta-agent emergence + RG | 1-2 weeks | Hierarchical system | High — emergence is complex |
| 4: Curvature + experiments | 1 week | New predictions + data | Medium — data wrangling |
| 5: Optimization + polish | 1 week | Publication-ready | Low |

**Total: 5-7 weeks** to publication-ready GPU codebase.

---

## Trade-off Analysis

### Why not full rewrite from scratch?
- `torch_core/` already exists with 23 passing tests — rebuilding this wastes 2+ weeks
- The geometry theory (`pullback_metrics.py`, `lie_algebra.py`, `geometry_base.py`) is correct and non-trivial to re-derive
- The experiment designs (`EXPERIMENTS.md`) are ready to execute — just need a new engine adapter

### Why not just patch the existing code?
- The agent-as-Python-object architecture fundamentally can't batch on GPU
- Hand-coded gradients in `gradient_engine.py` (56KB) are unmaintainable when autograd does it in 3 lines
- The meta/emergence modules have zero tests and multiple bugs — safer to rewrite cleanly

### Why `Omega ∈ GL(K)` instead of `phi ∈ gl(K)` with `Omega = exp(phi)`?

The old parameterization `Omega_i = exp(phi_i · G)` had several costs:
- **Computational**: Matrix exponential in every forward pass. For SO(3) we had Rodrigues; for general K, `torch.linalg.matrix_exp` on `(N, K, K)` tensors. This was the single most expensive operation in the pairwise transport computation.
- **Architectural**: Required choosing and maintaining Lie algebra generators `G_a`. For SO(K) these are well-defined, but the theory naturally lives on GL(K) (frame changes include scaling/shearing, not just rotations).
- **Constraint**: `exp: gl(K) → GL+(K)` only reaches the identity component. Restricting to `GL+(K)` or `SO(K)` is a modeling choice, not a mathematical necessity.

Storing `Omega_i ∈ GL(K)` directly:
- **Eliminates matrix exp** from the forward pass entirely
- **Transport** `Omega_{ij} = Omega_i Omega_j^{-1}` is batched matmul + batched inverse — trivially parallelizable
- **Autograd** through `Omega` works directly in `R^{K×K}`; no chain rule through `exp` needed
- **GL(K) is open** in `R^{K×K}`, so gradient descent stays in GL(K) for small learning rates. Monitor `det(Omega_i)` and condition number as health checks.
- **Initialization flexibility**: Start at `I` (identity frames), random orthogonal (QR), or perturbation `I + εR`

**Trade-off**: We lose the guarantee that `Omega_i ∈ SO(K)` (orthogonal, unit det). This means covariance transport `Sigma' = Omega Sigma Omega^T` is a congruence transform, not a similarity transform. Both preserve SPD. The congruence version also changes the "size" of the covariance (scales eigenvalues), which may be physically meaningful (frame changes that rescale uncertainty).

### Why not JAX instead of PyTorch?
- PyTorch has better debugging (eager mode), more ecosystem support, and the existing `torch_core/` is already PyTorch
- JAX's functional style would require rewriting all state management
- PyTorch 2.x compile mode gives comparable performance to JAX JIT for these workloads

### What if N > 10,000 agents?
- The O(N²) pairwise KL matrix becomes the bottleneck (~400MB for N=10K, K=5)
- Solution: sparse attention with top-k neighbors or ε-threshold
- Not needed for the current experiments (max N ≈ 3000 for ANES panel)
- Phase 5 addresses this with chunked computation

---

## Consequences

**What becomes easier:**
- Running the 8 empirical experiments on GPU with proper mass formula computation
- Adding new gauge curvature simulations (§5.6 predictions)
- Scaling to N > 100 agents for realistic social network simulations
- Debugging via autograd (no more hand-coded gradient errors)
- Testing via property-based checks on tensor operations
- Pairwise transport: `Omega_i Omega_j^{-1}` is a single batched matmul — no matrix exp, no generators, no Rodrigues
- Richer gauge group: GL(K) admits scaling/shearing frames, not just rotations

**What becomes harder:**
- The χ-weighted spatial integration from `geometry_base.py` doesn't have a direct GPU analog yet — this is deferred to a future phase
- Agents with heterogeneous state dimensions (different K per agent) require padding or ragged tensors
- The "it-from-bit" pullback metric requires spatial gradients (`np.gradient`) which need a torch port
- GL(K) frames can become ill-conditioned — need condition number monitoring and optional regularization (e.g., `loss += lambda * ||Omega^T Omega - I||` to softly encourage orthogonality if desired)

**What we'll need to revisit:**
- Spatial manifold support (currently 0D/1D/2D) — for now, focus on 0D (point agents) which covers all 8 experiments
- Stochastic dynamics (Langevin noise) — add in Phase 5 if needed
- Online learning / streaming data — future work, not needed for current studies
- Whether SO(K) restriction is needed for specific experiments — can always add soft orthogonality penalty
