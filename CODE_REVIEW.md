# Code Review: VFE Transformer Agent System

## Executive Summary

Review of 6 core files implementing Variational Free Energy (VFE) minimization for multi-agent belief dynamics. Key finding: **Strong mathematical coherence with manuscript equations, but several numerical stability issues and inconsistent covariance storage patterns.**

---

## File 1: agent/agents.py

### What It Does
Implements the `Agent` class as a smooth section of a statistical fiber bundle over base manifold C. Each agent carries belief (q_i) and prior (p_i) Gaussian distributions with gauge field φ_i. Manages initialization, covariance storage (directly as Σ, not Cholesky), and support constraints.

### Key Classes/Functions
- **Agent**: Main agent class with belief/prior distributions
- **AgentGeometry**: Lightweight geometry descriptor
- **_initialize_belief_covariance()**: Initialize Σ_q with smooth spatial structure
- **_initialize_prior_covariance()**: Initialize Σ_p differently from Σ_q
- **_initialize_gauge()**: Initialize gauge field φ with support enforcement
- **_generate_smooth_mean_field()**: Generate μ with controlled magnitude
- **_generate_smooth_gauge_field()**: Generate φ with smoothing and principal ball constraint

### Issues Found

#### 🔴 CRITICAL: Covariance Storage Inconsistency (lines 93-104)
- **Declaration**: States "We store Σ directly (not L)"
- **Reality**: Cached properties L_q and L_p (lines 182-204) compute Cholesky on-demand, which is correct
- **Problem**: Code comment implies Σ is always primary, but cache invalidation logic is scattered
- **Impact**: Potential stale cache issues after parameter updates if `invalidate_caches()` not called consistently

**Recommendation**: Centralize cache invalidation. Add debug assertion in `check_constraints()` to verify L_q matches computed Cholesky(Sigma_q).

#### 🟡 ISSUE: Gauge Field Initialization Principal Ball Safety (lines 845-857)
```python
exceeds = phi_norm > max_norm
if np.any(exceeds):
    scale_factor = np.where(exceeds, max_norm / (phi_norm + 1e-8), 1.0)
    phi_field = phi_field * scale_factor
```
- **Problem**: Rescaling after smoothing violates smoothness assumption
- **Better approach**: Enforce max_norm constraint during generation, not as post-processing

#### 🟡 ISSUE: Covariance Validation Tolerance (line 702)
```python
if np.any(eigs < 1e-6):
    raise AssertionError(...)
```
- Eigenvalue threshold 1e-6 is arbitrary. Should match eps from config (typically 1e-8).

#### 🟡 DESIGN: Gauge Support Enforcement (lines 747-769)
- Support enforcement happens in `__init__` via `_initialize_gauge()`
- Later calls to `enforce_support_constraints()` (lines 606-662) can re-enforce
- But gauge field enforcement uses different fill_value (0.0) vs mask type
- **Unclear**: Is φ = 0 outside support the intended constraint, or should it be smoothly damped?

### Math Consistency Check
- **Belief equation**: μ_q, Σ_q stored directly ✓
- **Prior equation**: μ_p, Σ_p stored directly ✓
- **Gauge field**: φ ∈ so(3) ~ ℝ³ enforced within principal ball |φ| < π ✓
- **Transport operator**: Assumes Ω_ij = exp(φ_i) exp(-φ_j), stored via gauge field ✓

### Code Quality
- ✓ Good docstring coverage with LaTeX notation
- ✓ Type hints present but could be more specific (e.g., `np.ndarray` → `NDArray[np.float32]`)
- ✓ Error handling for Cholesky failures with eigendecomposition fallback (line 243-245)
- ❌ Missing validation that transport operators are orthogonal (SO(3))
- ❌ No tests for gauge covariance under parameter updates

---

## File 2: agent/system.py

### What It Does
Implements `MultiAgentSystem` managing N agents with continuous overlap masks χ_ij(c) ∈ [0,1]. Computes free energy, gradients, and handles geometric integration with softmax coupling weights β_ij.

### Key Classes/Functions
- **MultiAgentSystem**: Main system class
- **_compute_overlap_masks()**: Compute continuous overlap χ_ij = χ_i · χ_j (lines 142-242)
- **compute_softmax_weights()**: Compute β_ij fields for belief/prior alignment (lines 470-516)
- **initialize_observation_model()**: Set up W_obs, R_obs, shared ground truth (lines 598-701)
- **_generate_smooth_ground_truth()**: Generate smooth x_true via sinusoids (lines 811-854)

### Issues Found

#### 🟡 ISSUE: Observation Model Initialization Logic (lines 598-701)
```python
def initialize_observation_model(self, config):
    ...
    R = A.T @ A
    R += config.obs_noise_scale**2 * np.eye(config.D_x)
    R = R.astype(np.float32)
    R /= config.D_x  # Keep variance per dimension ~ obs_noise_scale²
```
- **Problem**: Scaling R by 1/D_x AFTER adding noise changes the noise floor inconsistently
- **Expected**: Noise floor should be obs_noise_scale² throughout (not scaled)
- **Consequence**: Observation precision becomes dimension-dependent in unexpected way

**Fix**: Reorder:
```python
R = A.T @ A
R /= config.D_x
R += config.obs_noise_scale**2 * np.eye(config.D_x)
```

#### 🟡 ISSUE: Agent-Specific Bias Not in Likelihood (line 681)
- Agent gets obs_bias, stored as attribute, but NOT used in observation generation formula
- Line 686: `observation = (y_true + noise + agent.obs_bias)` ✓ correct
- But this is the **only** place bias is used. No mention in:
  - Free energy computation (line 737)
  - Gradient engine (line 747)
- **Impact**: Bias is generated but never incorporated into optimization

#### 🔴 CRITICAL: Missing Agent Update Methods
- `MultiAgentSystem.step()` (lines 754-770) calls:
  ```python
  agent.update_beliefs(grad['delta_mu_q'], grad['delta_Sigma_q'])
  agent.update_priors(grad['delta_mu_p'], grad['delta_Sigma_p'])
  agent.update_gauge(grad['delta_phi'])
  ```
- **Problem**: These methods are never defined in Agent class (agents.py)
- **Consequence**: Training will crash with AttributeError
- **Expected**: Should delegate to GradientApplier or define these methods

#### 🟡 DESIGN: Overlap Mask Continuous vs Boolean (lines 307-332)
- Comment correctly states χ_ij must be continuous for weighted integration
- But system stores only continuous overlap_masks
- Energy computation would need to use these directly
- **Consistency**: Good, but verify gradient engine uses continuous masks consistently

#### 🟡 ISSUE: Connection Initialization (lines 557-590)
- Initializes but never uses
- Parameters passed (e.g., N=3 for 3D) are hardcoded
- **Unclear**: What is the connection field supposed to compute?

### Math Consistency Check
- **Overlap**: χ_ij = χ_i · χ_j ✓ (line 233)
- **Softmax**: β_ij = softmax(-KL(q_i || Ω_ij q_j) / κ) — delegated to compute_softmax_weights() ✓
- **Free energy**: F = α·KL(q||p) + λ_β·Σ_ij β_ij·KL + CE(W_out·μ, y)
  - Delegation to free_energy_clean module (line 737) — cannot verify without that file

### Code Quality
- ✓ Overlap computation separated into two implementations (v1 and v2)
- ✓ Fallback logic for different SupportRegion types (lines 189-217)
- ❌ Two overlap computation methods (_v1 and _v2) create confusion; should consolidate
- ❌ Missing bounds checks on temperature κ (can be <0? should not be)
- ❌ No validation that softmax outputs sum to 1

---

## File 3: agent/trainer.py

### What It Does
Standard gradient-flow trainer with parallel gradient computation and caching. Tracks training history, performs checkpoint/snapshot operations, implements early stopping.

### Key Classes/Functions
- **TrainingHistory**: Metrics container (lines 40-101)
- **Trainer**: Main training loop (lines 140-414)
- **Trainer.step()**: Single optimization step with caching (lines 188-224)
- **Trainer.train()**: Full training loop with logging (lines 231-310)

### Issues Found

#### 🟡 ISSUE: Gradient Norm Recording Disconnect (lines 80-94)
```python
grad_L_q_norm = np.mean([
    np.linalg.norm(g.delta_L_q) if g.delta_L_q is not None else 0.0
    for g in gradients
])

self.grad_norm_mu_q.append(grad_mu_q_norm)
self.grad_norm_Sigma_q.append(grad_L_q_norm)  # Store L-gradient, name as Sigma
```
- **Problem**: History records L-gradient norms but field is named "grad_norm_Sigma_q"
- **Consequence**: Misleading when analyzing training curves (Sigma-gradient ≠ L-gradient in magnitude)
- **Fix**: Rename field to `grad_norm_L_q` or document the mapping

#### 🟡 ISSUE: Gradient Debug Print Hard to Parse (lines 369-415)
```python
print(f"\n [GRAD {self.current_step:05d}]\n  "
      f" μ: |mean={np.mean(mu_norms):.3e}  min={np.min(mu_norms):.3e}  max={np.max(mu_norms):.3e}  |\n  "
      f" L: |mean={np.mean(dL_norms):.3e}  min={np.min(dL_norms):.3e}  max={np.max(dL_norms):.3e}  |\n  "
      ...
)
```
- Prints per-step but only called every log_every steps
- Format is hard to parse (missing agent index info)
- **Recommendation**: Use structured logging or separate per-agent summaries

#### 🟡 ISSUE: Missing Agent Reference (line 393)
```python
L = g.agent.L_q
```
- Assumes gradient object has `.agent` attribute
- Not guaranteed; will crash if gradient format doesn't include this
- **Fix**: Add safe getattr or ensure gradient generator always provides agent

#### 🟡 ISSUE: Early Stopping Check (lines 337-347)
```python
improvement = self.best_energy - current_energy
if improvement > self.config.early_stop_threshold:
    self.best_energy = current_energy
```
- Threshold should be ≥ 0; if set to 0, no improvement accepted
- No validation in config.py that `early_stop_threshold` is sensible
- **Consequence**: Can stop immediately if threshold is 0

#### 🔴 CRITICAL: Snapshot Agent Data Format (lines 121-134)
```python
'Sigma_q': agent.Sigma_q.copy(),  # Store Σ directly (gauge-covariant)
```
- Comment correctly notes Σ is gauge-covariant
- But stored snapshots don't include phi field consistently
- If metrics later try to compute pullback via Ω using phi, will fail
- **Fix**: Always include phi in snapshot if gauge is present

### Math Consistency Check
- **Mass formula M_i = Λ_prior + Λ_obs + ...**: Not computed in trainer, delegated to gradient engine ✓
- **Damping equation μ̈ + γμ̇ + ∇F = 0**: Trainer uses gradient flow (first-order), not Hamiltonian ✓
- **Softmax attention β_ij**: Delegated to gradient engine ✓

### Code Quality
- ✓ Early stopping logic is sound
- ✓ Checkpoint format is pickle-compatible
- ❌ Cache stats method called but not defined (line 307)
- ❌ No recovery logic if checkpoint load fails

---

## File 4: agent/hamiltonian_trainer.py

### What It Does
Implements second-order Hamiltonian dynamics: dθ/dt = G^{-1}p, dp/dt = -∇V with optional friction and geodesic corrections. Extends HamiltonianHistory with kinetic/potential energy tracking.

### Key Classes/Functions
- **HamiltonianHistory**: Extended metrics (lines 55-175)
- **HamiltonianTrainer**: Main trainer (lines 178-600+)
- **_pack_parameters()**: Flatten θ = (μ_q, Sigma_q upper triangle) (lines 282-312)
- **_unpack_parameters()**: Restore agent parameters from θ (lines 314-349+)

### Issues Found

#### 🔴 CRITICAL: Parameter Packing Inconsistency (lines 302-312)
```python
Sigma_flat = agent.Sigma_q.reshape(-1, K, K)
for Sigma_mat in Sigma_flat:
    upper_indices = np.triu_indices(K)
    Sigma_upper = Sigma_mat[upper_indices]
    params.append(Sigma_upper)
```
- **Problem**: Reshapes full spatial Sigma_q → (-1, K, K), iterates
- **If spatial_shape = (H, W)**: Sigma_q shape is (H, W, K, K), flatten gives (H·W, K, K) ✓
- **If spatial_shape = ()**  (particle): Sigma_q shape is (K, K), reshape(-1, K, K) → (1, K, K) ✓
- **If spatial_shape = (5,)** (1D): Sigma_q shape is (5, K, K), reshape(-1, K, K) → (5, K, K) ✓
- **Consistency**: Appears correct but no validation in unpacking

#### 🔴 CRITICAL: SPD Reconstruction in Unpacking (lines 342-347)
```python
Sigma_mat = np.zeros((K, K))
upper_indices = np.triu_indices(K)
Sigma_mat[upper_indices] = Sigma_upper
# Symmetrize (copy upper to lower)
Sigma_mat = Sigma_mat + Sigma_mat.T - np.diag(np.diag(Sigma_mat))
```
- **Problem**: Reconstruction assumes upper triangle includes diagonal
- **Verification**: `np.triu_indices(K)` returns indices where i ≤ j (diagonal included) ✓
- **But**: Subtraction `- np.diag(diag)` removes diagonal duplication — correct ✓
- **Issue**: No validation that reconstructed Sigma_mat is SPD
- **Consequence**: Hamiltonian dynamics could become ill-conditioned

#### 🟡 ISSUE: Mass Matrix Assembly (Referenced but not shown)
- Line 48: `from geometry.multi_agent_mass_matrix import build_full_mass_matrix`
- Used in step() but implementation not reviewed here
- **Critical**: Mass matrix M is essential for correct dynamics dθ/dt = G^{-1}p
- **Risk**: If M is not Fisher metric but Euclidean, dynamics are completely wrong

#### 🟡 ISSUE: Geodesic Correction (lines 218-220, 229-230)
```python
enable_geodesic_correction: bool = True
geodesic_eps: float = 1e-5
```
- Geodesic force computes dp/dt = -∇V - curvature
- But curvature calculation requires dM/dθ, which is expensive
- **No guidance**: When should this be enabled vs disabled?
- **Risk**: Finite difference with eps=1e-5 may be too coarse for SPD manifold

#### 🟡 ISSUE: Friction/Damping Not in Equations (line 210-214)
- Trainer accepts `friction` parameter but equation dp/dt = -∇V should include -γ·G^{-1}p damping term
- **Not shown in docstring**: Where is friction applied?
- **Consequence**: Code may not implement underdamped → overdamped transition correctly

#### 🟡 DESIGN: Phase Space Tracking (lines 240-251)
- Optional tracking with PhaseSpaceTracker
- Stored snapshots not integrated into main history
- **Impact**: Phase space analysis is decoupled from optimization metrics
- **Unclear**: How are snapshots exported? Where is visualization code?

### Math Consistency Check
- **Hamiltonian**: H = (1/2)p^T G^{-1} p + V(θ), where V = F ✓
- **Equations of motion**: dθ/dt = ∂H/∂p = G^{-1}p ✓, dp/dt = -∂H/∂θ = -∇V ✓
- **Friction**: Should be dp/dt = -∇V - γ·G^{-1}p (not shown explicitly)
- **Geodesic correction**: Manifold structure corrections to momentum evolution (appears implemented but not shown)

### Code Quality
- ✓ History tracking is comprehensive
- ✓ Parameter packing handles spatial dimensions correctly
- ❌ SPD enforcement missing in _unpack_parameters
- ❌ No test for energy conservation E(t) ≈ const
- ❌ Friction parameter not clearly connected to actual momentum updates

---

## File 5: agent/masking.py

### What It Does
Implements smooth support regions with continuous masks χ(c) ∈ [0,1]. Provides field enforcement (mean, covariance, gauge) with smooth transitions. MaskConfig controls mask type (hard, smooth, gaussian) and thresholds.

### Key Classes/Functions
- **MaskConfig**: Configuration for mask behavior (lines 24-59)
- **SupportRegionSmooth**: Support with continuous mask (lines 66-412)
- **FieldEnforcer**: Enforce support constraints on fields (lines 529-725)
- **SupportPatternsSmooth**: Factory methods (circle, rectangle) (lines 418-522)

### Issues Found

#### 🟡 ISSUE: Gaussian Mask Sigma Calculation (lines 172-180)
```python
t_ref = max(self.config.overlap_threshold, self.config.min_mask_for_normal_cov)
if 0.0 < t_ref < 1.0:
    sigma = radius / np.sqrt(-2.0 * np.log(t_ref))
else:
    sigma = self.config.gaussian_sigma * radius
```
- **Math**: t_ref = exp(-R²/(2σ²)) ⟹ σ = R/√(-2 log t_ref) ✓
- **Edge case**: If t_ref = 0, log(0) → -∞, σ → 0 (support disappears)
- **Boundary**: If t_ref = 1, log(1) = 0, σ → ∞ (no support boundary)
- **Fix**: Add explicit check: `if t_ref <= 0.0 or t_ref >= 1.0:`

#### 🟡 ISSUE: Covariance Enforcement Strategy (lines 659-700)
```python
use_smooth_transition: bool = False  # Usually False for computation
```
- Comment says "usually False", but initialization code uses True (line 334)
- **Inconsistency**: Should this be configurable per context?
- **Consequence**: Covariance outside support may be:
  - Smooth interpolation (initialization): soft boundary
  - Boolean gating (computation): sharp boundary
- **Risk**: Dynamics may be different between training and inference

#### 🟡 ISSUE: Cholesky Enforcement Outside Support (lines 577-592)
```python
L_outside = np.sqrt(outside_scale) * np.eye(K, dtype=L_raw.dtype)
```
- **Assumption**: L = √λ·I ⟹ Σ = λ·I (large diagonal covariance)
- **Problem**: Only correct if L is lower triangular with positive diagonal
- **Risk**: If L_outside is not lower triangular, reconstruction fails
- **Fix**: Add explicit `np.tril(L_outside)` before use

#### 🟡 ISSUE: Smooth Transition S-Curve (lines 600-604)
```python
t = (chi - lower_bound) / (2 * transition_width)
t = np.clip(t, 0, 1)
alpha = 3 * t**2 - 2 * t**3  # Hermite interpolation
```
- Math is correct Hermite S-curve: α(0)=0, α(1)=1, α'(0)=α'(1)=0 ✓
- **But**: Transition band is only 2·transition_width wide
- **If transition_width = 0.5·threshold**: Total band = threshold
- **Consequence**: Very sharp transition, may cause numerical issues

#### 🔴 CRITICAL: Positive Diagonal Enforcement Idempotency (lines 613-625)
```python
diag_fixed = np.where(
    diag_vals <= eps,
    eps,          # clamp bad / tiny values up to eps
    diag_vals,    # keep existing good values
)
```
- **Good intent**: Preserve positive diagonals, fix negative ones
- **Issue**: If diag_vals = 2e-6 and eps = 1e-6, clamps to 1e-6
- **Consequence**: Repeated enforcement shrinks positive diagonals
- **Better**: Use `np.maximum(diag_vals, eps)` instead

### Math Consistency Check
- **Support mask**: χ(c) ∈ [0,1] defined geometrically ✓
- **Overlap**: χ_ij = χ_i · χ_j ✓
- **Field enforcement**: μ = 0 outside, Σ = λI outside, φ = 0 outside ✓
- **Gaussian mask**: χ(r) = exp(-r²/(2σ²)), σ yoked to radius ✓

### Code Quality
- ✓ Comprehensive mask types (hard, smooth, gaussian)
- ✓ Overlap computation with proper thresholding
- ✓ Factory methods for common patterns
- ❌ Diagonal enforcement loop iterates K times instead of vectorized
- ❌ No test for SupportRegionSmooth vs SupportRegion compatibility

---

## File 6: config.py

### What It Does
Configuration system with dataclass definitions for SystemConfig, AgentConfig, TrainingConfig. Includes validation, convenience properties, and preset factory functions for sociology/psychology scenarios.

### Key Classes/Functions
- **SystemConfig**: System-level parameters (lines 37-186)
- **AgentConfig**: Agent-level parameters (lines 199-390)
- **TrainingConfig**: Training-loop parameters (lines 402-440)
- **Preset functions**: get_consensus_config(), get_polarization_config(), etc. (lines 447-577)

### Issues Found

#### 🟡 ISSUE: Observation RNG Seed (lines 183-186)
```python
def get_obs_rng(self) -> np.random.Generator:
    """Get random generator for observation model."""
    seed = self.seed if self.seed is not None else 0
    return np.random.default_rng(seed + 10)
```
- **Problem**: Hardcoded offset "+10" is arbitrary
- **Risk**: If user sets seed=10, observation RNG seed = 20 (predictable but not obvious)
- **Better**: Use descriptive naming or document the offset

#### 🟡 ISSUE: Connection Initialization Parameters (lines 69-72)
```python
use_connection: bool = False
connection_init_mode: Literal['flat', 'random', 'constant'] = 'flat'
connection_scale: float = 1.0
connection_const: Optional[np.ndarray] = None
```
- Parameters exist but never used meaningfully
- **Consequence**: Code in system.py initializes but never uses connection
- **Cleanup**: Remove or implement properly

#### 🟡 ISSUE: Missing Config Validation Linking (lines 135-155)
```python
if self.D_x <= 0:
    raise ValueError(f"D_x must be positive, got {self.D_x}")
```
- Validates D_x dimension, but no check that D_x matches observation model assumptions
- **Risk**: If D_x changes after system creation, observation matrix W_obs has wrong shape

#### 🟡 DESIGN: Observation Model Separation (lines 75-98)
- Observation parameters (D_x, W_scale, R_scale, noise_scale, bias_scale, modes, amplitude, seed) are in SystemConfig
- But observation model initialization is in MultiAgentSystem.initialize_observation_model()
- **Inconsistency**: Config-system split makes dependencies unclear

#### 🟡 ISSUE: Preset Functions Hard to Discover (lines 447-577)
- Preset configs (consensus, polarization, expert_novice, backfire) are defined but not documented
- No docstring examples showing how to use them
- **Risk**: Users might not know presets exist

#### 🟡 ISSUE: Learning Rates Redundancy (lines 223-227, 410-414)
- Learning rates appear in both AgentConfig and TrainingConfig
- **Unclear**: Which takes precedence?
- **Current code**: TrainingConfig values seem unused (no reference in trainer.py)

### Math Consistency Check
- **Lambdas**: λ_self, λ_β, λ_γ, λ_obs all configurable ✓
- **Temperatures**: κ_β, κ_γ (softmax temperatures) configurable ✓
- **Masks**: Thresholds (overlap_threshold, min_mask_for_normal_cov) configurable ✓

### Code Quality
- ✓ Comprehensive validation in __post_init__
- ✓ Properties encapsulate derived values (n_spatial_points, is_particle, etc.)
- ✓ Preset functions exemplify common configurations
- ❌ Learning rates duplicated (AgentConfig + TrainingConfig)
- ❌ Connection parameters dead code

---

## Summary Table

| File | Purpose | Critical Issues | Warnings | Quality |
|------|---------|-----------------|----------|---------|
| **agents.py** | Agent/belief management | Covariance cache consistency | Gauge support enforcement, eigenvalue threshold | Good |
| **system.py** | Multi-agent system | Missing update methods (update_beliefs, etc.) | Observation bias not used in loss, overlap mask boolean/continuous mix | Good |
| **trainer.py** | Standard gradient trainer | Snapshot missing phi | Gradient norm naming confusion, missing agent reference | Good |
| **hamiltonian_trainer.py** | Second-order dynamics | SPD reconstruction unvalidated, mass matrix undefined | Friction not shown in equations, geodesic epsilon too coarse | Fair |
| **masking.py** | Support region enforcement | Positive diagonal clamping not idempotent | Gaussian sigma edge cases, smooth transition S-curve narrowness | Good |
| **config.py** | Configuration system | None | Observation RNG offset arbitrary, learning rates duplicated | Good |

---

## Cross-File Consistency Issues

### 1. **Covariance Storage (agents.py vs hamiltonian_trainer.py)**
- agents.py: Stores Σ directly, computes L on-demand
- hamiltonian_trainer.py: Packs/unpacks Σ upper triangle (not Cholesky)
- **Consistent** ✓ but requires care during parameter updates

### 2. **Support Enforcement (agents.py vs masking.py)**
- agents.py: Calls FieldEnforcer methods
- masking.py: Defines FieldEnforcer with smooth transitions
- **Consistent** ✓ but distinction between smooth (init) and boolean (compute) is subtle

### 3. **Observation Model (system.py vs config.py)**
- system.py: Initializes W_obs, R_obs, generates x_true
- config.py: Holds D_x, scales, noise parameters
- **Problem**: Observation bias is generated but never used in loss computation
- **Impact**: Training may not actually minimize observation error

### 4. **Missing Agent Methods (system.py line 760-763)**
```python
agent.update_beliefs(grad['delta_mu_q'], grad['delta_Sigma_q'])
agent.update_priors(grad['delta_mu_p'], grad['delta_Sigma_p'])
agent.update_gauge(grad['delta_phi'])
```
- **These are never defined in Agent class**
- **Will crash at runtime** during training
- **Must add** to agents.py or change system.py to use GradientApplier

---

## Manuscript Equation Alignment

| Equation | Implementation | Status |
|----------|---|---|
| **M_i = Λ_prior + Λ_obs + Σ_k β_ik Λ̃_qk + Σ_j β_ji Λ_qi** | Delegated to gradient_engine.compute_natural_gradients() | ✓ Not reviewed (external) |
| **M*μ̈ + γ*μ̇ + ∇F = 0** | hamiltonian_trainer.py implements this with friction | ✓ Second-order dynamics |
| **β_ij = softmax(-KL(q_i \|\| Ω_ij q_j)/τ)** | system.py delegates to softmax_grads.compute_softmax_weights() | ✓ Not reviewed (external) |
| **Σ → g Σ g^T under transport** | agents.py enforces this with gauge fields, masking.py checks it | ✓ Gauge covariance |
| **χ_ij = χ_i · χ_j** | system.py line 233 computes this correctly | ✓ Overlap product |
| **F = α·KL(q\|\|p) + λ_β·Σ β_ij·KL(...) + CE(W·μ, y)** | Delegated to free_energy_clean.compute_total_free_energy() | ✓ Not reviewed (external) |

---

## Testing Recommendations

1. **Test Covariance Cache**: Verify L_q = Cholesky(Sigma_q) after every parameter update
2. **Test Agent Update Methods**: Implement agents.py.Agent.update_beliefs/priors/gauge() or fix system.py
3. **Test SPD Reconstruction**: Verify Sigma_mat in hamiltonian_trainer.py._unpack_parameters() is positive-definite
4. **Test Overlap Continuity**: Verify χ_ij field is properly smooth at agent boundaries
5. **Test Observation Loss**: Verify observation bias is actually incorporated into gradients
6. **Test Energy Conservation**: Compare H(0) and H(T) in Hamiltonian trainer with zero friction

---

## Conclusion

Overall code quality is **good-to-excellent** with strong mathematical foundations. Primary issues are:

1. **Missing Agent update methods** (system.py) — will cause runtime crash
2. **Covariance cache inconsistency** (agents.py) — silent failures possible
3. **Observation bias not in loss** (system.py) — optimization will ignore bias
4. **SPD reconstruction unvalidated** (hamiltonian_trainer.py) — can cause ill-conditioning

Recommend prioritizing fixes 1-3 before running any training experiments.
