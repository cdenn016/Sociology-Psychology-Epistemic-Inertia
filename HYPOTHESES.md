# Hypothesis Generation: Epistemic Inertia Framework

**Generated from:** "The Inertia of Belief" manuscript
**Date:** 2026-03-15
**Method:** Systematic hypothesis derivation following `.claude/skills/hypothesis-generation/SKILL.md`

---

## Core Theoretical Claims

1. **Precision-as-Mass:** The Fisher information metric acts as an inertial mass tensor for belief dynamics; epistemic mass M = Lambda_prior + Lambda_observation + Lambda_social_in + Lambda_social_out
2. **Hamiltonian Dynamics:** Belief evolution follows second-order (Hamiltonian) dynamics with damping, not first-order gradient descent
3. **Sociological Unification:** DeGroot, Friedkin-Johnsen, bounded confidence, and echo chamber models emerge as limiting cases of VFE minimization
4. **Cognitive Bias Emergence:** Confirmation bias, belief perseverance, and continued influence effect are geometric consequences of epistemic inertia
5. **Social Attention Rigidity:** Influence directed toward an agent accumulates as inertial mass, making influential agents more rigid

---

## Category 1: Precision-Mass Identification

### H1.1: Overshoot Scales with Square Root of Precision

**Derived from:** Claim 1 (Precision-as-Mass)
**Prediction:** When agents encounter sudden evidence shifts, the overshoot distance scales as d_overshoot = |dot_mu| * sqrt(M/K). The ratio of overshoot for high-precision vs. low-precision agents scales as sqrt(Lambda_H / Lambda_L).
**Null hypothesis:** Overshoot is independent of prior precision, or scales linearly rather than as square root.
**Operationalization:** Measure participant confidence (as proxy for precision) on a topic via incentivized elicitation. Present strong counter-evidence. Track belief trajectory at 1-minute intervals for 20 minutes. Measure maximum deviation past the normative posterior.
**Required data/compute:** N >= 200 participants, pre-registered confidence bins (low/medium/high), repeated belief measurements
**Priority:** High -- directly tests the core mass identification
**Risk:** Overshoot could be confounded with anchoring-and-adjustment (Tversky & Kahneman) or reactance effects

### H1.2: Relaxation Time Scales Linearly with Precision

**Derived from:** Claim 1
**Prediction:** The characteristic time tau = M/gamma for a belief to decay to equilibrium after evidence exposure scales linearly with prior precision Lambda. Agents with twice the precision require twice the relaxation time, independent of evidence direction/magnitude.
**Null hypothesis:** Relaxation time depends primarily on evidence strength, not prior confidence. Or relaxation time scales sub-linearly/super-linearly with precision.
**Operationalization:** Longitudinal belief tracking (days to weeks) following a debunking intervention. Precision measured via betting tasks or confidence intervals. Fit exponential decay to belief trajectories and extract tau per participant.
**Required data/compute:** N >= 150, longitudinal tracking over 2-4 weeks, pre-registered precision measurement protocol
**Priority:** High -- clean, testable prediction distinguishing from standard Bayesian models
**Risk:** Confounded by memory effects, social reinforcement, or motivated reasoning independent of precision

### H1.3: Sensory Anchoring Effect

**Derived from:** Claim 1 (Observation precision term Lambda_o)
**Prediction:** Experts with high-fidelity instruments/data (high Lambda_o) exhibit greater belief inertia than novices with noisy observations, even when both hold equally strong prior beliefs (matched Lambda_p). The sensory precision contribution to mass is independent of and additive with prior precision.
**Null hypothesis:** Observation quality does not independently contribute to belief inertia once prior confidence is controlled.
**Operationalization:** Compare belief updating between: (a) experts with reliable data + strong priors, (b) experts with reliable data + weak priors, (c) novices with noisy data + strong priors, (d) novices with noisy data + weak priors. A 2x2 design separating Lambda_o from Lambda_p.
**Required data/compute:** N >= 60 per cell (240 total), domain-appropriate expertise/novice distinction
**Priority:** Medium-High -- tests additivity of mass contributions
**Risk:** Expertise confounds precision with motivation, identity, or domain knowledge

---

## Category 2: Second-Order Dynamics (Hamiltonian Ansatz)

### H2.1: Existence of Belief Oscillation

**Derived from:** Claim 2 (Hamiltonian dynamics)
**Prediction:** High-confidence agents confronting strong, credible counter-evidence exhibit non-monotonic belief trajectories: initial resistance, overshoot past normative posterior, followed by oscillation before settling.
**Null hypothesis:** All belief trajectories are monotonic (consistent with first-order gradient descent).
**Operationalization:** Dense temporal sampling (every 30-60 seconds over 20-30 minutes) of beliefs following counter-evidence exposure. Pre-screen for high-confidence participants on politically/morally charged topics. Use incentive-compatible elicitation.
**Required data/compute:** N >= 100 high-confidence participants, high-frequency belief sampling, pre-registered oscillation detection criterion
**Priority:** High -- the signature prediction of second-order dynamics
**Risk:** Non-monotonicity could arise from task confusion, demand effects, or recency bias in repeated measurements. Need careful controls.

### H2.2: Resonant Persuasion Frequency

**Derived from:** Claim 2
**Prediction:** Periodic persuasive messages achieve maximum belief change at resonance frequency omega_res = sqrt(K/M). This predicts a non-monotonic relationship between message timing interval and final belief change amplitude, with a peak at an intermediate frequency that depends on individual precision.
**Null hypothesis:** Belief change increases monotonically with message frequency (more exposure = more change) or is independent of timing.
**Operationalization:** Deliver persuasive messages at varied intervals (30s, 2min, 5min, 10min, 20min) holding total message content constant. Measure final belief change amplitude across conditions. Within each condition, test whether the resonant interval correlates with individual precision.
**Required data/compute:** N >= 50 per timing condition (250+ total), pre-registered timing intervals, individual precision measurement
**Priority:** High -- unique signature of second-order dynamics, no plausible first-order alternative
**Risk:** Attention fatigue, habituation, and memory decay confound timing effects. Need matched attention controls.

### H2.3: Critical Damping as Optimal Learning

**Derived from:** Claim 2 (Three dynamical regimes)
**Prediction:** The fastest belief equilibration occurs at critical damping gamma = 2*sqrt(K*M). Individuals or interventions operating near this regime should show fastest convergence to stable posteriors. Overdamped agents converge slowly; underdamped agents overshoot and oscillate.
**Null hypothesis:** Speed of belief convergence is independent of the gamma/sqrt(KM) ratio.
**Operationalization:** Manipulate damping (e.g., via cognitive load, time pressure, or pharmacological intervention affecting processing speed). Measure time-to-stable-posterior across damping conditions.
**Required data/compute:** Complex experimental design, potentially N >= 200 with pharmacological or cognitive load manipulation
**Priority:** Medium -- interesting but hard to manipulate damping independently
**Risk:** Damping is not directly manipulable; proxies (cognitive load, stress) may not map cleanly to gamma

---

## Category 3: Social Dynamics

### H3.1: Influence Accumulates as Rigidity (The Adams-Solzhenitsyn Effect)

**Derived from:** Claim 5 (Outgoing social inertia)
**Prediction:** Agents who receive sustained social attention (many followers attending to their beliefs) become more resistant to belief revision than equally confident but unattended individuals. The effect is proportional to sum_j beta_ji * Lambda_qi.
**Null hypothesis:** Belief rigidity depends only on individual confidence (Lambda_p + Lambda_o), not on how many others attend to the agent's beliefs.
**Operationalization:** Group deliberation paradigm. Randomly assign "influencer" vs. "observer" roles. Both groups hold equally strong initial beliefs. After deliberation, present identical counter-evidence to both groups. Measure belief change magnitude and relaxation dynamics.
**Required data/compute:** N >= 100 (50 influencers, 50 observers), matched on initial confidence
**Priority:** High -- the most sociologically important prediction; directly testable
**Risk:** Role assignment may induce identity/commitment effects beyond the attention mechanism. Need yoked controls.

### H3.2: Echo Chamber Formation Threshold

**Derived from:** Claim 3 (Echo chambers as VFE limit)
**Prediction:** Groups with belief separation ||mu_A - mu_B||^2 > 2*sigma^2*kappa*log(N) maintain stable polarization; below this threshold, consensus emerges. The critical temperature kappa_crit scales with initial separation squared divided by sigma^2*log(N).
**Null hypothesis:** Polarization is determined by network structure or initial opinion distribution, not by a threshold depending on uncertainty and attention temperature.
**Operationalization:** Agent-based simulation with N agents and varied initial separation, uncertainty sigma, and attention temperature kappa. Measure whether polarization persists as a function of the derived threshold variable.
**Required data/compute:** Computational (agent-based simulation), low resource requirement
**Priority:** Medium-High -- testable computationally before empirical validation
**Risk:** Real social dynamics have many confounds absent from the model; threshold may be washed out

### H3.3: Momentum Transfer Asymmetry

**Derived from:** Claim 2 (Epistemic momentum transfer)
**Prediction:** High-precision agents transfer more epistemic momentum to coupled neighbors than low-precision agents. In a dyadic interaction, the high-precision agent moves the low-precision agent's belief more than vice versa, proportional to the precision ratio.
**Null hypothesis:** Influence is symmetric regardless of precision, or depends only on network position.
**Operationalization:** Dyadic belief exchange paradigm. Pair participants with matched vs. mismatched precision (operationalized via expertise or confidence). Measure belief change in both partners after structured exchange.
**Required data/compute:** N >= 100 dyads (200 participants), pre-measured precision
**Priority:** Medium -- extends existing persuasion literature with quantitative predictions
**Risk:** Confounded by communication skill, persuasion ability, or status effects

### H3.4: Stubbornness Is Context-Dependent, Not a Trait

**Derived from:** Claim 3 (Friedkin-Johnsen derivation)
**Prediction:** The same individual exhibits different degrees of stubbornness alpha' = (alpha/Sigma_p) / (alpha/Sigma_p + lambda_beta/sigma^2) across different social contexts (varying lambda_beta), contradicting trait-based theories of resistance to influence.
**Null hypothesis:** Stubbornness is a stable individual difference (trait) that does not vary with social context.
**Operationalization:** Within-subjects design measuring the same individuals' resistance to social influence across domains where they have varying prior precision. Test whether stubbornness co-varies with precision and social coupling, not just with a stable individual trait score.
**Required data/compute:** N >= 100, within-subjects across 3+ domains, personality measures for comparison
**Priority:** Medium -- strong theoretical prediction but complex experimental design
**Risk:** Within-subject variation may be attributed to topic engagement or motivation rather than precision

---

## Category 4: Cognitive Bias Reinterpretation

### H4.1: Belief Perseverance Scales with Precision, Not Content

**Derived from:** Claim 4
**Prediction:** The duration of belief perseverance after debunking scales with prior precision (tau = M/gamma), not with the content or emotional valence of the belief. Controlling for precision, emotionally charged and neutral beliefs should persist for equal durations.
**Null hypothesis:** Belief perseverance depends on emotional valence, identity-relevance, or content, independent of precision.
**Operationalization:** Debunking paradigm across topics varying in emotional valence but matched for prior precision (measured via betting tasks). Track post-debunking belief trajectories and measure perseverance duration.
**Required data/compute:** N >= 200, topics matched on precision but varying in valence
**Priority:** High -- directly distinguishes geometric from motivational accounts of perseverance
**Risk:** Precision and emotional valence may be inherently correlated (we hold strong beliefs about things we care about)

### H4.2: Confirmation Bias Is Mass-Proportional

**Derived from:** Claim 4 (Confirmation bias as geometric effect)
**Prediction:** The asymmetry in response to confirming vs. disconfirming evidence scales with epistemic mass: ||d_mu_confirm|| / ||d_mu_disconfirm|| increases with M. Low-mass agents show symmetric updating; high-mass agents show strong asymmetry.
**Null hypothesis:** Confirmation bias is independent of confidence/precision, or is uniform across confidence levels.
**Operationalization:** Present matched confirming and disconfirming evidence to participants varying in prior confidence. Measure belief update magnitudes in both directions.
**Required data/compute:** N >= 200, within-subjects evidence direction manipulation
**Priority:** Medium-High -- connects to large existing literature on confirmation bias
**Risk:** Confirmation bias has multiple proposed mechanisms; hard to isolate the geometric contribution

### H4.3: Continued Influence as Momentum Decay

**Derived from:** Claim 4
**Prediction:** The continued influence of misinformation after correction follows an exponential decay with time constant tau = M/gamma, not a step function or power law. The decay rate is predicted by pre-correction precision.
**Null hypothesis:** Continued influence follows a different functional form (power law, step function) or is independent of initial encoding precision.
**Operationalization:** Misinformation paradigm with correction. Measure continued influence at multiple post-correction time points. Fit exponential vs. power law vs. step function decay models. Test whether decay rate correlates with pre-correction belief precision.
**Required data/compute:** N >= 150, longitudinal (multiple post-correction measurements), model comparison
**Priority:** Medium -- connects to active misinformation research
**Risk:** Memory decay processes may dominate over inertial dynamics at long timescales

---

## Category 5: Extensions and Scaling

### H5.1: Bounded Confidence Threshold Predicts Cluster Number

**Derived from:** Claim 3 (Bounded confidence as low-kappa limit)
**Prediction:** The effective confidence bound epsilon_eff = sigma*sqrt(2*kappa*log(N)) predicts the number of opinion clusters as approximately (opinion range) / epsilon_eff. This gives a quantitative relationship between attention temperature, uncertainty, population size, and fragmentation.
**Null hypothesis:** Number of opinion clusters depends on initial conditions, not on the derived threshold.
**Operationalization:** Agent-based simulation varying kappa, sigma, and N systematically. Count equilibrium clusters and compare to theoretical prediction.
**Required data/compute:** Computational only, low resource
**Priority:** Medium -- straightforward computational validation
**Risk:** Discrete agent dynamics may deviate from continuous-limit predictions for small N

### H5.2: Exponential Family Generalization

**Derived from:** Appendix E (Exponential family extension)
**Prediction:** The mass-as-Fisher-information identification M = nabla^2 A(theta) holds for non-Gaussian exponential families. Specifically, for Beta-distributed beliefs (natural for probability estimation), mass should equal the concentration parameter.
**Null hypothesis:** The Gaussian-specific results do not generalize; alternative inertia measures are needed for non-Gaussian beliefs.
**Operationalization:** Extend the simulation framework to Beta-distributed beliefs. Verify that inertial dynamics with Fisher-information mass produce the predicted oscillation/overshoot phenomena.
**Required data/compute:** Computational, moderate (new simulation code needed)
**Priority:** Medium -- important for theoretical generality
**Risk:** Non-Gaussian distributions may introduce qualitatively different dynamics not captured by the quadratic approximation

### H5.3: Hierarchical Emergence (Renormalization)

**Derived from:** Discussion Section 5.4 (Hierarchical extensions)
**Prediction:** When groups of agents achieve sufficient consensus (low within-group KL), they can be treated as a single meta-agent with emergent mass equal to the sum of individual masses plus inter-agent coupling terms. This renormalization is self-similar: meta-agents obey the same dynamics.
**Null hypothesis:** Group-level dynamics are qualitatively different from individual dynamics; coarse-graining introduces emergent properties not predicted by the individual framework.
**Operationalization:** Multi-scale agent-based simulation. Start with N individual agents, observe clustering, coarse-grain to meta-agents, compare meta-agent dynamics to theoretical prediction.
**Required data/compute:** Computational, moderate
**Priority:** Medium-High -- connects to the VFE-Transformer renormalization program
**Risk:** Coarse-graining may require additional terms or corrections not present in the current framework

---

## Prioritization Matrix

| Hypothesis | Impact | Feasibility | Novelty | Priority Score |
|-----------|--------|-------------|---------|----------------|
| H2.1 Belief oscillation | Very High | Medium | Very High | 5/5 |
| H3.1 Influence rigidity | Very High | Medium | Very High | 5/5 |
| H4.1 Perseverance = precision | High | Medium | High | 5/5 |
| H2.2 Resonant persuasion | Very High | Hard | Very High | 4.5/5 |
| H1.1 Overshoot sqrt scaling | High | Medium | High | 4.5/5 |
| H1.2 Relaxation time linear | High | Medium | High | 4.5/5 |
| H3.2 Echo chamber threshold | High | Easy (computational) | Medium | 4/5 |
| H4.2 Confirmation bias scaling | High | Medium | Medium | 4/5 |
| H3.4 Context-dependent stubbornness | High | Medium | High | 4/5 |
| H1.3 Sensory anchoring | Medium | Hard | Medium | 3.5/5 |
| H5.1 Bounded confidence clusters | Medium | Easy | Low | 3/5 |
| H3.3 Momentum transfer asymmetry | Medium | Medium | Medium | 3/5 |
| H4.3 Continued influence decay | Medium | Medium | Medium | 3/5 |
| H2.3 Critical damping optimal | Medium | Hard | Medium | 3/5 |
| H5.2 Exponential family | Medium | Medium | Low | 2.5/5 |
| H5.3 Hierarchical emergence | High | Hard | High | 4/5 |

---

## Recommended Experimental Program

### Phase 1: Computational Validation (Weeks 1-4)
- H3.2: Echo chamber threshold simulation
- H5.1: Bounded confidence cluster prediction
- H5.3: Hierarchical emergence simulation
- **Goal:** Verify analytical predictions computationally before committing to empirical work

### Phase 2: Core Empirical Tests (Months 2-6)
- H2.1: Belief oscillation (the flagship experiment)
- H1.2: Relaxation time scaling
- H3.1: Influence rigidity (the most sociologically impactful)
- **Goal:** Test the signature predictions that distinguish from first-order models

### Phase 3: Mechanism Dissection (Months 6-12)
- H4.1: Perseverance as precision
- H4.2: Confirmation bias scaling
- H1.1: Overshoot sqrt scaling
- **Goal:** Test the cognitive bias reinterpretation and quantitative scaling predictions

### Phase 4: Applications and Extensions (Year 2+)
- H2.2: Resonant persuasion (applied persuasion design)
- H3.4: Context-dependent stubbornness (challenges trait psychology)
- H5.2: Exponential family generalization
- **Goal:** Extend to applications and generalize the theoretical framework

---

## Integration with Other Skills

| Skill | Role in Hypothesis Testing |
|-------|---------------------------|
| **statistical-analysis** | Power analysis, effect size estimation, model comparison for all empirical hypotheses |
| **pymc** | Bayesian parameter estimation for tau, gamma, M from belief trajectory data |
| **networkx** | Graph-theoretic analysis of echo chamber and momentum transfer simulations |
| **umap-learn** | Visualization of belief trajectories and clustering in high-dimensional opinion space |
| **shap** | Feature importance for which mass components dominate in simulated dynamics |
| **scientific-writing** | Write up results for manuscript extension |
| **scientific-visualization** | Publication-quality figures for phase portraits, oscillation trajectories, polarization maps |
| **sympy** | Symbolic verification of Hessian derivations and exponential family extensions |
