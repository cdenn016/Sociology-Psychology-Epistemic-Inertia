# Peer Review: "The Inertia of Belief"

**Manuscript:** "The Inertia of Belief" by Robert C. Dennis
**Review Date:** 2026-03-15
**Reviewer:** Claude (AI-assisted structured review)
**Manuscript Type:** Original research article (theoretical/mathematical)

---

## Summary Statement

This manuscript presents a unified geometric framework grounding belief dynamics in information geometry, proposing that the Fisher information metric serves as an inertial mass tensor for epistemic dynamics. The central contribution is a Hamiltonian formulation of belief updating where precision equals inertial mass, yielding second-order (oscillatory, overshoot, resonant) dynamics absent from standard first-order Bayesian treatments. The paper demonstrates that several foundational sociological models (DeGroot, Friedkin-Johnsen, bounded confidence, echo chambers, Social Impact Theory) emerge as limiting cases of variational free energy minimization on statistical manifolds. The framework reinterprets cognitive biases (confirmation bias, belief perseverance, continued influence effect) as geometric consequences of epistemic inertia.

**Overall Recommendation:** Minor revisions

**Key Strengths:**
- Ambitious and genuinely novel unification of information geometry, opinion dynamics, and cognitive bias research under a single mathematical framework
- The derivations of classical sociological models as limiting cases are rigorous and clearly presented, with honest assessments of derivation quality (Table 5)
- The transparent distinction between geometrically necessary results (overdamped limits) and ansatz-dependent predictions (underdamped dynamics) is exemplary scientific practice
- The "outgoing social inertia" prediction (influence accumulates as rigidity) is a striking, testable, and sociologically important result

**Key Weaknesses:**
- The Hamiltonian ansatz, while well-motivated, remains empirically unvalidated; no simulation results or empirical data are presented
- The damping coefficient gamma's interpretation as inverse learning rate, while elegant, conflates a physical dissipation parameter with an optimization hyperparameter -- this identification needs stronger justification
- Several key predictions (oscillation, resonance) may be difficult to distinguish from alternative explanations (e.g., recency effects, anchoring-adjustment dynamics) without carefully designed experiments
- The manuscript's scope is very broad, spanning physics, information geometry, sociology, and psychology, which risks superficial treatment in each domain

---

## Major Comments

### M1. The Hamiltonian Ansatz Needs Stronger Motivation or Weakening of Claims

The manuscript forthrightly acknowledges (Section 2.6, boxed caveat) that the Hamiltonian formulation is an ansatz. However, the Results section (Section 3) then derives extensive predictions from this ansatz without clearly separating which results depend on it and which follow from geometry alone. While Table 5 addresses this for the classical model limits, the cognitive phenomena in Section 3 (overshoot, oscillation, resonance, perseverance) all depend on the ansatz.

**Recommendation:** Add a clear statement at the beginning of Section 3 reminding the reader that all predictions in that section are contingent on the Hamiltonian ansatz. Consider restructuring so that geometry-only results (Section 4: classical limits, confirmation bias) come before ansatz-dependent results (oscillation, resonance, overshoot). This would strengthen the paper by leading with the most rigorous contributions.

### M2. The Identification of gamma with Inverse Learning Rate Is Problematic

Section 3.3.1 identifies the damping coefficient gamma as the inverse learning rate from standard variational inference. While this provides an appealing connection, the identification is only valid in the overdamped limit. In the underdamped regime -- where the novel predictions live -- gamma no longer corresponds to any standard variational inference quantity. The manuscript needs to address what determines gamma in the underdamped regime and whether it is measurable independently of the phenomena it is meant to predict.

**Recommendation:** Discuss what gamma corresponds to physically/cognitively in the underdamped regime. Is it metabolic cost? Attention switching cost? Without an independent characterization, the framework risks circularity: gamma is chosen to produce the observed dynamics, which are then "predicted" by the framework.

### M3. Missing Simulation Results

The manuscript references phase portrait figures (Figure 1a, 1b) but presents no simulation results demonstrating the multi-agent dynamics, the emergence of classical models from the VFE framework, or the predicted oscillation/resonance phenomena in agent networks. Given that the repository contains a simulation suite, including even basic numerical demonstrations would substantially strengthen the paper.

**Recommendation:** Add at minimum: (1) a simulation showing DeGroot convergence emerging from the VFE framework in the overdamped limit, (2) a simulation demonstrating belief oscillation in the underdamped regime, and (3) a multi-agent simulation showing echo chamber formation with the predicted polarization threshold. These would verify the analytical predictions and demonstrate the framework's computational tractability.

### M4. The Observation Precision Contribution to Mass Needs More Careful Treatment

Section 2.5.3 derives observation precision Lambda_o as contributing to inertial mass. The interpretation in Section 2.5.5 ("sensory anchoring") argues that precise observations anchor beliefs. However, this conflates two distinct effects: (1) precise observations provide strong evidence for the *current* belief (anchoring), and (2) precise observations will also provide strong evidence for any *new* belief consistent with observations. The mass interpretation captures only (1), but a complete treatment should address how observation precision interacts with evidence for belief *change* vs. belief *maintenance*.

**Recommendation:** Clarify whether observation precision contributes to mass only when observations are consistent with the current belief, or unconditionally. Consider whether Lambda_o should be modulated by the agreement between current observations and current beliefs.

### M5. Attention Weights and Mass Are Circular

The attention weights beta_ij (Eq. 6) depend on KL divergences between agent beliefs, which in turn depend on agent states. But the mass matrix M_i also depends on beta_ij (through the social precision terms). This creates a circularity: mass determines dynamics, dynamics change beliefs, beliefs change attention, attention changes mass. While this is not necessarily a problem (it is simply a coupled dynamical system), the manuscript should explicitly acknowledge this coupling and discuss whether the system is well-posed (existence and uniqueness of solutions).

**Recommendation:** Add a brief discussion of well-posedness. Under what conditions do the coupled attention-mass-belief dynamics have unique solutions? Are there pathological cases (e.g., attention singularities when beliefs collapse)?

---

## Minor Comments

### m1. Abstract Length and Focus
The abstract is dense and could be tightened. The second paragraph introduces the Hamiltonian ansatz, proper time, and oscillation predictions alongside cognitive bias reinterpretation. Consider splitting into: (1) the geometric unification result (which is rigorous), and (2) the Hamiltonian/inertia framework (which is an ansatz with predictions).

### m2. Notation Consistency (Section 2.1)
The subscript nu in "mu_nu" (line 105) is introduced without definition. It appears to be a generic index for either q or p, but this should be stated explicitly.

### m3. Missing Definition of kappa vs. tau
The attention temperature appears as both kappa (Eq. 6, Section 2.4) and tau (Section 2.3). While they appear to serve different roles (kappa for attention selectivity, tau for proper time), the overlap in notation could confuse readers. Consider using a distinct symbol for proper time, such as s.

### m4. The "Proper Time" Section (2.3) Is Tangential
While intellectually interesting, the proper time construction in Section 2.3 is not used in any subsequent derivation. The Hamiltonian dynamics use coordinate time t, not proper time tau. Consider moving this to an appendix or clearly connecting it to the dynamical framework.

### m5. Claim About "Unique" Riemannian Metric (Section 2.4)
The manuscript states the Fisher metric is "the unique Riemannian metric on probability spaces" (line 289). This should be qualified: it is unique up to scaling *under the requirement of invariance under sufficient statistics* (Cencov's theorem). The theorem also applies only to finite sample spaces without additional assumptions. Add the appropriate qualification and citation to Cencov (1982).

### m6. DeGroot Derivation Step 3 (Section 4.1.3)
The gradient flow in Step 3 uses M_i^{-1} as the metric (natural gradient), but Step 1 assumes "overdamped dynamics." The overdamped limit of the Hamiltonian gives gradient descent gamma * d_mu/dt = -nabla F, not natural gradient descent M^{-1} nabla F. These are different unless gamma propto M, which is not assumed. Clarify this distinction.

### m7. Echo Chamber Stability Condition (Section 4.4.3)
The stability condition (Eq. after line 805) uses an approximation "(N/2) + (N/2)x" in the denominator, but the exact expression from Step 2 is "(N/2 - 1) + (N/2)x". For large N these agree, but the approximation should be noted explicitly.

### m8. Social Impact Theory Section Is Weak
The SIT correspondence (Section 4.6) is presented as "interpretive" rather than formal, and the manuscript acknowledges this. However, the section might benefit from being framed more carefully as a "conceptual mapping" rather than a "derivation," or moved to the Discussion section. Its placement alongside rigorous derivations (DeGroot, Friedkin-Johnsen) may mislead readers about the quality of the correspondence.

### m9. Missing Discussion of Related Work on Momentum in Optimization
The machine learning literature has extensively studied momentum methods (Polyak momentum, Nesterov acceleration, Adam optimizer) as second-order extensions of gradient descent. These share structural similarity with the proposed Hamiltonian dynamics. A brief discussion of connections and differences would strengthen the paper's positioning.

### m10. Typographical Issues
- Line 86: "fink2002" citation format should use \citep not \citet (it's parenthetical)
- Line 369: Missing space before "This leads to..."
- Line 1142: LaTeX line break "\\\\" before "Kaplowitz-Fink" creates awkward formatting
- Line 2004: "agents" should be "agent's" (possessive)
- Section 3.3.1: The equation of motion (line 409) is presented *after* being discussed in prose; consider reordering for clarity

### m11. References
- The bibliography should include Cencov (1982) for the uniqueness claim about the Fisher metric
- Consider citing Su, Boyd, and Candes (2016) on differential equation interpretations of optimization algorithms, which is structurally related
- The connection to Polyak's heavy ball method (1964) should be noted, as the damped oscillator equation is identical in form

---

## Questions for the Authors

1. **Falsifiability:** What specific experimental result would *falsify* the Hamiltonian ansatz (as opposed to merely constraining parameter values)? The framework appears to encompass both overdamped (standard Bayesian) and underdamped (oscillatory) regimes, raising the question of whether any data could be inconsistent with it.

2. **Precision dynamics:** The quasi-static precision assumption treats Lambda as slowly varying. But in learning, precision *increases* as evidence accumulates. How does the framework handle the fact that mass increases over time, potentially making the system *more* underdamped as learning progresses?

3. **Non-quadratic free energy:** The oscillation predictions rely on a quadratic approximation to the free energy around equilibrium. How robust are these predictions to the true (non-quadratic) free energy landscape, particularly for large belief displacements where the quadratic approximation breaks down?

4. **Gauge structure empirical content:** The paper acknowledges that all core results hold in the flat gauge (Omega_ij = I). Can you provide a concrete empirical scenario where the non-trivial gauge structure makes a different prediction than the flat case? Without this, the gauge machinery may be a formal luxury rather than a substantive contribution.

5. **Damping heterogeneity:** The framework assumes each agent has a damping coefficient gamma_i. In a social network, what determines inter-agent variation in gamma? If gamma is the inverse learning rate, different agents learning at different rates would produce heterogeneous damping. Has this been explored?

---

## Methodological and Statistical Rigor

This is a purely theoretical manuscript, so standard experimental methodology criteria (sample sizes, controls, randomization) do not directly apply. However:

- **Mathematical rigor:** The proofs and derivations are generally clear and correct. The DeGroot and Friedkin-Johnsen derivations are exact. The bounded confidence derivation is approximate (acknowledged). The echo chamber derivation makes reasonable large-N approximations.
- **Reproducibility:** The manuscript states code will be made available on GitHub. The analytical results are fully specified and verifiable.
- **Transparency:** The explicit ansatz caveat and the rigor assessment table (Table 5) are commendable. The distinction between rigorous and interpretive results is consistently maintained.

---

## Figure and Data Presentation

- **Figure 1 (Phase portraits):** The figures are referenced but their quality and content cannot be assessed from the LaTeX source alone. Ensure axes are labeled with units, trajectories are distinguishable, and the parameter values used are stated in the caption.
- **Tables:** Tables 1-5 are well-structured and informative. Table 5 (rigor assessment) is particularly valuable for reader guidance.

---

## Final Assessment

This is an ambitious, intellectually stimulating manuscript that proposes a genuinely novel theoretical framework. The core contribution -- deriving classical sociological models from variational free energy minimization -- is rigorous and valuable independent of the Hamiltonian ansatz. The inertial interpretation, while speculative, generates interesting and testable predictions. The main weaknesses are the absence of computational validation and the need for clearer separation between rigorous and speculative results. With the recommended revisions (primarily restructuring and adding simulation results), this manuscript would make a strong contribution to the intersection of information geometry, opinion dynamics, and cognitive science.

The manuscript would be suitable for journals such as the *Journal of Mathematical Sociology*, *Journal of Mathematical Psychology*, *Entropy*, or *Physical Review E* (social/complex systems section).
