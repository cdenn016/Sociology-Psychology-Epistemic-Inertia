# Line-by-Line Corrections for belief_inertia_unified.tex

## Immediate Fixes (Copy-Edit Level)

### Line 11
**Current:** ` \usepackage{subcaption}`
**Corrected:** `\usepackage{subcaption}`
(Remove leading space)

### Lines 45-48 (Abstract)
**Issue:** Text runs together awkwardly across lines
**Current:**
```latex
plays the role of an inertial mass tensor whereby confident beliefs resist change while uncertain beliefs update readily. This is not physics by analogy but
information geometry: the Fisher metric is the unique Riemannian metric on probability spaces, and second-order dynamics follow from its curvature.
```
**Suggestion:** Add line break or reflow for readability.

### Line 51 (Abstract)
**Issue:** Claims derivation of "diffusion of innovations" which doesn't appear in paper
**Current:**
```latex
and diffusion of innovations.
```
**Corrected:** Remove this phrase, or add derivation in Section 4

### Line 77 (Introduction)
**Same issue:** Remove "diffusion of innovations" reference unless derivation added

### Line 362
**Current:** `...trajectory.This leads to...`
**Corrected:** `...trajectory. This leads to...`
(Add space after period)

### Line 589
**Current:** `...evidence alone would have lead them.`
**Corrected:** `...evidence alone would have led them.`
(Past participle)

### Lines 1137-1138
**Issue:** Line break in author name creates bad formatting
**Current:**
```latex
The \\ Kaplowitz-Fink results
```
**Corrected:**
```latex
The Kaplowitz-Fink results
```
(Remove manual line break; let LaTeX handle it)

---

## Mathematical Notation Fixes

### Lines 142, 210-211, 234 (Observation Model)

**Issue:** Inconsistent notation for observation model. Uses both `μ_i` and `c_i` as conditioning variable.

**Line 142 currently:**
```latex
- \sum_i \underbrace{\mathbb{E}_{q_i}[\log p(o_i \mid \mu_i)]}_{\text{Sensory evidence}}
```

**Line 210-211 currently:**
```latex
- \underbrace{\sum_i \mathbb{E}_{q_i}[\log p(o_i|c_i)]}_{\text{accuracy}}
```

**Line 234 currently:**
```latex
For a Gaussian observation model $p(o_i|\mu_i) = \mathcal{N}(o_i \,|\, c_i, R_i)$
```

**Recommendation:** Choose one notation. If `c_i` is a hidden cause and `μ_i` is the belief about it, make this explicit:
```latex
For a Gaussian observation model $p(o_i|\theta) = \mathcal{N}(o_i; \theta, R_i)$ where $\theta$ is the hidden state and $\mu_i = \mathbb{E}_{q_i}[\theta]$:
```

Or simply use `μ_i` throughout if the belief mean directly predicts observations.

---

## Structural Issues

### Lines 280-287 (Important Caveat Box)

**Recommendation:** Move this BEFORE Section 3 (Results), not after the mass derivation. Current placement means readers encounter predictions for 8 pages before learning the Hamiltonian is an ansatz.

**Suggested location:** After line 197 (end of Section 2.5 intro paragraph), before Section 2.6

### Lines 399-400 (Damping Coefficient)

**Current:**
```latex
The damping coefficient $\gamma$ is not a new free parameter but inherits directly from standard variational inference: it is the inverse learning rate $\gamma = \eta^{-1}$ of gradient descent on the free energy.
```

**Suggested addition after this:**
```latex
To see this, note that in the overdamped limit where $M\ddot{\mu} \ll \gamma\dot{\mu}$, the equation of motion $M\ddot{\mu} + \gamma\dot{\mu} + \nabla F = 0$ reduces to $\gamma\dot{\mu} \approx -\nabla F$, yielding $\dot{\mu} = -\gamma^{-1}\nabla F = -\eta\nabla F$. This recovers standard gradient descent with learning rate $\eta = \gamma^{-1}$. The underdamped regime ($\gamma < 2\sqrt{KM}$) introduces genuinely new dynamics not present in first-order treatments.
```

---

## References.bib Duplicates to Remove

The following entries appear multiple times and should be deduplicated:

1. `Clark2013` - appears at lines 466-475 and 582-589
2. `Rovelli1996` - appears at lines 424-433, 800-807, and 1532-1541
3. `Jacobson1995` - appears at lines 413-422 and 682-689
4. `Verlinde2011` - appears at lines 402-411 and 700-707
5. `Hoffman2019` - appears at lines 478-485 and 1681-1687
6. `Arndt2014` - appears at lines 391-400 and 1800-1808
7. `Aspelmeyer2014` - appears at lines 380-389 and 1831-1839
8. `Wheeler1990` - appears at lines 309-316 and 446-455
9. `Fuchs2014` and `Fuchs2013` - similar content
10. `Chentsov1982` and `Cencov1982` - same work, different transliteration
11. `Ladyman2014` - appears twice (lines 1477-1487 and 1495-1505)
12. `lewandowsky2012` and `Lewandowsky2012` - same work, different case

**Recommendation:** Keep one version of each, using consistent capitalization (typically Title Case for BibTeX keys).

---

## Content Additions Needed

### Section 4: Add Diffusion of Innovations (if keeping claim)

If the claim is to be retained, add a subsection like:

```latex
\subsection{Diffusion of Innovations}

\subsubsection{Classical Formulation}

Rogers' diffusion model and the Bass model describe adoption dynamics through a population:
\begin{equation}
\frac{dN(t)}{dt} = [p + q\frac{N(t)}{M}][M - N(t)]
\end{equation}
where $N(t)$ is cumulative adopters, $M$ is market potential, $p$ is innovation coefficient, and $q$ is imitation coefficient.

\subsubsection{Derivation from VFE Framework}

[Show how binary adoption belief $q_i \in \{0, 1\}$ or continuous adoption propensity emerges from the social free energy with appropriate threshold dynamics]
```

**Alternative:** Remove the claim from abstract and introduction.

---

## Suggested Additions for Clarity

### After Equation (11) (Mass Formula)

Add clarifying remark:
```latex
We emphasize that this ``Hessian mass matrix'' depends on the full generative model---priors, observations, and social structure---and is distinct from the intrinsic Fisher-Rao metric, which depends only on the parametric family of the belief distribution itself. The identification of precision with inertia is thus not purely geometric but reflects the agent's embedding in their informational environment.
```

### Line 954-956 (SIT Caveat)

**Current:**
```latex
\subsubsection{Caveat}

This is an interpretive correspondence, not a formal derivation.
```

**Suggested enhancement:**
```latex
\subsubsection{Caveat}

This is an interpretive correspondence, not a formal derivation. Latané's original formulation involved physical proximity and temporal immediacy, whereas our $\beta_{ij}$ captures epistemic proximity. The correspondence suggests structural similarity between physical and epistemic influence, but a complete derivation would require either (i) showing how physical proximity enters the attention mechanism, or (ii) reformulating SIT in epistemic terms.
```

---

## Figure Verification

Ensure these files exist in the repository:
- `belief_inertia/phase_portrait_damped.png`
- `belief_inertia/phase_portrait_orbit.png`

If using a different directory structure for journal submission, update paths accordingly.

---

## Final Checklist Before Submission

- [ ] Remove or derive diffusion of innovations
- [ ] Fix observation model notation consistency
- [ ] Add damping coefficient derivation paragraph
- [ ] Correct typographical errors (lines 11, 362, 589, 1137-1138)
- [ ] Deduplicate references.bib
- [ ] Move caveat box earlier (optional but recommended)
- [ ] Verify figure paths
- [ ] Check JMS AI disclosure requirements
- [ ] Run LaTeX compilation to verify no errors
