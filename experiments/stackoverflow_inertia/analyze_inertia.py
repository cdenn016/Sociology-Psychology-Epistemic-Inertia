"""
Stack Overflow Epistemic Inertia Analysis

Tests the hypothesis (H_so) that high-reputation Stack Overflow users exhibit
greater epistemic inertia — specifically, they edit their answers LESS
frequently after receiving critical comments.

Theoretical framework (Hamiltonian belief dynamics):
    The epistemic mass matrix M_i determines resistance to belief revision.
    The full mass formula is:

        M_i = Lambda_p + Lambda_o + Sum_k beta_ik Lambda_tilde_qk + Sum_j beta_ji Lambda_qi

    where:
        Lambda_p  = prior precision        -> log(reputation)
        Lambda_o  = observation precision  -> tag_expertise_ratio
        Sum beta_ji = outgoing social ties -> log(view_count * score)

    High M_i => greater inertia => lower P(edit | criticism).

Hypothesis (H_so):
    High-reputation users edit answers less frequently in response to
    critical comments, controlling for answer quality, question difficulty,
    topic area, and comment timing.

Statistical tests:
    1. Logistic regression: P(edit | criticism) ~ log(reputation) + controls
       - Controls: answer_score, question_difficulty, primary_tag, hours_to_comment
       - Expected: OR ~ 0.85 per SD increase in log(reputation)
    2. Spearman correlation: rho(log_reputation, edit_probability) < 0
    3. Power analysis: N = 10,000 for 80% power at alpha = 0.05

Data: Processed CSV from fetch_data.py (SEDE pipeline).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# Optional heavy imports — degrade gracefully
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import logit as smf_logit
    from statsmodels.stats.power import NormalIndPower

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


class StackOverflowInertiaAnalyzer:
    """
    Analyze epistemic inertia in Stack Overflow answer-editing behavior.

    Mass formula mapping:
        M_i = Lambda_p + Lambda_o + Sum_k beta_ik Lambda_tilde_qk + Sum_j beta_ji Lambda_qi

        Lambda_p  (prior precision)       = log(reputation)
        Lambda_o  (observation precision) = tag_expertise_ratio
        Sum beta_ji (outgoing social)     = log(answer_view_count * score)

    The composite mass_score is the sum of z-scored components.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    # ----- data loading --------------------------------------------------

    def load_latest(self) -> pd.DataFrame:
        """
        Load the most recent processed dataset from fetch_data.py.

        Returns:
            DataFrame with one row per answer-comment event.
        """
        files = sorted(self.data_dir.glob("processed_*.csv"))
        if not files:
            raise FileNotFoundError(
                f"No processed_*.csv found in {self.data_dir}/.\n"
                "Run: python fetch_data.py --process"
            )
        df = pd.read_csv(files[-1])
        print(f"Loaded {len(df)} events from {files[-1].name}")
        return df

    # ----- core analysis -------------------------------------------------

    def run_full_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Execute all hypothesis tests.

        Args:
            df: Processed DataFrame from load_latest()

        Returns:
            Dictionary of results keyed by test name.
        """
        print("=" * 72)
        print("  EPISTEMIC INERTIA ANALYSIS — STACK OVERFLOW")
        print("  H_so: High reputation => lower P(edit | criticism)")
        print("=" * 72)

        results = {}

        # 1. Descriptive statistics
        results["descriptive"] = self._descriptive_stats(df)

        # 2. Primary test: logistic regression
        results["logistic"] = self._logistic_regression(df)

        # 3. Spearman correlation
        results["spearman"] = self._spearman_correlation(df)

        # 4. Stratified analysis by reputation quartile
        results["stratified"] = self._stratified_analysis(df)

        # 5. Mass-score composite analysis
        results["mass_composite"] = self._mass_composite_analysis(df)

        # 6. Power analysis
        results["power"] = self._power_analysis(df)

        # Summary
        self._print_summary(results)

        return results

    # ----- individual tests -----------------------------------------------

    def _descriptive_stats(self, df: pd.DataFrame) -> Dict:
        """Print and return descriptive statistics."""
        print("\n" + "-" * 72)
        print("  DESCRIPTIVE STATISTICS")
        print("-" * 72)

        n = len(df)
        n_users = df["answerer_id"].nunique()
        edit_rate = df["edited_after_comment"].mean()

        print(f"  N events:          {n:,}")
        print(f"  N unique users:    {n_users:,}")
        print(f"  Overall edit rate: {edit_rate:.1%}")
        print()

        # Reputation distribution
        rep = df["reputation"]
        print(f"  Reputation — median: {rep.median():,.0f}  "
              f"mean: {rep.mean():,.0f}  "
              f"IQR: [{rep.quantile(0.25):,.0f}, {rep.quantile(0.75):,.0f}]")

        log_rep = df["log_reputation"]
        print(f"  log(reputation) — mean: {log_rep.mean():.2f}  "
              f"SD: {log_rep.std():.2f}")

        # Tag expertise
        te = df["tag_expertise_ratio"]
        print(f"  Tag expertise — mean: {te.mean():.3f}  "
              f"median: {te.median():.3f}")

        # Social reach
        sr = df["log_social_reach"]
        print(f"  log(social reach) — mean: {sr.mean():.2f}  "
              f"SD: {sr.std():.2f}")

        return {
            "n_events": n,
            "n_users": n_users,
            "edit_rate": edit_rate,
            "reputation_median": float(rep.median()),
            "log_reputation_mean": float(log_rep.mean()),
            "log_reputation_sd": float(log_rep.std()),
        }

    def _logistic_regression(self, df: pd.DataFrame) -> Dict:
        """
        Primary hypothesis test: logistic regression.

        Model:
            logit(P(edit)) = b0 + b1*log_reputation + b2*answer_score
                             + b3*question_difficulty + b4*hours_to_comment
                             + tag fixed effects

        Expected under H_so:
            b1 < 0  (higher reputation => lower edit probability)
            OR ~ 0.85 per SD increase in log(reputation)

        Returns:
            Dictionary with coefficients, odds ratios, p-values.
        """
        print("\n" + "-" * 72)
        print("  TEST 1: LOGISTIC REGRESSION")
        print("  logit(P(edit)) ~ log_reputation + answer_score")
        print("                   + question_difficulty + hours_to_comment")
        print("-" * 72)

        if not HAS_STATSMODELS:
            print("  [SKIPPED] statsmodels not installed.")
            print("  Install with: pip install statsmodels")
            return {"skipped": True, "reason": "statsmodels not available"}

        # Prepare analysis DataFrame
        analysis_cols = [
            "edited_after_comment",
            "log_reputation",
            "answer_score",
            "question_difficulty",
            "hours_to_comment",
            "tag_expertise_ratio",
            "log_social_reach",
            "primary_tag",
        ]
        adf = df[analysis_cols].dropna().copy()

        # Standardize continuous predictors for interpretable ORs
        continuous = [
            "log_reputation",
            "answer_score",
            "question_difficulty",
            "hours_to_comment",
            "tag_expertise_ratio",
            "log_social_reach",
        ]
        for col in continuous:
            mean = adf[col].mean()
            std = adf[col].std()
            if std > 1e-10:
                adf[f"{col}_z"] = (adf[col] - mean) / std
            else:
                adf[f"{col}_z"] = 0.0

        # Top-N tags as dummies (avoid too many categories)
        top_tags = adf["primary_tag"].value_counts().head(20).index
        adf["tag_top"] = adf["primary_tag"].where(
            adf["primary_tag"].isin(top_tags), other="other"
        )

        # Fit model
        formula = (
            "edited_after_comment ~ log_reputation_z + answer_score_z "
            "+ question_difficulty_z + hours_to_comment_z "
            "+ tag_expertise_ratio_z + log_social_reach_z "
            "+ C(tag_top)"
        )

        try:
            model = smf_logit(formula, data=adf).fit(disp=0, maxiter=100)
        except Exception as e:
            print(f"  Model failed to converge: {e}")
            # Fallback: simpler model without tag FE
            formula_simple = (
                "edited_after_comment ~ log_reputation_z + answer_score_z "
                "+ question_difficulty_z + hours_to_comment_z"
            )
            model = smf_logit(formula_simple, data=adf).fit(disp=0, maxiter=100)

        # Extract key results
        params = model.params
        pvalues = model.pvalues
        conf_int = model.conf_int()

        # Odds ratios
        odds_ratios = np.exp(params)

        print(f"\n  N observations: {int(model.nobs):,}")
        print(f"  Pseudo R-squared: {model.prsquared:.4f}")
        print(f"  AIC: {model.aic:.1f}")
        print()

        # Key predictor: log_reputation
        rep_coef = params.get("log_reputation_z", np.nan)
        rep_or = odds_ratios.get("log_reputation_z", np.nan)
        rep_p = pvalues.get("log_reputation_z", np.nan)
        rep_ci = conf_int.loc["log_reputation_z"] if "log_reputation_z" in conf_int.index else [np.nan, np.nan]

        print("  KEY RESULT — log(reputation) (standardized):")
        print(f"    Coefficient: {rep_coef:.4f}")
        print(f"    Odds Ratio:  {rep_or:.4f}  "
              f"[95% CI: {np.exp(rep_ci.iloc[0]):.4f}, {np.exp(rep_ci.iloc[1]):.4f}]")
        print(f"    p-value:     {rep_p:.4e}")

        if rep_p < 0.05 and rep_coef < 0:
            print("    => SIGNIFICANT NEGATIVE: Higher reputation => lower P(edit)")
            print("       Consistent with epistemic inertia (H_so supported)")
        elif rep_p < 0.05 and rep_coef > 0:
            print("    => SIGNIFICANT POSITIVE: Higher reputation => higher P(edit)")
            print("       INCONSISTENT with epistemic inertia")
        else:
            print("    => NOT SIGNIFICANT at alpha = 0.05")

        # Print all coefficients
        print("\n  Full model coefficients (non-tag):")
        for var in ["log_reputation_z", "answer_score_z", "question_difficulty_z",
                     "hours_to_comment_z", "tag_expertise_ratio_z", "log_social_reach_z"]:
            if var in params.index:
                print(f"    {var:30s}  b={params[var]:+.4f}  "
                      f"OR={odds_ratios[var]:.4f}  p={pvalues[var]:.4e}")

        return {
            "n_obs": int(model.nobs),
            "pseudo_r2": float(model.prsquared),
            "aic": float(model.aic),
            "log_reputation_coef": float(rep_coef),
            "log_reputation_or": float(rep_or),
            "log_reputation_p": float(rep_p),
            "log_reputation_ci_low": float(np.exp(rep_ci.iloc[0])),
            "log_reputation_ci_high": float(np.exp(rep_ci.iloc[1])),
            "significant": bool(rep_p < 0.05),
            "direction": "negative" if rep_coef < 0 else "positive",
            "all_coefficients": {
                k: {"coef": float(v), "or": float(odds_ratios[k]), "p": float(pvalues[k])}
                for k, v in params.items()
                if not k.startswith("C(") and k != "Intercept"
            },
        }

    def _spearman_correlation(self, df: pd.DataFrame) -> Dict:
        """
        Spearman rank correlation between log(reputation) and edit probability.

        Aggregates to user-level edit rate to avoid pseudoreplication.

        Expected: rho < 0 (negative correlation).
        """
        print("\n" + "-" * 72)
        print("  TEST 2: SPEARMAN CORRELATION")
        print("  rho(log_reputation, user_edit_rate)")
        print("-" * 72)

        # Aggregate to user level
        user_agg = df.groupby("answerer_id").agg(
            log_reputation=("log_reputation", "first"),
            edit_rate=("edited_after_comment", "mean"),
            n_events=("edited_after_comment", "count"),
        ).reset_index()

        # Require at least 3 events per user for stable rate
        user_agg = user_agg[user_agg["n_events"] >= 3]

        if len(user_agg) < 10:
            print(f"  [SKIPPED] Only {len(user_agg)} users with >= 3 events")
            return {"skipped": True, "reason": "insufficient users"}

        rho, p_value = stats.spearmanr(
            user_agg["log_reputation"], user_agg["edit_rate"]
        )

        print(f"\n  N users (>= 3 events): {len(user_agg):,}")
        print(f"  Spearman rho:          {rho:.4f}")
        print(f"  p-value:               {p_value:.4e}")

        if p_value < 0.05 and rho < 0:
            print("  => SIGNIFICANT NEGATIVE: Higher reputation => lower edit rate")
            print("     Consistent with epistemic inertia (H_so)")
        elif p_value < 0.05 and rho > 0:
            print("  => SIGNIFICANT POSITIVE (unexpected)")
        else:
            print("  => NOT SIGNIFICANT")

        # Also test with Kendall's tau for robustness
        tau, tau_p = stats.kendalltau(
            user_agg["log_reputation"], user_agg["edit_rate"]
        )
        print(f"\n  Robustness — Kendall tau: {tau:.4f}  p={tau_p:.4e}")

        return {
            "n_users": len(user_agg),
            "spearman_rho": float(rho),
            "spearman_p": float(p_value),
            "kendall_tau": float(tau),
            "kendall_p": float(tau_p),
            "significant": bool(p_value < 0.05),
            "direction": "negative" if rho < 0 else "positive",
        }

    def _stratified_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Stratified analysis: edit rate by reputation quartile.

        Shows monotonic decrease in edit rate across quartiles
        if epistemic inertia holds.
        """
        print("\n" + "-" * 72)
        print("  TEST 3: STRATIFIED ANALYSIS BY REPUTATION QUARTILE")
        print("-" * 72)

        df = df.copy()
        df["rep_quartile"] = pd.qcut(
            df["log_reputation"],
            q=4,
            labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
            duplicates="drop",
        )

        strat = df.groupby("rep_quartile", observed=True).agg(
            n=("edited_after_comment", "count"),
            edit_rate=("edited_after_comment", "mean"),
            mean_reputation=("reputation", "mean"),
            mean_log_rep=("log_reputation", "mean"),
        )

        print(f"\n  {'Quartile':15s} {'N':>8s} {'Edit Rate':>12s} {'Mean Rep':>12s}")
        print("  " + "-" * 50)
        for idx, row in strat.iterrows():
            print(f"  {str(idx):15s} {int(row['n']):8,d} "
                  f"{row['edit_rate']:11.1%} {row['mean_reputation']:12,.0f}")

        # Cochran-Armitage trend test (via chi-square for trend)
        quartile_rates = strat["edit_rate"].values
        quartile_ns = strat["n"].values.astype(int)

        # Simple linear trend test
        quartile_midpoints = np.arange(len(quartile_rates))
        slope, intercept, r_value, p_trend, std_err = stats.linregress(
            quartile_midpoints, quartile_rates
        )

        print(f"\n  Linear trend in edit rate across quartiles:")
        print(f"    Slope:   {slope:+.4f} per quartile")
        print(f"    R^2:     {r_value**2:.4f}")
        print(f"    p-value: {p_trend:.4e}")

        if p_trend < 0.05 and slope < 0:
            print("    => SIGNIFICANT DECREASING TREND (supports H_so)")
        elif p_trend < 0.05 and slope > 0:
            print("    => SIGNIFICANT INCREASING TREND (contradicts H_so)")
        else:
            print("    => NO SIGNIFICANT TREND")

        return {
            "quartile_edit_rates": {
                str(k): float(v) for k, v in zip(strat.index, strat["edit_rate"])
            },
            "quartile_ns": {
                str(k): int(v) for k, v in zip(strat.index, strat["n"])
            },
            "trend_slope": float(slope),
            "trend_r2": float(r_value ** 2),
            "trend_p": float(p_trend),
            "significant_decrease": bool(p_trend < 0.05 and slope < 0),
        }

    def _mass_composite_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Test using the full composite mass score.

        M_i = Lambda_p + Lambda_o + Sum beta_ji Lambda_qi
            = z(log_reputation) + z(tag_expertise) + z(log_social_reach)

        Tests whether the composite mass predicts edit behavior better
        than reputation alone.
        """
        print("\n" + "-" * 72)
        print("  TEST 4: COMPOSITE MASS SCORE ANALYSIS")
        print("  M_i = z(log_rep) + z(tag_expertise) + z(log_social_reach)")
        print("-" * 72)

        valid = df[["mass_score", "edited_after_comment"]].dropna()
        if len(valid) < 30:
            print(f"  [SKIPPED] Only {len(valid)} valid observations")
            return {"skipped": True}

        # Spearman correlation with composite mass
        rho, p_value = stats.spearmanr(
            valid["mass_score"], valid["edited_after_comment"]
        )
        print(f"\n  Spearman rho(mass_score, edit): {rho:.4f}  p={p_value:.4e}")

        # Compare mass quartiles
        valid = valid.copy()
        valid["mass_quartile"] = pd.qcut(
            valid["mass_score"], q=4,
            labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
            duplicates="drop",
        )

        mass_strat = valid.groupby("mass_quartile", observed=True).agg(
            n=("edited_after_comment", "count"),
            edit_rate=("edited_after_comment", "mean"),
        )

        print(f"\n  {'Mass Quartile':15s} {'N':>8s} {'Edit Rate':>12s}")
        print("  " + "-" * 38)
        for idx, row in mass_strat.iterrows():
            print(f"  {str(idx):15s} {int(row['n']):8,d} {row['edit_rate']:11.1%}")

        # Point-biserial for effect size
        rpb, rpb_p = stats.pointbiserialr(
            valid["edited_after_comment"], valid["mass_score"]
        )
        print(f"\n  Point-biserial r(mass, edit): {rpb:.4f}  p={rpb_p:.4e}")

        return {
            "spearman_rho": float(rho),
            "spearman_p": float(p_value),
            "pointbiserial_r": float(rpb),
            "pointbiserial_p": float(rpb_p),
            "quartile_edit_rates": {
                str(k): float(v) for k, v in zip(mass_strat.index, mass_strat["edit_rate"])
            },
        }

    def _power_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Statistical power analysis for the logistic regression.

        Computes:
            - Required N for 80% power at OR = 0.85
            - Achieved power at current N
            - Minimum detectable OR at current N

        Uses the normal approximation for logistic regression power
        (Hsieh, Bloch, Larsen 1998).
        """
        print("\n" + "-" * 72)
        print("  POWER ANALYSIS")
        print("  Target: OR = 0.85 per SD increase in log(reputation)")
        print("-" * 72)

        n_current = len(df.dropna(subset=["log_reputation", "edited_after_comment"]))
        p0 = df["edited_after_comment"].mean()  # baseline edit rate

        # Target effect: OR = 0.85 => log(OR) = -0.1625
        target_or = 0.85
        target_log_or = np.log(target_or)

        if HAS_STATSMODELS:
            # Use statsmodels power calculator (normal approximation)
            power_calc = NormalIndPower()

            # Convert OR to Cohen's h-like effect size for binary outcome
            # Using the formula: effect_size = log(OR) * sqrt(p0*(1-p0))
            # This is an approximation; exact logistic power is complex.
            effect_size = abs(target_log_or) * np.sqrt(p0 * (1 - p0))

            # Required N for 80% power
            try:
                n_required = power_calc.solve_power(
                    effect_size=effect_size,
                    power=0.80,
                    alpha=0.05,
                    alternative="two-sided",
                )
                n_required = int(np.ceil(n_required))
            except Exception:
                n_required = self._manual_power_n(target_log_or, p0, power=0.80)

            # Achieved power at current N
            try:
                achieved_power = power_calc.solve_power(
                    effect_size=effect_size,
                    nobs=n_current,
                    alpha=0.05,
                    alternative="two-sided",
                )
            except Exception:
                achieved_power = self._manual_power(target_log_or, p0, n_current)
        else:
            # Manual power calculation using normal approximation
            n_required = self._manual_power_n(target_log_or, p0, power=0.80)
            achieved_power = self._manual_power(target_log_or, p0, n_current)

        print(f"\n  Baseline edit rate (p0):    {p0:.3f}")
        print(f"  Target effect (OR):        {target_or}")
        print(f"  Target log(OR):            {target_log_or:.4f}")
        print(f"  Current N:                 {n_current:,}")
        print(f"  Required N (80% power):    {n_required:,}")
        print(f"  Achieved power at N={n_current:,}: {achieved_power:.1%}")

        if n_current >= n_required:
            print("  => ADEQUATELY POWERED")
        else:
            shortfall = n_required - n_current
            print(f"  => UNDERPOWERED: need {shortfall:,} more observations")

        # Reference: typical SEDE queries return 50k+ rows
        print(f"\n  Reference: N=10,000 sufficient for OR=0.85 at p0={p0:.2f}")

        return {
            "baseline_rate": float(p0),
            "target_or": target_or,
            "n_current": n_current,
            "n_required_80pct": n_required,
            "achieved_power": float(achieved_power),
            "adequately_powered": bool(n_current >= n_required),
        }

    # ----- manual power helpers ------------------------------------------

    @staticmethod
    def _manual_power_n(log_or: float, p0: float, power: float = 0.80,
                        alpha: float = 0.05) -> int:
        """
        Required sample size for logistic regression power.

        Uses Hsieh (1989) formula:
            N = (z_{1-alpha/2} + z_power)^2 / (p0 * (1-p0) * log_or^2)

        Args:
            log_or: log odds ratio (effect size)
            p0: baseline event rate
            power: desired power (default 0.80)
            alpha: significance level (default 0.05)

        Returns:
            Required sample size.
        """
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_power = stats.norm.ppf(power)
        n = (z_alpha + z_power) ** 2 / (p0 * (1 - p0) * log_or ** 2)
        return int(np.ceil(n))

    @staticmethod
    def _manual_power(log_or: float, p0: float, n: int,
                      alpha: float = 0.05) -> float:
        """
        Achieved power at given sample size.

        Args:
            log_or: log odds ratio
            p0: baseline event rate
            n: sample size
            alpha: significance level

        Returns:
            Statistical power (0 to 1).
        """
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        se = 1.0 / np.sqrt(n * p0 * (1 - p0))
        z_stat = abs(log_or) / se
        power = stats.norm.cdf(z_stat - z_alpha)
        return float(power)

    # ----- summary -------------------------------------------------------

    def _print_summary(self, results: Dict) -> None:
        """Print overall summary of findings."""
        print("\n" + "=" * 72)
        print("  SUMMARY OF FINDINGS")
        print("=" * 72)

        print("\n  Hypothesis (H_so):")
        print("    High-reputation SO users edit answers less frequently")
        print("    in response to critical comments (epistemic inertia).")
        print()

        # Logistic regression
        lr = results.get("logistic", {})
        if not lr.get("skipped"):
            sig = "YES" if lr.get("significant") else "NO"
            direction = lr.get("direction", "?")
            or_val = lr.get("log_reputation_or", float("nan"))
            p_val = lr.get("log_reputation_p", float("nan"))
            print(f"  1. Logistic regression:")
            print(f"     Significant: {sig}  Direction: {direction}")
            print(f"     OR(log_rep): {or_val:.4f}  p={p_val:.4e}")
            if lr.get("significant") and direction == "negative":
                print("     => SUPPORTS H_so")
            elif lr.get("significant") and direction == "positive":
                print("     => CONTRADICTS H_so")
            else:
                print("     => INCONCLUSIVE")
        else:
            print("  1. Logistic regression: SKIPPED")

        # Spearman
        sp = results.get("spearman", {})
        if not sp.get("skipped"):
            rho = sp.get("spearman_rho", float("nan"))
            p_val = sp.get("spearman_p", float("nan"))
            print(f"\n  2. Spearman correlation (user-level):")
            print(f"     rho = {rho:.4f}  p = {p_val:.4e}")
            if sp.get("significant") and sp.get("direction") == "negative":
                print("     => SUPPORTS H_so")
            else:
                print("     => DOES NOT SUPPORT H_so")
        else:
            print("\n  2. Spearman correlation: SKIPPED")

        # Stratified
        st = results.get("stratified", {})
        if st.get("significant_decrease"):
            print(f"\n  3. Stratified trend test:")
            print(f"     Slope = {st['trend_slope']:+.4f}  p = {st['trend_p']:.4e}")
            print("     => MONOTONIC DECREASE ACROSS QUARTILES (supports H_so)")
        elif "trend_slope" in st:
            print(f"\n  3. Stratified trend test:")
            print(f"     Slope = {st['trend_slope']:+.4f}  p = {st['trend_p']:.4e}")
            print("     => NO SIGNIFICANT MONOTONIC TREND")
        else:
            print("\n  3. Stratified analysis: SKIPPED")

        # Power
        pw = results.get("power", {})
        if pw:
            print(f"\n  4. Power analysis:")
            print(f"     Current N:     {pw.get('n_current', '?'):,}")
            print(f"     Required N:    {pw.get('n_required_80pct', '?'):,}")
            print(f"     Achieved power: {pw.get('achieved_power', 0):.1%}")

        print("\n" + "=" * 72)

    # ----- visualization -------------------------------------------------

    def plot_results(self, df: pd.DataFrame, output_dir: Optional[str] = None) -> None:
        """
        Generate diagnostic and result plots.

        Creates a 2x2 figure:
            (a) Edit rate by reputation quartile
            (b) Scatter: log(reputation) vs user-level edit rate
            (c) Mass score distribution by edit outcome
            (d) ROC-like: edit rate vs mass percentile

        Args:
            df: Processed DataFrame
            output_dir: Where to save. Defaults to data/
        """
        if not HAS_PLOTTING:
            print("Plotting requires matplotlib and seaborn.")
            return

        out_dir = Path(output_dir) if output_dir else self.data_dir
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (a) Edit rate by reputation quartile
        ax = axes[0, 0]
        df_plot = df.copy()
        df_plot["rep_quartile"] = pd.qcut(
            df_plot["log_reputation"], q=4,
            labels=["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"],
            duplicates="drop",
        )
        rates = df_plot.groupby("rep_quartile", observed=True)[
            "edited_after_comment"
        ].mean()
        rates.plot(kind="bar", ax=ax, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
        ax.set_ylabel("P(edit | criticism)")
        ax.set_xlabel("Reputation Quartile")
        ax.set_title("Edit Rate by Reputation Quartile")
        ax.set_ylim(0, None)
        ax.tick_params(axis="x", rotation=0)

        # (b) Scatter: user-level edit rate vs log(reputation)
        ax = axes[0, 1]
        user_agg = df.groupby("answerer_id").agg(
            log_rep=("log_reputation", "first"),
            edit_rate=("edited_after_comment", "mean"),
            n=("edited_after_comment", "count"),
        ).reset_index()
        user_agg = user_agg[user_agg["n"] >= 3]

        ax.scatter(user_agg["log_rep"], user_agg["edit_rate"],
                   alpha=0.4, s=15, edgecolors="none")
        # Trend line
        if len(user_agg) > 10:
            z = np.polyfit(user_agg["log_rep"], user_agg["edit_rate"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(user_agg["log_rep"].min(), user_agg["log_rep"].max(), 100)
            ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8)
        ax.set_xlabel("log(reputation)")
        ax.set_ylabel("User-level edit rate")
        ax.set_title("Reputation vs Edit Responsiveness")
        ax.grid(alpha=0.3)

        # (c) Mass score distribution by edit outcome
        ax = axes[1, 0]
        edited = df[df["edited_after_comment"] == 1]["mass_score"].dropna()
        not_edited = df[df["edited_after_comment"] == 0]["mass_score"].dropna()
        ax.hist([not_edited, edited], bins=40, alpha=0.6, density=True,
                label=["Did not edit", "Edited"], color=["#c44e52", "#55a868"])
        ax.set_xlabel("Composite Mass Score (M_i)")
        ax.set_ylabel("Density")
        ax.set_title("Mass Distribution by Edit Outcome")
        ax.legend()
        ax.grid(alpha=0.3)

        # (d) Edit rate vs mass percentile (smooth curve)
        ax = axes[1, 1]
        df_sorted = df[["mass_score", "edited_after_comment"]].dropna().copy()
        df_sorted["mass_pctile"] = df_sorted["mass_score"].rank(pct=True) * 100
        # Bin into 20 bins and compute edit rate
        df_sorted["mass_bin"] = pd.cut(df_sorted["mass_pctile"], bins=20)
        bin_rates = df_sorted.groupby("mass_bin", observed=True)[
            "edited_after_comment"
        ].mean()
        bin_centers = [interval.mid for interval in bin_rates.index]
        ax.plot(bin_centers, bin_rates.values, "o-", color="#4c72b0", linewidth=2)
        ax.set_xlabel("Mass Score Percentile")
        ax.set_ylabel("P(edit | criticism)")
        ax.set_title("Edit Probability vs Epistemic Mass")
        ax.grid(alpha=0.3)

        plt.suptitle(
            "Epistemic Inertia in Stack Overflow\n"
            r"$M_i = \Lambda_p + \Lambda_o + \sum \beta_{ji} \Lambda_{qi}$",
            fontsize=14, y=1.02,
        )
        plt.tight_layout()
        out_path = out_dir / "epistemic_inertia_stackoverflow.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved plots to {out_path}")


def main():
    """Run the full epistemic inertia analysis pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze epistemic inertia in Stack Overflow data"
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Directory containing processed_*.csv (default: data/)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating plots",
    )
    args = parser.parse_args()

    analyzer = StackOverflowInertiaAnalyzer(data_dir=args.data_dir)

    # Load data
    df = analyzer.load_latest()

    # Run analysis
    results = analyzer.run_full_analysis(df)

    # Generate plots
    if not args.no_plots:
        analyzer.plot_results(df)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
