"""
Epistemic Inertia Analysis — OpenAlex Retraction Citation Data

Tests whether authors with high "epistemic mass" take longer to stop citing
retracted papers, using retraction events as natural experiments.

Hypotheses tested
-----------------
H1.2  Highly-cited authors take longer to stop citing retracted papers.
      relaxation_time ~ Lambda_p  (prior mass).
      Operationalized: Spearman(log(cited_by_count), relaxation_time) > 0.

H3.1  Authors who are themselves highly cited show slower post-retraction
      citation decay.
      Operationalized: In a Cox proportional-hazards model for time-to-
      stop-citing, the hazard ratio for high-cited authors is < 1
      (i.e. they are *slower* to stop).  Expected HR ~ 0.80-0.90.

Mass formula mapping (same as fetch_data.py):
    M_i = Lambda_p + Lambda_o + Sigma beta_ji

    Lambda_p  = log(career_citation_count + 1)
        Prior precision / reputation proxy.

    Lambda_o  = works_count in same concept
        Domain expertise / observation precision.

    Sigma beta_ji = cited_by_count
        Outgoing social coupling: how many cite THIS author.

    relaxation_time = years from retraction until author stops citing
        the retracted work (right-censored if still citing).

Statistical methods
-------------------
1. Spearman rank correlation: relaxation_time vs. each mass proxy.
2. Cox proportional hazards model (via lifelines):
       h(t | X) = h_0(t) * exp(beta_1 * lambda_p
                                + beta_2 * lambda_o
                                + beta_3 * log(beta_ji + 1))
   A hazard ratio < 1 for a mass proxy means higher mass -> slower
   cessation -> greater inertia.
3. Kaplan-Meier survival curves stratified by mass quartile.
4. Mann-Whitney U test comparing relaxation times between high/low
   mass groups.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# Optional heavy imports — gracefully degrade if unavailable
try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class RetractionInertiaAnalyzer:
    """
    Analyze epistemic inertia from retraction-citation tracking data.

    Expects CSV files produced by ``fetch_data.py``:
        authors_<ts>.csv   — author career stats + mass proxies
        tracking_<ts>.csv  — post-retraction citation tracking per author-paper pair
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_latest(self) -> Dict[str, pd.DataFrame]:
        """Load most recent data files by timestamp suffix."""
        def _latest(pattern: str) -> pd.DataFrame:
            files = sorted(self.data_dir.glob(pattern))
            if not files:
                raise FileNotFoundError(
                    f"No files matching {pattern} in {self.data_dir}"
                )
            return pd.read_csv(files[-1])

        return {
            "authors": _latest("authors_*.csv"),
            "tracking": _latest("tracking_*.csv"),
            "retractions": _latest("retractions_*.csv"),
            "citations": _latest("citations_*.csv"),
        }

    # ------------------------------------------------------------------
    # Prepare analysis dataset
    # ------------------------------------------------------------------

    def prepare_dataset(
        self,
        authors_df: pd.DataFrame,
        tracking_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge author mass proxies with post-retraction tracking data.

        Adds derived columns:
            - lambda_p: log(cited_by_count + 1)  [prior precision]
            - lambda_o: works_count              [domain expertise]
            - log_beta_ji: log(cited_by_count + 1) [social influence]
            - mass_composite: weighted sum of normalized proxies
            - event_observed: 1 if author stopped citing, 0 if right-censored
            - duration: relaxation_time_years (floored at 0)
        """
        merged = tracking_df.merge(authors_df, on="author_id", how="inner")

        if merged.empty:
            print("WARNING: merge produced 0 rows; check author_id overlap")
            return merged

        # Ensure numeric
        for col in ["cited_by_count", "works_count", "h_index"]:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

        # Mass proxies
        merged["lambda_p"] = np.log(merged["cited_by_count"] + 1)
        merged["lambda_o"] = merged["works_count"]
        merged["log_beta_ji"] = np.log(merged["cited_by_count"] + 1)

        # Composite mass (normalized then weighted)
        def _norm(s: pd.Series) -> pd.Series:
            r = s.max() - s.min()
            return (s - s.min()) / r if r > 1e-10 else pd.Series(0.5, index=s.index)

        merged["mass_composite"] = (
            0.4 * _norm(merged["lambda_p"])
            + 0.35 * _norm(merged["lambda_o"])
            + 0.25 * _norm(merged["log_beta_ji"])
        )

        # Survival analysis columns
        merged["event_observed"] = (~merged["still_citing"]).astype(int)
        merged["duration"] = merged["relaxation_time_years"].fillna(0).clip(lower=0)

        # For authors who never cited post-retraction, set duration = 0 and event = 1
        never_cited = merged["post_retraction_citation_count"] == 0
        merged.loc[never_cited, "duration"] = 0
        merged.loc[never_cited, "event_observed"] = 1

        # Remove rows with zero duration AND event observed
        # (these don't contribute to survival analysis)
        # Keep right-censored zeros (still_citing but duration 0) as they carry information
        valid = ~((merged["duration"] == 0) & (merged["event_observed"] == 1))
        dropped = (~valid).sum()
        merged = merged[valid].copy()
        print(f"Prepared dataset: {len(merged)} author-paper pairs "
              f"({dropped} immediate-stoppers removed)")

        return merged

    # ------------------------------------------------------------------
    # H1.2: Spearman correlation — relaxation time vs mass
    # ------------------------------------------------------------------

    def test_h1_2_spearman(self, df: pd.DataFrame) -> Dict:
        """
        H1.2: Highly-cited authors take longer to stop citing retracted papers.

        Computes Spearman rank correlation between each mass proxy and
        relaxation_time_years.

        Expected: positive rho (higher mass -> longer relaxation).
        """
        print("\n" + "=" * 72)
        print("H1.2: RELAXATION TIME SCALES WITH EPISTEMIC MASS (Spearman)")
        print("=" * 72)

        results = {}
        proxies = {
            "lambda_p (prior precision)": "lambda_p",
            "lambda_o (domain expertise)": "lambda_o",
            "log_beta_ji (social influence)": "log_beta_ji",
            "mass_composite": "mass_composite",
            "h_index": "h_index",
        }

        valid = df[df["duration"] > 0].copy()

        if len(valid) < 10:
            print("  Insufficient data for correlation test (need >= 10 pairs)")
            return {"error": "insufficient_data", "n": len(valid)}

        for label, col in proxies.items():
            if col not in valid.columns:
                continue
            rho, p = stats.spearmanr(valid[col], valid["duration"])
            sig = p < 0.05
            direction = "positive (as predicted)" if rho > 0 else "NEGATIVE (unexpected)"

            print(f"\n  {label}:")
            print(f"    rho = {rho:+.4f},  p = {p:.4e}  "
                  f"{'[SIGNIFICANT]' if sig else '[not significant]'}")
            if sig:
                print(f"    Interpretation: {direction}")

            results[col] = {"rho": rho, "p_value": p, "significant": sig, "n": len(valid)}

        return results

    # ------------------------------------------------------------------
    # H3.1: Cox proportional hazards
    # ------------------------------------------------------------------

    def test_h3_1_cox(self, df: pd.DataFrame) -> Dict:
        """
        H3.1: Authors highly cited themselves show slower post-retraction
        citation decay.

        Fits a Cox proportional hazards model:
            h(t | X) = h_0(t) exp(b1*lambda_p + b2*lambda_o + b3*log_beta_ji)

        A hazard ratio < 1 for a covariate means that higher values of that
        covariate are associated with SLOWER time-to-stop-citing (greater
        inertia).

        Expected: HR for lambda_p ~ 0.80-0.90.
        """
        print("\n" + "=" * 72)
        print("H3.1: COX PROPORTIONAL HAZARDS — TIME TO STOP CITING")
        print("=" * 72)

        if not HAS_LIFELINES:
            print("  lifelines not installed; skipping Cox model.")
            print("  Install with: pip install lifelines")
            return {"error": "lifelines_not_installed"}

        covariates = ["lambda_p", "lambda_o", "log_beta_ji"]
        surv_cols = covariates + ["duration", "event_observed"]
        surv_df = df[surv_cols].dropna()

        # Need duration > 0 for Cox
        surv_df = surv_df[surv_df["duration"] > 0].copy()

        if len(surv_df) < 20:
            print(f"  Insufficient data for Cox model (n={len(surv_df)}, need >= 20)")
            return {"error": "insufficient_data", "n": len(surv_df)}

        # Standardize covariates for numerical stability
        for col in covariates:
            mu, sigma = surv_df[col].mean(), surv_df[col].std()
            if sigma > 1e-10:
                surv_df[col] = (surv_df[col] - mu) / sigma

        cph = CoxPHFitter()
        try:
            cph.fit(
                surv_df,
                duration_col="duration",
                event_col="event_observed",
                show_progress=False,
            )
        except Exception as exc:
            print(f"  Cox fitting failed: {exc}")
            return {"error": str(exc)}

        print(f"\n  N = {len(surv_df)},  events = {surv_df['event_observed'].sum()}")
        print(f"  Concordance index = {cph.concordance_index_:.4f}")
        print()

        summary = cph.summary
        results = {}

        for covar in covariates:
            if covar not in summary.index:
                continue
            row = summary.loc[covar]
            hr = row["exp(coef)"]
            p_val = row["p"]
            ci_lo = row.get("exp(coef) lower 95%", float("nan"))
            ci_hi = row.get("exp(coef) upper 95%", float("nan"))

            print(f"  {covar}:")
            print(f"    HR = {hr:.4f}  (95% CI: {ci_lo:.4f} - {ci_hi:.4f})")
            print(f"    p  = {p_val:.4e}")

            if p_val < 0.05:
                if hr < 1.0:
                    interp = "CONFIRMED: higher mass -> slower cessation (inertia)"
                else:
                    interp = "UNEXPECTED: higher mass -> faster cessation"
                print(f"    >> {interp}")
            else:
                print(f"    >> Not significant at alpha=0.05")

            results[covar] = {
                "hazard_ratio": hr,
                "p_value": p_val,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "significant": p_val < 0.05,
            }

        results["concordance_index"] = cph.concordance_index_
        results["n"] = len(surv_df)
        results["n_events"] = int(surv_df["event_observed"].sum())

        return results

    # ------------------------------------------------------------------
    # Mann-Whitney U: high vs low mass relaxation times
    # ------------------------------------------------------------------

    def test_mann_whitney(self, df: pd.DataFrame) -> Dict:
        """
        Non-parametric comparison of relaxation times between high-mass
        and low-mass author groups (median split on mass_composite).

        Expected: high-mass group has significantly longer relaxation times.
        """
        print("\n" + "=" * 72)
        print("MANN-WHITNEY U: HIGH vs LOW MASS RELAXATION TIMES")
        print("=" * 72)

        valid = df[df["duration"] > 0].copy()
        if len(valid) < 10:
            print(f"  Insufficient data (n={len(valid)})")
            return {"error": "insufficient_data"}

        median_mass = valid["mass_composite"].median()
        high = valid[valid["mass_composite"] >= median_mass]["duration"]
        low = valid[valid["mass_composite"] < median_mass]["duration"]

        print(f"\n  High mass (n={len(high)}):  mean = {high.mean():.2f} yr,  "
              f"median = {high.median():.2f} yr")
        print(f"  Low mass  (n={len(low)}):   mean = {low.mean():.2f} yr,  "
              f"median = {low.median():.2f} yr")

        u_stat, p_val = stats.mannwhitneyu(high, low, alternative="greater")
        print(f"\n  U = {u_stat:.1f},  p = {p_val:.4e}")

        if p_val < 0.05:
            print("  >> SIGNIFICANT: High-mass authors take longer to stop citing")
        else:
            print("  >> Not significant")

        return {
            "high_mass_mean": high.mean(),
            "low_mass_mean": low.mean(),
            "high_mass_median": high.median(),
            "low_mass_median": low.median(),
            "u_statistic": u_stat,
            "p_value": p_val,
            "significant": p_val < 0.05,
        }

    # ------------------------------------------------------------------
    # Kaplan-Meier survival curves by mass quartile
    # ------------------------------------------------------------------

    def plot_kaplan_meier(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Plot Kaplan-Meier survival curves stratified by mass quartile.

        "Survival" here means the author is STILL citing the retracted paper.
        Higher mass should show curves that stay elevated longer.
        """
        if not HAS_LIFELINES or not HAS_PLOTTING:
            print("  Skipping KM plot (lifelines or matplotlib not available)")
            return

        valid = df[df["duration"] > 0].copy()
        if len(valid) < 20:
            print("  Too few data points for KM plot")
            return

        valid["mass_quartile"] = pd.qcut(
            valid["mass_composite"], q=4,
            labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
            duplicates="drop",
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        kmf = KaplanMeierFitter()

        for label in sorted(valid["mass_quartile"].unique()):
            subset = valid[valid["mass_quartile"] == label]
            kmf.fit(
                subset["duration"],
                event_observed=subset["event_observed"],
                label=str(label),
            )
            kmf.plot_survival_function(ax=ax)

        ax.set_xlabel("Years after retraction", fontsize=13)
        ax.set_ylabel("P(still citing retracted work)", fontsize=13)
        ax.set_title(
            "Kaplan-Meier: Persistence of Citation After Retraction\n"
            "by Author Epistemic Mass Quartile",
            fontsize=14,
        )
        ax.legend(title="Mass quartile", fontsize=11)
        ax.grid(alpha=0.3)

        out_path = output_dir / "km_retraction_inertia.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved Kaplan-Meier plot to {out_path}")

    # ------------------------------------------------------------------
    # Full results plot
    # ------------------------------------------------------------------

    def plot_results(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Generate a four-panel figure summarizing the analysis."""
        if not HAS_PLOTTING:
            print("  matplotlib not available; skipping plots")
            return

        valid = df[df["duration"] > 0].copy()
        if len(valid) < 10:
            print("  Insufficient data for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

        # 1. Scatter: lambda_p vs relaxation time
        ax = axes[0, 0]
        ax.scatter(valid["lambda_p"], valid["duration"], alpha=0.5, s=30, edgecolors="k", linewidth=0.3)
        z = np.polyfit(valid["lambda_p"], valid["duration"], 1)
        x_line = np.linspace(valid["lambda_p"].min(), valid["lambda_p"].max(), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), "r--", lw=2, label="OLS trend")
        ax.set_xlabel("Lambda_p = log(cited_by_count + 1)", fontsize=11)
        ax.set_ylabel("Relaxation time (years)", fontsize=11)
        ax.set_title("H1.2: Prior Mass vs Relaxation Time", fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

        # 2. Boxplot by mass quartile
        ax = axes[0, 1]
        valid["mass_q"] = pd.qcut(
            valid["mass_composite"], q=4,
            labels=["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"],
            duplicates="drop",
        )
        valid.boxplot(column="duration", by="mass_q", ax=ax)
        ax.set_xlabel("Epistemic Mass Quartile", fontsize=11)
        ax.set_ylabel("Relaxation Time (years)", fontsize=11)
        ax.set_title("Relaxation Time by Mass Quartile", fontsize=12)
        fig.suptitle("")  # remove pandas auto-title

        # 3. Scatter: h-index vs relaxation time
        ax = axes[1, 0]
        ax.scatter(valid["h_index"], valid["duration"], alpha=0.5, s=30, edgecolors="k", linewidth=0.3)
        ax.set_xlabel("h-index", fontsize=11)
        ax.set_ylabel("Relaxation time (years)", fontsize=11)
        ax.set_title("h-index vs Relaxation Time", fontsize=12)
        ax.grid(alpha=0.3)

        # 4. Distribution of relaxation times
        ax = axes[1, 1]
        median_mass = valid["mass_composite"].median()
        high = valid[valid["mass_composite"] >= median_mass]["duration"]
        low = valid[valid["mass_composite"] < median_mass]["duration"]
        ax.hist(
            [low, high], bins=15, alpha=0.6, density=True,
            label=["Low mass", "High mass"], color=["steelblue", "coral"],
        )
        ax.set_xlabel("Relaxation time (years)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title("Relaxation Time Distribution by Mass Group", fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        out_path = output_dir / "retraction_inertia_results.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved results plot to {out_path}")

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def print_summary(
        self,
        spearman_results: Dict,
        cox_results: Dict,
        mw_results: Dict,
    ) -> None:
        """Print a consolidated summary of all hypothesis tests."""
        print("\n" + "=" * 72)
        print("SUMMARY — EPISTEMIC INERTIA VIA RETRACTION CITATIONS")
        print("=" * 72)

        print("\nH1.2: Relaxation time scales with epistemic mass")
        print("-" * 50)
        if "error" not in spearman_results:
            for col, r in spearman_results.items():
                if isinstance(r, dict):
                    status = "CONFIRMED" if (r["significant"] and r["rho"] > 0) else "not confirmed"
                    print(f"  {col:30s}  rho={r['rho']:+.3f}  p={r['p_value']:.3e}  [{status}]")
        else:
            print(f"  Could not test: {spearman_results.get('error')}")

        print(f"\nH3.1: Cox PH model — high-mass authors slower to stop citing")
        print("-" * 50)
        if "error" not in cox_results:
            for key in ["lambda_p", "lambda_o", "log_beta_ji"]:
                if key in cox_results and isinstance(cox_results[key], dict):
                    r = cox_results[key]
                    hr = r["hazard_ratio"]
                    status = "INERTIA" if (r["significant"] and hr < 1.0) else ""
                    print(f"  {key:30s}  HR={hr:.4f}  p={r['p_value']:.3e}  {status}")
            ci = cox_results.get("concordance_index", float("nan"))
            print(f"  Concordance index: {ci:.4f}")
            print(f"  Expected HR range: 0.80 - 0.90 for lambda_p")
        else:
            print(f"  Could not test: {cox_results.get('error')}")

        print(f"\nMann-Whitney U: high vs low mass relaxation times")
        print("-" * 50)
        if "error" not in mw_results:
            print(f"  High mass mean: {mw_results['high_mass_mean']:.2f} yr")
            print(f"  Low mass mean:  {mw_results['low_mass_mean']:.2f} yr")
            print(f"  p = {mw_results['p_value']:.4e}  "
                  f"[{'SIGNIFICANT' if mw_results['significant'] else 'not significant'}]")
        else:
            print(f"  Could not test: {mw_results.get('error')}")

        print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run full epistemic inertia analysis on OpenAlex retraction data."""
    analyzer = RetractionInertiaAnalyzer(data_dir="data")

    # Load data
    print("Loading data ...")
    data = analyzer.load_latest()

    # Prepare merged dataset
    df = analyzer.prepare_dataset(data["authors"], data["tracking"])

    if df.empty or len(df) < 5:
        print("\nInsufficient data for analysis. Run fetch_data.py first.")
        return

    # Run hypothesis tests
    spearman_results = analyzer.test_h1_2_spearman(df)
    cox_results = analyzer.test_h3_1_cox(df)
    mw_results = analyzer.test_mann_whitney(df)

    # Plots
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    analyzer.plot_kaplan_meier(df, output_dir)
    analyzer.plot_results(df, output_dir)

    # Summary
    analyzer.print_summary(spearman_results, cox_results, mw_results)


if __name__ == "__main__":
    main()
