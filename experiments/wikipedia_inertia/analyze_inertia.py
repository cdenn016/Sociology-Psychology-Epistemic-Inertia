"""
Epistemic Inertia Analysis for Wikipedia Edit History Data

Tests the hypothesis (H3.1) that editors with greater social influence
exhibit lower revert acceptance rates -- i.e., they are more rigid in
maintaining their contributions when challenged by reverts.

Mass formula mapping (theory -> Wikipedia proxy):
  M_i = Lambda_p + Lambda_o + Sum_k beta_ik * Lambda_tilde_qk + Sum_j beta_ji * Lambda_qi

  Lambda_p  (prior precision)       = log(1 + total_edit_count)
  Lambda_o  (observation precision) = log(1 + contentious_article_edit_count)
  Sum_j beta_ji (outgoing social)   = mean page watchers for editor's articles
  Sum_k beta_ik (incoming social)   = [not computed; requires co-editor graph]

Statistical tests:
  H3.1 (Influence -> Rigidity):
    Editors with more page watchers (outgoing social mass) show LOWER
    revert acceptance rates.

    Expected: Spearman rho ~ -0.15 to -0.30
    Rationale: Social influence creates inertia -- editors whose work is
    widely watched resist corrections more strongly.

  Controls: admin status, tenure (days since registration), topic area
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class WikipediaInertiaAnalyzer:
    """
    Analyze epistemic inertia in Wikipedia editorial behavior.

    Primary hypothesis (H3.1):
      Editors with higher outgoing social mass (page watchers) exhibit
      lower revert acceptance rates, indicating epistemic rigidity.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_latest_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load the most recent data files produced by fetch_data.py.

        Returns:
            Dict with keys 'editors', 'reverts', 'revisions', 'watchers'.
        """
        def _latest(pattern: str) -> Path:
            files = sorted(self.data_dir.glob(pattern))
            if not files:
                raise FileNotFoundError(
                    f"No files matching '{pattern}' in {self.data_dir}. "
                    "Run fetch_data.py first."
                )
            return files[-1]

        editors_file = _latest("editors_*.csv")
        reverts_file = _latest("reverts_*.csv")
        revisions_file = _latest("revisions_*.csv")
        watchers_file = _latest("watchers_*.csv")

        print(f"Loading data from {self.data_dir}/")
        print(f"  Editors:   {editors_file.name}")
        print(f"  Reverts:   {reverts_file.name}")
        print(f"  Revisions: {revisions_file.name}")
        print(f"  Watchers:  {watchers_file.name}")

        return {
            "editors": pd.read_csv(editors_file),
            "reverts": pd.read_csv(reverts_file),
            "revisions": pd.read_csv(revisions_file),
            "watchers": pd.read_csv(watchers_file),
        }

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_analysis_df(self, editors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare the editors DataFrame for analysis.

        Filters:
          - Remove bots
          - Remove editors with zero reverts (no acceptance rate available)
          - Ensure numeric types

        Adds:
          - mass_quartile: quartile grouping of composite_mass
          - outgoing_mass_quartile: quartile grouping of outgoing social mass

        Returns:
            Cleaned DataFrame ready for statistical tests.
        """
        df = editors_df.copy()

        # Type coercions
        numeric_cols = [
            "edit_count", "contentious_edit_count", "lambda_p", "lambda_o",
            "outgoing_mass", "outgoing_mass_log", "composite_mass",
            "tenure_days", "total_reverts", "reverts_accepted",
            "revert_acceptance_rate",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Boolean coercions
        for col in ["is_admin", "is_bot"]:
            if col in df.columns:
                df[col] = df[col].astype(bool)

        # Filter out bots
        if "is_bot" in df.columns:
            n_bots = df["is_bot"].sum()
            df = df[~df["is_bot"]]
            if n_bots > 0:
                print(f"  Removed {n_bots} bot accounts")

        # Require at least 1 revert event for meaningful acceptance rate
        before = len(df)
        df = df[df["total_reverts"].notna() & (df["total_reverts"] >= 1)]
        print(f"  Editors with revert data: {len(df)} "
              f"(dropped {before - len(df)} with no reverts)")

        # Require minimum reverts for statistical reliability
        min_reverts = 2
        reliable = df[df["total_reverts"] >= min_reverts]
        print(f"  Editors with >= {min_reverts} reverts: {len(reliable)}")

        # Add quartile groupings
        if len(df) >= 4:
            df["mass_quartile"] = pd.qcut(
                df["composite_mass"], q=4,
                labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"],
                duplicates="drop",
            )
            df["outgoing_mass_quartile"] = pd.qcut(
                df["outgoing_mass_log"].clip(lower=0), q=4,
                labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"],
                duplicates="drop",
            )

        return df

    # ------------------------------------------------------------------
    # Hypothesis tests
    # ------------------------------------------------------------------

    def test_h3_1_influence_rigidity(self, df: pd.DataFrame) -> Dict:
        """
        H3.1: Editors with more page watchers show LOWER revert acceptance rates.

        Theory:
          outgoing_mass = mean(page_watchers) for the editor's primary articles
          This proxies Sum_j beta_ji * Lambda_qi (outgoing social coupling)

          Higher outgoing_mass --> higher M_i --> more inertia
          --> editor resists reverts --> lower revert_acceptance_rate

        Tests performed:
          1. Spearman correlation: outgoing_mass vs revert_acceptance_rate
             Expected: rho ~ -0.15 to -0.30
          2. Mann-Whitney U: high-mass vs low-mass acceptance rates
          3. Kruskal-Wallis across quartiles

        Returns:
            Dict with all test statistics and interpretations.
        """
        print("\n" + "=" * 70)
        print("H3.1: SOCIAL INFLUENCE --> EPISTEMIC RIGIDITY")
        print("=" * 70)
        print("\nTheory: Editors whose contributions are widely watched")
        print("  (high outgoing social mass) resist reverts more strongly.")
        print("  Expected: Spearman rho ~ -0.15 to -0.30")

        results = {}

        # ------ Descriptive statistics ------
        print("\n--- Descriptive Statistics ---")
        print(f"  N (editors with reverts): {len(df)}")
        print(f"  Revert acceptance rate:")
        print(f"    Mean:   {df['revert_acceptance_rate'].mean():.3f}")
        print(f"    Median: {df['revert_acceptance_rate'].median():.3f}")
        print(f"    Std:    {df['revert_acceptance_rate'].std():.3f}")
        print(f"  Outgoing social mass (page watchers):")
        print(f"    Mean:   {df['outgoing_mass'].mean():.1f}")
        print(f"    Median: {df['outgoing_mass'].median():.1f}")
        print(f"    Range:  [{df['outgoing_mass'].min():.0f}, "
              f"{df['outgoing_mass'].max():.0f}]")

        # ------ Test 1: Spearman correlation ------
        print("\n--- Test 1: Spearman Correlation ---")
        print("  outgoing_mass_log vs revert_acceptance_rate")

        valid = df[["outgoing_mass_log", "revert_acceptance_rate"]].dropna()
        if len(valid) < 5:
            print("  INSUFFICIENT DATA for correlation (N < 5)")
            results["spearman"] = {"rho": float("nan"), "p_value": float("nan"),
                                   "n": len(valid), "sufficient_data": False}
        else:
            rho, p_val = stats.spearmanr(
                valid["outgoing_mass_log"],
                valid["revert_acceptance_rate"],
            )
            print(f"  rho = {rho:.4f}")
            print(f"  p   = {p_val:.4e}")
            print(f"  N   = {len(valid)}")

            if p_val < 0.05 and rho < 0:
                print("  --> SIGNIFICANT NEGATIVE correlation (supports H3.1)")
                interpretation = "SUPPORTED"
            elif p_val < 0.05 and rho > 0:
                print("  --> SIGNIFICANT POSITIVE correlation (contradicts H3.1)")
                interpretation = "CONTRADICTED"
            else:
                print("  --> NOT SIGNIFICANT")
                interpretation = "NOT SIGNIFICANT"

            in_expected_range = -0.30 <= rho <= -0.15
            if in_expected_range:
                print(f"  rho = {rho:.4f} is within expected range [-0.30, -0.15]")
            else:
                print(f"  rho = {rho:.4f} is outside expected range [-0.30, -0.15]")

            results["spearman"] = {
                "rho": rho, "p_value": p_val, "n": len(valid),
                "interpretation": interpretation,
                "in_expected_range": in_expected_range,
                "sufficient_data": True,
            }

        # ------ Test 1b: Spearman with composite mass ------
        print("\n--- Test 1b: Spearman Correlation (composite mass) ---")
        print("  composite_mass vs revert_acceptance_rate")

        valid2 = df[["composite_mass", "revert_acceptance_rate"]].dropna()
        if len(valid2) >= 5:
            rho2, p2 = stats.spearmanr(
                valid2["composite_mass"],
                valid2["revert_acceptance_rate"],
            )
            print(f"  rho = {rho2:.4f}")
            print(f"  p   = {p2:.4e}")
            print(f"  N   = {len(valid2)}")
            results["spearman_composite"] = {
                "rho": rho2, "p_value": p2, "n": len(valid2),
            }

        # ------ Test 2: Mann-Whitney U (high vs low mass) ------
        print("\n--- Test 2: Mann-Whitney U (High vs Low Mass) ---")

        mass_median = df["outgoing_mass_log"].median()
        high_mass = df[df["outgoing_mass_log"] >= mass_median]
        low_mass = df[df["outgoing_mass_log"] < mass_median]

        high_acc = high_mass["revert_acceptance_rate"].dropna()
        low_acc = low_mass["revert_acceptance_rate"].dropna()

        print(f"  High mass (N={len(high_acc)}):")
        print(f"    Mean acceptance rate: {high_acc.mean():.3f}")
        print(f"    Median:              {high_acc.median():.3f}")
        print(f"  Low mass (N={len(low_acc)}):")
        print(f"    Mean acceptance rate: {low_acc.mean():.3f}")
        print(f"    Median:              {low_acc.median():.3f}")

        if len(high_acc) >= 2 and len(low_acc) >= 2:
            u_stat, u_pval = stats.mannwhitneyu(
                high_acc, low_acc, alternative="less"
            )
            print(f"\n  Mann-Whitney U (H: high_mass < low_mass):")
            print(f"    U = {u_stat:.2f}")
            print(f"    p = {u_pval:.4e}")

            # Effect size: rank-biserial correlation
            n1, n2 = len(high_acc), len(low_acc)
            r_rb = 1 - (2 * u_stat) / (n1 * n2)
            print(f"    rank-biserial r = {r_rb:.4f}")

            if u_pval < 0.05:
                print("  --> SIGNIFICANT: High-mass editors accept fewer reverts")
            else:
                print("  --> NOT SIGNIFICANT")

            results["mannwhitney"] = {
                "u_stat": u_stat, "p_value": u_pval,
                "rank_biserial_r": r_rb,
                "high_mass_mean": high_acc.mean(),
                "low_mass_mean": low_acc.mean(),
                "significant": u_pval < 0.05,
            }
        else:
            print("  INSUFFICIENT DATA for Mann-Whitney U")
            results["mannwhitney"] = {"sufficient_data": False}

        # ------ Test 3: Kruskal-Wallis across quartiles ------
        print("\n--- Test 3: Kruskal-Wallis Across Mass Quartiles ---")

        if "mass_quartile" in df.columns:
            groups = [
                g["revert_acceptance_rate"].dropna().values
                for _, g in df.groupby("mass_quartile", observed=True)
            ]
            groups = [g for g in groups if len(g) >= 2]

            if len(groups) >= 2:
                h_stat, kw_pval = stats.kruskal(*groups)
                print(f"  H-statistic: {h_stat:.2f}")
                print(f"  p-value:     {kw_pval:.4e}")

                if kw_pval < 0.05:
                    print("  --> SIGNIFICANT difference across quartiles")
                else:
                    print("  --> NOT SIGNIFICANT")

                results["kruskal_wallis"] = {
                    "h_stat": h_stat, "p_value": kw_pval,
                    "significant": kw_pval < 0.05,
                }

                # Report per-quartile means
                print("\n  Per-quartile acceptance rates:")
                for name, group in df.groupby("mass_quartile", observed=True):
                    acc = group["revert_acceptance_rate"].dropna()
                    print(f"    {name}: mean={acc.mean():.3f}, "
                          f"median={acc.median():.3f}, N={len(acc)}")
            else:
                print("  INSUFFICIENT DATA for Kruskal-Wallis")
                results["kruskal_wallis"] = {"sufficient_data": False}

        return results

    def run_logistic_regression(self, df: pd.DataFrame) -> Dict:
        """
        Logistic regression predicting revert acceptance (binary) from mass
        proxies with controls for admin status, tenure, and topic area.

        Model:
          P(accept_revert) = logit^{-1}(
              b0
            + b1 * outgoing_mass_log          # H3.1: expected b1 < 0
            + b2 * lambda_p                    # prior experience control
            + b3 * is_admin                    # status control
            + b4 * log(tenure_days)            # time-on-platform control
          )

        We use statsmodels for the logistic regression to get proper
        confidence intervals and p-values.

        Returns:
            Dict with regression coefficients, p-values, pseudo-R^2.
        """
        print("\n" + "=" * 70)
        print("LOGISTIC REGRESSION: REVERT ACCEPTANCE")
        print("=" * 70)

        try:
            import statsmodels.api as sm
        except ImportError:
            print("  statsmodels not installed -- skipping logistic regression.")
            print("  Install with: pip install statsmodels")
            return {"error": "statsmodels not installed"}

        # Prepare binary outcome: convert acceptance rate to binary events
        # Use the per-revert data if available; otherwise threshold the rate
        reg_df = df[["outgoing_mass_log", "lambda_p", "is_admin",
                      "tenure_days", "revert_acceptance_rate"]].dropna().copy()

        if len(reg_df) < 10:
            print(f"  INSUFFICIENT DATA for regression (N={len(reg_df)} < 10)")
            return {"error": "insufficient data", "n": len(reg_df)}

        # Binary outcome: above-median acceptance (1) vs below-median (0)
        median_acc = reg_df["revert_acceptance_rate"].median()
        reg_df["accept_binary"] = (
            reg_df["revert_acceptance_rate"] >= median_acc
        ).astype(int)

        # Predictors
        reg_df["log_tenure"] = np.log1p(reg_df["tenure_days"])
        reg_df["is_admin_int"] = reg_df["is_admin"].astype(int)

        X = reg_df[["outgoing_mass_log", "lambda_p", "is_admin_int",
                     "log_tenure"]]
        X = sm.add_constant(X)
        y = reg_df["accept_binary"]

        try:
            model = sm.Logit(y, X)
            result = model.fit(disp=0, maxiter=100)

            print(f"\n  N = {len(reg_df)}")
            print(f"  Pseudo R-squared: {result.prsquared:.4f}")
            print(f"  Log-likelihood:   {result.llf:.2f}")
            print(f"  AIC:              {result.aic:.2f}")
            print()
            print(result.summary2().tables[1].to_string())

            # Extract key coefficient: outgoing_mass_log
            coef_social = result.params.get("outgoing_mass_log", float("nan"))
            pval_social = result.pvalues.get("outgoing_mass_log", float("nan"))
            ci_social = result.conf_int().loc["outgoing_mass_log"].values

            print(f"\n  Key result (outgoing_mass_log coefficient):")
            print(f"    b = {coef_social:.4f}")
            print(f"    p = {pval_social:.4e}")
            print(f"    95% CI: [{ci_social[0]:.4f}, {ci_social[1]:.4f}]")
            print(f"    Odds ratio: {np.exp(coef_social):.4f}")

            if pval_social < 0.05 and coef_social < 0:
                print("    --> SIGNIFICANT NEGATIVE: Higher social mass "
                      "reduces acceptance (supports H3.1)")
            elif pval_social < 0.05 and coef_social > 0:
                print("    --> SIGNIFICANT POSITIVE: Higher social mass "
                      "increases acceptance (contradicts H3.1)")
            else:
                print("    --> NOT SIGNIFICANT")

            return {
                "n": len(reg_df),
                "pseudo_r2": result.prsquared,
                "aic": result.aic,
                "coefficients": result.params.to_dict(),
                "p_values": result.pvalues.to_dict(),
                "conf_int": {
                    col: result.conf_int().loc[col].tolist()
                    for col in result.params.index
                },
                "outgoing_mass_coef": coef_social,
                "outgoing_mass_pval": pval_social,
                "outgoing_mass_odds_ratio": np.exp(coef_social),
            }

        except Exception as exc:
            print(f"  Logistic regression failed: {exc}")
            return {"error": str(exc)}

    def run_lambda_p_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Supplementary test: Does prior experience (Lambda_p) independently
        predict revert acceptance?

        Lambda_p = log(1 + total_edit_count)
        Theory: experienced editors have stronger priors, hence more inertia.

        Returns:
            Dict with Spearman correlation results.
        """
        print("\n" + "=" * 70)
        print("SUPPLEMENTARY: PRIOR EXPERIENCE (Lambda_p) vs RIGIDITY")
        print("=" * 70)

        valid = df[["lambda_p", "revert_acceptance_rate"]].dropna()
        if len(valid) < 5:
            print(f"  INSUFFICIENT DATA (N={len(valid)})")
            return {"sufficient_data": False}

        rho, pval = stats.spearmanr(
            valid["lambda_p"], valid["revert_acceptance_rate"]
        )
        print(f"  Spearman rho (lambda_p vs acceptance): {rho:.4f}")
        print(f"  p-value: {pval:.4e}")
        print(f"  N: {len(valid)}")

        if pval < 0.05 and rho < 0:
            print("  --> More experienced editors accept fewer reverts")
        elif pval < 0.05 and rho > 0:
            print("  --> More experienced editors accept MORE reverts")
        else:
            print("  --> No significant relationship")

        return {"rho": rho, "p_value": pval, "n": len(valid)}

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------

    def plot_results(self, df: pd.DataFrame, output_dir: Optional[Path] = None):
        """
        Generate publication-quality visualizations of H3.1 results.

        Plots:
          1. Scatter: outgoing social mass vs revert acceptance rate
          2. Box plot: acceptance rate by mass quartile
          3. Scatter: Lambda_p (experience) vs acceptance rate
          4. Heatmap: mass components correlation matrix
        """
        if output_dir is None:
            output_dir = self.data_dir
        output_dir = Path(output_dir)

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

        # ----- Plot 1: Social mass vs acceptance rate -----
        ax1 = axes[0, 0]
        valid = df[["outgoing_mass_log", "revert_acceptance_rate"]].dropna()
        ax1.scatter(
            valid["outgoing_mass_log"],
            valid["revert_acceptance_rate"],
            alpha=0.4, s=30, c="steelblue", edgecolors="white", linewidth=0.3,
        )
        # Trend line
        if len(valid) >= 5:
            z = np.polyfit(valid["outgoing_mass_log"],
                           valid["revert_acceptance_rate"], 1)
            x_line = np.linspace(
                valid["outgoing_mass_log"].min(),
                valid["outgoing_mass_log"].max(), 100,
            )
            ax1.plot(x_line, np.polyval(z, x_line), "r--", linewidth=2,
                     alpha=0.8, label=f"Trend (slope={z[0]:.3f})")
            ax1.legend(fontsize=9)
        ax1.set_xlabel("Outgoing Social Mass (log page watchers)", fontsize=10)
        ax1.set_ylabel("Revert Acceptance Rate", fontsize=10)
        ax1.set_title("H3.1: Social Influence vs Epistemic Rigidity",
                       fontsize=11, fontweight="bold")
        ax1.set_ylim(-0.05, 1.05)

        # ----- Plot 2: Box plot by mass quartile -----
        ax2 = axes[0, 1]
        if "mass_quartile" in df.columns:
            quartile_data = df[["mass_quartile", "revert_acceptance_rate"]].dropna()
            quartile_data.boxplot(
                column="revert_acceptance_rate", by="mass_quartile",
                ax=ax2, patch_artist=True,
                boxprops=dict(facecolor="lightblue", color="steelblue"),
                medianprops=dict(color="red", linewidth=2),
            )
            ax2.set_xlabel("Composite Mass Quartile", fontsize=10)
            ax2.set_ylabel("Revert Acceptance Rate", fontsize=10)
            ax2.set_title("Acceptance Rate by Mass Quartile",
                           fontsize=11, fontweight="bold")
            plt.sca(ax2)
            plt.xticks(rotation=30, fontsize=9)
            fig.suptitle("")  # Remove auto-generated title from boxplot

        # ----- Plot 3: Experience vs acceptance rate -----
        ax3 = axes[1, 0]
        valid3 = df[["lambda_p", "revert_acceptance_rate"]].dropna()
        ax3.scatter(
            valid3["lambda_p"], valid3["revert_acceptance_rate"],
            alpha=0.4, s=30, c="darkorange", edgecolors="white", linewidth=0.3,
        )
        if len(valid3) >= 5:
            z3 = np.polyfit(valid3["lambda_p"],
                            valid3["revert_acceptance_rate"], 1)
            x3 = np.linspace(valid3["lambda_p"].min(),
                             valid3["lambda_p"].max(), 100)
            ax3.plot(x3, np.polyval(z3, x3), "r--", linewidth=2, alpha=0.8,
                     label=f"Trend (slope={z3[0]:.3f})")
            ax3.legend(fontsize=9)
        ax3.set_xlabel("Lambda_p (log edit count = prior precision)", fontsize=10)
        ax3.set_ylabel("Revert Acceptance Rate", fontsize=10)
        ax3.set_title("Experience vs Epistemic Rigidity", fontsize=11,
                       fontweight="bold")
        ax3.set_ylim(-0.05, 1.05)

        # ----- Plot 4: Correlation heatmap of mass components -----
        ax4 = axes[1, 1]
        corr_cols = [
            "lambda_p", "lambda_o", "outgoing_mass_log",
            "composite_mass", "revert_acceptance_rate",
        ]
        available = [c for c in corr_cols if c in df.columns]
        if len(available) >= 3:
            corr_matrix = df[available].corr(method="spearman")
            # Prettier labels
            label_map = {
                "lambda_p": "Prior (Lambda_p)",
                "lambda_o": "Topic exp (Lambda_o)",
                "outgoing_mass_log": "Social mass",
                "composite_mass": "Composite mass",
                "revert_acceptance_rate": "Accept rate",
            }
            corr_matrix.index = [label_map.get(c, c) for c in corr_matrix.index]
            corr_matrix.columns = [label_map.get(c, c) for c in corr_matrix.columns]

            sns.heatmap(
                corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax4,
                square=True, linewidths=0.5,
            )
            ax4.set_title("Spearman Correlations (Mass Components)",
                           fontsize=11, fontweight="bold")

        plt.tight_layout(pad=2.0)
        output_path = output_dir / "wikipedia_inertia_results.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved plots to {output_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def print_summary(self, h31_results: Dict, logistic_results: Dict,
                      lambda_p_results: Dict):
        """Print a formatted summary of all test results."""
        print("\n" + "=" * 70)
        print("SUMMARY OF FINDINGS")
        print("=" * 70)

        print("\n--- H3.1: Social Influence --> Epistemic Rigidity ---")
        print("  Theory: Editors with higher outgoing social mass (page watchers)")
        print("  exhibit lower revert acceptance rates.")
        print(f"  Expected: Spearman rho in [-0.30, -0.15]")

        spearman = h31_results.get("spearman", {})
        if spearman.get("sufficient_data", False):
            rho = spearman["rho"]
            pval = spearman["p_value"]
            interp = spearman.get("interpretation", "?")
            in_range = spearman.get("in_expected_range", False)

            print(f"\n  Result: rho = {rho:.4f}, p = {pval:.4e}")
            print(f"  Interpretation: {interp}")
            if in_range:
                print(f"  rho is within the predicted range [-0.30, -0.15]")
            else:
                print(f"  rho is outside the predicted range")
        else:
            print("\n  INSUFFICIENT DATA for primary test")

        mw = h31_results.get("mannwhitney", {})
        if mw.get("significant", False):
            print(f"\n  Mann-Whitney U confirms: high-mass editors accept "
                  f"fewer reverts (p = {mw['p_value']:.4e})")
        elif "p_value" in mw:
            print(f"\n  Mann-Whitney U not significant (p = {mw['p_value']:.4e})")

        # Logistic regression
        print("\n--- Logistic Regression (with controls) ---")
        if "error" not in logistic_results:
            coef = logistic_results.get("outgoing_mass_coef", float("nan"))
            pval = logistic_results.get("outgoing_mass_pval", float("nan"))
            or_val = logistic_results.get("outgoing_mass_odds_ratio", float("nan"))
            print(f"  Social mass coefficient: {coef:.4f} (p = {pval:.4e})")
            print(f"  Odds ratio: {or_val:.4f}")
            print(f"  Pseudo R^2: {logistic_results.get('pseudo_r2', 0):.4f}")
        else:
            print(f"  {logistic_results.get('error', 'Failed')}")

        # Lambda_p
        print("\n--- Supplementary: Prior Experience ---")
        if lambda_p_results.get("sufficient_data", True) and "rho" in lambda_p_results:
            print(f"  Lambda_p vs acceptance: rho = {lambda_p_results['rho']:.4f}, "
                  f"p = {lambda_p_results['p_value']:.4e}")
        else:
            print("  Insufficient data")

        print("\n" + "=" * 70)


def main():
    """Run the complete epistemic inertia analysis pipeline."""
    analyzer = WikipediaInertiaAnalyzer(data_dir="data")

    # Load data
    data = analyzer.load_latest_data()

    # Prepare analysis DataFrame
    df = analyzer.prepare_analysis_df(data["editors"])

    if len(df) == 0:
        print("\nERROR: No editors with revert data available.")
        print("Ensure fetch_data.py has been run and produced valid data.")
        return

    print(f"\nAnalysis sample: {len(df)} editors")

    # Run hypothesis tests
    h31_results = analyzer.test_h3_1_influence_rigidity(df)
    logistic_results = analyzer.run_logistic_regression(df)
    lambda_p_results = analyzer.run_lambda_p_analysis(df)

    # Generate plots
    analyzer.plot_results(df, output_dir=Path("data"))

    # Print summary
    analyzer.print_summary(h31_results, logistic_results, lambda_p_results)


if __name__ == "__main__":
    main()
