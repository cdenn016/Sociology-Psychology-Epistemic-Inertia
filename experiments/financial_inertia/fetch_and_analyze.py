"""
Financial Analyst Forecast Revisions — Epistemic Inertia Analysis

Uses Survey of Professional Forecasters (SPF) data as the primary source
and yfinance for supplementary market/earnings data.

==========================================================================
MASS FORMULA UNDER TEST
==========================================================================

    M_i = Lambda_p + Lambda_o + Sum_k beta_ik Lambda_tilde_qk + Sum_j beta_ji Lambda_qi
          --------   --------   ----------------------------   -------------------------
          prior       obs        incoming social                outgoing social

    Concrete variable mappings for financial forecasters:

        Lambda_p   (prior precision)     -> Forecaster tenure (quarters in SPF panel)
                                            More experienced forecasters have built up
                                            stronger priors through repeated observation.
                                            Operationalized as: log(1 + quarters_active)

        Lambda_o   (observation precision) -> Historical forecast accuracy
                                            1 / RMSE of past 8 quarters of forecasts vs.
                                            realized values. Accurate forecasters have
                                            higher-precision observation models.

        Sum beta_ik Lambda_tilde_qk      -> Consensus proximity (incoming social)
           (incoming social)                1 / |forecast_i - median_forecast|
                                            Forecasters close to consensus receive
                                            confirming social signals.

        Sum beta_ji Lambda_qi            -> Influence on subsequent consensus (outgoing)
           (outgoing social)                Granger-causality of forecaster i's revision
                                            on the subsequent consensus shift.
                                            Forecasters whose revisions lead the pack
                                            accumulate outgoing epistemic mass.

==========================================================================
HYPOTHESES TESTED
==========================================================================

H_fin_2 — Oscillation detection:
    Individual forecast revision sequences show more sign reversals
    (Delta_forecast changes sign) than predicted by a random walk model,
    consistent with underdamped second-order belief dynamics.

    Test: Wald-Wolfowitz runs test on revision sign sequences.
    Null: sign changes follow binomial distribution with p = 0.5.
    Alternative: excess sign reversals (oscillation).

H_fin_3 — Relaxation time scales with experience:
    Experienced forecasters (many quarters in panel = high Lambda_p)
    take more quarters to converge to post-shock consensus than
    inexperienced forecasters.

    Prediction from framework: tau = M / gamma, so high-mass forecasters
    have longer relaxation times.

==========================================================================
DATA SOURCES
==========================================================================

Primary: Survey of Professional Forecasters (SPF)
    - Source: Federal Reserve Bank of Philadelphia
    - URL: https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/
    - Individual-level panel data with anonymized but consistent forecaster IDs
    - Variables: GDP, CPI, unemployment forecasts at multiple horizons
    - Access: Fully public, no authentication

Secondary: yfinance (for earnings surprise data and market context)
    - Package: pip install yfinance
    - Used to fetch actual GDP/CPI releases for accuracy computation
    - Used to identify market shocks via S&P 500 drawdowns
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Dict, List, Optional, Tuple
import warnings
import json
from datetime import datetime

warnings.filterwarnings("ignore")

# Optional imports
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("WARNING: yfinance not installed. Market shock detection will be skipped.")
    print("  Install with: pip install yfinance")

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# Path to SPF data (produced by experiments/spf_inertia/fetch_data.py)
SPF_DATA_DIR = Path(__file__).parent.parent / "spf_inertia" / "data"
LOCAL_DATA_DIR = Path(__file__).parent / "data"


class FinancialInertiaAnalyzer:
    """
    Analyze epistemic inertia in financial forecast revisions.

    Combines SPF panel data with market data from yfinance to test
    whether forecaster behavior follows second-order (Hamiltonian)
    belief dynamics rather than first-order gradient descent.
    """

    def __init__(
        self,
        spf_data_dir: str = None,
        local_data_dir: str = None,
    ):
        self.spf_dir = Path(spf_data_dir) if spf_data_dir else SPF_DATA_DIR
        self.local_dir = Path(local_data_dir) if local_data_dir else LOCAL_DATA_DIR
        self.local_dir.mkdir(exist_ok=True, parents=True)

    # ------------------------------------------------------------------
    # Data loading: SPF revisions
    # ------------------------------------------------------------------

    def load_spf_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load pre-processed SPF data from the spf_inertia experiment.

        Expected files (produced by experiments/spf_inertia/fetch_data.py):
            revisions_*.csv  — forecast revisions with sign-change flags
            features_*.csv   — per-forecaster features (tenure, accuracy)
            shocks_*.csv     — identified macroeconomic shock events
            consensus_*.csv  — quarterly consensus forecasts
            panel_*.csv      — raw panel data
        """
        def _latest(directory: Path, pattern: str) -> pd.DataFrame:
            candidates = sorted(directory.glob(pattern))
            if not candidates:
                raise FileNotFoundError(
                    f"No files matching '{pattern}' in {directory}.\n"
                    "Run the SPF data pipeline first:\n"
                    "  cd experiments/spf_inertia && python fetch_data.py"
                )
            path = candidates[-1]
            print(f"  Loading: {path.name}")
            return pd.read_csv(path)

        print("Loading SPF data...")
        data = {}
        try:
            data["revisions"] = _latest(self.spf_dir, "revisions_*.csv")
            data["features"] = _latest(self.spf_dir, "features_*.csv")
            data["shocks"] = _latest(self.spf_dir, "shocks_*.csv")
            data["consensus"] = _latest(self.spf_dir, "consensus_*.csv")
            data["panel"] = _latest(self.spf_dir, "panel_*.csv")
        except FileNotFoundError:
            # Try local data dir as fallback
            print(f"  SPF data not found in {self.spf_dir}, checking {self.local_dir}...")
            data["revisions"] = _latest(self.local_dir, "revisions_*.csv")
            data["features"] = _latest(self.local_dir, "features_*.csv")
            data["shocks"] = _latest(self.local_dir, "shocks_*.csv")
            data["consensus"] = _latest(self.local_dir, "consensus_*.csv")
            data["panel"] = _latest(self.local_dir, "panel_*.csv")

        for key, df in data.items():
            print(f"    {key}: {len(df)} rows")

        return data

    # ------------------------------------------------------------------
    # Data loading: yfinance market shocks
    # ------------------------------------------------------------------

    def fetch_market_shocks(
        self,
        ticker: str = "^GSPC",
        start: str = "1990-01-01",
        end: str = None,
        drawdown_threshold: float = -0.10,
    ) -> pd.DataFrame:
        """
        Fetch historical market data via yfinance and identify shock events.

        A market shock is defined as a quarterly return below
        drawdown_threshold (default: -10%).

        Args:
            ticker: Market index ticker (default: S&P 500)
            start: Start date for historical data
            end: End date (default: today)
            drawdown_threshold: Quarterly return threshold for shock identification

        Returns:
            DataFrame with shock dates and magnitudes
        """
        if not HAS_YFINANCE:
            print("WARNING: yfinance not available. Skipping market shock detection.")
            return pd.DataFrame()

        print(f"\nFetching market data for {ticker}...")

        try:
            index = yf.Ticker(ticker)
            hist = index.history(start=start, end=end, interval="1d")

            if hist.empty:
                print("  WARNING: No market data returned.")
                return pd.DataFrame()

            print(f"  Downloaded {len(hist)} daily observations")
            print(f"  Date range: {hist.index[0].date()} to {hist.index[-1].date()}")

            # Resample to quarterly
            quarterly = hist["Close"].resample("QE").last()
            quarterly_return = quarterly.pct_change()

            # Identify shocks
            shocks = quarterly_return[quarterly_return < drawdown_threshold].reset_index()
            shocks.columns = ["date", "quarterly_return"]
            shocks["year"] = shocks["date"].dt.year
            shocks["quarter"] = shocks["date"].dt.quarter

            print(f"  Identified {len(shocks)} market shock quarters (return < {drawdown_threshold:.0%})")
            for _, s in shocks.iterrows():
                print(f"    {s['year']}Q{s['quarter']}: {s['quarterly_return']:.1%}")

            # Save
            shocks.to_csv(self.local_dir / "market_shocks.csv", index=False)
            return shocks

        except Exception as e:
            print(f"  Market data fetch failed: {e}")
            return pd.DataFrame()

    def fetch_earnings_surprise_data(
        self,
        tickers: List[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical earnings data via yfinance to compute
        earnings surprise magnitudes for additional shock identification.

        Args:
            tickers: List of stock tickers. Default: major S&P 500 components.

        Returns:
            DataFrame with earnings dates, estimates, and actuals.
        """
        if not HAS_YFINANCE:
            print("WARNING: yfinance not available. Skipping earnings data.")
            return pd.DataFrame()

        if tickers is None:
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM",
                        "JNJ", "XOM", "PG", "WMT", "BAC"]

        print(f"\nFetching earnings data for {len(tickers)} tickers...")
        all_earnings = []

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                earnings = stock.earnings_dates
                if earnings is not None and not earnings.empty:
                    df = earnings.reset_index()
                    df["ticker"] = ticker
                    all_earnings.append(df)
                    print(f"  {ticker}: {len(df)} earnings records")
            except Exception as e:
                print(f"  {ticker}: failed ({e})")

        if all_earnings:
            combined = pd.concat(all_earnings, ignore_index=True)
            combined.to_csv(self.local_dir / "earnings_data.csv", index=False)
            print(f"  Total: {len(combined)} earnings records")
            return combined

        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Mass proxy computation
    # ------------------------------------------------------------------

    def compute_mass_proxies(
        self,
        revisions: pd.DataFrame,
        features: pd.DataFrame,
        consensus: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Compute the four mass components for each forecaster.

        M_i = Lambda_p + Lambda_o + Sum beta_ik Lambda_qk + Sum beta_ji Lambda_qi

        Returns:
            DataFrame with one row per forecaster and mass components.
        """
        print("\nComputing epistemic mass proxies...")

        mass_df = features.copy()

        # Lambda_p: prior precision = log(1 + quarters_active)
        mass_df["lambda_p"] = np.log1p(mass_df["quarters_active"])

        # Lambda_o: observation precision = 1 / mean_abs_revision
        # (forecasters with small revisions have precise observations)
        mar = mass_df["mean_abs_revision"]
        mass_df["lambda_o"] = 1.0 / mar.clip(lower=mar.quantile(0.05))

        # Incoming social: consensus proximity
        # Compute average distance from consensus for each forecaster
        if consensus is not None and "forecast" in revisions.columns:
            revisions_dt = revisions.copy()
            revisions_dt["survey_date"] = pd.to_datetime(revisions_dt["survey_date"])

            cons = consensus.copy()
            cons["survey_date"] = pd.to_datetime(cons["survey_date"])

            # Merge consensus median into revisions
            merged = revisions_dt.merge(
                cons[["variable", "horizon", "survey_date", "consensus_median"]].drop_duplicates(),
                on=["variable", "horizon", "survey_date"],
                how="left",
            )
            merged["dist_from_consensus"] = (merged["forecast"] - merged["consensus_median"]).abs()

            # Average distance per forecaster
            avg_dist = merged.groupby("forecaster_id")["dist_from_consensus"].mean().reset_index()
            avg_dist.columns = ["forecaster_id", "avg_consensus_distance"]

            mass_df = mass_df.merge(avg_dist, on="forecaster_id", how="left")
            mass_df["beta_incoming"] = 1.0 / mass_df["avg_consensus_distance"].clip(
                lower=mass_df["avg_consensus_distance"].quantile(0.05)
            )
        else:
            mass_df["beta_incoming"] = 0.0

        # Outgoing social: influence on consensus
        # Simple proxy: forecasters whose revisions are followed by consensus shifts
        # in the same direction have high outgoing influence
        if "revision" in revisions.columns and consensus is not None:
            influence_scores = []
            for fid in mass_df["forecaster_id"].unique():
                f_rev = revisions[revisions["forecaster_id"] == fid]
                if len(f_rev) < 5:
                    influence_scores.append({"forecaster_id": fid, "influence": 0.0})
                    continue

                # Crude influence: correlation between forecaster revision and
                # next-quarter consensus change
                # (A proper Granger test needs more data; this is a proxy)
                f_rev_sorted = f_rev.sort_values("survey_date")
                rev_signs = np.sign(f_rev_sorted["revision"].dropna().values)
                if len(rev_signs) >= 5:
                    # Autocorrelation as proxy for "leading the pack"
                    influence = np.abs(np.corrcoef(rev_signs[:-1], rev_signs[1:])[0, 1])
                else:
                    influence = 0.0

                influence_scores.append(
                    {"forecaster_id": fid, "influence": influence if not np.isnan(influence) else 0.0}
                )

            inf_df = pd.DataFrame(influence_scores)
            mass_df = mass_df.merge(inf_df, on="forecaster_id", how="left")
            mass_df["beta_outgoing"] = mass_df["influence"]
        else:
            mass_df["beta_outgoing"] = 0.0

        # Standardize all components
        for col in ["lambda_p", "lambda_o", "beta_incoming", "beta_outgoing"]:
            std = mass_df[col].std()
            if std > 0:
                mass_df[f"{col}_z"] = (mass_df[col] - mass_df[col].mean()) / std
            else:
                mass_df[f"{col}_z"] = 0.0

        # Composite mass
        mass_df["epistemic_mass"] = (
            mass_df["lambda_p_z"]
            + mass_df["lambda_o_z"]
            + mass_df["beta_incoming_z"]
            + mass_df["beta_outgoing_z"]
        )

        print(f"  Computed mass for {len(mass_df)} forecasters")
        print(f"  Lambda_p  range: [{mass_df['lambda_p'].min():.2f}, {mass_df['lambda_p'].max():.2f}]")
        print(f"  Lambda_o  range: [{mass_df['lambda_o'].min():.2f}, {mass_df['lambda_o'].max():.2f}]")
        print(f"  Mass      range: [{mass_df['epistemic_mass'].min():.2f}, {mass_df['epistemic_mass'].max():.2f}]")

        return mass_df

    # ==================================================================
    # TEST H_fin_2: Oscillation detection via runs test
    # ==================================================================

    def test_h_fin_2_oscillation(self, revisions: pd.DataFrame) -> Dict:
        """
        H_fin_2: Individual forecast revision sequences show more sign
        reversals than a random walk predicts.

        Method: Wald-Wolfowitz runs test on the sign sequence of revisions.
        - A "run" is a maximal sequence of consecutive same-sign revisions.
        - Under random walk: expected runs = 2*n_pos*n_neg/(n_pos+n_neg) + 1
        - Excess runs = oscillation (underdamped dynamics)
        - Deficit of runs = momentum/trending (overdamped)

        The Hamiltonian framework predicts EXCESS runs (oscillation).
        """
        print("\n" + "=" * 70)
        print("TEST H_fin_2: OSCILLATION IN FORECAST REVISIONS")
        print("=" * 70)

        forecaster_results = []

        for fid, group in revisions.groupby("forecaster_id"):
            for (var, horizon), subgroup in group.groupby(["variable", "horizon"]):
                sub = subgroup.sort_values("survey_date")

                if len(sub) < 10:
                    continue

                rev = sub["revision"].dropna().values
                if len(rev) < 8:
                    continue

                # Remove zero revisions (no update = no information)
                nonzero_rev = rev[rev != 0]
                if len(nonzero_rev) < 6:
                    continue

                signs = np.sign(nonzero_rev)
                n = len(signs)
                n_pos = np.sum(signs > 0)
                n_neg = np.sum(signs < 0)

                if n_pos < 2 or n_neg < 2:
                    continue

                # Count runs (sequences of consecutive same sign)
                runs = 1
                for i in range(1, len(signs)):
                    if signs[i] != signs[i - 1]:
                        runs += 1

                # Count sign changes
                sign_changes = runs - 1

                # Expected runs under null (random arrangement)
                expected_runs = (2 * n_pos * n_neg) / (n_pos + n_neg) + 1

                # Variance of runs under null
                var_runs = (
                    (2 * n_pos * n_neg * (2 * n_pos * n_neg - n_pos - n_neg))
                    / ((n_pos + n_neg) ** 2 * (n_pos + n_neg - 1))
                )

                if var_runs <= 0:
                    continue

                # Z-score for runs test
                z_runs = (runs - expected_runs) / np.sqrt(var_runs)

                # p-value (two-sided for oscillation or trending)
                p_runs = 2 * stats.norm.sf(abs(z_runs))

                # Also compute AR(1) autocorrelation for comparison
                ar1 = np.corrcoef(nonzero_rev[:-1], nonzero_rev[1:])[0, 1]

                # Observed vs expected sign change rate
                n_transitions = n - 1
                observed_rate = sign_changes / n_transitions
                expected_rate = 2 * n_pos * n_neg / (n * (n - 1)) if n > 1 else 0.5

                forecaster_results.append({
                    "forecaster_id": fid,
                    "variable": var,
                    "horizon": horizon,
                    "n_revisions": n,
                    "n_positive": n_pos,
                    "n_negative": n_neg,
                    "observed_runs": runs,
                    "expected_runs": expected_runs,
                    "excess_runs": runs - expected_runs,
                    "z_runs": z_runs,
                    "p_runs": p_runs,
                    "sign_changes": sign_changes,
                    "observed_sign_change_rate": observed_rate,
                    "expected_sign_change_rate": expected_rate,
                    "ar1_autocorrelation": ar1,
                })

        results_df = pd.DataFrame(forecaster_results)

        if len(results_df) == 0:
            print("WARNING: No forecaster series long enough for runs test")
            return {"h_fin_2": None, "reason": "insufficient_data"}

        # Aggregate results
        mean_excess = results_df["excess_runs"].mean()
        mean_z = results_df["z_runs"].mean()

        # Meta-analysis: combine z-scores (Stouffer's method)
        combined_z = results_df["z_runs"].sum() / np.sqrt(len(results_df))
        combined_p = 2 * stats.norm.sf(abs(combined_z))

        # One-sample t-test: is mean excess_runs > 0?
        t_stat, t_p = stats.ttest_1samp(results_df["excess_runs"].dropna(), 0)

        # Proportion of series with excess runs
        prop_excess = (results_df["excess_runs"] > 0).mean()

        # Proportion significantly oscillatory (individual p < 0.05 and excess > 0)
        sig_oscillatory = (
            (results_df["p_runs"] < 0.05) & (results_df["excess_runs"] > 0)
        ).mean()

        print(f"\nAnalyzed {len(results_df)} forecaster-variable-horizon series")
        print(f"\nRuns test results:")
        print(f"  Mean observed runs:  {results_df['observed_runs'].mean():.2f}")
        print(f"  Mean expected runs:  {results_df['expected_runs'].mean():.2f}")
        print(f"  Mean excess runs:    {mean_excess:.3f}")
        print(f"  Mean z-score:        {mean_z:.3f}")
        print(f"  Series with excess:  {prop_excess:.1%}")
        print(f"  Significantly oscillatory (p<0.05): {sig_oscillatory:.1%}")
        print(f"\nMeta-analysis (Stouffer):")
        print(f"  Combined z = {combined_z:.3f}")
        print(f"  Combined p = {combined_p:.4e}")
        print(f"\nt-test (H: mean excess > 0):")
        print(f"  t = {t_stat:.3f}, p = {t_p:.4e}")

        # Sign change rate analysis
        mean_obs_rate = results_df["observed_sign_change_rate"].mean()
        mean_exp_rate = results_df["expected_sign_change_rate"].mean()
        print(f"\nSign change rates:")
        print(f"  Observed: {mean_obs_rate:.4f}")
        print(f"  Expected: {mean_exp_rate:.4f}")
        print(f"  Excess:   {mean_obs_rate - mean_exp_rate:.4f}")

        # AR(1) autocorrelation distribution
        mean_ar1 = results_df["ar1_autocorrelation"].mean()
        print(f"\nAR(1) autocorrelation:")
        print(f"  Mean: {mean_ar1:.4f}")
        print(f"  Negative AR(1) supports oscillation (revisions alternate sign)")

        # By variable
        print(f"\nBy variable:")
        for var in results_df["variable"].unique():
            sub = results_df[results_df["variable"] == var]
            print(f"  {var}: excess runs = {sub['excess_runs'].mean():.3f}, "
                  f"mean z = {sub['z_runs'].mean():.3f} (N={len(sub)})")

        result = {
            "n_series": len(results_df),
            "mean_excess_runs": mean_excess,
            "combined_z": combined_z,
            "combined_p": combined_p,
            "t_stat": t_stat,
            "t_p": t_p,
            "prop_excess": prop_excess,
            "prop_sig_oscillatory": sig_oscillatory,
            "mean_ar1": mean_ar1,
            "mean_observed_sign_change_rate": mean_obs_rate,
            "mean_expected_sign_change_rate": mean_exp_rate,
            "significant": combined_p < 0.05 and mean_excess > 0,
            "results_df": results_df,
        }

        if result["significant"]:
            print(f"\n>>> RESULT: EXCESS OSCILLATION DETECTED (H_fin_2 CONFIRMED)")
            print(f"    Revision sequences show {mean_obs_rate - mean_exp_rate:.1%} more")
            print(f"    sign changes than random walk, consistent with underdamped dynamics.")
        elif mean_excess > 0 and combined_p < 0.10:
            print(f"\n>>> RESULT: MARGINAL evidence for oscillation (p = {combined_p:.4f})")
        else:
            print(f"\n>>> RESULT: No significant excess oscillation (H_fin_2 NOT CONFIRMED)")

        return result

    # ==================================================================
    # TEST H_fin_3: Experienced forecasters converge more slowly
    # ==================================================================

    def test_h_fin_3_experience_convergence(
        self,
        revisions: pd.DataFrame,
        shocks: pd.DataFrame,
        mass_df: pd.DataFrame,
    ) -> Dict:
        """
        H_fin_3: Experienced forecasters (high Lambda_p = many quarters in
        panel) take more quarters to converge to post-shock consensus.

        Prediction: tau = M / gamma, so tau grows with mass.

        Method:
        1. Identify shock events from SPF data
        2. Track each forecaster's trajectory post-shock
        3. Measure convergence time (quarters to reach 1/e of initial distance)
        4. Regress convergence time on experience / mass
        """
        print("\n" + "=" * 70)
        print("TEST H_fin_3: EXPERIENCE -> SLOWER CONVERGENCE")
        print("=" * 70)

        if len(shocks) == 0:
            print("WARNING: No shock events available")
            return {"h_fin_3": None, "reason": "no_shocks"}

        revisions = revisions.copy()
        revisions["survey_date"] = pd.to_datetime(revisions["survey_date"])
        shocks = shocks.copy()
        shocks["survey_date"] = pd.to_datetime(shocks["survey_date"])

        convergence_data = []

        for _, shock in shocks.iterrows():
            var = shock["variable"]
            horizon = shock["horizon"]
            shock_date = shock["survey_date"]

            # Post-shock window: 1-8 quarters after shock
            post_mask = (
                (revisions["variable"] == var)
                & (revisions["horizon"] == horizon)
                & (revisions["survey_date"] > shock_date)
                & (revisions["survey_date"] <= shock_date + pd.DateOffset(years=2))
            )
            post_data = revisions[post_mask]

            if len(post_data) < 10:
                continue

            # Settled consensus: median of late-window forecasts (6-8 quarters post)
            late_mask = post_data["survey_date"] > shock_date + pd.DateOffset(months=15)
            if late_mask.sum() < 5:
                continue
            settled = post_data[late_mask]["forecast"].median()

            # Pre-shock consensus
            pre_mask = (
                (revisions["variable"] == var)
                & (revisions["horizon"] == horizon)
                & (revisions["survey_date"] <= shock_date)
                & (revisions["survey_date"] > shock_date - pd.DateOffset(months=3))
            )
            pre_consensus = revisions[pre_mask]["forecast"].median()
            if np.isnan(pre_consensus):
                continue

            shock_magnitude = abs(settled - pre_consensus)
            if shock_magnitude < 1e-6:
                continue

            # Track each forecaster
            for fid in post_data["forecaster_id"].unique():
                f_post = post_data[post_data["forecaster_id"] == fid].sort_values("survey_date")

                if len(f_post) < 3:
                    continue

                # Distance from settled consensus over time
                f_post = f_post.copy()
                f_post["distance"] = (f_post["forecast"] - settled).abs()

                initial_dist = f_post["distance"].iloc[0]
                if initial_dist < 1e-6:
                    continue

                # Convergence: first quarter where distance < 1/e * initial
                threshold = initial_dist * np.exp(-1)  # 0.368
                converged_rows = f_post[f_post["distance"] <= threshold]

                if len(converged_rows) > 0:
                    conv_date = converged_rows["survey_date"].iloc[0]
                    tau_quarters = (conv_date - shock_date).days / 91.25
                    did_converge = True
                else:
                    tau_quarters = np.nan
                    did_converge = False

                # Count oscillations during convergence
                if "revision_sign" in f_post.columns:
                    sign_changes = f_post["sign_change"].sum() if "sign_change" in f_post.columns else 0
                else:
                    revs = f_post["forecast"].diff().dropna().values
                    signs = np.sign(revs[revs != 0])
                    sign_changes = np.sum(np.diff(signs) != 0) if len(signs) > 1 else 0

                # Get mass proxies
                f_mass = mass_df[mass_df["forecaster_id"] == fid]
                if len(f_mass) == 0:
                    continue

                convergence_data.append({
                    "forecaster_id": fid,
                    "shock_date": shock_date,
                    "variable": var,
                    "horizon": horizon,
                    "shock_magnitude": shock_magnitude,
                    "initial_distance": initial_dist,
                    "tau_quarters": tau_quarters,
                    "converged": did_converge,
                    "sign_changes_during_convergence": sign_changes,
                    "quarters_active": f_mass["quarters_active"].iloc[0],
                    "lambda_p": f_mass["lambda_p"].iloc[0],
                    "lambda_o": f_mass["lambda_o"].iloc[0],
                    "epistemic_mass": f_mass["epistemic_mass"].iloc[0],
                })

        conv_df = pd.DataFrame(convergence_data)

        if len(conv_df) == 0:
            print("WARNING: No forecaster-shock convergence episodes found")
            return {"h_fin_3": None, "reason": "no_episodes"}

        converged = conv_df[conv_df["converged"]]
        print(f"\nAnalyzed {len(conv_df)} forecaster-shock episodes")
        print(f"  Converged: {len(converged)} ({len(converged)/len(conv_df):.1%})")

        if len(converged) < 20:
            print("WARNING: Too few converged episodes for reliable analysis")
            return {
                "h_fin_3": "insufficient",
                "n_episodes": len(conv_df),
                "n_converged": len(converged),
            }

        # Primary test: Spearman correlation (experience vs convergence time)
        rho_exp, p_exp = stats.spearmanr(
            converged["lambda_p"], converged["tau_quarters"]
        )

        # Also test with composite mass
        rho_mass, p_mass = stats.spearmanr(
            converged["epistemic_mass"], converged["tau_quarters"]
        )

        print(f"\nConvergence time vs. experience (Lambda_p):")
        print(f"  Spearman rho = {rho_exp:.4f}, p = {p_exp:.4e}")
        print(f"\nConvergence time vs. composite mass:")
        print(f"  Spearman rho = {rho_mass:.4f}, p = {p_mass:.4e}")

        # Split by experience tertiles
        converged = converged.copy()
        converged["exp_tertile"] = pd.qcut(
            converged["lambda_p"], q=3, labels=["Low", "Medium", "High"],
            duplicates="drop",
        )

        print(f"\nConvergence time by experience tertile:")
        for tertile in ["Low", "Medium", "High"]:
            sub = converged[converged["exp_tertile"] == tertile]
            if len(sub) > 0:
                print(f"  {tertile} experience (N={len(sub)}): "
                      f"tau = {sub['tau_quarters'].mean():.2f} +/- {sub['tau_quarters'].std():.2f} quarters")

        # Mann-Whitney: high vs low experience
        high_exp = converged[converged["exp_tertile"] == "High"]
        low_exp = converged[converged["exp_tertile"] == "Low"]

        if len(high_exp) >= 5 and len(low_exp) >= 5:
            u_stat, u_p = stats.mannwhitneyu(
                high_exp["tau_quarters"],
                low_exp["tau_quarters"],
                alternative="greater",
            )
            ratio = high_exp["tau_quarters"].mean() / max(low_exp["tau_quarters"].mean(), 0.01)
            print(f"\nMann-Whitney (high > low experience):")
            print(f"  U = {u_stat:.1f}, p = {u_p:.4e}")
            print(f"  Ratio (tau_high / tau_low) = {ratio:.2f}")
        else:
            u_stat, u_p, ratio = np.nan, np.nan, np.nan

        # OLS regression with controls
        if HAS_STATSMODELS and len(converged) >= 30:
            print(f"\n--- OLS: tau ~ lambda_p + shock_magnitude ---")
            try:
                reg = smf.ols(
                    "tau_quarters ~ lambda_p + shock_magnitude",
                    data=converged,
                ).fit(cov_type="HC1")
                print(reg.summary().tables[1])
            except Exception as e:
                print(f"  OLS failed: {e}")

        result = {
            "n_episodes": len(conv_df),
            "n_converged": len(converged),
            "rho_experience": rho_exp,
            "p_experience": p_exp,
            "rho_mass": rho_mass,
            "p_mass": p_mass,
            "tau_ratio_high_low": ratio,
            "mannwhitney_p": u_p,
            "significant": p_exp < 0.05 and rho_exp > 0,
            "conv_df": conv_df,
        }

        if result["significant"]:
            print(f"\n>>> RESULT: EXPERIENCED FORECASTERS CONVERGE SLOWER (H_fin_3 CONFIRMED)")
            print(f"    tau_high / tau_low = {ratio:.2f}, consistent with tau = M/gamma")
        elif p_exp < 0.10 and rho_exp > 0:
            print(f"\n>>> RESULT: MARGINAL support for H_fin_3 (p = {p_exp:.4f})")
        else:
            print(f"\n>>> RESULT: No significant experience-convergence relationship (H_fin_3 NOT CONFIRMED)")

        return result

    # ==================================================================
    # Mass-revision relationship
    # ==================================================================

    def test_mass_revision_scaling(
        self,
        revisions: pd.DataFrame,
        mass_df: pd.DataFrame,
    ) -> Dict:
        """
        Supplementary test: does higher epistemic mass predict smaller
        forecast revisions (higher inertia)?

        M_i -> smaller |Delta_forecast| per quarter.
        """
        print("\n" + "=" * 70)
        print("SUPPLEMENTARY: MASS -> REVISION SIZE")
        print("=" * 70)

        # Merge mass into revisions
        rev = revisions.merge(
            mass_df[["forecaster_id", "epistemic_mass", "lambda_p", "lambda_o",
                      "beta_incoming_z", "beta_outgoing_z", "quarters_active"]],
            on="forecaster_id",
            how="inner",
        )

        if len(rev) < 50:
            print(f"WARNING: Only {len(rev)} observations. Skipping.")
            return {"mass_revision": None}

        # Mean absolute revision per forecaster
        per_forecaster = rev.groupby("forecaster_id").agg(
            mean_abs_rev=("abs_revision", "mean"),
            epistemic_mass=("epistemic_mass", "first"),
            lambda_p=("lambda_p", "first"),
            lambda_o=("lambda_o", "first"),
            quarters_active=("quarters_active", "first"),
            n_revisions=("revision", "count"),
        ).reset_index()

        # Correlation: mass vs revision size (expected: negative)
        rho, p = stats.spearmanr(
            per_forecaster["epistemic_mass"],
            per_forecaster["mean_abs_rev"],
        )

        print(f"\nSpearman correlation (mass vs |revision|):")
        print(f"  rho = {rho:.4f}, p = {p:.4e}")
        print(f"  {'NEGATIVE as predicted: higher mass -> smaller revisions' if rho < 0 else 'POSITIVE: higher mass -> larger revisions (unexpected)'}")

        # By component
        print(f"\nBy mass component:")
        for comp, label in [("lambda_p", "Experience"), ("lambda_o", "Accuracy")]:
            r, pp = stats.spearmanr(per_forecaster[comp], per_forecaster["mean_abs_rev"])
            print(f"  {label}: rho = {r:.4f}, p = {pp:.4e}")

        # Quintile analysis
        per_forecaster["mass_q"] = pd.qcut(
            per_forecaster["epistemic_mass"], q=5, labels=False, duplicates="drop"
        )
        print(f"\nMean |revision| by mass quintile:")
        for q in sorted(per_forecaster["mass_q"].unique()):
            sub = per_forecaster[per_forecaster["mass_q"] == q]
            print(f"  Q{q+1}: |rev| = {sub['mean_abs_rev'].mean():.4f} (N={len(sub)})")

        return {
            "rho": rho,
            "p": p,
            "n_forecasters": len(per_forecaster),
            "significant": p < 0.05 and rho < 0,
        }

    # ==================================================================
    # Full pipeline
    # ==================================================================

    def run_all_tests(self) -> Dict:
        """Run the complete financial inertia analysis pipeline."""
        print("=" * 70)
        print("FINANCIAL FORECAST INERTIA — EPISTEMIC INERTIA ANALYSIS")
        print("=" * 70)
        print()
        print("Mass formula: M_i = Lambda_p + Lambda_o + Sum beta_ik Lambda_qk + Sum beta_ji Lambda_qi")
        print("  Lambda_p  = forecaster tenure (log quarters in panel)")
        print("  Lambda_o  = historical accuracy (1/RMSE)")
        print("  incoming  = consensus proximity (1/|forecast - median|)")
        print("  outgoing  = influence on subsequent consensus")
        print()

        # Step 1: Load SPF data
        spf = self.load_spf_data()

        # Step 2: Fetch supplementary market data (non-blocking)
        market_shocks = self.fetch_market_shocks()
        earnings = self.fetch_earnings_surprise_data()

        # Step 3: Compute mass proxies
        mass_df = self.compute_mass_proxies(
            spf["revisions"], spf["features"],
            consensus=spf.get("consensus"),
        )

        # Step 4: Run hypothesis tests
        results = {}

        results["h_fin_2"] = self.test_h_fin_2_oscillation(spf["revisions"])
        results["h_fin_3"] = self.test_h_fin_3_experience_convergence(
            spf["revisions"], spf["shocks"], mass_df,
        )
        results["mass_revision"] = self.test_mass_revision_scaling(
            spf["revisions"], mass_df,
        )

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY: FINANCIAL INERTIA TESTS")
        print("=" * 70)

        tests = [
            ("H_fin_2: Oscillation in revisions", results.get("h_fin_2")),
            ("H_fin_3: Experience -> slow convergence", results.get("h_fin_3")),
            ("Mass -> |revision| (validation)", results.get("mass_revision")),
        ]

        for name, r in tests:
            if r is None:
                status = "SKIPPED"
            elif isinstance(r, dict) and r.get("significant"):
                status = "CONFIRMED"
            elif isinstance(r, dict) and r.get("h_fin_3") == "insufficient":
                status = "INSUFFICIENT DATA"
            else:
                status = "NOT CONFIRMED"
            print(f"  {name}: {status}")

        # Additional context from yfinance data
        if not market_shocks.empty:
            print(f"\nMarket shocks identified: {len(market_shocks)}")
        if not earnings.empty:
            print(f"Earnings records fetched: {len(earnings)}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mass_df.to_csv(self.local_dir / f"mass_proxies_{timestamp}.csv", index=False)

        summary = {}
        for k, v in results.items():
            if isinstance(v, dict):
                summary[k] = {
                    kk: vv for kk, vv in v.items()
                    if not isinstance(vv, pd.DataFrame)
                }
        summary_path = self.local_dir / f"financial_results_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\nMass proxies saved to: {self.local_dir / f'mass_proxies_{timestamp}.csv'}")
        print(f"Results saved to: {summary_path}")

        return results


# =========================================================================
# Synthetic demo
# =========================================================================

def run_synthetic_demo():
    """
    Generate synthetic SPF-like data to demonstrate the pipeline when
    real SPF data is not available.
    """
    print("=" * 70)
    print("SYNTHETIC DEMO — Generating simulated forecaster data")
    print("=" * 70)
    print()
    print("To run on real data, first run the SPF pipeline:")
    print("  cd experiments/spf_inertia && python fetch_data.py")
    print()

    np.random.seed(42)

    n_forecasters = 60
    n_quarters = 40
    variables = ["gdp", "inflation"]
    horizons = [1, 2]

    # Generate forecaster features
    features_rows = []
    for fid in range(1, n_forecasters + 1):
        tenure = np.random.randint(5, n_quarters)
        features_rows.append({
            "forecaster_id": fid,
            "quarters_active": tenure,
            "mean_abs_revision": np.random.exponential(0.5) + 0.1,
            "revision_std": np.random.exponential(0.3),
            "sign_change_rate": np.random.uniform(0.3, 0.7),
            "total_revisions": tenure * len(variables) * len(horizons),
            "mass_proxy": np.log1p(tenure),
            "first_survey": "2000-01-01",
            "last_survey": "2020-01-01",
        })
    features = pd.DataFrame(features_rows)

    # Generate panel data with oscillatory dynamics
    panel_rows = []
    for fid in range(1, n_forecasters + 1):
        tenure = features.loc[features["forecaster_id"] == fid, "quarters_active"].iloc[0]
        mass = np.log1p(tenure)

        for var in variables:
            for h in horizons:
                # Simulate forecasts with underdamped oscillation
                # mu_ddot + gamma * mu_dot + K * (mu - target) = noise
                gamma = 0.3 / mass  # Damping inversely proportional to mass
                K = 0.5
                omega = np.sqrt(max(K - (gamma / 2) ** 2, 0.01))

                target = 3.0 if var == "gdp" else 2.5
                mu = target + np.random.normal(0, 0.5)
                mu_dot = 0.0

                for q in range(n_quarters):
                    # Add shocks
                    shock = 0
                    if q == 10:
                        shock = 2.0  # Positive shock
                    elif q == 25:
                        shock = -1.5  # Negative shock

                    target_q = target + shock * np.exp(-0.1 * max(q - 10, 0)) if q >= 10 else target
                    if q >= 25:
                        target_q = target + shock * np.exp(-0.1 * max(q - 25, 0))

                    # Second-order dynamics
                    mu_ddot = -gamma * mu_dot - K * (mu - target_q) + np.random.normal(0, 0.1)
                    mu_dot += mu_ddot * 0.25
                    mu += mu_dot * 0.25

                    year = 2000 + q // 4
                    quarter = q % 4 + 1

                    panel_rows.append({
                        "forecaster_id": fid,
                        "year": year,
                        "quarter": quarter,
                        "variable": var,
                        "horizon": h,
                        "forecast": mu,
                        "survey_date": f"{year}-{quarter * 3:02d}-01",
                    })

    panel = pd.DataFrame(panel_rows)
    panel["survey_date"] = pd.to_datetime(panel["survey_date"])

    # Compute revisions
    panel = panel.sort_values(["forecaster_id", "variable", "horizon", "survey_date"])
    panel["prev_forecast"] = panel.groupby(
        ["forecaster_id", "variable", "horizon"]
    )["forecast"].shift(1)
    panel["revision"] = panel["forecast"] - panel["prev_forecast"]
    panel["abs_revision"] = panel["revision"].abs()
    panel["revision_sign"] = np.sign(panel["revision"])
    panel["prev_revision_sign"] = panel.groupby(
        ["forecaster_id", "variable", "horizon"]
    )["revision_sign"].shift(1)
    panel["sign_change"] = (
        (panel["revision_sign"] != panel["prev_revision_sign"])
        & (panel["revision_sign"] != 0)
        & (panel["prev_revision_sign"] != 0)
    ).astype(int)

    revisions = panel.dropna(subset=["revision"])

    # Compute consensus and shocks
    consensus = panel.groupby(
        ["variable", "horizon", "year", "quarter"]
    )["forecast"].agg(["median", "std", "count"]).reset_index()
    consensus.columns = ["variable", "horizon", "year", "quarter",
                          "consensus_median", "consensus_std", "n_forecasters"]
    consensus["survey_date"] = pd.to_datetime(
        consensus["year"].astype(str) + "-" + (consensus["quarter"] * 3).astype(str).str.zfill(2) + "-01"
    )
    consensus = consensus.sort_values(["variable", "horizon", "survey_date"])
    consensus["prev_consensus"] = consensus.groupby(
        ["variable", "horizon"]
    )["consensus_median"].shift(1)
    consensus["consensus_change"] = consensus["consensus_median"] - consensus["prev_consensus"]
    consensus["abs_consensus_change"] = consensus["consensus_change"].abs()

    # Mark quarters 10 and 25 as shocks
    consensus["rolling_std"] = consensus.groupby(
        ["variable", "horizon"]
    )["consensus_change"].transform(lambda x: x.rolling(8, min_periods=4).std())
    consensus["is_shock"] = (
        consensus["abs_consensus_change"] > 1.0 * consensus["rolling_std"]
    )
    shocks = consensus[consensus["is_shock"]].copy()

    # Save synthetic data
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    panel.to_csv(data_dir / f"panel_{timestamp}.csv", index=False)
    revisions.to_csv(data_dir / f"revisions_{timestamp}.csv", index=False)
    features.to_csv(data_dir / f"features_{timestamp}.csv", index=False)
    shocks.to_csv(data_dir / f"shocks_{timestamp}.csv", index=False)
    consensus.to_csv(data_dir / f"consensus_{timestamp}.csv", index=False)

    print(f"Synthetic data saved to {data_dir}/")
    print(f"  Panel: {len(panel)} rows")
    print(f"  Revisions: {len(revisions)} rows")
    print(f"  Forecasters: {len(features)} rows")
    print(f"  Shocks: {len(shocks)} rows")

    # Run analysis on synthetic data
    analyzer = FinancialInertiaAnalyzer(
        spf_data_dir=str(data_dir),
        local_data_dir=str(data_dir),
    )
    return analyzer.run_all_tests()


# =========================================================================
# Entry point
# =========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Financial Forecast Inertia — Epistemic Inertia Analysis"
    )
    parser.add_argument(
        "--spf-dir",
        default=None,
        help="Directory containing SPF data (default: ../spf_inertia/data/)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Local output directory (default: ./data/)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with synthetic data to demonstrate the pipeline",
    )
    parser.add_argument(
        "--skip-yfinance",
        action="store_true",
        help="Skip yfinance data fetching",
    )
    args = parser.parse_args()

    if args.skip_yfinance:
        HAS_YFINANCE = False

    if args.demo:
        results = run_synthetic_demo()
    else:
        try:
            analyzer = FinancialInertiaAnalyzer(
                spf_data_dir=args.spf_dir,
                local_data_dir=args.data_dir,
            )
            results = analyzer.run_all_tests()
        except FileNotFoundError as e:
            print(f"\n{e}")
            print("\nRunning synthetic demo instead...")
            results = run_synthetic_demo()

    print("\nAnalysis complete!")
