"""
ANES Panel Study — Epistemic Inertia Analysis

Tests two core hypotheses from the Hamiltonian belief dynamics framework
using the American National Election Studies (ANES) panel data.

==========================================================================
MASS FORMULA UNDER TEST
==========================================================================

    M_i = Lambda_p + Lambda_o + Sum_k beta_ik Lambda_tilde_qk + Sum_j beta_ji Lambda_qi
          --------   --------   ----------------------------   -------------------------
          prior       obs        incoming social                outgoing social

    Concrete ANES variable mappings:
        Lambda_p   (prior precision)     -> Response certainty scale (1-5)
                                            e.g., "How certain are you of your position
                                            on [issue]?" Higher certainty = higher prior
                                            precision = more resistance to updating.

        Lambda_o   (observation precision) -> Political knowledge score
                                            Composite of civics quiz items (interviewer-
                                            assessed factual knowledge questions).
                                            More knowledge = more precise observations.

        Sum beta_ik Lambda_tilde_qk      -> Discussion network size and heterogeneity
           (incoming social)                "How many days in the past week did you talk
                                            about politics?" x network size.
                                            Social input that could shift beliefs.

        Sum beta_ji Lambda_qi            -> Persuasion attempts (outgoing influence)
           (outgoing social)                "During the campaign, did you try to show
                                            anyone why they should vote for or against
                                            one of the parties or candidates?"
                                            Frequency of trying to convince others.
                                            The Adams-Solzhenitsyn prediction: those
                                            who broadcast beliefs become anchored to them.

==========================================================================
DATA ACQUISITION — MANUAL DOWNLOAD REQUIRED
==========================================================================

ANES data requires free registration. It cannot be programmatically fetched.

Step 1: Go to https://electionstudies.org/data-center/
Step 2: Create a free account (academic email not required)
Step 3: Download one of the following panel datasets:

    RECOMMENDED — ANES 2016-2020 Panel Study:
        https://electionstudies.org/data-center/2016-2020-panel-study/
        File: anes_timeseries_2016_2020_panel_csv.zip
        OR:   anes_timeseries_2016_2020_panel_stata14.zip  (.dta format)

    ALTERNATIVE — ANES Cumulative Data File (1948-2020):
        https://electionstudies.org/data-center/anes-time-series-cumulative-data-file/
        File: anes_timeseries_cdf_csv_20220916.zip

Step 4: Extract and place CSV or .dta file into:
        experiments/anes_inertia/data/

==========================================================================
KEY VARIABLES TO INCLUDE IN YOUR DOWNLOAD
==========================================================================

For the 2016-2020 Panel, the critical variables are:

    RESPONDENT IDENTIFIER:
        V160001_orig  — case ID (links respondent across waves)

    ISSUE POSITIONS (measured in both waves):
        V161178  — Government services spending (7-point scale)
        V161198  — Defense spending (7-point scale)
        V161232  — Aid to blacks (7-point scale)
        V161196  — Government health insurance (7-point scale)
        V161215  — Abortion (4-point scale)
        V161181  — Jobs/environment tradeoff (7-point scale)
        V161184  — Immigration level (5-point scale)

        Wave 2 equivalents: V201246, V201258, V201318, V201262, V201336,
                            V201249, V201252

    CERTAINTY / CONFIDENCE (the crucial Lambda_p proxy):
        V161179  — How sure: government services
        V161199  — How sure: defense spending
        V161233  — How sure: aid to blacks
        V161197  — How sure: health insurance

        (Coded 1=not at all sure, 5=extremely sure)
        Wave 2: V201247, V201259, V201319, V201263

    PARTISAN IDENTITY:
        V161158x — Party ID 7-point scale
        V161155  — Strength of party identification

    POLITICAL KNOWLEDGE (Lambda_o proxy):
        V162072  — Knowledge: what job does Biden hold
        V162073a — Knowledge: which party controls House
        V162074a — Knowledge: which party controls Senate
        (Score = sum of correct answers, 0-3)

    PERSUASION / OUTGOING SOCIAL (Sum beta_ji proxy):
        V161003  — Talk to anyone about voting for a candidate
        V161002  — Discuss politics: days in past week

    EMOTIONAL ENGAGEMENT:
        V161082  — How angry does [candidate] make you
        V161083  — How afraid does [candidate] make you
        V161086  — How hopeful does [candidate] make you

==========================================================================
HYPOTHESES TESTED
==========================================================================

H4.1 — Perseverance is proportional to Precision, not content:
    Within-person analysis: does issue-specific certainty predict
    cross-wave stability better than partisan identity?
    Expected: beta_certainty > beta_partisanship in multilevel model.

H3.4 — Context-dependent stubbornness:
    The same individual shows different stability across issues,
    predicted by their issue-specific certainty, NOT by a global
    "stubbornness" trait. If stubbornness were a fixed trait, we would
    see high ICC for stability across issues within person; but if
    certainty drives it, the within-person variance in stability should
    be explained by within-person variance in certainty.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Dict, List, Optional, Tuple
import warnings
import sys
import json
from datetime import datetime

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation if not installed
# ---------------------------------------------------------------------------
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not installed. Multilevel models will be skipped.")
    print("  Install with: pip install statsmodels")


# =========================================================================
# CONFIGURATION — map ANES variable names to analysis roles
# =========================================================================

# 2016-2020 Panel variable mapping
PANEL_CONFIG = {
    "respondent_id": "V160001_orig",
    # (issue_name, wave1_position, wave2_position, wave1_certainty, wave2_certainty)
    "issues": [
        {
            "name": "gov_services",
            "pos_w1": "V161178",
            "pos_w2": "V201246",
            "cert_w1": "V161179",
            "cert_w2": "V201247",
            "scale_points": 7,
        },
        {
            "name": "defense_spending",
            "pos_w1": "V161198",
            "pos_w2": "V201258",
            "cert_w1": "V161199",
            "cert_w2": "V201259",
            "scale_points": 7,
        },
        {
            "name": "aid_blacks",
            "pos_w1": "V161232",
            "pos_w2": "V201318",
            "cert_w1": "V161233",
            "cert_w2": "V201319",
            "scale_points": 7,
        },
        {
            "name": "health_insurance",
            "pos_w1": "V161196",
            "pos_w2": "V201262",
            "cert_w1": "V161197",
            "cert_w2": "V201263",
            "scale_points": 7,
        },
        {
            "name": "abortion",
            "pos_w1": "V161215",
            "pos_w2": "V201336",
            "cert_w1": None,  # No certainty item for abortion
            "cert_w2": None,
            "scale_points": 4,
        },
        {
            "name": "immigration",
            "pos_w1": "V161184",
            "pos_w2": "V201252",
            "cert_w1": None,
            "cert_w2": None,
            "scale_points": 5,
        },
    ],
    "partisanship": {
        "party_id_7pt": "V161158x",
        "party_strength": "V161155",
    },
    "knowledge": {
        "biden_job": "V162072",
        "house_control": "V162073a",
        "senate_control": "V162074a",
    },
    "persuasion": {
        "talk_about_voting": "V161003",
        "discuss_politics_days": "V161002",
    },
    "emotion": {
        "angry": "V161082",
        "afraid": "V161083",
        "hopeful": "V161086",
    },
}


class ANESInertiaAnalyzer:
    """
    Analyze epistemic inertia in ANES panel data.

    Tests whether belief persistence is driven by prior precision (certainty)
    rather than partisan identity or dispositional stubbornness.
    """

    def __init__(self, data_dir: str = "data", config: dict = None):
        self.data_dir = Path(data_dir)
        self.config = config or PANEL_CONFIG
        self.results = {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, filename: str = None) -> pd.DataFrame:
        """
        Load ANES panel data from CSV or Stata (.dta) file.

        Args:
            filename: Specific file to load. If None, searches data_dir for
                      any .csv or .dta file matching 'anes*panel*'.

        Returns:
            Raw ANES panel DataFrame.
        """
        if filename:
            path = self.data_dir / filename
        else:
            # Auto-detect
            candidates = (
                list(self.data_dir.glob("*anes*panel*.csv"))
                + list(self.data_dir.glob("*anes*panel*.dta"))
                + list(self.data_dir.glob("*anes*.csv"))
                + list(self.data_dir.glob("*anes*.dta"))
                + list(self.data_dir.glob("*.csv"))
                + list(self.data_dir.glob("*.dta"))
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No data files found in {self.data_dir}/.\n"
                    "Please download ANES panel data from:\n"
                    "  https://electionstudies.org/data-center/\n"
                    "See module docstring for detailed instructions."
                )
            path = candidates[0]

        print(f"Loading data from: {path}")

        if path.suffix == ".dta":
            df = pd.read_stata(path, convert_categoricals=False)
        else:
            df = pd.read_csv(path, low_memory=False)

        print(f"  Loaded {len(df)} respondents, {len(df.columns)} variables")
        return df

    # ------------------------------------------------------------------
    # Variable preparation
    # ------------------------------------------------------------------

    def _recode_missing(self, series: pd.Series) -> pd.Series:
        """Recode ANES missing-data codes to NaN.

        ANES uses negative values for various types of missing data:
            -9 = refused, -8 = don't know, -7 = no post-election interview,
            -6 = no pre-election interview, -5 = interview breakoff,
            -1 = inapplicable
        """
        s = pd.to_numeric(series, errors="coerce")
        s[s < 0] = np.nan
        return s

    def prepare_long_format(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Reshape ANES data into long format for multilevel modeling.

        Output columns:
            respondent_id, issue, position_w1, position_w2, certainty_w1,
            certainty_w2, stability, certainty_mean, partisanship,
            party_strength, knowledge_score, persuasion_score,
            emotional_engagement
        """
        id_col = self.config["respondent_id"]

        # Check which variables actually exist in the data
        available = set(raw.columns)

        rows = []
        issues_with_certainty = 0

        for issue_cfg in self.config["issues"]:
            pos_w1 = issue_cfg["pos_w1"]
            pos_w2 = issue_cfg["pos_w2"]
            cert_w1 = issue_cfg["cert_w1"]
            cert_w2 = issue_cfg["cert_w2"]
            scale_pts = issue_cfg["scale_points"]
            issue_name = issue_cfg["name"]

            # Skip if position variables not in dataset
            if pos_w1 not in available or pos_w2 not in available:
                print(f"  Skipping {issue_name}: position variables not found")
                continue

            has_certainty = (
                cert_w1 is not None
                and cert_w2 is not None
                and cert_w1 in available
                and cert_w2 in available
            )
            if has_certainty:
                issues_with_certainty += 1

            for idx in raw.index:
                row_data = {"respondent_id": raw.at[idx, id_col], "issue": issue_name}

                # Positions
                p1 = self._recode_missing(pd.Series([raw.at[idx, pos_w1]])).iloc[0]
                p2 = self._recode_missing(pd.Series([raw.at[idx, pos_w2]])).iloc[0]

                if np.isnan(p1) or np.isnan(p2):
                    continue

                # Normalize positions to 0-1 scale for cross-issue comparability
                row_data["position_w1"] = p1
                row_data["position_w2"] = p2
                row_data["position_w1_norm"] = (p1 - 1) / max(scale_pts - 1, 1)
                row_data["position_w2_norm"] = (p2 - 1) / max(scale_pts - 1, 1)

                # Stability = 1 - |normalized change|
                # Higher stability = less change = more inertia
                abs_change = abs(row_data["position_w2_norm"] - row_data["position_w1_norm"])
                row_data["abs_change"] = abs_change
                row_data["stability"] = 1.0 - abs_change
                row_data["position_changed"] = int(p1 != p2)

                # Certainty (Lambda_p proxy)
                if has_certainty:
                    c1 = self._recode_missing(pd.Series([raw.at[idx, cert_w1]])).iloc[0]
                    c2 = self._recode_missing(pd.Series([raw.at[idx, cert_w2]])).iloc[0]
                    row_data["certainty_w1"] = c1
                    row_data["certainty_w2"] = c2
                    row_data["certainty_mean"] = np.nanmean([c1, c2])
                else:
                    row_data["certainty_w1"] = np.nan
                    row_data["certainty_w2"] = np.nan
                    row_data["certainty_mean"] = np.nan

                rows.append(row_data)

        if not rows:
            raise ValueError(
                "No valid issue-respondent observations could be constructed.\n"
                "Check that the variable names in PANEL_CONFIG match your data file.\n"
                f"Available columns (first 20): {sorted(available)[:20]}"
            )

        long_df = pd.DataFrame(rows)

        # Add respondent-level variables
        long_df = self._add_respondent_level_vars(long_df, raw)

        print(f"\n  Long-format dataset: {len(long_df)} issue-respondent observations")
        print(f"  Respondents: {long_df['respondent_id'].nunique()}")
        print(f"  Issues: {long_df['issue'].nunique()} ({issues_with_certainty} with certainty)")
        print(f"  Mean stability: {long_df['stability'].mean():.3f}")

        return long_df

    def _add_respondent_level_vars(
        self, long_df: pd.DataFrame, raw: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge respondent-level predictors into long-format data."""
        id_col = self.config["respondent_id"]
        available = set(raw.columns)

        # Build respondent-level features
        resp_features = raw[[id_col]].copy()
        resp_features.columns = ["respondent_id"]

        # Partisanship (absolute value = strength regardless of direction)
        pid_var = self.config["partisanship"]["party_id_7pt"]
        if pid_var in available:
            pid = self._recode_missing(raw[pid_var])
            # 7-point scale centered at 4 (Independent)
            resp_features["partisanship"] = pid
            resp_features["partisan_strength"] = (pid - 4).abs()
        else:
            resp_features["partisanship"] = np.nan
            resp_features["partisan_strength"] = np.nan

        # Party strength (separate measure)
        pstr_var = self.config["partisanship"]["party_strength"]
        if pstr_var in available:
            resp_features["party_strength_raw"] = self._recode_missing(raw[pstr_var])
        else:
            resp_features["party_strength_raw"] = np.nan

        # Political knowledge score (Lambda_o proxy)
        knowledge_vars = self.config["knowledge"]
        k_scores = []
        for kname, kvar in knowledge_vars.items():
            if kvar in available:
                k = self._recode_missing(raw[kvar])
                # Recode to 0/1 correct (ANES typically: 1=correct, 5=incorrect)
                k_binary = (k == 1).astype(float)
                k_binary[k.isna()] = np.nan
                k_scores.append(k_binary)

        if k_scores:
            resp_features["knowledge_score"] = pd.concat(k_scores, axis=1).mean(
                axis=1, skipna=True
            )
        else:
            resp_features["knowledge_score"] = np.nan

        # Persuasion / outgoing social (Sum beta_ji proxy)
        persuasion_vars = self.config["persuasion"]
        p_scores = []
        for pname, pvar in persuasion_vars.items():
            if pvar in available:
                p = self._recode_missing(raw[pvar])
                p_scores.append(p)

        if p_scores:
            # Standardize and average
            p_standardized = [
                (s - s.mean()) / s.std() if s.std() > 0 else s * 0
                for s in p_scores
            ]
            resp_features["persuasion_score"] = pd.concat(
                p_standardized, axis=1
            ).mean(axis=1, skipna=True)
        else:
            resp_features["persuasion_score"] = np.nan

        # Emotional engagement
        emotion_vars = self.config["emotion"]
        e_scores = []
        for ename, evar in emotion_vars.items():
            if evar in available:
                e = self._recode_missing(raw[evar])
                e_scores.append(e)

        if e_scores:
            e_standardized = [
                (s - s.mean()) / s.std() if s.std() > 0 else s * 0
                for s in e_scores
            ]
            resp_features["emotional_engagement"] = pd.concat(
                e_standardized, axis=1
            ).mean(axis=1, skipna=True)
        else:
            resp_features["emotional_engagement"] = np.nan

        # Merge into long format
        long_df = long_df.merge(resp_features, on="respondent_id", how="left")

        return long_df

    # ------------------------------------------------------------------
    # Epistemic mass computation
    # ------------------------------------------------------------------

    def compute_epistemic_mass(self, long_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the full epistemic mass M_i for each respondent-issue pair.

        M_i = Lambda_p + Lambda_o + Sum_k beta_ik Lambda_tilde_qk + Sum_j beta_ji Lambda_qi

        Operationalized as:
            Lambda_p    = certainty_mean (issue-specific, standardized)
            Lambda_o    = knowledge_score (respondent-level, standardized)
            incoming    = 0 (not directly measured in ANES; absorbed into intercept)
            outgoing    = persuasion_score (respondent-level, standardized)

        Returns long_df with added column 'epistemic_mass'.
        """
        df = long_df.copy()

        # Standardize components
        def _zscore(s):
            std = s.std()
            if std == 0 or np.isnan(std):
                return s * 0
            return (s - s.mean()) / std

        cert_z = _zscore(df["certainty_mean"].fillna(df["certainty_mean"].median()))
        know_z = _zscore(df["knowledge_score"].fillna(df["knowledge_score"].median()))
        pers_z = _zscore(df["persuasion_score"].fillna(df["persuasion_score"].median()))

        # Composite mass (equally weighted for now; can be estimated from data)
        df["lambda_p"] = cert_z
        df["lambda_o"] = know_z
        df["beta_outgoing"] = pers_z
        df["epistemic_mass"] = cert_z + know_z + pers_z

        print(f"\n  Epistemic mass computed:")
        print(f"    Lambda_p  (certainty):  mean={cert_z.mean():.3f}, std={cert_z.std():.3f}")
        print(f"    Lambda_o  (knowledge):  mean={know_z.mean():.3f}, std={know_z.std():.3f}")
        print(f"    Beta_out  (persuasion): mean={pers_z.mean():.3f}, std={pers_z.std():.3f}")
        print(f"    M_i       (total mass): mean={df['epistemic_mass'].mean():.3f}, "
              f"std={df['epistemic_mass'].std():.3f}")

        return df

    # ==================================================================
    # TEST H4.1: Perseverance proportional to Precision, not content
    # ==================================================================

    def test_h4_1_precision_vs_partisanship(
        self, long_df: pd.DataFrame
    ) -> Dict:
        """
        H4.1: Within-person analysis — does issue-specific certainty predict
        cross-wave stability better than partisan identity?

        Method:
            Multilevel model: stability ~ certainty + partisan_strength
                              + emotional_engagement + knowledge
                              + (1 | respondent_id)

        The critical comparison is beta_certainty vs. beta_partisanship.
        If the inertia framework is correct, certainty (precision of prior)
        should dominate partisan identity in predicting stability.
        """
        print("\n" + "=" * 70)
        print("TEST H4.1: PERSEVERANCE ~ PRECISION, NOT CONTENT")
        print("=" * 70)

        # Filter to issues with certainty measures
        df_cert = long_df.dropna(subset=["certainty_mean"]).copy()

        if len(df_cert) < 100:
            print(f"WARNING: Only {len(df_cert)} observations with certainty data.")
            print("Need at least 100 for meaningful analysis.")
            return {"h4_1": None, "reason": "insufficient_certainty_data"}

        print(f"\nUsing {len(df_cert)} observations ({df_cert['respondent_id'].nunique()} respondents)")

        # Standardize predictors for coefficient comparison
        for col in ["certainty_mean", "partisan_strength", "emotional_engagement",
                     "knowledge_score", "persuasion_score"]:
            if col in df_cert.columns:
                std = df_cert[col].std()
                if std > 0:
                    df_cert[f"{col}_z"] = (df_cert[col] - df_cert[col].mean()) / std
                else:
                    df_cert[f"{col}_z"] = 0.0

        # ------ Approach 1: OLS with clustered standard errors ------
        print("\n--- Approach 1: OLS with respondent-clustered SEs ---")

        predictors = []
        for p in ["certainty_mean_z", "partisan_strength_z",
                   "emotional_engagement_z", "knowledge_score_z"]:
            if p in df_cert.columns and df_cert[p].notna().sum() > 50:
                predictors.append(p)

        if not predictors:
            print("WARNING: No valid predictors available.")
            return {"h4_1": None, "reason": "no_valid_predictors"}

        # Drop NaNs for regression
        reg_df = df_cert[["stability"] + predictors + ["respondent_id"]].dropna()
        print(f"  Regression sample: {len(reg_df)} observations")

        X = sm.add_constant(reg_df[predictors]) if HAS_STATSMODELS else None
        y = reg_df["stability"]

        ols_result = {}
        if HAS_STATSMODELS:
            # OLS with clustered standard errors
            ols = sm.OLS(y, X).fit(
                cov_type="cluster",
                cov_kwds={"groups": reg_df["respondent_id"]},
            )
            print(ols.summary().tables[1])

            for p in predictors:
                ols_result[p] = {
                    "coef": ols.params[p],
                    "se": ols.bse[p],
                    "t": ols.tvalues[p],
                    "p": ols.pvalues[p],
                }

            # Critical comparison
            if "certainty_mean_z" in ols_result and "partisan_strength_z" in ols_result:
                b_cert = ols_result["certainty_mean_z"]["coef"]
                b_part = ols_result["partisan_strength_z"]["coef"]
                print(f"\n  CRITICAL COMPARISON:")
                print(f"    beta_certainty       = {b_cert:.4f}")
                print(f"    beta_partisanship    = {b_part:.4f}")
                print(f"    Ratio (cert/partisan)= {b_cert / b_part:.2f}" if b_part != 0 else "")
                print(f"    {'>>> CERTAINTY DOMINATES (H4.1 CONFIRMED)' if abs(b_cert) > abs(b_part) else '>>> PARTISANSHIP DOMINATES (H4.1 NOT CONFIRMED)'}")
        else:
            # Fallback: scipy-based OLS
            from numpy.linalg import lstsq

            X_np = np.column_stack([np.ones(len(reg_df))] + [reg_df[p].values for p in predictors])
            betas, residuals, rank, sv = lstsq(X_np, y.values, rcond=None)

            print(f"  Coefficients (basic OLS, no clustered SEs):")
            for i, p in enumerate(["intercept"] + predictors):
                print(f"    {p}: {betas[i]:.4f}")

            if len(predictors) >= 2:
                print(f"\n  CRITICAL COMPARISON:")
                print(f"    beta_certainty    = {betas[1]:.4f}")
                print(f"    beta_partisanship = {betas[2]:.4f}")

        # ------ Approach 2: Multilevel model (if statsmodels available) ------
        mlm_result = {}
        if HAS_STATSMODELS and len(reg_df) >= 200:
            print("\n--- Approach 2: Multilevel model (issues nested in respondents) ---")
            try:
                formula = "stability ~ " + " + ".join(predictors)
                mlm = smf.mixedlm(
                    formula,
                    data=reg_df,
                    groups=reg_df["respondent_id"],
                ).fit(reml=True)

                print(mlm.summary().tables[1])

                for p in predictors:
                    mlm_result[p] = {
                        "coef": mlm.fe_params[p],
                        "se": mlm.bse_fe[p],
                        "z": mlm.tvalues[p],
                        "p": mlm.pvalues[p],
                    }

                # Random effects variance = respondent-level "trait stubbornness"
                re_var = mlm.cov_re.iloc[0, 0] if hasattr(mlm.cov_re, "iloc") else float(mlm.cov_re)
                resid_var = mlm.scale
                icc = re_var / (re_var + resid_var) if (re_var + resid_var) > 0 else 0

                print(f"\n  Random effects:")
                print(f"    Respondent variance (trait stubbornness): {re_var:.4f}")
                print(f"    Residual variance (issue-specific):       {resid_var:.4f}")
                print(f"    ICC (proportion due to trait):             {icc:.4f}")
                print(f"    {'  -> Low ICC supports H3.4: stubbornness is context-dependent' if icc < 0.3 else '  -> High ICC suggests trait-like stubbornness'}")

                mlm_result["icc"] = icc
                mlm_result["re_variance"] = re_var
                mlm_result["residual_variance"] = resid_var

            except Exception as e:
                print(f"  Mixed model failed: {e}")
                print("  Falling back to OLS results only.")

        # ------ Approach 3: Within-person correlation ------
        print("\n--- Approach 3: Within-person certainty-stability correlation ---")
        within_corrs = []
        for rid, rgroup in df_cert.groupby("respondent_id"):
            if len(rgroup) < 3:
                continue
            cert = rgroup["certainty_mean"]
            stab = rgroup["stability"]
            if cert.std() > 0 and stab.std() > 0:
                r, p = stats.pearsonr(cert, stab)
                within_corrs.append({"respondent_id": rid, "r": r, "p": p, "n": len(rgroup)})

        if within_corrs:
            wc_df = pd.DataFrame(within_corrs)
            mean_r = wc_df["r"].mean()
            t_stat, t_p = stats.ttest_1samp(wc_df["r"].dropna(), 0)

            print(f"  Within-person correlations (certainty <-> stability):")
            print(f"    N respondents: {len(wc_df)}")
            print(f"    Mean r = {mean_r:.4f}")
            print(f"    t-test (H: mean r > 0): t = {t_stat:.3f}, p = {t_p:.4e}")
            print(f"    {'>>> SIGNIFICANT: certainty predicts stability within-person' if t_p < 0.05 and mean_r > 0 else '>>> NOT SIGNIFICANT'}")
        else:
            wc_df = pd.DataFrame()
            mean_r = np.nan

        result = {
            "n_observations": len(df_cert),
            "n_respondents": df_cert["respondent_id"].nunique(),
            "ols_result": ols_result,
            "mlm_result": mlm_result,
            "within_person_mean_r": mean_r,
            "within_person_n": len(wc_df) if len(within_corrs) > 0 else 0,
        }

        # Determine if H4.1 is confirmed
        confirmed = False
        if ols_result and "certainty_mean_z" in ols_result and "partisan_strength_z" in ols_result:
            confirmed = abs(ols_result["certainty_mean_z"]["coef"]) > abs(
                ols_result["partisan_strength_z"]["coef"]
            )
        result["h4_1_confirmed"] = confirmed

        return result

    # ==================================================================
    # TEST H3.4: Context-dependent stubbornness
    # ==================================================================

    def test_h3_4_context_dependent(self, long_df: pd.DataFrame) -> Dict:
        """
        H3.4: The same individual shows different stability across issues,
        predicted by their issue-specific certainty, not by a global
        stubbornness trait.

        Evidence for H3.4:
        1. High within-person variance in stability across issues
        2. Certainty explains within-person variance in stability
        3. Low ICC from random-intercept model (stability not a trait)

        Evidence against H3.4:
        1. Low within-person variance (everyone is equally stable across issues)
        2. ICC > 0.5 (most variation is between-person = trait-like)
        """
        print("\n" + "=" * 70)
        print("TEST H3.4: CONTEXT-DEPENDENT STUBBORNNESS")
        print("=" * 70)

        df = long_df.copy()

        # 1. Compute within-person variance in stability
        resp_stats = df.groupby("respondent_id").agg(
            n_issues=("stability", "count"),
            mean_stability=("stability", "mean"),
            std_stability=("stability", "std"),
            var_stability=("stability", "var"),
        ).reset_index()

        # Need multiple issues per person
        resp_multi = resp_stats[resp_stats["n_issues"] >= 3]
        print(f"\nRespondents with 3+ issues: {len(resp_multi)}")

        if len(resp_multi) < 30:
            print("WARNING: Too few respondents with multiple issues for within-person analysis.")
            return {"h3_4": None, "reason": "insufficient_within_person_data"}

        mean_within_var = resp_multi["var_stability"].mean()
        between_var = resp_multi["mean_stability"].var()

        print(f"\n  Mean within-person variance in stability: {mean_within_var:.4f}")
        print(f"  Between-person variance in mean stability: {between_var:.4f}")
        print(f"  Ratio (within/between): {mean_within_var / between_var:.2f}" if between_var > 0 else "")

        # 2. ICC decomposition
        grand_mean = df["stability"].mean()
        total_var = df["stability"].var()

        print(f"\n  Grand mean stability: {grand_mean:.3f}")
        print(f"  Total variance: {total_var:.4f}")

        # Naive ICC = between-person var / total var
        if total_var > 0:
            naive_icc = between_var / total_var
            print(f"  Naive ICC (between/total): {naive_icc:.4f}")
        else:
            naive_icc = np.nan

        # 3. Within-person analysis: certainty predicts issue-specific stability
        df_cert = df.dropna(subset=["certainty_mean"]).copy()

        if len(df_cert) < 100:
            print("WARNING: Insufficient certainty data for within-person analysis.")
            within_result = None
        else:
            # Person-mean-center certainty to isolate within-person effect
            person_means = df_cert.groupby("respondent_id")["certainty_mean"].transform("mean")
            df_cert["certainty_within"] = df_cert["certainty_mean"] - person_means
            df_cert["certainty_between"] = person_means

            print("\n--- Within-person regression: stability ~ certainty_within + certainty_between ---")

            if HAS_STATSMODELS:
                try:
                    # Mundlak approach: include both within and between
                    reg_data = df_cert[
                        ["stability", "certainty_within", "certainty_between", "respondent_id"]
                    ].dropna()

                    if len(reg_data) >= 50:
                        model = smf.mixedlm(
                            "stability ~ certainty_within + certainty_between",
                            data=reg_data,
                            groups=reg_data["respondent_id"],
                        ).fit(reml=True)

                        print(model.summary().tables[1])

                        b_within = model.fe_params.get("certainty_within", np.nan)
                        p_within = model.pvalues.get("certainty_within", np.nan)

                        print(f"\n  Within-person effect of certainty: beta = {b_within:.4f}, p = {p_within:.4e}")
                        if p_within < 0.05 and b_within > 0:
                            print("  >>> CONFIRMED: When a person is more certain about an issue,")
                            print("      they are more stable on that issue (within-person).")
                            print("      This supports H3.4: stubbornness is context-dependent.")
                        else:
                            print("  >>> NOT CONFIRMED at p < 0.05")

                        within_result = {
                            "beta_within": b_within,
                            "p_within": p_within,
                            "beta_between": model.fe_params.get("certainty_between", np.nan),
                            "p_between": model.pvalues.get("certainty_between", np.nan),
                        }
                    else:
                        print(f"  Only {len(reg_data)} observations after dropna. Skipping.")
                        within_result = None

                except Exception as e:
                    print(f"  Within-person model failed: {e}")
                    within_result = None
            else:
                # Fallback: simple within-person correlation
                within_corr, within_p = stats.pearsonr(
                    df_cert["certainty_within"].dropna(),
                    df_cert.loc[df_cert["certainty_within"].notna(), "stability"],
                )
                print(f"  Within-person correlation (certainty_within vs stability):")
                print(f"    r = {within_corr:.4f}, p = {within_p:.4e}")
                within_result = {"r_within": within_corr, "p_within": within_p}

        # 4. Test: does a "stubbornness trait" model fit worse than
        #    a certainty-driven model?
        print("\n--- Variance decomposition ---")

        # Within-person R-squared: how much of issue-specific stability
        # does certainty explain beyond person-level means?
        if len(df_cert) >= 50 and "certainty_within" in df_cert.columns:
            cw = df_cert["certainty_within"].dropna()
            stab = df_cert.loc[cw.index, "stability"]
            if len(cw) > 10:
                slope, intercept, r_value, p_val, se = stats.linregress(cw, stab)
                print(f"  Within-person R-squared (certainty): {r_value**2:.4f}")
                print(f"  This means certainty explains {r_value**2:.1%} of within-person")
                print(f"  variation in stability across issues.")

        result = {
            "n_respondents_multi_issue": len(resp_multi),
            "mean_within_person_var": mean_within_var,
            "between_person_var": between_var,
            "naive_icc": naive_icc,
            "within_person_result": within_result,
            "h3_4_confirmed": (
                within_result is not None
                and within_result.get("p_within", 1.0) < 0.05
                and within_result.get("beta_within", within_result.get("r_within", 0)) > 0
            ),
        }

        if result["h3_4_confirmed"]:
            print("\n>>> RESULT: H3.4 CONFIRMED — stability is context-dependent,")
            print("    driven by issue-specific certainty, not a global trait.")
        elif naive_icc is not None and naive_icc < 0.3:
            print("\n>>> PARTIAL SUPPORT: Low ICC suggests stability is NOT trait-like,")
            print("    but certainty effect is not significant at p < 0.05.")
        else:
            print("\n>>> RESULT: H3.4 NOT CONFIRMED")

        return result

    # ==================================================================
    # Epistemic mass validation: does M predict stability?
    # ==================================================================

    def test_mass_predicts_stability(self, long_df: pd.DataFrame) -> Dict:
        """
        Validation test: does the composite epistemic mass M_i predict
        cross-wave stability?

        This is the fundamental prediction of the framework: higher mass
        agents should show smaller belief updates (higher stability).
        """
        print("\n" + "=" * 70)
        print("VALIDATION: EPISTEMIC MASS -> STABILITY")
        print("=" * 70)

        df = long_df.dropna(subset=["epistemic_mass", "stability"]).copy()

        if len(df) < 50:
            print(f"WARNING: Only {len(df)} valid observations. Skipping.")
            return {"validation": None}

        # Overall correlation
        r_overall, p_overall = stats.spearmanr(df["epistemic_mass"], df["stability"])
        print(f"\n  Spearman correlation (M_i vs stability):")
        print(f"    rho = {r_overall:.4f}, p = {p_overall:.4e}")

        # By component
        print(f"\n  By mass component:")
        component_results = {}
        for comp, label in [
            ("lambda_p", "Lambda_p (certainty)"),
            ("lambda_o", "Lambda_o (knowledge)"),
            ("beta_outgoing", "Beta_outgoing (persuasion)"),
        ]:
            if comp in df.columns:
                valid = df[[comp, "stability"]].dropna()
                if len(valid) >= 20:
                    r, p = stats.spearmanr(valid[comp], valid["stability"])
                    print(f"    {label}: rho = {r:.4f}, p = {p:.4e}")
                    component_results[comp] = {"rho": r, "p": p}

        # Quintile analysis
        df["mass_quintile"] = pd.qcut(
            df["epistemic_mass"], q=5, labels=False, duplicates="drop"
        )
        quintile_stability = df.groupby("mass_quintile")["stability"].agg(["mean", "std", "count"])
        print(f"\n  Stability by mass quintile:")
        for q in quintile_stability.index:
            row = quintile_stability.loc[q]
            print(f"    Q{q+1}: stability = {row['mean']:.3f} +/- {row['std']:.3f} (N={int(row['count'])})")

        # Monotonicity test
        q_means = quintile_stability["mean"].values
        is_monotonic = all(q_means[i] <= q_means[i + 1] for i in range(len(q_means) - 1))
        print(f"  Monotonically increasing: {'YES' if is_monotonic else 'NO'}")

        result = {
            "rho_overall": r_overall,
            "p_overall": p_overall,
            "component_results": component_results,
            "monotonic_quintiles": is_monotonic,
            "significant": p_overall < 0.05 and r_overall > 0,
        }

        if result["significant"]:
            print(f"\n>>> RESULT: Epistemic mass SIGNIFICANTLY predicts stability")
            print(f"    Higher mass -> higher stability, as predicted by M d2mu/dt2 + ... = 0")
        else:
            print(f"\n>>> RESULT: Mass-stability relationship not significant")

        return result

    # ==================================================================
    # Full pipeline
    # ==================================================================

    def run_all_tests(self, filename: str = None) -> Dict:
        """Run the complete ANES epistemic inertia analysis pipeline."""
        print("=" * 70)
        print("ANES PANEL STUDY — EPISTEMIC INERTIA ANALYSIS")
        print("=" * 70)
        print()
        print("Mass formula: M_i = Lambda_p + Lambda_o + Sum beta_ik Lambda_qk + Sum beta_ji Lambda_qi")
        print("  Lambda_p  = response certainty (issue-specific)")
        print("  Lambda_o  = political knowledge score")
        print("  outgoing  = persuasion attempt frequency")
        print()

        # Step 1: Load data
        raw = self.load_data(filename)

        # Step 2: Reshape to long format
        long_df = self.prepare_long_format(raw)

        # Step 3: Compute epistemic mass
        long_df = self.compute_epistemic_mass(long_df)

        # Step 4: Run tests
        results = {}
        results["h4_1"] = self.test_h4_1_precision_vs_partisanship(long_df)
        results["h3_4"] = self.test_h3_4_context_dependent(long_df)
        results["mass_validation"] = self.test_mass_predicts_stability(long_df)

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY: ANES EPISTEMIC INERTIA TESTS")
        print("=" * 70)

        tests = [
            ("H4.1 Precision > Partisanship", results["h4_1"]),
            ("H3.4 Context-dependent stubbornness", results["h3_4"]),
            ("Mass -> Stability validation", results["mass_validation"]),
        ]

        for name, r in tests:
            if r is None:
                status = "SKIPPED"
            elif r.get("h4_1_confirmed") or r.get("h3_4_confirmed") or r.get("significant"):
                status = "CONFIRMED"
            else:
                status = "NOT CONFIRMED"
            print(f"  {name}: {status}")

        # Save long-format data for further analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outpath = self.data_dir / f"anes_long_format_{timestamp}.csv"
        long_df.to_csv(outpath, index=False)
        print(f"\nLong-format data saved to: {outpath}")

        # Save results summary
        summary = {
            k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (pd.DataFrame, dict))}
            for k, v in results.items()
            if isinstance(v, dict)
        }
        summary_path = self.data_dir / f"anes_results_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Results summary saved to: {summary_path}")

        return results


# =========================================================================
# Synthetic demo (when no real data is available)
# =========================================================================

def run_synthetic_demo():
    """
    Generate synthetic data that mirrors ANES structure to demonstrate
    the analysis pipeline and expected results.

    The synthetic data is designed to have the properties predicted by
    the inertia framework:
    - Certainty predicts stability (H4.1)
    - Same person varies in stability across issues (H3.4)
    - Partisanship has weaker effect than certainty
    """
    print("=" * 70)
    print("SYNTHETIC DEMO — No ANES data found, generating simulated data")
    print("=" * 70)
    print()
    print("To run on real data, download from https://electionstudies.org/data-center/")
    print("See module docstring for full instructions.")
    print()

    np.random.seed(42)
    n_respondents = 800
    issues = ["gov_services", "defense_spending", "aid_blacks", "health_insurance"]
    n_issues = len(issues)

    rows = []
    for rid in range(1, n_respondents + 1):
        # Respondent-level traits
        partisanship = np.random.choice(range(1, 8))  # 1-7 scale
        partisan_strength = abs(partisanship - 4)
        knowledge = np.random.uniform(0, 1)
        persuasion = np.random.normal(0, 1)
        emotion = np.random.normal(0, 1)

        # Small respondent-level "trait stubbornness" (should be small per H3.4)
        trait_stability = np.random.normal(0, 0.05)

        for issue in issues:
            # Issue-specific certainty (the key predictor per H4.1)
            certainty = np.random.uniform(1, 5)

            # Generate position at wave 1
            pos_w1 = np.random.choice(range(1, 8))

            # Stability driven primarily by certainty, weakly by partisanship
            # This encodes the H4.1 prediction
            stability_prob = (
                0.3                                    # base stability
                + 0.12 * (certainty / 5)               # certainty effect (strong)
                + 0.04 * (partisan_strength / 3)       # partisanship effect (weak)
                + 0.02 * knowledge                     # knowledge effect (small)
                + trait_stability                      # person-level trait (small)
            )
            stability_prob = np.clip(stability_prob, 0.05, 0.95)

            # Generate wave 2 position
            if np.random.random() < stability_prob:
                pos_w2 = pos_w1  # No change
            else:
                # Random change (bounded by scale)
                shift = np.random.choice([-2, -1, 1, 2])
                pos_w2 = np.clip(pos_w1 + shift, 1, 7)

            rows.append({
                "V160001_orig": rid,
                f"V161178": pos_w1 if issue == "gov_services" else -1,
                f"V201246": pos_w2 if issue == "gov_services" else -1,
                f"V161179": certainty if issue == "gov_services" else -1,
                f"V201247": certainty + np.random.normal(0, 0.3) if issue == "gov_services" else -1,
                f"V161198": pos_w1 if issue == "defense_spending" else -1,
                f"V201258": pos_w2 if issue == "defense_spending" else -1,
                f"V161199": certainty if issue == "defense_spending" else -1,
                f"V201259": certainty + np.random.normal(0, 0.3) if issue == "defense_spending" else -1,
                f"V161232": pos_w1 if issue == "aid_blacks" else -1,
                f"V201318": pos_w2 if issue == "aid_blacks" else -1,
                f"V161233": certainty if issue == "aid_blacks" else -1,
                f"V201319": certainty + np.random.normal(0, 0.3) if issue == "aid_blacks" else -1,
                f"V161196": pos_w1 if issue == "health_insurance" else -1,
                f"V201262": pos_w2 if issue == "health_insurance" else -1,
                f"V161197": certainty if issue == "health_insurance" else -1,
                f"V201263": certainty + np.random.normal(0, 0.3) if issue == "health_insurance" else -1,
                "V161158x": partisanship,
                "V161155": max(1, partisan_strength),
                "V162072": 1 if knowledge > 0.6 else 5,
                "V162073a": 1 if knowledge > 0.4 else 5,
                "V162074a": 1 if knowledge > 0.5 else 5,
                "V161003": 1 if persuasion > 0.5 else 2,
                "V161002": max(0, int(persuasion * 2 + 3)),
                "V161082": max(1, int(emotion * 1.5 + 3)),
                "V161083": max(1, int(emotion * 1.2 + 2.5)),
                "V161086": max(1, int(-emotion * 1.0 + 3)),
            })

    # Build a wide-format "raw" DataFrame that mimics ANES structure
    # We need one row per respondent with all issue variables
    # Aggregate: for each respondent, take valid (non -1) values per variable
    raw_rows = {}
    for row in rows:
        rid = row["V160001_orig"]
        if rid not in raw_rows:
            raw_rows[rid] = dict(row)
        else:
            # Merge: take non-missing values
            for k, v in row.items():
                if k == "V160001_orig":
                    continue
                if isinstance(v, (int, float)) and v != -1 and (
                    k not in raw_rows[rid] or raw_rows[rid][k] == -1
                ):
                    raw_rows[rid][k] = v

    raw_df = pd.DataFrame(list(raw_rows.values()))
    print(f"Generated synthetic data: {len(raw_df)} respondents, {len(raw_df.columns)} variables")

    # Save synthetic data
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True, parents=True)
    raw_df.to_csv(data_dir / "anes_synthetic_panel.csv", index=False)

    # Run analysis
    analyzer = ANESInertiaAnalyzer(data_dir=str(data_dir))
    return analyzer.run_all_tests("anes_synthetic_panel.csv")


# =========================================================================
# Entry point
# =========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ANES Panel Study — Epistemic Inertia Analysis"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing ANES data file (default: data/)",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Specific data file to load (default: auto-detect)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with synthetic data to demonstrate the pipeline",
    )
    args = parser.parse_args()

    if args.demo:
        results = run_synthetic_demo()
    else:
        try:
            analyzer = ANESInertiaAnalyzer(data_dir=args.data_dir)
            results = analyzer.run_all_tests(filename=args.file)
        except FileNotFoundError as e:
            print(f"\n{e}")
            print("\nRunning synthetic demo instead...")
            results = run_synthetic_demo()

    print("\nAnalysis complete!")
