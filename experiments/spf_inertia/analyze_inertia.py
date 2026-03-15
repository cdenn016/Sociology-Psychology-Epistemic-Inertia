"""
SPF Epistemic Inertia Analysis

Tests three core predictions from the Hamiltonian belief dynamics framework:

Test 1 — OSCILLATION (H2.1): Individual forecast revision sequences show
    non-monotonic trajectories (sign reversals) more often than an AR(1) null.

Test 2 — RELAXATION TIME (H1.2): Experienced forecasters take longer to
    converge to post-shock consensus. τ ∝ M/γ ∝ experience.

Test 3 — OVERSHOOT (H1.1): High-precision forecasters overshoot the eventual
    settled consensus by more. d_overshoot ∝ √(Λ_p).

Mass formula mapping:
    M_i = Λ_p + Λ_o + Σ β_ik Λ̃_qk + Σ β_ji Λ_qi

    Λ_p → log(quarters_in_panel)       [experience = prior precision]
    Λ_o → 1/RMSE_historical            [accuracy = observation precision]
    Σ β_ik → 1/|forecast - median|     [consensus proximity]
    Σ β_ji → influence_score           [Granger-causality on consensus]
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SPFInertiaAnalyzer:
    """Analyze epistemic inertia in Survey of Professional Forecasters data."""

    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)

    def load_latest(self) -> Dict[str, pd.DataFrame]:
        """Load most recent data files."""
        def latest(pattern):
            files = sorted(self.data_dir.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No files matching {pattern} in {self.data_dir}")
            return pd.read_csv(files[-1])

        return {
            'revisions': latest('revisions_*.csv'),
            'features': latest('features_*.csv'),
            'shocks': latest('shocks_*.csv'),
            'consensus': latest('consensus_*.csv'),
            'panel': latest('panel_*.csv'),
        }

    # ------------------------------------------------------------------
    # TEST 1: Oscillation Detection (H2.1)
    # ------------------------------------------------------------------
    def test_oscillation(self, revisions: pd.DataFrame) -> Dict:
        """
        Test whether forecast revisions show more sign changes (oscillation)
        than predicted by an AR(1) or random walk null model.

        Under gradient descent (first-order dynamics), belief changes should
        be monotonic — the sign of revision should persist. Under Hamiltonian
        dynamics (second-order), beliefs overshoot and oscillate, producing
        excess sign reversals.

        Test: Compare observed sign-change rate to expected rate under null.
        Null model: If revisions are i.i.d. draws from a symmetric distribution,
        P(sign change) = 0.5. If revisions are AR(1) with positive autocorrelation
        (ρ > 0), P(sign change) < 0.5.
        """
        print("=" * 70)
        print("TEST 1: BELIEF OSCILLATION (H2.1)")
        print("=" * 70)

        # Per-forecaster sign change analysis
        forecaster_stats = []

        for fid, group in revisions.groupby('forecaster_id'):
            for (var, horizon), subgroup in group.groupby(['variable', 'horizon']):
                sub = subgroup.sort_values('survey_date')

                if len(sub) < 8:  # Need enough data
                    continue

                rev = sub['revision'].dropna().values
                if len(rev) < 6:
                    continue

                # Count sign changes
                signs = np.sign(rev[rev != 0])
                if len(signs) < 4:
                    continue

                sign_changes = np.sum(np.diff(signs) != 0)
                n_transitions = len(signs) - 1

                # AR(1) autocorrelation
                if len(rev) > 2:
                    ar1_corr = np.corrcoef(rev[:-1], rev[1:])[0, 1]
                else:
                    ar1_corr = np.nan

                # Expected sign change rate under null
                # For AR(1) with autocorrelation ρ, P(sign change) ≈ (1 - ρ) / 2
                if not np.isnan(ar1_corr):
                    expected_rate = (1 - max(0, ar1_corr)) / 2
                else:
                    expected_rate = 0.5

                observed_rate = sign_changes / n_transitions if n_transitions > 0 else 0

                forecaster_stats.append({
                    'forecaster_id': fid,
                    'variable': var,
                    'horizon': horizon,
                    'n_revisions': len(rev),
                    'sign_changes': sign_changes,
                    'n_transitions': n_transitions,
                    'observed_sign_change_rate': observed_rate,
                    'ar1_autocorrelation': ar1_corr,
                    'expected_sign_change_rate': expected_rate,
                    'excess_oscillation': observed_rate - expected_rate,
                })

        stats_df = pd.DataFrame(forecaster_stats)

        if len(stats_df) == 0:
            print("WARNING: No forecaster series long enough for oscillation analysis")
            return {'oscillation': None}

        # Aggregate results
        mean_observed = stats_df['observed_sign_change_rate'].mean()
        mean_expected = stats_df['expected_sign_change_rate'].mean()
        mean_excess = stats_df['excess_oscillation'].mean()

        # One-sample t-test: is excess oscillation significantly > 0?
        t_stat, p_value = stats.ttest_1samp(
            stats_df['excess_oscillation'].dropna(), 0
        )

        # Wilcoxon signed-rank test (nonparametric)
        try:
            w_stat, w_pvalue = stats.wilcoxon(
                stats_df['excess_oscillation'].dropna(),
                alternative='greater'
            )
        except ValueError:
            w_stat, w_pvalue = np.nan, np.nan

        print(f"\nAnalyzed {len(stats_df)} forecaster-variable-horizon series")
        print(f"\nSign change rates:")
        print(f"  Observed: {mean_observed:.4f}")
        print(f"  Expected (AR(1) null): {mean_expected:.4f}")
        print(f"  Excess: {mean_excess:.4f}")
        print(f"\nt-test (H: excess > 0):")
        print(f"  t = {t_stat:.3f}, p = {p_value:.4e}")
        print(f"  {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'}")
        print(f"\nWilcoxon signed-rank:")
        print(f"  W = {w_stat:.1f}, p = {w_pvalue:.4e}")

        # Breakdown by variable
        print(f"\nBy variable:")
        for var in stats_df['variable'].unique():
            sub = stats_df[stats_df['variable'] == var]
            print(f"  {var}: excess = {sub['excess_oscillation'].mean():.4f} "
                  f"(N={len(sub)})")

        result = {
            'n_series': len(stats_df),
            'mean_observed_rate': mean_observed,
            'mean_expected_rate': mean_expected,
            'mean_excess': mean_excess,
            't_stat': t_stat,
            'p_value': p_value,
            'w_stat': w_stat,
            'w_pvalue': w_pvalue,
            'significant': p_value < 0.05,
            'stats_df': stats_df,
        }

        if result['significant'] and mean_excess > 0:
            print("\n>>> RESULT: Excess oscillation detected — consistent with H2.1")
        elif result['significant'] and mean_excess < 0:
            print("\n>>> RESULT: Less oscillation than expected — inconsistent with H2.1")
        else:
            print("\n>>> RESULT: No significant excess oscillation")

        return result

    # ------------------------------------------------------------------
    # TEST 2: Relaxation Time (H1.2)
    # ------------------------------------------------------------------
    def test_relaxation_time(self, revisions: pd.DataFrame,
                              shocks: pd.DataFrame,
                              features: pd.DataFrame) -> Dict:
        """
        Test whether experienced forecasters (high Λ_p) take longer to
        converge to post-shock consensus.

        Prediction: τ = M/γ, so experienced forecasters (high M) have
        longer relaxation times.
        """
        print("\n" + "=" * 70)
        print("TEST 2: RELAXATION TIME SCALING (H1.2)")
        print("=" * 70)

        if len(shocks) == 0:
            print("WARNING: No shocks identified")
            return {'relaxation': None}

        revisions['survey_date'] = pd.to_datetime(revisions['survey_date'])
        shocks['survey_date'] = pd.to_datetime(shocks['survey_date'])

        relaxation_data = []

        for _, shock in shocks.iterrows():
            var = shock['variable']
            horizon = shock['horizon']
            shock_date = shock['survey_date']

            # Get consensus after shock settles (4-8 quarters later)
            post_shock_mask = (
                (revisions['variable'] == var) &
                (revisions['horizon'] == horizon) &
                (revisions['survey_date'] > shock_date) &
                (revisions['survey_date'] <= shock_date + pd.DateOffset(years=2))
            )

            post_shock = revisions[post_shock_mask]
            if len(post_shock) < 10:
                continue

            # Settled consensus = median of forecasts 6-8 quarters post-shock
            late_mask = post_shock['survey_date'] > shock_date + pd.DateOffset(months=15)
            if late_mask.sum() < 5:
                continue

            settled_consensus = post_shock[late_mask]['forecast'].median()

            # For each forecaster, measure time to converge
            for fid in post_shock['forecaster_id'].unique():
                f_post = post_shock[post_shock['forecaster_id'] == fid].sort_values('survey_date')

                if len(f_post) < 3:
                    continue

                # Compute distance from settled consensus over time
                f_post = f_post.copy()
                f_post['distance'] = (f_post['forecast'] - settled_consensus).abs()

                # Relaxation time = first quarter where distance < threshold
                initial_distance = f_post['distance'].iloc[0]
                if initial_distance < 1e-6:
                    continue

                threshold = initial_distance * 0.37  # 1/e decay
                converged = f_post[f_post['distance'] <= threshold]

                if len(converged) > 0:
                    convergence_date = converged['survey_date'].iloc[0]
                    relaxation_quarters = (
                        (convergence_date - shock_date).days / 91.25
                    )
                else:
                    relaxation_quarters = np.nan  # Did not converge

                # Get forecaster experience
                f_features = features[features['forecaster_id'] == fid]
                if len(f_features) == 0:
                    continue

                experience = f_features['quarters_active'].iloc[0]

                relaxation_data.append({
                    'forecaster_id': fid,
                    'shock_date': shock_date,
                    'variable': var,
                    'horizon': horizon,
                    'experience': experience,
                    'mass_proxy': np.log1p(experience),
                    'initial_distance': initial_distance,
                    'relaxation_quarters': relaxation_quarters,
                    'converged': not np.isnan(relaxation_quarters),
                })

        relax_df = pd.DataFrame(relaxation_data)

        if len(relax_df) == 0:
            print("WARNING: No forecaster-shock episodes found")
            return {'relaxation': None}

        converged = relax_df[relax_df['converged']]
        print(f"\nAnalyzed {len(relax_df)} forecaster-shock episodes")
        print(f"  Converged: {len(converged)} ({len(converged)/len(relax_df):.1%})")

        if len(converged) < 20:
            print("WARNING: Too few converged episodes for reliable analysis")
            return {'relaxation': {'n_episodes': len(relax_df), 'n_converged': len(converged)}}

        # Spearman correlation: experience vs relaxation time
        corr, p_val = stats.spearmanr(
            converged['mass_proxy'],
            converged['relaxation_quarters']
        )

        print(f"\nRelaxation time vs. experience (mass proxy):")
        print(f"  Spearman ρ = {corr:.4f}")
        print(f"  p-value = {p_val:.4e}")

        # Split by experience
        exp_median = converged['experience'].median()
        high_exp = converged[converged['experience'] >= exp_median]
        low_exp = converged[converged['experience'] < exp_median]

        print(f"\nHigh experience (N={len(high_exp)}):")
        print(f"  Mean relaxation time: {high_exp['relaxation_quarters'].mean():.2f} quarters")
        print(f"\nLow experience (N={len(low_exp)}):")
        print(f"  Mean relaxation time: {low_exp['relaxation_quarters'].mean():.2f} quarters")

        # Mann-Whitney test
        u_stat, u_pval = stats.mannwhitneyu(
            high_exp['relaxation_quarters'],
            low_exp['relaxation_quarters'],
            alternative='greater'
        )
        print(f"\nMann-Whitney U (H: high_exp > low_exp):")
        print(f"  U = {u_stat:.1f}, p = {u_pval:.4e}")

        result = {
            'n_episodes': len(relax_df),
            'n_converged': len(converged),
            'spearman_rho': corr,
            'spearman_p': p_val,
            'high_exp_mean_tau': high_exp['relaxation_quarters'].mean(),
            'low_exp_mean_tau': low_exp['relaxation_quarters'].mean(),
            'mannwhitney_p': u_pval,
            'significant': p_val < 0.05 and corr > 0,
            'relax_df': relax_df,
        }

        if result['significant']:
            ratio = result['high_exp_mean_tau'] / max(result['low_exp_mean_tau'], 0.01)
            print(f"\n>>> RESULT: Relaxation time scales with experience (ratio={ratio:.2f})")
            print("    Consistent with τ = M/γ prediction (H1.2)")
        else:
            print("\n>>> RESULT: No significant experience-relaxation scaling")

        return result

    # ------------------------------------------------------------------
    # TEST 3: Overshoot Magnitude (H1.1)
    # ------------------------------------------------------------------
    def test_overshoot(self, revisions: pd.DataFrame,
                        shocks: pd.DataFrame,
                        features: pd.DataFrame) -> Dict:
        """
        Test whether high-precision forecasters overshoot more.

        Prediction: d_overshoot = |μ̇| √(M/K)
        More experienced/accurate forecasters should overshoot the eventual
        consensus by a larger amount when they do eventually revise.
        """
        print("\n" + "=" * 70)
        print("TEST 3: OVERSHOOT MAGNITUDE (H1.1)")
        print("=" * 70)

        revisions['survey_date'] = pd.to_datetime(revisions['survey_date'])
        shocks['survey_date'] = pd.to_datetime(shocks['survey_date'])

        overshoot_data = []

        for _, shock in shocks.iterrows():
            var = shock['variable']
            horizon = shock['horizon']
            shock_date = shock['survey_date']

            # Get post-shock data
            post_mask = (
                (revisions['variable'] == var) &
                (revisions['horizon'] == horizon) &
                (revisions['survey_date'] > shock_date) &
                (revisions['survey_date'] <= shock_date + pd.DateOffset(years=2))
            )

            post_shock = revisions[post_mask]
            if len(post_shock) < 10:
                continue

            # Settled consensus
            late_mask = post_shock['survey_date'] > shock_date + pd.DateOffset(months=15)
            if late_mask.sum() < 5:
                continue
            settled = post_shock[late_mask]['forecast'].median()

            # Consensus direction (shock moved consensus up or down?)
            pre_mask = (
                (revisions['variable'] == var) &
                (revisions['horizon'] == horizon) &
                (revisions['survey_date'] <= shock_date) &
                (revisions['survey_date'] > shock_date - pd.DateOffset(months=3))
            )
            pre_consensus = revisions[pre_mask]['forecast'].median()
            if np.isnan(pre_consensus):
                continue

            shock_direction = np.sign(settled - pre_consensus)

            # For each forecaster, measure overshoot
            for fid in post_shock['forecaster_id'].unique():
                f_post = post_shock[post_shock['forecaster_id'] == fid].sort_values('survey_date')

                if len(f_post) < 3:
                    continue

                # Maximum deviation past the settled consensus
                deviations = (f_post['forecast'] - settled) * shock_direction
                max_overshoot = deviations.max()

                if max_overshoot <= 0:  # No overshoot
                    continue

                # Get mass proxy
                f_feat = features[features['forecaster_id'] == fid]
                if len(f_feat) == 0:
                    continue

                overshoot_data.append({
                    'forecaster_id': fid,
                    'shock_date': shock_date,
                    'variable': var,
                    'experience': f_feat['quarters_active'].iloc[0],
                    'mass_proxy': np.log1p(f_feat['quarters_active'].iloc[0]),
                    'sqrt_mass': np.sqrt(np.log1p(f_feat['quarters_active'].iloc[0])),
                    'overshoot': max_overshoot,
                    'initial_distance': abs(pre_consensus - settled),
                })

        overshoot_df = pd.DataFrame(overshoot_data)

        if len(overshoot_df) < 20:
            print(f"WARNING: Only {len(overshoot_df)} overshoot episodes found")
            return {'overshoot': None}

        print(f"\nAnalyzed {len(overshoot_df)} overshoot episodes")

        # Test: overshoot ∝ √(mass)
        corr_sqrt, p_sqrt = stats.spearmanr(
            overshoot_df['sqrt_mass'],
            overshoot_df['overshoot']
        )

        # Also test linear scaling (alternative hypothesis)
        corr_lin, p_lin = stats.spearmanr(
            overshoot_df['mass_proxy'],
            overshoot_df['overshoot']
        )

        print(f"\nOvershoot vs √(mass):")
        print(f"  Spearman ρ = {corr_sqrt:.4f}, p = {p_sqrt:.4e}")
        print(f"\nOvershoot vs mass (linear):")
        print(f"  Spearman ρ = {corr_lin:.4f}, p = {p_lin:.4e}")

        # Theory predicts √ scaling; if √ fits better than linear, that's evidence
        print(f"\n{'√(mass) fits better' if abs(corr_sqrt) > abs(corr_lin) else 'Linear mass fits better'}")

        result = {
            'n_episodes': len(overshoot_df),
            'corr_sqrt_mass': corr_sqrt,
            'p_sqrt_mass': p_sqrt,
            'corr_linear_mass': corr_lin,
            'p_linear_mass': p_lin,
            'sqrt_better': abs(corr_sqrt) > abs(corr_lin),
            'significant': p_sqrt < 0.05 and corr_sqrt > 0,
            'overshoot_df': overshoot_df,
        }

        if result['significant']:
            print(f"\n>>> RESULT: Overshoot scales with √(mass) — consistent with H1.1")
        else:
            print(f"\n>>> RESULT: No significant overshoot-mass scaling")

        return result

    def run_all_tests(self) -> Dict:
        """Run all epistemic inertia tests."""
        data = self.load_latest()

        results = {}
        results['oscillation'] = self.test_oscillation(data['revisions'])
        results['relaxation'] = self.test_relaxation_time(
            data['revisions'], data['shocks'], data['features']
        )
        results['overshoot'] = self.test_overshoot(
            data['revisions'], data['shocks'], data['features']
        )

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY: SPF EPISTEMIC INERTIA TESTS")
        print("=" * 70)

        tests = [
            ('H2.1 Oscillation', results['oscillation']),
            ('H1.2 Relaxation', results['relaxation']),
            ('H1.1 Overshoot', results['overshoot']),
        ]

        for name, r in tests:
            if r is None:
                status = "SKIPPED (insufficient data)"
            elif r.get('significant'):
                status = "CONFIRMED"
            else:
                status = "NOT CONFIRMED"
            print(f"  {name}: {status}")

        return results


if __name__ == '__main__':
    analyzer = SPFInertiaAnalyzer(data_dir='data')
    results = analyzer.run_all_tests()
