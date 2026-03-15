"""
Wikipedia Edit Wars as Belief Oscillation

Direct test of H2.1 (Belief Oscillation): Articles in edit wars exhibit
oscillation with frequency ω ≈ √(K/M), where K is evidence strength
and M is editor epistemic mass.

Edit wars are the most direct observable manifestation of belief oscillation
in the real world: article content literally oscillates between two states
as editors revert each other.

Framework mapping:
    M_i = Λ_p + Λ_o + Σ_k β_ik Λ̃_qk + Σ_j β_ji Λ_qi

    Λ_p → log(editor_total_edits)   [experience/prior precision]
    Λ_o → edits_in_topic_category   [domain expertise]
    Σ β_ji → page_watchers          [how many watch their edits]

    Oscillation prediction:
        ω = √(K/M) where K = # reliable sources supporting each side
        Higher mass editors → lower oscillation frequency
        τ = 2M/γ = period of oscillation

Data source: Wikipedia MediaWiki API (fully public, no auth)
"""

import numpy as np
import pandas as pd
import requests
import time
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from scipy import stats, signal


# Articles known for edit wars (from Wikipedia:Lamest_edit_wars and similar)
CONTENTIOUS_ARTICLES = [
    'Abortion',
    'Gun_control',
    'Climate_change',
    'Evolution',
    'Israeli%E2%80%93Palestinian_conflict',
    'Homosexuality',
    'Intelligent_design',
    'Kashmir_conflict',
    'Crimea',
    'COVID-19_pandemic',
    'Donald_Trump',
    'Hillary_Clinton',
    'Chiropractic',
    'Homeopathy',
    'Acupuncture',
    'Circumcision',
    'Anarchism',
    'Capitalism',
    'Communism',
    'Genetically_modified_food',
]

# Control articles (low controversy)
CONTROL_ARTICLES = [
    'Mathematics',
    'Solar_System',
    'Water',
    'Carbon',
    'DNA',
    'Photosynthesis',
    'Periodic_table',
    'Speed_of_light',
    'Oxygen',
    'Hydrogen',
]


class WikipediaOscillationDetector:
    """Detect and analyze belief oscillation in Wikipedia edit wars."""

    API_URL = 'https://en.wikipedia.org/w/api.php'

    def __init__(self, output_dir: str = 'data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'EpistemicInertiaResearch/1.0 (academic research; epistemic.inertia@research.org)'
        })

    def fetch_revisions(self, title: str,
                         limit: int = 500) -> List[Dict]:
        """Fetch revision history for an article."""
        revisions = []
        params = {
            'action': 'query',
            'titles': title,
            'prop': 'revisions',
            'rvprop': 'ids|timestamp|user|userid|size|comment|sha1',
            'rvlimit': min(limit, 500),
            'rvdir': 'newer',  # Oldest first
            'format': 'json',
        }

        while len(revisions) < limit:
            try:
                response = self.session.get(self.API_URL, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                pages = data.get('query', {}).get('pages', {})
                for page_id, page_data in pages.items():
                    for rev in page_data.get('revisions', []):
                        revisions.append({
                            'title': title,
                            'revid': rev['revid'],
                            'parentid': rev.get('parentid', 0),
                            'timestamp': rev['timestamp'],
                            'user': rev.get('user', ''),
                            'userid': rev.get('userid', 0),
                            'size': rev.get('size', 0),
                            'comment': rev.get('comment', ''),
                            'sha1': rev.get('sha1', ''),
                        })

                # Check for continuation
                if 'continue' in data:
                    params['rvcontinue'] = data['continue']['rvcontinue']
                else:
                    break

            except Exception as e:
                print(f"  Error fetching revisions for {title}: {e}")
                break

            time.sleep(0.5)

        return revisions

    def detect_reverts(self, revisions: List[Dict]) -> pd.DataFrame:
        """
        Detect reverts in a revision sequence.

        A revert occurs when revision N has the same SHA1 as an earlier
        revision M, meaning the content was restored to a previous state.
        """
        df = pd.DataFrame(revisions)
        if len(df) == 0:
            return pd.DataFrame()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Track SHA1 hashes to detect content restoration
        sha1_first_seen = {}
        reverts = []

        for idx, row in df.iterrows():
            sha1 = row['sha1']
            if not sha1:
                continue

            if sha1 in sha1_first_seen:
                # This is a revert — content matches an earlier version
                original_idx = sha1_first_seen[sha1]
                reverts.append({
                    'revert_revid': row['revid'],
                    'revert_user': row['user'],
                    'revert_timestamp': row['timestamp'],
                    'original_revid': original_idx,
                    'title': row['title'],
                })
            else:
                sha1_first_seen[sha1] = row['revid']

        # Also detect reverts from edit comments
        revert_keywords = ['revert', 'rvv', 'undid', 'undo', 'restored',
                          'rv/', 'reverted']

        for idx, row in df.iterrows():
            comment = str(row.get('comment', '')).lower()
            if any(kw in comment for kw in revert_keywords):
                # Check if already captured by SHA1
                existing = [r for r in reverts if r['revert_revid'] == row['revid']]
                if not existing:
                    reverts.append({
                        'revert_revid': row['revid'],
                        'revert_user': row['user'],
                        'revert_timestamp': row['timestamp'],
                        'original_revid': None,
                        'title': row['title'],
                    })

        return pd.DataFrame(reverts) if reverts else pd.DataFrame()

    def compute_oscillation_metrics(self, revisions: List[Dict],
                                      reverts: pd.DataFrame) -> Dict:
        """
        Compute oscillation metrics for an article.

        Key metrics:
        - Revert rate: reverts / total edits
        - Oscillation period: median time between consecutive reverts
        - Oscillation frequency: 1 / period
        - Edit war intensity: reverts per unit time
        """
        df = pd.DataFrame(revisions)
        if len(df) == 0 or len(reverts) == 0:
            return {}

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        reverts = reverts.copy()
        reverts['revert_timestamp'] = pd.to_datetime(reverts['revert_timestamp'])

        total_edits = len(df)
        total_reverts = len(reverts)
        revert_rate = total_reverts / max(total_edits, 1)

        # Time span
        time_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400  # Days

        # Oscillation period (time between consecutive reverts)
        revert_times = reverts['revert_timestamp'].sort_values()
        if len(revert_times) > 1:
            inter_revert_intervals = revert_times.diff().dt.total_seconds() / 3600  # Hours
            inter_revert_intervals = inter_revert_intervals.dropna()

            median_period_hours = inter_revert_intervals.median()
            mean_period_hours = inter_revert_intervals.mean()

            # Oscillation frequency
            if median_period_hours > 0:
                frequency = 1.0 / median_period_hours  # Per hour
            else:
                frequency = np.nan
        else:
            median_period_hours = np.nan
            mean_period_hours = np.nan
            frequency = np.nan

        # Editor mass statistics
        editor_edit_counts = df.groupby('user').size()
        unique_reverters = reverts['revert_user'].nunique()

        return {
            'total_edits': total_edits,
            'total_reverts': total_reverts,
            'revert_rate': revert_rate,
            'time_span_days': time_span,
            'median_revert_period_hours': median_period_hours,
            'mean_revert_period_hours': mean_period_hours,
            'oscillation_frequency': frequency,
            'unique_editors': len(editor_edit_counts),
            'unique_reverters': unique_reverters,
            'mean_editor_edits': editor_edit_counts.mean(),
            'max_editor_edits': editor_edit_counts.max(),
        }

    def fetch_editor_mass(self, username: str) -> Optional[Dict]:
        """Fetch editor statistics as mass proxies."""
        params = {
            'action': 'query',
            'list': 'users',
            'ususers': username,
            'usprop': 'editcount|registration|groups',
            'format': 'json',
        }

        try:
            response = self.session.get(self.API_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            users = data.get('query', {}).get('users', [])
            if not users:
                return None

            user = users[0]
            if 'missing' in user:
                return None

            edit_count = user.get('editcount', 0)
            registration = user.get('registration', '')
            groups = user.get('groups', [])
            is_admin = 'sysop' in groups

            return {
                'username': username,
                'edit_count': edit_count,
                'mass_proxy': np.log1p(edit_count),  # Λ_p
                'is_admin': is_admin,
                'registration': registration,
                'groups': ','.join(groups),
            }

        except Exception:
            return None

    def analyze_article(self, title: str, max_revisions: int = 500) -> Dict:
        """Complete analysis of one article."""
        print(f"\nAnalyzing: {title}")

        # Fetch revisions
        revisions = self.fetch_revisions(title, limit=max_revisions)
        print(f"  {len(revisions)} revisions fetched")

        if len(revisions) < 10:
            return {}

        # Detect reverts
        reverts = self.detect_reverts(revisions)
        print(f"  {len(reverts)} reverts detected")

        # Compute oscillation metrics
        metrics = self.compute_oscillation_metrics(revisions, reverts)
        metrics['title'] = title

        # Fetch mass data for top editors
        df = pd.DataFrame(revisions)
        top_editors = df['user'].value_counts().head(20).index.tolist()
        editor_masses = []

        for editor in top_editors:
            mass_data = self.fetch_editor_mass(editor)
            if mass_data:
                editor_masses.append(mass_data)
            time.sleep(0.3)

        if editor_masses:
            mass_df = pd.DataFrame(editor_masses)
            metrics['mean_editor_mass'] = mass_df['mass_proxy'].mean()
            metrics['max_editor_mass'] = mass_df['mass_proxy'].max()
            metrics['admin_fraction'] = mass_df['is_admin'].mean()
        else:
            metrics['mean_editor_mass'] = np.nan
            metrics['max_editor_mass'] = np.nan
            metrics['admin_fraction'] = np.nan

        print(f"  Revert rate: {metrics.get('revert_rate', 0):.3f}")
        print(f"  Oscillation period: {metrics.get('median_revert_period_hours', np.nan):.1f} hours")
        print(f"  Mean editor mass: {metrics.get('mean_editor_mass', np.nan):.2f}")

        return metrics

    def run_full_analysis(self) -> pd.DataFrame:
        """
        Analyze all contentious and control articles.

        Tests:
        1. Contentious articles have higher oscillation frequency than controls
        2. Oscillation frequency inversely correlates with editor mass: ω ∝ 1/√M
        3. Revert rate correlates with editor mass asymmetry
        """
        print("=" * 70)
        print("WIKIPEDIA EDIT WAR OSCILLATION ANALYSIS (H2.1)")
        print("=" * 70)

        all_results = []

        # Contentious articles
        print("\n--- CONTENTIOUS ARTICLES ---")
        for title in CONTENTIOUS_ARTICLES:
            result = self.analyze_article(title)
            if result:
                result['category'] = 'contentious'
                all_results.append(result)
            time.sleep(1)

        # Control articles
        print("\n--- CONTROL ARTICLES ---")
        for title in CONTROL_ARTICLES:
            result = self.analyze_article(title)
            if result:
                result['category'] = 'control'
                all_results.append(result)
            time.sleep(1)

        if not all_results:
            print("ERROR: No results")
            return pd.DataFrame()

        df = pd.DataFrame(all_results)
        df.to_csv(self.output_dir / 'oscillation_results.csv', index=False)

        # Statistical tests
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        contentious = df[df['category'] == 'contentious']
        control = df[df['category'] == 'control']

        # Test 1: Contentious articles have higher revert rates
        if len(contentious) > 0 and len(control) > 0:
            u_stat, p_val = stats.mannwhitneyu(
                contentious['revert_rate'].dropna(),
                control['revert_rate'].dropna(),
                alternative='greater'
            )
            print(f"\nTest 1: Contentious vs Control revert rates")
            print(f"  Contentious mean: {contentious['revert_rate'].mean():.4f}")
            print(f"  Control mean: {control['revert_rate'].mean():.4f}")
            print(f"  Mann-Whitney p = {p_val:.4e}")
            print(f"  {'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'}")

        # Test 2: Oscillation frequency vs editor mass
        valid = df[df['oscillation_frequency'].notna() & df['mean_editor_mass'].notna()]
        if len(valid) > 5:
            corr, p_val = stats.spearmanr(
                valid['mean_editor_mass'],
                valid['oscillation_frequency']
            )
            print(f"\nTest 2: Editor mass vs oscillation frequency")
            print(f"  Spearman ρ = {corr:.4f}, p = {p_val:.4e}")
            print(f"  Prediction: ρ < 0 (higher mass → lower frequency)")
            print(f"  {'CONSISTENT with ω ∝ 1/√M' if corr < 0 else 'INCONSISTENT'}")

        # Test 3: Oscillation frequency vs √(1/mass)
        valid['inv_sqrt_mass'] = 1.0 / np.sqrt(valid['mean_editor_mass'])
        if len(valid) > 5:
            corr2, p_val2 = stats.spearmanr(
                valid['inv_sqrt_mass'],
                valid['oscillation_frequency']
            )
            print(f"\nTest 3: 1/√(mass) vs oscillation frequency")
            print(f"  Spearman ρ = {corr2:.4f}, p = {p_val2:.4e}")
            print(f"  {'POSITIVE (expected)' if corr2 > 0 else 'Not as expected'}")

        return df


if __name__ == '__main__':
    detector = WikipediaOscillationDetector(output_dir='data')
    results = detector.run_full_analysis()
    print("\nAnalysis complete!")
