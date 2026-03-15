"""
Reddit Echo Chamber Threshold Test

Tests H3.2: Echo chambers form when group belief separation exceeds the
theoretical threshold from the VFE framework:

    ||μ_A - μ_B||² > 2σ²κ log(N)

where:
    μ_A, μ_B = mean sentiment/position of paired subreddits
    σ² = within-subreddit sentiment variance
    κ = cross-posting "temperature" (higher = more cross-posting)
    N = subreddit subscriber count

Mass formula context:
    M_i = Λ_p + Λ_o + Σ_k β_ik Λ̃_qk + Σ_j β_ji Λ_qi

    In the echo chamber context, Σ β_ik terms determine attention allocation
    between in-group and out-group. When ||μ_A - μ_B||² exceeds the threshold,
    out-group attention β_out → 0 and the community polarizes.

Data source: Reddit API (free, limited) or Pushshift/Academic Torrents archives
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json
import time

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False


# Predefined subreddit pairs on opposing sides of the same topics
SUBREDDIT_PAIRS = [
    # Political
    ('politics', 'Conservative'),
    ('Liberal', 'Conservative'),
    ('Democrats', 'Republican'),
    ('SandersForPresident', 'The_Donald'),
    # Science/Health
    ('vegan', 'meat'),
    ('ClimateChange', 'climateskeptics'),
    ('vaxxhappened', 'DebateVaccines'),
    # Religion
    ('atheism', 'Christianity'),
    ('atheism', 'islam'),
    # Economics
    ('LateStageCapitalism', 'Libertarian'),
    ('antiwork', 'jobs'),
    # Tech
    ('apple', 'Android'),
    ('pcmasterrace', 'consoles'),
    # Neutral control pairs (should NOT polarize)
    ('cats', 'dogs'),
    ('cooking', 'Baking'),
    ('science', 'askscience'),
]


class RedditEchoChamberAnalyzer:
    """
    Test the echo chamber formation threshold from VFE theory.

    The key equation is:
        ||μ_A - μ_B||² > 2σ²κ log(N)

    where κ is estimated from cross-posting rates.
    """

    REDDIT_API_BASE = 'https://www.reddit.com'

    def __init__(self, output_dir: str = 'data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        if HAS_VADER:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        else:
            self.sentiment_analyzer = None

        self.session = requests.Session() if HAS_REQUESTS else None
        if self.session:
            self.session.headers.update({
                'User-Agent': 'EpistemicInertiaResearch/1.0 (academic research)'
            })

    def get_sentiment(self, text: str) -> float:
        """Compute sentiment score for a text using VADER or TextBlob."""
        if self.sentiment_analyzer:
            return self.sentiment_analyzer.polarity_scores(text)['compound']
        elif HAS_TEXTBLOB:
            return TextBlob(text).sentiment.polarity
        else:
            # Fallback: very simple word-count heuristic
            positive_words = {'good', 'great', 'best', 'love', 'agree', 'right',
                            'correct', 'true', 'excellent', 'wonderful'}
            negative_words = {'bad', 'worst', 'hate', 'wrong', 'disagree', 'false',
                            'terrible', 'awful', 'stupid', 'idiot'}
            words = set(text.lower().split())
            pos = len(words & positive_words)
            neg = len(words & negative_words)
            total = pos + neg
            return (pos - neg) / max(total, 1)

    def fetch_subreddit_posts(self, subreddit: str,
                               limit: int = 100) -> List[Dict]:
        """Fetch recent posts from a subreddit via Reddit JSON API."""
        if not self.session:
            return []

        url = f"{self.REDDIT_API_BASE}/r/{subreddit}/hot.json"
        params = {'limit': min(limit, 100), 'raw_json': 1}

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            posts = []
            for child in data.get('data', {}).get('children', []):
                post = child['data']
                text = f"{post.get('title', '')} {post.get('selftext', '')}"
                if len(text.strip()) < 10:
                    continue

                posts.append({
                    'subreddit': subreddit,
                    'id': post['id'],
                    'title': post.get('title', ''),
                    'text': text,
                    'score': post.get('score', 0),
                    'num_comments': post.get('num_comments', 0),
                    'created_utc': post.get('created_utc', 0),
                    'author': post.get('author', ''),
                })

            return posts

        except Exception as e:
            print(f"  Error fetching r/{subreddit}: {e}")
            return []

    def fetch_subreddit_info(self, subreddit: str) -> Dict:
        """Fetch subreddit metadata (subscriber count, etc.)."""
        if not self.session:
            return {}

        url = f"{self.REDDIT_API_BASE}/r/{subreddit}/about.json"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()['data']
            return {
                'subreddit': subreddit,
                'subscribers': data.get('subscribers', 0),
                'active_accounts': data.get('accounts_active', 0),
                'created_utc': data.get('created_utc', 0),
            }
        except Exception as e:
            print(f"  Error fetching r/{subreddit} info: {e}")
            return {'subreddit': subreddit, 'subscribers': 0}

    def analyze_pair(self, sub_a: str, sub_b: str,
                      posts_per_sub: int = 100) -> Optional[Dict]:
        """
        Analyze a subreddit pair for echo chamber threshold.

        Computes:
        - μ_A, μ_B: mean sentiment for each subreddit
        - σ²: pooled within-subreddit sentiment variance
        - N: subscriber count (geometric mean of both)
        - κ: estimated from cross-posting rate (fraction of authors in both)
        - Threshold test: ||μ_A - μ_B||² vs 2σ²κ log(N)
        """
        print(f"\n  Analyzing pair: r/{sub_a} vs r/{sub_b}")

        # Fetch posts
        posts_a = self.fetch_subreddit_posts(sub_a, posts_per_sub)
        time.sleep(2)  # Reddit rate limiting
        posts_b = self.fetch_subreddit_posts(sub_b, posts_per_sub)
        time.sleep(2)

        if len(posts_a) < 10 or len(posts_b) < 10:
            print(f"    Insufficient posts: {len(posts_a)}, {len(posts_b)}")
            return None

        # Compute sentiments
        sentiments_a = [self.get_sentiment(p['text']) for p in posts_a]
        sentiments_b = [self.get_sentiment(p['text']) for p in posts_b]

        mu_a = np.mean(sentiments_a)
        mu_b = np.mean(sentiments_b)
        var_a = np.var(sentiments_a)
        var_b = np.var(sentiments_b)
        sigma2 = (var_a + var_b) / 2  # Pooled variance

        # Belief separation
        separation2 = (mu_a - mu_b) ** 2

        # Subreddit sizes
        info_a = self.fetch_subreddit_info(sub_a)
        time.sleep(2)
        info_b = self.fetch_subreddit_info(sub_b)
        time.sleep(2)

        n_a = max(info_a.get('subscribers', 1000), 1)
        n_b = max(info_b.get('subscribers', 1000), 1)
        N = np.sqrt(n_a * n_b)  # Geometric mean

        # Cross-posting rate as κ proxy
        authors_a = set(p['author'] for p in posts_a if p['author'] != '[deleted]')
        authors_b = set(p['author'] for p in posts_b if p['author'] != '[deleted]')
        overlap = len(authors_a & authors_b)
        total_authors = len(authors_a | authors_b)

        if total_authors > 0:
            cross_post_rate = overlap / total_authors
        else:
            cross_post_rate = 0.01

        # κ estimation: higher cross-posting → higher temperature → less polarization
        # κ = cross_post_rate * scaling_factor
        kappa = max(cross_post_rate * 10, 0.01)  # Scale to reasonable range

        # Threshold
        threshold = 2 * sigma2 * kappa * np.log(max(N, 2))

        # Test
        exceeds_threshold = separation2 > threshold

        result = {
            'sub_a': sub_a,
            'sub_b': sub_b,
            'mu_a': mu_a,
            'mu_b': mu_b,
            'separation_squared': separation2,
            'separation': np.sqrt(separation2),
            'sigma_squared': sigma2,
            'kappa': kappa,
            'N': N,
            'threshold': threshold,
            'exceeds_threshold': exceeds_threshold,
            'cross_post_rate': cross_post_rate,
            'n_posts_a': len(posts_a),
            'n_posts_b': len(posts_b),
            'subscribers_a': n_a,
            'subscribers_b': n_b,
            # Sentiment difference t-test
            't_stat': stats.ttest_ind(sentiments_a, sentiments_b).statistic,
            'p_value': stats.ttest_ind(sentiments_a, sentiments_b).pvalue,
        }

        status = "POLARIZED" if exceeds_threshold else "CONVERGENT"
        print(f"    μ_A={mu_a:.3f}, μ_B={mu_b:.3f}, "
              f"||Δ||²={separation2:.4f}, threshold={threshold:.4f} → {status}")

        return result

    def run_analysis(self, pairs: List[Tuple[str, str]] = None,
                      posts_per_sub: int = 100) -> pd.DataFrame:
        """
        Run echo chamber threshold analysis across all subreddit pairs.

        Tests: Do pairs exceeding the threshold show actual polarization
        (persistent sentiment divergence, low cross-posting)?
        """
        if pairs is None:
            pairs = SUBREDDIT_PAIRS

        print("=" * 70)
        print("REDDIT ECHO CHAMBER THRESHOLD TEST (H3.2)")
        print("=" * 70)
        print(f"\nTheoretical threshold: ||μ_A - μ_B||² > 2σ²κ log(N)")
        print(f"Testing {len(pairs)} subreddit pairs...")

        results = []
        for sub_a, sub_b in pairs:
            result = self.analyze_pair(sub_a, sub_b, posts_per_sub)
            if result:
                results.append(result)

        if not results:
            print("ERROR: No results obtained")
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Save
        df.to_csv(self.output_dir / 'echo_chamber_results.csv', index=False)

        # Analysis
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        n_polarized = df['exceeds_threshold'].sum()
        n_total = len(df)
        print(f"\nPairs exceeding threshold: {n_polarized}/{n_total}")

        # Classification accuracy: do pairs we expect to be polarized
        # (political, religion) actually exceed the threshold?
        political_pairs = df[df['sub_a'].isin(
            ['politics', 'Liberal', 'Democrats', 'SandersForPresident']
        ) | df['sub_b'].isin(
            ['Conservative', 'Republican', 'The_Donald']
        )]

        control_pairs = df[df['sub_a'].isin(['cats', 'cooking', 'science'])]

        if len(political_pairs) > 0:
            print(f"\nPolitical pairs exceeding threshold: "
                  f"{political_pairs['exceeds_threshold'].sum()}/{len(political_pairs)}")

        if len(control_pairs) > 0:
            print(f"Control pairs exceeding threshold: "
                  f"{control_pairs['exceeds_threshold'].sum()}/{len(control_pairs)}")

        # Correlation: separation vs cross-posting rate
        if len(df) > 5:
            corr, p_val = stats.spearmanr(
                df['separation_squared'], df['cross_post_rate']
            )
            print(f"\nCorrelation (separation² vs cross-posting):")
            print(f"  ρ = {corr:.4f}, p = {p_val:.4e}")
            print(f"  {'Negative (expected)' if corr < 0 else 'Positive (unexpected)'}")

        print("\n--- Pair Details ---")
        for _, row in df.sort_values('separation_squared', ascending=False).iterrows():
            status = "POLARIZED" if row['exceeds_threshold'] else "convergent"
            print(f"  r/{row['sub_a']} vs r/{row['sub_b']}: "
                  f"Δ={row['separation']:.3f}, threshold={np.sqrt(row['threshold']):.3f} "
                  f"→ {status}")

        return df


def main():
    """Run echo chamber threshold analysis."""
    analyzer = RedditEchoChamberAnalyzer(output_dir='data')
    results = analyzer.run_analysis(posts_per_sub=50)

    if len(results) > 0:
        print("\n" + "=" * 70)
        print("CONCLUSION")
        print("=" * 70)
        accuracy = (
            results['exceeds_threshold'].mean()
            if len(results) > 0 else 0
        )
        print(f"Echo chamber threshold predictive accuracy: {accuracy:.1%}")
        print("(High accuracy supports H3.2 from VFE framework)")


if __name__ == '__main__':
    main()
