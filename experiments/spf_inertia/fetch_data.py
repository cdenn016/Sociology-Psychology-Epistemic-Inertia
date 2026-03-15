"""
Survey of Professional Forecasters — Data Fetcher

Downloads individual-level panel data from the Federal Reserve Bank of
Philadelphia to test epistemic inertia predictions:
  H2.1: Belief oscillation after macroeconomic shocks
  H1.2: Relaxation time scales with forecaster experience (precision)
  H1.1: Overshoot magnitude scales with sqrt(precision)

Data source: https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/
Fully public, no authentication required.
"""

import requests
import zipfile
import io
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


class SPFDataFetcher:
    """Fetch Survey of Professional Forecasters individual-level data."""

    # Public download URLs for SPF microdata
    URLS = {
        'gdp': 'https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/survey-of-professional-forecasters/data-files/files/individual_rgdp.xlsx',
        'inflation': 'https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/survey-of-professional-forecasters/data-files/files/individual_cpi.xlsx',
        'unemployment': 'https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/survey-of-professional-forecasters/data-files/files/individual_unemp.xlsx',
        'tbill': 'https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/survey-of-professional-forecasters/data-files/files/individual_tbill.xlsx',
    }

    # Alternative: CSV format
    URLS_CSV = {
        'gdp': 'https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/survey-of-professional-forecasters/data-files/files/individual_rgdp.csv',
        'inflation': 'https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/survey-of-professional-forecasters/data-files/files/individual_cpi.csv',
        'unemployment': 'https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/survey-of-professional-forecasters/data-files/files/individual_unemp.csv',
    }

    def __init__(self, output_dir: str = 'data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'EpistemicInertiaResearch/1.0 (academic research)'
        })

    def fetch_variable(self, variable: str = 'gdp') -> pd.DataFrame:
        """
        Fetch individual-level forecasts for a variable.

        The SPF data has columns:
        - ID: forecaster identifier (anonymized but consistent)
        - YEAR, QUARTER: survey date
        - RGDP1..RGDP6 (or CPI1..CPI6): forecasts for different horizons

        Returns:
            DataFrame with forecaster panel data
        """
        # Try CSV first, then Excel
        for url_dict, reader in [(self.URLS_CSV, 'csv'), (self.URLS, 'excel')]:
            if variable not in url_dict:
                continue

            url = url_dict[variable]
            print(f"Fetching {variable} forecasts from {url}...")

            try:
                response = self.session.get(url, timeout=60)
                response.raise_for_status()

                if reader == 'csv':
                    df = pd.read_csv(io.StringIO(response.text))
                else:
                    df = pd.read_excel(io.BytesIO(response.content))

                print(f"  Downloaded {len(df)} rows, {len(df.columns)} columns")
                print(f"  Columns: {list(df.columns[:10])}...")
                return df

            except Exception as e:
                print(f"  Failed with {reader}: {e}")
                continue

        raise RuntimeError(f"Could not fetch {variable} data from any source")

    def build_panel(self, variables: list = None) -> pd.DataFrame:
        """
        Build a tidy panel dataset from raw SPF data.

        Reshapes wide-format forecasts into long format:
        (forecaster_id, year, quarter, variable, horizon, forecast)
        """
        if variables is None:
            variables = ['gdp', 'inflation', 'unemployment']

        all_data = []

        for var in variables:
            try:
                raw = self.fetch_variable(var)
            except Exception as e:
                print(f"Skipping {var}: {e}")
                continue

            # Identify ID and date columns
            id_col = [c for c in raw.columns if c.upper() in ('ID', 'INDID', 'IND')][0] \
                if any(c.upper() in ('ID', 'INDID', 'IND') for c in raw.columns) else raw.columns[0]

            year_col = [c for c in raw.columns if 'YEAR' in c.upper()][0] \
                if any('YEAR' in c.upper() for c in raw.columns) else raw.columns[1]

            quarter_col = [c for c in raw.columns if 'QUARTER' in c.upper() or c.upper() == 'QTR'][0] \
                if any('QUARTER' in c.upper() or c.upper() == 'QTR' for c in raw.columns) else raw.columns[2]

            # Identify forecast columns (variable prefix + horizon number)
            var_prefix = {
                'gdp': ['RGDP', 'NGDP', 'GDP'],
                'inflation': ['CPI', 'CORECPI', 'PCE'],
                'unemployment': ['UNEMP', 'UNRATE'],
            }.get(var, [var.upper()])

            forecast_cols = []
            for col in raw.columns:
                for prefix in var_prefix:
                    if col.upper().startswith(prefix) and any(c.isdigit() for c in col):
                        forecast_cols.append(col)
                        break

            if not forecast_cols:
                print(f"  Warning: no forecast columns found for {var}")
                print(f"  Available columns: {list(raw.columns)}")
                continue

            print(f"  Found {len(forecast_cols)} forecast columns for {var}: {forecast_cols[:5]}...")

            # Melt to long format
            for fc in forecast_cols:
                # Extract horizon from column name
                horizon = ''.join(c for c in fc if c.isdigit())
                if not horizon:
                    continue

                subset = raw[[id_col, year_col, quarter_col, fc]].copy()
                subset.columns = ['forecaster_id', 'year', 'quarter', 'forecast']
                subset['variable'] = var
                subset['horizon'] = int(horizon)
                subset['forecast'] = pd.to_numeric(subset['forecast'], errors='coerce')
                subset = subset.dropna(subset=['forecast'])

                all_data.append(subset)

        if not all_data:
            raise RuntimeError("No forecast data could be parsed")

        panel = pd.concat(all_data, ignore_index=True)

        # Sort and compute survey date
        panel['survey_date'] = pd.to_datetime(
            panel['year'].astype(str) + 'Q' + panel['quarter'].astype(str)
        )

        panel = panel.sort_values(['forecaster_id', 'variable', 'horizon', 'survey_date'])

        print(f"\nBuilt panel: {len(panel)} observations")
        print(f"  Forecasters: {panel['forecaster_id'].nunique()}")
        print(f"  Variables: {panel['variable'].unique()}")
        print(f"  Date range: {panel['survey_date'].min()} to {panel['survey_date'].max()}")

        return panel

    def compute_revisions(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Compute forecast revisions (Δforecast) between consecutive surveys.

        This is the key belief update measure: how much does forecaster i
        change their forecast from quarter t to quarter t+1?
        """
        panel = panel.sort_values(['forecaster_id', 'variable', 'horizon', 'survey_date'])

        # Compute revision = forecast_t - forecast_{t-1}
        panel['prev_forecast'] = panel.groupby(
            ['forecaster_id', 'variable', 'horizon']
        )['forecast'].shift(1)

        panel['revision'] = panel['forecast'] - panel['prev_forecast']
        panel['abs_revision'] = panel['revision'].abs()

        # Compute revision sign (for oscillation detection)
        panel['revision_sign'] = np.sign(panel['revision'])
        panel['prev_revision_sign'] = panel.groupby(
            ['forecaster_id', 'variable', 'horizon']
        )['revision_sign'].shift(1)

        # Sign change = potential oscillation
        panel['sign_change'] = (
            (panel['revision_sign'] != panel['prev_revision_sign']) &
            (panel['revision_sign'] != 0) &
            (panel['prev_revision_sign'] != 0)
        ).astype(int)

        revisions = panel.dropna(subset=['revision'])

        print(f"\nRevisions computed: {len(revisions)} observations")
        print(f"  Mean |revision|: {revisions['abs_revision'].mean():.4f}")
        print(f"  Sign changes: {revisions['sign_change'].sum()} ({revisions['sign_change'].mean():.1%})")

        return revisions

    def compute_forecaster_features(self, revisions: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-forecaster features as mass matrix proxies.

        Mass components:
        - Λ_p (prior precision) → experience = quarters in panel
        - Λ_o (observation precision) → accuracy = 1 / RMSE
        - Incoming social → consensus proximity
        - Outgoing social → influence on subsequent consensus
        """
        features = []

        for fid in revisions['forecaster_id'].unique():
            f_data = revisions[revisions['forecaster_id'] == fid]

            # Experience (quarters in panel)
            quarters_active = f_data['survey_date'].nunique()

            # First and last survey date
            first_survey = f_data['survey_date'].min()
            last_survey = f_data['survey_date'].max()

            # Mean absolute revision (inverse = "steadiness")
            mean_abs_rev = f_data['abs_revision'].mean()

            # Sign change rate (oscillation frequency)
            sign_change_rate = f_data['sign_change'].mean()

            # Revision magnitude stats
            rev_std = f_data['revision'].std()

            features.append({
                'forecaster_id': fid,
                'quarters_active': quarters_active,
                'first_survey': first_survey,
                'last_survey': last_survey,
                'mean_abs_revision': mean_abs_rev,
                'revision_std': rev_std,
                'sign_change_rate': sign_change_rate,
                'total_revisions': len(f_data),
            })

        features_df = pd.DataFrame(features)

        # Compute mass proxy (experience-weighted)
        features_df['mass_proxy'] = np.log1p(features_df['quarters_active'])

        print(f"\nForecaster features: {len(features_df)} forecasters")
        print(f"  Mean quarters active: {features_df['quarters_active'].mean():.1f}")
        print(f"  Mean |revision|: {features_df['mean_abs_revision'].mean():.4f}")

        return features_df

    def identify_shocks(self, panel: pd.DataFrame,
                        threshold_sd: float = 1.0) -> pd.DataFrame:
        """
        Identify macroeconomic shocks as quarters where consensus forecast
        shifts by more than threshold_sd standard deviations.

        These are the "evidence events" that test whether forecasters
        exhibit oscillation, overshoot, and differential relaxation.
        """
        # Compute consensus (median) forecast per quarter per variable per horizon
        consensus = panel.groupby(
            ['variable', 'horizon', 'year', 'quarter']
        )['forecast'].agg(['median', 'std', 'count']).reset_index()

        consensus.columns = ['variable', 'horizon', 'year', 'quarter',
                            'consensus_median', 'consensus_std', 'n_forecasters']

        consensus['survey_date'] = pd.to_datetime(
            consensus['year'].astype(str) + 'Q' + consensus['quarter'].astype(str)
        )

        consensus = consensus.sort_values(['variable', 'horizon', 'survey_date'])

        # Compute consensus change
        consensus['prev_consensus'] = consensus.groupby(
            ['variable', 'horizon']
        )['consensus_median'].shift(1)

        consensus['consensus_change'] = consensus['consensus_median'] - consensus['prev_consensus']
        consensus['abs_consensus_change'] = consensus['consensus_change'].abs()

        # Rolling standard deviation for normalization
        consensus['rolling_std'] = consensus.groupby(
            ['variable', 'horizon']
        )['consensus_change'].transform(
            lambda x: x.rolling(window=8, min_periods=4).std()
        )

        # Identify shocks
        consensus['is_shock'] = (
            consensus['abs_consensus_change'] >
            threshold_sd * consensus['rolling_std']
        )

        shocks = consensus[consensus['is_shock']].copy()

        print(f"\nIdentified {len(shocks)} shock events (threshold: {threshold_sd} SD)")
        for var in shocks['variable'].unique():
            n = len(shocks[shocks['variable'] == var])
            print(f"  {var}: {n} shocks")

        return shocks, consensus

    def run_pipeline(self, variables: list = None) -> dict:
        """Run complete data collection and preprocessing pipeline."""
        print("=" * 70)
        print("SPF EPISTEMIC INERTIA DATA PIPELINE")
        print("=" * 70)

        # Step 1: Fetch data
        panel = self.build_panel(variables)

        # Step 2: Compute revisions
        revisions = self.compute_revisions(panel)

        # Step 3: Compute forecaster features
        features = self.compute_forecaster_features(revisions)

        # Step 4: Identify shocks
        shocks, consensus = self.identify_shocks(panel)

        # Step 5: Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        panel.to_csv(self.output_dir / f'panel_{timestamp}.csv', index=False)
        revisions.to_csv(self.output_dir / f'revisions_{timestamp}.csv', index=False)
        features.to_csv(self.output_dir / f'features_{timestamp}.csv', index=False)
        shocks.to_csv(self.output_dir / f'shocks_{timestamp}.csv', index=False)
        consensus.to_csv(self.output_dir / f'consensus_{timestamp}.csv', index=False)

        print(f"\nAll data saved to {self.output_dir}/")

        return {
            'panel': panel,
            'revisions': revisions,
            'features': features,
            'shocks': shocks,
            'consensus': consensus,
        }


if __name__ == '__main__':
    fetcher = SPFDataFetcher(output_dir='data')
    data = fetcher.run_pipeline()
    print("\nData collection complete!")
