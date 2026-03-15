"""Run complete SPF epistemic inertia pipeline: fetch → analyze."""

import sys
from pathlib import Path

from fetch_data import SPFDataFetcher
from analyze_inertia import SPFInertiaAnalyzer


def main():
    data_dir = 'data'
    skip_fetch = '--skip-fetch' in sys.argv

    if not skip_fetch:
        print("STEP 1: Fetching SPF data...")
        fetcher = SPFDataFetcher(output_dir=data_dir)
        fetcher.run_pipeline()
    else:
        print("STEP 1: Skipping fetch (using cached data)")

    print("\nSTEP 2: Running epistemic inertia analysis...")
    analyzer = SPFInertiaAnalyzer(data_dir=data_dir)
    results = analyzer.run_all_tests()

    print("\nPipeline complete!")
    return results


if __name__ == '__main__':
    main()
