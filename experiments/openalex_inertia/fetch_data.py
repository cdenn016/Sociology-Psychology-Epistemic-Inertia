"""
OpenAlex + CrossRef Data Fetcher for Epistemic Inertia via Retraction Events

Uses paper retractions as natural experiments to measure epistemic inertia:
after a paper is retracted, how long do citing authors continue to cite it?

The "relaxation time" (years until an author stops citing the retracted work)
is predicted to scale with the author's epistemic mass.

Mass formula mapping (from Hamiltonian belief dynamics):
    M_i = Lambda_p + Lambda_o + Sigma beta_ik Lambda_qk + Sigma beta_ji Lambda_qi

    Lambda_p (prior precision)        = log(career_citation_count)
        Proxy for accumulated reputation / h-index.  Authors with large
        career citation counts have high prior precision, making them
        resistant to updating (high inertia).

    Lambda_o (observation precision)  = works_count in same concept/field
        Domain expertise: authors with many publications in the retracted
        paper's field have stronger observational models, contributing to
        mass in that epistemic direction.

    Sigma beta_ji (outgoing social)   = cited_by_count of the citing author
        How many OTHER researchers cite this author.  High values mean the
        author's beliefs propagate widely, creating social coupling terms
        that resist individual belief change.

    relaxation_time                   = years from retraction until author's
        last citation of the retracted work (or right-censored if still
        citing at data collection time).

Data sources:
    - OpenAlex API (https://api.openalex.org/) -- no auth, polite email header
    - CrossRef API (https://api.crossref.org/) -- for retraction metadata

Rate limiting: 100ms between requests (polite pool).
"""

import requests
import json
import time
import math
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENALEX_BASE = "https://api.openalex.org"
CROSSREF_BASE = "https://api.crossref.org"

# Polite-pool email -- replace with your own for higher rate limits
POLITE_EMAIL = "epistemic.inertia.research@example.org"

# Delay between API calls (seconds)
REQUEST_DELAY = 0.1

# Maximum items to page through per endpoint call
MAX_PAGES = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _openalex_headers() -> Dict[str, str]:
    """Return headers for polite OpenAlex access."""
    return {
        "User-Agent": f"EpistemicInertiaResearch/1.0 (mailto:{POLITE_EMAIL})",
        "Accept": "application/json",
    }


def _sleep():
    """Polite delay between API requests."""
    time.sleep(REQUEST_DELAY)


def _extract_year(date_str: Optional[str]) -> Optional[int]:
    """Extract year from an ISO-style date string."""
    if not date_str:
        return None
    try:
        return int(date_str[:4])
    except (ValueError, TypeError):
        return None


def _short_id(openalex_id: str) -> str:
    """Strip URL prefix from an OpenAlex ID: 'https://openalex.org/W123' -> 'W123'."""
    if openalex_id and "/" in openalex_id:
        return openalex_id.rsplit("/", 1)[-1]
    return openalex_id or ""


# ---------------------------------------------------------------------------
# Data Fetcher
# ---------------------------------------------------------------------------

class OpenAlexRetractionFetcher:
    """
    Fetch retraction-event citation data from OpenAlex and CrossRef.

    Pipeline
    --------
    1. Identify retracted papers (via OpenAlex type:retraction + CrossRef)
    2. For each retracted paper, collect pre-retraction citing works
    3. For each citing author, gather career statistics (mass proxies)
    4. Track post-retraction citations to compute relaxation time
    5. Save structured CSVs for downstream analysis
    """

    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.session = requests.Session()
        self.session.headers.update(_openalex_headers())

    # ------------------------------------------------------------------
    # Step 1: Find retracted papers
    # ------------------------------------------------------------------

    def fetch_retracted_papers_openalex(
        self,
        max_results: int = 50,
        concept_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Query OpenAlex for retraction notices and extract retracted DOIs.

        OpenAlex indexes retraction notices as works whose ``type`` is
        ``"retraction"``.  Each notice references the original retracted
        paper via its ``related_works`` field.

        Args:
            max_results: Cap on number of retraction notices to retrieve.
            concept_id: Optional OpenAlex concept ID to restrict field
                        (e.g. "C86803240" for Biology).

        Returns:
            List of dicts with keys:
                retraction_id, retracted_work_id, retraction_year,
                retracted_doi, retracted_title
        """
        print("Fetching retraction notices from OpenAlex ...")

        retracted = []
        cursor = "*"
        collected = 0

        while collected < max_results and cursor:
            params = {
                "filter": "type:retraction",
                "per_page": min(50, max_results - collected),
                "cursor": cursor,
                "mailto": POLITE_EMAIL,
            }
            if concept_id:
                params["filter"] += f",concepts.id:{concept_id}"

            try:
                resp = self.session.get(
                    f"{OPENALEX_BASE}/works", params=params, timeout=30
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                print(f"  OpenAlex request failed: {exc}")
                break

            results = data.get("results", [])
            if not results:
                break

            for work in results:
                retraction_year = _extract_year(
                    work.get("publication_date")
                )
                related = work.get("related_works", [])
                retracted_work_id = related[0] if related else None

                retracted.append({
                    "retraction_notice_id": _short_id(work.get("id", "")),
                    "retracted_work_id": _short_id(retracted_work_id) if retracted_work_id else None,
                    "retraction_year": retraction_year,
                    "retracted_doi": work.get("doi"),
                    "retraction_title": work.get("title"),
                })
                collected += 1
                if collected >= max_results:
                    break

            cursor = data.get("meta", {}).get("next_cursor")
            _sleep()

        print(f"  Found {len(retracted)} retraction notices")
        return retracted

    def fetch_retracted_papers_crossref(
        self, max_results: int = 50
    ) -> List[Dict]:
        """
        Query CrossRef for retraction notices using the update-type filter.

        CrossRef marks retractions with ``update-type: retraction`` and
        links back to the original DOI via ``update-to``.

        Returns:
            List of dicts with retracted_doi, retraction_year, source.
        """
        print("Fetching retraction notices from CrossRef ...")

        retracted = []
        offset = 0
        rows_per = 50

        while len(retracted) < max_results:
            params = {
                "filter": "update-type:retraction",
                "rows": min(rows_per, max_results - len(retracted)),
                "offset": offset,
                "mailto": POLITE_EMAIL,
            }

            try:
                resp = self.session.get(
                    f"{CROSSREF_BASE}/works", params=params, timeout=30
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                print(f"  CrossRef request failed: {exc}")
                break

            items = data.get("message", {}).get("items", [])
            if not items:
                break

            for item in items:
                update_to = item.get("update-to", [])
                retracted_doi = update_to[0].get("DOI") if update_to else None
                pub_date = item.get("published-print", item.get("published-online", {}))
                year_parts = pub_date.get("date-parts", [[None]])[0] if pub_date else [None]
                year = year_parts[0] if year_parts else None

                retracted.append({
                    "retraction_notice_doi": item.get("DOI"),
                    "retracted_doi": retracted_doi,
                    "retraction_year": year,
                    "source": "crossref",
                })

            offset += rows_per
            _sleep()

        print(f"  Found {len(retracted)} CrossRef retractions")
        return retracted

    # ------------------------------------------------------------------
    # Step 2: Get citing works for a retracted paper
    # ------------------------------------------------------------------

    def fetch_citing_works(
        self, work_id: str, max_citers: int = 200
    ) -> List[Dict]:
        """
        Retrieve works that cite a given OpenAlex work ID.

        For each citing work we record:
            - citing_work_id, citing_doi, publication_year
            - author IDs (first & last author for tractability)

        Args:
            work_id: Short OpenAlex work ID (e.g. "W2104976013").
            max_citers: Upper bound on citing works to collect.

        Returns:
            List of citing-work dicts.
        """
        citers = []
        cursor = "*"

        while len(citers) < max_citers and cursor:
            params = {
                "filter": f"cites:{work_id}",
                "per_page": min(50, max_citers - len(citers)),
                "cursor": cursor,
                "select": "id,doi,publication_date,authorships,cited_by_count",
                "mailto": POLITE_EMAIL,
            }

            try:
                resp = self.session.get(
                    f"{OPENALEX_BASE}/works", params=params, timeout=30
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                print(f"    Citing-works request failed for {work_id}: {exc}")
                break

            results = data.get("results", [])
            if not results:
                break

            for w in results:
                authors = w.get("authorships", [])
                author_ids = [
                    _short_id(a.get("author", {}).get("id", ""))
                    for a in authors
                    if a.get("author", {}).get("id")
                ]
                citers.append({
                    "citing_work_id": _short_id(w.get("id", "")),
                    "citing_doi": w.get("doi"),
                    "citation_year": _extract_year(w.get("publication_date")),
                    "cited_by_count": w.get("cited_by_count", 0),
                    "author_ids": author_ids,
                })

            cursor = data.get("meta", {}).get("next_cursor")
            _sleep()

        return citers

    # ------------------------------------------------------------------
    # Step 3: Fetch author career statistics (mass proxies)
    # ------------------------------------------------------------------

    def fetch_author_stats(self, author_id: str) -> Optional[Dict]:
        """
        Retrieve career-level statistics for a single OpenAlex author.

        Mass proxy mapping:
            Lambda_p  = log(cited_by_count + 1)   [career citation count]
            Lambda_o  = works_count                [publication volume]
            beta_ji   = cited_by_count             [social outgoing influence]

        Args:
            author_id: Short OpenAlex author ID (e.g. "A5023888391").

        Returns:
            Dict with author metadata and mass proxy fields, or None.
        """
        url = f"{OPENALEX_BASE}/authors/{author_id}"
        params = {"mailto": POLITE_EMAIL}

        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            a = resp.json()
        except Exception:
            return None

        cited_by = a.get("cited_by_count", 0)
        works_count = a.get("works_count", 0)

        # Concept breakdown: count of works per top-level concept
        concepts = a.get("x_concepts", []) or []
        top_concept = concepts[0].get("display_name", "unknown") if concepts else "unknown"
        top_concept_count = concepts[0].get("count", 0) if concepts else 0

        return {
            "author_id": _short_id(a.get("id", "")),
            "display_name": a.get("display_name"),
            "works_count": works_count,
            "cited_by_count": cited_by,
            "h_index": a.get("summary_stats", {}).get("h_index", 0),
            "i10_index": a.get("summary_stats", {}).get("i10_index", 0),
            "top_concept": top_concept,
            "top_concept_count": top_concept_count,
            # Mass proxies (pre-computed for convenience)
            "lambda_p": math.log(cited_by + 1),
            "lambda_o": works_count,
            "beta_ji_outgoing": cited_by,
        }

    # ------------------------------------------------------------------
    # Step 4: Track post-retraction citation behaviour per author
    # ------------------------------------------------------------------

    def track_post_retraction_citations(
        self,
        author_id: str,
        retracted_work_id: str,
        retraction_year: int,
        max_works: int = 500,
    ) -> Dict:
        """
        Determine whether *author_id* continued citing *retracted_work_id*
        after it was retracted and, if so, for how many years.

        Searches the author's publications after *retraction_year* and
        checks whether any of them cite the retracted work.

        Returns:
            Dict with:
                post_retraction_citation_count: int
                last_citation_year: int or None
                relaxation_time_years: float or None (years after retraction
                    until last citation; None = never cited post-retraction)
                still_citing: bool (True if cited in the last 2 years)
        """
        post_citations = 0
        last_cite_year = None
        cursor = "*"
        pages = 0

        while cursor and pages < MAX_PAGES:
            params = {
                "filter": (
                    f"authorships.author.id:{author_id},"
                    f"cites:{retracted_work_id},"
                    f"from_publication_date:{retraction_year}-01-01"
                ),
                "per_page": 50,
                "cursor": cursor,
                "select": "id,publication_date",
                "mailto": POLITE_EMAIL,
            }

            try:
                resp = self.session.get(
                    f"{OPENALEX_BASE}/works", params=params, timeout=30
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                break

            results = data.get("results", [])
            if not results:
                break

            for w in results:
                yr = _extract_year(w.get("publication_date"))
                if yr and yr >= retraction_year:
                    post_citations += 1
                    if last_cite_year is None or yr > last_cite_year:
                        last_cite_year = yr

            cursor = data.get("meta", {}).get("next_cursor")
            pages += 1
            _sleep()

        current_year = datetime.now().year
        if last_cite_year is not None:
            relaxation_time = last_cite_year - retraction_year
            still_citing = (current_year - last_cite_year) <= 2
        else:
            relaxation_time = None
            still_citing = False

        return {
            "post_retraction_citation_count": post_citations,
            "last_citation_year": last_cite_year,
            "relaxation_time_years": relaxation_time,
            "still_citing": still_citing,
        }

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        max_retractions: int = 30,
        max_citers_per_paper: int = 100,
        max_authors_per_paper: int = 40,
    ) -> Dict[str, pd.DataFrame]:
        """
        End-to-end data collection pipeline.

        Steps
        -----
        1. Fetch retracted papers from OpenAlex + CrossRef.
        2. For each retracted paper, fetch citing works.
        3. Collect unique citing-author IDs; fetch career stats.
        4. For each (author, retracted_paper) pair, track post-retraction
           citation behaviour and compute relaxation time.
        5. Save four CSVs: retractions, citations, authors, tracking.

        Args:
            max_retractions: Number of retraction events to sample.
            max_citers_per_paper: Max citing works per retracted paper.
            max_authors_per_paper: Max authors to profile per retracted paper.

        Returns:
            Dict of DataFrames keyed by table name.
        """
        # ---- 1. Retracted papers ----
        retracted_oa = self.fetch_retracted_papers_openalex(
            max_results=max_retractions
        )
        retracted_cr = self.fetch_retracted_papers_crossref(
            max_results=max_retractions
        )

        # Merge and deduplicate on retracted DOI
        all_retracted = []
        seen_dois = set()

        for r in retracted_oa:
            work_id = r.get("retracted_work_id")
            doi = r.get("retracted_doi")
            key = doi or work_id
            if key and key not in seen_dois:
                seen_dois.add(key)
                all_retracted.append({
                    "retracted_work_id": work_id,
                    "retracted_doi": doi,
                    "retraction_year": r.get("retraction_year"),
                    "source": "openalex",
                })

        for r in retracted_cr:
            doi = r.get("retracted_doi")
            if doi and doi not in seen_dois:
                seen_dois.add(doi)
                all_retracted.append({
                    "retracted_work_id": None,  # resolve below
                    "retracted_doi": doi,
                    "retraction_year": r.get("retraction_year"),
                    "source": "crossref",
                })

        # For CrossRef-only entries, resolve DOI -> OpenAlex work ID
        for entry in all_retracted:
            if entry["retracted_work_id"] is None and entry["retracted_doi"]:
                oa_id = self._resolve_doi_to_openalex(entry["retracted_doi"])
                entry["retracted_work_id"] = oa_id

        # Filter to entries with a valid work ID and retraction year
        all_retracted = [
            r for r in all_retracted
            if r.get("retracted_work_id") and r.get("retraction_year")
        ]

        print(f"\nTotal unique retracted papers with IDs: {len(all_retracted)}")
        retractions_df = pd.DataFrame(all_retracted)

        if retractions_df.empty:
            print("No retracted papers found. Exiting pipeline.")
            return {"retractions": retractions_df}

        # ---- 2. Citing works ----
        print("\nFetching citing works for each retracted paper ...")
        all_citations = []

        for _, row in tqdm(retractions_df.iterrows(), total=len(retractions_df)):
            work_id = row["retracted_work_id"]
            citers = self.fetch_citing_works(work_id, max_citers=max_citers_per_paper)
            for c in citers:
                c["retracted_work_id"] = work_id
                c["retraction_year"] = row["retraction_year"]
            all_citations.extend(citers)

        citations_df = pd.DataFrame(all_citations)
        print(f"  Collected {len(citations_df)} total citing works")

        if citations_df.empty:
            print("No citing works found. Exiting pipeline.")
            return {
                "retractions": retractions_df,
                "citations": citations_df,
            }

        # ---- 3. Author statistics ----
        # Flatten author IDs from citing works
        author_paper_pairs = []
        for _, row in citations_df.iterrows():
            for aid in (row.get("author_ids") or []):
                author_paper_pairs.append({
                    "author_id": aid,
                    "retracted_work_id": row["retracted_work_id"],
                    "retraction_year": row["retraction_year"],
                    "citation_year": row["citation_year"],
                })

        pairs_df = pd.DataFrame(author_paper_pairs)
        unique_authors = pairs_df["author_id"].unique()
        print(f"\nFetching career stats for {len(unique_authors)} unique authors ...")

        author_stats = []
        for aid in tqdm(unique_authors[:max_authors_per_paper * len(retractions_df)]):
            stats = self.fetch_author_stats(aid)
            if stats:
                author_stats.append(stats)
            _sleep()

        authors_df = pd.DataFrame(author_stats)
        print(f"  Collected stats for {len(authors_df)} authors")

        # ---- 4. Post-retraction tracking ----
        # For each (author, retracted_work) pair, track post-retraction cites
        # Limit to authors we successfully fetched stats for
        known_authors = set(authors_df["author_id"]) if not authors_df.empty else set()
        tracking_pairs = (
            pairs_df[pairs_df["author_id"].isin(known_authors)]
            .drop_duplicates(subset=["author_id", "retracted_work_id"])
        )

        print(f"\nTracking post-retraction citations for {len(tracking_pairs)} author-paper pairs ...")
        tracking_records = []

        for _, pair in tqdm(tracking_pairs.iterrows(), total=len(tracking_pairs)):
            result = self.track_post_retraction_citations(
                author_id=pair["author_id"],
                retracted_work_id=pair["retracted_work_id"],
                retraction_year=int(pair["retraction_year"]),
            )
            result["author_id"] = pair["author_id"]
            result["retracted_work_id"] = pair["retracted_work_id"]
            result["retraction_year"] = int(pair["retraction_year"])
            tracking_records.append(result)

        tracking_df = pd.DataFrame(tracking_records)
        print(f"  Tracked {len(tracking_df)} pairs")

        # ---- 5. Save ----
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        retractions_df.to_csv(
            self.output_dir / f"retractions_{timestamp}.csv", index=False
        )
        citations_df.to_csv(
            self.output_dir / f"citations_{timestamp}.csv", index=False
        )
        authors_df.to_csv(
            self.output_dir / f"authors_{timestamp}.csv", index=False
        )
        tracking_df.to_csv(
            self.output_dir / f"tracking_{timestamp}.csv", index=False
        )

        metadata = {
            "timestamp": timestamp,
            "num_retracted_papers": len(retractions_df),
            "num_citing_works": len(citations_df),
            "num_authors": len(authors_df),
            "num_tracking_pairs": len(tracking_df),
            "params": {
                "max_retractions": max_retractions,
                "max_citers_per_paper": max_citers_per_paper,
                "max_authors_per_paper": max_authors_per_paper,
            },
        }

        with open(self.output_dir / f"metadata_{timestamp}.json", "w") as fh:
            json.dump(metadata, fh, indent=2)

        print(f"\nData saved to {self.output_dir}/")
        print(f"  - {len(retractions_df)} retracted papers")
        print(f"  - {len(citations_df)} citing works")
        print(f"  - {len(authors_df)} author profiles")
        print(f"  - {len(tracking_df)} post-retraction tracking records")

        return {
            "retractions": retractions_df,
            "citations": citations_df,
            "authors": authors_df,
            "tracking": tracking_df,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_doi_to_openalex(self, doi: str) -> Optional[str]:
        """Resolve a DOI to an OpenAlex work ID."""
        clean_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
        url = f"{OPENALEX_BASE}/works/doi:{clean_doi}"
        params = {"mailto": POLITE_EMAIL, "select": "id"}

        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return _short_id(resp.json().get("id", ""))
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fetcher = OpenAlexRetractionFetcher(output_dir="data")

    data = fetcher.run_full_pipeline(
        max_retractions=30,
        max_citers_per_paper=100,
        max_authors_per_paper=40,
    )

    print("\nData collection complete!")
    print("Ready for epistemic inertia analysis (analyze_inertia.py)")
