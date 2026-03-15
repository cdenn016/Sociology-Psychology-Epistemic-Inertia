"""
Wikipedia Edit History Data Fetcher for Epistemic Inertia Analysis

Fetches revision histories from contentious Wikipedia articles to test the
hypothesis that editors with higher "epistemic mass" exhibit greater rigidity
in the face of reverts -- i.e., they are less likely to accept reverts of
their contributions.

Theory (mass matrix mapping to Wikipedia proxies):
  M_i = Lambda_p + Lambda_o + Sum_k beta_ik * Lambda_tilde_qk + Sum_j beta_ji * Lambda_qi

  Wikipedia proxy mapping:
    Lambda_p  (prior precision)       = log(total_edit_count)  -- editing experience
    Lambda_o  (observation precision) = log(article_edit_count) -- topic-specific experience
    Sum_j beta_ji (outgoing social)   = page watchers count for the editor's primary pages
    Sum_k beta_ik (incoming social)   = number of co-editors on shared articles

  High M_i --> Lower revert acceptance rate (epistemic rigidity)

Data source: Wikipedia MediaWiki API (https://en.wikipedia.org/w/api.php)
No authentication required for read-only queries.
"""

import requests
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from collections import defaultdict


# ---------------------------------------------------------------------------
# Contentious articles selected from Wikipedia's list of controversial topics.
# These pages have high revert rates, making them ideal for studying epistemic
# inertia in editorial disputes.
# ---------------------------------------------------------------------------
CONTENTIOUS_ARTICLES = [
    "Abortion",
    "Climate change",
    "Gun control",
    "Israeli-Palestinian conflict",
    "Donald Trump",
    "COVID-19 pandemic",
    "Race and intelligence",
    "Intelligent design",
    "Homeopathy",
    "Scientology",
    "Circumcision",
    "Kashmir conflict",
    "Death penalty",
    "Iraq War",
    "Nuclear power",
    "Genetically modified food",
    "Vaccination",
    "2020 United States presidential election",
    "Crimea",
    "Muhammad",
]


class WikipediaRevertFetcher:
    """
    Fetches revision histories from contentious Wikipedia articles and
    identifies revert events to measure epistemic inertia.

    A 'revert' is detected when a revision restores the SHA-1 hash of a
    previous revision, indicating that an editor undid someone else's work.
    """

    API_URL = "https://en.wikipedia.org/w/api.php"
    # MediaWiki API etiquette: identify the client in User-Agent
    HEADERS = {
        "User-Agent": "EpistemicInertiaResearch/1.0 (academic research; "
                      "epistemic-inertia-study)"
    }

    def __init__(self, output_dir: str = "data", max_revisions: int = 500):
        """
        Args:
            output_dir: Directory where CSV outputs will be saved.
            max_revisions: Max revisions to fetch per article (API limit
                           awareness -- 500 for anonymous users).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self.max_revisions = max_revisions

    # ------------------------------------------------------------------
    # Core API helpers
    # ------------------------------------------------------------------

    def _api_get(self, params: Dict, retries: int = 3) -> Dict:
        """
        Make a GET request to the MediaWiki API with retry logic.

        Respects rate limits by sleeping between retries and between
        successive calls (MediaWiki asks for <= 200 req/s for bots).
        """
        params.setdefault("format", "json")
        params.setdefault("formatversion", "2")

        for attempt in range(retries):
            try:
                resp = self.session.get(self.API_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    print(f"  API error: {data['error'].get('info', data['error'])}")
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                return data
            except (requests.RequestException, json.JSONDecodeError) as exc:
                print(f"  Request failed (attempt {attempt + 1}/{retries}): {exc}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        return {}

    # ------------------------------------------------------------------
    # Revision fetching
    # ------------------------------------------------------------------

    def fetch_revisions(self, title: str) -> List[Dict]:
        """
        Fetch revision history for a single article.

        Uses action=query&prop=revisions with rvprop to retrieve:
          - revid, parentid, user, timestamp, sha1, size, comment

        Paginates via 'rvcontinue' until max_revisions reached.

        Returns:
            List of revision dicts, oldest-first.
        """
        print(f"  Fetching revisions for '{title}'...")
        revisions = []
        rvcontinue = None

        while len(revisions) < self.max_revisions:
            params = {
                "action": "query",
                "prop": "revisions",
                "titles": title,
                "rvprop": "ids|user|userid|timestamp|sha1|size|comment|flags",
                "rvlimit": min(50, self.max_revisions - len(revisions)),
                "rvdir": "older",  # newest first, we reverse later
            }
            if rvcontinue:
                params["rvcontinue"] = rvcontinue

            data = self._api_get(params)
            if not data:
                break

            pages = data.get("query", {}).get("pages", [])
            if not pages:
                break

            page = pages[0] if isinstance(pages, list) else list(pages.values())[0]
            page_revisions = page.get("revisions", [])
            if not page_revisions:
                break

            for rev in page_revisions:
                rev["page_title"] = title
                rev["pageid"] = page.get("pageid", 0)

            revisions.extend(page_revisions)

            # Check for continuation token
            cont = data.get("continue", {})
            if "rvcontinue" in cont:
                rvcontinue = cont["rvcontinue"]
            else:
                break

            # Rate limiting: be polite to Wikipedia servers
            time.sleep(0.5)

        # Reverse so oldest revision comes first (chronological order)
        revisions.reverse()
        print(f"    Got {len(revisions)} revisions")
        return revisions

    # ------------------------------------------------------------------
    # Revert detection
    # ------------------------------------------------------------------

    def detect_reverts(self, revisions: List[Dict]) -> List[Dict]:
        """
        Detect reverts by finding revisions whose SHA-1 matches an earlier
        revision's SHA-1 (i.e., the content was restored to a prior state).

        For each revert, we record:
          - reverting_user: the editor who performed the revert
          - reverted_user: the editor whose work was undone
          - original_revid: the revision that was restored
          - revert_revid: the revision that performed the revert
          - reverted_revid: the revision(s) that were undone

        Returns:
            List of revert event dicts.
        """
        # Build a map from SHA-1 -> list of (index, revision)
        sha1_history: Dict[str, List[Tuple[int, Dict]]] = defaultdict(list)
        revert_events = []

        for idx, rev in enumerate(revisions):
            sha1 = rev.get("sha1", "")
            if not sha1:
                sha1_history[sha1].append((idx, rev))
                continue

            # Check if this SHA-1 appeared before (content restored)
            if sha1 in sha1_history and len(sha1_history[sha1]) > 0:
                # This is a revert -- content matches a previous revision
                original_idx, original_rev = sha1_history[sha1][-1]

                # The reverted edits are those between the original and this one
                if idx > original_idx + 1:
                    # Collect unique reverted users (editors whose work was undone)
                    reverted_users = set()
                    reverted_revids = []
                    for between_idx in range(original_idx + 1, idx):
                        between_rev = revisions[between_idx]
                        reverted_users.add(between_rev.get("user", ""))
                        reverted_revids.append(between_rev.get("revid", 0))

                    reverting_user = rev.get("user", "")
                    # Only count if reverting user is different from reverted user(s)
                    if reverting_user not in reverted_users:
                        for reverted_user in reverted_users:
                            if not reverted_user:
                                continue
                            revert_events.append({
                                "page_title": rev.get("page_title", ""),
                                "reverting_user": reverting_user,
                                "reverted_user": reverted_user,
                                "revert_revid": rev.get("revid", 0),
                                "original_revid": original_rev.get("revid", 0),
                                "reverted_revids": json.dumps(reverted_revids),
                                "revert_timestamp": rev.get("timestamp", ""),
                                "original_timestamp": original_rev.get("timestamp", ""),
                                "num_edits_reverted": idx - original_idx - 1,
                            })

            sha1_history[sha1].append((idx, rev))

        return revert_events

    def detect_re_reverts(self, revert_events: List[Dict],
                          revisions: List[Dict]) -> List[Dict]:
        """
        For each revert, check whether the *reverted* editor re-reverted
        (i.e., undid the revert) within the subsequent revisions.

        A re-revert means the editor did NOT accept the revert -- they
        fought back.  Acceptance = no re-revert within the look-ahead window.

        This is the key behavioral signal for epistemic inertia:
          High mass editors --> low revert_accepted (they fight back)

        Returns:
            revert_events list, each augmented with 'revert_accepted' bool.
        """
        # Build a quick lookup: revid -> index in revision list
        revid_to_idx = {}
        for idx, rev in enumerate(revisions):
            revid_to_idx[rev.get("revid", 0)] = idx

        look_ahead = 20  # Check next N revisions for a re-revert

        for event in revert_events:
            revert_idx = revid_to_idx.get(event["revert_revid"], -1)
            reverted_user = event["reverted_user"]
            event["revert_accepted"] = True  # default: accepted

            if revert_idx < 0:
                continue

            # Look ahead for the reverted user re-reverting
            for future_idx in range(revert_idx + 1,
                                    min(revert_idx + look_ahead + 1,
                                        len(revisions))):
                future_rev = revisions[future_idx]
                if future_rev.get("user", "") == reverted_user:
                    comment = (future_rev.get("comment", "") or "").lower()
                    # Heuristics: revert-like edit summaries
                    revert_keywords = ["revert", "undo", "undid", "rv ",
                                       "restored", "reverted"]
                    if any(kw in comment for kw in revert_keywords):
                        event["revert_accepted"] = False
                        break

        return revert_events

    # ------------------------------------------------------------------
    # Editor metadata
    # ------------------------------------------------------------------

    def fetch_user_info(self, usernames: List[str]) -> Dict[str, Dict]:
        """
        Fetch user metadata using action=query&list=users.

        Retrieves:
          - editcount (total edits across Wikipedia) --> Lambda_p proxy
          - registration date --> tenure
          - groups (admin, bureaucrat, etc.) --> control variable

        Args:
            usernames: List of Wikipedia usernames.

        Returns:
            Dict mapping username -> user info dict.
        """
        print(f"\nFetching info for {len(usernames)} editors...")
        user_info = {}

        # API supports up to 50 users per request
        batch_size = 50
        for i in range(0, len(usernames), batch_size):
            batch = usernames[i:i + batch_size]
            params = {
                "action": "query",
                "list": "users",
                "ususers": "|".join(batch),
                "usprop": "editcount|registration|groups|blockinfo",
            }

            data = self._api_get(params)
            if not data:
                continue

            users = data.get("query", {}).get("users", [])
            for u in users:
                name = u.get("name", "")
                if "missing" in u or "invalid" in u:
                    continue
                user_info[name] = {
                    "username": name,
                    "edit_count": u.get("editcount", 0),
                    "registration": u.get("registration", ""),
                    "groups": json.dumps(u.get("groups", [])),
                    "is_admin": "sysop" in u.get("groups", []),
                    "is_bot": "bot" in u.get("groups", []),
                }

            time.sleep(0.5)
            if (i // batch_size) % 5 == 0 and i > 0:
                print(f"  Processed {i + len(batch)}/{len(usernames)} users")

        print(f"  Retrieved info for {len(user_info)} editors")
        return user_info

    def fetch_page_watchers(self, titles: List[str]) -> Dict[str, int]:
        """
        Fetch page watcher counts using action=query&prop=info.

        Page watchers approximate the social reach of an editor's primary
        contributions -- the outgoing social coupling Sum_j beta_ji.

        Note: Wikipedia returns 'watchers' only if the count exceeds a
        privacy threshold (typically 30).  Pages below threshold return None.

        Args:
            titles: List of article titles.

        Returns:
            Dict mapping title -> watcher count (0 if unavailable).
        """
        print(f"\nFetching watcher counts for {len(titles)} pages...")
        watcher_counts = {}

        batch_size = 50
        for i in range(0, len(titles), batch_size):
            batch = titles[i:i + batch_size]
            params = {
                "action": "query",
                "prop": "info",
                "inprop": "watchers",
                "titles": "|".join(batch),
            }

            data = self._api_get(params)
            if not data:
                continue

            pages = data.get("query", {}).get("pages", {})
            if isinstance(pages, list):
                page_list = pages
            else:
                page_list = pages.values()

            for page in page_list:
                title = page.get("title", "")
                # 'watchers' may be absent if below privacy threshold
                watchers = page.get("watchers", 0) or 0
                watcher_counts[title] = watchers

            time.sleep(0.5)

        print(f"  Retrieved watcher counts for {len(watcher_counts)} pages")
        return watcher_counts

    # ------------------------------------------------------------------
    # Mass computation
    # ------------------------------------------------------------------

    def compute_editor_mass_proxies(
        self,
        user_info: Dict[str, Dict],
        editor_revisions: Dict[str, List[Dict]],
        watcher_counts: Dict[str, int],
    ) -> pd.DataFrame:
        """
        Compute epistemic mass proxies for each editor.

        Mass formula mapping (theory -> Wikipedia proxy):
        -------------------------------------------------------
        M_i = Lambda_p + Lambda_o + Sum_k beta_ik Lambda_tilde_qk + Sum_j beta_ji Lambda_qi

        Lambda_p   (prior precision)
            = log(1 + total_edit_count)
            Editors with more experience have stronger priors.

        Lambda_o   (observation precision)
            = log(1 + contentious_article_edit_count)
            Topic-specific editing experience.

        Sum_j beta_ji * Lambda_qi   (outgoing social mass)
            = mean(page_watchers) for the editor's primary articles
            Editors whose contributions are watched by many people
            carry more social inertia.

        Sum_k beta_ik * Lambda_tilde_qk   (incoming social mass)
            = number of distinct co-editors on shared articles
            Not computed here (would require full co-editor graph).
        -------------------------------------------------------

        Returns:
            DataFrame with columns: username, lambda_p, lambda_o,
            outgoing_mass, composite_mass, edit_count, tenure_days,
            is_admin, is_bot
        """
        import numpy as np

        records = []
        now = datetime.utcnow()

        for username, info in user_info.items():
            edit_count = info.get("edit_count", 0)
            # Lambda_p: prior precision from global experience
            lambda_p = float(np.log1p(edit_count))

            # Lambda_o: topic-specific experience (edits in our article set)
            user_revs = editor_revisions.get(username, [])
            contentious_edit_count = len(user_revs)
            lambda_o = float(np.log1p(contentious_edit_count))

            # Sum_j beta_ji: outgoing social influence
            # Mean watcher count across the editor's primary articles
            user_articles = set(r.get("page_title", "") for r in user_revs)
            article_watchers = [
                watcher_counts.get(a, 0) for a in user_articles
            ]
            outgoing_mass = (
                float(np.mean(article_watchers)) if article_watchers else 0.0
            )

            # Tenure in days
            reg = info.get("registration", "")
            if reg:
                try:
                    reg_dt = datetime.strptime(reg, "%Y-%m-%dT%H:%M:%SZ")
                    tenure_days = (now - reg_dt).days
                except (ValueError, TypeError):
                    tenure_days = 0
            else:
                tenure_days = 0

            # Composite mass (weighted combination)
            # Weights reflect theoretical importance:
            #   prior experience (0.4) + topic experience (0.2) + social (0.4)
            composite_mass = (
                0.4 * lambda_p
                + 0.2 * lambda_o
                + 0.4 * float(np.log1p(outgoing_mass))
            )

            records.append({
                "username": username,
                "edit_count": edit_count,
                "contentious_edit_count": contentious_edit_count,
                "lambda_p": lambda_p,
                "lambda_o": lambda_o,
                "outgoing_mass": outgoing_mass,
                "outgoing_mass_log": float(np.log1p(outgoing_mass)),
                "composite_mass": composite_mass,
                "tenure_days": tenure_days,
                "is_admin": info.get("is_admin", False),
                "is_bot": info.get("is_bot", False),
                "groups": info.get("groups", "[]"),
            })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self, articles: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Execute the full data collection pipeline.

        Steps:
          1. Fetch revision histories for each contentious article
          2. Detect revert events (SHA-1 matching)
          3. Detect re-reverts (editor fights back)
          4. Fetch editor metadata (edit counts, groups)
          5. Fetch page watcher counts (social influence proxy)
          6. Compute mass proxies for each editor
          7. Save all data to CSV

        Args:
            articles: List of article titles (defaults to CONTENTIOUS_ARTICLES).

        Returns:
            Dict of DataFrames: 'revisions', 'reverts', 'editors', 'watchers'
        """
        if articles is None:
            articles = CONTENTIOUS_ARTICLES

        print("=" * 70)
        print("WIKIPEDIA EPISTEMIC INERTIA -- DATA COLLECTION")
        print("=" * 70)
        print(f"Articles: {len(articles)}")
        print(f"Max revisions per article: {self.max_revisions}")
        print()

        # ----- Step 1: Fetch all revisions -----
        all_revisions = []
        for title in articles:
            revisions = self.fetch_revisions(title)
            all_revisions.extend(revisions)

        print(f"\nTotal revisions fetched: {len(all_revisions)}")

        # ----- Step 2 & 3: Detect reverts and re-reverts per article -----
        all_revert_events = []
        for title in articles:
            article_revs = [r for r in all_revisions
                            if r.get("page_title") == title]
            if not article_revs:
                continue

            reverts = self.detect_reverts(article_revs)
            reverts = self.detect_re_reverts(reverts, article_revs)
            all_revert_events.extend(reverts)

        print(f"\nTotal revert events detected: {len(all_revert_events)}")

        # ----- Step 4: Collect unique editors and fetch metadata -----
        # Gather all editors involved in reverts
        editor_names = set()
        for event in all_revert_events:
            editor_names.add(event["reverting_user"])
            editor_names.add(event["reverted_user"])

        # Also track which revisions belong to each editor
        editor_revisions: Dict[str, List[Dict]] = defaultdict(list)
        for rev in all_revisions:
            user = rev.get("user", "")
            if user:
                editor_revisions[user].append(rev)

        # Filter to registered editors (skip IPs)
        registered_editors = [
            name for name in editor_names
            if name and not _is_ip_address(name)
        ]

        user_info = self.fetch_user_info(registered_editors)

        # Filter out bots
        user_info = {
            k: v for k, v in user_info.items() if not v.get("is_bot", False)
        }
        print(f"  Non-bot registered editors: {len(user_info)}")

        # ----- Step 5: Fetch page watcher counts -----
        watcher_counts = self.fetch_page_watchers(articles)

        # ----- Step 6: Compute mass proxies -----
        editors_df = self.compute_editor_mass_proxies(
            user_info, editor_revisions, watcher_counts
        )

        # ----- Step 7: Compute per-editor revert acceptance rates -----
        reverts_df = pd.DataFrame(all_revert_events)
        if not reverts_df.empty:
            # For each reverted_user, compute acceptance rate
            acceptance = (
                reverts_df.groupby("reverted_user")
                .agg(
                    total_reverts=("revert_accepted", "count"),
                    reverts_accepted=("revert_accepted", "sum"),
                )
                .reset_index()
            )
            acceptance["revert_acceptance_rate"] = (
                acceptance["reverts_accepted"] / acceptance["total_reverts"]
            )
            acceptance.rename(
                columns={"reverted_user": "username"}, inplace=True
            )

            # Merge acceptance rates into editors DataFrame
            editors_df = editors_df.merge(
                acceptance[["username", "total_reverts", "reverts_accepted",
                            "revert_acceptance_rate"]],
                on="username",
                how="left",
            )
        else:
            editors_df["total_reverts"] = 0
            editors_df["reverts_accepted"] = 0
            editors_df["revert_acceptance_rate"] = float("nan")

        # ----- Save outputs -----
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        revisions_df = pd.DataFrame(all_revisions)
        revisions_path = self.output_dir / f"revisions_{timestamp}.csv"
        revisions_df.to_csv(revisions_path, index=False)
        print(f"\nSaved {len(revisions_df)} revisions to {revisions_path}")

        reverts_path = self.output_dir / f"reverts_{timestamp}.csv"
        reverts_df.to_csv(reverts_path, index=False)
        print(f"Saved {len(reverts_df)} revert events to {reverts_path}")

        editors_path = self.output_dir / f"editors_{timestamp}.csv"
        editors_df.to_csv(editors_path, index=False)
        print(f"Saved {len(editors_df)} editor profiles to {editors_path}")

        watchers_df = pd.DataFrame(
            list(watcher_counts.items()),
            columns=["page_title", "watchers"],
        )
        watchers_path = self.output_dir / f"watchers_{timestamp}.csv"
        watchers_df.to_csv(watchers_path, index=False)
        print(f"Saved {len(watchers_df)} watcher counts to {watchers_path}")

        print("\n" + "=" * 70)
        print("DATA COLLECTION COMPLETE")
        print("=" * 70)
        print(f"\nSummary:")
        print(f"  Articles analyzed:    {len(articles)}")
        print(f"  Total revisions:      {len(revisions_df)}")
        print(f"  Revert events:        {len(reverts_df)}")
        print(f"  Editors profiled:     {len(editors_df)}")
        if not editors_df.empty and "revert_acceptance_rate" in editors_df.columns:
            valid = editors_df["revert_acceptance_rate"].dropna()
            if len(valid) > 0:
                print(f"  Mean acceptance rate: {valid.mean():.3f}")
                print(f"  Editors with reverts: {len(valid)}")

        return {
            "revisions": revisions_df,
            "reverts": reverts_df,
            "editors": editors_df,
            "watchers": watchers_df,
        }


def _is_ip_address(name: str) -> bool:
    """Check if a username looks like an IP address (anonymous editor)."""
    parts = name.split(".")
    if len(parts) == 4:
        try:
            return all(0 <= int(p) <= 255 for p in parts)
        except ValueError:
            pass
    # IPv6
    if ":" in name and all(c in "0123456789abcdefABCDEF:" for c in name):
        return True
    return False


def main():
    """Run Wikipedia epistemic inertia data collection."""
    fetcher = WikipediaRevertFetcher(
        output_dir="data",
        max_revisions=500,
    )
    results = fetcher.run()

    print("\nData files saved to data/ directory.")
    print("Next step: run analyze_inertia.py to test H3.1")


if __name__ == "__main__":
    main()
