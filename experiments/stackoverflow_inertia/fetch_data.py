"""
Stack Overflow Data Fetcher for Epistemic Inertia Analysis

Provides SQL queries for the Stack Exchange Data Explorer (SEDE) to collect
data testing whether high-reputation users exhibit greater epistemic inertia
when their answers receive critical comments.

Theory (Hamiltonian belief dynamics):
    The epistemic mass matrix M_i governs resistance to belief revision.
    Higher mass => slower, smaller updates in response to new evidence.

    M_i = Lambda_p + Lambda_o + Sum_k beta_ik Lambda_tilde_qk + Sum_j beta_ji Lambda_qi

    Stack Overflow proxy mapping:
        Lambda_p  (prior precision)       = log(reputation)
        Lambda_o  (observation precision) = tag expertise ratio (answers in tag / total answers)
        Sum beta_ji (outgoing social)     = answer view_count * score (audience reach)

    Prediction (H_so): High-reputation users edit their answers LESS frequently
    after receiving critical comments, because their epistemic mass is higher.

Data source: https://data.stackexchange.com/
    SEDE provides a web SQL interface over the public Stack Overflow data dump.
    The queries below can be pasted directly into the SEDE web interface.
    Results are downloaded as CSV files for offline analysis.

Usage:
    Option A (automated): Run this script to save SQL queries and process
             downloaded CSV files.
    Option B (manual):    Copy each query from QUERIES dict, paste into
             data.stackexchange.com, run, download CSV, place in data/ dir.

    Then run:
        python analyze_inertia.py
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# SEDE SQL Queries
# ---------------------------------------------------------------------------
# These queries are designed for the Stack Exchange Data Explorer web interface
# at https://data.stackexchange.com/stackoverflow/query/new
# Paste each query, click "Run Query", and download results as CSV.
# ---------------------------------------------------------------------------

QUERIES: Dict[str, str] = {

    # ------------------------------------------------------------------
    # Query 1: Answers that received critical comments
    # ------------------------------------------------------------------
    # Finds answers where a comment contains language suggesting the answer
    # is wrong, incomplete, or should be corrected. Captures the answer
    # metadata, the comment metadata, and whether the answer was edited
    # after the critical comment was posted.
    # ------------------------------------------------------------------
    "critical_comments": """
-- Answers that received critical comments with edit response tracking
-- Paste into https://data.stackexchange.com/stackoverflow/query/new
-- Download result as CSV -> data/critical_comments.csv

SELECT
    a.Id                    AS answer_id,
    a.ParentId              AS question_id,
    a.OwnerUserId           AS answerer_id,
    a.Score                 AS answer_score,
    a.ViewCount             AS answer_view_count,
    a.CreationDate          AS answer_created,
    a.LastEditDate          AS answer_last_edit,
    a.CommentCount          AS answer_comment_count,
    c.Id                    AS comment_id,
    c.UserId                AS commenter_id,
    c.Score                 AS comment_score,
    c.CreationDate          AS comment_created,
    c.Text                  AS comment_text,
    q.Tags                  AS question_tags,
    q.ViewCount             AS question_view_count,
    q.Score                 AS question_score,
    q.AnswerCount           AS question_answer_count,
    -- Did the author edit AFTER this comment?
    CASE
        WHEN a.LastEditDate IS NOT NULL
             AND a.LastEditDate > c.CreationDate
        THEN 1
        ELSE 0
    END                     AS edited_after_comment
FROM Posts a
INNER JOIN Comments c
    ON c.PostId = a.Id
INNER JOIN Posts q
    ON q.Id = a.ParentId
WHERE
    a.PostTypeId = 2                          -- answers only
    AND a.OwnerUserId IS NOT NULL             -- not deleted/anonymous
    AND c.UserId <> a.OwnerUserId             -- comment by someone else
    AND a.Score >= 0                          -- non-negative answers
    AND (
        c.Text LIKE '%wrong%'
        OR c.Text LIKE '%incorrect%'
        OR c.Text LIKE '%mistake%'
        OR c.Text LIKE '%doesn''t work%'
        OR c.Text LIKE '%does not work%'
        OR c.Text LIKE '%not correct%'
        OR c.Text LIKE '%actually%'
        OR c.Text LIKE '%should be%'
        OR c.Text LIKE '%but this%'
        OR c.Text LIKE '%however%'
        OR c.Text LIKE '%bug%'
        OR c.Text LIKE '%error%'
        OR c.Text LIKE '%fix%'
        OR c.Text LIKE '%update%your%answer%'
    )
    AND c.CreationDate > '2020-01-01'         -- recent data
    AND a.CreationDate  > '2019-01-01'
ORDER BY a.Score DESC
""",

    # ------------------------------------------------------------------
    # Query 2: User reputation and activity statistics
    # ------------------------------------------------------------------
    # Fetches reputation, answer count, and account age for all answerers
    # identified in Query 1.  Run Query 1 first, note the answerer_ids,
    # or run this query standalone for top answerers.
    # ------------------------------------------------------------------
    "user_stats": """
-- User reputation and activity statistics
-- Paste into https://data.stackexchange.com/stackoverflow/query/new
-- Download result as CSV -> data/user_stats.csv

SELECT
    u.Id                    AS user_id,
    u.Reputation            AS reputation,
    u.CreationDate          AS account_created,
    u.UpVotes               AS total_upvotes,
    u.DownVotes             AS total_downvotes,
    u.Views                 AS profile_views,
    (
        SELECT COUNT(*)
        FROM Posts p
        WHERE p.OwnerUserId = u.Id
          AND p.PostTypeId = 2
    )                       AS total_answers,
    (
        SELECT COUNT(*)
        FROM Posts p
        WHERE p.OwnerUserId = u.Id
          AND p.PostTypeId = 1
    )                       AS total_questions,
    (
        SELECT ISNULL(SUM(p.Score), 0)
        FROM Posts p
        WHERE p.OwnerUserId = u.Id
          AND p.PostTypeId = 2
    )                       AS total_answer_score
FROM Users u
WHERE u.Reputation >= 100              -- filter out very low-activity users
  AND u.Id IN (
      -- Restrict to users who authored answers with critical comments
      SELECT DISTINCT a.OwnerUserId
      FROM Posts a
      INNER JOIN Comments c ON c.PostId = a.Id
      WHERE a.PostTypeId = 2
        AND a.OwnerUserId IS NOT NULL
        AND c.UserId <> a.OwnerUserId
        AND (
            c.Text LIKE '%wrong%'
            OR c.Text LIKE '%incorrect%'
            OR c.Text LIKE '%mistake%'
            OR c.Text LIKE '%doesn''t work%'
            OR c.Text LIKE '%does not work%'
            OR c.Text LIKE '%not correct%'
        )
        AND c.CreationDate > '2020-01-01'
  )
ORDER BY u.Reputation DESC
""",

    # ------------------------------------------------------------------
    # Query 3: Tag-level expertise for each answerer
    # ------------------------------------------------------------------
    # Computes per-tag answer counts for each user, enabling calculation
    # of Lambda_o = (answers_in_tag / total_answers).
    # ------------------------------------------------------------------
    "tag_expertise": """
-- Tag-level expertise: answers per tag per user
-- Paste into https://data.stackexchange.com/stackoverflow/query/new
-- Download result as CSV -> data/tag_expertise.csv

SELECT
    p.OwnerUserId           AS user_id,
    t.TagName               AS tag_name,
    COUNT(*)                AS answers_in_tag
FROM Posts p
INNER JOIN PostTags pt ON pt.PostId = p.ParentId   -- tags on the question
INNER JOIN Tags t      ON t.Id = pt.TagId
WHERE p.PostTypeId = 2                             -- answers
  AND p.OwnerUserId IS NOT NULL
  AND p.OwnerUserId IN (
      SELECT DISTINCT a.OwnerUserId
      FROM Posts a
      INNER JOIN Comments c ON c.PostId = a.Id
      WHERE a.PostTypeId = 2
        AND a.OwnerUserId IS NOT NULL
        AND c.UserId <> a.OwnerUserId
        AND (
            c.Text LIKE '%wrong%'
            OR c.Text LIKE '%incorrect%'
            OR c.Text LIKE '%mistake%'
        )
        AND c.CreationDate > '2020-01-01'
  )
GROUP BY p.OwnerUserId, t.TagName
HAVING COUNT(*) >= 3                              -- meaningful expertise
ORDER BY p.OwnerUserId, COUNT(*) DESC
""",

    # ------------------------------------------------------------------
    # Query 4: Post edit history (for edit magnitude calculation)
    # ------------------------------------------------------------------
    # Retrieves the revision history for answers that received critical
    # comments.  Allows computing edit_magnitude as the Levenshtein-like
    # ratio between pre- and post-edit body text.
    # ------------------------------------------------------------------
    "edit_history": """
-- Edit history for answers with critical comments
-- Paste into https://data.stackexchange.com/stackoverflow/query/new
-- Download result as CSV -> data/edit_history.csv

SELECT
    ph.PostId               AS answer_id,
    ph.RevisionGUID         AS revision_guid,
    ph.CreationDate         AS revision_date,
    ph.UserId               AS editor_id,
    ph.PostHistoryTypeId    AS edit_type,
    -- PostHistoryTypeId: 4 = Edit Title, 5 = Edit Body, 6 = Edit Tags
    LEN(ph.Text)            AS revision_length,
    ph.Comment              AS edit_comment
FROM PostHistory ph
WHERE ph.PostHistoryTypeId IN (5, 8)   -- 5 = Edit Body, 8 = Rollback Body
  AND ph.PostId IN (
      SELECT DISTINCT a.Id
      FROM Posts a
      INNER JOIN Comments c ON c.PostId = a.Id
      WHERE a.PostTypeId = 2
        AND a.OwnerUserId IS NOT NULL
        AND c.UserId <> a.OwnerUserId
        AND (
            c.Text LIKE '%wrong%'
            OR c.Text LIKE '%incorrect%'
            OR c.Text LIKE '%mistake%'
        )
        AND c.CreationDate > '2020-01-01'
  )
ORDER BY ph.PostId, ph.CreationDate
""",
}


class StackOverflowDataProcessor:
    """
    Process CSV files downloaded from SEDE into analysis-ready DataFrames.

    Workflow:
        1. User runs SQL queries at data.stackexchange.com (or copies from QUERIES dict)
        2. User downloads CSV results into data/ directory
        3. This class loads, cleans, and merges the CSVs
        4. Computes epistemic mass proxies and edit response variables
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)

    # ----- public entry point -------------------------------------------

    def save_queries(self, output_path: Optional[str] = None) -> None:
        """
        Save all SEDE SQL queries to a single .sql file for easy copy-paste.

        Args:
            output_path: Where to save. Defaults to data/sede_queries.sql
        """
        out = Path(output_path) if output_path else self.data_dir / "sede_queries.sql"
        with open(out, "w") as f:
            for name, sql in QUERIES.items():
                f.write(f"-- =========================================\n")
                f.write(f"-- Query: {name}\n")
                f.write(f"-- Save result as: data/{name}.csv\n")
                f.write(f"-- =========================================\n")
                f.write(sql.strip())
                f.write("\n\n\n")
        print(f"Saved {len(QUERIES)} SEDE queries to {out}")
        print("Instructions:")
        print("  1. Open https://data.stackexchange.com/stackoverflow/query/new")
        print("  2. Paste each query and click 'Run Query'")
        print("  3. Click 'Download CSV' and save to data/ directory")
        print("  4. Run: python analyze_inertia.py")

    def load_and_process(self) -> pd.DataFrame:
        """
        Load downloaded SEDE CSV files, merge, and compute derived variables.

        Expected files in data/:
            critical_comments.csv   (from Query 1)
            user_stats.csv          (from Query 2)
            tag_expertise.csv       (from Query 3)
            edit_history.csv        (from Query 4, optional)

        Returns:
            Merged DataFrame ready for analysis, with columns:
                answer_id, answerer_id, answer_score, question_tags,
                log_reputation, tag_expertise_ratio, social_reach,
                edited_after_comment, edit_magnitude, ...
        """
        # Load core tables
        comments_df = self._load_csv("critical_comments")
        users_df = self._load_csv("user_stats")
        tags_df = self._load_csv("tag_expertise")
        edits_df = self._load_csv("edit_history", required=False)

        print(f"Loaded {len(comments_df)} critical comment events")
        print(f"Loaded {len(users_df)} user profiles")
        print(f"Loaded {len(tags_df)} tag-expertise rows")
        if edits_df is not None:
            print(f"Loaded {len(edits_df)} edit history rows")

        # ------ Compute mass proxies on users table --------------------

        # Lambda_p (prior precision) = log(reputation)
        users_df["log_reputation"] = np.log1p(
            pd.to_numeric(users_df["reputation"], errors="coerce").fillna(1)
        )

        # Account age in years
        users_df["account_created"] = pd.to_datetime(
            users_df["account_created"], errors="coerce"
        )
        ref_date = pd.Timestamp("2024-06-01")  # approx SEDE dump date
        users_df["account_age_years"] = (
            (ref_date - users_df["account_created"]).dt.total_seconds() / (365.25 * 86400)
        ).clip(lower=0)

        # ------ Compute tag expertise (Lambda_o) per user ---------------

        tag_totals = tags_df.groupby("user_id")["answers_in_tag"].sum().rename(
            "total_tag_answers"
        )
        tags_df = tags_df.merge(tag_totals, on="user_id", how="left")
        tags_df["tag_expertise_ratio"] = (
            tags_df["answers_in_tag"] / tags_df["total_tag_answers"]
        )

        # ------ Merge everything onto the comments table -----------------

        # Merge user stats
        merged = comments_df.merge(
            users_df[
                [
                    "user_id",
                    "reputation",
                    "log_reputation",
                    "total_answers",
                    "total_answer_score",
                    "account_age_years",
                ]
            ],
            left_on="answerer_id",
            right_on="user_id",
            how="left",
        )

        # Extract primary tag from question_tags (first tag)
        merged["primary_tag"] = merged["question_tags"].apply(self._extract_primary_tag)

        # Merge tag expertise for the primary tag
        merged = merged.merge(
            tags_df[["user_id", "tag_name", "tag_expertise_ratio"]],
            left_on=["answerer_id", "primary_tag"],
            right_on=["user_id", "tag_name"],
            how="left",
            suffixes=("", "_tag"),
        )
        merged["tag_expertise_ratio"] = merged["tag_expertise_ratio"].fillna(0.0)

        # ------ Compute social reach: Sum beta_ji -----------------------
        # Proxy: view_count * score = how many people consumed this answer
        merged["answer_view_count"] = pd.to_numeric(
            merged.get("answer_view_count", pd.Series(dtype=float)), errors="coerce"
        ).fillna(0)
        merged["social_reach"] = (
            merged["answer_view_count"] * merged["answer_score"].clip(lower=0)
        )
        merged["log_social_reach"] = np.log1p(merged["social_reach"])

        # ------ Compute edit response variables --------------------------

        merged["edited_after_comment"] = pd.to_numeric(
            merged["edited_after_comment"], errors="coerce"
        ).fillna(0).astype(int)

        # Time between answer creation and critical comment (hours)
        merged["answer_created"] = pd.to_datetime(
            merged["answer_created"], errors="coerce"
        )
        merged["comment_created"] = pd.to_datetime(
            merged["comment_created"], errors="coerce"
        )
        merged["hours_to_comment"] = (
            (merged["comment_created"] - merged["answer_created"]).dt.total_seconds() / 3600
        ).clip(lower=0)

        # ------ Compute edit magnitude from edit history -----------------
        if edits_df is not None:
            edit_magnitude = self._compute_edit_magnitude(edits_df, comments_df)
            merged = merged.merge(edit_magnitude, on="answer_id", how="left")
        else:
            merged["edit_magnitude"] = np.nan

        # ------ Compute composite epistemic mass -------------------------
        # M_i = Lambda_p + Lambda_o + Sum beta_ji Lambda_qi
        #
        # Standardize components before summing so they contribute
        # comparably to the composite.
        merged["mass_score"] = (
            self._zscore(merged["log_reputation"])
            + self._zscore(merged["tag_expertise_ratio"])
            + self._zscore(merged["log_social_reach"])
        )

        # ------ Question difficulty proxy --------------------------------
        merged["question_difficulty"] = (
            merged["question_answer_count"].clip(lower=1).apply(np.log)
        )

        # ------ Clean up and save ----------------------------------------
        keep_cols = [
            "answer_id",
            "question_id",
            "answerer_id",
            "answer_score",
            "answer_view_count",
            "answer_created",
            "comment_id",
            "commenter_id",
            "comment_score",
            "comment_created",
            "primary_tag",
            "question_view_count",
            "question_score",
            "question_answer_count",
            "question_difficulty",
            "edited_after_comment",
            "edit_magnitude",
            "hours_to_comment",
            "reputation",
            "log_reputation",
            "total_answers",
            "total_answer_score",
            "account_age_years",
            "tag_expertise_ratio",
            "social_reach",
            "log_social_reach",
            "mass_score",
        ]
        # Keep only columns that exist
        keep_cols = [c for c in keep_cols if c in merged.columns]
        result = merged[keep_cols].copy()

        # Drop duplicates (an answer can have multiple critical comments;
        # keep the earliest one per answer)
        result = result.sort_values("comment_created").drop_duplicates(
            subset=["answer_id"], keep="first"
        )

        # Save processed dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.data_dir / f"processed_{timestamp}.csv"
        result.to_csv(out_path, index=False)
        print(f"\nSaved processed dataset: {out_path}")
        print(f"  {len(result)} answer-comment events")
        print(f"  {result['answerer_id'].nunique()} unique answerers")
        print(f"  Edit rate: {result['edited_after_comment'].mean():.1%}")

        # Save metadata
        meta = {
            "timestamp": timestamp,
            "n_events": len(result),
            "n_users": int(result["answerer_id"].nunique()),
            "edit_rate": float(result["edited_after_comment"].mean()),
            "median_reputation": float(result["reputation"].median()),
            "source": "Stack Exchange Data Explorer (SEDE)",
        }
        with open(self.data_dir / f"metadata_{timestamp}.json", "w") as f:
            json.dump(meta, f, indent=2)

        return result

    # ----- private helpers -----------------------------------------------

    def _load_csv(self, name: str, required: bool = True) -> Optional[pd.DataFrame]:
        """Load a CSV file from data directory, trying several filename patterns."""
        patterns = [
            f"{name}.csv",
            f"{name}_*.csv",
        ]
        for pat in patterns:
            matches = sorted(self.data_dir.glob(pat))
            if matches:
                df = pd.read_csv(matches[-1])
                # Normalize column names to snake_case
                df.columns = [self._to_snake_case(c) for c in df.columns]
                return df

        if required:
            raise FileNotFoundError(
                f"Missing {name}.csv in {self.data_dir}/\n"
                f"Run the '{name}' query at data.stackexchange.com and "
                f"save the result as data/{name}.csv"
            )
        return None

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert 'CamelCase' or 'Column Name' to 'snake_case'."""
        # Handle spaces
        name = name.strip().replace(" ", "_")
        # CamelCase -> snake_case
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    @staticmethod
    def _extract_primary_tag(tags_str) -> str:
        """Extract first tag from SO tag format '<python><pandas>'."""
        if not isinstance(tags_str, str):
            return "unknown"
        match = re.search(r"<([^>]+)>", tags_str)
        return match.group(1) if match else "unknown"

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        """Standardize to z-scores, handling constant series."""
        s = pd.to_numeric(series, errors="coerce")
        std = s.std()
        if std < 1e-10:
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std

    @staticmethod
    def _compute_edit_magnitude(
        edits_df: pd.DataFrame, comments_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute edit magnitude for each answer as the relative change in
        body length between the revision just before the critical comment
        and the first revision after.

        edit_magnitude = |len_after - len_before| / len_before

        Returns:
            DataFrame with columns [answer_id, edit_magnitude]
        """
        edits_df = edits_df.copy()
        edits_df["revision_date"] = pd.to_datetime(
            edits_df["revision_date"], errors="coerce"
        )
        edits_df["revision_length"] = pd.to_numeric(
            edits_df["revision_length"], errors="coerce"
        )

        # Get earliest critical comment per answer
        comments_df = comments_df.copy()
        comments_df["comment_created"] = pd.to_datetime(
            comments_df.get("comment_created", pd.Series(dtype="datetime64[ns]")),
            errors="coerce",
        )
        first_comment = (
            comments_df.sort_values("comment_created")
            .drop_duplicates(subset=["answer_id"], keep="first")[
                ["answer_id", "comment_created"]
            ]
        )

        results = []
        for _, row in first_comment.iterrows():
            aid = row["answer_id"]
            cdate = row["comment_created"]
            if pd.isna(cdate):
                continue

            post_edits = edits_df[edits_df["answer_id"] == aid].sort_values(
                "revision_date"
            )
            if len(post_edits) < 2:
                continue

            before = post_edits[post_edits["revision_date"] <= cdate]
            after = post_edits[post_edits["revision_date"] > cdate]

            if len(before) == 0 or len(after) == 0:
                continue

            len_before = before.iloc[-1]["revision_length"]
            len_after = after.iloc[0]["revision_length"]

            if pd.isna(len_before) or pd.isna(len_after) or len_before == 0:
                continue

            magnitude = abs(len_after - len_before) / len_before
            results.append({"answer_id": aid, "edit_magnitude": magnitude})

        if results:
            return pd.DataFrame(results)
        return pd.DataFrame(columns=["answer_id", "edit_magnitude"])


def print_queries() -> None:
    """Print all SEDE queries to stdout for manual copy-paste."""
    for name, sql in QUERIES.items():
        print("=" * 72)
        print(f"  QUERY: {name}")
        print(f"  Save result as: data/{name}.csv")
        print("=" * 72)
        print(sql.strip())
        print("\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stack Overflow epistemic inertia data pipeline"
    )
    parser.add_argument(
        "--print-queries",
        action="store_true",
        help="Print SEDE SQL queries to stdout for manual use",
    )
    parser.add_argument(
        "--save-queries",
        action="store_true",
        help="Save SEDE SQL queries to data/sede_queries.sql",
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Process downloaded CSV files into analysis-ready dataset",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory for CSV files (default: data/)",
    )
    args = parser.parse_args()

    processor = StackOverflowDataProcessor(data_dir=args.data_dir)

    if args.print_queries:
        print_queries()
    elif args.save_queries:
        processor.save_queries()
    elif args.process:
        df = processor.load_and_process()
        print(f"\nProcessing complete. Shape: {df.shape}")
    else:
        # Default: save queries and print instructions
        processor.save_queries()
        print("\nTo process downloaded CSVs:")
        print(f"  python {__file__} --process --data-dir {args.data_dir}")
