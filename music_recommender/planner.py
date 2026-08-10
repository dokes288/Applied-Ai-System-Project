"""
Agentic Planner module — orchestrates retrieval → guardrails → recommendation
in an explicit plan-and-execute loop with step-by-step logging.

Decision-making process
------------------------
Step 1  Parse & validate inputs via Guardrails.
Step 2  If a free-text query is provided, retrieve a pre-filtered candidate
        list from the catalog.
Step 3  Run the MusicRecommender scorer on the candidate pool.
Step 4  (Optional reflection) If fewer than ``min_results`` candidates are
        found, relax constraints and retry once.
Step 5  Return results together with a full execution trace so the user can
        audit every decision.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .guardrails import Guardrails
from .recommender import MusicRecommender
from .retrieval import MusicCatalog


class AgenticPlanner:
    """
    Plan-and-execute controller for music recommendations.

    Parameters
    ----------
    catalog:
        Optional custom :class:`MusicCatalog`.
    min_results:
        If the first recommendation pass returns fewer than this many results
        with a positive score the planner will retry with relaxed constraints.
    """

    def __init__(
        self,
        catalog: Optional[MusicCatalog] = None,
        min_results: int = 3,
    ) -> None:
        self.catalog = catalog or MusicCatalog()
        self.recommender = MusicRecommender(self.catalog)
        self.min_results = min_results

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        tags: Optional[List[str]] = None,
        bpm: Optional[int] = None,
        top_n: int = 5,
        query: Optional[str] = None,
    ) -> Dict:
        """
        Execute the full agentic plan and return a structured result.

        Returns
        -------
        dict with keys:
          ``success``   – bool
          ``results``   – list of recommendation dicts
          ``trace``     – list of step description strings
          ``errors``    – list of validation error strings (may be empty)
        """
        trace: List[str] = []
        errors: List[str] = []

        # ── Step 1: validate inputs ──────────────────────────────────
        trace.append("Step 1: Validating inputs via Guardrails.")
        validation = Guardrails.validate(
            genre=genre, mood=mood, bpm=bpm, top_n=top_n, query=query, tags=tags
        )
        if not validation:
            errors.extend(validation.errors)
            trace.append(f"  ✗ Validation failed: {validation.errors}")
            return {"success": False, "results": [], "trace": trace, "errors": errors}
        trace.append("  ✓ All inputs are valid.")

        # ── Step 2: retrieve candidates ──────────────────────────────
        trace.append("Step 2: Retrieving candidate songs from catalog.")
        if query:
            candidates = self.catalog.search(query)
            trace.append(
                f"  Keyword search for '{query}' → {len(candidates)} hit(s)."
            )
            if not candidates:
                trace.append("  No keyword matches; falling back to full catalog.")
        else:
            candidates = self.catalog.all_songs()
            trace.append(f"  Full catalog loaded: {len(candidates)} song(s).")

        # ── Step 3: score & rank ─────────────────────────────────────
        trace.append("Step 3: Scoring and ranking candidates.")
        results = self.recommender.recommend(
            genre=genre, mood=mood, tags=tags, bpm=bpm, top_n=top_n, query=query
        )
        positive_results = [r for r in results if r["score"] > 0]
        trace.append(
            f"  Returned {len(results)} result(s), "
            f"{len(positive_results)} with positive score."
        )

        # ── Step 4: reflect & retry ──────────────────────────────────
        if len(positive_results) < self.min_results and (genre or mood):
            trace.append(
                f"Step 4: Only {len(positive_results)} result(s) have a positive score "
                f"(min_results={self.min_results}). Relaxing constraints and retrying."
            )
            relaxed = self.recommender.recommend(
                genre=None, mood=mood, tags=tags, bpm=bpm, top_n=top_n
            )
            if len([r for r in relaxed if r["score"] > 0]) > len(positive_results):
                results = relaxed
                trace.append(
                    f"  Relaxed run produced {len([r for r in relaxed if r['score'] > 0])} "
                    "positive-score result(s). Using relaxed results."
                )
            else:
                trace.append("  Relaxed run did not improve results. Keeping original.")
        else:
            trace.append("Step 4: Result quality is acceptable; no retry needed.")

        # ── Step 5: return ───────────────────────────────────────────
        trace.append("Step 5: Returning final recommendations.")
        return {
            "success": True,
            "results": results,
            "trace": trace,
            "errors": [],
        }

    def explain_run(self, **kwargs) -> str:
        """Run the planner and return a human-readable trace + results."""
        outcome = self.run(**kwargs)
        lines = ["=== Agentic Planner Execution Trace ==="]
        for step in outcome["trace"]:
            lines.append(f"  {step}")
        lines.append("")
        if outcome["success"]:
            lines.append("=== Final Recommendations ===")
            for i, r in enumerate(outcome["results"], 1):
                lines.append(
                    f"  {i}. {r['title']} — {r['artist']}"
                    f" [{r['genre']}, {r['mood']}, {r['bpm']} BPM]"
                    f"  score={r['score']}"
                )
        else:
            lines.append("Errors:")
            for e in outcome["errors"]:
                lines.append(f"  • {e}")
        return "\n".join(lines)
