"""
Evaluation harness for the VibeMatch RAG pipeline (stretch: Test Harness).

Runs the system on a fixed set of predefined inputs and prints a per-case
summary with a **confidence rating** and a **pass/fail** verdict, plus overall
averages. Exits non-zero if the pass rate falls below a threshold, so it can gate
a build. It runs on the deterministic offline path, so it needs no API key and
produces the same numbers every run.

This complements `reliability.py`: reliability aggregates a few numeric metrics
into a go/no-go gate, while this harness gives a readable per-input report with a
confidence score for each recommendation.

Confidence rating (0.0-1.0), a heuristic combining two signals:
    score_component  = min(1.0, top_score / 6.0)          # how strong the top pick is
    margin_component = min(1.0, (top_score - runner_up) / 2.0)  # how clearly it wins
    confidence       = 0.6 * score_component + 0.4 * margin_component
A high-scoring pick that clearly beats the field scores near 1.0; a weak pick in
a tight field scores low.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import sys

from src.recommender import load_songs
from src.rag import recommend_rag, grounding_check

# (query, expected_top_genre or None). None means "no in-catalog genre expected"
# (edge / off-catalog cases) — those pass on graceful, grounded handling.
EVAL_CASES: List[Tuple[str, Optional[str]]] = [
    ("upbeat happy pop, something popular", "pop"),
    ("chill lofi for studying, instrumental", "lofi"),
    ("intense rock for a workout, popular hits", "rock"),
    ("nostalgic 80s synthwave, nothing explicit", "synthwave"),
    ("ambient space music, no vocals", "ambient"),
    ("niche jazz, relaxed and warm", "jazz"),
    ("something indie pop and happy", "indie pop"),
    ("polka accordion music from Mars", None),        # off-catalog -> graceful
    ("dreamy calm music to fall asleep to", None),    # no explicit genre -> graceful
]

PASS_RATE_THRESHOLD = 0.85


def confidence(retrieved: List[Tuple[Dict, float, str]]) -> float:
    """Heuristic 0.0-1.0 confidence for the top recommendation."""
    if not retrieved:
        return 0.0
    top_score = retrieved[0][1]
    runner_up = retrieved[1][1] if len(retrieved) > 1 else 0.0
    score_component = min(1.0, top_score / 6.0)
    margin_component = min(1.0, max(0.0, (top_score - runner_up) / 2.0))
    return round(0.6 * score_component + 0.4 * margin_component, 2)


def run_eval(songs: List[Dict]) -> Dict:
    catalog_genres = {s["genre"] for s in songs}
    rows = []
    passed = 0
    conf_sum = 0.0
    for query, expected in EVAL_CASES:
        result = recommend_rag(query, songs, k=5, use_llm="offline")
        grounded, _ = grounding_check(result.answer, result.retrieved, songs)
        top = result.retrieved[0][0] if result.retrieved else None
        top_genre = top["genre"] if top else None
        conf = confidence(result.retrieved)
        conf_sum += conf

        # Pass criteria: always grounded; if a catalog genre was expected, the
        # #1 result must be in that genre. Edge cases (expected None) pass on
        # graceful, grounded handling alone.
        if expected in catalog_genres:
            ok = grounded and (top_genre == expected)
            criteria = f"top genre == {expected}, grounded"
        else:
            ok = grounded and bool(result.retrieved)
            criteria = "graceful + grounded (no in-catalog genre)"
        passed += int(ok)
        rows.append({
            "query": query,
            "criteria": criteria,
            "top_pick": top["title"] if top else None,
            "top_genre": top_genre,
            "confidence": conf,
            "grounded": grounded,
            "result": "PASS" if ok else "FAIL",
        })

    n = len(EVAL_CASES)
    return {
        "rows": rows,
        "passed": passed,
        "total": n,
        "pass_rate": round(passed / n, 2) if n else 1.0,
        "avg_confidence": round(conf_sum / n, 2) if n else 0.0,
    }


def print_report(report: Dict) -> bool:
    print("=" * 78)
    print(f" VibeMatch evaluation harness  ({report['total']} predefined inputs, offline path)")
    print("=" * 78)
    print(f"{'input':<42}{'top pick':<18}{'conf':>6}{'':>2}{'result'}")
    print("-" * 78)
    for r in report["rows"]:
        q = (r["query"][:39] + "...") if len(r["query"]) > 42 else r["query"]
        pick = (r["top_pick"] or "-")[:16]
        print(f"{q:<42}{pick:<18}{r['confidence']:>6.2f}  {r['result']}")
    print("-" * 78)
    print(f"Passed: {report['passed']}/{report['total']}"
          f"  |  pass rate: {report['pass_rate']:.2f} (threshold {PASS_RATE_THRESHOLD:.2f})"
          f"  |  avg confidence: {report['avg_confidence']:.2f}")
    ok = report["pass_rate"] >= PASS_RATE_THRESHOLD
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


def main() -> None:
    songs = load_songs("data/songs.csv")
    report = run_eval(songs)
    ok = print_report(report)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
