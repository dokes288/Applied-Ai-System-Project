"""
Reliability & testing system for the VibeMatch RAG pipeline.

This is the second advanced AI feature: a harness that *measures how well and
how consistently* the AI-assisted recommender behaves, and fails (non-zero exit)
if any metric drops below a threshold. It runs entirely on the deterministic
OFFLINE path (`use_llm="offline"`), so it needs no API key and produces the same
numbers on every machine -- a reproducible quality gate you can wire into CI.

Metrics
    1. Parse determinism   -- same query parsed twice yields identical profiles.
    2. Parse accuracy      -- parsed fields match hand-labeled expectations.
    3. Retrieval precision -- when a genre is requested and exists in the
                              catalog, the #1 result is in that genre (P@1).
    4. Grounding rate       -- generated answers mention only retrieved songs
                              (the RAG safety property the grounding guard
                              enforces).
    5. End-to-end determinism -- same query yields the identical final answer.
"""

from __future__ import annotations

from typing import Dict, List
import sys

from src.recommender import load_songs
from src.rag import recommend_rag, offline_parse, grounding_check

# (query, {field: expected_value}) -- only the listed fields are checked.
LABELED_QUERIES = [
    ("I want upbeat happy pop, something popular", {"genre": "pop", "mood": "happy", "prefers_popular": True}),
    ("chill lofi to study to, acoustic and mellow", {"genre": "lofi", "mood": "chill", "likes_acoustic": True}),
    ("intense rock for a workout", {"genre": "rock", "mood": "intense", "energy": 0.9}),
    ("nostalgic 80s synthwave, nothing explicit", {"genre": "synthwave", "preferred_decade": 1980, "allow_explicit": False}),
    ("instrumental ambient with no lyrics", {"genre": "ambient", "preferred_language": "instrumental"}),
    ("niche jazz, relaxed and warm", {"genre": "jazz", "mood": "relaxed", "prefers_popular": False}),
]

# Quality gate. Tuned to the current deterministic pipeline; lower a number here
# only when you intend to accept a regression.
THRESHOLDS = {
    "parse_determinism": 1.0,
    "parse_accuracy": 0.90,
    "retrieval_precision_at_1": 1.0,
    "grounding_rate": 1.0,
    "e2e_determinism": 1.0,
}


def _catalog_genres(songs: List[Dict]) -> set:
    return {s["genre"] for s in songs}


def run_reliability(songs: List[Dict]) -> Dict:
    genres = _catalog_genres(songs)

    parse_det_hits = 0
    parse_field_hits = parse_field_total = 0
    retr_hits = retr_total = 0
    grounded_hits = grounded_total = 0
    e2e_det_hits = 0

    details = []
    for query, expected in LABELED_QUERIES:
        # 1. Parse determinism
        p1, p2 = offline_parse(query), offline_parse(query)
        det = (p1.to_prefs() == p2.to_prefs())
        parse_det_hits += int(det)

        # 2. Parse accuracy (per labeled field)
        prefs = p1.to_prefs()
        field_ok = 0
        for field_name, want in expected.items():
            got = prefs.get(field_name)
            if isinstance(want, float):
                ok = got is not None and abs(float(got) - want) < 1e-9
            else:
                ok = (got == want)
            field_ok += int(ok)
            parse_field_total += 1
        parse_field_hits += field_ok

        # 3 + 4 + 5. Retrieval, grounding, end-to-end determinism (offline).
        r1 = recommend_rag(query, songs, k=5, use_llm="offline")
        r2 = recommend_rag(query, songs, k=5, use_llm="offline")

        if expected.get("genre") in genres:
            retr_total += 1
            top_genre = r1.retrieved[0][0]["genre"] if r1.retrieved else None
            retr_hits += int(top_genre == expected["genre"])

        grounded_total += 1
        grounded, hallucinated = grounding_check(r1.answer, r1.retrieved, songs)
        grounded_hits += int(grounded)

        e2e_det_hits += int(r1.answer == r2.answer)

        details.append({
            "query": query,
            "parse_deterministic": det,
            "fields_correct": f"{field_ok}/{len(expected)}",
            "top_pick": r1.retrieved[0][0]["title"] if r1.retrieved else None,
            "grounded": grounded,
        })

    n = len(LABELED_QUERIES)
    metrics = {
        "parse_determinism": parse_det_hits / n,
        "parse_accuracy": parse_field_hits / parse_field_total if parse_field_total else 1.0,
        "retrieval_precision_at_1": retr_hits / retr_total if retr_total else 1.0,
        "grounding_rate": grounded_hits / grounded_total if grounded_total else 1.0,
        "e2e_determinism": e2e_det_hits / n,
    }
    passed = {m: metrics[m] >= THRESHOLDS[m] for m in metrics}
    return {"metrics": metrics, "passed": passed, "details": details, "n": n}


def print_report(report: Dict) -> bool:
    """Print the metrics table. Returns True if every metric passed."""
    print("=" * 60)
    print(f" VibeMatch reliability report  ({report['n']} labeled queries)")
    print("=" * 60)
    print(f"{'metric':<28}{'score':>8}{'threshold':>12}{'':>4}")
    print("-" * 60)
    all_ok = True
    for metric, score in report["metrics"].items():
        ok = report["passed"][metric]
        all_ok = all_ok and ok
        print(f"{metric:<28}{score:>8.2f}{THRESHOLDS[metric]:>12.2f}{'  OK' if ok else '  FAIL'}")
    print("-" * 60)
    print("RESULT:", "PASS" if all_ok else "FAIL")
    return all_ok


def main() -> None:
    songs = load_songs("data/songs.csv")
    report = run_reliability(songs)
    all_ok = print_report(report)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
