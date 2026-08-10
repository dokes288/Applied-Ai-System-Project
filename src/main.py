"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

Scoring modes (Strategy pattern):
    python -m src.main                 # default "balanced" mode
    python -m src.main genre-first     # run every profile in Genre-First mode
    python -m src.main mood-first
    python -m src.main energy-focused
    python -m src.main compare         # show one profile ranked by ALL modes
    python -m src.main diversity       # show top-5 with vs without the diversity penalty
    python -m src.main table           # show each profile's top-5 as a formatted table

AI features:
    python -m src.main ask "nostalgic 80s synthwave, nothing explicit"
                                       # natural-language RAG recommendation
                                       # (uses Claude when ANTHROPIC_API_KEY is
                                       #  set; deterministic offline path otherwise)
    python -m src.main reliability     # run the reliability/quality gate
"""

import logging
import os
import sys
import textwrap

from src.recommender import (
    load_songs,
    recommend_songs,
    BALANCED,
    STRATEGIES,
    ScoringStrategy,
)

# Use `tabulate` if it's installed for a nicely-formatted grid; otherwise fall
# back to a self-contained ASCII renderer so the table works with no extra
# dependency. (tabulate ships transitively with pandas/streamlit here.)
try:
    from tabulate import tabulate as _tabulate
    _HAVE_TABULATE = True
except ImportError:  # pragma: no cover - exercised only when tabulate is absent
    _HAVE_TABULATE = False

REASON_WRAP_WIDTH = 46  # wrap the Reasons column so wide rows stay readable


def print_recommendations(
    name: str, user_prefs: dict, songs: list, k: int = 5, strategy: ScoringStrategy = BALANCED
) -> None:
    divider = "=" * 68
    # Descriptive name on top; the profile's actual key/values underneath, so
    # the readable label no longer hides the concrete prefs it stands for.
    details = ", ".join(f"{key}={value}" for key, value in user_prefs.items())
    print(f"\n{divider}")
    print(f" USER PROFILE: {name}")
    print(f" ({details})")
    print(divider)

    results = recommend_songs(user_prefs, songs, k=k, strategy=strategy)
    if not results:
        print("\n  No recommendations found.")
        return

    for rank, (song, score, explanation) in enumerate(results, start=1):
        print(f"\n{rank}. {song['title']} — {song['artist']}")
        print(f"   Score: {score:.2f}")
        reasons = explanation.split(" | ")
        print("   Reasons:")
        for reason in reasons:
            print(f"     - {reason}")


def print_mode_comparison(name: str, user_prefs: dict, songs: list, k: int = 5) -> None:
    """Ranks one profile under every scoring mode, side by side, so the effect
    of switching strategies is easy to see."""
    details = ", ".join(f"{key}={value}" for key, value in user_prefs.items())
    print(f"\nProfile: {name}  ({details})")
    for strategy in STRATEGIES.values():
        results = recommend_songs(user_prefs, songs, k=k, strategy=strategy)
        picks = ", ".join(f"{song['title']} ({score:.2f})" for song, score, _ in results)
        print(f"\n  [{strategy.name}] {strategy.description}")
        print(f"    {picks}")


def print_diversity_comparison(name: str, user_prefs: dict, songs: list, k: int = 5) -> None:
    """Shows the top-k for one profile with and without the diversity penalty,
    so the de-duplication of artists/genres is visible."""
    details = ", ".join(f"{key}={value}" for key, value in user_prefs.items())
    print(f"\nProfile: {name}  ({details})")

    def _line(song):
        return f"{song['title']} — {song['artist']} [{song['genre']}]"

    print("\n  Without diversity penalty:")
    for rank, (song, score, _) in enumerate(recommend_songs(user_prefs, songs, k=k), 1):
        print(f"    {rank}. {_line(song)}  ({score:.2f})")

    print("\n  With diversity penalty (artist -1.5 each, genre -0.75 each):")
    for rank, (song, score, explanation) in enumerate(
        recommend_songs(user_prefs, songs, k=k, diversity=True), 1
    ):
        note = ""
        for part in explanation.split(" | "):
            if part.startswith("Diversity penalty"):
                note = f"   <- {part}"
        print(f"    {rank}. {_line(song)}  ({score:.2f}){note}")


# The four canonical demo profiles. These are the ones whose output is pasted
# into README.md's "Sample Recommendation Output" section, so their prefs must
# stay in sync with that documented output.
STANDARD_PROFILES = [
    (
        "High-Energy Pop",
        {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False},
    ),
    (
        "Chill Lofi",
        {"genre": "lofi", "mood": "chill", "energy": 0.4, "likes_acoustic": True},
    ),
    (
        "Deep Intense Rock",
        {"genre": "rock", "mood": "intense", "energy": 0.9, "likes_acoustic": False},
    ),
    (
        # Deliberate near-miss: "metal"/"angry" match no song in the catalog,
        # so both categorical gates score 0.0 for every track. Recommendations
        # here are decided purely by energy + acoustic similarity, showing how
        # the ranking behaves when the gates never fire (max reachable = 3.0).
        "Off-Catalog Taste (no genre/mood match)",
        {"genre": "metal", "mood": "angry", "energy": 0.85, "likes_acoustic": False},
    ),
]

# Adversarial / edge-case profiles: each is built to probe a specific weakness
# in the scoring logic. Their output is documented in model_card.md.
ADVERSARIAL_PROFILES = [
    (
        # Conflicting taste: a "chill" mood paired with near-max energy. Nothing
        # rejects the contradiction -- the chill mood gate still fires (+1.0)
        # while the 0.95 energy target fights it, so the two signals cancel.
        "Conflicting: Chill Mood + High Energy",
        {"genre": "lofi", "mood": "chill", "energy": 0.95, "likes_acoustic": True},
    ),
    (
        # Out-of-range input: energy accidentally passed on a 0-100 scale. The
        # target is clamped to 1.0 and the energy term is floored at 0.0, so the
        # genre/mood gates survive instead of being cancelled by a negative term.
        "Out-of-Range Energy (0-100 scale mixup)",
        {"genre": "pop", "mood": "happy", "energy": 2.0, "likes_acoustic": False},
    ),
    (
        # Opt-in popularity, "wants popular": identical to High-Energy Pop but
        # with prefers_popular=True, so popular tracks (popularity >= 70) earn
        # an extra +1.0 that indifferent profiles never see.
        "Wants Popular (prefers_popular=True)",
        {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False, "prefers_popular": True},
    ),
    (
        # Opt-in popularity, "wants niche": niche tracks (popularity <= 30) earn
        # +0.75, surfacing under-the-radar picks the default ranking buries.
        "Wants Niche (prefers_popular=False)",
        {"genre": "lofi", "mood": "chill", "energy": 0.4, "likes_acoustic": True, "prefers_popular": False},
    ),
]


def _wrap_reasons(explanation: str, width: int = REASON_WRAP_WIDTH) -> list:
    """Splits an explanation ("a | b | c") into individually-wrapped reason lines."""
    lines = []
    for reason in explanation.split(" | "):
        lines.extend(textwrap.wrap(reason, width) or [reason])
    return lines


def _ascii_table(headers: list, rows: list, aligns: list) -> str:
    """Minimal dependency-free table renderer. Each cell is a list of lines, so
    multi-line cells (like the wrapped Reasons column) render correctly."""
    widths = [len(h) for h in headers]
    for cells in rows:
        for i, lines in enumerate(cells):
            for line in lines:
                widths[i] = max(widths[i], len(line))

    def fmt(text, i):
        return text.rjust(widths[i]) if aligns[i] == "right" else text.ljust(widths[i])

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [sep, "| " + " | ".join(fmt(h, i) for i, h in enumerate(headers)) + " |", sep]
    for cells in rows:
        height = max(len(c) for c in cells)
        for r in range(height):
            row_line = [fmt(c[r] if r < len(c) else "", i) for i, c in enumerate(cells)]
            out.append("| " + " | ".join(row_line) + " |")
        out.append(sep)
    return "\n".join(out)


def print_recommendations_table(
    name: str, user_prefs: dict, songs: list, k: int = 5,
    strategy: ScoringStrategy = BALANCED, diversity: bool = False,
) -> None:
    """Prints one profile's top-k as a formatted table that includes the full
    reasons for every score. Uses tabulate when available, ASCII otherwise."""
    details = ", ".join(f"{key}={value}" for key, value in user_prefs.items())
    print(f"\nUSER PROFILE: {name}  ({details})")

    results = recommend_songs(user_prefs, songs, k=k, strategy=strategy, diversity=diversity)
    if not results:
        print("  No recommendations found.")
        return

    headers = ["#", "Title", "Artist", "Score", "Reasons"]
    if _HAVE_TABULATE:
        rows = [
            [rank, song["title"], song["artist"], f"{score:.2f}",
             "\n".join(_wrap_reasons(explanation))]
            for rank, (song, score, explanation) in enumerate(results, start=1)
        ]
        print(_tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        rows = [
            [[str(rank)], [song["title"]], [song["artist"]], [f"{score:.2f}"],
             _wrap_reasons(explanation)]
            for rank, (song, score, explanation) in enumerate(results, start=1)
        ]
        print(_ascii_table(headers, rows, aligns=["right", "left", "left", "right", "left"]))


def _banner(title: str) -> None:
    bar = "#" * 68
    print(f"\n{bar}\n# {title}\n{bar}")


def _select_mode(argv: list) -> str:
    """Reads the mode from the command line. Returns a key into STRATEGIES, or
    the special value 'compare'. Falls back to 'balanced' for no/unknown args."""
    if len(argv) < 2:
        return "balanced"
    arg = argv[1].strip().lower()
    if arg in ("compare", "diversity", "table") or arg in STRATEGIES:
        return arg
    print(f"Unknown mode '{argv[1]}'. Options: {', '.join(STRATEGIES)}, compare, diversity, table, ask, reliability.")
    print("Falling back to 'balanced'.")
    return "balanced"


def print_rag_result(result) -> None:
    """Pretty-print a RAG pipeline result: parsed profile, retrieved songs, and
    the grounded natural-language answer."""
    prefs = {k: v for k, v in result.profile.to_prefs().items() if v not in ("", None)}
    _banner("VIBEMATCH — NATURAL LANGUAGE REQUEST")
    print(f'\nYou asked: "{result.query_text}"')
    print(f"\nParsed profile ({result.parse_engine}): {prefs}")
    print("\nRetrieved (content-based engine):")
    for rank, (song, score, _reasons) in enumerate(result.retrieved, start=1):
        print(f"  {rank}. {song['title']} — {song['artist']} [{song['genre']}]  ({score:.2f})")
    print(f"\nRecommendation ({result.generate_engine}):")
    for line in textwrap.wrap(result.answer, 76) or [result.answer]:
        print(f"  {line}")
    if result.warnings:
        print("\nGuardrail notes:")
        for w in result.warnings:
            print(f"  - {w}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO if os.environ.get("VIBEMATCH_DEBUG") else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # AI feature 1 — RAG: natural-language request. Everything after "ask" is
    # the query, so it can contain spaces without quoting on some shells.
    if len(sys.argv) >= 2 and sys.argv[1].strip().lower() == "ask":
        from src.rag import recommend_rag
        query = " ".join(sys.argv[2:]).strip()
        if not query:
            print('Usage: python -m src.main ask "your request in plain English"')
            return
        songs = load_songs("data/songs.csv")
        print_rag_result(recommend_rag(query, songs, k=5))
        return

    # AI feature 2 — reliability/quality gate. Exits non-zero on regression.
    if len(sys.argv) >= 2 and sys.argv[1].strip().lower() == "reliability":
        from src.reliability import main as reliability_main
        reliability_main()
        return

    songs = load_songs("data/songs.csv")
    mode = _select_mode(sys.argv)

    # Compare mode: rank one representative profile under every strategy so the
    # difference between modes is obvious at a glance.
    if mode == "compare":
        _banner("SCORING-MODE COMPARISON")
        for name, prefs in STANDARD_PROFILES:
            print_mode_comparison(name, prefs, songs, k=3)
        return

    # Diversity mode: show each profile's top-5 with and without the penalty so
    # the artist/genre de-duplication is obvious.
    if mode == "diversity":
        _banner("DIVERSITY / FAIRNESS PENALTY  —  before vs after")
        for name, prefs in STANDARD_PROFILES:
            print_diversity_comparison(name, prefs, songs, k=5)
        return

    # Table mode: formatted table (with full reasons) for each profile.
    if mode == "table":
        engine = "tabulate" if _HAVE_TABULATE else "ASCII fallback"
        _banner(f"TOP RECOMMENDATIONS TABLE  ({engine})")
        for name, prefs in STANDARD_PROFILES:
            print_recommendations_table(name, prefs, songs, k=5)
        return

    strategy = STRATEGIES[mode]
    _banner(f"STANDARD PROFILES  —  {strategy.name} mode")
    print(f"  ({strategy.description})")
    for name, prefs in STANDARD_PROFILES:
        print_recommendations(name, prefs, songs, k=5, strategy=strategy)

    _banner(f"ADVERSARIAL / EDGE-CASE PROFILES  —  {strategy.name} mode")
    for name, prefs in ADVERSARIAL_PROFILES:
        print_recommendations(name, prefs, songs, k=5, strategy=strategy)


if __name__ == "__main__":
    main()
