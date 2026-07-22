"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.
"""

from src.recommender import load_songs, recommend_songs


def print_recommendations(name: str, user_prefs: dict, songs: list, k: int = 5) -> None:
    divider = "=" * 68
    # Descriptive name on top; the profile's actual key/values underneath, so
    # the readable label no longer hides the concrete prefs it stands for.
    details = ", ".join(f"{key}={value}" for key, value in user_prefs.items())
    print(f"\n{divider}")
    print(f" USER PROFILE: {name}")
    print(f" ({details})")
    print(divider)

    results = recommend_songs(user_prefs, songs, k=k)
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


def _banner(title: str) -> None:
    bar = "#" * 68
    print(f"\n{bar}\n# {title}\n{bar}")


def main() -> None:
    songs = load_songs("data/songs.csv")

    _banner("STANDARD PROFILES")
    for name, prefs in STANDARD_PROFILES:
        print_recommendations(name, prefs, songs, k=5)

    _banner("ADVERSARIAL / EDGE-CASE PROFILES")
    for name, prefs in ADVERSARIAL_PROFILES:
        print_recommendations(name, prefs, songs, k=5)


if __name__ == "__main__":
    main()
