"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.
"""

from src.recommender import load_songs, recommend_songs


def print_recommendations(label: str, user_prefs: dict, songs: list, k: int = 5) -> None:
    divider = "=" * 68
    print(f"\n{divider}")
    print(f" USER PROFILE: {label}")
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


def main() -> None:
    songs = load_songs("data/songs.csv")

    profiles = [
        (
            "genre=pop, mood=happy, energy=0.8, likes_acoustic=False",
            {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False},
        ),
        (
            "genre=lofi, mood=chill, energy=0.4, likes_acoustic=True",
            {"genre": "lofi", "mood": "chill", "energy": 0.4, "likes_acoustic": True},
        ),
        (
            "genre=rock, mood=intense, energy=0.9, likes_acoustic=False",
            {"genre": "rock", "mood": "intense", "energy": 0.9, "likes_acoustic": False},
        ),
    ]

    for label, prefs in profiles:
        print_recommendations(label, prefs, songs, k=5)


if __name__ == "__main__":
    main()
