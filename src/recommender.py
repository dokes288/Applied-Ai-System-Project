from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import csv


# Tunable weights — change these to run experiments (see README).
WEIGHT_GENRE = 3.5
WEIGHT_MOOD = 3.0
WEIGHT_ENERGY = 2.5
WEIGHT_VALENCE = 1.0
WEIGHT_TEMPO = 0.5
WEIGHT_ACOUSTIC_MATCH = 1.5
WEIGHT_ACOUSTIC_NON_MATCH = 1.0
WEIGHT_POPULARITY_MATCH = 1.0       # user prefers popular tracks, song clears the popularity bar
WEIGHT_POPULARITY_NON_MATCH = 0.75   # user prefers niche/under-the-radar tracks, song is obscure enough

# Threshold logic mirrors WEIGHT_ACOUSTIC_MATCH/NON_MATCH exactly: a hard
# cutoff rather than a continuous scale, consistent with how acousticness
# is already scored below.
POPULARITY_MATCH_THRESHOLD = 70.0     # song.popularity >= this counts as "popular"
POPULARITY_NON_MATCH_THRESHOLD = 30.0  # song.popularity <= this counts as "niche"

MOOD_VALENCE_TARGETS = {
    "happy": 0.85,
    "chill": 0.60,
    "relaxed": 0.70,
    "intense": 0.45,
    "moody": 0.40,
    "focused": 0.55,
}


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    popularity: float = 50.0  # 0-100; defaults to neutral so existing Song(...) calls still work


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    prefers_popular: Optional[bool] = None  # True=wants popular, False=wants niche, None=indifferent


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # Same Pythonic shape as recommend_songs(): comprehension judges
        # every song, sorted() ranks the result without mutating anything.
        scored_songs = [
            (score, reasons, song)
            for song, (score, reasons) in ((s, self._score_song(user, s)) for s in self.songs)
        ]
        ranked = sorted(scored_songs, key=lambda item: (-item[0], item[2].title))
        return [song for _, _, song in ranked[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        score, reasons = self._score_song(user, song)
        if reasons:
            return " | ".join(reasons)
        return f"This song fits your taste profile with a score of {score:.2f}."

    def _score_song(self, user: UserProfile, song: Song) -> Tuple[float, List[str]]:
        # Delegates to score_song() rather than duplicating its logic, so the
        # OOP and functional paths cannot drift apart again -- both call the
        # exact same code, on dict views of the same dataclass fields.
        return score_song(
            {
                "favorite_genre": user.favorite_genre,
                "favorite_mood": user.favorite_mood,
                "target_energy": user.target_energy,
                "likes_acoustic": user.likes_acoustic,
                "prefers_popular": user.prefers_popular,
            },
            {
                "genre": song.genre,
                "mood": song.mood,
                "energy": song.energy,
                "acousticness": song.acousticness,
                "popularity": song.popularity,
            },
        )


def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    print(f"Loading songs from {csv_path}...")
    songs: List[Dict] = []
    path = Path(csv_path)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            songs.append(
                {
                    "id": int(row["id"]),
                    "title": row["title"],
                    "artist": row["artist"],
                    "genre": row["genre"],
                    "mood": row["mood"],
                    "energy": float(row["energy"]),
                    "tempo_bpm": float(row["tempo_bpm"]),
                    "valence": float(row["valence"]),
                    "danceability": float(row["danceability"]),
                    "acousticness": float(row["acousticness"]),
                    "popularity": float(row["popularity"]) if row.get("popularity") not in (None, "") else 50.0,
                }
            )

    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences using the finalized
    baseline Algorithm Recipe:

        score = genre_match(+2.0) + mood_match(+1.0) + energy_similarity(up to +2.0)
              + acoustic_similarity(up to +1.0) + popularity_bonus(up to +1.0)

    Max score is 6.0 when the user is indifferent to popularity
    (prefers_popular is None), or 7.0 when they state a popularity
    preference and the song clears the matching threshold.

    energy_similarity is clamped to [0.0, 2.0]: target_energy is first
    clamped into [0, 1], and the term is floored at 0.0, so an
    out-of-range energy value can never subtract from the total.

    popularity_bonus is opt-in and threshold-based (mirroring the acoustic
    weights): +1.0 if prefers_popular is True and song.popularity >= 70,
    +0.75 if prefers_popular is False and song.popularity <= 30, else 0.0.
    prefers_popular=None (the default) always scores 0.0 here.

    Genre and mood are strict binary gates (normalized equality, NOT
    substring matching -- "pop" does not credit against "indie pop").
    This matches what was actually validated during design: the
    Sunrise City / Storm Runner test cases assumed exact-match gates.

    Energy is a continuous similarity term:
        energy_similarity = 2.0 * (1 - |song.energy - target_energy|)

    Acoustic fit is a continuous similarity term, same style as energy
    (not a hard threshold like the original 7-component formula's
    acoustic term -- that version left a 0.3-0.7 acousticness dead zone
    earning nothing in either direction, which didn't fit a from-scratch
    recipe built around continuous distance rather than ported gates):
        target_acousticness = 1.0 if likes_acoustic else 0.0
        acoustic_similarity  = 1.0 * (1 - |acousticness - target_acousticness|)
    Every acousticness value now contributes proportionally, in whichever
    direction the user's preference points.

    Recommender._score_song() delegates to this same function, so the
    OOP and functional paths agree on both scores and rankings for the
    same profile/catalog. _compute_score() (the full 7-component
    weighted formula: genre 3.5, mood 3.0, energy 2.5, ...) is no longer
    called by either path -- it's left in place, unused, in case
    external code still references it directly.

    Required by recommend_songs() and src/main.py
    """
    favorite_genre = user_prefs.get("genre") or user_prefs.get("favorite_genre", "")
    favorite_mood = user_prefs.get("mood") or user_prefs.get("favorite_mood", "")
    target_energy = user_prefs.get("energy")
    if target_energy is None:
        target_energy = user_prefs.get("target_energy", 0.5)
    target_energy = float(target_energy)
    # Clamp to the valid [0, 1] domain so an out-of-range target (e.g. energy
    # accidentally passed on a 0-100 scale) can't push energy_similarity
    # negative and quietly cancel out the genre/mood gates.
    target_energy = min(1.0, max(0.0, target_energy))
    likes_acoustic = bool(user_prefs.get("likes_acoustic", False))
    prefers_popular = user_prefs.get("prefers_popular")  # True/False/None (indifferent)

    genre = str(song.get("genre", ""))
    mood = str(song.get("mood", ""))
    energy = float(song.get("energy", 0.0))
    acousticness = float(song.get("acousticness", 0.0))
    popularity = float(song.get("popularity", 50.0))

    score = 0.0
    reasons: List[str] = []

    if _normalize(genre) == _normalize(str(favorite_genre)):
        score += 2.0
        reasons.append(f"Genre match: {genre} (+2.0)")

    if _normalize(mood) == _normalize(str(favorite_mood)):
        score += 1.0
        reasons.append(f"Mood match: {mood} (+1.0)")

    # Floored at 0.0 as defense-in-depth alongside the target clamp above: an
    # out-of-range song.energy must never subtract from the total.
    energy_similarity = max(0.0, 2.0 * (1.0 - abs(energy - target_energy)))
    score += energy_similarity
    reasons.append(
        f"Energy similarity: {energy:.2f} vs target {target_energy:.2f} (+{energy_similarity:.2f})"
    )

    target_acousticness = 1.0 if likes_acoustic else 0.0
    acoustic_similarity = 1.0 * (1.0 - abs(acousticness - target_acousticness))
    score += acoustic_similarity
    preference_label = "acoustic" if likes_acoustic else "non-acoustic"
    reasons.append(
        f"Acoustic fit: acousticness {acousticness:.2f} vs {preference_label} preference (+{acoustic_similarity:.2f})"
    )

    # Popularity is opt-in: only scored when the user states a preference.
    # None (indifferent) earns nothing either way, so profiles that omit it --
    # including every profile in src/main.py -- keep their original scores.
    # Threshold-based to mirror the WEIGHT_ACOUSTIC_MATCH/NON_MATCH style.
    if prefers_popular is True and popularity >= POPULARITY_MATCH_THRESHOLD:
        score += WEIGHT_POPULARITY_MATCH
        reasons.append(
            f"Popularity match: {popularity:.0f} vs popular preference (+{WEIGHT_POPULARITY_MATCH:.2f})"
        )
    elif prefers_popular is False and popularity <= POPULARITY_NON_MATCH_THRESHOLD:
        score += WEIGHT_POPULARITY_NON_MATCH
        reasons.append(
            f"Niche pick: popularity {popularity:.0f} vs niche preference (+{WEIGHT_POPULARITY_NON_MATCH:.2f})"
        )

    return round(score, 2), reasons


def _explain(reasons: List[str]) -> str:
    """Formats scoring reasons into a single user-facing explanation string."""
    return " | ".join(reasons) if reasons else "No strong match found."


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.

    Pythonic form: a single list comprehension judges every song with
    score_song(), producing (song, score, explanation) triples -- then
    sorted() (not .sort()) ranks that list as a pure expression, rather
    than mutating a list built up with .append() in a for-loop.

    Tie-break: score descending, then title ascending, so ties resolve
    the same way regardless of catalog order.

    Required by src/main.py
    """
    scored_songs = [
        (song, score, _explain(reasons))
        for song, (score, reasons) in ((s, score_song(user_prefs, s)) for s in songs)
    ]
    return sorted(scored_songs, key=lambda item: (-item[1], item[0].get("title", "")))[:k]


def _compute_score(
    genre: str,
    mood: str,
    energy: float,
    acousticness: float,
    valence: float,
    tempo_bpm: float,
    favorite_genre: str,
    favorite_mood: str,
    target_energy: float,
    likes_acoustic: bool,
    popularity: float = 50.0,
    prefers_popular: Optional[bool] = None,
) -> Tuple[float, List[str]]:
    """Computes the full weighted compatibility score and reason list for one song."""
    score = 0.0
    reasons: List[str] = []

    if _matches(genre, favorite_genre):
        score += WEIGHT_GENRE
        reasons.append(f"Genre matches your favorite style: {genre}")

    if _matches(mood, favorite_mood):
        score += WEIGHT_MOOD
        reasons.append(f"Mood matches your preference: {mood}")

    energy_gap = abs(energy - target_energy)
    energy_score = max(0.0, WEIGHT_ENERGY - (energy_gap * 3.0))
    score += energy_score
    if energy_score > 0.0:
        reasons.append(f"Energy is close to your target ({energy:.2f} vs {target_energy:.2f})")

    target_valence = MOOD_VALENCE_TARGETS.get(_normalize(favorite_mood), 0.6)
    valence_gap = abs(valence - target_valence)
    valence_score = max(0.0, WEIGHT_VALENCE - (valence_gap * 2.0))
    score += valence_score
    if valence_score >= 0.5:
        reasons.append(f"Valence fits your mood ({valence:.2f})")

    expected_tempo = 70 + (target_energy * 80)
    tempo_gap = abs(tempo_bpm - expected_tempo)
    tempo_score = max(0.0, WEIGHT_TEMPO - (tempo_gap / 60.0))
    score += tempo_score
    if tempo_score >= 0.25:
        reasons.append(f"Tempo ({tempo_bpm:.0f} BPM) fits your energy level")

    if likes_acoustic:
        if acousticness >= 0.7:
            score += WEIGHT_ACOUSTIC_MATCH
            reasons.append("You tend to like acoustic tracks")
    else:
        if acousticness <= 0.3:
            score += WEIGHT_ACOUSTIC_NON_MATCH
            reasons.append("The track has a less acoustic profile")

    # prefers_popular is three-state: True (wants popular), False (wants
    # niche/under-the-radar), or None (no stated preference — indifferent,
    # earns no bonus either way). Threshold-based to match the acoustic
    # bonus's style above rather than a continuously scaled bonus.
    if prefers_popular is True:
        if popularity >= POPULARITY_MATCH_THRESHOLD:
            score += WEIGHT_POPULARITY_MATCH
            reasons.append("This track is a well-known crowd favorite")
    elif prefers_popular is False:
        if popularity <= POPULARITY_NON_MATCH_THRESHOLD:
            score += WEIGHT_POPULARITY_NON_MATCH
            reasons.append("This is a niche, under-the-radar pick")

    return round(score, 2), reasons


def _matches(left: str, right: str) -> bool:
    """Returns True when two preference strings match after normalization."""
    left_norm = _normalize(left)
    right_norm = _normalize(right)
    return bool(left_norm and right_norm and (left_norm == right_norm or right_norm in left_norm or left_norm in right_norm))


def _normalize(value: str) -> str:
    """Normalizes text by trimming, lowercasing, and removing spaces."""
    return str(value).strip().lower().replace(" ", "")
