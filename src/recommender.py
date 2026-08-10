from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
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

# --- Advanced feature weights (all OPT-IN: only scored when the user profile
# supplies the matching preference key, so profiles that do not set them keep
# their original scores). See score_song() for the exact rules. ---------------
WEIGHT_MOOD_TAGS = 1.0        # scaled by the fraction of desired detailed tags a song carries
WEIGHT_DECADE_EXACT = 1.0     # song.release_decade == preferred_decade
WEIGHT_DECADE_ADJACENT = 0.5  # song is exactly one decade away
WEIGHT_LANGUAGE = 1.0         # song.language == preferred_language (exact, normalized)
WEIGHT_INSTRUMENTAL = 1.0     # continuous similarity to target_instrumentalness
WEIGHT_EXPLICIT_PENALTY = 2.0  # subtracted when the user disallows explicit content and the song is explicit

# --- Diversity / fairness penalties (applied during RE-RANKING, not per-song
# scoring). See recommend_songs(diversity=True) / _select_diverse(). ----------
# The rule: as the top list is built one pick at a time, a candidate is
# penalized for every already-chosen song that shares its artist (strong) or its
# genre (softer). This stops one artist or genre from monopolizing the top-k.
DIVERSITY_ARTIST_PENALTY = 1.5   # subtracted per already-listed song by the same artist
DIVERSITY_GENRE_PENALTY = 0.75   # subtracted per already-listed song of the same genre

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


# --- Strategy pattern: interchangeable ranking modes -------------------------
# Each ScoringStrategy encapsulates one "ranking approach" as the set of weights
# score_song() applies to the four core signals. Swapping the strategy object
# swaps the ranking behavior without touching the scoring code -- that is the
# whole point of the Strategy pattern: the algorithm family (how much each
# signal counts) varies independently of the client that uses it.
@dataclass(frozen=True)
class ScoringStrategy:
    """One ranking mode. Frozen so named strategies are safe shared constants."""
    name: str
    description: str
    weight_genre: float = 2.0
    weight_mood: float = 1.0
    weight_energy: float = 2.0
    weight_acoustic: float = 1.0


# BALANCED reproduces the original baseline recipe exactly (genre 2.0, mood 1.0,
# energy 2.0, acoustic 1.0), so default behavior, tests, and documented output
# are unchanged. The other modes tilt the weights toward one signal.
BALANCED = ScoringStrategy(
    "Balanced", "Baseline recipe: genre 2.0, mood 1.0, energy 2.0, acoustic 1.0.",
)
GENRE_FIRST = ScoringStrategy(
    "Genre-First", "Genre dominates; other signals only break ties.",
    weight_genre=4.0, weight_mood=1.0, weight_energy=1.0, weight_acoustic=1.0,
)
MOOD_FIRST = ScoringStrategy(
    "Mood-First", "Mood dominates; good for 'match my vibe' over 'match my genre'.",
    weight_genre=1.0, weight_mood=4.0, weight_energy=1.0, weight_acoustic=1.0,
)
ENERGY_FOCUSED = ScoringStrategy(
    "Energy-Focused", "Energy dominates; great for workouts/focus by intensity.",
    weight_genre=1.0, weight_mood=1.0, weight_energy=4.0, weight_acoustic=1.0,
)

# Registry so callers (e.g. main.py's CLI) can look a strategy up by short name.
STRATEGIES: Dict[str, ScoringStrategy] = {
    "balanced": BALANCED,
    "genre-first": GENRE_FIRST,
    "mood-first": MOOD_FIRST,
    "energy-focused": ENERGY_FOCUSED,
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
    # --- Advanced features (all have defaults so existing Song(...) calls in the
    # tests still work without supplying them). ---
    release_decade: int = 0                                  # e.g. 1980, 2010, 2020; 0 = unknown
    mood_tags: List[str] = field(default_factory=list)       # detailed tags, e.g. ["nostalgic", "euphoric"]
    language: str = ""                                       # e.g. "english", "instrumental"
    instrumentalness: float = 0.0                            # 0.0 (vocal) - 1.0 (purely instrumental)
    explicit: bool = False                                   # explicit-content flag


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
    # --- Advanced-feature preferences. All default to None ("indifferent"), so a
    # profile that omits them scores exactly as it did before these existed. ---
    desired_mood_tags: Optional[List[str]] = None    # detailed tags the listener wants, e.g. ["nostalgic"]
    preferred_decade: Optional[int] = None           # e.g. 2010; scores by distance in decades
    preferred_language: Optional[str] = None         # e.g. "english"; exact-match bonus
    target_instrumentalness: Optional[float] = None  # 0.0-1.0; continuous similarity bonus
    allow_explicit: Optional[bool] = None            # False = penalize explicit tracks; None = do not care


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """

    def __init__(self, songs: List[Song], strategy: ScoringStrategy = BALANCED):
        self.songs = songs
        # The strategy is injected here (Strategy pattern): the Recommender holds
        # a reference to an interchangeable ranking mode and delegates weighting
        # to it. Defaults to BALANCED so existing callers behave unchanged.
        self.strategy = strategy

    def recommend(self, user: UserProfile, k: int = 5, diversity: bool = False) -> List[Song]:
        # Same Pythonic shape as recommend_songs(): comprehension judges
        # every song, sorted() ranks the result without mutating anything.
        # When diversity=True, delegate ranking to the shared _select_diverse()
        # re-ranker so the OOP and functional paths stay consistent.
        if diversity:
            scored = [
                (song, score, reasons)
                for song, (score, reasons) in ((s, self._score_song(user, s)) for s in self.songs)
            ]
            return [song for song, _, _ in _select_diverse(scored, k)]
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
                "desired_mood_tags": user.desired_mood_tags,
                "preferred_decade": user.preferred_decade,
                "preferred_language": user.preferred_language,
                "target_instrumentalness": user.target_instrumentalness,
                "allow_explicit": user.allow_explicit,
            },
            {
                "genre": song.genre,
                "mood": song.mood,
                "energy": song.energy,
                "acousticness": song.acousticness,
                "popularity": song.popularity,
                "mood_tags": song.mood_tags,
                "release_decade": song.release_decade,
                "language": song.language,
                "instrumentalness": song.instrumentalness,
                "explicit": song.explicit,
            },
            self.strategy,
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
                    # --- Advanced features. Each falls back to a neutral default
                    # when the column is missing, so pre-existing CSVs still load. ---
                    "release_decade": int(row["release_decade"]) if row.get("release_decade") not in (None, "") else 0,
                    "mood_tags": _parse_tags(row.get("mood_tags", "")),
                    "language": row.get("language", "") or "",
                    "instrumentalness": float(row["instrumentalness"]) if row.get("instrumentalness") not in (None, "") else 0.0,
                    "explicit": _parse_bool(row.get("explicit", "")),
                }
            )

    return songs


def _parse_tags(raw: str) -> List[str]:
    """Splits a pipe-separated tag string ("nostalgic|euphoric") into a clean list."""
    return [tag.strip() for tag in str(raw).split("|") if tag.strip()]


def _parse_bool(raw) -> bool:
    """Parses a CSV truthy string ("true"/"1"/"yes") into a bool; blank/unknown = False."""
    return str(raw).strip().lower() in ("true", "1", "yes", "y")


def score_song(user_prefs: Dict, song: Dict, strategy: ScoringStrategy = BALANCED) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences using the finalized
    baseline Algorithm Recipe:

        score = genre_match(+2.0) + mood_match(+1.0) + energy_similarity(up to +2.0)
              + acoustic_similarity(up to +1.0) + popularity_bonus(up to +1.0)
              + [advanced opt-in terms, see below]

    Base max is 6.0 when the user is indifferent to every optional term.
    Each optional term is scored ONLY when the profile supplies its key, so
    a profile that omits them behaves exactly like the baseline recipe:

      * popularity_bonus     up to +1.00  (prefers_popular True/False)
      * mood_tags_bonus      up to +1.00  (desired_mood_tags: list of detailed tags)
      * decade_bonus         up to +1.00  (preferred_decade: exact +1.0, one decade off +0.5)
      * language_bonus       up to +1.00  (preferred_language: exact normalized match)
      * instrumental_bonus   up to +1.00  (target_instrumentalness: continuous 1-|diff|)
      * explicit_penalty          -2.00   (allow_explicit is False and song.explicit is True)

    With every positive optional term firing, the ceiling is 11.0; the
    explicit penalty can drive an otherwise-good match below zero on purpose,
    acting as a soft content filter rather than a hard exclusion.

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
    earning nothing in either direction, which did not fit a from-scratch
    recipe built around continuous distance rather than ported gates):
        target_acousticness = 1.0 if likes_acoustic else 0.0
        acoustic_similarity  = 1.0 * (1 - |acousticness - target_acousticness|)
    Every acousticness value now contributes proportionally, in whichever
    direction the user's preference points.

    Recommender._score_song() delegates to this same function, so the
    OOP and functional paths agree on both scores and rankings for the
    same profile/catalog. _compute_score() (the full 7-component
    weighted formula: genre 3.5, mood 3.0, energy 2.5, ...) is no longer
    called by either path -- it is left in place, unused, in case
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
    # accidentally passed on a 0-100 scale) cannot push energy_similarity
    # negative and quietly cancel out the genre/mood gates.
    target_energy = min(1.0, max(0.0, target_energy))
    likes_acoustic = bool(user_prefs.get("likes_acoustic", False))
    prefers_popular = user_prefs.get("prefers_popular")  # True/False/None (indifferent)
    desired_mood_tags = user_prefs.get("desired_mood_tags")      # list or None
    preferred_decade = user_prefs.get("preferred_decade")        # int or None
    preferred_language = user_prefs.get("preferred_language")    # str or None
    target_instrumentalness = user_prefs.get("target_instrumentalness")  # float or None
    allow_explicit = user_prefs.get("allow_explicit")           # bool or None

    genre = str(song.get("genre", ""))
    mood = str(song.get("mood", ""))
    energy = float(song.get("energy", 0.0))
    acousticness = float(song.get("acousticness", 0.0))
    popularity = float(song.get("popularity", 50.0))
    song_mood_tags = song.get("mood_tags", []) or []
    song_decade = song.get("release_decade") or 0
    song_language = str(song.get("language", ""))
    instrumentalness = float(song.get("instrumentalness", 0.0))
    explicit = bool(song.get("explicit", False))

    score = 0.0
    reasons: List[str] = []

    if _normalize(genre) == _normalize(str(favorite_genre)):
        score += strategy.weight_genre
        reasons.append(f"Genre match: {genre} (+{strategy.weight_genre:.1f})")

    if _normalize(mood) == _normalize(str(favorite_mood)):
        score += strategy.weight_mood
        reasons.append(f"Mood match: {mood} (+{strategy.weight_mood:.1f})")

    # Floored at 0.0 as defense-in-depth alongside the target clamp above: an
    # out-of-range song.energy must never subtract from the total.
    energy_similarity = max(0.0, strategy.weight_energy * (1.0 - abs(energy - target_energy)))
    score += energy_similarity
    reasons.append(
        f"Energy similarity: {energy:.2f} vs target {target_energy:.2f} (+{energy_similarity:.2f})"
    )

    target_acousticness = 1.0 if likes_acoustic else 0.0
    acoustic_similarity = strategy.weight_acoustic * (1.0 - abs(acousticness - target_acousticness))
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

    # --- Advanced opt-in terms. Each fires only when the profile supplies its
    # key, so baseline profiles are untouched. ---

    # Detailed mood tags: partial credit scaled by how many of the desired tags
    # the song carries (set overlap), so "nostalgic + euphoric" rewards a song
    # tagged with either, fully rewards one tagged with both.
    if desired_mood_tags:
        wanted = {_normalize(t) for t in desired_mood_tags if str(t).strip()}
        have = {_normalize(t) for t in song_mood_tags if str(t).strip()}
        matched = wanted & have
        if wanted and matched:
            tag_bonus = WEIGHT_MOOD_TAGS * (len(matched) / len(wanted))
            score += tag_bonus
            reasons.append(
                f"Mood tags: matched {sorted(matched)} ({len(matched)}/{len(wanted)}) (+{tag_bonus:.2f})"
            )

    # Release decade: exact decade earns full credit, one decade away earns half,
    # anything further earns nothing (a soft nostalgia/era preference).
    if preferred_decade is not None and song_decade:
        decades_apart = abs(int(preferred_decade) - int(song_decade)) // 10
        if decades_apart == 0:
            score += WEIGHT_DECADE_EXACT
            reasons.append(f"Era match: {int(song_decade)}s (+{WEIGHT_DECADE_EXACT:.2f})")
        elif decades_apart == 1:
            score += WEIGHT_DECADE_ADJACENT
            reasons.append(
                f"Era near-match: {int(song_decade)}s vs {int(preferred_decade)}s (+{WEIGHT_DECADE_ADJACENT:.2f})"
            )

    # Language: exact normalized match only.
    if preferred_language:
        if _normalize(preferred_language) == _normalize(song_language):
            score += WEIGHT_LANGUAGE
            reasons.append(f"Language match: {song_language} (+{WEIGHT_LANGUAGE:.2f})")

    # Instrumentalness: continuous similarity, same shape as the acoustic term.
    if target_instrumentalness is not None:
        instr_similarity = max(0.0, WEIGHT_INSTRUMENTAL * (1.0 - abs(instrumentalness - float(target_instrumentalness))))
        score += instr_similarity
        reasons.append(
            f"Instrumental fit: {instrumentalness:.2f} vs target {float(target_instrumentalness):.2f} (+{instr_similarity:.2f})"
        )

    # Explicit content: a soft penalty (not a hard exclusion) when the listener
    # opts out of explicit tracks. Can push a score below zero on purpose.
    if allow_explicit is False and explicit:
        score -= WEIGHT_EXPLICIT_PENALTY
        reasons.append(f"Explicit content (unwanted) (-{WEIGHT_EXPLICIT_PENALTY:.2f})")

    return round(score, 2), reasons


def _explain(reasons: List[str]) -> str:
    """Formats scoring reasons into a single user-facing explanation string."""
    return " | ".join(reasons) if reasons else "No strong match found."


def _field(song, name: str, default: str = "") -> str:
    """Reads an attribute from a song whether it is a dict or a Song dataclass."""
    if isinstance(song, dict):
        return song.get(name, default)
    return getattr(song, name, default)


def _select_diverse(
    scored: List[Tuple], k: int,
    artist_penalty: float = DIVERSITY_ARTIST_PENALTY,
    genre_penalty: float = DIVERSITY_GENRE_PENALTY,
) -> List[Tuple]:
    """
    Greedy diversity re-ranking (a lightweight Maximal-Marginal-Relevance).

    `scored` is a list of (song, base_score, reasons) triples. The list is built
    one pick at a time: at each step every remaining candidate's *effective*
    score is its base score minus a penalty for each already-chosen song that
    shares its artist (artist_penalty) or genre (genre_penalty). The best
    effective score wins (ties broken by title, ascending, to stay deterministic).

    Because penalties grow with each same-artist/same-genre pick already made,
    a second song by an artist already in the list must be clearly better than
    the alternatives to earn its slot -- so one artist or genre cannot dominate.

    Returns (song, effective_score, reasons) triples, with a diversity note
    appended to the reasons of any pick that was penalized.
    """
    selected: List[Tuple] = []
    remaining = list(scored)

    while remaining and len(selected) < k:
        best_idx = -1
        best_key = None  # (effective_score, -"title") maximized
        best_payload = None

        for idx, (song, base, reasons) in enumerate(remaining):
            artist = _field(song, "artist")
            genre = _field(song, "genre")
            a_count = sum(1 for s, _, _ in selected if _field(s, "artist") == artist)
            g_count = sum(1 for s, _, _ in selected if _field(s, "genre") == genre)
            effective = base - artist_penalty * a_count - genre_penalty * g_count

            title = _field(song, "title")
            # Maximize effective score; break ties by title ascending. We compare
            # (effective, title) where a lower title should win a tie, so track
            # the best explicitly instead of relying on tuple ordering of str.
            if (
                best_key is None
                or effective > best_key[0]
                or (effective == best_key[0] and title < best_key[1])
            ):
                note_bits = []
                if a_count:
                    note_bits.append(f"artist '{artist}' already listed x{a_count} (-{artist_penalty * a_count:.2f})")
                if g_count:
                    note_bits.append(f"genre '{genre}' already listed x{g_count} (-{genre_penalty * g_count:.2f})")
                new_reasons = reasons + ([f"Diversity penalty: {'; '.join(note_bits)}"] if note_bits else [])
                best_key = (effective, title)
                best_idx = idx
                best_payload = (song, round(effective, 2), new_reasons)

        selected.append(best_payload)
        remaining.pop(best_idx)

    return selected


def recommend_songs(
    user_prefs: Dict, songs: List[Dict], k: int = 5, strategy: ScoringStrategy = BALANCED,
    diversity: bool = False,
) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.

    Pythonic form: a single list comprehension judges every song with
    score_song(), producing (song, score, explanation) triples -- then
    sorted() (not .sort()) ranks that list as a pure expression, rather
    than mutating a list built up with .append() in a for-loop.

    `strategy` selects the ranking mode (Strategy pattern). Defaults to
    BALANCED, which reproduces the original recipe exactly.

    `diversity` (default False) applies a greedy diversity/fairness penalty via
    _select_diverse(): as the top-k is built, songs are penalized for sharing an
    artist or genre with picks already in the list, so no single artist or genre
    dominates the results. Defaults off so baseline output is unchanged.

    Tie-break: score descending, then title ascending, so ties resolve
    the same way regardless of catalog order.

    Required by src/main.py
    """
    scored = [
        (song, score, reasons)
        for song, (score, reasons) in ((s, score_song(user_prefs, s, strategy)) for s in songs)
    ]
    if diversity:
        chosen = _select_diverse(scored, k)
        return [(song, score, _explain(reasons)) for song, score, reasons in chosen]
    ranked = sorted(scored, key=lambda item: (-item[1], _field(item[0], "title")))
    return [(song, score, _explain(reasons)) for song, score, reasons in ranked[:k]]


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
