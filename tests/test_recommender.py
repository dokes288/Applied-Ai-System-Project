from src.recommender import (
    Song,
    UserProfile,
    Recommender,
    score_song,
    recommend_songs,
    BALANCED,
    GENRE_FIRST,
    MOOD_FIRST,
    ENERGY_FOCUSED,
    STRATEGIES,
)

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_acoustic_preference_boosts_acoustic_tracks():
    user = UserProfile(
        favorite_genre="lofi",
        favorite_mood="chill",
        target_energy=0.4,
        likes_acoustic=True,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=1)

    assert results[0].title == "Chill Lofi Loop"
    assert results[0].acousticness >= 0.7


# --- Adversarial / edge-case regressions -------------------------------------

def _pop_song(**overrides) -> dict:
    base = {"genre": "pop", "mood": "happy", "energy": 0.8, "acousticness": 0.2, "popularity": 90.0}
    base.update(overrides)
    return base


def test_out_of_range_target_energy_never_goes_negative():
    # energy=2.0 (e.g. a 0-100 scale slipping through) must not drive the
    # energy term negative and cancel out the genre/mood gates.
    prefs = {"genre": "pop", "mood": "happy", "energy": 2.0, "likes_acoustic": False}
    score, _ = score_song(prefs, _pop_song(energy=0.82))
    # Full genre+mood (3.0) survives; target clamps to 1.0 so energy term is
    # 2.0*(1-|0.82-1.0|) = 1.64, plus acoustic fit -- comfortably above 3.0.
    assert score > 3.0


def test_energy_term_is_floored_at_zero():
    # Maximal energy mismatch contributes 0, never a negative number.
    prefs = {"genre": "x", "mood": "x", "energy": 1.0, "likes_acoustic": False}
    score, _ = score_song(prefs, {"genre": "y", "mood": "y", "energy": 0.0, "acousticness": 0.0})
    # No categorical match, energy fully off (floored to 0), acoustic perfect (1.0).
    assert score == 1.0


def test_prefers_popular_true_boosts_popular_track():
    indifferent, _ = score_song({"genre": "pop", "mood": "happy", "energy": 0.8}, _pop_song())
    wants_popular, reasons = score_song(
        {"genre": "pop", "mood": "happy", "energy": 0.8, "prefers_popular": True}, _pop_song()
    )
    assert wants_popular == round(indifferent + 1.0, 2)
    assert any("Popularity match" in r for r in reasons)


def test_prefers_niche_boosts_obscure_track():
    _, reasons = score_song(
        {"genre": "pop", "mood": "happy", "energy": 0.8, "prefers_popular": False},
        _pop_song(popularity=20.0),
    )
    assert any("Niche pick" in r for r in reasons)


def test_prefers_popular_none_is_a_true_noop():
    # The default (indifferent) must not change scores -- protects the
    # documented sample output in the README.
    with_field, _ = score_song(
        {"genre": "pop", "mood": "happy", "energy": 0.8, "prefers_popular": None}, _pop_song()
    )
    without_field, _ = score_song({"genre": "pop", "mood": "happy", "energy": 0.8}, _pop_song())
    assert with_field == without_field


# --- Advanced feature regressions --------------------------------------------

def _adv_song(**overrides) -> dict:
    base = {
        "genre": "synthwave", "mood": "moody", "energy": 0.75, "acousticness": 0.2,
        "popularity": 60.0, "release_decade": 1980, "mood_tags": ["nostalgic", "dreamy"],
        "language": "english", "instrumentalness": 0.4, "explicit": False,
    }
    base.update(overrides)
    return base


def test_advanced_features_are_all_opt_in():
    # A profile that supplies none of the advanced keys must score exactly the
    # same as before the features existed (protects README/model_card output).
    base = {"genre": "synthwave", "mood": "moody", "energy": 0.75, "likes_acoustic": False}
    plain, _ = score_song(base, _adv_song())
    # Advanced fields present on the SONG but no matching prefs -> still baseline.
    assert plain == round(2.0 + 1.0 + 2.0 + 0.8, 2)  # genre+mood+energy+acoustic


def test_mood_tags_give_partial_credit_by_overlap():
    one, _ = score_song({"desired_mood_tags": ["nostalgic"]}, _adv_song())
    both, reasons = score_song({"desired_mood_tags": ["nostalgic", "dreamy"]}, _adv_song())
    # Song has both tags: 1/1 -> +1.0; 2/2 -> +1.0; a 1-of-2 miss would be +0.5.
    half, _ = score_song({"desired_mood_tags": ["nostalgic", "aggressive"]}, _adv_song())
    assert both - one == 0.0          # song already had both, so full credit either way
    assert round(both - half, 2) == 0.5  # aggressive is absent -> only half the tags match
    assert any("Mood tags" in r for r in reasons)


def test_decade_exact_beats_adjacent_beats_far():
    exact, _ = score_song({"preferred_decade": 1980}, _adv_song())
    adjacent, _ = score_song({"preferred_decade": 1990}, _adv_song())
    far, _ = score_song({"preferred_decade": 2020}, _adv_song())
    assert exact > adjacent > far
    assert round(exact - adjacent, 2) == 0.5   # 1.0 exact vs 0.5 adjacent
    assert round(adjacent - far, 2) == 0.5     # 0.5 adjacent vs 0.0 far


def test_language_exact_match_bonus():
    _, reasons = score_song({"preferred_language": "English"}, _adv_song())  # case-insensitive
    assert any("Language match" in r for r in reasons)
    no_match, _ = score_song({"preferred_language": "spanish"}, _adv_song())
    match, _ = score_song({"preferred_language": "english"}, _adv_song())
    assert round(match - no_match, 2) == 1.0


def test_instrumentalness_is_continuous():
    close, _ = score_song({"target_instrumentalness": 0.4}, _adv_song(instrumentalness=0.4))
    far, _ = score_song({"target_instrumentalness": 0.4}, _adv_song(instrumentalness=0.9))
    assert close > far  # closer instrumentalness scores higher


def test_explicit_penalty_only_when_opted_out():
    clean_pref, _ = score_song({}, _adv_song(explicit=True))          # no allow_explicit key
    opted_out, reasons = score_song({"allow_explicit": False}, _adv_song(explicit=True))
    assert round(clean_pref - opted_out, 2) == 2.0
    assert any("Explicit content" in r for r in reasons)
    # Non-explicit song is never penalized even when opting out.
    not_penalized, _ = score_song({"allow_explicit": False}, _adv_song(explicit=False))
    baseline, _ = score_song({}, _adv_song(explicit=False))
    assert not_penalized == baseline


# --- Scoring strategies (Strategy pattern) -----------------------------------

def test_balanced_strategy_matches_default():
    # BALANCED must reproduce the no-strategy default exactly, so existing
    # behavior and documented output are unchanged.
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}
    default_score, _ = score_song(prefs, _pop_song())
    balanced_score, _ = score_song(prefs, _pop_song(), BALANCED)
    assert default_score == balanced_score


def test_genre_first_weights_genre_more_than_mood():
    # Same song matched only on genre scores higher under Genre-First than a
    # song matched only on mood; Mood-First flips that.
    genre_only = {"genre": "pop", "mood": "zzz", "energy": 0.5}
    mood_only = {"genre": "zzz", "mood": "happy", "energy": 0.5}
    song = _pop_song(energy=0.5)  # genre=pop, mood=happy
    g_score, _ = score_song(genre_only, song, GENRE_FIRST)
    m_score, _ = score_song(mood_only, song, GENRE_FIRST)
    assert g_score > m_score  # genre weighted 4.0 vs mood 1.0
    # Under Mood-First the ranking of the two match types reverses.
    g2, _ = score_song(genre_only, song, MOOD_FIRST)
    m2, _ = score_song(mood_only, song, MOOD_FIRST)
    assert m2 > g2


def test_strategy_changes_ranking_order():
    # A concrete order flip for a pop/happy listener between two songs:
    #   Gym Hero      = pop genre  but "intense" mood  (matches genre only)
    #   Rooftop Lights = "happy" mood but indie-pop genre (matches mood only)
    # Genre-First should rank Gym Hero first; Mood-First should flip to Rooftop.
    pop_fan = UserProfile(favorite_genre="pop", favorite_mood="happy",
                          target_energy=0.8, likes_acoustic=False)
    songs = [
        Song(1, "Gym Hero", "Max Pulse", "pop", "intense", 0.93, 132, 0.77, 0.88, 0.05),
        Song(2, "Rooftop Lights", "Indigo Parade", "indie pop", "happy", 0.76, 124, 0.81, 0.82, 0.35),
    ]
    genre_first = [s.title for s in Recommender(songs, GENRE_FIRST).recommend(pop_fan, k=2)]
    mood_first = [s.title for s in Recommender(songs, MOOD_FIRST).recommend(pop_fan, k=2)]
    assert genre_first[0] == "Gym Hero"        # genre weighted heavily
    assert mood_first[0] == "Rooftop Lights"   # mood weighted heavily
    assert genre_first != mood_first


def test_strategy_registry_is_complete():
    assert set(STRATEGIES) == {"balanced", "genre-first", "mood-first", "energy-focused"}
    assert STRATEGIES["energy-focused"] is ENERGY_FOCUSED


# --- Diversity / fairness penalty --------------------------------------------

def _catalog_dicts():
    # Two strong picks by the SAME artist and genre, plus two others.
    return [
        {"title": "A1", "artist": "Dup", "genre": "pop", "mood": "happy", "energy": 0.8, "acousticness": 0.10},
        {"title": "A2", "artist": "Dup", "genre": "pop", "mood": "happy", "energy": 0.8, "acousticness": 0.12},
        {"title": "B1", "artist": "Other", "genre": "rock", "mood": "happy", "energy": 0.8, "acousticness": 0.10},
        {"title": "C1", "artist": "Third", "genre": "jazz", "mood": "happy", "energy": 0.8, "acousticness": 0.10},
    ]


def test_diversity_off_is_unchanged():
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}
    songs = _catalog_dicts()
    default = [t for t, _, _ in ((s["title"], sc, ex) for s, sc, ex in recommend_songs(prefs, songs, 4))]
    explicit_off = [s["title"] for s, _, _ in recommend_songs(prefs, songs, 4, diversity=False)]
    assert default == explicit_off
    # Both same-artist songs sit at the top when diversity is off.
    assert default[:2] == ["A1", "A2"]


def test_diversity_penalty_demotes_second_same_artist():
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}
    songs = _catalog_dicts()
    diverse = recommend_songs(prefs, songs, 4, diversity=True)
    titles = [s["title"] for s, _, _ in diverse]
    # A1 still wins, but A2 (same artist AND genre) is pushed below the other
    # artists instead of taking the #2 slot.
    assert titles[0] == "A1"
    assert diverse[1][0]["artist"] != "Dup"
    assert titles.index("A2") > titles.index("B1")
    # The demoted pick carries a diversity-penalty note in its explanation.
    a2_explanation = next(ex for s, _, ex in diverse if s["title"] == "A2")
    assert "Diversity penalty" in a2_explanation


def test_diversity_reduces_genre_domination_in_oop_path():
    # Real-catalog-shaped case: three lofi tracks (two by LoRoom) would sweep the
    # top 3; the penalty must break that up.
    user = UserProfile(favorite_genre="lofi", favorite_mood="chill",
                       target_energy=0.4, likes_acoustic=True)
    songs = [
        Song(1, "Library Rain", "Paper Lanterns", "lofi", "chill", 0.35, 72, 0.60, 0.58, 0.86),
        Song(2, "Midnight Coding", "LoRoom", "lofi", "chill", 0.42, 78, 0.56, 0.62, 0.71),
        Song(3, "Focus Flow", "LoRoom", "lofi", "focused", 0.40, 80, 0.59, 0.60, 0.78),
        Song(4, "Spacewalk Thoughts", "Orbit Bloom", "ambient", "chill", 0.28, 60, 0.65, 0.41, 0.92),
    ]
    rec = Recommender(songs)
    plain = [s.genre for s in rec.recommend(user, k=3)]
    diverse = [s.genre for s in rec.recommend(user, k=3, diversity=True)]
    assert plain.count("lofi") == 3           # un-penalized: all lofi
    assert diverse.count("lofi") < 3          # penalty forces in a non-lofi track
    assert "ambient" in diverse
