from src.recommender import Song, UserProfile, Recommender, score_song

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
