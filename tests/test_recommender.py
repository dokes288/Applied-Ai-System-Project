"""
Structured tests for the Music Recommender Simulation System.

Covers:
  - MusicCatalog retrieval
  - MusicRecommender scoring and ranking
  - Guardrails validation
  - AgenticPlanner plan-and-execute loop (happy path + edge cases)
"""

import pytest

from music_recommender.retrieval import MusicCatalog, Song
from music_recommender.recommender import MusicRecommender, score_song
from music_recommender.guardrails import Guardrails
from music_recommender.planner import AgenticPlanner


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture()
def small_catalog():
    """Three-song catalog for deterministic testing."""
    return MusicCatalog(
        songs=[
            Song("Alpha", "Artist A", "pop", 120, "happy", ["dance"]),
            Song("Beta",  "Artist B", "rock", 90,  "calm",  ["guitar"]),
            Song("Gamma", "Artist C", "pop", 150, "energetic", ["80s", "dance"]),
        ]
    )


@pytest.fixture()
def recommender(small_catalog):
    return MusicRecommender(small_catalog)


@pytest.fixture()
def planner(small_catalog):
    return AgenticPlanner(small_catalog, min_results=2)


# ──────────────────────────────────────────────────────────────────────
# MusicCatalog
# ──────────────────────────────────────────────────────────────────────

class TestMusicCatalog:
    def test_len_default_catalog(self):
        catalog = MusicCatalog()
        assert len(catalog) > 0

    def test_by_genre_returns_correct_songs(self, small_catalog):
        results = small_catalog.by_genre("pop")
        assert len(results) == 2
        assert all(s.genre == "pop" for s in results)

    def test_by_genre_case_insensitive(self, small_catalog):
        assert small_catalog.by_genre("POP") == small_catalog.by_genre("pop")

    def test_by_mood(self, small_catalog):
        results = small_catalog.by_mood("calm")
        assert len(results) == 1
        assert results[0].title == "Beta"

    def test_search_by_artist(self, small_catalog):
        results = small_catalog.search("Artist A")
        assert any(s.title == "Alpha" for s in results)

    def test_search_by_tag(self, small_catalog):
        results = small_catalog.search("80s")
        assert any(s.title == "Gamma" for s in results)

    def test_search_empty_query_returns_empty(self, small_catalog):
        assert small_catalog.search("") == []

    def test_genres_unique(self, small_catalog):
        genres = small_catalog.genres()
        assert len(genres) == len(set(genres))

    def test_add_song(self, small_catalog):
        before = len(small_catalog)
        small_catalog.add_song(Song("Delta", "Artist D", "jazz", 70, "calm", []))
        assert len(small_catalog) == before + 1


# ──────────────────────────────────────────────────────────────────────
# score_song
# ──────────────────────────────────────────────────────────────────────

class TestScoreSong:
    def test_genre_match_adds_three(self):
        song = Song("X", "Y", "pop", 100, "happy", [])
        s = score_song(song, genre="pop")
        assert s == 3.0

    def test_mood_match_adds_two(self):
        song = Song("X", "Y", "pop", 100, "happy", [])
        s = score_song(song, mood="happy")
        assert s == 2.0

    def test_tag_overlap_adds_per_tag(self):
        song = Song("X", "Y", "pop", 100, "happy", ["dance", "80s"])
        s = score_song(song, tags=["dance", "80s", "missing"])
        assert s == 2.0

    def test_bpm_exact_match_adds_one(self):
        song = Song("X", "Y", "pop", 100, "happy", [])
        s = score_song(song, bpm=100)
        assert s == pytest.approx(1.0)

    def test_bpm_far_apart_adds_zero(self):
        song = Song("X", "Y", "pop", 200, "happy", [])
        s = score_song(song, bpm=100)
        assert s == pytest.approx(0.0)

    def test_no_criteria_scores_zero(self):
        song = Song("X", "Y", "pop", 100, "happy", [])
        assert score_song(song) == 0.0


# ──────────────────────────────────────────────────────────────────────
# MusicRecommender
# ──────────────────────────────────────────────────────────────────────

class TestMusicRecommender:
    def test_recommend_returns_at_most_top_n(self, recommender):
        results = recommender.recommend(top_n=2)
        assert len(results) <= 2

    def test_recommend_genre_filter_scores_pop_highest(self, recommender):
        results = recommender.recommend(genre="pop", top_n=3)
        assert results[0]["genre"] == "pop"

    def test_recommend_sorted_by_score_descending(self, recommender):
        results = recommender.recommend(genre="pop", mood="happy", top_n=3)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_returns_dict_keys(self, recommender):
        results = recommender.recommend(top_n=1)
        assert set(results[0].keys()) == {
            "title", "artist", "genre", "mood", "bpm", "tags", "score"
        }

    def test_recommend_top_n_one(self, recommender):
        results = recommender.recommend(top_n=1)
        assert len(results) == 1

    def test_recommend_invalid_top_n_raises(self, recommender):
        with pytest.raises(ValueError):
            recommender.recommend(top_n=0)

    def test_recommend_with_query(self, recommender):
        results = recommender.recommend(query="80s")
        assert any(r["title"] == "Gamma" for r in results)

    def test_explain_contains_formula(self, recommender):
        explanation = recommender.explain(genre="pop")
        assert "genre match" in explanation
        assert "mood match" in explanation

    def test_recommend_query_no_match_falls_back(self, recommender):
        # query that matches nothing → full catalog fallback
        results = recommender.recommend(query="zzz_no_match", top_n=3)
        assert len(results) > 0


# ──────────────────────────────────────────────────────────────────────
# Guardrails
# ──────────────────────────────────────────────────────────────────────

class TestGuardrails:
    def test_valid_input_passes(self):
        result = Guardrails.validate(genre="pop", mood="happy", bpm=120, top_n=5)
        assert result.valid is True
        assert result.errors == []

    def test_invalid_genre_fails(self):
        result = Guardrails.validate_genre("not_a_genre")
        assert not result.valid
        assert any("genre" in e.lower() or "Unknown" in e for e in result.errors)

    def test_invalid_mood_fails(self):
        result = Guardrails.validate_mood("furious")
        assert not result.valid

    def test_bpm_below_min_fails(self):
        result = Guardrails.validate_bpm(5)
        assert not result.valid

    def test_bpm_above_max_fails(self):
        result = Guardrails.validate_bpm(500)
        assert not result.valid

    def test_valid_bpm_passes(self):
        assert Guardrails.validate_bpm(120).valid is True

    def test_top_n_zero_fails(self):
        assert Guardrails.validate_top_n(0).valid is False

    def test_top_n_exceeds_max_fails(self):
        assert Guardrails.validate_top_n(100).valid is False

    def test_query_too_long_fails(self):
        long_query = "a" * 201
        assert Guardrails.validate_query(long_query).valid is False

    def test_tags_not_list_fails(self):
        result = Guardrails.validate_tags("dance")  # type: ignore[arg-type]
        assert not result.valid

    def test_too_many_tags_fails(self):
        result = Guardrails.validate_tags(["t"] * 11)
        assert not result.valid

    def test_aggregate_validation_collects_multiple_errors(self):
        result = Guardrails.validate(genre="bad_genre", mood="bad_mood")
        assert not result.valid
        assert len(result.errors) >= 2

    def test_bool_conversion_true(self):
        result = Guardrails.validate(genre="pop")
        assert bool(result) is True

    def test_bool_conversion_false(self):
        result = Guardrails.validate(genre="invalid_genre")
        assert bool(result) is False


# ──────────────────────────────────────────────────────────────────────
# AgenticPlanner
# ──────────────────────────────────────────────────────────────────────

class TestAgenticPlanner:
    def test_happy_path_returns_success(self, planner):
        outcome = planner.run(genre="pop", mood="happy", top_n=3)
        assert outcome["success"] is True
        assert len(outcome["results"]) > 0

    def test_trace_has_steps(self, planner):
        outcome = planner.run(genre="pop", top_n=2)
        assert any("Step 1" in t for t in outcome["trace"])
        assert any("Step 5" in t for t in outcome["trace"])

    def test_invalid_genre_returns_failure(self, planner):
        outcome = planner.run(genre="invalid_genre")
        assert outcome["success"] is False
        assert len(outcome["errors"]) > 0
        assert outcome["results"] == []

    def test_results_sorted_by_score(self, planner):
        outcome = planner.run(genre="pop", top_n=3)
        scores = [r["score"] for r in outcome["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_explain_run_returns_string(self, planner):
        text = planner.explain_run(genre="pop", top_n=2)
        assert "Agentic Planner" in text
        assert "Recommendations" in text

    def test_explain_run_failure_shows_errors(self, planner):
        text = planner.explain_run(genre="bad_genre")
        assert "Errors" in text

    def test_query_filters_candidates(self, planner):
        outcome = planner.run(query="80s", top_n=2)
        assert outcome["success"] is True
        assert any(r["title"] == "Gamma" for r in outcome["results"])

    def test_no_args_returns_success(self, planner):
        outcome = planner.run()
        assert outcome["success"] is True

    def test_retry_on_low_results(self):
        # Only pop songs with mood "happy"; requesting genre="classical" (no match)
        # and mood="energetic" (no match) → 0 positive-score results → retry triggered.
        catalog = MusicCatalog(songs=[
            Song("Pop1", "P", "pop", 100, "happy", []),
            Song("Pop2", "Q", "pop", 110, "happy", []),
        ])
        # min_results=3 means any fewer positive results triggers the retry
        planner = AgenticPlanner(catalog, min_results=3)
        outcome = planner.run(genre="classical", mood="energetic", top_n=3)
        assert outcome["success"] is True
        retry_steps = [t for t in outcome["trace"] if "Relaxing" in t]
        assert len(retry_steps) >= 1
