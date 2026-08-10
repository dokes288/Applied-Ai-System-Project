"""Tests for the RAG pipeline and reliability harness (offline/deterministic)."""

from src.recommender import load_songs
from src.rag import (
    offline_parse,
    offline_generate,
    grounding_check,
    recommend_rag,
    load_artist_notes,
    VibeQuery,
)
from src.reliability import run_reliability, THRESHOLDS
from src.evaluate import run_eval, confidence, PASS_RATE_THRESHOLD


def _catalog():
    return load_songs("data/songs.csv")


# --- Parsing ------------------------------------------------------------------

def test_offline_parse_extracts_core_fields():
    q = offline_parse("chill lofi to study to, acoustic and mellow")
    assert q.genre == "lofi"
    assert q.mood == "chill"
    assert q.likes_acoustic is True
    assert q.energy < 0.5  # "study"/"mellow"/"chill" -> low energy


def test_offline_parse_handles_advanced_cues():
    q = offline_parse("nostalgic 80s synthwave, nothing explicit, no lyrics")
    assert q.genre == "synthwave"
    assert q.preferred_decade == 1980
    assert q.allow_explicit is False
    assert q.preferred_language == "instrumental"
    assert "nostalgic" in (q.desired_mood_tags or [])


def test_offline_parse_is_deterministic():
    text = "intense rock for a workout, popular hits"
    assert offline_parse(text).to_prefs() == offline_parse(text).to_prefs()


def test_indie_pop_matches_before_pop():
    # The longer genre token must win so "indie pop" is not captured as "pop".
    assert offline_parse("something indie pop and happy").genre == "indie pop"


def test_word_boundary_pop_not_matched_inside_popular():
    # Regression: "popular" must NOT be parsed as genre "pop"; "rock" should win,
    # and "popular" should still set the popularity preference.
    q = offline_parse("intense rock for a workout, popular hits")
    assert q.genre == "rock"
    assert q.prefers_popular is True


# --- Retrieval + end-to-end ---------------------------------------------------

def test_rag_offline_pipeline_retrieves_and_answers():
    songs = _catalog()
    result = recommend_rag("upbeat happy pop, something popular", songs, k=5, use_llm="offline")
    assert result.parse_engine == "offline"
    assert result.generate_engine == "offline"
    assert len(result.retrieved) == 5
    assert result.retrieved[0][0]["genre"] == "pop"   # top pick honors the parsed genre
    assert result.answer.strip() != ""
    # The answer must name the top retrieved song (grounded generation).
    assert result.retrieved[0][0]["title"].lower() in result.answer.lower()


def test_rag_end_to_end_deterministic():
    songs = _catalog()
    a = recommend_rag("nostalgic 80s synthwave", songs, use_llm="offline")
    b = recommend_rag("nostalgic 80s synthwave", songs, use_llm="offline")
    assert a.answer == b.answer
    assert [s["title"] for s, _, _ in a.retrieved] == [s["title"] for s, _, _ in b.retrieved]


# --- Grounding guard ----------------------------------------------------------

def test_grounding_check_passes_for_offline_answer():
    songs = _catalog()
    result = recommend_rag("chill lofi", songs, use_llm="offline")
    grounded, hallucinated = grounding_check(result.answer, result.retrieved, songs)
    assert grounded is True
    assert hallucinated == []


def test_grounding_check_flags_a_hallucinated_song():
    songs = _catalog()
    # Retrieve a set that does NOT include "Storm Runner", then craft an answer
    # that names it -> the guard must flag it as hallucinated.
    retrieved = recommend_songs_stub = [
        (s, 5.0, "reason") for s in songs if s["title"] == "Library Rain"
    ]
    answer = "You should listen to Storm Runner, it is great."
    grounded, hallucinated = grounding_check(answer, retrieved, songs)
    assert grounded is False
    assert "Storm Runner" in hallucinated


def test_offline_generate_only_uses_retrieved_songs():
    songs = _catalog()
    retrieved = [(s, 6.0, "Genre match: lofi (+2.0)") for s in songs if s["title"] == "Library Rain"]
    answer = offline_generate("chill lofi", retrieved)
    grounded, _ = grounding_check(answer, retrieved, songs)
    assert grounded is True


# --- Reliability harness ------------------------------------------------------

def test_reliability_meets_thresholds():
    songs = _catalog()
    report = run_reliability(songs)
    for metric, threshold in THRESHOLDS.items():
        assert report["metrics"][metric] >= threshold, (
            f"{metric} = {report['metrics'][metric]:.2f} < {threshold}"
        )


# --- Stretch: RAG enhancement (artist notes as a 2nd source) ------------------

def test_artist_notes_load():
    notes = load_artist_notes()
    assert "neon echo" in notes and "loroom" in notes
    assert "synth" in notes["neon echo"].lower()


def test_artist_notes_contain_no_catalog_song_titles():
    # Safety: notes must never mention a song title, or the grounding guard would
    # flag a legitimate note as a hallucination.
    songs = _catalog()
    titles = [s["title"].lower() for s in songs]
    blob = " ".join(load_artist_notes().values()).lower()
    for title in titles:
        assert title not in blob, f"artist note leaks song title: {title}"


def test_notes_enhancement_enriches_answer_and_stays_grounded():
    songs = _catalog()
    plain = recommend_rag("chill lofi for studying", songs, use_llm="offline", use_notes=False)
    enriched = recommend_rag("chill lofi for studying", songs, use_llm="offline", use_notes=True)
    # Same retrieval, but the enriched answer is longer and cites artist context.
    assert [s["title"] for s, _, _ in plain.retrieved] == [s["title"] for s, _, _ in enriched.retrieved]
    assert enriched.notes_used  # at least one artist note was retrieved
    assert len(enriched.answer) > len(plain.answer)
    assert "About" in enriched.answer
    # Grounding still holds with notes on.
    grounded, hallucinated = grounding_check(enriched.answer, enriched.retrieved, songs)
    assert grounded is True and hallucinated == []


# --- Stretch: evaluation harness ----------------------------------------------

def test_confidence_is_bounded():
    songs = _catalog()
    r = recommend_rag("intense rock, popular hits", songs, use_llm="offline")
    c = confidence(r.retrieved)
    assert 0.0 <= c <= 1.0
    assert confidence([]) == 0.0


def test_evaluation_harness_passes():
    songs = _catalog()
    report = run_eval(songs)
    assert report["pass_rate"] >= PASS_RATE_THRESHOLD
    assert 0.0 <= report["avg_confidence"] <= 1.0
    for row in report["rows"]:
        assert 0.0 <= row["confidence"] <= 1.0
        assert row["result"] in ("PASS", "FAIL")
