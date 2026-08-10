"""
Recommender module — core recommendation logic.

Scoring uses a weighted combination of:
  - genre match          (weight 3)
  - mood match           (weight 2)
  - tag overlap          (weight 1 each)
  - BPM proximity        (weight 1, scaled)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .retrieval import MusicCatalog, Song


def _bpm_score(target_bpm: int, song_bpm: int) -> float:
    """Return a 0–1 BPM similarity score (1 = identical, 0 = ≥60 BPM apart)."""
    diff = abs(target_bpm - song_bpm)
    return max(0.0, 1.0 - diff / 60.0)


def score_song(
    song: Song,
    genre: Optional[str] = None,
    mood: Optional[str] = None,
    tags: Optional[List[str]] = None,
    bpm: Optional[int] = None,
) -> float:
    """
    Compute a relevance score for *song* given the user preferences.

    Higher is more relevant.
    """
    score = 0.0

    if genre and song.genre.lower() == genre.lower():
        score += 3.0

    if mood and song.mood.lower() == mood.lower():
        score += 2.0

    if tags:
        tags_lower = [t.lower() for t in tags]
        overlap = sum(1 for t in song.tags if t.lower() in tags_lower)
        score += float(overlap)

    if bpm is not None:
        score += _bpm_score(bpm, song.bpm)

    return score


class MusicRecommender:
    """
    Ranks catalog songs according to user preferences and returns
    the top-N recommendations.

    Parameters
    ----------
    catalog:
        A :class:`MusicCatalog` instance.  A default catalog is created if
        none is provided.
    """

    def __init__(self, catalog: Optional[MusicCatalog] = None) -> None:
        self.catalog = catalog or MusicCatalog()

    def recommend(
        self,
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        tags: Optional[List[str]] = None,
        bpm: Optional[int] = None,
        top_n: int = 5,
        query: Optional[str] = None,
    ) -> List[Dict]:
        """
        Return the top-N recommended songs as a list of dicts.

        Parameters
        ----------
        genre:
            Preferred genre (e.g. ``"pop"``, ``"rock"``).
        mood:
            Preferred mood (e.g. ``"happy"``, ``"calm"``).
        tags:
            Preferred tags (e.g. ``["80s", "synthwave"]``).
        bpm:
            Target beats-per-minute.
        top_n:
            Maximum number of results.
        query:
            Optional free-text search that pre-filters the catalog before
            scoring.
        """
        if top_n < 1:
            raise ValueError("top_n must be at least 1")

        # Pre-filter by keyword query when provided
        if query:
            pool = self.catalog.search(query)
            if not pool:
                # Fall back to full catalog when query yields nothing
                pool = self.catalog.all_songs()
        else:
            pool = self.catalog.all_songs()

        scored = [
            (
                song,
                score_song(song, genre=genre, mood=mood, tags=tags, bpm=bpm),
            )
            for song in pool
        ]

        # Sort by score descending, then title for deterministic tie-breaking
        scored.sort(key=lambda x: (-x[1], x[0].title))

        top = scored[:top_n]
        return [
            {
                "title": s.title,
                "artist": s.artist,
                "genre": s.genre,
                "mood": s.mood,
                "bpm": s.bpm,
                "tags": s.tags,
                "score": round(sc, 3),
            }
            for s, sc in top
        ]

    def explain(
        self,
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        tags: Optional[List[str]] = None,
        bpm: Optional[int] = None,
        top_n: int = 5,
    ) -> str:
        """
        Return a human-readable explanation of the recommendation decision.
        """
        results = self.recommend(genre=genre, mood=mood, tags=tags, bpm=bpm, top_n=top_n)
        lines = [
            "=== Music Recommender — Decision Explanation ===",
            f"  Requested genre : {genre or 'any'}",
            f"  Requested mood  : {mood or 'any'}",
            f"  Requested tags  : {tags or []}",
            f"  Target BPM      : {bpm or 'any'}",
            "",
            "Scoring formula (weights):",
            "  genre match  → +3.0",
            "  mood match   → +2.0",
            "  tag overlap  → +1.0 per tag",
            "  BPM diff     → +0.0–1.0 (1 = exact, 0 at ≥60 BPM apart)",
            "",
            f"Top {top_n} recommendations:",
        ]
        for i, r in enumerate(results, 1):
            lines.append(
                f"  {i}. {r['title']} — {r['artist']}"
                f" [{r['genre']}, {r['mood']}, {r['bpm']} BPM]"
                f"  score={r['score']}"
            )
        return "\n".join(lines)
