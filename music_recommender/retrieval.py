"""
Retrieval module — provides a simple in-memory music catalog with
keyword and genre-based lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Song:
    """Represents a single song in the catalog."""

    title: str
    artist: str
    genre: str
    bpm: int
    mood: str  # e.g. "happy", "sad", "energetic", "calm"
    tags: List[str] = field(default_factory=list)

    def matches_query(self, query: str) -> bool:
        """Return True if the query string appears in any song field."""
        query_lower = query.lower()
        searchable = " ".join(
            [self.title, self.artist, self.genre, self.mood] + self.tags
        ).lower()
        return query_lower in searchable


# ---------------------------------------------------------------------------
# Built-in demo catalog (no external API calls — fully offline)
# ---------------------------------------------------------------------------
_DEFAULT_CATALOG: List[Song] = [
    Song("Blinding Lights", "The Weeknd", "pop", 171, "energetic", ["80s", "synthwave"]),
    Song("Levitating", "Dua Lipa", "pop", 103, "happy", ["disco", "dance"]),
    Song("Shape of You", "Ed Sheeran", "pop", 96, "happy", ["tropical"]),
    Song("Bohemian Rhapsody", "Queen", "rock", 144, "dramatic", ["classic", "opera"]),
    Song("Hotel California", "Eagles", "rock", 75, "calm", ["classic", "guitar"]),
    Song("Smells Like Teen Spirit", "Nirvana", "rock", 117, "energetic", ["grunge"]),
    Song("God's Plan", "Drake", "hip-hop", 77, "calm", ["trap", "rnb"]),
    Song("HUMBLE.", "Kendrick Lamar", "hip-hop", 150, "energetic", ["conscious", "trap"]),
    Song("Lose Yourself", "Eminem", "hip-hop", 87, "energetic", ["motivational"]),
    Song("Clair de Lune", "Claude Debussy", "classical", 60, "calm", ["piano", "impressionist"]),
    Song("Symphony No. 5", "Beethoven", "classical", 108, "dramatic", ["orchestral"]),
    Song("Four Seasons", "Vivaldi", "classical", 132, "happy", ["baroque", "violin"]),
    Song("Bad Guy", "Billie Eilish", "pop", 135, "energetic", ["alternative", "dark-pop"]),
    Song("Starboy", "The Weeknd", "pop", 186, "energetic", ["r&b", "synthpop"]),
    Song("Redbone", "Childish Gambino", "soul", 96, "calm", ["psychedelic", "funk"]),
    Song("Superstition", "Stevie Wonder", "soul", 100, "happy", ["funk", "classic"]),
    Song("Lose Control", "Teddy Swims", "soul", 92, "sad", ["rnb"]),
    Song("As It Was", "Harry Styles", "pop", 174, "sad", ["indie-pop"]),
    Song("Running Up That Hill", "Kate Bush", "rock", 121, "dramatic", ["80s", "art-rock"]),
    Song("Flowers", "Miley Cyrus", "pop", 118, "happy", ["empowerment"]),
]


class MusicCatalog:
    """
    In-memory music catalog with genre and keyword retrieval.

    Parameters
    ----------
    songs:
        Optional custom song list.  Defaults to the built-in demo catalog.
    """

    def __init__(self, songs: Optional[List[Song]] = None) -> None:
        self._songs: List[Song] = songs if songs is not None else list(_DEFAULT_CATALOG)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def all_songs(self) -> List[Song]:
        """Return the entire catalog."""
        return list(self._songs)

    def by_genre(self, genre: str) -> List[Song]:
        """Return songs that match *genre* (case-insensitive)."""
        genre_lower = genre.lower()
        return [s for s in self._songs if s.genre.lower() == genre_lower]

    def by_mood(self, mood: str) -> List[Song]:
        """Return songs that match *mood* (case-insensitive)."""
        mood_lower = mood.lower()
        return [s for s in self._songs if s.mood.lower() == mood_lower]

    def search(self, query: str) -> List[Song]:
        """Full-text search across title, artist, genre, mood, and tags."""
        if not query or not query.strip():
            return []
        return [s for s in self._songs if s.matches_query(query.strip())]

    def add_song(self, song: Song) -> None:
        """Add a song to the catalog at runtime."""
        self._songs.append(song)

    def genres(self) -> List[str]:
        """Return the unique genres present in the catalog."""
        seen: set = set()
        result = []
        for s in self._songs:
            if s.genre not in seen:
                seen.add(s.genre)
                result.append(s.genre)
        return result

    def __len__(self) -> int:
        return len(self._songs)
