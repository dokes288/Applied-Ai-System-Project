"""
Guardrails module — validates user inputs and enforces content policies
before passing them to the recommender.

Decision-making rationale
--------------------------
1. **Input validation** — BPM must be in a realistic musical range (20–300).
2. **Allowlist** — Only known genres and moods are accepted to prevent
   garbage queries from producing misleading results.
3. **Query sanitisation** — Overly long or blank queries are rejected.
4. **Transparency** — Every rejection includes a human-readable reason so
   the caller can fix the request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

ALLOWED_GENRES = {
    "pop", "rock", "hip-hop", "classical", "jazz", "soul",
    "country", "electronic", "r&b", "metal", "folk", "reggae",
}

ALLOWED_MOODS = {
    "happy", "sad", "energetic", "calm", "dramatic", "romantic",
    "angry", "melancholic", "uplifting",
}

_MAX_QUERY_LEN = 200
_BPM_MIN = 20
_BPM_MAX = 300
_MAX_TOP_N = 50


@dataclass
class ValidationResult:
    """Holds the outcome of a guardrail validation pass."""

    valid: bool
    errors: List[str]

    def __bool__(self) -> bool:
        return self.valid


class Guardrails:
    """
    Static validation helpers for recommendation requests.

    All methods are pure (no side effects) and return a
    :class:`ValidationResult`.
    """

    @staticmethod
    def validate(
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        bpm: Optional[int] = None,
        top_n: int = 5,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> ValidationResult:
        """
        Validate all recommendation parameters at once.

        Returns
        -------
        ValidationResult
            ``.valid`` is ``True`` only when *all* checks pass.
        """
        errors: List[str] = []

        if genre is not None:
            genre_errors = Guardrails.validate_genre(genre).errors
            errors.extend(genre_errors)

        if mood is not None:
            mood_errors = Guardrails.validate_mood(mood).errors
            errors.extend(mood_errors)

        if bpm is not None:
            bpm_errors = Guardrails.validate_bpm(bpm).errors
            errors.extend(bpm_errors)

        if top_n is not None:
            top_n_errors = Guardrails.validate_top_n(top_n).errors
            errors.extend(top_n_errors)

        if query is not None:
            query_errors = Guardrails.validate_query(query).errors
            errors.extend(query_errors)

        if tags is not None:
            tags_errors = Guardrails.validate_tags(tags).errors
            errors.extend(tags_errors)

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    @staticmethod
    def validate_genre(genre: str) -> ValidationResult:
        errors = []
        if not isinstance(genre, str) or not genre.strip():
            errors.append("genre must be a non-empty string.")
        elif genre.lower() not in ALLOWED_GENRES:
            errors.append(
                f"Unknown genre '{genre}'. Allowed: {sorted(ALLOWED_GENRES)}."
            )
        return ValidationResult(valid=len(errors) == 0, errors=errors)

    @staticmethod
    def validate_mood(mood: str) -> ValidationResult:
        errors = []
        if not isinstance(mood, str) or not mood.strip():
            errors.append("mood must be a non-empty string.")
        elif mood.lower() not in ALLOWED_MOODS:
            errors.append(
                f"Unknown mood '{mood}'. Allowed: {sorted(ALLOWED_MOODS)}."
            )
        return ValidationResult(valid=len(errors) == 0, errors=errors)

    @staticmethod
    def validate_bpm(bpm: int) -> ValidationResult:
        errors = []
        if not isinstance(bpm, int):
            errors.append("bpm must be an integer.")
        elif not (_BPM_MIN <= bpm <= _BPM_MAX):
            errors.append(
                f"BPM {bpm} out of realistic range [{_BPM_MIN}–{_BPM_MAX}]."
            )
        return ValidationResult(valid=len(errors) == 0, errors=errors)

    @staticmethod
    def validate_top_n(top_n: int) -> ValidationResult:
        errors = []
        if not isinstance(top_n, int) or top_n < 1:
            errors.append("top_n must be a positive integer.")
        elif top_n > _MAX_TOP_N:
            errors.append(f"top_n exceeds maximum allowed value of {_MAX_TOP_N}.")
        return ValidationResult(valid=len(errors) == 0, errors=errors)

    @staticmethod
    def validate_query(query: str) -> ValidationResult:
        errors = []
        if not isinstance(query, str):
            errors.append("query must be a string.")
        elif len(query) > _MAX_QUERY_LEN:
            errors.append(
                f"query is too long ({len(query)} chars). Maximum: {_MAX_QUERY_LEN}."
            )
        return ValidationResult(valid=len(errors) == 0, errors=errors)

    @staticmethod
    def validate_tags(tags: List[str]) -> ValidationResult:
        errors = []
        if not isinstance(tags, list):
            errors.append("tags must be a list of strings.")
        elif not all(isinstance(t, str) for t in tags):
            errors.append("Every tag must be a string.")
        elif len(tags) > 10:
            errors.append("Too many tags (max 10).")
        return ValidationResult(valid=len(errors) == 0, errors=errors)
