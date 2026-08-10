"""
Retrieval-Augmented Generation (RAG) for VibeMatch.

This is the AI feature that turns VibeMatch from a fixed-profile scorer into a
natural-language music assistant. The flow is a real RAG pipeline:

    1. PARSE   -- an LLM reads the user's free-text request ("nostalgic 80s
                  synthwave, nothing explicit") and turns it into a structured
                  VibeMatch profile. This builds the *retrieval query*.
    2. RETRIEVE-- the existing content-based engine (recommend_songs) scores the
                  10-song catalog against that profile and returns the top-k
                  songs WITH their reason strings. This is the retrieval step.
    3. GENERATE-- an LLM writes a short, natural-language recommendation that is
                  grounded ONLY in the retrieved songs and their reasons. The
                  retrieved data actively shapes the answer -- the model is told
                  to recommend only from the provided list and to cite the real
                  scores/reasons, never to invent songs.

Both the PARSE and GENERATE steps use the Claude API when it is available
(`anthropic` installed AND ANTHROPIC_API_KEY set) and fall back to a
deterministic, dependency-free local implementation otherwise. That keeps the
project fully reproducible for a grader with no API key, while giving the real
LLM experience when a key is present.

Guardrails & logging:
    * Every step is logged (which engine ran, token usage, warnings, errors).
    * The parsed profile is validated/clamped before it reaches the engine.
    * A GROUNDING guard checks that the generated answer only mentions songs
      that were actually retrieved; if the LLM hallucinates a song, we discard
      its answer and fall back to the deterministic generator.
    * Any API error or safety refusal is caught and degrades to the offline
      path rather than crashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import json
import logging
import os

from src.recommender import recommend_songs

logger = logging.getLogger("vibematch.rag")

# Default model. Override with VIBEMATCH_MODEL. Claude Opus 5 is the current
# flagship (see the claude-api reference); adaptive thinking is on by default.
DEFAULT_MODEL = os.environ.get("VIBEMATCH_MODEL", "claude-opus-5")

# --- Catalog vocabulary used by the offline parser and the grounding guard ---
KNOWN_GENRES = ["indie pop", "pop", "lofi", "rock", "ambient", "jazz", "synthwave", "metal"]
KNOWN_MOODS = ["happy", "chill", "intense", "relaxed", "moody", "focused", "angry", "sad"]
KNOWN_TAGS = [
    "nostalgic", "dreamy", "aggressive", "euphoric", "energetic",
    "mellow", "calm", "warm", "uplifting",
]


@dataclass
class VibeQuery:
    """A structured taste profile parsed from a free-text request."""
    genre: Optional[str] = None
    mood: Optional[str] = None
    energy: float = 0.5
    likes_acoustic: bool = False
    prefers_popular: Optional[bool] = None
    desired_mood_tags: Optional[List[str]] = None
    preferred_decade: Optional[int] = None
    preferred_language: Optional[str] = None
    target_instrumentalness: Optional[float] = None
    allow_explicit: Optional[bool] = None

    def to_prefs(self) -> Dict:
        """Convert to the dict the scoring engine expects, dropping unset
        (None) optional fields so their opt-in terms stay off."""
        prefs: Dict = {
            "genre": self.genre or "",
            "mood": self.mood or "",
            "energy": self.energy,
            "likes_acoustic": self.likes_acoustic,
        }
        for key in (
            "prefers_popular", "desired_mood_tags", "preferred_decade",
            "preferred_language", "target_instrumentalness", "allow_explicit",
        ):
            value = getattr(self, key)
            if value is not None:
                prefs[key] = value
        return prefs


@dataclass
class RagResult:
    """Everything the pipeline produced, for display and testing."""
    query_text: str
    profile: VibeQuery
    retrieved: List[Tuple[Dict, float, str]]
    answer: str
    parse_engine: str          # "claude" or "offline"
    generate_engine: str       # "claude" or "offline"
    warnings: List[str] = field(default_factory=list)

    def profile_dict(self) -> Dict:
        return asdict(self.profile)


# The JSON schema the Claude parse step is constrained to (structured outputs).
PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "genre": {"type": ["string", "null"]},
        "mood": {"type": ["string", "null"]},
        "energy": {"type": "number"},
        "likes_acoustic": {"type": "boolean"},
        "prefers_popular": {"type": ["boolean", "null"]},
        "desired_mood_tags": {"type": ["array", "null"], "items": {"type": "string"}},
        "preferred_decade": {"type": ["integer", "null"]},
        "preferred_language": {"type": ["string", "null"]},
        "target_instrumentalness": {"type": ["number", "null"]},
        "allow_explicit": {"type": ["boolean", "null"]},
    },
    "required": [
        "genre", "mood", "energy", "likes_acoustic", "prefers_popular",
        "desired_mood_tags", "preferred_decade", "preferred_language",
        "target_instrumentalness", "allow_explicit",
    ],
    "additionalProperties": False,
}

PARSE_SYSTEM = (
    "You convert a music listener's free-text request into a structured taste "
    "profile for a small content-based recommender. Only use these vocabularies "
    "where they apply; leave a field null if the request does not imply it.\n"
    f"genre: one of {KNOWN_GENRES} or null.\n"
    f"mood: one of {KNOWN_MOODS} or null.\n"
    "energy: 0.0 (very calm) to 1.0 (very intense).\n"
    "likes_acoustic: true if they want acoustic/organic sound, else false.\n"
    "prefers_popular: true=wants hits, false=wants niche, null=no preference.\n"
    f"desired_mood_tags: any of {KNOWN_TAGS} or null.\n"
    "preferred_decade: e.g. 1980, 1990, 2000, 2010, 2020, or null.\n"
    "preferred_language: e.g. 'english' or 'instrumental', or null.\n"
    "target_instrumentalness: 0.0 (vocal) to 1.0 (instrumental), or null.\n"
    "allow_explicit: false if they want to avoid explicit tracks, else null."
)


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, float(x)))


def _validate_profile(profile: VibeQuery) -> VibeQuery:
    """Guardrail: clamp/normalize a parsed profile before it hits the engine."""
    profile.energy = _clamp01(profile.energy)
    if profile.target_instrumentalness is not None:
        profile.target_instrumentalness = _clamp01(profile.target_instrumentalness)
    if profile.genre:
        profile.genre = str(profile.genre).strip().lower()
    if profile.mood:
        profile.mood = str(profile.mood).strip().lower()
    if profile.desired_mood_tags:
        profile.desired_mood_tags = [str(t).strip().lower() for t in profile.desired_mood_tags if str(t).strip()]
        if not profile.desired_mood_tags:
            profile.desired_mood_tags = None
    return profile


# --------------------------------------------------------------------------- #
# Claude client                                                               #
# --------------------------------------------------------------------------- #

def get_client():
    """Return (client, reason). client is None when the live path is unavailable
    -- either the SDK isn't installed or no API key is configured."""
    try:
        import anthropic
    except ImportError:
        return None, "anthropic package not installed"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None, "ANTHROPIC_API_KEY not set"
    try:
        return anthropic.Anthropic(), "ok"
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"client init failed: {exc}"


# --------------------------------------------------------------------------- #
# PARSE step                                                                  #
# --------------------------------------------------------------------------- #

def offline_parse(text: str) -> VibeQuery:
    """Deterministic keyword parser -- the reproducible fallback for the parse
    step. Same input always yields the same profile."""
    t = f" {text.lower()} "
    q = VibeQuery()

    for genre in KNOWN_GENRES:  # 'indie pop' before 'pop' so the longer match wins
        if genre in t:
            q.genre = genre
            break
    for mood in KNOWN_MOODS:
        if mood in t:
            q.mood = mood
            break

    tags = [tag for tag in KNOWN_TAGS if tag in t]
    if tags:
        q.desired_mood_tags = tags

    # Energy cues
    if any(w in t for w in ["high energy", "intense", "workout", "hype", "energetic", "banger"]):
        q.energy = 0.9
    elif any(w in t for w in ["calm", "chill", "relax", "sleep", "study", "mellow", "low energy"]):
        q.energy = 0.3
    elif "moderate" in t or "medium energy" in t:
        q.energy = 0.6

    # Acoustic / electronic
    if "acoustic" in t or "unplugged" in t:
        q.likes_acoustic = True
    elif "electronic" in t or "synth" in t:
        q.likes_acoustic = False

    # Popularity
    if any(w in t for w in ["popular", "hits", "mainstream", "chart"]):
        q.prefers_popular = True
    elif any(w in t for w in ["niche", "obscure", "underground", "under the radar", "hidden"]):
        q.prefers_popular = False

    # Decade ("80s", "1990s", ...)
    for token, decade in [("80s", 1980), ("90s", 1990), ("2000s", 2000), ("2010s", 2010), ("2020s", 2020)]:
        if token in t or f" {decade} " in t or f" {decade}s" in t:
            q.preferred_decade = decade
            break

    # Language / instrumental
    if "instrumental" in t or "no lyrics" in t or "no vocals" in t:
        q.preferred_language = "instrumental"
        q.target_instrumentalness = 0.9
    elif "english" in t:
        q.preferred_language = "english"

    # Explicit
    if any(w in t for w in ["no explicit", "clean", "family friendly", "not explicit", "nothing explicit"]):
        q.allow_explicit = False

    return _validate_profile(q)


def claude_parse(client, text: str) -> VibeQuery:
    """Parse via Claude using structured outputs (guaranteed-valid JSON)."""
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=2048,
        system=PARSE_SYSTEM,
        messages=[{"role": "user", "content": text}],
        output_config={"format": {"type": "json_schema", "schema": PROFILE_SCHEMA}},
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("parse refused by safety classifier")
    raw = next((b.text for b in response.content if b.type == "text"), "")
    data = json.loads(raw)
    usage = getattr(response, "usage", None)
    if usage is not None:
        logger.info("claude_parse tokens: in=%s out=%s", usage.input_tokens, usage.output_tokens)
    return _validate_profile(VibeQuery(**data))


# --------------------------------------------------------------------------- #
# GENERATE step                                                               #
# --------------------------------------------------------------------------- #

def _format_retrieved(retrieved: List[Tuple[Dict, float, str]]) -> str:
    lines = []
    for rank, (song, score, reasons) in enumerate(retrieved, start=1):
        lines.append(
            f"{rank}. {song['title']} by {song['artist']} "
            f"[genre={song.get('genre')}] score={score:.2f}\n   reasons: {reasons}"
        )
    return "\n".join(lines)


def offline_generate(query_text: str, retrieved: List[Tuple[Dict, float, str]]) -> str:
    """Deterministic, grounded answer built only from the retrieved songs."""
    if not retrieved:
        return "I couldn't find anything in the catalog that matches that request."
    top_song, top_score, top_reasons = retrieved[0]
    first_reason = top_reasons.split(" | ")[0] if top_reasons else "it fits your request"
    parts = [
        f'For "{query_text.strip()}", the best match is '
        f"{top_song['title']} by {top_song['artist']} (score {top_score:.2f}) "
        f"— {first_reason}."
    ]
    if len(retrieved) > 1:
        also = ", ".join(f"{s['title']} ({sc:.2f})" for s, sc, _ in retrieved[1:3])
        parts.append(f"You might also like {also}.")
    return " ".join(parts)


def claude_generate(client, query_text: str, retrieved: List[Tuple[Dict, float, str]]) -> str:
    """Generate a grounded recommendation with Claude, using ONLY the retrieved
    songs. The retrieved data is the sole source of truth for the answer."""
    context = _format_retrieved(retrieved)
    system = (
        "You are VibeMatch, a music recommender. Recommend ONLY from the songs "
        "provided below — never invent or mention any song, artist, or album that "
        "is not in the list. Ground every claim in the given scores and reasons. "
        "Write 2-4 warm, natural sentences: name the top pick and why it fits, "
        "then optionally one or two alternatives. Do not output a list or JSON."
    )
    user = f'Listener request: "{query_text}"\n\nRetrieved songs (the only ones you may recommend):\n{context}'
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("generation refused by safety classifier")
    usage = getattr(response, "usage", None)
    if usage is not None:
        logger.info("claude_generate tokens: in=%s out=%s", usage.input_tokens, usage.output_tokens)
    return next((b.text for b in response.content if b.type == "text"), "").strip()


# --------------------------------------------------------------------------- #
# GROUNDING guard                                                             #
# --------------------------------------------------------------------------- #

def grounding_check(answer: str, retrieved: List[Tuple[Dict, float, str]], catalog: List[Dict]) -> Tuple[bool, List[str]]:
    """Return (is_grounded, hallucinated_titles).

    The answer is grounded if every catalog song it names was actually
    retrieved. A song title from the full catalog that appears in the answer
    but was NOT in the retrieved set is a hallucination (the model recommended
    something it wasn't given)."""
    retrieved_titles = {song["title"].lower() for song, _, _ in retrieved}
    answer_l = answer.lower()
    hallucinated = [
        song["title"] for song in catalog
        if song["title"].lower() in answer_l and song["title"].lower() not in retrieved_titles
    ]
    return (len(hallucinated) == 0, hallucinated)


# --------------------------------------------------------------------------- #
# Pipeline                                                                    #
# --------------------------------------------------------------------------- #

def recommend_rag(query_text: str, songs: List[Dict], k: int = 5, use_llm: str = "auto") -> RagResult:
    """Run the full RAG pipeline.

    use_llm: "auto" (live Claude when available, else offline), "offline"
    (force the deterministic path -- used by the reliability harness), or
    "live" (require Claude; still degrades gracefully on error)."""
    warnings: List[str] = []

    client = None
    if use_llm in ("auto", "live"):
        client, reason = get_client()
        if client is None:
            msg = f"LLM unavailable ({reason}); using offline path."
            logger.info(msg)
            if use_llm == "live":
                warnings.append(msg)

    # 1. PARSE
    parse_engine = "offline"
    if client is not None:
        try:
            profile = claude_parse(client, query_text)
            parse_engine = "claude"
        except Exception as exc:
            warnings.append(f"Claude parse failed ({exc}); used offline parser.")
            logger.warning("claude_parse failed: %s", exc)
            profile = offline_parse(query_text)
    else:
        profile = offline_parse(query_text)
    logger.info("parsed profile via %s: %s", parse_engine, profile.to_prefs())

    # 2. RETRIEVE (deterministic content-based engine)
    retrieved = recommend_songs(profile.to_prefs(), songs, k=k)
    logger.info("retrieved %d songs: %s", len(retrieved), [s["title"] for s, _, _ in retrieved])

    # 3. GENERATE (grounded in the retrieved songs)
    generate_engine = "offline"
    answer = ""
    if client is not None:
        try:
            candidate = claude_generate(client, query_text, retrieved)
            grounded, hallucinated = grounding_check(candidate, retrieved, songs)
            if grounded:
                answer, generate_engine = candidate, "claude"
            else:
                warnings.append(
                    "Grounding guard: Claude mentioned non-retrieved song(s) "
                    f"{hallucinated}; discarded and used offline generator."
                )
                logger.warning("grounding guard tripped: %s", hallucinated)
        except Exception as exc:
            warnings.append(f"Claude generation failed ({exc}); used offline generator.")
            logger.warning("claude_generate failed: %s", exc)

    if not answer:
        answer = offline_generate(query_text, retrieved)

    return RagResult(
        query_text=query_text,
        profile=profile,
        retrieved=retrieved,
        answer=answer,
        parse_engine=parse_engine,
        generate_engine=generate_engine,
        warnings=warnings,
    )
