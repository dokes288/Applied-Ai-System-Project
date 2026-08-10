# 🎵 VibeMatch — An Explainable, AI-Assisted Music Recommender

> A content-based music recommender you can talk to in plain English, that
> explains every pick, runs with **or without** an API key, and ships with an
> automated reliability gate.

**Original project (AI-110, Modules 1–3): "VibeMatch — Music Recommender
Simulation."** The original project was a rule-based, content-based recommender:
each of 10 songs is stored as structured metadata (genre, mood, energy, tempo,
valence, acousticness, …), a user describes their taste as a short profile, and
the system scores every song and returns the top matches, each with a
plain-language explanation of *why* it ranked well. Its goal was to show, at a
miniature scale, how real recommenders turn user preferences + item features into
ranked predictions — and to make the resulting biases (genre over-weighting, tiny
catalog, no behavioral data) visible and discussable.

---

## Summary — what this project does and why it matters

This project extends that rule-based recommender into an **AI system** while
keeping everything that made the original explainable. Two capabilities are new:

1. **Natural-language recommendations (RAG).** Instead of hand-writing a profile,
   you type a request like *"nostalgic 80s synthwave, nothing explicit."* A
   language model turns that into a structured profile, the recommender retrieves
   the best-matching songs **with their reasons**, and a language model writes a
   recommendation **grounded only in those retrieved songs**.
2. **A reliability / quality gate.** An automated harness measures how correctly
   and consistently the pipeline behaves and fails the build if any metric
   regresses.

**Why it matters:** it demonstrates the two ideas most production AI systems are
built on — **retrieval-augmented generation** (ground the model in real data so it
does not make things up) and **evaluation** (measure the AI, do not just trust it) —
in a small, fully reproducible package. Crucially, it runs **with the Claude API
when a key is present and a deterministic offline fallback when it is not**, so
anyone can clone it and get a working result immediately.

---

## Architecture Overview

The full diagram lives in **[system_diagram.md](system_diagram.md)** (Mermaid
source). In short, a request flows **input → process → output** through three
stages, with automated checks and humans at the edges:

```
👤 user request
      │  (natural language)
      ▼
  CLI  ──►  1. PARSE ──► 🛡️ validate ──► 2. RETRIEVE ──► 3. GENERATE ──► 🛡️ grounding guard ──► answer ──► 👤 user
 src/main.py   (LLM or          profile     recommend_songs()   (LLM or        only retrieved                     reads &
               offline)                     over songs.csv       offline)       songs allowed?                    judges
```

**Main components**

| Component | File | Role |
|-----------|------|------|
| **CLI / orchestrator** | `src/main.py` | Entry point; routes `ask`, `reliability`, and the scoring modes |
| **Retriever** | `src/recommender.py` + `data/songs.csv` | Content-based scorer: score → sort → explain; returns top-k songs **with reason strings** |
| **RAG pipeline (the "agent" steps)** | `src/rag.py` | Parse free text → retrieve → generate a grounded answer; Claude (`claude-opus-5`) live or a deterministic offline implementation |
| **Evaluator / tester** | `src/reliability.py`, `tests/` | Metrics + pass/fail gate; `pytest` unit tests |

**Where AI results are checked** (🛡️ automated, 👤 human): a **profile guardrail**
validates/clamps the parsed profile before scoring; a **grounding guard** discards
any generated answer that names a song that was not retrieved; the **reliability
gate + pytest** block on regressions; and humans (the listener, and the developer
reviewing the PASS/FAIL report) make the final judgment.

---

## Setup Instructions

Requires **Python 3.10+**. Runs on Windows, macOS, and Linux.

```bash
# 1. Clone and enter the project
git clone https://github.com/dokes288/Applied-Ai-System-Project.git
cd Applied-Ai-System-Project

# 2. (optional but recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\Activate.ps1       # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run it — no API key needed (deterministic offline path)
python -m src.main ask "chill lofi for studying, instrumental"
python -m src.main reliability
pytest -q
```

**Optional — enable the live Claude path.** Set an Anthropic API key and the same
commands automatically use the real LLM (`claude-opus-5`) for the parse and
generate steps; without a key they use the offline fallback. No code changes.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."     # macOS / Linux
$env:ANTHROPIC_API_KEY = "sk-ant-..."     # Windows PowerShell
```

See **[.env.example](.env.example)** for all environment variables
(`ANTHROPIC_API_KEY`, `VIBEMATCH_MODEL`, `VIBEMATCH_DEBUG`).

**Other commands**

```bash
python -m src.main                 # the 4 classic demo profiles (baseline recipe)
python -m src.main compare         # one profile ranked under all scoring modes
python -m src.main diversity       # top-5 with vs. without the diversity penalty
python -m src.main table           # formatted table output (with full reasons)
```

---

## Sample Interactions

All outputs below are **real** runs on the deterministic offline path (no API
key), so they reproduce exactly. With a key set, the parse/generate steps are
produced by Claude instead, but the retrieval and grounding are identical.

### Example 1 — Natural-language request (RAG)

**Input**
```bash
python -m src.main ask "nostalgic 80s synthwave, nothing explicit, no lyrics"
```
**Output**
```text
You asked: "nostalgic 80s synthwave, nothing explicit, no lyrics"

Parsed profile (offline): {'genre': 'synthwave', 'energy': 0.5, 'likes_acoustic': False,
  'desired_mood_tags': ['nostalgic'], 'preferred_decade': 1980,
  'preferred_language': 'instrumental', 'target_instrumentalness': 0.9, 'allow_explicit': False}

Retrieved (content-based engine):
  1. Night Drive Loop — Neon Echo [synthwave]  (6.78)
  2. Coffee Shop Stories — Slow Stereo [jazz]  (5.05)
  3. Library Rain — Paper Lanterns [lofi]  (4.84)
  4. Midnight Coding — LoRoom [lofi]  (4.08)
  5. Focus Flow — LoRoom [lofi]  (4.00)

Recommendation (offline):
  For "nostalgic 80s synthwave, nothing explicit, no lyrics", the best match is
  Night Drive Loop by Neon Echo (score 6.78) — Genre match: synthwave (+2.0).
  You might also like Coffee Shop Stories (5.05), Library Rain (4.84).
```
The free text is correctly parsed into a synthwave / 1980s / instrumental /
no-explicit profile, and *Night Drive Loop* — the catalog's 1980s synthwave
track — is retrieved and recommended. The answer only names retrieved songs.

### Example 2 — A different request, different result

**Input**
```bash
python -m src.main ask "chill lofi for studying, instrumental"
```
**Output (recommendation)**
```text
Retrieved (content-based engine):
  1. Library Rain — Paper Lanterns [lofi]  (7.04)
  2. Midnight Coding — LoRoom [lofi]  (7.00)
  3. Focus Flow — LoRoom [lofi]  (6.00)
  ...
Recommendation (offline):
  For "chill lofi for studying, instrumental", the best match is Library Rain by
  Paper Lanterns (score 7.04) — Genre match: lofi (+2.0). You might also like
  Midnight Coding (7.00), Focus Flow (6.00).
```

### Example 3 — Classic profile-based recommendation with reasons

**Input**
```bash
python -m src.main            # runs the "High-Energy Pop" demo profile, among others
```
**Output (top pick)**
```text
1. Sunrise City — Neon Echo
   Score: 5.78
   Reasons:
     - Genre match: pop (+2.0)
     - Mood match: happy (+1.0)
     - Energy similarity: 0.82 vs target 0.80 (+1.96)
     - Acoustic fit: acousticness 0.18 vs non-acoustic preference (+0.82)
```
Every recommendation ships with the exact reasons it scored well — the
explainability the whole project is built around. (Full four-profile output is in
[Reference: Full Sample Recommendation Output](#reference-full-sample-recommendation-output).)

---

## Design Decisions & Trade-offs

- **Content-based, not collaborative.** VibeMatch scores songs purely on their own
  attributes. **Trade-off:** it works with zero behavioral data and is fully
  explainable (great for a portfolio demo and cold-start), but it cannot make the
  social "people like you also loved…" discoveries a real hybrid system does.
- **Explainability over cleverness.** Every score is a sum of named, inspectable
  terms, and every recommendation lists its reasons. **Trade-off:** a simple
  additive model is easy to reason about and audit, but it is blind to signals it
  does not score (valence, tempo) — see Testing Summary.
- **RAG grounded in retrieval, not a bare chatbot.** The LLM's answer is
  constructed *from* the retrieved songs and their real scores; it is told never to
  mention a song it was not given, and a **grounding guard** enforces that at
  runtime. **Trade-off:** the answer is factual and on-catalog, at the cost of the
  free-form creativity an ungrounded model would show.
- **Live LLM + deterministic offline fallback.** Both AI steps use Claude when a
  key is available and a deterministic local implementation otherwise. **Trade-off:**
  more code to maintain two paths, but the project is reproducible for anyone (a
  grader with no key still gets a working, identical-every-run result) and testable
  offline.
- **Opt-in advanced features.** Extra signals (decade, mood tags, language,
  instrumentalness, explicit filter) and scoring modes only fire when a request
  asks for them, so the documented baseline behavior never silently changes.
- **Strategy pattern for scoring modes.** Ranking modes are interchangeable
  `ScoringStrategy` objects, so adding a mode never edits the scoring code.
- **Structured outputs for parsing.** The live parse step uses the model's
  structured-output mode, guaranteeing a valid profile object rather than
  hand-parsing free text.

These decisions came out of real experiments — e.g. switching genre matching from
substring to strict equality, and the acoustic term from a hard threshold to a
continuous score — documented in
[Reference: Experiments](#reference-experiments).

---

## Testing Summary

**What worked**
- **31 automated tests pass** (`pytest -q`): 21 for the recommender, 10 for the
  RAG pipeline and reliability harness.
- **Reliability gate passes at 1.00 on every metric** (`python -m src.main
  reliability`): parse determinism, parse accuracy, retrieval precision@1,
  grounding rate, and end-to-end determinism — and it exits non-zero on
  regression, so it is CI-ready.
- The **offline path** was verified end-to-end (this environment has no API key),
  proving the reproducibility guarantee holds.
- The **grounding guard** is tested directly: a crafted answer naming a
  non-retrieved song is correctly flagged as a hallucination and discarded.

**What did not work at first / what I had to fix**
- An early energy term could go **negative** on out-of-range input and cancel the
  genre/mood match; fixed by clamping the target to `[0,1]` and flooring the term
  at 0 (covered by a regression test).
- A previously advertised popularity preference was a **silent no-op** because the
  CSV had no popularity column; fixed by adding the data and wiring it in as an
  opt-in term.
- Genre matching by substring let "indie pop" credit against "pop"; switched to
  strict equality to match the validated behavior.

**What I learned**
- The model is **only as good as the signals it scores.** A "metal, angry"
  request surfaces upbeat pop, because the scorer ignores valence (dark vs.
  bright) and tempo — the exact features that define the genre. Weighting cannot
  fix a missing signal.
- For anything with an LLM, **a guardrail plus a measurement harness matters as
  much as the feature itself** — the grounding guard and reliability gate are what
  make the AI output trustworthy.

---

## Reflection (brief)

Building this made the AI system pattern concrete: **retrieve real data, ground
the model in it, then measure the result** — the same shape whether the catalog
is 10 songs or a hundred million. The most valuable habit it reinforced was
treating AI output as something to *verify*, not trust: the grounding guard and
the reliability gate did more for the system's credibility than any single prompt.

> My full, graded **responsible-AI reflection** — how I collaborated with AI, one
> helpful and one flawed AI suggestion, and the system's limitations — is in
> **[model_card.md](model_card.md)**. The agentic build workflow (prompts,
> generated changes, manual verification) is in
> **[ai_interactions.md](ai_interactions.md)**.

---
---

# Reference

Deeper technical detail for readers who want it. None of this is required to run
the project.

## Reference: How the Recommender Scores

This is the actual scoring logic in `score_song()` / `Recommender._score_song()`
(both paths delegate to the same function, so they always agree):

```
score = genre_match + mood_match + energy_similarity + acoustic_similarity + popularity_bonus

genre_match          = 2.0 if song.genre == user.favorite_genre else 0.0
mood_match           = 1.0 if song.mood == user.favorite_mood  else 0.0
target_energy        = clamp(user.target_energy, 0.0, 1.0)   # out-of-range input is clamped
energy_similarity    = max(0.0, 2.0 * (1 - abs(song.energy - target_energy)))  # floored at 0
target_acousticness  = 1.0 if user.likes_acoustic else 0.0
acoustic_similarity  = 1.0 * (1 - abs(song.acousticness - target_acousticness))

# popularity_bonus is OPT-IN: only scored when the user states a preference.
popularity_bonus     = 1.00 if user.prefers_popular is True  and song.popularity >= 70 else
                       0.75 if user.prefers_popular is False and song.popularity <= 30 else
                       0.0   # prefers_popular is None (default) => always 0.0

max possible score = 6.0 (prefers_popular = None) / 7.0 (popularity preference stated and met)
```

**Data flow:** score every song → sort by score descending (ties broken by title,
ascending, for determinism) → return the top `k` → attach an explanation built
from the reasons that earned points. Genre outweighs mood 2:1 (genre is the
primary style filter); energy and acoustic fit are continuous distance terms, so
there is no "dead zone" where a near-match earns nothing.

### Song features

| Feature | Example | Role in scoring |
|---------|---------|-----------------|
| `genre` | pop, lofi, rock | Exact match to favorite genre — binary gate |
| `mood` | happy, chill, intense | Exact match to preferred mood — binary gate |
| `energy` | 0.0–1.0 | Continuous similarity to target energy |
| `acousticness` | 0.0–1.0 | Continuous similarity to acoustic preference |
| `popularity` | 0–100 | Opt-in bonus when `prefers_popular` is set |
| `release_decade`, `mood_tags`, `language`, `instrumentalness`, `explicit` | — | Opt-in advanced signals (below) |
| `tempo_bpm`, `valence`, `danceability` | — | Loaded, not yet scored |

## Reference: Advanced Features, Scoring Modes & Diversity

All three layers are **opt-in and backward-compatible** — a request that does not
use them scores exactly as the baseline does.

**Advanced song features** (scored only when the request supplies the matching preference):

| Attribute | Preference key | How it scores |
|-----------|----------------|---------------|
| `release_decade` | `preferred_decade` | exact decade **+1.0**, one decade away **+0.5** |
| `mood_tags` (e.g. `nostalgic\|dreamy`) | `desired_mood_tags` | **+1.0 × fraction** of wanted tags the song has |
| `language` | `preferred_language` | exact match **+1.0** |
| `instrumentalness` | `target_instrumentalness` | continuous **+1.0 × (1 − |diff|)** |
| `explicit` | `allow_explicit` | **−2.0** penalty when the listener opts out |

**Scoring modes (Strategy pattern):** `balanced` (default), `genre-first`,
`mood-first`, `energy-focused` — interchangeable weight sets. Run
`python -m src.main compare` to see one profile ranked under all four.

**Diversity / fairness penalty** (`python -m src.main diversity`): a re-ranking
step that subtracts **−1.5** per already-listed same-artist song and **−0.75** per
same-genre song, so one artist or genre cannot monopolize the top-k.

## Reference: AI Features in depth (RAG + Reliability)

The RAG pipeline ([src/rag.py](src/rag.py)) runs **parse → retrieve → generate**:
an LLM (or offline parser) turns free text into a profile; `recommend_songs`
retrieves the top-k songs with reasons; an LLM (or offline template) writes an
answer grounded only in those songs. Live steps use the **Claude API**
(`claude-opus-5`) when `ANTHROPIC_API_KEY` is set; otherwise a deterministic
offline path runs. Guardrails: profile validation/clamping, a grounding guard
(discards hallucinated answers), refusal/error handling that degrades to offline,
and step-level logging (`VIBEMATCH_DEBUG=1`).

The reliability harness ([src/reliability.py](src/reliability.py)) scores the
pipeline on labeled queries and exits non-zero on regression:

| Metric | Checks | Current |
|--------|--------|---------|
| `parse_determinism` | same query → identical profile twice | 1.00 |
| `parse_accuracy` | parsed fields match hand-labeled expectations | 1.00 |
| `retrieval_precision_at_1` | requested genre (when in catalog) is #1 | 1.00 |
| `grounding_rate` | answers name only retrieved songs | 1.00 |
| `e2e_determinism` | same query → identical final answer | 1.00 |

## Reference: Full Sample Recommendation Output

Real output from `python -m src.main` against `data/songs.csv` (baseline recipe).
Each profile's top-5 is shown in its own block.

### Profile 1 — High-Energy Pop
```text
 USER PROFILE: High-Energy Pop  (genre=pop, mood=happy, energy=0.8, likes_acoustic=False)

1. Sunrise City — Neon Echo        Score: 5.78  (genre +2.0 | mood +1.0 | energy +1.96 | acoustic +0.82)
2. Gym Hero — Max Pulse            Score: 4.69  (genre +2.0 | energy +1.74 | acoustic +0.95)
3. Rooftop Lights — Indigo Parade  Score: 3.57  (mood +1.0 | energy +1.92 | acoustic +0.65)
4. Night Drive Loop — Neon Echo    Score: 2.68  (energy +1.90 | acoustic +0.78)
5. Storm Runner — Voltline         Score: 2.68  (energy +1.78 | acoustic +0.90)
```

### Profile 2 — Chill Lofi
```text
 USER PROFILE: Chill Lofi  (genre=lofi, mood=chill, energy=0.4, likes_acoustic=True)

1. Library Rain — Paper Lanterns   Score: 5.76  (genre +2.0 | mood +1.0 | energy +1.90 | acoustic +0.86)
2. Midnight Coding — LoRoom        Score: 5.67  (genre +2.0 | mood +1.0 | energy +1.96 | acoustic +0.71)
3. Focus Flow — LoRoom             Score: 4.78  (genre +2.0 | energy +2.00 | acoustic +0.78)
4. Spacewalk Thoughts — Orbit Bloom Score: 3.68 (mood +1.0 | energy +1.76 | acoustic +0.92)
5. Coffee Shop Stories — Slow Stereo Score: 2.83 (energy +1.94 | acoustic +0.89)
```

### Profile 3 — Deep Intense Rock
```text
 USER PROFILE: Deep Intense Rock  (genre=rock, mood=intense, energy=0.9, likes_acoustic=False)

1. Storm Runner — Voltline         Score: 5.88  (genre +2.0 | mood +1.0 | energy +1.98 | acoustic +0.90)
2. Gym Hero — Max Pulse            Score: 3.89  (mood +1.0 | energy +1.94 | acoustic +0.95)
3. Sunrise City — Neon Echo        Score: 2.66  (energy +1.84 | acoustic +0.82)
4. Night Drive Loop — Neon Echo    Score: 2.48  (energy +1.70 | acoustic +0.78)
5. Rooftop Lights — Indigo Parade  Score: 2.37  (energy +1.72 | acoustic +0.65)
```

### Profile 4 — Off-Catalog Taste (deliberate near-miss)
```text
 USER PROFILE: Off-Catalog Taste  (genre=metal, mood=angry, energy=0.85, likes_acoustic=False)

1. Gym Hero — Max Pulse            Score: 2.79   ← no genre/mood match fires;
2. Storm Runner — Voltline         Score: 2.78     every score stays below 3.0,
3. Sunrise City — Neon Echo        Score: 2.76     decided purely by energy + acoustic
4. Night Drive Loop — Neon Echo    Score: 2.58
5. Rooftop Lights — Indigo Parade  Score: 2.47
```
`metal`/`angry` match no song, so both categorical gates score 0.0 and ranking is
decided purely by energy + acoustic similarity — every score falls below **3.0**,
the most any song can earn without the genre (2.0) and mood (1.0) gates firing.

## Reference: How Real Streaming Platforms Work

Spotify, YouTube Music, and Apple Music use **hybrid** recommenders:

- **Collaborative filtering** (behavior-based): learns from likes, skips, replays,
  and playlist co-occurrence — "users like you also liked…". Strong at discovery;
  weak at cold-start; prone to popularity bias.
- **Content-based filtering** (attribute-based): learns from what the item *is* —
  genre, audio features, embeddings, lyrics. Works for brand-new songs; can
  over-recommend similar-sounding tracks.

Production systems blend both and add **context** (workout vs. focus),
**diversity controls**, and **fairness** goals. VibeMatch is purely
content-based — one honest slice of that stack, made fully explainable.

## Reference: Experiments

| Change | What happened |
|--------|---------------|
| Genre/mood matching: substring vs. strict equality | Substring let "indie pop" credit against "pop" (*Rooftop Lights* 4.92); strict equality dropped it to 2.92 — matching validated behavior. |
| Acoustic term: hard threshold vs. continuous | The threshold left a 0.3–0.7 dead zone; the continuous term makes every value contribute proportionally. |
| No tie-break vs. tie-break by title | Added `(-score, title)` so equal scores resolve identically every run. |
| Doubling energy weight (sensitivity test) | Reshuffled half the profiles but did not fix the "metal → pop" problem — proof that weighting cannot replace a missing signal (valence/tempo). |

## Limitations and Risks

- **Tiny catalog** — 10 handcrafted songs; no long-tail discovery.
- **No behavioral data** — no skips, repeats, or "people like you" signals.
- **Blind to some signals** — valence, tempo, and danceability are loaded but not
  scored, so mood color and intensity do not register.
- **Feature/representation bias** — over-weighting genre/mood, and a small curated
  dataset, can hide adjacent-genre matches and underrepresent artists or regions.
- **Filter-bubble risk** — even simple rules can reinforce existing taste.

A fuller, graded discussion is in **[model_card.md](model_card.md)**.
