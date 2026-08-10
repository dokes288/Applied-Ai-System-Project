# Applied AI System Project
## Music Recommender Simulation System

An end-to-end, modular AI system that recommends music based on user preferences using retrieval, scoring logic, agentic planning, and built-in guardrails.

---

## Project Goals

| Goal | Implementation |
|------|---------------|
| Extend a prior mini-project into a cohesive AI system | Fully offline Music Recommender built in Python |
| Implement modular components (retrieval, logic, agentic planning) | `retrieval.py`, `recommender.py`, `planner.py` |
| Test & evaluate reliability and guardrails | 47 structured pytest tests in `tests/` |
| Document AI decision-making clearly | Inline docstrings + `explain()` / `explain_run()` APIs |
| Professional portfolio entry | This README |

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  AgenticPlanner                      │
│  Step 1 ─ Guardrails.validate()                     │
│  Step 2 ─ MusicCatalog.search() / all_songs()       │
│  Step 3 ─ MusicRecommender.recommend()              │
│  Step 4 ─ Reflection & constraint relaxation        │
│  Step 5 ─ Return results + full trace               │
└─────────────────────────────────────────────────────┘
         │            │              │
    retrieval.py  recommender.py  guardrails.py
```

### Module Overview

| Module | Class | Responsibility |
|--------|-------|---------------|
| `retrieval.py` | `MusicCatalog` | Stores songs; exposes genre, mood, and keyword search |
| `recommender.py` | `MusicRecommender` | Scores and ranks songs by weighted criteria |
| `guardrails.py` | `Guardrails` | Validates inputs; enforces allowlists and ranges |
| `planner.py` | `AgenticPlanner` | Orchestrates the full plan-and-execute loop |

---

## Scoring Formula

The recommender uses a transparent, auditable weighted score:

| Criterion | Weight |
|-----------|--------|
| Genre match | +3.0 |
| Mood match | +2.0 |
| Tag overlap | +1.0 per tag |
| BPM proximity | +0.0 – 1.0 (1 = exact, 0 at ≥ 60 BPM difference) |

---

## Guardrails

All user inputs are validated **before** any recommendation is computed:

- **Genre** — must be one of the known genres (`pop`, `rock`, `hip-hop`, etc.)
- **Mood** — must be one of the known moods (`happy`, `sad`, `energetic`, etc.)
- **BPM** — must be in the realistic range 20 – 300
- **top_n** — positive integer, max 50
- **Query** — string, max 200 characters
- **Tags** — list of strings, max 10 items

Every rejection returns a human-readable reason via `ValidationResult.errors`.

---

## Agentic Planning Loop

The `AgenticPlanner` implements an explicit **plan → execute → reflect** cycle:

1. **Validate** all inputs via `Guardrails`.
2. **Retrieve** candidates (keyword search or full catalog).
3. **Score & rank** using `MusicRecommender`.
4. **Reflect** — if fewer than `min_results` candidates have a positive score, relax the genre constraint and retry once.
5. **Return** final recommendations with a complete execution trace.

Every run returns a `trace` list that explains every decision made, supporting transparency and auditability.

---

## Quick Start

```python
from music_recommender import AgenticPlanner

planner = AgenticPlanner()

# Full agentic run with explanation
print(planner.explain_run(genre="pop", mood="happy", bpm=120, top_n=5))
```

**Example output:**
```
=== Agentic Planner Execution Trace ===
  Step 1: Validating inputs via Guardrails.
  ✓ All inputs are valid.
  Step 2: Retrieving candidate songs from catalog.
  Full catalog loaded: 20 song(s).
  Step 3: Scoring and ranking candidates.
  Returned 5 result(s), 5 with positive score.
  Step 4: Result quality is acceptable; no retry needed.
  Step 5: Returning final recommendations.

=== Final Recommendations ===
  1. Levitating — Dua Lipa [pop, happy, 103 BPM]  score=5.983
  2. Flowers — Miley Cyrus [pop, happy, 118 BPM]  score=5.7
  ...
```

### Using individual modules

```python
from music_recommender import MusicCatalog, MusicRecommender, Guardrails

# Validate inputs first
result = Guardrails.validate(genre="rock", mood="energetic", bpm=120)
if result:
    catalog = MusicCatalog()
    rec = MusicRecommender(catalog)
    songs = rec.recommend(genre="rock", mood="energetic", bpm=120, top_n=3)
    print(rec.explain(genre="rock", mood="energetic", bpm=120, top_n=3))
else:
    print("Invalid input:", result.errors)
```

---

## Project Structure

```
Applied-Ai-System-Project/
├── music_recommender/
│   ├── __init__.py       # Public API exports
│   ├── retrieval.py      # MusicCatalog + Song dataclass
│   ├── recommender.py    # MusicRecommender + score_song()
│   ├── guardrails.py     # Guardrails + ValidationResult
│   └── planner.py        # AgenticPlanner (plan-and-execute)
└── tests/
    └── test_recommender.py   # 47 structured pytest tests
```

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Evaluation & Reliability

The system is validated through 47 structured tests covering:

- **Retrieval**: genre/mood/keyword filtering, edge cases (empty query, unknown genre)
- **Scoring**: each weight component in isolation and in combination
- **Guardrails**: all validation rules including boundary conditions
- **Planner**: happy path, invalid inputs, retry/reflection loop, query pre-filtering

---

## AI Decision-Making Transparency

All AI decisions in this system are:

1. **Deterministic** — same inputs always produce the same ranked output.
2. **Explainable** — `explain()` and `explain_run()` expose the formula and every step.
3. **Auditable** — every `AgenticPlanner.run()` returns a `trace` list.
4. **Bounded** — guardrails prevent out-of-range inputs from reaching the model.

---

## License

MIT

