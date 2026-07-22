# 🎵 Music Recommender Simulation

## Project Summary

This project simulates a small music recommender called **VibeMatch**. Each song is stored as structured metadata (genre, mood, energy, tempo, valence, and more). A user describes their taste with a short profile, and the system scores every song in a 10-track catalog. The highest-scoring songs become recommendations, each with a plain-language explanation of why it ranked well.

The goal is to show how real-world AI recommenders work at a miniature scale: turn preferences and item features into ranked predictions, then reflect on what the system gets right, what it misses, and where bias can appear.

---

## How The System Works

Real-world recommenders almost never rely on a single signal. Platforms like Spotify and YouTube run **hybrid** systems that blend collaborative filtering — learning from what other users with similar taste played, skipped, or saved — with content-based filtering, which scores an item purely on its own attributes (genre, tempo, energy, and so on). Collaborative signals are powerful because they surface unexpected matches a feature-based system would never find on its own, but they need behavioral data to work at all, so they struggle with brand-new users and brand-new songs. Content-based signals solve exactly that cold-start gap, at the cost of only ever recommending things that *resemble* what a listener already engages with. **VibeMatch prioritizes the content-based half of that picture, deliberately and transparently:** it scores every song purely on how well its metadata and audio features match a stated taste profile, weighting genre and mood as binary gates (2.0 and 1.0 points respectively) and treating energy and acousticness as continuous similarity terms on top. That tradeoff is the point of this simulation — by never looking at other users' behavior, VibeMatch stays fully explainable (every recommendation ships with the exact reasons it scored well) at the cost of the discovery and cold-start resilience a real collaborative signal would add.

### Algorithm Recipe

This is the actual scoring logic implemented in `score_song()` / `Recommender._score_song()` (both paths delegate to the same function, so they always agree):

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

The energy term is guarded twice: `target_energy` is clamped into `[0, 1]` and the
term is floored at `0.0`, so an out-of-range energy value (e.g. one accidentally
passed on a 0â100 scale) can never subtract from the score and cancel the gates.
`popularity_bonus` defaults to `0.0` for every profile that doesn't set
`prefers_popular`, so all sample output below is unchanged by its addition.

**Data flow:** `User Prefs` → loop over every song in the catalog, scoring genre match, mood match, energy similarity, and acoustic similarity → sum into a total per song → sort all songs by score, descending → return the top `k`.

Genre outweighs mood 2:1, since genre is treated as the primary style filter and mood as a secondary preference signal. Energy and acoustic fit are both continuous distance terms rather than hard thresholds — every value contributes proportionally, in whichever direction the preference points, so there's no "dead zone" where a song earns nothing regardless of how close it is.

**Potential biases to expect:**

- This system might over-prioritize genre, ignoring great songs that match the user's mood but come from a different genre — e.g., an intense rock track could outrank an intense pop track purely because pop is the stated favorite genre.
- Genre and mood are exact-match only, so "close enough" genres (e.g., "pop" vs. "indie pop") get zero credit, identical to a totally unrelated genre — there's no partial-credit mechanism.
- With only four signals scored, a song that's a near-perfect match on valence, tempo, or danceability gets no credit for it — those features are loaded but not yet wired into scoring.
- Because genre + mood alone are worth 3.0 combined — half the max score — a song can win almost entirely on categorical match even with a mediocre energy/acoustic fit, while a song with excellent energy and acoustic fit but no genre/mood match can't out-rank it. Worth watching for in testing, especially with a small catalog where few songs share both categorical fields.

### Song features

Each `Song` stores:

| Feature | Example | Role in scoring |
|---------|---------|-----------------|
| `genre` | pop, lofi, rock | Exact match to user's favorite genre — binary gate, no partial credit |
| `mood` | happy, chill, intense | Exact match to user's preferred mood — binary gate, no partial credit |
| `energy` | 0.0–1.0 | Continuous similarity to user's target energy |
| `acousticness` | 0.0–1.0 | Continuous similarity to acoustic or non-acoustic preference |
| `popularity` | 0–100 | Opt-in threshold bonus, only when `prefers_popular` is set: +1.0 if popular (≥70) and user wants popular; +0.75 if niche (≤30) and user wants niche |
| `tempo_bpm` | 118 | Loaded, not yet used in scoring |
| `valence` | 0.0–1.0 | Loaded, not yet used in scoring |
| `danceability` | 0.0–1.0 | Loaded, not yet used in scoring |

### User profile

Each `UserProfile` (or dict in the functional API) stores:

- `favorite_genre` — e.g. `"pop"`
- `favorite_mood` — e.g. `"happy"`
- `target_energy` — float from 0.0 (calm) to 1.0 (intense)
- `likes_acoustic` — whether the user prefers acoustic-sounding tracks (`True`) or non-acoustic tracks (`False`)
- `prefers_popular` — `True` (wants popular tracks), `False` (wants niche/under-the-radar), or `None` (indifferent, the default — scores no popularity bonus either way)

### Scoring rule

For every song, `score_song()` adds points from four components:

```
score = genre_match
      + mood_match
      + energy_similarity
      + acoustic_similarity
```

This is the same formula documented in full under [Algorithm Recipe](#algorithm-recipe) above:

- Genre match: **2.0** if `song.genre == user.favorite_genre` exactly, else **0.0** — no partial credit
- Mood match: **1.0** if `song.mood == user.favorite_mood` exactly, else **0.0** — no partial credit
- Energy similarity: up to **2.0**, continuous — `2.0 * (1 - |song.energy - user.target_energy|)`
- Acoustic similarity: up to **1.0**, continuous — `1.0 * (1 - |song.acousticness - target_acousticness|)`, where `target_acousticness` is `1.0` if `user.likes_acoustic` else `0.0`

Max possible score: **6.0**. `valence`, `tempo_bpm`, and `danceability` are loaded by `load_songs()` but not currently wired into any of these terms — see [Limitations and Risks](#limitations-and-risks).

### Choosing recommendations

1. Score every song in the catalog with `score_song()`.
2. Sort by score, descending — ties broken by title, ascending, so results are deterministic regardless of catalog order.
3. Return the top `k` songs.
4. Attach an explanation built from the reasons that earned points.

```text
UserProfile + Song catalog
        │
        ▼
  score each song  ──►  sort by score, tie-break by title  ──►  top k + explanations
```

Both an **OOP** path (`Recommender.recommend()`) and a **functional** path (`recommend_songs()`) exist. `Recommender._score_song()` delegates directly to `score_song()` rather than duplicating its logic, so the two paths always agree on scores and rankings for the same profile/catalog.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   python -m src.main
   ```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Advanced Features, Scoring Modes, and Diversity

Beyond the baseline four-signal recipe, VibeMatch adds three optional layers. All
three are **opt-in and backward-compatible**: a profile (or run) that doesn't use
them scores exactly as the baseline does, so the sample output below is unchanged.

### 1. Advanced song features

Five richer attributes were added to `data/songs.csv` and to scoring. Each is
scored **only when the user profile supplies the matching preference**, so they
never affect a baseline profile.

| Attribute | Preference key | How it scores |
|-----------|----------------|---------------|
| `release_decade` (e.g. 1980, 2010) | `preferred_decade` | exact decade **+1.0**, one decade away **+0.5**, else 0 |
| `mood_tags` (e.g. `nostalgic\|dreamy`) | `desired_mood_tags` | **+1.0 × fraction** of the wanted tags the song carries |
| `language` (e.g. `english`) | `preferred_language` | exact normalized match **+1.0** |
| `instrumentalness` (0.0–1.0) | `target_instrumentalness` | continuous **+1.0 × (1 − |diff|)** |
| `explicit` (true/false) | `allow_explicit` | **−2.0** penalty when the listener opts out and the song is explicit |

Base max stays 6.0 for an indifferent profile; each opt-in term can add up to
another point (or, for explicit, subtract). Example: a "Nostalgic 80s" profile
(`preferred_decade=1980`, `desired_mood_tags=["nostalgic","dreamy"]`,
`preferred_language="english"`) correctly ranks *Night Drive Loop* first.

### 2. Scoring modes (Strategy pattern)

Four interchangeable ranking modes re-weight the four core signals. They are
implemented with the **Strategy pattern** (`ScoringStrategy` objects in a
`STRATEGIES` registry), so switching modes never touches the scoring code.

| Mode | Genre | Mood | Energy | Acoustic | Use it for |
|------|-------|------|--------|----------|-----------|
| `balanced` (default) | 2.0 | 1.0 | 2.0 | 1.0 | the original recipe |
| `genre-first` | 4.0 | 1.0 | 1.0 | 1.0 | "stay in my genre" |
| `mood-first` | 1.0 | 4.0 | 1.0 | 1.0 | "match my vibe over my genre" |
| `energy-focused` | 1.0 | 1.0 | 4.0 | 1.0 | workouts / focus by intensity |

```bash
python -m src.main genre-first     # run every profile in Genre-First mode
python -m src.main compare         # rank one profile under ALL modes, side by side
```

`compare` makes the effect obvious — e.g. for a pop/happy listener, Genre-First
ranks *Gym Hero* (pop) above *Rooftop Lights*, while Mood-First flips them
(*Rooftop Lights* is the "happy" match).

### 3. Diversity / fairness penalty

An opt-in **diversity penalty** stops one artist or genre from monopolizing the
top results. It is a **re-ranking** step (not per-song scoring), because it
depends on what has already been picked. As the top-k list is built one slot at a
time, each candidate's effective score is reduced by:

- **−1.5** for each already-listed song by the **same artist**
- **−0.75** for each already-listed song of the **same genre**

The best effective score wins each slot, so a second song by an already-listed
artist must be clearly better to earn its place.

```bash
python -m src.main diversity       # show each profile's top-5 with vs without the penalty
```

Example (Chill Lofi): without the penalty the top 3 are all lofi (two by
*LoRoom*); with it, *Focus Flow* (the 2nd LoRoom/lofi track) drops to #5 and an
ambient and a jazz track surface instead.

### 4. Formatted table output

`python -m src.main table` prints each profile's top-5 as a table (including the
full reasons for each score). It uses [`tabulate`](https://pypi.org/project/tabulate/)
when installed and falls back to a dependency-free ASCII renderer otherwise.

```text
+-----+--------------+-----------+---------+------------------------------------------------+
|   # | Title        | Artist    |   Score | Reasons                                        |
+=====+==============+===========+=========+================================================+
|   1 | Sunrise City | Neon Echo |    5.78 | Genre match: pop (+2.0)                        |
|     |              |           |         | Mood match: happy (+1.0)                       |
|     |              |           |         | Energy similarity: 0.82 vs target 0.80 (+1.96) |
|     |              |           |         | Acoustic fit: acousticness 0.18 vs non-        |
|     |              |           |         | acoustic preference (+0.82)                    |
+-----+--------------+-----------+---------+------------------------------------------------+
```

---

## Sample Recommendation Output

Real output from `python -m src.main`, run against `data/songs.csv` with the current recipe (genre +2.0, mood +1.0, energy similarity up to +2.0, acoustic similarity up to +1.0). Verified identically in both the sandbox and a local Windows run — same scores, same rankings, down to the decimal. Each profile's top-5 recommendations are shown in its own block below.

### Profile 1 — High-Energy Pop

```text
====================================================================
 USER PROFILE: High-Energy Pop
 (genre=pop, mood=happy, energy=0.8, likes_acoustic=False)
====================================================================

1. Sunrise City — Neon Echo
   Score: 5.78
   Reasons:
     - Genre match: pop (+2.0)
     - Mood match: happy (+1.0)
     - Energy similarity: 0.82 vs target 0.80 (+1.96)
     - Acoustic fit: acousticness 0.18 vs non-acoustic preference (+0.82)

2. Gym Hero — Max Pulse
   Score: 4.69
   Reasons:
     - Genre match: pop (+2.0)
     - Energy similarity: 0.93 vs target 0.80 (+1.74)
     - Acoustic fit: acousticness 0.05 vs non-acoustic preference (+0.95)

3. Rooftop Lights — Indigo Parade
   Score: 3.57
   Reasons:
     - Mood match: happy (+1.0)
     - Energy similarity: 0.76 vs target 0.80 (+1.92)
     - Acoustic fit: acousticness 0.35 vs non-acoustic preference (+0.65)

4. Night Drive Loop — Neon Echo
   Score: 2.68
   Reasons:
     - Energy similarity: 0.75 vs target 0.80 (+1.90)
     - Acoustic fit: acousticness 0.22 vs non-acoustic preference (+0.78)

5. Storm Runner — Voltline
   Score: 2.68
   Reasons:
     - Energy similarity: 0.91 vs target 0.80 (+1.78)
     - Acoustic fit: acousticness 0.10 vs non-acoustic preference (+0.90)
```

### Profile 2 — Chill Lofi

```text
====================================================================
 USER PROFILE: Chill Lofi
 (genre=lofi, mood=chill, energy=0.4, likes_acoustic=True)
====================================================================

1. Library Rain — Paper Lanterns
   Score: 5.76
   Reasons:
     - Genre match: lofi (+2.0)
     - Mood match: chill (+1.0)
     - Energy similarity: 0.35 vs target 0.40 (+1.90)
     - Acoustic fit: acousticness 0.86 vs acoustic preference (+0.86)

2. Midnight Coding — LoRoom
   Score: 5.67
   Reasons:
     - Genre match: lofi (+2.0)
     - Mood match: chill (+1.0)
     - Energy similarity: 0.42 vs target 0.40 (+1.96)
     - Acoustic fit: acousticness 0.71 vs acoustic preference (+0.71)

3. Focus Flow — LoRoom
   Score: 4.78
   Reasons:
     - Genre match: lofi (+2.0)
     - Energy similarity: 0.40 vs target 0.40 (+2.00)
     - Acoustic fit: acousticness 0.78 vs acoustic preference (+0.78)

4. Spacewalk Thoughts — Orbit Bloom
   Score: 3.68
   Reasons:
     - Mood match: chill (+1.0)
     - Energy similarity: 0.28 vs target 0.40 (+1.76)
     - Acoustic fit: acousticness 0.92 vs acoustic preference (+0.92)

5. Coffee Shop Stories — Slow Stereo
   Score: 2.83
   Reasons:
     - Energy similarity: 0.37 vs target 0.40 (+1.94)
     - Acoustic fit: acousticness 0.89 vs acoustic preference (+0.89)
```

### Profile 3 — Deep Intense Rock

```text
====================================================================
 USER PROFILE: Deep Intense Rock
 (genre=rock, mood=intense, energy=0.9, likes_acoustic=False)
====================================================================

1. Storm Runner — Voltline
   Score: 5.88
   Reasons:
     - Genre match: rock (+2.0)
     - Mood match: intense (+1.0)
     - Energy similarity: 0.91 vs target 0.90 (+1.98)
     - Acoustic fit: acousticness 0.10 vs non-acoustic preference (+0.90)

2. Gym Hero — Max Pulse
   Score: 3.89
   Reasons:
     - Mood match: intense (+1.0)
     - Energy similarity: 0.93 vs target 0.90 (+1.94)
     - Acoustic fit: acousticness 0.05 vs non-acoustic preference (+0.95)

3. Sunrise City — Neon Echo
   Score: 2.66
   Reasons:
     - Energy similarity: 0.82 vs target 0.90 (+1.84)
     - Acoustic fit: acousticness 0.18 vs non-acoustic preference (+0.82)

4. Night Drive Loop — Neon Echo
   Score: 2.48
   Reasons:
     - Energy similarity: 0.75 vs target 0.90 (+1.70)
     - Acoustic fit: acousticness 0.22 vs non-acoustic preference (+0.78)

5. Rooftop Lights — Indigo Parade
   Score: 2.37
   Reasons:
     - Energy similarity: 0.76 vs target 0.90 (+1.72)
     - Acoustic fit: acousticness 0.35 vs non-acoustic preference (+0.65)
```

### Profile 4 — Off-Catalog Taste (no genre/mood match)

```text
====================================================================
 USER PROFILE: Off-Catalog Taste (no genre/mood match)
 (genre=metal, mood=angry, energy=0.85, likes_acoustic=False)
====================================================================

1. Gym Hero — Max Pulse
   Score: 2.79
   Reasons:
     - Energy similarity: 0.93 vs target 0.85 (+1.84)
     - Acoustic fit: acousticness 0.05 vs non-acoustic preference (+0.95)

2. Storm Runner — Voltline
   Score: 2.78
   Reasons:
     - Energy similarity: 0.91 vs target 0.85 (+1.88)
     - Acoustic fit: acousticness 0.10 vs non-acoustic preference (+0.90)

3. Sunrise City — Neon Echo
   Score: 2.76
   Reasons:
     - Energy similarity: 0.82 vs target 0.85 (+1.94)
     - Acoustic fit: acousticness 0.18 vs non-acoustic preference (+0.82)

4. Night Drive Loop — Neon Echo
   Score: 2.58
   Reasons:
     - Energy similarity: 0.75 vs target 0.85 (+1.80)
     - Acoustic fit: acousticness 0.22 vs non-acoustic preference (+0.78)

5. Rooftop Lights — Indigo Parade
   Score: 2.47
   Reasons:
     - Energy similarity: 0.76 vs target 0.85 (+1.82)
     - Acoustic fit: acousticness 0.35 vs non-acoustic preference (+0.65)
```

This fourth profile is a deliberate near-miss: `metal`/`angry` match no song in the catalog, so both categorical gates score 0.0 for every track and ranking is decided purely by energy + acoustic similarity. Note that every score falls below **3.0** — the most any song can earn without the genre (2.0) and mood (1.0) gates firing.

Both the OOP `Recommender` path and the functional `recommend_songs()` path produce identical scores and rankings for the same profile/catalog, since `Recommender._score_song()` delegates directly to `score_song()` rather than duplicating its logic. Note the deterministic tie-break visible above: `Night Drive Loop` and `Storm Runner` tie at 2.68 in the first profile and resolve alphabetically, not by catalog insertion order.

**Screenshot or video** *(optional)*: <!-- Insert a screenshot or demo video link here -->

---

## How Major Streaming Platforms Predict What You'll Love Next

Spotify, YouTube Music, Apple Music, and similar services use **hybrid** recommenders that combine multiple signals. Two core approaches differ in *what data they learn from*:

### Collaborative filtering (behavior-based)

The system learns from **what users do**, not from audio alone.

- **User–user CF:** "Users similar to you also liked these songs."
- **Item–item CF:** "People who liked Song A also liked Song B."
- **Playlist/session co-occurrence:** Spotify famously treats songs that appear together on playlists or in the same listening session as similar, even when their genres differ.

**Strengths:** discovers unexpected matches, leverages crowd wisdom, improves as data grows.

**Weaknesses:** cold-start for new users and new songs, popularity bias, filter bubbles.

**Data types:** likes, skips, replays, saves, playlist adds, follow actions, listening duration, session context (time of day, device).

### Content-based filtering (attribute-based)

The system learns from **what the item is**.

- **Metadata:** genre, artist, release year, tags.
- **Audio features:** tempo, energy, danceability, valence, acousticness (similar to our CSV).
- **Deep audio analysis:** spectrograms, embeddings from neural networks (Spotify's Echo Nest heritage).
- **Text/NLP:** lyrics, reviews, web mentions.

**Strengths:** works for new songs with no play history; explains "sounds like X"; good for niche taste.

**Weaknesses:** can over-recommend similar-sounding tracks; misses social/cultural discovery.

**Data types:** BPM, key, loudness, mood tags, lyrical themes, artist similarity vectors.

### How platforms combine them

| Platform | Collaborative signals | Content signals |
|----------|----------------------|-----------------|
| **Spotify** | Playlist co-occurrence, listening history, Discover Weekly models | Audio CNN features, NLP on lyrics/metadata, artist graphs |
| **YouTube Music** | Watch/listen sequences, skip/replay patterns, taste communities | Audio embeddings, video metadata, search history from Google ecosystem |

Production systems also add **context** (workout vs focus), **diversity controls**, and **fairness** goals so recommendations are not only accurate but varied and representative.

Our VibeMatch simulation is **purely content-based**: it never sees other users' behavior, which mirrors only one slice of a real platform stack.

---

## Experiments You Tried

| Change | What happened |
|--------|---------------|
| Genre/mood matching: substring (`_matches()`) vs. strict equality | With substring matching, "indie pop" credited against a "pop" preference (e.g. *Rooftop Lights* scored 4.92). Switched to strict equality to match what was actually validated during design — *Rooftop Lights* dropped to 2.92 once genre stopped crediting. |
| Acoustic term: hard threshold vs. continuous similarity | The threshold version (`+1.0` if `acousticness >= 0.7`, `+0.5` if `<= 0.3`) left a 0.3–0.7 dead zone earning nothing either way. Switched to a continuous similarity term (`1.0 * (1 - |diff|)`) so every acousticness value contributes proportionally, regardless of preference direction. |
| No tie-break vs. deterministic tie-break by title | Without a secondary sort key, songs with equal scores (e.g. *Night Drive Loop* and *Storm Runner*, both 2.68 for a pop/happy profile) resolved by catalog insertion order — not guaranteed stable if the CSV row order changed. Added `(-score, title)` as the sort key so ties resolve the same way every run. |
| `likes_acoustic=True` vs. `likes_acoustic=False` | Same song, opposite acoustic_similarity direction — e.g. *Sunrise City* (acousticness 0.18) scores `+0.82` for a non-acoustic preference but only `+0.18` for an acoustic one. |

These experiments show that **scoring rules are design choices with real, testable tradeoffs** — matching strategy, threshold-vs-continuous scoring, and tie-break rules all visibly change which songs rank where, even holding the catalog and profile fixed.

---

## Limitations and Risks

- **Tiny catalog** — only 10 handcrafted songs; no long-tail discovery.
- **No behavioral data** — no skips, repeats, or "people like you" signals.
- **No lyrics or culture** — cannot understand language, nostalgia, or social trends.
- **Feature bias** — overweighting genre/mood can hide great matches in adjacent genres.
- **Filter bubble risk** — even simple rules can reinforce existing preferences.
- **Representation gaps** — a small, curated dataset may underrepresent artists, languages, or regions.

See `model_card.md` for a fuller discussion.

---

## Reflection

This project showed me that recommenders are **ranking engines**: they convert structured signals (user prefs + item features) into an ordered list of likely favorites. The math here is simple addition, but the pattern is the same at Spotify scale — score, sort, explain (or at least justify internally).

It also highlighted **bias and fairness**. A system that rewards genre match and popularity-like features can keep users inside familiar lanes. Real platforms fight this with diversity rules and collaborative discovery, but those same tools can still amplify mainstream artists or under-recommend music from underrepresented communities. Building even a toy recommender makes those tradeoffs visible early.

Read and complete the model card for more detail:

[**Model Card**](model_card.md)
