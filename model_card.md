# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

**VibeMatch 1.0** — a rule-based, content-filtering music recommender for classroom use.

---

## 2. Intended Use

This recommender is designed for **education and demonstration**. It ranks songs from a small CSV catalog against a fictional user profile and returns explainable results.

**Appropriate uses:**

- Learning how content-based recommendation works
- Experimenting with feature weights and user profiles
- Discussing bias, limitations, and comparison to production systems

**Not intended for:**

- Production deployment
- Real listener personalization at scale
- Legal, medical, or high-stakes decision making

---

## 3. How the Model Works

VibeMatch is a **weighted rule-based scorer**, not a trained machine-learning model.

1. **Input:** a user profile (favorite genre, mood, target energy, acoustic preference) and a song catalog.
2. **Scoring:** each song earns points for matching genre, mood, energy, valence, tempo, and acoustic profile.
3. **Output:** the top *k* songs sorted by total score, with human-readable reasons.

### Scoring breakdown

| Signal | Max points | Logic |
|--------|------------|-------|
| Genre match | 3.5 | Exact or substring match (e.g. `"pop"` in `"indie pop"`) |
| Mood match | 3.0 | Same matching rule as genre |
| Energy fit | 2.5 | Decays as `\|song.energy − target_energy\|` grows |
| Valence fit | 1.0 | Mood maps to expected valence; decay on gap |
| Tempo fit | 0.5 | Expected BPM derived from target energy |
| Acoustic fit | 1.0–1.5 | Bonus for high acousticness if user likes acoustic, or low acousticness if not |

### Architecture

- **OOP API:** `Recommender.recommend(UserProfile, k)`
- **Functional API:** `recommend_songs(dict, songs, k)`
- Both call shared `_compute_score()` to avoid inconsistent results.

---

## 4. Data

| Source | Description |
|--------|-------------|
| `data/songs.csv` | 10 fictional tracks with genre, mood, energy, tempo, valence, danceability, acousticness |

**Genres represented:** pop, lofi, rock, ambient, jazz, synthwave, indie pop.

**Not included:** real user listening logs, lyrics, audio waveforms, artist popularity, geographic or demographic data.

---

## 5. Strengths

- **Transparent:** every point has an explanation string.
- **Fast:** linear scan over a tiny catalog; no training step.
- **Flexible:** weights at top of `recommender.py` are easy to tune.
- **Cold-start friendly (for songs):** new songs can be scored immediately from metadata.
- **Good for clear profiles:** "happy pop at high energy" produces sensible top picks.

---

## 6. Limitations and Bias

### Technical limitations

- Catalog size is fixed at 10 songs.
- `danceability` is loaded but not yet used in scoring.
- No learning from feedback (likes/skips do not update the model).
- Genre/mood labels are coarse and subjective.

### Bias risks

| Risk | Example in this system |
|------|--------------------------|
| **Popularity proxy bias** | Not modeled here, but in production CF favors heavily streamed tracks. |
| **Genre stereotyping** | Mood→valence mapping assumes "happy = high valence" for all users. |
| **Filter bubble** | Strong genre/mood weights keep users in one lane. |
| **Representation bias** | Small handcrafted catalog may omit languages, regions, and artist diversity. |
| **Acoustic binary** | `likes_acoustic` is boolean; real taste is more nuanced. |

### Comparison to production systems

Real platforms combine this kind of **content filtering** with **collaborative filtering** (playlists, co-listening, skips) and **context** (time, activity). VibeMatch intentionally isolates the content-based slice so the mechanics stay visible.

---

## 7. Evaluation

### Automated tests (`pytest`)

- Pop/happy/high-energy user → pop happy track ranks first.
- `explain_recommendation()` returns a non-empty string.

### Manual profile checks

| Profile | Expected top results | Observed |
|---------|---------------------|----------|
| pop / happy / energy 0.8 | *Sunrise City*, *Rooftop Lights* | ✓ Strong genre+mood+energy alignment |
| lofi / chill / acoustic | *Midnight Coding*, *Library Rain* | ✓ Genre, mood, acoustic bonuses align |
| rock / intense / energy 0.9 | *Storm Runner*, *Gym Hero* | ✓ High energy intense/rock tracks rise |

### What it gets wrong

- Cannot recommend outside the 10-song catalog.
- *Gym Hero* ranks for happy pop users because genre matches even though mood is "intense" not "happy."
- Adjacent-genre discovery (e.g. synthwave for a pop fan) only happens via partial string match, not true similarity.

---

## 8. Future Work

- Incorporate `danceability` and artist-level similarity.
- Add lightweight collaborative filtering from fake multi-user play history.
- Diversity penalty so top-5 results are not all the same genre.
- User feedback loop: adjust weights after simulated likes/skips.
- Optional Streamlit UI for interactive profile tuning.

---

## 9. Personal Reflection

Building VibeMatch made the recommendation pipeline concrete: **features in, score out, rank, explain**. Production systems add billions of behavioral events and neural embeddings, but the output is still a ranked list shaped by design choices someone made.

The exercise also surfaced fairness questions. When rules favor obvious metadata matches, users with eclectic or cross-genre taste may be poorly served. When catalogs are small or homogenous, whole styles of music never appear. Those issues exist in this toy system on purpose — they are the same categories of harm that matter at scale on Spotify or YouTube, where data volume amplifies both discovery and bias.
