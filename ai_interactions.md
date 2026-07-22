# AI Interactions Log

> **Stretch features only.** Only fill in the sections that apply to stretch features you attempted. If you did not attempt a stretch feature, leave its section blank or delete it. This file is not required for the core project.

---

## Agentic Workflow (SF8)

> Document your experience using an AI agent (e.g., Cursor Agent, Claude, Copilot) to make multi-step changes autonomously.

**What task did you give the agent?**

Add 5 or more complex attributes to the dataset that were not in the baseline
data (things like Release Decade and Detailed Mood Tags), then update both
`data/songs.csv` and the scoring logic in `src/recommender.py` so the scoring
actually uses them. This was a multi-step change: edit the data file, edit two
dataclasses, edit the loader, edit the scoring function, keep the OOP and
functional paths in sync, and add tests — all in one pass.

**Prompts used:**

Main prompt:

> "Add 5 or more complex attributes to the dataset that are not currently
> present, such as Song Popularity (0–100), Release Decade, or Detailed Mood
> Tags (e.g. 'nostalgic', 'aggressive', 'euphoric'). Update both
> `data/songs.csv` and the scoring logic in `src/recommender.py` so scoring
> accounts for the new attributes. Document the workflow in `ai_interactions.md`."

Follow-up guidance I gave during the work:

- "Make the new signals opt-in so the documented baseline output and existing
  tests don't change unless a profile actually asks for them."
- "Verify the math stays valid and run the tests before committing."

**What did the agent generate or change?**

Five new attributes were added (popularity already existed, so these are all new):

| Attribute | Type | CSV column | How it is scored |
|-----------|------|------------|------------------|
| `release_decade` | int (e.g. 1980, 2010) | `release_decade` | exact decade +1.0, one decade off +0.5, else 0 |
| `mood_tags` | list of tags | `mood_tags` (pipe-separated, e.g. `nostalgic\|dreamy`) | +1.0 × (fraction of the listener's wanted tags the song has) |
| `language` | text | `language` (e.g. `english`, `instrumental`) | exact match +1.0 |
| `instrumentalness` | float 0–1 | `instrumentalness` | continuous +1.0 × (1 − |song − target|) |
| `explicit` | bool | `explicit` (`true`/`false`) | −2.0 penalty when the listener opts out |

Files the agent edited:

- **`data/songs.csv`** — added the five columns and values for all 10 songs
  (decades from the 1980s–2020s, detailed mood tags, a mix of English and
  instrumental tracks, instrumentalness values, and one explicit track,
  *Storm Runner*, to exercise the penalty).
- **`src/recommender.py`** — new weight constants; five new fields on the `Song`
  dataclass and five optional preference fields on `UserProfile` (all with
  defaults); `load_songs()` updated to parse the new columns with safe fallbacks
  (plus helpers `_parse_tags()` and `_parse_bool()`); `score_song()` extended to
  score all five signals as **opt-in** terms; and `Recommender._score_song`
  updated to pass the new fields through so the OOP and functional paths agree.
- **`tests/test_recommender.py`** — 6 new regression tests (opt-in no-op,
  mood-tag partial credit, decade exact/adjacent/far, language match, continuous
  instrumentalness, explicit penalty).

Key design choice the agent followed: every new signal defaults to "indifferent"
and only scores when the profile supplies its key. This keeps the documented
sample output in `README.md` / `model_card.md` unchanged.

**What did you verify or fix manually?**

I ran the code and checked the numbers instead of trusting the explanation:

1. **Baseline unchanged.** The standard "High-Energy Pop" profile still returns
   *Sunrise City* at **5.78**, identical to the documented output — so the opt-in
   design worked.
2. **Tests pass.** `python -m pytest -q` → **14 passed** (8 original + 6 new).
3. **New signals fire correctly.** An advanced "Nostalgic 80s" profile pushed the
   right track (*Night Drive Loop*, an 1980s nostalgic dreamy English synthwave
   song) to #1 at **9.68**, with each new signal itemized in the explanation:

   ```text
   Night Drive Loop   9.68
     Genre match: synthwave (+2.0) | Mood match: moody (+1.0)
     | Energy similarity: 0.75 vs target 0.75 (+2.00)
     | Acoustic fit: 0.22 vs non-acoustic preference (+0.78)
     | Mood tags: matched ['dreamy', 'nostalgic'] (2/2) (+1.00)
     | Era match: 1980s (+1.00) | Language match: english (+1.00)
     | Instrumental fit: 0.40 vs target 0.30 (+0.90)
   ```
4. **Explicit penalty works.** With `allow_explicit=False`, *Storm Runner*
   (the one explicit track) dropped by exactly 2.0; a non-explicit song was not
   penalized.

Things I had to watch for / correct:

- The new dataclass fields needed **defaults**, or the existing tests that build
  `Song(...)` without them would crash.
- `load_songs()` needed **fallbacks** for missing columns so an older CSV still
  loads.
- I double-checked that the agent passed the new fields through
  `Recommender._score_song` — it's easy to add a field to `score_song()` and
  forget the OOP path, which would silently ignore the new features. Both paths
  return the same score for the same profile.

**How to reproduce:**

```bash
python -m pytest -q          # 14 passed
python -m src.main           # baseline + adversarial profiles (unchanged)
```

---

## Design Pattern (SF10)

> Document how AI helped you choose or implement a design pattern.

**Which design pattern did you use?**

The **Strategy pattern**. The goal was to let a user switch between multiple
ranking modes ("Genre-First," "Mood-First," "Energy-Focused," and the default
"Balanced") without rewriting the scoring code. The Strategy pattern fits exactly
this situation: it captures a family of interchangeable algorithms (here, the set
of weights applied to each signal) behind a common shape, so the behavior can be
swapped at runtime.

**How did AI help you brainstorm or implement it?**

I asked the AI to brainstorm a simple design pattern that would keep the code
modular while supporting several scoring modes. It weighed a few options and
recommended Strategy over the alternatives, with reasons:

- **Strategy (chosen)** — each mode is just a bundle of weights; the scoring
  function stays single-source and reads the weights from whichever strategy is
  passed in. Adding a new mode is one new object, no new branching.
- **If/elif branching (rejected)** — a `if mode == "genre-first": ...` chain
  inside `score_song()` would grow messy and force edits to the core function
  every time a mode is added.
- **Subclass-per-mode (rejected as overkill)** — a full class hierarchy with an
  overridden `score()` per mode would duplicate the shared scoring logic and risk
  the OOP and functional paths drifting apart again (a bug we had fixed earlier).

The AI's practical suggestion was a lightweight, frozen `ScoringStrategy`
dataclass holding the four core weights, plus named module-level instances and a
registry dict for CLI lookup. I liked that it kept one copy of the scoring math
and made "Balanced" reproduce the original numbers exactly, so nothing already
documented would change. I also had it inject the strategy into the `Recommender`
constructor (constructor injection) so the OOP path uses the same mechanism.

**How does the pattern appear in your final code?**

In `src/recommender.py`:

- `ScoringStrategy` (a frozen dataclass) is the strategy type — it holds
  `weight_genre`, `weight_mood`, `weight_energy`, `weight_acoustic`.
- `BALANCED`, `GENRE_FIRST`, `MOOD_FIRST`, `ENERGY_FOCUSED` are the concrete
  strategies, collected in the `STRATEGIES` registry.
- `score_song(user_prefs, song, strategy=BALANCED)` applies the chosen strategy's
  weights instead of hard-coded numbers. `recommend_songs(..., strategy=BALANCED)`
  passes it through, and `Recommender(songs, strategy=BALANCED)` stores it and
  uses it in `_score_song` (constructor injection).

In `src/main.py`:

- A user switches modes from the command line, e.g. `python -m src.main mood-first`
  or `python -m src.main compare` (which ranks one profile under every mode side
  by side). `_select_mode()` maps the CLI argument to a strategy via the registry.

**Verification:** `BALANCED` was confirmed to reproduce the exact baseline scores
(a unit test asserts `score_song(..., BALANCED) == score_song(...)` default), and
`compare` mode shows the modes genuinely reshuffle results — e.g. for a pop/happy
listener, Genre-First ranks *Gym Hero* (pop) above *Rooftop Lights*, while
Mood-First flips them (*Rooftop Lights* is the "happy" match). 18 tests pass.
