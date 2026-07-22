# 🎧 Model Card: VibeMatch

## Model Name

**VibeMatch 1.0** — a "match your vibe" music recommender for the classroom.

It is a small, rule-based demo. It is not a trained AI model.

---

## Goal / Task

VibeMatch tries to guess which songs a person will like.

You give it a short taste profile (your genre, mood, energy level, and whether you like acoustic music). It looks at a small song list and picks the 5 songs that best fit. For each pick, it also shows the reasons it chose that song.

The point is to show, at a tiny scale, how a real recommender works: turn preferences into a score, sort by score, and explain the result.

---

## Data Used

All songs come from one file: `data/songs.csv`.

- **Size:** 10 made-up songs. That is the whole catalog.
- **Each song has:** title, artist, genre, mood, energy, tempo, valence, danceability, acousticness, and popularity.
- **Genres in the list:** pop, lofi, rock, ambient, jazz, synthwave, indie pop.
- **Used in scoring:** genre, mood, energy, acousticness, and popularity (popularity only when the user asks for it).
- **Stored but not used:** tempo, valence, and danceability. They are in the file but the scorer ignores them.

**Limits:**

- The catalog is tiny, so there is not much to discover.
- There are no real users and no listening history (no likes, skips, or replays).
- There are no lyrics and no real audio.
- The popularity numbers are invented for the demo, not real chart data.

---

## Algorithm Summary

VibeMatch gives each song points. More points means a better match. It adds up the points, sorts the songs from high to low, and returns the top 5.

A song can earn points from four things (plus one optional one):

- **Genre — up to 2 points.** The song gets 2 points only if its genre is the exact word you picked. "pop" does not count for "indie pop". No match means 0.
- **Mood — up to 1 point.** Same idea: 1 point only for an exact mood match, else 0.
- **Energy — up to 2 points.** The closer the song's energy is to your target energy, the more points. A perfect match gets the full 2. A big gap gets close to 0.
- **Acoustic feel — up to 1 point.** If you like acoustic music, acoustic songs score high. If you don't, electronic songs score high. The closer the fit, the more points.
- **Popularity — optional, up to 1 point.** Off by default. If you say you want popular songs, a popular song adds 1 point. If you say you want niche songs, an obscure song adds 0.75. If you don't state a preference, this adds nothing.

The most a song can score is 6 points normally, or 7 if you turn on the popularity preference.

Genre and mood are simple yes/no checks. Energy and acoustic feel are "how close" checks that give partial points.

---

## Advanced Features / Modes / Diversity

Three optional layers sit on top of the basic recipe. All three are **opt-in**:
if a listener does not use them, the results are exactly the same as before. This
keeps the sample output in this card and the README correct.

### Advanced song features

We added five richer attributes to each song, and the scorer uses them **only
when the listener asks for them**:

- **Release decade** (e.g. 1980, 2010) — bonus for the right era.
- **Detailed mood tags** (e.g. "nostalgic", "dreamy", "aggressive") — partial
  credit for how many of the listener's wanted tags a song has.
- **Language** (e.g. English, instrumental) — bonus for an exact match.
- **Instrumentalness** (0–1) — bonus for being close to the listener's target.
- **Explicit flag** — a penalty when the listener asks to avoid explicit songs.

These let the recommender handle much more specific tastes, like "nostalgic 80s
English synthwave, no explicit tracks."

### Scoring modes

The listener can switch between four ranking modes that change how much each
signal counts:

| Mode | What it favors |
|------|----------------|
| Balanced (default) | the original recipe |
| Genre-First | staying inside the chosen genre |
| Mood-First | matching the vibe over the genre |
| Energy-Focused | matching intensity (good for workouts/focus) |

Run them with `python -m src.main <mode>`, or `python -m src.main compare` to see
one listener ranked by every mode. Under the hood this uses the **Strategy design
pattern** (see `ai_interactions.md`), so adding a new mode does not require
changing the scoring code.

### Diversity / fairness penalty

By default the recommender can return several songs by the same artist or in the
same genre. The optional **diversity penalty** prevents that. As the top list is
built, a song loses points for each song already on the list that shares its
artist (−1.5 each) or genre (−0.75 each). This is a re-ranking step, so it looks
at what has already been chosen — something the normal per-song score cannot do.

Run `python -m src.main diversity` to see each listener's top 5 with and without
the penalty. Example: for a lofi/chill listener, the plain top 3 are all lofi
(two by the same artist, *LoRoom*); with the penalty, one of those drops out and
an ambient track and a jazz track take its place. The trade-off is that a
slightly lower-scoring song can outrank a higher one to keep the list varied —
which is the point of a fairness rule.

### Readable table output

`python -m src.main table` prints the top picks as a formatted table that
includes the full reasons for each score. It uses the `tabulate` library when
available and falls back to a plain-ASCII table otherwise.

---

## Observed Behavior / Biases

We noticed several clear patterns. They are not bugs. They come straight from the simple rules.

- **Genre counts more than mood.** Genre is worth 2 points and mood is worth 1. So a song in the right genre but the wrong mood can beat a song in the right mood but the wrong genre.
- **It only understands exact words.** If your genre is not an exact word in the catalog, you get 0 genre points. A metal fan gets nothing, even though "rock" is very close to metal. The system has no idea two genres are related.
- **The energy score has a blind spot in the middle.** The songs are either low energy or high energy, with a gap in between. So someone who wants medium energy can never get a great energy match. Low-energy and high-energy tastes are served well; medium tastes are not.
- **It is blind to the "feel" of a song.** It does not use valence (happy vs. dark sound), tempo, or danceability. So a bright cheerful song and a dark moody song can score exactly the same.
- **It can push everyone toward the same few songs.** With few signals and only 10 songs, many people end up with overlapping lists. That is a small version of a "filter bubble."

The clearest example is the "Gym Hero" pattern, explained in the next section.

---

## Evaluation Process

We tested VibeMatch three ways: everyday profiles, edge-case profiles, and a weight experiment. We also compared profiles side by side.

**Automated tests (`pytest`):** 8 tests pass. They check that a pop/happy/high-energy listener gets a pop happy song first, that explanations are never blank, that acoustic fans get acoustic songs, that out-of-range energy never breaks the score, and that the popularity preference works.

**Everyday profiles we ran** (full output is in `README.md`):

| Profile | What the listener asked for | Top 5 (score) |
|---------|------------------------------|---------------|
| **High-Energy Pop** | pop, happy, energy 0.8, non-acoustic | Sunrise City (5.78), Gym Hero (4.69), Rooftop Lights (3.57), Night Drive Loop (2.68), Storm Runner (2.68) |
| **Chill Lofi** | lofi, chill, energy 0.4, acoustic | Library Rain (5.76), Midnight Coding (5.67), Focus Flow (4.78), Spacewalk Thoughts (3.68), Coffee Shop Stories (2.83) |
| **Deep Intense Rock** | rock, intense, energy 0.9, non-acoustic | Storm Runner (5.88), Gym Hero (3.89), Sunrise City (2.66), Night Drive Loop (2.48), Rooftop Lights (2.37) |
| **Off-Catalog Metal** | metal, angry, energy 0.85, non-acoustic | Gym Hero (2.79), Storm Runner (2.78), Sunrise City (2.76), Night Drive Loop (2.58), Rooftop Lights (2.47) |

**What surprised us:**

- A song tagged "intense" (*Gym Hero*) keeps ranking high for people who asked for "happy."
- The metal fan gets served cheerful pop, because there is no metal in the list.
- Some ranks are decided by the alphabet. *Night Drive Loop* and *Storm Runner* tie at 2.68, so the order comes from the song title, not the music.
- The rock fan and the metal fan want almost the same thing but get very different results.

**Comparing profiles two at a time** (what changed, and why it makes sense):

- **High-Energy Pop vs. Chill Lofi.** Opposite tastes, and their top 5s share zero songs. The pop fan wants loud and electronic and gets bright synth-pop. The lofi fan wants calm and acoustic and gets soft mellow tracks. This makes sense: they asked for opposite energy and opposite acoustic feel, which are the two things that most separate songs.
- **High-Energy Pop vs. Deep Intense Rock.** Both want loud, non-acoustic music, so they pull from the same pool of loud songs — just in a different order. The genre/mood label breaks the tie: *Sunrise City* (pop, happy) leads for the pop fan, and *Storm Runner* (rock, intense) leads for the rock fan.
- **Chill Lofi vs. Deep Intense Rock.** These are near mirror images: calm-and-acoustic versus loud-and-electronic. No shared songs. Flipping those two settings sends the listener to the opposite half of the catalog.
- **Deep Intense Rock vs. Off-Catalog Metal.** Both want loud and aggressive music. The only real difference is the words. "rock" and "intense" exist in the catalog, so *Storm Runner* wins clearly at 5.88. "metal" and "angry" match nothing, so the same *Storm Runner* drops to #2 and every score falls below 3.0. Same taste, very different result, only because of vocabulary.
- **High-Energy Pop vs. Off-Catalog Metal.** For the pop fan the labels match, so the top pick is a confident 5.78. For the metal fan nothing matches, so the top four are all within 0.2 points — the ranking gets shaky when the labels do not apply.
- **Wants Popular vs. Wants Niche.** Same base taste, opposite popularity switch. "Wants popular" lifts chart-toppers (*Sunrise City* to 6.78). "Wants niche" lifts obscure tracks (*Library Rain* to 6.51). Middle-of-the-road songs get no boost either way.

**Weight experiment (sensitivity test):** We doubled the energy weight and halved the genre weight. It changed half the rankings, so the system is clearly sensitive to its weights. But it did not fix the metal problem — it just handed #1 to whichever song's energy was nearest the target (a bright pop song). Lesson: no amount of energy weighting can replace the missing "feel" signals.

### Why "Gym Hero" keeps showing up for "Happy Pop" (plain language)

Think of the recommender as a checklist with four boxes. A song earns points for each box it ticks:

1. **Genre** — is it the genre you wanted? (2 points)
2. **Mood** — is it the mood you wanted? (1 point)
3. **Energy** — is its energy close to yours? (up to 2 points)
4. **Acoustic feel** — does it match your acoustic taste? (up to 1 point)

The "Happy Pop" listener wants pop, happy, high-energy, non-acoustic. Now look at *Gym Hero*:

- It is a **pop** song. Ticks the genre box (2 points). ✅
- It is **very high energy**. Nearly full energy points. ✅
- It is **very electronic**. Nearly full acoustic points. ✅
- Its mood is **"intense," not "happy."** Misses the 1 mood point. ❌

So *Gym Hero* ticks three of four boxes and misses only the smallest one. Three strong matches easily beat one small miss, so it lands at #2.

The big takeaway: **the system counts the boxes separately and never asks if they make sense together.** It does not know that a "happy" song and an "intense" workout song feel different. Because "happy" is only 1 point out of 6, a high-energy pop song can score high for a happy-pop fan even when it does not really sound happy. That is not a coding mistake — it is just what happens when mood is one small yes/no box.

### Edge-case profiles

These profiles are built to try to trick the scorer. They live in `ADVERSARIAL_PROFILES` in `src/main.py` and print after the everyday profiles when you run `python -m src.main`. Below is the real output for each.

**1. Conflicting preferences — a chill mood with very high energy.** Nothing stops the contradiction. The "chill" mood still scores, but the high energy target fights it, so the winners land about a point lower than a normal chill profile.

```text
====================================================================
 USER PROFILE: Conflicting: Chill Mood + High Energy
 (genre=lofi, mood=chill, energy=0.95, likes_acoustic=True)
====================================================================

1. Library Rain — Paper Lanterns
   Score: 4.66
   Reasons:
     - Genre match: lofi (+2.0)
     - Mood match: chill (+1.0)
     - Energy similarity: 0.35 vs target 0.95 (+0.80)
     - Acoustic fit: acousticness 0.86 vs acoustic preference (+0.86)

2. Midnight Coding — LoRoom
   Score: 4.65
   Reasons:
     - Genre match: lofi (+2.0)
     - Mood match: chill (+1.0)
     - Energy similarity: 0.42 vs target 0.95 (+0.94)
     - Acoustic fit: acousticness 0.71 vs acoustic preference (+0.71)

3. Focus Flow — LoRoom
   Score: 3.68
   Reasons:
     - Genre match: lofi (+2.0)
     - Energy similarity: 0.40 vs target 0.95 (+0.90)
     - Acoustic fit: acousticness 0.78 vs acoustic preference (+0.78)

4. Spacewalk Thoughts — Orbit Bloom
   Score: 2.58
   Reasons:
     - Mood match: chill (+1.0)
     - Energy similarity: 0.28 vs target 0.95 (+0.66)
     - Acoustic fit: acousticness 0.92 vs acoustic preference (+0.92)

5. Storm Runner — Voltline
   Score: 2.02
   Reasons:
     - Energy similarity: 0.91 vs target 0.95 (+1.92)
     - Acoustic fit: acousticness 0.10 vs acoustic preference (+0.10)
```

**2. Out-of-range input — energy typed on a 0–100 scale (2.0).** The safety check holds. The target is clamped to 1.00 and the energy points can never go negative, so the genre and mood points survive. Every score stays positive.

```text
====================================================================
 USER PROFILE: Out-of-Range Energy (0-100 scale mixup)
 (genre=pop, mood=happy, energy=2.0, likes_acoustic=False)
====================================================================

1. Sunrise City — Neon Echo
   Score: 5.46
   Reasons:
     - Genre match: pop (+2.0)
     - Mood match: happy (+1.0)
     - Energy similarity: 0.82 vs target 1.00 (+1.64)
     - Acoustic fit: acousticness 0.18 vs non-acoustic preference (+0.82)

2. Gym Hero — Max Pulse
   Score: 4.81
   Reasons:
     - Genre match: pop (+2.0)
     - Energy similarity: 0.93 vs target 1.00 (+1.86)
     - Acoustic fit: acousticness 0.05 vs non-acoustic preference (+0.95)

3. Rooftop Lights — Indigo Parade
   Score: 3.17
   Reasons:
     - Mood match: happy (+1.0)
     - Energy similarity: 0.76 vs target 1.00 (+1.52)
     - Acoustic fit: acousticness 0.35 vs non-acoustic preference (+0.65)

4. Storm Runner — Voltline
   Score: 2.72
   Reasons:
     - Energy similarity: 0.91 vs target 1.00 (+1.82)
     - Acoustic fit: acousticness 0.10 vs non-acoustic preference (+0.90)

5. Night Drive Loop — Neon Echo
   Score: 2.28
   Reasons:
     - Energy similarity: 0.75 vs target 1.00 (+1.50)
     - Acoustic fit: acousticness 0.22 vs non-acoustic preference (+0.78)
```

**3. Wants popular.** Same as High-Energy Pop, but now asking for popular songs. Popular tracks (popularity 70 or higher) get an extra point. *Sunrise City* jumps to 6.78. *Night Drive Loop* (popularity 64) gets nothing, so the cutoff is a hard line.

```text
====================================================================
 USER PROFILE: Wants Popular (prefers_popular=True)
 (genre=pop, mood=happy, energy=0.8, likes_acoustic=False, prefers_popular=True)
====================================================================

1. Sunrise City — Neon Echo
   Score: 6.78
   Reasons:
     - Genre match: pop (+2.0)
     - Mood match: happy (+1.0)
     - Energy similarity: 0.82 vs target 0.80 (+1.96)
     - Acoustic fit: acousticness 0.18 vs non-acoustic preference (+0.82)
     - Popularity match: 92 vs popular preference (+1.00)

2. Gym Hero — Max Pulse
   Score: 5.69
   Reasons:
     - Genre match: pop (+2.0)
     - Energy similarity: 0.93 vs target 0.80 (+1.74)
     - Acoustic fit: acousticness 0.05 vs non-acoustic preference (+0.95)
     - Popularity match: 88 vs popular preference (+1.00)

3. Rooftop Lights — Indigo Parade
   Score: 4.57
   Reasons:
     - Mood match: happy (+1.0)
     - Energy similarity: 0.76 vs target 0.80 (+1.92)
     - Acoustic fit: acousticness 0.35 vs non-acoustic preference (+0.65)
     - Popularity match: 71 vs popular preference (+1.00)

4. Storm Runner — Voltline
   Score: 3.68
   Reasons:
     - Energy similarity: 0.91 vs target 0.80 (+1.78)
     - Acoustic fit: acousticness 0.10 vs non-acoustic preference (+0.90)
     - Popularity match: 78 vs popular preference (+1.00)

5. Night Drive Loop — Neon Echo
   Score: 2.68
   Reasons:
     - Energy similarity: 0.75 vs target 0.80 (+1.90)
     - Acoustic fit: acousticness 0.22 vs non-acoustic preference (+0.78)
```

**4. Wants niche.** Same as Chill Lofi, but now asking for obscure songs. Niche tracks (popularity 30 or lower) get 0.75 points. *Library Rain* climbs to 6.51. Middle-popularity songs like *Midnight Coding* and *Focus Flow* get nothing.

```text
====================================================================
 USER PROFILE: Wants Niche (prefers_popular=False)
 (genre=lofi, mood=chill, energy=0.4, likes_acoustic=True, prefers_popular=False)
====================================================================

1. Library Rain — Paper Lanterns
   Score: 6.51
   Reasons:
     - Genre match: lofi (+2.0)
     - Mood match: chill (+1.0)
     - Energy similarity: 0.35 vs target 0.40 (+1.90)
     - Acoustic fit: acousticness 0.86 vs acoustic preference (+0.86)
     - Niche pick: popularity 22 vs niche preference (+0.75)

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
   Score: 4.43
   Reasons:
     - Mood match: chill (+1.0)
     - Energy similarity: 0.28 vs target 0.40 (+1.76)
     - Acoustic fit: acousticness 0.92 vs acoustic preference (+0.92)
     - Niche pick: popularity 18 vs niche preference (+0.75)

5. Coffee Shop Stories — Slow Stereo
   Score: 3.58
   Reasons:
     - Energy similarity: 0.37 vs target 0.40 (+1.94)
     - Acoustic fit: acousticness 0.89 vs acoustic preference (+0.89)
     - Niche pick: popularity 27 vs niche preference (+0.75)
```

---

## Intended Use and Non-Intended Use

**What it is for:**

- Learning how content-based recommendation works.
- Playing with weights and taste profiles to see what changes.
- Talking about bias, fairness, and limits in a safe, tiny example.

**What it is not for:**

- Real apps or real listeners.
- Personalizing music at any real scale.
- Any important decision (money, health, legal, hiring, and so on).
- Judging or ranking real artists.

This is a teaching toy. The biases are here on purpose so they are easy to see.

---

## Ideas for Improvement

If we kept building VibeMatch, we would start here:

1. **Score the "feel" of a song.** Use valence and tempo so mood actually counts. This would stop a metal fan from being handed bright, cheerful pop.
2. **Add genre similarity.** Let close genres earn partial credit (metal is close to rock, indie pop is close to pop). Right now an unknown genre word scores nothing.
3. **Add a diversity rule (and a bigger catalog).** Stop the top 5 from being all one genre or one artist, and add more songs so there is more to discover — including some medium-energy tracks to fill the current gap.

---

## Personal Reflection

**My biggest learning moment.** The scoring rules are design choices, not facts. Small changes moved the results in ways I did not expect. Doubling the energy weight was supposed to help the metal fan. Instead it handed the top spot to a cheerful pop song. That taught me that a recommender is only as good as the signals it looks at. If it never reads the "feel" of a song, no weighting can fix that.

**How AI tools helped, and when I checked them.** AI helped me move fast. It spotted a dead setting (`prefers_popular`) that looked wired up but did nothing, and it drafted the fix, the tests, and the plain-language write-ups. But I learned not to trust it blindly. When it wanted to wire popularity in, we found the song file had no popularity column at all, so the "fix" would still have done nothing until we added real data. It also gave me a terminal command that broke because of quote characters, and the output showed a scrambled dash that was really just an encoding setting. So the rule I landed on: let AI draft, then run the code, read the actual output, and confirm the numbers myself. `pytest` and real runs were the truth, not the explanation.

**What surprised me about simple algorithms.** The math here is just adding a few numbers. There is no learning and no AI model. Yet the output really does feel like a recommendation. It ranks songs, and it explains itself in a way that sounds reasonable. That was the eye-opener: "score, sort, explain" is enough to feel smart, even when the system does not understand music at all. It also means the flaws feel reasonable too, like "Gym Hero" showing up for a happy-pop fan, which makes them easy to miss.

**What I would try next.** I would score valence and tempo first, because that is the missing piece behind most of the odd results. Then I would add genre similarity so close genres get partial credit. After that, a diversity rule and a bigger, more varied catalog. Longer term, I would like to add a little bit of feedback (likes and skips) so the system can actually learn, instead of only matching labels.
