# Testing & Reliability

This project proves it works with four independent reliability methods, all
runnable by anyone with no API key (deterministic offline path):

1. **Automated tests** — `pytest` unit tests for the recommender and the RAG pipeline.
2. **Reliability metrics** — a harness that scores the pipeline and fails on regression.
3. **Logging & error handling** — every step is logged; a grounding guard and
   refusal/error handling degrade to the offline path instead of crashing.
4. **Manual evaluation** — labeled inputs (including edge cases) checked against
   explicit criteria, recorded in the parseable table below.

## Summary

> **32 of 32 automated tests pass. Manual evaluation: 6 of 6 inputs handled
> correctly, including empty and off-catalog edge cases. The reliability harness
> reports 1.00 on all 5 metrics.** Testing caught one real bug — the offline
> parser matched the genre "pop" inside the word "popular", so *"intense rock …
> popular hits"* was mis-parsed as pop; accuracy improved after adding
> word-boundary matching, now locked by a regression test. The off-catalog
> request (*"polka accordion music from Mars"*) is handled gracefully (no crash,
> grounded answer) but cannot be truly satisfied by a 10-song catalog — a known
> limitation, not a failure.

Reproduce everything:

```bash
pytest -q                        # 32 passed
python -m src.main reliability   # all metrics 1.00, RESULT: PASS (exit 0)
```

## 1. Automated tests (`pytest`)

| Suite | Tests | Result |
|-------|------:|--------|
| `tests/test_recommender.py` (scoring, modes, diversity, advanced features) | 21 | Pass |
| `tests/test_rag.py` (parsing, retrieval, grounding, determinism, reliability) | 11 | Pass |
| **Total** | **32** | **Pass** |

## 2. Reliability metrics (`python -m src.main reliability`)

Run over 6 labeled queries on the deterministic offline path. The command exits
non-zero if any metric drops below its threshold.

| Metric | What it checks | Score | Threshold | Result |
|--------|----------------|------:|----------:|--------|
| `parse_determinism` | same query parsed twice yields the identical profile | 1.00 | 1.00 | Pass |
| `parse_accuracy` | parsed fields match hand-labeled expectations | 1.00 | 0.90 | Pass |
| `retrieval_precision_at_1` | requested genre (when in catalog) is the #1 result | 1.00 | 1.00 | Pass |
| `grounding_rate` | generated answers name only retrieved songs | 1.00 | 1.00 | Pass |
| `e2e_determinism` | same query yields the identical final answer | 1.00 | 1.00 | Pass |

## 3. Logging & error handling

- Every pipeline step logs which engine ran (Claude vs offline), token usage, and
  any guardrail trip. Set `VIBEMATCH_DEBUG=1` to see INFO-level logs.
- The **grounding guard** discards any generated answer that names a song that was
  not retrieved (a hallucination) and falls back to the deterministic generator.
- API errors and safety refusals are caught and degrade to the offline path, so
  the `ask` command never crashes.

## 4. Manual evaluation

Each input was run through the pipeline and the output judged against explicit
criteria. Results are real and reproducible.

| Test Input | Evaluation Criteria | Result |
|------------|---------------------|--------|
| `"nostalgic 80s synthwave, nothing explicit, no lyrics"` | Parses synthwave + 1980s; top pick is the catalog's 80s synthwave track; answer grounded | Pass — Night Drive Loop (6.78), grounded |
| `"chill lofi for studying, instrumental"` | Parses lofi; top pick is lofi; answer grounded | Pass — Library Rain (7.04), grounded |
| `"intense rock for a workout, popular hits"` | Parses rock + wants-popular; top pick is rock | Pass (after fix) — Storm Runner (6.88); initially mis-parsed "pop" from "popular", fixed with word-boundary matching |
| `"niche jazz, relaxed and warm"` | Parses jazz + wants-niche; top pick is jazz | Pass — Coffee Shop Stories (6.72), grounded |
| `"polka accordion music from Mars"` (off-catalog) | No crash; degrades gracefully; answer grounded | Pass (graceful) — no genre match, ranks by energy/acoustic only (top score 2.28); cannot truly satisfy an off-catalog request (known limitation) |
| `""` (empty input) | Handles gracefully, no crash | Pass — prints a usage message, exits 0 |

## Machine-readable results

```json
{
  "automated_tests": { "passed": 32, "total": 32 },
  "reliability_metrics": {
    "parse_determinism": 1.0,
    "parse_accuracy": 1.0,
    "retrieval_precision_at_1": 1.0,
    "grounding_rate": 1.0,
    "e2e_determinism": 1.0
  },
  "manual_evaluation": [
    { "input": "nostalgic 80s synthwave, nothing explicit, no lyrics", "criteria": "parses synthwave+1980; top pick 80s synthwave; grounded", "result": "pass", "top_pick": "Night Drive Loop", "score": 6.78 },
    { "input": "chill lofi for studying, instrumental", "criteria": "parses lofi; top pick lofi; grounded", "result": "pass", "top_pick": "Library Rain", "score": 7.04 },
    { "input": "intense rock for a workout, popular hits", "criteria": "parses rock + wants-popular; top pick rock", "result": "pass_after_fix", "top_pick": "Storm Runner", "score": 6.88, "note": "initial substring bug fixed with word-boundary matching" },
    { "input": "niche jazz, relaxed and warm", "criteria": "parses jazz + wants-niche; top pick jazz", "result": "pass", "top_pick": "Coffee Shop Stories", "score": 6.72 },
    { "input": "polka accordion music from Mars", "criteria": "no crash; graceful degradation; grounded", "result": "pass_graceful", "top_pick": "Night Drive Loop", "score": 2.28, "note": "off-catalog request cannot be truly satisfied by a 10-song catalog" },
    { "input": "", "criteria": "handles gracefully, no crash", "result": "pass", "note": "prints usage message, exits 0" }
  ]
}
```
