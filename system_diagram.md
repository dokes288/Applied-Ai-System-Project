# VibeMatch — System Diagram

A view of how the project is organized: the main components, how data flows
(input → process → output), and where humans and automated testing check the AI
results. The source below is Mermaid, so the structure is readable directly from
the file (and renders automatically on GitHub).

```mermaid
flowchart TD
    %% ---------- Input ----------
    U["👤 User<br/>free-text request<br/>e.g. &quot;nostalgic 80s synthwave, nothing explicit&quot;"]
    U -->|"INPUT: natural language"| CLI["CLI entrypoint<br/>src/main.py · ask mode"]

    %% ---------- RAG pipeline (PROCESS) ----------
    subgraph RAG["RAG Pipeline · src/rag.py  (PROCESS)"]
        direction TB
        P{"1 · PARSE<br/>Claude available?"}
        P -->|"yes: Claude API<br/>claude-opus-5"| PC["claude_parse<br/>structured output"]
        P -->|"no key / SDK absent"| PO["offline_parse<br/>deterministic keyword rules"]
        PC --> V["🛡️ GUARDRAIL<br/>validate + clamp profile"]
        PO --> V

        V --> R["2 · RETRIEVE<br/>recommend_songs()"]

        R --> G{"3 · GENERATE<br/>Claude available?"}
        G -->|"yes: Claude API"| GC["claude_generate<br/>grounded-only prompt"]
        G -->|"no key / SDK absent"| GO["offline_generate<br/>deterministic template"]
        GC --> GG{"🛡️ GROUNDING GUARD<br/>answer names only<br/>retrieved songs?"}
        GG -->|"yes"| ANS["grounded recommendation"]
        GG -->|"no — hallucination:<br/>discard LLM answer"| GO
        GO --> ANS
    end

    CLI --> P

    %% ---------- Retriever data source ----------
    subgraph DATA["Retriever source (DATA)"]
        ENG["Content-based scorer<br/>src/recommender.py<br/>score → sort → explain"]
        CSV[("data/songs.csv<br/>10-song catalog +<br/>features & reasons")]
        ENG --> CSV
    end
    R <-->|"top-k songs + reason strings"| ENG

    %% ---------- Output ----------
    ANS -->|"OUTPUT: grounded answer"| OUT["👤 User reads &amp; judges<br/>the recommendation"]

    %% ---------- Evaluation & Testing (human-in-the-loop check) ----------
    subgraph EVAL["Evaluation &amp; Testing  (no API key needed — deterministic offline path)"]
        direction TB
        REL["Reliability harness<br/>src/reliability.py<br/>parse-accuracy · retrieval P@1 ·<br/>grounding · determinism · thresholds"]
        PT["pytest<br/>tests/test_rag.py ·<br/>tests/test_recommender.py"]
        GATE{"all metrics ≥ threshold?"}
        REL --> GATE
        GATE -->|"yes"| PASS["PASS (exit 0)"]
        GATE -->|"no"| FAIL["FAIL (exit 1) — blocks"]
        DEV["👤 Developer reviews<br/>PASS/FAIL report + test run"]
        PASS --> DEV
        FAIL --> DEV
        PT --> DEV
    end

    RAG -. "offline path measured by" .-> REL
    RAG -. "unit-tested by" .-> PT

    %% ---------- styling ----------
    classDef human fill:#fde68a,stroke:#b45309,color:#111827;
    classDef guard fill:#bbf7d0,stroke:#15803d,color:#111827;
    classDef data  fill:#e0e7ff,stroke:#4338ca,color:#111827;
    class U,OUT,DEV human;
    class V,GG guard;
    class ENG,CSV data;
```

## How to read it

- **Input → Process → Output.** A natural-language request enters through the
  CLI, flows through the three RAG stages (**parse → retrieve → generate**), and
  leaves as a grounded recommendation the user reads.
- **Main components.**
  - **Retriever** — `src/recommender.py` scoring the `data/songs.csv` catalog and
    returning top‑k songs *with their reason strings*.
  - **AI parse/generate (the "agent" steps)** — `src/rag.py` calls Claude
    (`claude-opus-5`) when a key is present, or a deterministic offline
    implementation otherwise, so the system is reproducible with no key.
  - **Evaluator / tester** — `src/reliability.py` (metrics + a pass/fail gate)
    and `pytest`.
- **Where AI results are checked (🛡️ / 👤).**
  - **Grounding guard** — automatically verifies the generated answer names only
    songs that were actually retrieved; a hallucinated answer is discarded and
    the deterministic generator is used instead.
  - **Profile guardrail** — validates/clamps the parsed profile before scoring.
  - **Reliability gate + pytest** — measure the pipeline and block on regression.
  - **Humans** — the user judges the final recommendation; the developer reviews
    the PASS/FAIL report and test results.
