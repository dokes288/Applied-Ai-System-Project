flowchart TD
    U["👤 User\nfree-text request"] --> CLI["CLI entrypoint\nsrc/main.py"]

    subgraph PIP["RAG pipeline"]
        direction TB
        P{"1 · PARSE"}
        P -->|Claude available| CP["Claude parse\nstructured profile"]
        P -->|No key / offline| OP["Offline parse\ndeterministic parser"]
        CP --> V["🛡️ Validate + clamp profile"]
        OP --> V

        V --> R["2 · RETRIEVE\nrecommend_songs()"]
        R --> G{"3 · GENERATE"}
        G -->|Claude available| CG["Claude generate\ngrounded prompt"]
        G -->|No key / offline| OG["Offline generate\ntemplate-based answer"]
        CG --> GG{"🛡️ Grounding guard"}
        OG --> GG
        GG -->|Allowed| A["Grounded recommendation"]
        GG -->|Hallucinated song| OG
    end

    CLI --> P

    subgraph DATA["Retriever data"]
        direction LR
        ENG["Content-based scorer\nsrc/recommender.py"]
        CSV[("data/songs.csv")]
        ENG --> CSV
    end

    R <-->|top-k songs + reasons| ENG
    A --> OUT["👤 User reads & judges result"]

    subgraph TEST["Evaluation & testing"]
        direction TB
        REL["Reliability harness\nsrc/reliability.py"]
        PY["pytest\ntests/test_rag.py + tests/test_recommender.py"]
        DEV["👤 Developer reviews PASS/FAIL report"]
        REL --> DEV
        PY --> DEV
    end

    R -.-> REL
    PIP -.-> PY
