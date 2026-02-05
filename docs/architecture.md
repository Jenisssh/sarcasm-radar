# Architecture

A one-page mental model of the system.

## Component layout

```
                              ┌─────────────────────┐
                              │  HuggingFace        │
                              │  iSarcasm + SemEval │
                              └──────────┬──────────┘
                                         │ scripts/download_data.py
                                         ▼
              ┌─────────────────────────────────────┐
              │  data/raw/                          │
              │  ─ isarcasm.parquet                 │
              │  ─ semeval_isarcasm.parquet         │
              │  (gitignored)                       │
              └─────────────────────────────────────┘
                                         │
                                         │           data/curated/indian_english.csv
                                         │                 │ (committed seed labels)
                                         ▼                 ▼
                              ┌─────────────────────────────┐
                              │  sarcasm_radar.data.load    │
                              │  ─ load_isarcasm            │
                              │  ─ load_semeval_isarcasm    │
                              │  ─ load_curated_supplement  │
                              │  ─ merge_corpora (dedup)    │
                              └──────────────┬──────────────┘
                                             │
                              ┌──────────────▼──────────────┐
                              │  data.clean                 │
                              │  URLs, mentions, hashtags,  │
                              │  repeats, HTML, whitespace  │
                              └──────────────┬──────────────┘
                                             │
            ┌────────────────────────────────┼────────────────────────────────┐
            │                                │                                │
   ┌────────▼─────────┐          ┌───────────▼──────────┐         ┌──────────▼─────────┐
   │ models.baseline  │          │ models.transformer   │         │ models.multilingual│
   │ TfidfVectorizer  │          │ DistilBERT           │         │ XLM-RoBERTa preset │
   │ + LogReg(bal)    │          │ + HF Trainer         │         │ + per-register     │
   │                  │          │                      │         │ comparison         │
   └────────┬─────────┘          └───────────┬──────────┘         └──────────┬─────────┘
            │                                │                               │
            └────────────────────────────────┼───────────────────────────────┘
                                             │
                              ┌──────────────▼──────────────┐
                              │  evaluation.metrics         │
                              │  macro-F1, per-class P/R/F1,│
                              │  confusion matrix           │
                              └──────────────┬──────────────┘
                                             │
                              ┌──────────────▼──────────────┐
                              │  models.inference           │
                              │  ─ ModelArtifacts (frozen)  │
                              │  ─ load_artifacts (dispatch)│
                              │  ─ predict_single / batch   │
                              └──────────────┬──────────────┘
                                             │
                  ┌──────────────────────────┼──────────────────────────┐
                  │                                                     │
       ┌──────────▼──────────┐                                  ┌──────▼──────────┐
       │ api/main.py FastAPI │                                  │ app/streamlit_  │
       │ /health /predict    │                                  │ app.py (Plotly  │
       │ /explain (LIME)     │ ◄────────────── HTTP ─────────── │ gauge + token   │
       └─────────────────────┘                                  │ weight bar)     │
                                                                └─────────────────┘
```

## Request paths

**`POST /predict`**

```
client ─► PredictRequest (Pydantic, extra='forbid', text 1..4000 chars)
            │ 422 on bad payload
            ▼
        inference.predict_single
            │ dispatches on artifacts.kind
            ▼
        baseline | transformer  ──► P(sarcastic)
            │
            ▼
        decision = P >= threshold ? 'SARCASTIC' : 'NOT_SARCASTIC'
            │
            ▼
        PredictResponse JSON
```

**`POST /explain`**

```
client ─► PredictRequest
            │
            ▼
        LimeTextExplainer(class_names=["not_sarcastic", "sarcastic"])
        explain_instance(text, predict_fn, num_samples=200, num_features=10)
            │
            ▼
        as_list(label=1)  →  [(token, weight), ...]
            │
            ▼
        ExplainResponse JSON
```

## Build & deploy

- **Local dev:** `make install-dev`, `make serve` + `make app` in two terminals.
- **Container:** `docker compose up` builds both images and brings up
  api + streamlit on `sarcasm-radar-net`. API mounts `./models`
  read-only; streamlit calls `http://api:8000` over the internal
  network.
- **Production targets (future work):** Fly.io for the API
  (CPU-only, single region — transformer inference fits in 1GB
  RAM), Streamlit Cloud for the demo.

## Reproducibility

- Fixed `random_seed=42` everywhere (`sarcasm_radar.config.settings`).
- HuggingFace Trainer's own `seed` is wired through
  `TransformerTrainConfig.seed`.
- Dependencies pinned in `requirements.txt` (runtime),
  `requirements-dev.txt` (dev). Snapshot date documented in each file.
- Notebooks are checked in without outputs — re-run after
  `make data && make train` to populate the result tables.

## Testing surface

```
tests/
├── test_smoke.py            package + config + fixture
├── test_load.py              loaders + merge + normalisation
├── test_download_data.py     HF download script (mocked)
├── test_clean.py             text cleaning (8 transformation categories)
├── test_curate.py            curation validation + the 50 seed labels
├── test_metrics.py           macro-F1, per-class, confusion matrix
├── test_baseline.py          TF-IDF + LR pipeline + Hinglish input
├── test_transformer.py       DistilBERT config + save/load + lazy import
├── test_multilingual.py      XLM-R preset + per-register comparison
├── test_inference.py         unified wrapper (baseline + transformer)
└── test_api.py               FastAPI integration via TestClient + stubs
```

160+ tests, mypy strict, ruff clean.
