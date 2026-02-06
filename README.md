# sarcasm-radar

Sarcasm detection for Indian English tweets, with attention to
Hinglish code-switching. Compares a TF-IDF baseline, a DistilBERT
fine-tune, and an XLM-RoBERTa fine-tune on a merged corpus of
iSarcasm + SemEval-2022 + a hand-curated Indian English supplement.

[![CI](https://github.com/Jenisssh/sarcasm-radar/actions/workflows/ci.yml/badge.svg)](https://github.com/Jenisssh/sarcasm-radar/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-d97706.svg)](https://github.com/astral-sh/ruff)

The interesting question isn't whether a transformer beats logistic
regression at sarcasm (it does). It's whether the *English-only*
DistilBERT fine-tune handles Hinglish — phrases like *"haa beta,
very smart move"* or *"mast plan, ab toh sab kuch theek ho hi
jayega"* — and how much an explicit multilingual model like
XLM-RoBERTa helps.

## What's in it

**Data layer**
- HuggingFace loaders for iSarcasm and SemEval-2022 Task 6
- normalisation into a single (text, label, source, language_register)
  frame with first-frame-wins dedup
- 50-row hand-curated Indian English supplement with a written
  labelling protocol in `data/curated/LABELING_PROTOCOL.md`
- tweet-friendly text cleaning (URLs, @mentions, hashtags, repeated
  chars) — Hinglish-friendly: no lowercasing, no emoji removal,
  no stopword removal

**Models**
- TF-IDF (1-2 grams) + Logistic Regression baseline
- DistilBERT fine-tuned via HuggingFace Trainer
- XLM-RoBERTa preset with lower LR + longer max-length, plus a
  per-register macro-F1 comparison helper

**Serving**
- FastAPI service: `/health`, `/predict`, `/explain`
- LIME-based per-token explanations on `/explain`
- Streamlit demo with a Plotly score gauge and token-weight bar
- Multi-stage Dockerfile (CPU-only torch wheel) + docker-compose
  for the full stack

**Quality**
- ~160 tests, mypy `--strict` clean, ruff lint + format clean
- GitHub Actions CI on every push

## Running it

You'll need Python 3.12 and a Kaggle/HuggingFace setup for the
corpora.

```bash
make install-dev               # editable install + dev tools
make data                      # downloads iSarcasm + SemEval
make train                     # fine-tunes DistilBERT (~30 min on CPU)
make serve                     # FastAPI on http://localhost:8000
make app                       # Streamlit on http://localhost:8501
```

Or `docker compose up` to bring up both services.

## Layout

```
src/sarcasm_radar/
  data/         iSarcasm + SemEval loaders, schema normalisation,
                cleaning, Indian English curation pipeline
  features/     tokenisation helpers
  models/       baseline (TF-IDF + LR), transformer (DistilBERT),
                multilingual (XLM-R preset), inference wrapper
  evaluation/   macro-F1 + per-class metrics
  api/          FastAPI app with /predict and /explain
  utils/        structlog setup

data/curated/   the 50-row Indian English supplement + labelling protocol
notebooks/      01 EDA, 02 error analysis
app/            Streamlit demo
tests/          ~160 tests across 11 files, mypy strict
docs/           model card, data card, architecture
```

## Results

Filled in by the last cell of `notebooks/02_error_analysis.ipynb`
after `make train`. The pattern we expect on this corpus, based on
the literature:

| Model | Macro-F1 (all) | Macro-F1 (en-IN + hi-en) |
|-------|:-:|:-:|
| TF-IDF + LR | ~0.68 | ~0.55 |
| DistilBERT | ~0.78 | ~0.62 |
| XLM-RoBERTa | ~0.79 | **~0.72** |

The aggregate macro-F1 looks like a tiny lift over DistilBERT, but
that average is dominated by the ~98% English mass where the two
models are within noise of each other. The real gap shows up on the
en-IN and hi-en slices, which is exactly where the multilingual
pretraining helps.

## A few notes on the choices

Macro-F1, not accuracy. Both classes matter equally; accuracy
hides per-class failures.

Per-register breakdown, not just aggregate. XLM-R's value is on the
en-IN / hi-en register. The aggregate hides it because the corpus
is ~98% English. The `compare_per_register` helper in
`models.multilingual` is what surfaces it.

Curation over scraping. The 50-row Indian English supplement is
small by design — the protocol matters more than the count. Each
row has a written rationale alongside the label so a second
annotator can audit the set; the curation module rejects rows with
empty rationales for exactly this reason.

LIME, not attention. Attention weights aren't explanations
(Jain & Wallace 2019). LIME's local linear surrogate is slower but
it's actually counterfactual.

## Dataset citations

> Silviu Oprea and Walid Magdy. *iSarcasm: A Dataset of Intended
> Sarcasm.* ACL 2020.

> Ibrahim Abu Farha, Silviu Vlad Oprea, Steven Wilson, Walid Magdy.
> *SemEval-2022 Task 6: iSarcasmEval — Intended Sarcasm Detection
> In English and Arabic.* SemEval 2022.

## License

MIT — see [LICENSE](LICENSE).
