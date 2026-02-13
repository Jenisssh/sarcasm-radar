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
regression at sarcasm — it does. It's whether an English-only
DistilBERT fine-tune handles Hinglish — phrases like *"haa beta,
very smart move"* or *"mast plan, ab toh sab kuch theek ho hi
jayega"* — and whether an explicit multilingual model like
XLM-RoBERTa actually helps. (Spoiler from the Results section: on a
small dataset, it doesn't — it collapses.)

## What's in it

**Data layer**
- HuggingFace loaders for iSarcasm and SemEval-2022 Task 6
- normalisation into a single `(text, label, source, language_register)`
  frame with first-frame-wins dedup
- 50-row hand-curated Indian English supplement + written
  labelling protocol in `data/curated/LABELING_PROTOCOL.md`
- tweet-friendly text cleaning (URLs, @mentions, hashtags,
  repeated chars) — Hinglish-friendly defaults (no lowercasing,
  no emoji removal, no stopword removal)

**Models**
- TF-IDF (1-2 grams) + Logistic Regression baseline
- DistilBERT fine-tuned via HuggingFace Trainer
- XLM-RoBERTa preset with lower LR + longer max-length, plus a
  per-register macro-F1 comparison helper

**Serving**
- FastAPI: `/health`, `/predict`, `/explain` (LIME-based)
- Streamlit demo with a Plotly score gauge and token-weight bar
- Multi-stage Dockerfile (CPU-only torch wheel) + docker-compose

**Quality**
- ~160 tests, mypy `--strict` clean, ruff lint + format clean
- GitHub Actions CI on every push

## How it fits together

```
       iSarcasm + SemEval                  data/curated/indian_english.csv
       (HuggingFace, gitignored)           (50 hand-labelled rows, committed)
              │                                        │
              └──────────────┬─────────────────────────┘
                             ▼
                   data.load.merge_corpora
                  (text, label, source, language_register)
                             │
                             ▼
                       data.clean
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
       TF-IDF + LR      DistilBERT       XLM-RoBERTa
        baseline        fine-tune        fine-tune
            └────────────────┼────────────────┘
                             ▼
                     models.inference
                  (unified predict_single)
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
              FastAPI               Streamlit
            /predict /explain        demo app
```

## Running it

You'll need Python 3.12 and a HuggingFace setup for the corpora.

```bash
make install-dev               # editable install + dev tools
make data                      # downloads iSarcasm + SemEval
make train                     # fine-tunes DistilBERT (~30 min on CPU)
make serve                     # FastAPI on http://localhost:8000
make app                       # Streamlit on http://localhost:8501
```

Or `docker compose up` to bring up both services.

## Talking to the API

```bash
curl -sX POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"text":"mast plan, ab toh sab kuch theek ho hi jayega"}'
```

```json
{
  "text": "mast plan, ab toh sab kuch theek ho hi jayega",
  "probability": 0.91,
  "decision": "SARCASTIC",
  "threshold": 0.5,
  "model_version": "v0.1.0"
}
```

`/explain` returns the same fields plus a `tokens` array with LIME
per-token weights — positive pushes toward sarcastic, negative
toward not.

## Growing the curated supplement

```python
from sarcasm_radar.data.curate import append_label

append_label(
    text="kya mast plan banaya hai bhai, sab ka time waste",
    label=1,
    register="hi-en",
    rationale="praising a plan that wastes everyone's time",
)
```

The helper validates each row (rejects empty text, empty rationale,
or bad label / register) and appends to
`data/curated/indian_english.csv` with the correct header. Don't
edit the CSV by hand — the rationale field is enforced
non-empty for quality-control reasons.

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

Trained on the SemEval-2022 Task 6 iSarcasmEval English set (2,773
train / 694 test tweets, stratified 80/20). The 50-row curated
Indian English set is held out entirely as a cross-domain probe —
no model trains on it. Reproduce on a free Colab GPU with
[`notebooks/03_train_colab.ipynb`](notebooks/03_train_colab.ipynb).

| Model | Macro-F1 (iSarcasmEval test) | Macro-F1 (Indian English probe, n=50) |
|-------|:-:|:-:|
| TF-IDF + LR | 0.545 | 0.523 |
| **DistilBERT** | **0.588** | **0.696** |
| XLM-RoBERTa | 0.428 † | 0.333 † |

**DistilBERT is the production model.** 0.588 macro-F1 sounds
modest, but iSarcasmEval is a deliberately hard benchmark — the
SemEval-2022 Task 6 shared-task systems scored in the 0.57–0.61
range, so this is competitive with the published competition
results. DistilBERT also holds up on the curated Indian English
probe (0.696), so a model trained only on Western tweets
generalises reasonably to the en-IN / hi-en register.

† **XLM-RoBERTa-base collapsed to the majority class** during
fine-tuning — per-class F1 on the test set was 0.857 (not-sarcastic)
and **0.000 (sarcastic)**. It learned the trivial all-negative
solution and never predicts the minority class. This is a known
instability of XLM-R-base on small (~2.7k rows), class-imbalanced
data without a class-weighted loss. The honest finding: the
multilingual model is *not* the win here — the lighter DistilBERT
is the practical choice, and XLM-R would need more data plus
class-weighting and a lower learning rate to be viable. Full
write-up in [`docs/model_card.md`](docs/model_card.md).

## A few notes on the choices

Macro-F1, not accuracy. Both classes matter equally; accuracy
hides per-class failures.

The Indian English set is a held-out probe, not training data. 50
rows can't train anything — but it's a clean cross-domain test:
models trained on Western iSarcasm tweets, scored on curated
en-IN / hi-en text they never saw. `compare_per_register` in
`models.multilingual` does the per-register split. (It's also how
the XLM-R collapse showed up clearly — see Results.)

Curation over scraping. The 50-row Indian English supplement is
small by design — the protocol matters more than the count. Each
row has a written rationale alongside the label so a second
annotator can audit the set; the curation module rejects rows
with empty rationales for exactly this reason.

LIME, not attention. Attention weights aren't explanations
(Jain & Wallace 2019). LIME's local linear surrogate is slower —
about 3-5 seconds per call on the transformer with 200 perturbations
— but it's actually counterfactual.

## What this doesn't do (yet)

- No retraining cadence. Tweet slang moves fast; a real deployment
  would need a monthly or quarterly refresh on a fresh time slice.
- The supplement is 50 rows. Enough to validate the pipeline, not
  enough to characterise Indian English sarcasm. The labelling
  protocol is what makes it growable.
- LIME on a transformer is too slow for high-throughput scoring.
  `/explain` is for human-in-the-loop review, not real-time.

## Dataset citations

> Silviu Oprea and Walid Magdy. *iSarcasm: A Dataset of Intended
> Sarcasm.* ACL 2020.

> Ibrahim Abu Farha, Silviu Vlad Oprea, Steven Wilson, Walid Magdy.
> *SemEval-2022 Task 6: iSarcasmEval — Intended Sarcasm Detection
> In English and Arabic.* SemEval 2022.

## License

MIT — see [LICENSE](LICENSE).
