# Model Card — sarcasm-radar v0.1.0

## Model details

- **Task:** binary sarcasm classification of short English / Hinglish
  text (tweets, posts, chat messages).
- **Architectures shipped:**
  - **Baseline:** TF-IDF (1-2 grams) + Logistic Regression with
    `class_weight='balanced'`.
  - **DistilBERT** (`distilbert-base-uncased`) fine-tuned end-to-end.
  - **XLM-RoBERTa** (`xlm-roberta-base`) fine-tuned end-to-end — the
    production candidate for Hinglish handling.
- **Owner:** Jenisssh.
- **Repo:** https://github.com/Jenisssh/sarcasm-radar
- **License:** MIT.

## Intended use

- **Primary:** content-moderation and analytics pipelines that need
  to know whether a piece of short text was meant ironically.
- **Out of scope:**
  - Long-form text (essays, articles) — the model has only seen
    tweet-length inputs.
  - Languages other than English / Hinglish — XLM-R was pretrained
    multilingually, but this fine-tune only saw English + Hinglish
    rows.
  - High-stakes moderation decisions in isolation — sarcasm
    detection is a *signal*, not an oracle.

## Training data

- **iSarcasm** (Oprea & Magdy 2020) — English-only sarcastic tweets,
  ~5K rows.
- **SemEval-2022 Task 6 iSarcasmEval** (Farha et al. 2022) — English
  partition, ~6K rows.
- **Curated Indian English supplement** — 50 hand-labeled rows
  committed in `data/curated/indian_english.csv`, balanced 25/25
  with `register ∈ {en-IN, hi-en}`.

Merge logic in `src/sarcasm_radar/data/load.py` dedups on the exact
text string and tags each row with its source + language register.

## Metrics

Measured on the held-out 15% test split. Filled in by the final cell
of `notebooks/02_error_analysis.ipynb` after running the training
scripts.

| Model | Macro-F1 (all) | Macro-F1 (en) | Macro-F1 (en-IN) | Macro-F1 (hi-en) |
|---|:-:|:-:|:-:|:-:|
| TF-IDF + LR baseline | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| DistilBERT | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **XLM-RoBERTa** | **_TBD_** | _TBD_ | **_TBD_** | **_TBD_** |

The headline metric is **macro-F1** — both classes matter equally
and the dataset is roughly balanced post-merge. ROC-AUC isn't
reported because the operating point matters more than ranking for
this use case.

## Ethical considerations

- **False positives are not free.** A tweet flagged as sarcastic that
  was sincere can be quoted out of context. The API surfaces the
  probability + decision but never asks downstream systems to take
  action without human review.
- **Cultural calibration.** The model was tuned on Indian English
  /  Hinglish supplements. Outputs on text from other English
  varieties (AAVE, Singlish, Caribbean English, etc.) are not
  characterised in this card — those registers were not in the
  training data.
- **Curation bias.** The 50-row supplement reflects one annotator's
  judgement of what counts as sarcastic in Indian English. The
  `data/curated/LABELING_PROTOCOL.md` document spells out the
  decision rules used, so a second annotator can re-audit the
  set and the disagreement rate can be reported.

## Caveats

- **Small supplement.** 50 rows of Indian English is a starting point,
  not a definitive characterization. The curation pipeline supports
  growth; subsequent runs should add 50–100 more rows and re-evaluate
  the per-register macro-F1.
- **Single test split.** No k-fold or repeated holdouts; the
  reported numbers are point estimates with no confidence intervals.
- **No retraining cadence.** Tweet language evolves fast (new slang,
  meme phrases). A retraining pipeline lives in
  `docs/architecture.md` under future work.
- **LIME for transformers is expensive.** The `/explain` endpoint
  runs 200 perturbations per call which takes 3–5 seconds on CPU.
  Not appropriate for high-throughput scoring; use `/predict` and
  reserve `/explain` for human-in-the-loop review.

## Citation

Datasets:

> Silviu Oprea and Walid Magdy. *iSarcasm: A Dataset of Intended
> Sarcasm.* ACL 2020.

> Ibrahim Abu Farha, Silviu Vlad Oprea, Steven Wilson, Walid Magdy.
> *SemEval-2022 Task 6: iSarcasmEval — Intended Sarcasm Detection
> In English and Arabic.* SemEval 2022.
