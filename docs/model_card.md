# Model Card — sarcasm-radar v0.1.0

## Model details

- **Task:** binary sarcasm classification of short English / Hinglish
  text (tweets, posts, chat messages).
- **Architectures evaluated:**
  - **Baseline:** TF-IDF (1-2 grams) + Logistic Regression with
    `class_weight='balanced'`.
  - **DistilBERT** (`distilbert-base-uncased`) fine-tuned end-to-end —
    **the production model** (see Metrics).
  - **XLM-RoBERTa** (`xlm-roberta-base`) fine-tuned end-to-end —
    evaluated, but collapsed to the majority class on this dataset;
    **not shipped**. See Metrics and Caveats.
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

The v0.1.0 results below were produced on the **SemEval-2022
iSarcasmEval English set** (2,773 train / 694 test tweets). The
iSarcasm loader is wired up and tested, but that corpus was not
part of this training run.

## Metrics

Trained on the SemEval-2022 Task 6 iSarcasmEval English set, 80/20
stratified split. The 50-row curated Indian English set is held out
as a cross-domain probe — no model trains on it. Reproduce with
`notebooks/03_train_colab.ipynb` on a free Colab GPU.

| Model | Macro-F1 (iSarcasmEval test) | Macro-F1 (Indian English probe, n=50) |
|---|:-:|:-:|
| TF-IDF + LR baseline | 0.545 | 0.523 |
| **DistilBERT** (production) | **0.588** | **0.696** |
| XLM-RoBERTa | 0.428 | 0.333 |

The headline metric is **macro-F1** — both classes matter equally.

**DistilBERT is the model the API serves.** 0.588 macro-F1 on
iSarcasmEval is competitive with the SemEval-2022 Task 6 shared-task
systems (0.57–0.61 range — the benchmark is deliberately hard). It
also holds up on the curated Indian English probe (0.696), so a
model trained only on Western tweets generalises reasonably to the
en-IN / hi-en register.

**XLM-RoBERTa collapsed.** On the test set its per-class F1 was
0.857 (not-sarcastic) and **0.000 (sarcastic)** — it learned the
trivial all-negative solution and never predicts the minority
class. macro-F1 0.428 is just the average of those two. See Caveats
for the diagnosis.

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

- **XLM-RoBERTa-base collapsed under fine-tuning.** On this dataset
  (~2.7k training rows, class-imbalanced, no class-weighted loss in
  the transformer trainer) XLM-R-base settled into predicting
  *not-sarcastic* for everything — sarcastic-class F1 = 0.00. This is
  a documented instability of XLM-R-base on small datasets: it's
  sensitive to learning rate and warmup and frequently degenerates
  without class-weighting or a lower LR. The honest read — the
  multilingual model is not viable at this data volume as
  configured; DistilBERT is the practical choice. Re-running with a
  class-weighted loss, a lower learning rate, and more training data
  is the documented next step.
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
