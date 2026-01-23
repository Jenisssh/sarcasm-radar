# sarcasm-radar

Sarcasm detection for Indian English tweets, with attention to Hinglish
code-switching and culturally local phrasings. Fine-tunes a transformer on
public sarcasm corpora plus a small hand-curated Indian English supplement.

> Status: scaffolding. The repo is being built out over the next 3 weeks.

## Why this

Most public sarcasm datasets (iSarcasm, SARC) are Western Reddit/Twitter
text. A model trained on them generalises badly to phrases like
*"haa beta, very smart move"* or
*"mast plan, ab toh sab kuch theek ho hi jayega"* — the Indian English
register has its own conventions that the model has never seen.

This project addresses that with two things working together:

- a small hand-labeled supplement of ~50 Indian English examples, expandable
  via the curation pipeline in `src/sarcasm_radar/data/curate.py`
- a multilingual transformer (XLM-RoBERTa) compared head-to-head against
  English-only DistilBERT, with error analysis split by language register

## Layout (target)

```
src/sarcasm_radar/
  data/        loaders, cleaning, curation
  features/    tokenization helpers
  models/      TF-IDF + LR baseline, transformer fine-tunes
  evaluation/  metrics, error analysis
  api/         FastAPI service

notebooks/     EDA, comparison, error breakdown
app/           Streamlit demo
tests/         pytest, mypy strict, ruff
docs/          model card, data card, architecture
```

## License

MIT.
