# Data

The corpora this project uses are **not** committed to the repo (gitignored
under `data/raw/` and `data/processed/`). Only the curated Indian English
supplement under `data/curated/` is version-controlled, since it's small
and is the unique contribution of this project.

## Sources

| Corpus | Source | Size | License |
|---|---|---|---|
| iSarcasm | Oprea & Magdy 2020 — HuggingFace `silicone/sarcasm` | ~5K tweets | research-use |
| SemEval-2022 Task 6 iSarcasmEval | Farha et al. 2022 — HuggingFace `iSarcasm/iSarcasmEval` | ~6K English entries | research-use |
| Indian English supplement (this repo) | hand-labeled in `data/curated/indian_english.csv` | ~50 entries (and growing) | CC-BY-SA 4.0 |

## How to fetch

```bash
make data
# or:
python -m scripts.download_data
```

Downloads land in `data/raw/`. The merging + cleaning pipeline produces
`data/processed/{train,val,test}.parquet`.

## Schema

After the loader normalises everything, each row has two columns:

| Column | Type | Notes |
|---|---|---|
| `text` | string | tweet content, lightly cleaned (URLs and @mentions normalised, see `data/clean.py`) |
| `label` | int | 1 = sarcastic, 0 = not sarcastic |
| `source` | string | which corpus the row came from (kept for stratification + error analysis) |
| `language_register` | string | `en` for English, `hi-en` for Hinglish, `en-IN` for Indian English |

## Layout

```
data/
├── raw/         # downloaded corpora — gitignored
├── curated/     # hand-labeled Indian English supplement — committed
├── processed/   # train/val/test parquets — gitignored
└── README.md    # this file
```

A `.gitkeep` file in `raw/` and `processed/` preserves the structure.
