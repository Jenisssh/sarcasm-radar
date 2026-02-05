# Data Card — sarcasm-radar Training Corpora

## Sources

| Corpus | Source | Rows | License | Notes |
|---|---|---|---|---|
| iSarcasm | Oprea & Magdy 2020 (HuggingFace `mteb/silicone-sarcasm` or equivalent) | ~5K | research-use | English tweets, intentional sarcasm labelled by the author |
| SemEval-2022 Task 6 iSarcasmEval | Farha et al. 2022 (HuggingFace `iabufarha/iSarcasmEval`) | ~6K | research-use | English partition only; Arabic excluded |
| Indian English supplement | hand-labelled in `data/curated/indian_english.csv` | 50 (and growing) | CC-BY-SA 4.0 | balanced 25/25, mix of `en-IN` and `hi-en` |

## Schema (post-normalisation)

After the loader merges all sources, every row has four columns:

| Column | Type | Notes |
|---|---|---|
| `text` | string | the tweet/utterance, single line |
| `label` | int | 1 = sarcastic, 0 = not sarcastic |
| `source` | string | `isarcasm`, `semeval`, or `curated_indian_english` |
| `language_register` | string | `en`, `en-IN`, or `hi-en` |

## How to fetch

```bash
make data
# or:
python -m scripts.download_data
```

This downloads the public corpora via HuggingFace `datasets`, lands
them as parquet files under `data/raw/`, and leaves the curated
supplement alone (already committed). The merging happens in
`sarcasm_radar.data.load.merge_corpora` — duplicates are dropped on
the exact `text` string, first-frame-wins on conflicts.

## Known limitations

- **Domain.** Tweet-length text. Don't fine-tune the model on
  paragraphs and expect it to generalise.
- **English bias.** Even with XLM-RoBERTa, ~98% of the merged
  corpus is English. The 50-row supplement helps but doesn't fix
  the imbalance.
- **Author-labelled sarcasm.** iSarcasm uses the *author's* claim
  of intent, not a third-party reading. This is more principled than
  crowdsourced labels but skews toward overt sarcasm.
- **Time window.** The Western corpora are from 2018-2022; the
  supplement was labelled in January 2026. Slang from the gap
  isn't well-represented.
- **Hinglish romanisation variance.** *"haa beta"* vs *"hain beta"*
  vs *"haan beta"* — the model has only seen the first of these
  in the 50-row supplement.

## How to extend

Adding more curated rows:

```python
from sarcasm_radar.data.curate import append_label

append_label(
    text="kya mast plan banaya hai bhai, sab ka time waste",
    label=1,
    register="hi-en",
    rationale="praising a plan that wastes everyone's time",
)
```

The helper validates each row (rejects empty text / empty rationale
/ bad label / bad register) and appends to
`data/curated/indian_english.csv` with the correct header. Don't
edit the CSV by hand.

## Citation

> Silviu Oprea and Walid Magdy. *iSarcasm: A Dataset of Intended
> Sarcasm.* ACL 2020.

> Ibrahim Abu Farha, Silviu Vlad Oprea, Steven Wilson, Walid Magdy.
> *SemEval-2022 Task 6: iSarcasmEval — Intended Sarcasm Detection
> In English and Arabic.* SemEval 2022.
