# Labeling Protocol — Indian English Sarcasm Supplement

This document covers *how* rows in `indian_english.csv` are decided.
It exists because sarcasm is inherently subjective; without a written
protocol, two annotators (or one annotator at different times) will
disagree noisily.

## Schema

Each row has four fields:

| Field | Type | Notes |
|---|---|---|
| `text` | string | cleaned utterance, single line |
| `label` | int | 0 = not sarcastic, 1 = sarcastic |
| `register` | string | `en-IN` (Indian English) or `hi-en` (Hinglish code-switching) |
| `rationale` | string | short note on *why* this label, for quality control |

## Decision rules

1. **Read the utterance literally first.** If the surface meaning is
   the intended meaning, label `0`.
2. **Check for context-contradicting markers.** If the surface meaning
   is the *opposite* of what the speaker plausibly means in the
   referenced situation, label `1`. Common markers:
   - over-praise of a clearly bad outcome (*"great service"* + clearly bad service)
   - "obviously", "clearly", "totally" + a non-obvious / unfortunate fact
   - Hinglish: *"haa beta"*, *"waah"*, *"kamaal"*, *"shaandar"* + a bad outcome
3. **Tone words alone don't make sarcasm.** *"That biryani was amazing"*
   is positive unless context contradicts.
4. **Borderline cases default to `0`.** Sarcasm classifiers err
   toward over-prediction; the dataset should not encourage it.

## Register

- `en-IN`: English-only with Indian English usages (e.g. *"prepone"*,
  *"do the needful"*, Indian place names, IST timing references).
- `hi-en`: code-switched. At least one Hindi / Urdu / regional word.
  Romanised Hinglish counts.

## Rationale field

Always one short clause explaining the *signal* that decided the label.
Two reasons to keep it:

1. **Quality control.** Re-reading the rationale months later is the
   fastest way to find a row I'd label differently now.
2. **Inter-annotator agreement.** When a second annotator joins, they
   can see what cue I keyed on for each row.

## How to add rows

```python
from sarcasm_radar.data.curate import append_label

append_label(
    text="haa beta, kamaal ka idea",
    label=1,
    register="hi-en",
    rationale="ironic praise — 'kamaal' for a bad idea",
)
```

The helper validates the row and appends with the right header. Don't
edit `indian_english.csv` by hand if you can avoid it.

## Current corpus

50 seed rows, balanced 25 sarcastic / 25 non-sarcastic. Mix of `en-IN`
and `hi-en`. Will grow as the curation pass continues.
