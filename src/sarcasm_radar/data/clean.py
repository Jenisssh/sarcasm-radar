"""Text cleaning for tweets, with Hinglish-friendly defaults.

Decisions baked in here, with the reasons:

- **URLs** → the literal token ``<URL>``. The model should learn that
  "this tweet contains a link" matters, without memorising specific
  domains the val/test set won't contain.
- **@mentions** → ``<USER>``. The ``@`` alone leaks identity that
  doesn't generalise.
- **Hashtags** → the ``#`` is stripped but the word stays. ``#Modi`` and
  ``Modi`` should mean the same thing to the model.
- **HTML entities** → decoded (``&amp;`` → ``&``, ``&lt;`` → ``<``, ...).
- **Repeated characters** → 4+ collapsed to 3 (``soooooo`` → ``sooo``).
- **Whitespace** → multiple spaces collapsed to one, edges trimmed.

Decisions deliberately *not* made here:

- **No lowercasing.** Uppercase often signals shouting / sarcasm in
  tweet text, and lowercasing Hinglish romanisation throws away
  legitimate variation.
- **No emoji removal.** Emojis carry strong sarcasm signal (🙄, 😒, 👏).
- **No stopword removal.** Stopwords (``the``, ``so``, ``well``) carry
  ironic register; removing them is harmful for sarcasm detection.
- **No transliteration of Hinglish.** Conservative by design — the
  multilingual transformer in week 2 handles code-switching natively.

Every individual cleaning step is an opt-out flag, so the same function
serves both the preprocessing pipeline and the EDA notebooks.
"""

from __future__ import annotations

import html
import re
from typing import Any

import pandas as pd

URL_PLACEHOLDER = "<URL>"
USER_PLACEHOLDER = "<USER>"

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
_HASHTAG_PATTERN = re.compile(r"#(\w+)")
_WHITESPACE_PATTERN = re.compile(r"\s+")
# 4 or more of the same char collapse to 3 (so "sooooo" -> "sooo")
_REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{3,}")


def clean_text(
    text: str,
    *,
    replace_urls: bool = True,
    replace_mentions: bool = True,
    strip_hashtag_hash: bool = True,
    collapse_repeats: bool = True,
    decode_html: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    """Return a normalised version of ``text``.

    Each transformation can be disabled with the corresponding flag. The
    function never raises on bad input — non-string values are coerced
    with ``str()``.
    """
    if not isinstance(text, str):
        text = str(text)

    if decode_html:
        text = html.unescape(text)
    if replace_urls:
        text = _URL_PATTERN.sub(URL_PLACEHOLDER, text)
    if replace_mentions:
        text = _MENTION_PATTERN.sub(USER_PLACEHOLDER, text)
    if strip_hashtag_hash:
        text = _HASHTAG_PATTERN.sub(r"\1", text)
    if collapse_repeats:
        text = _REPEATED_CHAR_PATTERN.sub(r"\1\1\1", text)
    if collapse_whitespace:
        text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def clean_series(series: pd.Series, **kwargs: Any) -> pd.Series:
    """Vectorised ``clean_text`` over a pandas Series.

    ``**kwargs`` forwards to :func:`clean_text` so callers can flip
    individual steps off without writing a lambda.
    """
    return series.apply(lambda x: clean_text(x, **kwargs))
