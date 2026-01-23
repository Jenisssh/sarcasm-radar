"""Shared pytest fixtures.

``tiny_sarcasm_df`` is a 20-row toy dataset following the project schema
(`text`, `label`) so unit tests run in milliseconds without needing the
real corpus on disk.
"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def tiny_sarcasm_df() -> pd.DataFrame:
    """Tiny toy dataset with a mix of sarcastic / not-sarcastic examples."""
    rows = [
        ("Great, another Monday. My favorite day.", 1),
        ("Oh wow, the trains are late again. Shocking.", 1),
        ("haa beta, very smart move", 1),
        ("mast plan, ab toh sab kuch theek ho hi jayega", 1),
        ("This is the best news I've heard all week.", 1),
        ("Loved waiting in line for 2 hours, 10/10 experience", 1),
        ("Sure, blame the intern. Classic.", 1),
        ("yeah right, like that's ever going to happen", 1),
        ("waah kya timing hai bhai", 1),
        ("amazing service, only took them 3 emails to respond", 1),
        ("The weather is nice today.", 0),
        ("Just finished my morning run, feeling great.", 0),
        ("Looking forward to the team lunch tomorrow.", 0),
        ("She gave a really thoughtful presentation.", 0),
        ("Mumbai local was on time today, surprised!", 0),
        ("Picked up the new book from the library.", 0),
        ("The project deadline is next Friday.", 0),
        ("Had a great chat with my mentor over coffee.", 0),
        ("Diwali plans are coming together nicely.", 0),
        ("Just submitted my final assignment for the semester.", 0),
    ]
    return pd.DataFrame(rows, columns=["text", "label"])
