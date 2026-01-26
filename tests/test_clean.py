"""Tests for sarcasm_radar.data.clean."""

from __future__ import annotations

import pandas as pd
import pytest

from sarcasm_radar.data.clean import (
    URL_PLACEHOLDER,
    USER_PLACEHOLDER,
    clean_series,
    clean_text,
)


class TestURLs:
    def test_http_url_replaced(self) -> None:
        out = clean_text("see https://example.com for details")
        assert URL_PLACEHOLDER in out
        assert "example.com" not in out

    def test_https_url_replaced(self) -> None:
        out = clean_text("https://t.co/abc123 is cool")
        assert URL_PLACEHOLDER in out

    def test_www_url_replaced(self) -> None:
        out = clean_text("check www.example.com")
        assert URL_PLACEHOLDER in out

    def test_url_replacement_can_be_disabled(self) -> None:
        out = clean_text("see https://example.com", replace_urls=False)
        assert "https://example.com" in out


class TestMentions:
    def test_mention_replaced(self) -> None:
        out = clean_text("hey @alice nice catch")
        assert USER_PLACEHOLDER in out
        assert "@alice" not in out

    def test_multiple_mentions(self) -> None:
        out = clean_text("@alice and @bob_42 chatting")
        assert out.count(USER_PLACEHOLDER) == 2

    def test_mention_with_email_like_in_text(self) -> None:
        # Just an email-shaped substring with @ shouldn't be aggressively masked
        out = clean_text("contact admin@site.com")
        # ``@site`` matches our pattern — known limitation; document via test
        assert USER_PLACEHOLDER in out

    def test_mention_replacement_can_be_disabled(self) -> None:
        out = clean_text("hey @alice", replace_mentions=False)
        assert "@alice" in out


class TestHashtags:
    def test_hash_stripped_word_kept(self) -> None:
        out = clean_text("loving #Diwali vibes")
        assert "#Diwali" not in out
        assert "Diwali" in out

    def test_underscore_in_hashtag(self) -> None:
        out = clean_text("#new_year_2026 already")
        assert "new_year_2026" in out

    def test_stripping_can_be_disabled(self) -> None:
        out = clean_text("loving #Diwali", strip_hashtag_hash=False)
        assert "#Diwali" in out


class TestRepeats:
    def test_long_run_collapses_to_three(self) -> None:
        out = clean_text("sooooooo cool")
        assert "soooo" not in out
        assert "sooo " in out  # three Os

    def test_three_runs_are_kept(self) -> None:
        # Exactly 3 repeats should pass through unchanged
        out = clean_text("nooo way")
        assert "nooo " in out

    def test_collapse_can_be_disabled(self) -> None:
        out = clean_text("sooooo", collapse_repeats=False)
        assert out == "sooooo"


class TestHTML:
    def test_amp_decoded(self) -> None:
        assert "&" in clean_text("AT&amp;T announced")

    def test_lt_gt_decoded(self) -> None:
        out = clean_text("&lt;3 you")
        assert "<3" in out

    def test_decode_can_be_disabled(self) -> None:
        out = clean_text("AT&amp;T", decode_html=False)
        assert "&amp;" in out


class TestWhitespace:
    def test_multiple_spaces_collapsed(self) -> None:
        assert clean_text("hello   world") == "hello world"

    def test_tabs_and_newlines_collapsed(self) -> None:
        assert clean_text("hello\n\tworld") == "hello world"

    def test_edges_trimmed(self) -> None:
        assert clean_text("  hello world  ") == "hello world"


class TestPreservation:
    def test_case_preserved(self) -> None:
        out = clean_text("OBVIOUSLY this is fine")
        assert "OBVIOUSLY" in out

    def test_emoji_preserved(self) -> None:
        out = clean_text("great plan 🙄")
        assert "🙄" in out

    def test_hindi_script_preserved(self) -> None:
        out = clean_text("कमाल का प्लान")
        assert "कमाल" in out

    def test_hinglish_preserved(self) -> None:
        out = clean_text("haa beta mast plan")
        # All tokens still present in original case
        for token in ("haa", "beta", "mast", "plan"):
            assert token in out


class TestRobustness:
    def test_empty_string(self) -> None:
        assert clean_text("") == ""

    def test_only_whitespace(self) -> None:
        assert clean_text("   \n\t  ") == ""

    def test_non_string_coerced(self) -> None:
        # Non-strings get str()'d — shouldn't raise
        assert clean_text(12345) == "12345"  # type: ignore[arg-type]

    def test_none_coerced_to_string(self) -> None:
        out = clean_text(None)  # type: ignore[arg-type]
        assert out == "None"


class TestSeries:
    def test_clean_series_applies_to_each_row(self) -> None:
        s = pd.Series(["check https://x.com", "@alice nice #Diwali"])
        out = clean_series(s)
        assert URL_PLACEHOLDER in out.iloc[0]
        assert USER_PLACEHOLDER in out.iloc[1]
        assert "Diwali" in out.iloc[1]
        assert "#" not in out.iloc[1]

    def test_kwargs_forward_to_clean_text(self) -> None:
        s = pd.Series(["see https://x.com"])
        out = clean_series(s, replace_urls=False)
        assert "https://x.com" in out.iloc[0]

    def test_preserves_index(self) -> None:
        s = pd.Series(["a", "b"], index=["x", "y"])
        out = clean_series(s)
        assert list(out.index) == ["x", "y"]


@pytest.mark.parametrize(
    "raw,expected_in",
    [
        ("Soooooooo glad to be stuck in traffic 🚗", "Sooo"),
        ("RT @newsbot: BIG NEWS &amp; analysis at https://example.com", USER_PLACEHOLDER),
        ("absolutely #thrilled to receive 50 emails", "thrilled"),
    ],
)
def test_end_to_end_examples(raw: str, expected_in: str) -> None:
    assert expected_in in clean_text(raw)
