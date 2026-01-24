"""Download the public sarcasm corpora from HuggingFace.

Outputs (under ``data/raw/``):

    isarcasm.parquet           normalised iSarcasm-style English tweets
    semeval_isarcasm.parquet   SemEval-2022 Task 6 English partition

The HuggingFace dataset IDs below are the ones we settled on after a
short audit on Jan 24. If the upstream dataset moves, override with
the ``--isarcasm-id`` / ``--semeval-id`` flags.

Usage
-----
    python -m scripts.download_data
    python -m scripts.download_data --force                       # re-download
    python -m scripts.download_data --skip-semeval                # iSarcasm only
    python -m scripts.download_data --isarcasm-id user/other_id   # alternate source

Exit codes
----------
0  success (or every requested file already present)
1  failed to import the ``datasets`` library
2  HuggingFace download raised
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from sarcasm_radar.config import settings
from sarcasm_radar.utils.logging import get_logger

log = get_logger("sarcasm_radar.scripts.download_data")

DEFAULT_ISARCASM_ID = "mteb/silicone-sarcasm"
DEFAULT_SEMEVAL_ID = "iabufarha/iSarcasmEval"


def _load_split(dataset_id: str, split: str = "train") -> Any:
    """Lazy import so the rest of the package doesn't require ``datasets``."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "huggingface 'datasets' is required. " "Run `pip install datasets`."
        ) from e
    return load_dataset(dataset_id, split=split)


def download_isarcasm(
    dest: Path,
    *,
    dataset_id: str = DEFAULT_ISARCASM_ID,
    split: str = "train",
    force: bool = False,
) -> Path:
    """Fetch iSarcasm and save as parquet under ``dest/isarcasm.parquet``."""
    out = dest / "isarcasm.parquet"
    if out.exists() and not force:
        log.info("isarcasm_already_present", path=str(out))
        return out

    log.info("downloading_isarcasm", dataset_id=dataset_id, split=split)
    ds = _load_split(dataset_id, split=split)
    df = ds.to_pandas()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("isarcasm_saved", path=str(out), rows=len(df))
    return out


def download_semeval(
    dest: Path,
    *,
    dataset_id: str = DEFAULT_SEMEVAL_ID,
    split: str = "train",
    force: bool = False,
) -> Path:
    """Fetch SemEval iSarcasmEval and save as parquet."""
    out = dest / "semeval_isarcasm.parquet"
    if out.exists() and not force:
        log.info("semeval_already_present", path=str(out))
        return out

    log.info("downloading_semeval", dataset_id=dataset_id, split=split)
    ds = _load_split(dataset_id, split=split)
    df = ds.to_pandas()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("semeval_saved", path=str(out), rows=len(df))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    parser.add_argument("--skip-isarcasm", action="store_true")
    parser.add_argument("--skip-semeval", action="store_true")
    parser.add_argument("--isarcasm-id", default=DEFAULT_ISARCASM_ID)
    parser.add_argument("--semeval-id", default=DEFAULT_SEMEVAL_ID)
    args = parser.parse_args(argv)

    settings.data_raw.mkdir(parents=True, exist_ok=True)

    try:
        if not args.skip_isarcasm:
            download_isarcasm(
                settings.data_raw,
                dataset_id=args.isarcasm_id,
                force=args.force,
            )
        if not args.skip_semeval:
            download_semeval(
                settings.data_raw,
                dataset_id=args.semeval_id,
                force=args.force,
            )
    except RuntimeError as e:
        log.error("import_error", error=str(e))
        return 1
    except Exception as e:
        log.error("download_failed", error=str(e), error_type=type(e).__name__)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
