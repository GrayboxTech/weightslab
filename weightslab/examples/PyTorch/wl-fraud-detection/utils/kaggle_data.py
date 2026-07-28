"""Resolve or download the real Kaggle "Credit Card Fraud Detection" (ULB) CSV.

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

There is no anonymous/unauthenticated mirror of this dataset, so a from-scratch
run needs exactly one of:

  * a previously cached copy (this module remembers where it put it),
  * ``kagglehub`` installed + Kaggle credentials configured, or
  * a manual download, pointed at via ``dataset.csv_path`` / ``WL_FRAUD_CSV_PATH``.

Kept in its own module (no ``weightslab``/``torch`` import) so the resolution
logic is unit-testable offline (see ``test_fraud_detection.py``).
"""

from __future__ import annotations

import os
import shutil
from typing import Optional

DATASET_SLUG = "mlg-ulb/creditcardfraud"
CSV_FILENAME = "creditcard.csv"

MANUAL_DOWNLOAD_HELP = f"""
Could not find or download the credit-card-fraud CSV.

This is a real Kaggle dataset ({DATASET_SLUG}) — there is no anonymous
mirror, so getting the data takes one of these one-time steps:

  1) Recommended — Kaggle API credentials (free Kaggle account):
       pip install kagglehub
       # then either:
       python -c "import kagglehub; kagglehub.login()"   # interactive, or
       # place your token at ~/.kaggle/kaggle.json
       #   (Kaggle.com > Settings > API > "Create New Token"), or
       # export KAGGLE_USERNAME=... KAGGLE_KEY=...
     Re-run this script — it will download and cache the CSV automatically.

  2) Manual download:
       https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
     Download `creditcard.csv` and either:
       - place it at: {{cache_dir}}/{CSV_FILENAME}, or
       - point `dataset.csv_path` in config.yaml (or env WL_FRAUD_CSV_PATH)
         directly at the file.
""".strip()


def default_cache_dir() -> str:
    """``<example_dir>/.cache`` — already covered by the repo-wide .gitignore."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache")


def _try_kagglehub_download() -> Optional[str]:
    """Best-effort download via the optional ``kagglehub`` dependency.

    Returns the local path to ``creditcard.csv``, or ``None`` if kagglehub
    isn't installed, isn't authenticated, or the download otherwise fails —
    callers fall back to the manual-download instructions in that case.
    """
    try:
        import kagglehub
    except ImportError:
        return None

    try:
        dataset_dir = kagglehub.dataset_download(DATASET_SLUG)
    except Exception as e:
        print(
            f"[fraud-data] kagglehub download failed ({e}); "
            "falling back to manual-download instructions.",
            flush=True,
        )
        return None

    csv_path = os.path.join(dataset_dir, CSV_FILENAME)
    return csv_path if os.path.isfile(csv_path) else None


def resolve_csv_path(
    explicit_path: Optional[str] = None, cache_dir: Optional[str] = None
) -> str:
    """Locate ``creditcard.csv``, downloading + caching it via kagglehub if needed.

    Resolution order: ``explicit_path`` -> ``WL_FRAUD_CSV_PATH`` env var -> a
    previously cached copy under ``cache_dir`` -> a fresh ``kagglehub`` download
    (cached for next time). Raises ``FileNotFoundError`` with manual-download
    instructions if none of those work.
    """
    cache_dir = cache_dir or default_cache_dir()
    cached_path = os.path.join(cache_dir, CSV_FILENAME)

    for candidate in (explicit_path, os.environ.get("WL_FRAUD_CSV_PATH"), cached_path):
        if candidate and os.path.isfile(candidate):
            return candidate

    downloaded_path = _try_kagglehub_download()
    if downloaded_path:
        os.makedirs(cache_dir, exist_ok=True)
        if os.path.abspath(downloaded_path) != os.path.abspath(cached_path):
            try:
                shutil.copy2(downloaded_path, cached_path)
                return cached_path
            except OSError:
                pass
        return downloaded_path

    raise FileNotFoundError(MANUAL_DOWNLOAD_HELP.format(cache_dir=cache_dir))
