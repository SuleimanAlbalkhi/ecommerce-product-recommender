"""Loading of the cleaned product CSV produced by scripts/clean.py."""

import pandas as pd


def load_data(path: str, sample_n: int | None = None) -> pd.DataFrame:
    """Load the cleaned product CSV (category, name, description), optionally sampling sample_n rows."""
    df = pd.read_csv(path)
    # the CSV should already be clean, but I check again here. One bad
    # category value would silently break the category filter in the app.
    df = df.dropna(subset=["category"])
    df["category"] = df["category"].astype(str).str.strip()
    if sample_n is not None:
        # fixed seed, so the sample is always the same
        # (test_reproducible_sample depends on this)
        df = df.sample(n=min(sample_n, len(df)), random_state=42)
    # very important: after reset_index the row label, the row position and
    # the TF-IDF matrix row are all the same number. Both recommenders
    # depend on this, so do not remove it.
    return df.reset_index(drop=True)
