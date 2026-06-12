"""Loading of the cleaned product CSV produced by scripts/clean.py."""

import pandas as pd


def load_data(path: str, sample_n: int | None = None) -> pd.DataFrame:
    """Load the cleaned product CSV (category, name, description), optionally sampling sample_n rows."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["category"])
    df["category"] = df["category"].astype(str).str.strip()
    if sample_n is not None:
        # fixed seed: reproducible sampling (test_data_loader.py::test_reproducible_sample relies on it)
        df = df.sample(n=min(sample_n, len(df)), random_state=42)
    return df.reset_index(drop=True)
