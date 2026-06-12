import os
import pytest
import pandas as pd
from data_loader import load_data

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "ecommerceDataset_clean.csv")


def test_load_returns_dataframe():
    df = load_data(DATA_PATH, sample_n=50)
    assert isinstance(df, pd.DataFrame)


def test_columns_present():
    df = load_data(DATA_PATH, sample_n=50)
    assert list(df.columns) == ["category", "name", "description"]


def test_sample_n_respected():
    df = load_data(DATA_PATH, sample_n=50)
    assert len(df) == 50


def test_reproducible_sample():
    df1 = load_data(DATA_PATH, sample_n=50)
    df2 = load_data(DATA_PATH, sample_n=50)
    assert df1["category"].tolist() == df2["category"].tolist()
