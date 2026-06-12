import numpy as np
import pandas as pd
import pytest
from recommender import build_tfidf_matrix, history_recommender, popularity_recommender

PRODUCTS = [
    ("Electronics", "wireless bluetooth headphones noise cancelling"),
    ("Electronics", "smartphone 5G camera battery"),
    ("Electronics", "laptop ultrabook thin light processor"),
    ("Electronics", "smartwatch fitness tracker heart rate"),
    ("Clothing", "cotton t-shirt casual summer wear"),
    ("Clothing", "denim jeans slim fit blue"),
    ("Clothing", "winter jacket waterproof hood"),
    ("Books", "python programming data science guide"),
    ("Books", "machine learning deep neural network"),
    ("Books", "history ancient civilizations world"),
    ("Toys", "building blocks creative children set"),
    ("Toys", "remote control car fast race"),
    ("Home", "coffee maker espresso machine"),
]


@pytest.fixture
def df():
    rows = [(cat, f"Product {i}", desc) for i, (cat, desc) in enumerate(PRODUCTS)]
    return pd.DataFrame(rows, columns=["category", "name", "description"])


@pytest.fixture
def tfidf(df):
    _, matrix = build_tfidf_matrix(df)
    return matrix


def test_popularity_ranks_by_frequency(df):
    histories = {
        "u1": [0, 0, 4, 7],   # ignored when current_user_id="u1"
        "u2": [4, 5, 4],
        "u3": [4, 7, 5],
    }
    result = popularity_recommender(df, histories, current_user_id="u1", top_n=3)
    assert result.index.tolist()[0] == 4  # 4 viewed most by others (3 times)


def test_popularity_excludes_current_user(df):
    histories = {"alice": [0, 0, 0, 0], "bob": [9]}
    result = popularity_recommender(df, histories, current_user_id="alice", top_n=2)
    assert 9 in result.index
    assert result.index.tolist()[0] == 9  # only Bob's view counts


def test_popularity_fills_when_empty(df):
    result = popularity_recommender(df, {}, current_user_id="alice", top_n=5)
    assert len(result) == 5


def test_popularity_filters_category(df):
    histories = {"u2": [0, 4, 7, 10]}  # mixed categories
    result = popularity_recommender(
        df, histories, current_user_id="u1", category="Electronics", top_n=3
    )
    assert (result["category"] == "Electronics").all()


def test_history_returns_top_n(df, tfidf):
    result = history_recommender(df, tfidf, history_indices=[0], top_n=3)
    assert len(result) == 3


def test_history_excludes_seen(df, tfidf):
    seen = [0, 1, 2]
    result = history_recommender(df, tfidf, history_indices=seen, top_n=5)
    assert not any(idx in result.index for idx in seen)


def test_history_single_item(df, tfidf):
    result = history_recommender(df, tfidf, history_indices=[7], top_n=3)
    assert len(result) == 3
    assert 7 not in result.index


def test_history_preserves_original_index(df, tfidf):
    result = history_recommender(df, tfidf, history_indices=[0], top_n=5)
    assert result.index.isin(df.index).all()


def test_history_filters_by_category(df, tfidf):
    # history items are Electronics; filter to Books anyway
    result = history_recommender(
        df, tfidf, history_indices=[0, 1], category="Books", top_n=3
    )
    assert (result["category"] == "Books").all()


def test_history_single_item_decay_irrelevant(df, tfidf):
    # With one history item the weight is always [1.0] after normalisation — decay has no effect.
    result_09 = history_recommender(df, tfidf, history_indices=[7], decay=0.9, top_n=3)
    result_00 = history_recommender(df, tfidf, history_indices=[7], decay=0.0, top_n=3)
    assert result_09.index.tolist() == result_00.index.tolist()


def test_history_category_filter_below_top_n(df, tfidf):
    # Electronics has items 0-3; with 3 seen the only valid candidate is index 3.
    # top_n=4 must not crash and must return exactly that one valid item.
    result = history_recommender(
        df, tfidf, history_indices=[0, 1, 2], category="Electronics", top_n=4
    )
    assert len(result) == 1
    assert result.index.tolist() == [3]
    assert (result["category"] == "Electronics").all()


def test_history_decay_weights_recent():
    # Controlled vocabulary: items 0-1 use only "alpha", items 2-3 use only "beta".
    # With decay=0.0, history [0, 2] collapses to item 2 (beta) alone → top rec must be item 3.
    rows = [
        ("A", "P0", "alpha alpha alpha"),
        ("A", "P1", "alpha alpha"),
        ("B", "P2", "beta beta beta"),
        ("B", "P3", "beta beta"),
    ]
    mini_df = pd.DataFrame(rows, columns=["category", "name", "description"])
    _, mini_tfidf = build_tfidf_matrix(mini_df)
    result = history_recommender(mini_df, mini_tfidf, history_indices=[0, 2], decay=0.0, top_n=1)
    assert result.index.tolist() == [3]
