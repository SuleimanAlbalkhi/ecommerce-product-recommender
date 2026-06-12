from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_tfidf_matrix(df: pd.DataFrame) -> tuple[TfidfVectorizer, csr_matrix]:
    """Fit TF-IDF on product name + description and return (vectorizer, matrix)."""
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    text = df["name"].fillna("") + " " + df["description"].fillna("")
    matrix = vectorizer.fit_transform(text)
    return vectorizer, matrix


def popularity_recommender(
    df: pd.DataFrame,
    all_histories: dict[str, list[int]],
    current_user_id: str | None = None,
    category: str | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rank products by view count across all other users; return top_n most viewed. Rows beyond the ranked set are filled with a deterministic (seeded) sample."""
    counts: Counter = Counter()
    for uid, hist in all_histories.items():
        if uid == current_user_id:
            continue
        counts.update(hist)
    # history entries are row labels; df must keep the default RangeIndex from
    # load_data so that labels, positions, and TF-IDF matrix rows all coincide
    ranked = [idx for idx, _ in counts.most_common() if idx in df.index]
    candidates = df.loc[ranked]
    if category:
        candidates = candidates[candidates["category"] == category]
    if len(candidates) >= top_n:
        return candidates.head(top_n)
    pool = df[df["category"] == category] if category else df
    pool = pool.drop(candidates.index, errors="ignore")
    needed = top_n - len(candidates)
    filler = pool.sample(n=min(needed, len(pool)), random_state=42)
    return pd.concat([candidates, filler])


def history_recommender(
    df: pd.DataFrame,
    tfidf_matrix: csr_matrix,
    history_indices: list[int],
    category: str | None = None,
    top_n: int = 10,
    decay: float = 0.9,
) -> pd.DataFrame:
    """Return top_n products most similar to history, optionally filtered by category."""
    n = tfidf_matrix.shape[0]
    history_indices = [i for i in history_indices if 0 <= i < n]
    if not history_indices:
        return df.head(0)
    n_hist = len(history_indices)
    weights = np.array([decay ** (n_hist - 1 - i) for i in range(n_hist)], dtype=float)
    weights /= weights.sum()
    vectors = tfidf_matrix[history_indices]
    mean_vector = np.asarray(vectors.multiply(weights[:, np.newaxis]).sum(axis=0))
    scores = cosine_similarity(mean_vector, tfidf_matrix).flatten()
    scores[history_indices] = -np.inf
    if category:
        scores[df["category"].values != category] = -np.inf
    order = np.argsort(scores)[::-1]
    top_indices = [i for i in order[:top_n] if np.isfinite(scores[i])]
    # positional iloc is correct here because df rows align 1:1 with tfidf_matrix
    # rows (default RangeIndex guaranteed by load_data's reset_index)
    return df.iloc[top_indices]
