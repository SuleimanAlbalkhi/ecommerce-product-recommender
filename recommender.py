from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_tfidf_matrix(df: pd.DataFrame) -> tuple[TfidfVectorizer, csr_matrix]:
    """Fit TF-IDF on product name + description and return (vectorizer, matrix)."""
    # 5000 words are enough. The full vocabulary is much bigger, but the
    # extra words are rare and only cost memory. Stop words like "the"
    # appear in every text, so they carry no information.
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    # I use the name plus the description. The name is short but very
    # precise, so products with a weak description still get a good match.
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
    """Rank products by other users' view counts; pad with a seeded sample if needed."""
    counts: Counter = Counter()
    for uid, hist in all_histories.items():
        if uid == current_user_id:
            continue
        counts.update(hist)
    # the history stores row labels. This only works because load_data
    # resets the index, so label, position and matrix row are the same number.
    ranked = [idx for idx, _ in counts.most_common() if idx in df.index]
    candidates = df.loc[ranked]
    if category:
        candidates = candidates[candidates["category"] == category]
    if len(candidates) >= top_n:
        return candidates.head(top_n)
    pool = df[df["category"] == category] if category else df
    pool = pool.drop(candidates.index, errors="ignore")
    needed = top_n - len(candidates)
    # the seed is fixed on purpose. Streamlit runs the whole script again
    # after every click, and without the seed the filler products would
    # change every time. That looks broken to the user.
    filler = pool.sample(n=min(needed, len(pool)), random_state=42)
    return pd.concat([candidates, filler])


def build_category_profiles(
    df: pd.DataFrame,
    tfidf_matrix: csr_matrix,
    history_indices: list[int],
    decay: float = 0.9,
) -> dict[str, np.ndarray]:
    """Build one decay weighted TF-IDF profile vector per viewed category."""
    n = tfidf_matrix.shape[0]
    # the history can contain old indices, for example when the dataset
    # changed while a session was open. I drop them instead of crashing.
    history_indices = [i for i in history_indices if 0 <= i < n]
    # THE KEY IDEA: one profile per category instead of one global mean.
    # A global mean over mixed clicks lands between the categories in
    # vector space, so it is close to nothing (centroid dilution). I group
    # the clicks by category and run the decay inside each group only.
    # A click in Books must not age the Electronics profile.
    groups: dict[str, list[int]] = {}
    for idx in history_indices:
        groups.setdefault(df["category"].iat[idx], []).append(idx)
    profiles: dict[str, np.ndarray] = {}
    for cat, idxs in groups.items():
        k = len(idxs)
        weights = np.array([decay ** (k - 1 - i) for i in range(k)], dtype=float)
        # normalize, so the profile does not depend on the history length
        weights /= weights.sum()
        vectors = tfidf_matrix[idxs]
        profiles[cat] = np.asarray(vectors.multiply(weights[:, np.newaxis]).sum(axis=0))
    return profiles


def _allocate_slots(
    cat_counts: dict[str, int], last_seen: dict[str, int], top_n: int
) -> dict[str, int]:
    """Split top_n slots across categories proportional to view counts (largest remainder)."""
    total = sum(cat_counts.values())
    quotas = {cat: top_n * c / total for cat, c in cat_counts.items()}
    slots = {cat: int(q) for cat, q in quotas.items()}
    leftover = top_n - sum(slots.values())
    # remainder ties go to the category the user clicked most recently,
    # because that is the freshest signal about their current interest
    order = sorted(
        quotas,
        key=lambda cat: (quotas[cat] - slots[cat], last_seen[cat]),
        reverse=True,
    )
    for cat in order[:leftover]:
        slots[cat] += 1
    return slots


def _top_in_category(
    df: pd.DataFrame,
    tfidf_matrix: csr_matrix,
    profile: np.ndarray,
    category: str,
    seen: set[int],
    k: int,
) -> list[int]:
    """Return up to k unseen product indices of one category, ranked by profile similarity."""
    scores = cosine_similarity(profile, tfidf_matrix).flatten()
    # I set excluded products to -inf instead of removing them from the
    # array. Removing them would break the position match with the matrix.
    scores[df["category"].values != category] = -np.inf
    scores[list(seen)] = -np.inf
    order = np.argsort(scores)[::-1]
    return [i for i in order[:k] if np.isfinite(scores[i])]


def history_recommender(
    df: pd.DataFrame,
    tfidf_matrix: csr_matrix,
    history_indices: list[int],
    category: str | None = None,
    top_n: int = 10,
    decay: float = 0.9,
) -> pd.DataFrame:
    """Recommend top_n products from per-category profiles, optionally for one category."""
    profiles = build_category_profiles(df, tfidf_matrix, history_indices, decay)
    if not profiles:
        return df.head(0)
    n = tfidf_matrix.shape[0]
    seen = {i for i in history_indices if 0 <= i < n}
    if category:
        # no clicks in this category means no profile to match against.
        # I return nothing and the app falls back to the popularity list.
        if category not in profiles:
            return df.head(0)
        picks = _top_in_category(df, tfidf_matrix, profiles[category], category, seen, top_n)
        return df.iloc[picks]
    cats = [df["category"].iat[i] for i in history_indices if 0 <= i < n]
    last_seen = {cat: pos for pos, cat in enumerate(cats)}
    slots = _allocate_slots(Counter(cats), last_seen, top_n)
    picks = []
    spare = 0
    # a category with few unseen products can not always fill its quota.
    # I carry the spare slots over to the next category, so the grid
    # stays as full as possible.
    for cat in sorted(slots, key=lambda c: (slots[c], last_seen[c]), reverse=True):
        got = _top_in_category(df, tfidf_matrix, profiles[cat], cat, seen, slots[cat] + spare)
        spare = slots[cat] + spare - len(got)
        picks.extend(got)
    # iloc works here because the df rows and the matrix rows are in the
    # same order (load_data resets the index)
    return df.iloc[picks]
