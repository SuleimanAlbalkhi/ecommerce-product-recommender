# Implementation Plan

## 1. `scripts/clean.py`
Standalone CLI pipeline that turns the raw Kaggle dump into the CSV the app consumes. Run once; the app never touches the raw file.

Two phases (step IDs match the log output):
- **Phase 1 — `clean()`**: repair UTF-8/CP1252 mojibake, drop empty descriptions, drop exact `(category, description)` duplicates, drop descriptions appearing in more than one category, normalise whitespace, resolve HTML entities, strip URLs and e-mail addresses.
- **Phase 2 — `refine()`**: truncate descriptions >500 chars at a sentence boundary (then dedup again) and extract a `name` column from the leading part of each description.

Input: `data/raw/ecommerceDataset.csv` (no header, ~50,000 rows, latin-1).
Output: `data/processed/ecommerceDataset_clean.csv` (UTF-8, header, columns `category, name, description`, 27,715 rows).

---

## 2. `data_loader.py`
Reads the **processed** CSV and returns a ready-to-use DataFrame. All text cleaning already happened in `scripts/clean.py`.

| Function | Signature | Description |
|---|---|---|
| `load_data` | `(path: str, sample_n: int \| None = None) -> pd.DataFrame` | Read the cleaned CSV (header row, columns `category, name, description`), drop rows with missing category, strip category whitespace, optionally sample `sample_n` rows with a fixed seed, and reset to a default RangeIndex. |

The `reset_index` matters: downstream, row labels, positions, and TF-IDF matrix rows must all coincide.

---

## 3. `recommender.py`
Pure ML logic — no Streamlit imports allowed. All functions are stateless and accept a DataFrame plus pre-built artefacts.

| Function | Signature | Description |
|---|---|---|
| `build_tfidf_matrix` | `(df: pd.DataFrame) -> tuple[TfidfVectorizer, csr_matrix]` | Fit a `TfidfVectorizer` (max 5,000 features, English stop words) on `name + " " + description`, return `(vectorizer, tfidf_matrix)`. |
| `popularity_recommender` | `(df, all_histories: dict[str, list[int]], current_user_id: str \| None = None, category: str \| None = None, top_n: int = 10) -> pd.DataFrame` | Cold-start path: aggregate view counts across all *other* users' histories, rank by frequency, return the `top_n` most viewed. Optional category filter; pads with a deterministic (seeded) sample when fewer than `top_n` ranked candidates exist. |
| `history_recommender` | `(df, tfidf_matrix, history_indices: list[int], category: str \| None = None, top_n: int = 10, decay: float = 0.9) -> pd.DataFrame` | Personalised path: exponentially weighted mean of the history's TF-IDF vectors (recent views weighted higher via `decay`), cosine similarity against the full matrix, return the `top_n` best matches excluding already-seen indices, optionally filtered by category. Preserves the original DataFrame index. |

---

## 4. `app.py`
Streamlit entry point — UI only. Calls `data_loader` and `recommender` functions; contains no ML logic itself.

| Section | Description |
|---|---|
| `load_products()` | `@st.cache_data`-wrapped `load_data` call: loads the full processed dataset once. |
| `cached_tfidf(_df)` | `@st.cache_resource` wrapper around `build_tfidf_matrix` (the leading underscore on `_df` tells Streamlit not to hash the DataFrame). Lives in `app.py` only, never in `recommender.py`. |
| `init_session_state(user_id)` | Maintains `st.session_state["histories"]` (one view-history list per simulated user) and points `st.session_state["history"]` at the selected user's list. |
| `render_sidebar(categories)` | Dropdown for the simulated user (Alice/Bob/Carol/Dave), category filter select-box, clear-history button. Returns `(user_id, selected_category)`. |
| `render_product_card(row, idx)` | One product tile: Pillow-generated placeholder image, category, name, truncated description. The View button appends `idx` to the user's history and triggers a rerun. |
| `main()` | Loads data + TF-IDF (cached), renders the sidebar, then branches: empty history → `popularity_recommender` (cold start), non-empty → `history_recommender`. Renders results in a 5-column grid. |

---

## 5. Tests
Small synthetic DataFrames so tests run instantly without touching the real dataset.

- `tests/test_rec.py` — `recommender.py` behaviour: popularity ranking/frequency, exclusion of the current user, filler when histories are empty, category filtering, top-n counts, exclusion of seen items, index preservation, decay weighting, and category-filter edge cases below `top_n`.
- `tests/test_data_loader.py` — `load_data` against the real processed CSV: returns a DataFrame, expected columns, `sample_n` respected, sampling reproducible (fixed seed).
