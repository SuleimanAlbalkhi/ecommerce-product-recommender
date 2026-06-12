# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does
Streamlit app (Python only, no backend) that demonstrates product recommendation:
- Cold-start users → popular products ranked by other users' view frequency
- Users with history → content-based filtering via TF-IDF cosine similarity

## Stack (exact versions)
- Python 3.14
- streamlit>=1.35
- pandas                  (CSV loading)
- scikit-learn            (TfidfVectorizer, cosine_similarity)
- numpy
- pytest                  (always run after changes)
- Pillow                  (placeholder product images)

## File roles — do NOT merge these
- app.py                    → Streamlit UI only. No ML logic here.
- recommender.py            → Pure functions. No Streamlit imports.
- data_loader.py            → Load the processed CSV. Returns a clean DataFrame.
- scripts/clean.py          → Offline cleaning pipeline (raw CSV → processed CSV). Never imported by the app.
- tests/test_rec.py         → pytest tests for recommender.py
- tests/test_data_loader.py → pytest tests for data_loader.py

## Run commands
```
streamlit run app.py     # start app
pytest tests/ -v         # always run after any change to recommender.py
pytest tests/ -v -k "test_name"  # run a single test
```

## Code conventions
- All functions have type hints and a one-line docstring
- `@st.cache_resource` for the TF-IDF matrix (expensive, build once)
- `@st.cache_data` for data loading
- `st.session_state["history"]` → list of product indices the user clicked
- `st.session_state["user_id"]` → simulated user (dropdown in sidebar)
- Keep each function under 30 lines

## Data
- App consumes `data/processed/ecommerceDataset_clean.csv` (UTF-8, header row, 27,715 rows)
- Columns: `category, name, description`
- Raw file `data/raw/ecommerceDataset.csv` (no header, ~50,000 rows, latin-1, mojibake artifacts)
  is handled ONLY by `scripts/clean.py` — the app never reads it
- `load_data(path, sample_n=...)` supports seeded sampling for fast dev runs; the app loads the full dataset

## Recommendation logic (implement exactly this)
### Cold start (no history)
```
popularity_recommender(df, all_histories, current_user_id=None, category=None, top_n=10)
→ aggregate view counts across all other users' histories (excluding current user)
→ rank products by view frequency, return top_n most viewed
→ if category given, filter first; pad with random samples if fewer than top_n results
```

### With history
```
history_recommender(df, tfidf_matrix, history_indices, category=None, top_n=10, decay=0.9)
→ exponentially weighted mean of TF-IDF vectors (recent views weighted higher)
→ cosine_similarity(weighted_mean, tfidf_matrix)
→ return top_n indices (exclude already-seen)
```

## What NOT to do
- No collaborative filtering (no user-item matrix needed for this demo)
- No external API calls
- No database — everything in memory
- No authentication
