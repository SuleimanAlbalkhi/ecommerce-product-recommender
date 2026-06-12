# Product Recommendation System

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-%E2%89%A51.49-FF4B4B)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-F7931E)
![Tests](https://img.shields.io/badge/tests-20%20passing-brightgreen)

A content-based product recommendation demo built with Streamlit. It serves
personalized recommendations over a real e-commerce dataset of **~27,700 products**
across four categories (Household, Books, Clothing & Accessories, Electronics) —
entirely in memory, with no backend, database, or external API.

## Key features

- **Cold-start handling** — new users see products ranked by what other users viewed
- **Category-segmented user profiles** — one TF-IDF profile vector per viewed
  category, which prevents *centroid dilution* when a user browses mixed categories
- **Per-category time decay** — recent clicks weigh more, but a click in Books never
  ages the Electronics profile
- **Proportional slot mixing** — the recommendation grid mirrors the user's actual
  interest distribution across categories
- **Simulated multi-user sessions** — switch users in the sidebar; each keeps an
  independent click history
- **Reproducible data pipeline** — one script turns the raw, mojibake-ridden CSV
  into the clean dataset the app consumes

## How it works

### Cold start (no click history)

Products are ranked by view frequency across all *other* users' histories
(`popularity_recommender`). An optional category filter is applied first; if fewer
than `top_n` products remain, the grid is padded with a deterministic seeded sample
so reruns stay stable.

### With history (content-based)

1. **Profile building** — the click history is grouped by product category. For each
   category, a profile vector is computed as the normalized weighted mean of the
   products' TF-IDF vectors, with exponential recency weights applied *within that
   category only*:

   ```
   weight(click_i) = decay^(n - 1 - i),   decay = 0.9
   ```

   The newest click in a category gets weight 1.0; older clicks fade out gradually.

2. **Scoring** — candidate products are ranked by cosine similarity against their own
   category's profile vector. Already-seen products are excluded.

3. **Mixing** — without a category filter, the `top_n` slots are apportioned across
   the viewed categories proportional to view counts (largest-remainder method, ties
   resolved toward the most recently clicked category). With a filter, only that
   category's profile is used; if the user has no history there, the app falls back
   to the popularity ranking.

The TF-IDF matrix (5,000 features, English stop words removed, product name +
description) is built once and cached with `@st.cache_resource`.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py

# 3. Run the test suite
pytest tests/ -v
```

Developed and tested on Python 3.14.

**Try it:** pick a simulated user in the sidebar, click **View** on a few products
from different categories, and watch the recommendations adapt — the grid mix follows
your click distribution, and each category stays sharp instead of blurring together.

## Project structure

```
├── app.py                     # Streamlit UI only — no ML logic
├── recommender.py             # Pure recommendation functions (no Streamlit imports)
├── data_loader.py             # Loads the processed CSV into a clean DataFrame
├── scripts/
│   └── clean.py               # Offline pipeline: raw CSV → processed CSV
├── tests/
│   ├── test_rec.py            # Recommender unit tests
│   └── test_data_loader.py    # Data loader unit tests
└── data/
    ├── raw/                   # Original dataset (latin-1, no header, ~50k rows)
    └── processed/             # Cleaned UTF-8 dataset consumed by the app
```

The separation is strict by design: `recommender.py` contains only pure, testable
functions; `app.py` contains only UI; the raw dataset is read exclusively by the
offline pipeline.

## Data pipeline

The app never touches the raw file. To rebuild the processed dataset from scratch:

```bash
python scripts/clean.py
```

The pipeline:

1. Drops empty and NaN descriptions
2. Repairs encoding damage (double-encoded UTF-8, CP1252 mojibake), resolves HTML
   entities, strips URLs and e-mail addresses, normalizes whitespace
3. Removes descriptions that appear in more than one category (label conflicts)
4. Deduplicates exact (category, description) pairs
5. Truncates very long descriptions at sentence boundaries and extracts a product
   name column

Result: 27,715 clean rows with `category, name, description` columns.

## Design decisions & scope

| Decision | Rationale |
|---|---|
| Content-based only, no collaborative filtering | The demo focuses on TF-IDF profiles; no user-item matrix is needed |
| Everything in memory | Zero-setup demo: clone, install, run |
| Per-category profiles instead of one global vector | A single mean vector over mixed-category clicks lands "between" categories and matches nothing well |
| Seeded sampling for grid padding | Streamlit reruns the script on every interaction; unseeded randomness would make the UI look broken |

## Tech stack

Python · Streamlit · pandas · scikit-learn (TfidfVectorizer, cosine similarity) ·
NumPy · SciPy · Pillow · pytest
