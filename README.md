# Product Recommendation Demo

A Streamlit app that demonstrates content-based product recommendation on a real
e-commerce dataset (~27,700 products in 4 categories). Everything runs in memory —
no backend, no database, no external APIs.

## How it recommends

| Situation | Strategy |
|---|---|
| New user (no clicks yet) | **Popularity**: products ranked by how often *other* simulated users viewed them |
| User with click history | **Category-segmented TF-IDF profiles**: one profile vector per viewed category, built as a recency-weighted mean of TF-IDF vectors. Time decay applies only within each category, so a click in Books never dilutes the Electronics profile. Recommendation slots are split across categories proportional to the user's view counts. |

Pick a simulated user in the sidebar, click **View** on products, and watch the
recommendations adapt. Filtering to a category you have never clicked falls back to
the popularity list.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py        # start the app
pytest tests/ -v            # run the test suite
```

Developed and tested on Python 3.14.

## Project layout

| File | Role |
|---|---|
| `app.py` | Streamlit UI only — no ML logic |
| `recommender.py` | Pure recommendation functions (TF-IDF, profiles, ranking) |
| `data_loader.py` | Loads the processed CSV into a clean DataFrame |
| `scripts/clean.py` | Offline pipeline: raw CSV → processed CSV (mojibake repair, dedup, name extraction) |
| `tests/` | pytest suites for the recommender and the data loader |
| `data/raw/` | Original dataset (latin-1, no header, ~50k rows) — only read by `scripts/clean.py` |
| `data/processed/` | Cleaned UTF-8 dataset consumed by the app |

## Data pipeline

The app never touches the raw file. To rebuild the processed dataset:

```bash
python scripts/clean.py
```

The pipeline drops empty and duplicate descriptions, repairs UTF-8/CP1252 mojibake,
strips URLs and HTML entities, truncates very long descriptions at sentence
boundaries, and extracts a product name column.
