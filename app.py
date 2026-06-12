import io
import os

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from data_loader import load_data
from recommender import build_tfidf_matrix, history_recommender, popularity_recommender

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "processed", "ecommerceDataset_clean.csv")
# I need some fake users here. The popularity ranking only works
# when there are other users who already viewed something.
SIMULATED_USERS = ["Alice", "Bob", "Carol", "Dave"]
CATEGORY_COLORS = {
    "Household": "#E74C3C",
    "Books": "#5BAD6F",
    "Clothing & Accessories": "#E87040",
    "Electronics": "#4A90D9",
}


@st.cache_data
def load_products() -> pd.DataFrame:
    """Load and cache the full product dataset."""
    return load_data(DATA_PATH)


@st.cache_resource
def cached_tfidf(_df: pd.DataFrame):
    """Build and cache the TF-IDF matrix (leading underscore tells Streamlit not to hash _df)."""
    # I use cache_resource and not cache_data. The matrix is big,
    # so I want to build it one time and share it, not copy it around.
    return build_tfidf_matrix(_df)


def placeholder_image(category: str) -> io.BytesIO:
    """Return a solid-colour PNG BytesIO for a product card."""
    # the dataset has no product images, so I draw a simple colored box instead
    color = CATEGORY_COLORS.get(category, "#95A5A6")
    img = Image.new("RGB", (300, 150), color)
    draw = ImageDraw.Draw(img)
    draw.text((10, 65), category, fill="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def init_session_state(user_id: str) -> None:
    """Swap in the selected user's own history list on every render."""
    if "histories" not in st.session_state:
        st.session_state["histories"] = {}
    if user_id not in st.session_state["histories"]:
        st.session_state["histories"][user_id] = []
    st.session_state["user_id"] = user_id
    st.session_state["history"] = st.session_state["histories"][user_id]


def render_sidebar(categories: list[str]) -> tuple[str, str | None]:
    """Render user picker and category filter; return (user_id, category|None)."""
    with st.sidebar:
        st.header("Settings")
        user_id = st.selectbox("Simulated user", SIMULATED_USERS)
        raw_cat = st.selectbox("Filter category", ["All"] + categories)
        user_history = st.session_state.get("histories", {}).get(user_id, [])
        if st.button("Clear history"):
            st.session_state["histories"][user_id].clear()
            st.rerun()
        st.caption(f"Viewed: {len(user_history)} products")
    return user_id, (None if raw_cat == "All" else raw_cat)


def brief(text: str, max_chars: int = 80) -> str:
    """Return the first sentence of text, capped at max_chars characters."""
    # only for the look: when every card shows about the same text length,
    # the grid stays nice and even
    if not isinstance(text, str):
        return ""
    sentence = text.split(".")[0].strip()
    if len(sentence) > max_chars:
        sentence = sentence[:max_chars].rsplit(" ", 1)[0] + "…"
    return sentence


def render_product_card(row: pd.Series, idx: int) -> None:
    """Display one product tile; clicking View adds idx to history."""
    st.image(placeholder_image(row["category"]), use_container_width=True)
    st.caption(f"**{row['category']}**")
    st.markdown(f"**{row['name']}**")
    st.write(brief(row["description"]))
    if st.button("View", key=f"view_{idx}"):
        # I save every product only once. If I would count repeat clicks,
        # one single product could dominate the recommendations.
        if idx not in st.session_state["history"]:
            st.session_state["history"].append(idx)
        st.rerun()


def main() -> None:
    """Wire together data, recommenders, and UI for one Streamlit render pass."""
    st.set_page_config(page_title="Product Recommender", layout="wide")
    st.title("Product Recommendation Demo")

    df = load_products()
    _, tfidf_matrix = cached_tfidf(df)
    categories = sorted(df["category"].dropna().unique().tolist())

    # the sidebar runs before init_session_state, so the selected user
    # maybe has no history entry yet. That is why render_sidebar uses .get().
    user_id, selected_category = render_sidebar(categories)
    init_session_state(user_id)

    history = st.session_state["history"]

    if history:
        label = f"Recommended in {selected_category}" if selected_category else "Recommended for you"
        st.subheader(label)
        recs = history_recommender(
            df, tfidf_matrix, history, category=selected_category, top_n=10
        )
        if recs.empty and selected_category:
            # the user has no clicks in this category yet, so there is no
            # profile to match against. Popular products are the best guess.
            st.info(f"No history in {selected_category} yet — showing popular products.")
            recs = popularity_recommender(
                df,
                st.session_state["histories"],
                current_user_id=user_id,
                category=selected_category,
                top_n=10,
            )
        # the category filter plus removing already seen products can leave
        # less than 10 results. I show a short info text, so the half empty
        # grid does not look like a bug.
        elif len(recs) < 10:
            st.info(f"Only {len(recs)} new recommendations available for this filter.")
    else:
        label = f"Popular in {selected_category}" if selected_category else "Popular with other users"
        st.subheader(label)
        recs = popularity_recommender(
            df,
            st.session_state["histories"],
            current_user_id=user_id,
            category=selected_category,
            top_n=10,
        )

    cols = st.columns(5)
    for i, (idx, row) in enumerate(recs.iterrows()):
        with cols[i % 5]:
            render_product_card(row, idx)


if __name__ == "__main__":
    main()
