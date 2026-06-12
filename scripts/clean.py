"""Dataset cleaning pipeline for data/raw/ecommerceDataset.csv.

The numbered labels below (2.x / R*) are the step IDs printed in the log
output, listed in execution order.

Phase 1: clean()
  2.1  Drop rows with empty / NaN descriptions
  2.2  Clean every description text:
       a) repair UTF-8 bytes that were read as latin-1 (e.g. Ã¢â€šÂ¬ → €)
       b) fix CP1252 mojibake (e.g. â€™ → ')
       c) resolve HTML entities (&amp; → &, &lt; → < etc.)
       d) strip URLs (http://, www.) and e-mail addresses
       e) normalise whitespace and tabs (\\t, \\r, \\n → space; repeated spaces → one)
  2.3  Drop descriptions that occur in more than one category (cross-category conflict)
  2.4  Drop exact (category, description) duplicates

Phase 2: refine()  [applied after clean()]
  R1  Truncate descriptions >500 chars at the last sentence boundary; dedup again afterwards
  R2  Add 'name' column: product name extracted from the leading part of the description

Short descriptions are kept on purpose. The extracted name alone gives
enough TF-IDF signal, so a minimum length filter is not needed.

Input:  data/raw/ecommerceDataset.csv             (no header, ~50,000 rows, latin-1)
Output: data/processed/ecommerceDataset_clean.csv (UTF-8)
Columns: category, name, description
"""

import html
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw" / "ecommerceDataset.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT = OUT_DIR / "ecommerceDataset_clean.csv"


# map from broken cp1252 bytes to the correct unicode characters.
# I wrote this map myself instead of using the ftfy library, because
# the broken characters in this dataset are few and well known.
CP1252_C1_MAP = {
    0x80: "€", 0x82: "‚", 0x83: "ƒ", 0x84: "„",
    0x85: "…", 0x86: "†", 0x87: "‡", 0x88: "ˆ",
    0x89: "‰", 0x8a: "Š", 0x8b: "‹", 0x8c: "Œ",
    0x8e: "Ž", 0x91: "'", 0x92: "'", 0x93: "“",
    0x94: "”", 0x95: "•", 0x96: "–", 0x97: "—",
    0x98: "˜", 0x99: "™", 0x9a: "š", 0x9b: "›",
    0x9c: "œ", 0x9e: "ž", 0x9f: "Ÿ",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
WS_RE = re.compile(r"\s+")

# Product-name separators, discovered by studying 400+ examples (100/category).
# Order does NOT determine priority. The earliest match in the text wins.
# Tuples of (compiled_pattern, include_period_in_name)
_NAME_SEPS: list[tuple[re.Pattern[str], bool]] = [
    # Books section headers
    (re.compile(r"\bReview\b"),                    False),
    (re.compile(r"About the Author"),              False),
    (re.compile(r"About the Book"),                False),
    (re.compile(r"Book Description"),              False),
    (re.compile(r"From the Back Cover"),           False),
    (re.compile(r"From the Inside Flap"),          False),
    (re.compile(r"From the Author"),               False),
    # Generic section headers (all categories)
    (re.compile(r"Product [Dd]escription"),        False),
    (re.compile(r"Descriptions?\s+\d"),            False),
    (re.compile(r"\bDescription\s*:", re.I),       False),
    (re.compile(r"\bFeatures\s*:", re.I),          False),
    # Attribute key:value separators
    (re.compile(r"Style [Nn]ame:"),                False),
    (re.compile(r"Flavou?r [Nn]ame:", re.I),       False),
    (re.compile(r"Colou?r [Nn]ame:", re.I),        False),
    (re.compile(r"Colou?r:", re.I),                False),
    (re.compile(r"Size [Nn]ame:", re.I),           False),
    (re.compile(r"Size:", re.I),                   False),
    (re.compile(r"\bMaterial\s*:", re.I),          False),
    (re.compile(r"\bDimensions?\s*:", re.I),       False),
    (re.compile(r"\bCapacity\s*:", re.I),          False),
    (re.compile(r"Configuration:", re.I),          False),
    (re.compile(r"Specifications?:", re.I),        False),
    (re.compile(r"Item [Pp]ackage"),               False),
    (re.compile(r"\|"),                            False),
    # Structural: closing paren followed immediately by body text (capital letter)
    # m.start() lands on the space after ')'; desc[:m.start()] retains the ')'
    (re.compile(r"(?<=\)) (?=[A-Z])"),             False),
    # Sentence boundary: keep the period as part of the name
    (re.compile(r"\. (?=[A-Z])"),                  True),
]


def _fix_utf8_mojibake(s: str) -> str:
    """Repair UTF-8 bytes that were mistakenly decoded as latin-1."""
    # only some rows have this double encoding problem. For normal text
    # the encode/decode fails and I simply keep the original string.
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return s


def clean_description(s: str) -> str:
    """Apply the text-level cleaning steps (2.2 a-e) to one description."""
    s = _fix_utf8_mojibake(s)
    s = s.translate(CP1252_C1_MAP)
    s = html.unescape(s)
    s = URL_RE.sub(" ", s)
    s = EMAIL_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s


def extract_product_name(desc: str) -> str:
    """Return the product name extracted from the leading part of desc."""
    best_pos: int = len(desc)
    best_keep_period: bool = False
    for pat, keep_period in _NAME_SEPS:
        m = pat.search(desc)
        if m and m.start() < best_pos:
            best_pos = m.start()
            best_keep_period = keep_period
    if best_pos == len(desc):
        return desc.strip()
    cut = best_pos + 1 if best_keep_period else best_pos
    name = desc[:cut].strip()
    return name if name else desc.strip()


def log(msg: str) -> None:
    print(msg, flush=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 1: drop empty/duplicate/conflicting rows and clean every description."""
    n0 = len(df)

    mask_empty = df["description"].fillna("").str.strip().eq("")
    n_empty = int(mask_empty.sum())
    df = df.loc[~mask_empty].copy()
    log(f"2.1 dropped empty/NaN descriptions: {n_empty}")

    log("2.2 cleaning descriptions ...")
    df["description"] = df["description"].map(clean_description)

    mask_empty2 = df["description"].eq("")
    n_empty2 = int(mask_empty2.sum())
    if n_empty2:
        df = df.loc[~mask_empty2].copy()
        log(f"    dropped {n_empty2} rows that became empty after cleaning")

    cats_per_desc = df.groupby("description")["category"].nunique()
    conflict_descs = cats_per_desc[cats_per_desc > 1].index
    mask_conflict = df["description"].isin(conflict_descs)
    n_conflict_rows = int(mask_conflict.sum())
    df = df.loc[~mask_conflict].copy()
    log(
        f"2.3 dropped cross-category duplicates: "
        f"{len(conflict_descs)} descriptions, {n_conflict_rows} rows"
    )

    before = len(df)
    df = df.drop_duplicates(subset=["category", "description"], keep="first").copy()
    log(f"2.4 dropped exact (cat,desc) duplicates: {before - len(df):,}")

    log(f"    rows after clean(): {len(df):,}  (from {n0:,})")
    return df


def truncate_at_sentence(s: str, max_len: int = 500) -> str:
    """Truncate s to at most max_len chars, cutting at the latest sentence-like boundary."""
    # 500 characters are enough. A very long description gets too much
    # weight in the TF-IDF similarity and pushes other products away.
    if len(s) <= max_len:
        return s

    prefix = s[:max_len]

    # I try to cut at a real sentence end first. If there is none, I take
    # a weaker break like a comma. A cut in the middle of a word is the
    # last option.
    for separator in [". ", "! ", "? ", "; ", ", ", ".", ",", " "]:
        pos = prefix.rfind(separator)
        if pos != -1:
            return s[:pos + len(separator)].rstrip()

    return prefix.rstrip() + "…"


def refine(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 2: truncate long descriptions, dedup again, and extract the 'name' column."""
    desc_len = df["description"].str.len()
    mask_long = desc_len > 500
    n_long = int(mask_long.sum())
    df.loc[mask_long, "description"] = (
        df.loc[mask_long, "description"].map(truncate_at_sentence)
    )
    log(f"R1 flag_long (>500 chars) truncated at sentence boundary: {n_long:,}")
    # Truncation can create new (category, description) duplicates, so I
    # remove them one more time
    before = len(df)
    df = df.drop_duplicates(subset=["category", "description"], keep="first").copy()
    log(f"   post-truncation dedup: {before - len(df)} rows removed")

    df["name"] = df["description"].map(extract_product_name)
    log("R2 product_name column added")
    return df[["category", "name", "description"]]


def main() -> int:
    """Run the full pipeline: read raw CSV, clean(), refine(), write processed CSV."""
    log(f"Reading {RAW} ...")
    df = pd.read_csv(
        RAW,
        header=None,
        names=["category", "description"],
        dtype=str,
        encoding="latin-1",
        # without this, pandas reads text like "None" or "NA" as missing
        # values and real products would get lost
        keep_default_na=False,
        na_values=[""],
    )
    log(f"  loaded rows: {len(df):,}")

    df = clean(df)
    df = refine(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    log(f"\nWrote {OUT}")
    log(f"  final rows: {len(df):,}")
    log(f"  desc len max: {df['description'].str.len().max()}")
    log("\nCategory distribution:")
    dist = df["category"].value_counts()
    for cat, count in dist.items():
        log(f"  {cat:<25} {count:>6,}  ({count/len(df):.2%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
