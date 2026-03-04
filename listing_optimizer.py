"""
Listing Optimizer - Proof of Concept
Uses Listings.csv only. Analyzes guest reviews for positive and negative sentiment.
- Positive: used for content (generate updated listing title).
- Negative: shown to clients to analyze work orders, maintenance, and owner scoring.
- Analyses are stored in a JSON "table"; UI shows last refreshed + option to refresh + history (last 6 months).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from groq import Groq

# --- Config: from TOML secrets (Streamlit Cloud) or .env ---
LISTINGS_CSV = "Listings_1.csv"
# Persistent "table" for analyses (simulates DB; one JSON file)
DATA_DIR = Path(__file__).resolve().parent / "data"
ANALYSES_JSON = DATA_DIR / "sentiment_analyses.json"
MONTHS_HISTORY = 6


def _get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default) or default
    except Exception:
        return os.getenv(key, default)


LLAMA_API_KEY = _get_secret("LLAMA_API_KEY", os.getenv("LLAMA_API_KEY", ""))
LLAMA_MODEL = _get_secret("LLAMA_MODEL", os.getenv("LLAMA_MODEL", "llama-3.3-70b-versatile"))

# Session state keys
SENTIMENT_CACHE = "sentiment_cache"
WORK_LOGS = "work_logs"
VIEWING_ANALYSIS_INDEX = "viewing_analysis_index"  # 0 = latest, 1+ = history
LAST_LISTING_ID = "last_listing_id"  # reset history view when property changes


def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_analyses_store() -> list[dict]:
    """Load full analyses table from JSON."""
    _ensure_data_dir()
    if not ANALYSES_JSON.exists():
        return []
    try:
        with open(ANALYSES_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_analysis(listing_id: str, listing_name: str, positive: str, negative: str) -> dict:
    """Append one analysis to the store; return the new record."""
    _ensure_data_dir()
    now = datetime.now(timezone.utc).isoformat()
    record = {
        "listing_id": listing_id,
        "listing_name": listing_name,
        "analyzed_at": now,
        "positive": positive,
        "negative": negative,
    }
    rows = load_analyses_store()
    rows.append(record)
    with open(ANALYSES_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return record


def get_analyses_for_listing(listing_id: str, months: int = MONTHS_HISTORY) -> list[dict]:
    """Analyses for this listing in the last `months` months, newest first."""
    rows = load_analyses_store()
    cutoff = datetime.now(timezone.utc) - timedelta(days=months * 31)
    out = []
    for r in rows:
        if r.get("listing_id") != listing_id:
            continue
        try:
            at = datetime.fromisoformat(r["analyzed_at"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        if at >= cutoff:
            out.append(r)
    out.sort(key=lambda x: x["analyzed_at"], reverse=True)
    return out


def relative_time(iso_ts: str) -> str:
    """Human-readable 'last refreshed' e.g. 'just now', '5 min ago', '2 hours ago', 'Mar 3, 2025'."""
    try:
        at = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        if at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - at
        if delta.total_seconds() < 60:
            return "just now"
        if delta.total_seconds() < 3600:
            m = int(delta.total_seconds() / 60)
            return f"{m} min ago"
        if delta.days == 0:
            h = int(delta.total_seconds() / 3600)
            return f"{h} hour{'s' if h != 1 else ''} ago"
        if delta.days == 1:
            return "yesterday"
        if delta.days < 7:
            return f"{delta.days} days ago"
        return at.strftime("%b %d, %Y %I:%M %p")
    except Exception:
        return iso_ts[:19] if len(iso_ts) > 19 else iso_ts


@st.cache_data
def load_listings():
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(LISTINGS_CSV, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        df = pd.read_csv(LISTINGS_CSV, encoding="utf-8", encoding_errors="replace")
    if "Review" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "Review"})
    df = df[df["Review"].notna() & (df["Review"].astype(str).str.strip() != "")].copy()
    return df


def extract_sentiment(client: Groq, review_text: str) -> tuple[str, str]:
    """Returns (positive_bullets, negative_bullets). Both as bullet points from the raw review."""
    prompt = """Analyze this guest review and output two sections. Extract directly from the raw review text; list as short bullet points, one per line, each line starting with "- ".

First section: POSITIVE — only what guests liked or praised. If none, write "- No clear positive highlights found."

Second section: NEGATIVE — what guests complained about or found lacking (for work orders, maintenance, owner scoring). If none, write "- No clear negative feedback found."

Format your reply exactly like this (include the labels on their own lines):
POSITIVE:
- point one
- point two

NEGATIVE:
- point one
- point two

Review:
"""
    prompt += review_text[:4000]
    try:
        r = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
        )
        text = (r.choices[0].message.content or "").strip()
        positive, negative = "", ""
        if "NEGATIVE:" in text:
            parts = text.split("NEGATIVE:", 1)
            positive = parts[0].replace("POSITIVE:", "").strip()
            negative = parts[1].strip()
        else:
            positive = text.replace("POSITIVE:", "").strip()
        return positive, negative
    except Exception as e:
        return f"Error: {e}", ""


def generate_listing_content(client: Groq, positive_bullets: str, property_type: str, city: str) -> tuple[str, str]:
    """Returns (title, description_paragraphs) for the listing. Description is 2–3 short paragraphs."""
    prompt = f"""Using ONLY these positive guest highlights from reviews, create an Airbnb listing.

1) First line: a catchy listing TITLE, maximum 200 characters. Include property type and location if helpful.
2) Then write 2–3 short, flowing PARAGRAPHS for the listing description (what guests will read). Use a warm, inviting tone. Base the content only on the positive highlights below.

Format your reply exactly like this (include the labels on their own lines):
TITLE:
(one line, max 200 chars)

DESCRIPTION:
(2–3 paragraphs here)

Positive highlights from reviews:
{positive_bullets}

Property type: {property_type}. City: {city}.
"""
    try:
        r = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
        )
        text = (r.choices[0].message.content or "").strip()
        title, description = "", ""
        if "DESCRIPTION:" in text:
            parts = text.split("DESCRIPTION:", 1)
            title = parts[0].replace("TITLE:", "").strip().strip('"')[:200]
            description = parts[1].strip()
        else:
            title = text.replace("TITLE:", "").strip().strip('"')[:200]
        return title, description
    except Exception as e:
        return f"Error: {e}", ""


def main():
    st.set_page_config(page_title="Listing Optimizer", page_icon="🏠", layout="wide")
    st.title("🏠 Listing Optimizer")
    st.caption("Stored analyses · Last refreshed · Refresh on demand · History (last 6 months)")

    if WORK_LOGS not in st.session_state:
        st.session_state[WORK_LOGS] = []
    if VIEWING_ANALYSIS_INDEX not in st.session_state:
        st.session_state[VIEWING_ANALYSIS_INDEX] = 0
    if LAST_LISTING_ID not in st.session_state:
        st.session_state[LAST_LISTING_ID] = None

    df = load_listings()
    if df.empty:
        st.error("No listings with reviews found.")
        return

    df["_label"] = df["name"].fillna("") + " (ID: " + df["listing_id"].astype(str) + ")"
    options = df["_label"].tolist()
    choice = st.selectbox("Select property", options, index=0, key="property_select")
    idx = df[df["_label"] == choice].index[0]
    row = df.loc[idx]
    listing_id = str(row.get("listing_id", ""))
    listing_name = str(row.get("name", ""))
    review_text = str(row.get("Review", ""))

    if st.session_state.get(LAST_LISTING_ID) != listing_id:
        st.session_state[LAST_LISTING_ID] = listing_id
        st.session_state[VIEWING_ANALYSIS_INDEX] = 0

    analyses = get_analyses_for_listing(listing_id, months=MONTHS_HISTORY)

    if not analyses:
        if not LLAMA_API_KEY:
            st.warning("Set **LLAMA_API_KEY** in secrets or .env to run sentiment analysis.")
        else:
            if st.button("Run sentiment analysis", type="primary", use_container_width=True):
                with st.spinner("Analyzing positive and negative sentiment…"):
                    client = Groq(api_key=LLAMA_API_KEY)
                    pos, neg = extract_sentiment(client, review_text)
                    save_analysis(listing_id, listing_name, pos, neg)
                st.rerun()
    else:
        def _history_label(i: int, r: dict) -> str:
            at = r["analyzed_at"]
            try:
                dt = datetime.fromisoformat(at.replace("Z", "+00:00"))
                return dt.strftime("%b %d, %Y %I:%M %p") if i > 0 else f"Latest — {dt.strftime('%b %d, %Y %I:%M %p')}"
            except Exception:
                return at[:16] if i > 0 else f"Latest — {at[:16]}"

        viewing_index = st.session_state.get(VIEWING_ANALYSIS_INDEX, 0)
        if viewing_index >= len(analyses):
            viewing_index = 0
        current = analyses[viewing_index]
        positive_text = current["positive"]
        negative_text = current["negative"]
        analyzed_at = current["analyzed_at"]

        st.subheader("Review sentiment")
        col_meta, col_btn = st.columns([3, 1])
        with col_meta:
            st.caption(f"**Last refreshed:** {relative_time(analyzed_at)}")
        with col_btn:
            if viewing_index == 0 and LLAMA_API_KEY:
                if st.button("Refresh analysis", key="refresh_analysis"):
                    with st.spinner("Re-analyzing…"):
                        client = Groq(api_key=LLAMA_API_KEY)
                        pos, neg = extract_sentiment(client, review_text)
                        save_analysis(listing_id, listing_name, pos, neg)
                    st.session_state[VIEWING_ANALYSIS_INDEX] = 0
                    st.rerun()

        with st.expander("Analysis history (last 6 months)", expanded=False):
            hist_options = [_history_label(i, r) for i, r in enumerate(analyses)]
            sel = st.selectbox("View snapshot", hist_options, index=viewing_index, key="history_select")
            new_idx = hist_options.index(sel)
            if new_idx != viewing_index:
                st.session_state[VIEWING_ANALYSIS_INDEX] = new_idx
                st.rerun()
            st.caption(f"{len(analyses)} snapshot(s) in the last {MONTHS_HISTORY} months.")

        tab_pos, tab_neg = st.tabs(["Positive sentiment", "Negative sentiment"])

        with tab_pos:
            st.subheader("Positive sentiment (from reviews)")
            st.markdown(positive_text)
            st.divider()
            if st.button("Generate updated listing", type="primary", key="gen_listing"):
                if not LLAMA_API_KEY:
                    st.error("LLAMA_API_KEY is not set.")
                else:
                    with st.spinner("Generating listing (title + paragraphs)…"):
                        client = Groq(api_key=LLAMA_API_KEY)
                        title, description = generate_listing_content(
                            client, positive_text,
                            str(row.get("property_type", "")),
                            str(row.get("city", "")),
                        )
                    st.subheader("Generated listing")
                    st.write("**Title** (≤200 chars): ", title)
                    st.caption(f"Length: {len(title)} characters")
                    st.write("**Description** (paragraphs):")
                    st.write(description)

        with tab_neg:
            st.subheader("Negative sentiment")
            st.caption("Use this feedback to analyze work orders, maintenance, and owner scoring.")
            st.markdown(negative_text)
            st.divider()
            st.subheader("Log for work orders, maintenance & owner scoring")
            with st.form("work_log_form"):
                category = st.selectbox(
                    "Use for",
                    ["Work order", "Maintenance", "Owner scoring", "Other"],
                    key="work_log_category",
                )
                description = st.text_area("Notes (optional)", key="work_log_desc", height=80)
                submitted = st.form_submit_button("Log for analysis")
            if submitted:
                st.session_state[WORK_LOGS].append({
                    "listing_id": listing_id,
                    "listing_name": listing_name,
                    "category": category,
                    "description": description or "(No notes)",
                })
                st.success("Logged. Clients can use this for work orders, maintenance, and owner scoring.")
                st.balloons()

        session_logs = [w for w in st.session_state[WORK_LOGS] if w["listing_id"] == listing_id]
        if session_logs:
            st.divider()
            with st.expander("Logged for this property (work orders / maintenance / owner scoring)"):
                for w in session_logs:
                    st.write(f"**{w['category']}** — {w['description']}")

    st.divider()
    st.caption("Data: " + LISTINGS_CSV + " · Model: " + LLAMA_MODEL)


if __name__ == "__main__":
    main()
