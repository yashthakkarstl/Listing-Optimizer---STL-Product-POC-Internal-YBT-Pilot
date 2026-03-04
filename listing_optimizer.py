"""
Listing Optimizer - Proof of Concept
Uses Listings.csv only. Analyzes guest reviews for positive and negative sentiment.
- Positive: used for content (generate updated listing title).
- Negative: shown to clients for work orders, maintenance, and owner scoring (raise ticket).
"""
from __future__ import annotations

import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from groq import Groq

# --- Config: from TOML secrets (Streamlit Cloud) or .env ---
LISTINGS_CSV = "Listings_1.csv"


def _get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default) or default
    except Exception:
        return os.getenv(key, default)


LLAMA_API_KEY = _get_secret("LLAMA_API_KEY", os.getenv("LLAMA_API_KEY", ""))
LLAMA_MODEL = _get_secret("LLAMA_MODEL", os.getenv("LLAMA_MODEL", "llama-3.3-70b-versatile"))

# Session state keys
SENTIMENT_CACHE = "sentiment_cache"  # dict of listing_id -> {"positive": str, "negative": str}
TICKETS = "tickets"  # list of {"listing_id", "listing_name", "category", "description"}


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
    """Returns (positive_sentiment, negative_sentiment) as short paragraphs."""
    prompt = """Analyze this guest review and output two sections. Use 2–3 short, flowing paragraphs per section (no bullet points).

First section: POSITIVE — only what guests liked or praised. Use a listing-style tone. If none, write "No clear positive highlights found."

Second section: NEGATIVE — only what guests complained about or found lacking (for maintenance/work orders). If none, write "No clear negative feedback found."

Format your reply exactly like this (include the labels on their own lines):
POSITIVE:
(paragraphs here)

NEGATIVE:
(paragraphs here)

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


def generate_listing_title(client: Groq, positive_sentiment: str, property_type: str, city: str) -> str:
    prompt = f"""Using ONLY these positive guest highlights, write one amazing Airbnb listing title. Include property type and location if helpful.
- Maximum 200 characters.
- Output ONLY the title, no quotes, no explanation.

Positive highlights:
{positive_sentiment}

Property type: {property_type}. City: {city}.
"""
    try:
        r = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
        )
        title = (r.choices[0].message.content or "").strip().strip('"')
        return title[:200]
    except Exception as e:
        return f"Error generating title: {e}"


def main():
    st.set_page_config(page_title="Listing Optimizer", page_icon="🏠", layout="wide")
    st.title("🏠 Listing Optimizer")
    st.caption("Analyze review sentiment · Use positive for listing content · Use negative for work orders & tickets")

    if SENTIMENT_CACHE not in st.session_state:
        st.session_state[SENTIMENT_CACHE] = {}
    if TICKETS not in st.session_state:
        st.session_state[TICKETS] = []

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

    # Run sentiment analysis
    review_text = str(row.get("Review", ""))
    cached = st.session_state[SENTIMENT_CACHE].get(listing_id)

    if not cached:
        if not LLAMA_API_KEY:
            st.warning("Set **LLAMA_API_KEY** in secrets or .env to run sentiment analysis.")
        else:
            if st.button("Run sentiment analysis", type="primary", use_container_width=True):
                with st.spinner("Analyzing positive and negative sentiment…"):
                    client = Groq(api_key=LLAMA_API_KEY)
                    pos, neg = extract_sentiment(client, review_text)
                    st.session_state[SENTIMENT_CACHE][listing_id] = {"positive": pos, "negative": neg}
                st.rerun()
    else:
        positive_text = cached["positive"]
        negative_text = cached["negative"]

        tab_pos, tab_neg = st.tabs(["Positive sentiment", "Negative sentiment"])

        with tab_pos:
            st.subheader("Positive sentiment (for listing content)")
            st.write(positive_text)
            st.divider()
            if st.button("Generate updated listing", type="primary", key="gen_listing"):
                if not LLAMA_API_KEY:
                    st.error("LLAMA_API_KEY is not set.")
                else:
                    with st.spinner("Generating 200-character listing title…"):
                        client = Groq(api_key=LLAMA_API_KEY)
                        title = generate_listing_title(
                            client, positive_text,
                            str(row.get("property_type", "")),
                            str(row.get("city", "")),
                        )
                    st.success("Generated listing title (≤200 chars)")
                    st.info(title)
                    st.caption(f"Length: {len(title)} characters")

        with tab_neg:
            st.subheader("Negative sentiment (for work orders & maintenance)")
            st.write(negative_text)
            st.divider()
            st.subheader("Raise ticket")
            with st.form("raise_ticket_form"):
                category = st.selectbox(
                    "Category",
                    ["Maintenance", "Repair", "Cleaning", "Amenity issue", "Other"],
                    key="ticket_category",
                )
                description = st.text_area("Description (optional)", key="ticket_desc", height=80)
                submitted = st.form_submit_button("Submit ticket")
            if submitted:
                st.session_state[TICKETS].append({
                    "listing_id": listing_id,
                    "listing_name": listing_name,
                    "category": category,
                    "description": description or "(No description)",
                })
                st.success("Ticket raised. It will be used for work orders and owner scoring.")
                st.balloons()

        # Optional: show raised tickets for this session
        session_tickets = [t for t in st.session_state[TICKETS] if t["listing_id"] == listing_id]
        if session_tickets:
            st.divider()
            with st.expander("Tickets raised for this property (this session)"):
                for t in session_tickets:
                    st.write(f"**{t['category']}** — {t['description']}")

    st.divider()
    st.caption("Data: " + LISTINGS_CSV + " · Model: " + LLAMA_MODEL)


if __name__ == "__main__":
    main()
