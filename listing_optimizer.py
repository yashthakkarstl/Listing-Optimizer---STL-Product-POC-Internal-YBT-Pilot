"""
Listing Optimizer - Proof of Concept
Uses Listings.csv only. Extracts positive sentiment from the Review column
and generates a 200-character listing title via LLM (Llama).
"""
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from groq import Groq

# --- Config (set your key before running) ---
LISTINGS_CSV = "Listings_1.csv"
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama-3.3-70b-versatile")


@st.cache_data
def load_listings():
    # CSV may be Windows-encoded (cp1252); try UTF-8 first, then fallback
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(LISTINGS_CSV, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        df = pd.read_csv(LISTINGS_CSV, encoding="utf-8", encoding_errors="replace")
    # Ensure Review column exists (last column)
    if "Review" not in df.columns:
        last_col = df.columns[-1]
        df = df.rename(columns={last_col: "Review"})
    # Drop rows with no review text
    df = df[df["Review"].notna() & (df["Review"].astype(str).str.strip() != "")].copy()
    return df


def extract_positive_sentiment(client: Groq, review_text: str) -> str:
    prompt = """From this guest review, extract ONLY the positive things guests said. Write them as 2–3 short, flowing paragraphs (no bullet points). Use a listing-style tone, like descriptive copy for a property. If there are no clear positives, say "No clear positive highlights found."
Review:
"""
    prompt += review_text[:4000]  # cap length
    try:
        r = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Error extracting sentiment: {e}"


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
    st.set_page_config(page_title="Listing Optimizer - STL Product POC", page_icon="🏠", layout="wide")
    st.title("🏠 Listing Optimizer - STL Product POC")
    st.caption("Use guest reviews to generate a compelling listing title (Listings.csv only)")

    df = load_listings()
    if df.empty:
        st.error("No listings with reviews found in Listings.csv.")
        return

    # Listing selector
    df["_label"] = df["name"].fillna("") + " (ID: " + df["listing_id"].astype(str) + ")"
    options = df["_label"].tolist()
    choice = st.selectbox("Select a listing", options, index=0)
    idx = df[df["_label"] == choice].index[0]
    row = df.loc[idx]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current listing")
        st.write("**Title:**", row.get("name", "—"))
        st.write("**Property type:**", row.get("property_type", "—"))
        st.write("**City:**", row.get("city", "—"))
    with col2:
        st.subheader("Guest review (source)")
        review = str(row.get("Review", ""))[:2000]
        st.text_area("Review text", value=review, height=120, disabled=True)

    st.divider()
    if not LLAMA_API_KEY:
        st.warning("Set **LLAMA_API_KEY** in your environment (or in a `.env` file) before generating.")
    if st.button("Generate listing title", type="primary", use_container_width=True):
        if not LLAMA_API_KEY:
            st.error("LLAMA_API_KEY is not set. Add your key and restart.")
            return
        with st.spinner("Extracting positive sentiment…"):
            client = Groq(api_key=LLAMA_API_KEY)
            positive = extract_positive_sentiment(client, str(row.get("Review", "")))
        st.subheader("Positive sentiment from reviews")
        st.write(positive)
        with st.spinner("Generating 200-character listing title…"):
            title = generate_listing_title(
                client, positive,
                str(row.get("property_type", "")),
                str(row.get("city", "")),
            )
        st.subheader("Generated listing title (≤200 chars)")
        st.success(title)
        st.caption(f"Length: {len(title)} characters")

    st.divider()
    st.caption("Data: Listings_1.csv · Model: " + LLAMA_MODEL)


if __name__ == "__main__":
    main()
