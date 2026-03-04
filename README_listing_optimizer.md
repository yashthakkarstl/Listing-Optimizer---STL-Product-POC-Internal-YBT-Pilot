# Listing Optimizer (PoC)

Streamlit app that uses **Listings.csv only**. It reads the **Review** column, extracts positive sentiment, and uses the Llama model to generate a 200-character listing title.

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your API key and model** (choose one):
   - **Environment variables:**
     ```bash
     set LLAMA_API_KEY=your_groq_api_key
     set LLAMA_MODEL=llama-3.3-70b-versatile
     ```
   - **Or create a `.env` file** in this folder:
     ```
     LLAMA_API_KEY=your_groq_api_key
     LLAMA_MODEL=llama-3.3-70b-versatile
     ```
   Use your Groq API key from [console.groq.com](https://console.groq.com).

## Run

From the folder containing `Listings.csv` and `listing_optimizer.py`:

```bash
streamlit run listing_optimizer.py
```

Select a listing, then click **Generate listing title** to see positive sentiment and the new title.
