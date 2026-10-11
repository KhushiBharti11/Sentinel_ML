# SENTINEL ML Demo

A standalone Streamlit research prototype for intelligent institutional complaint analysis.

## Features

- Multi-label complaint categorization using the saved trained classifier
- Sentence Transformer embeddings (`all-MiniLM-L6-v2`)
- Rule-based LOW / MEDIUM / HIGH / CRITICAL priority score
- Keyword-based institutional risk indicators
- Top-5 semantically related complaints from the pilot reference set

## Files

- `app.py` — Streamlit interface and inference pipeline
- `requirements.txt` — Python dependencies
- `sentinel_demo_bundle.joblib` — generated in Colab from the trained classifier, reference embeddings, category columns, and reference complaints

## Run locally

1. Create a virtual environment (recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place `sentinel_demo_bundle.joblib` in the same folder as `app.py`.
4. Start the app:

   ```bash
   streamlit run app.py
   ```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository for this demo.
2. Upload `app.py`, `requirements.txt`, and `README.md`.
3. Upload `sentinel_demo_bundle.joblib` only if it is safe to share with the people who can access the repository and app.
4. In Streamlit Community Cloud, create an app from that repository and select `app.py` as the entry point.
5. Wait for dependency installation and model download to complete.

## Privacy and research limitations

- This is a pilot/research prototype, not a production governance system.
- Do not publish personally identifiable or sensitive complaint text. Use a private repository and review who can access the deployed app.
- A private repository does not automatically make every deployed app private; review Streamlit's app visibility settings.
- The saved reference complaints may be shown to users through similarity results. Remove or anonymize any sensitive records before deployment.
- Category scores are not calibrated probabilities.
- Priority weights and thresholds are manually specified prototype assumptions, not learned or validated policy.
- Keyword-based risk detection is limited and can miss real hazards or flag benign mentions.
- Similarity is not proof of duplication; results require human review.
- The app does not write to Supabase or connect to the main SENTINEL application.
