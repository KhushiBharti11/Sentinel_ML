import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- Page configuration --------------------
st.set_page_config(
    page_title="SENTINEL | Complaint Intelligence",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px;}
      .sentinel-header {
        padding: 1.4rem 1.6rem; border-radius: 16px;
        background: linear-gradient(120deg, #12243a, #234a70);
        color: white; margin-bottom: 1.2rem;
      }
      .sentinel-subtitle {color: #d9e6f2; margin-top: -0.35rem;}
      .small-note {font-size: 0.86rem; color: #687385;}
    </style>
    <div class="sentinel-header">
      <h1>🛡️ SENTINEL</h1>
      <div class="sentinel-subtitle">Institutional Communication &amp; Intelligent Complaint Analysis</div>
      <p style="margin-bottom:0.2rem">Standalone research prototype · Pilot dataset</p>
    </div>
    """,
    unsafe_allow_html=True,
)

BUNDLE_PATH = os.environ.get("SENTINEL_BUNDLE_PATH", "sentinel_demo_bundle.joblib")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CATEGORY_LABELS = {
    "hostel_accommodation": "Hostel / Accommodation",
    "placement_career_services": "Placement / Career Services",
    "infrastructure_facilities": "Infrastructure / Facilities",
    "academic_issues": "Academic Issues",
    "faculty_related": "Faculty Related",
    "administration": "Administration",
    "management": "Management",
}

FREQUENCY_MAPPING = {
    "Rarely": 1,
    "Sometimes": 2,
    "Often": 3,
    "Always": 4,
}

# These are the prototype's existing keyword rules, not a validated safety classifier.
RISK_TERMS = [
    "water", "washroom", "unhygienic", "lpg", "fire", "electricity",
    "electrical", "clogging", "termite", "falling", "broken", "leak"
]


@st.cache_resource(show_spinner="Loading SENTINEL model bundle...")
def load_bundle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model bundle not found at '{path}'. Add sentinel_demo_bundle.joblib "
            "to the project root or set SENTINEL_BUNDLE_PATH."
        )
    return joblib.load(path)


@st.cache_resource(show_spinner="Loading Sentence Transformer model...")
def load_embedding_model(model_name):
    return SentenceTransformer(model_name)


def detect_risk(text):
    text_lower = str(text).lower()
    return [term for term in RISK_TERMS if term in text_lower]


def calculate_priority(seriousness, frequency, complaint_text):
    if int(seriousness) not in [1, 2, 3, 4, 5]:
        raise ValueError("Seriousness must be from 1 to 5.")
    if frequency not in FREQUENCY_MAPPING:
        raise ValueError("Choose a valid frequency.")

    seriousness_norm = (float(seriousness) - 1) / 4
    frequency_norm = (FREQUENCY_MAPPING[frequency] - 1) / 3
    risk_indicators = detect_risk(complaint_text)
    risk_flag = int(bool(risk_indicators))

    # Prototype policy weights; these are manually specified, not learned.
    score = (
        0.50 * seriousness_norm
        + 0.30 * frequency_norm
        + 0.20 * risk_flag
    )

    if score < 0.40:
        level = "LOW"
    elif score < 0.60:
        level = "MEDIUM"
    elif score < 0.80:
        level = "HIGH"
    else:
        level = "CRITICAL"

    return {
        "priority_score": round(float(score), 3),
        "priority_level": level,
        "risk_indicators": risk_indicators,
    }


def category_display_name(category):
    return CATEGORY_LABELS.get(category, category.replace("_", " ").title())


def analyze_complaint(text, seriousness, frequency, bundle, embedding_model, top_k=5):
    # Use the same embedding model and classifier feature space as the Colab experiment.
    complaint_embedding = embedding_model.encode(
        [text], convert_to_numpy=True, show_progress_bar=False
    )
    model = bundle["category_model"]
    category_columns = list(bundle["category_columns"])
    category_probabilities = model.predict_proba(complaint_embedding)[0]

    category_results = []
    for category, probability in zip(category_columns, category_probabilities):
        category_results.append({
            "category": category,
            "label": category_display_name(category),
            "score": float(probability),
            "predicted": bool(probability >= 0.50),
        })
    category_results.sort(key=lambda item: item["score"], reverse=True)

    reference_embeddings = np.asarray(bundle["reference_embeddings"])
    reference_complaints = bundle["reference_complaints"]
    similarities = cosine_similarity(complaint_embedding, reference_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    related = []
    for idx in top_indices:
        record = reference_complaints[int(idx)]
        related.append({
            "complaint_id": record.get("complaint_id", f"Reference {int(idx)+1}"),
            "similarity": float(similarities[int(idx)]),
            "complaint_text": record.get("complaint_text", ""),
        })

    priority = calculate_priority(seriousness, frequency, text)
    return {
        "category_results": category_results,
        "related_complaints": related,
        **priority,
    }


# -------------------- Load resources --------------------
try:
    bundle = load_bundle(BUNDLE_PATH)
    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
except Exception as exc:
    st.error("SENTINEL could not load its model resources.")
    st.exception(exc)
    st.info(
        "Confirm that sentinel_demo_bundle.joblib is in the repository root "
        "and that the Streamlit deployment has installed requirements.txt."
    )
    st.stop()

# -------------------- Input form --------------------
st.subheader("Submit a complaint")
st.write("Enter a complaint and its reported seriousness and frequency to analyze it.")

with st.form("complaint_form"):
    complaint_text = st.text_area(
        "Complaint description",
        placeholder="Example: The hostel mess food quality is very poor...",
        height=130,
        help="Avoid entering names, phone numbers, student IDs, or other personal information.",
    )
    col1, col2 = st.columns(2)
    with col1:
        seriousness = st.select_slider(
            "Seriousness",
            options=[1, 2, 3, 4, 5],
            value=4,
            help="Use the student's reported seriousness, where 1 is minor and 5 is major.",
        )
    with col2:
        frequency = st.selectbox(
            "How often does this problem occur?",
            options=["Rarely", "Sometimes", "Often", "Always"],
            index=2,
        )
    submitted = st.form_submit_button("Analyze complaint", type="primary", use_container_width=True)

if submitted:
    if not complaint_text.strip():
        st.warning("Please enter a complaint before analyzing it.")
        st.stop()

    with st.spinner("Analyzing complaint..."):
        result = analyze_complaint(
            complaint_text.strip(),
            seriousness,
            frequency,
            bundle,
            embedding_model,
        )

    st.divider()
    st.subheader("Analysis results")

    predicted = [item for item in result["category_results"] if item["predicted"]]
    category_col, priority_col, risk_col = st.columns(3)

    with category_col:
        st.markdown("#### Predicted categories")
        if predicted:
            for item in predicted:
                st.write(f"**{item['label']}** — {item['score']:.1%}")
        else:
            st.info("No category crossed the prototype threshold of 0.50.")
        st.caption("Scores are model outputs, not calibrated probabilities.")

    with priority_col:
        st.markdown("#### Priority")
        level = result["priority_level"]
        st.metric("Priority level", level, help="Based on a manually specified prototype rule.")
        st.metric("Priority score", f"{result['priority_score']:.3f}")
        st.caption("Weights: 50% seriousness, 30% frequency, 20% keyword-risk flag.")

    with risk_col:
        st.markdown("#### Rule-based risk indicators")
        if result["risk_indicators"]:
            st.write(", ".join(result["risk_indicators"]))
        else:
            st.write("No configured risk keywords detected.")
        st.caption("This is not a comprehensive safety assessment.")

    with st.expander("View all category scores"):
        score_df = pd.DataFrame([
            {"Category": item["label"], "Model score": round(item["score"], 3),
             "Selected at threshold 0.50": "Yes" if item["predicted"] else "No"}
            for item in result["category_results"]
        ])
        st.dataframe(score_df, use_container_width=True, hide_index=True)

    st.markdown("#### Top 5 related complaints")
    related_rows = []
    for item in result["related_complaints"]:
        related_rows.append({
            "Reference ID": item["complaint_id"],
            "Cosine similarity": round(item["similarity"], 3),
            "Complaint": item["complaint_text"],
        })
    st.dataframe(pd.DataFrame(related_rows), use_container_width=True, hide_index=True)
    st.caption(
        "Related complaints are ranked by semantic similarity. A high score does not "
        "prove that two complaints are duplicates; an administrator should review them."
    )

    st.info(
        "Research prototype using a small pilot dataset. Outputs are advisory and "
        "must not be treated as validated automated decisions."
    )

with st.expander("About this prototype"):
    st.write(
        "SENTINEL demonstrates multi-label complaint categorization, rule-based priority "
        "scoring, keyword-based risk indicators, and semantic retrieval of related complaints. "
        "The current pilot is small and may contain label noise and class imbalance. "
        "Priority weights and thresholds are manually specified prototype assumptions."
    )
