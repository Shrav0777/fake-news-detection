"""
app.py — Streamlit Web Application for Fake News Detection
===========================================================
A streamlined, high-performance Streamlit interface dedicated to news
prediction using trained Natural Language Processing (NLP) models.

Features:
    🔍 Predict — Single headline/article prediction with confidence scores.
    📂 Batch Analysis — Upload CSV datasets for bulk predictions and download results.

All theoretical concepts, dataset statistics, preprocessing pipeline details,
and model evaluation benchmarks are documented in README.md.

Usage:
    streamlit run app.py
"""

import os
import io
import streamlit as st
import pandas as pd

# --- Path configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE CONFIG & STYLING                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a polished look
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown label,
    section[data-testid="stSidebar"] .stMarkdown span {
        color: #e0e0e0 !important;
    }

    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #0f3460 0%, #533483 50%, #e94560 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(15, 52, 96, 0.3);
    }
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.05rem;
        margin: 0;
    }

    /* Result cards */
    .result-card {
        padding: 1.8rem 2rem;
        border-radius: 14px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    }
    .result-real {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
    }
    .result-fake {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        color: white;
    }
    .result-card h2 {
        font-size: 2rem;
        margin: 0 0 0.5rem 0;
        font-weight: 700;
    }
    .result-card p {
        font-size: 1rem;
        margin: 0;
        opacity: 0.92;
    }

    /* Metric card */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.4rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.25);
        margin-bottom: 0.8rem;
    }
    .metric-card h3 {
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0 0 0.4rem 0;
        opacity: 0.85;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }

    /* Info box */
    .info-box {
        background: #f8f9fa;
        border-left: 4px solid #3498db;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
        font-size: 0.95rem;
        color: #2c3e50;
    }

    /* Warning box */
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
        font-size: 0.92rem;
        color: #856404;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Disclaimer */
    .disclaimer {
        background: linear-gradient(135deg, #ffecd2, #fcb69f);
        padding: 0.9rem 1.2rem;
        border-radius: 10px;
        font-size: 0.88rem;
        color: #7c4a1e;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HELPER FUNCTIONS                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def check_model_ready() -> bool:
    """Check if model artifacts exist."""
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    return os.path.isfile(vec_path) and os.path.isfile(model_path)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown("## 📰 Fake News Detector")
    st.caption("AI-Powered Misinformation Detection")
    st.markdown("---")
    st.markdown("### 🔍 Predict")
    st.markdown(
        "Enter news content or upload a CSV dataset to check whether "
        "articles are classified as **REAL** or **FAKE**."
    )



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PREDICT PAGE                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

st.markdown(
    '<div class="main-header">'
    '<h1>🔍 News Prediction</h1>'
    '<p>Enter a news headline or article text to check if it is predicted as Real or Fake</p>'
    '</div>',
    unsafe_allow_html=True,
)

if not check_model_ready():
    st.error(
        "⚠️ **Model not found!** Please run the training script first:\n\n"
        "```bash\npython train_model.py\n```"
    )
    st.stop()

# Import prediction module (only when model is ready)
from predict import predict_single, combine_headline_body, predict_dataframe

# --- Input mode selection ---
input_mode = st.radio(
    "Input mode",
    ["Combined text", "Separate headline & article"],
    horizontal=True,
)

if input_mode == "Combined text":
    user_text = st.text_area(
        "Paste the news headline and/or article text below:",
        height=200,
        placeholder="Enter or paste news content here...",
        key="combined_input",
    )
else:
    col_h, col_b = st.columns([1, 2])
    with col_h:
        headline = st.text_input(
            "📌 Headline",
            placeholder="Enter the news headline...",
            key="headline_input",
        )
    with col_b:
        body = st.text_area(
            "📄 Article body",
            height=180,
            placeholder="Enter the full article text...",
            key="body_input",
        )
    user_text = combine_headline_body(headline, body)

# --- Prediction buttons ---
col_btn, col_clear = st.columns([1, 4])
with col_btn:
    predict_btn = st.button("🔎 Predict", type="primary", use_container_width=True)
with col_clear:
    clear_btn = st.button("🗑️ Clear", use_container_width=False)

if clear_btn:
    st.rerun()

if predict_btn:
    if not user_text or not user_text.strip():
        st.warning("⚠️ Please enter some text before clicking Predict.")
    elif len(user_text.strip()) < 10:
        st.warning(
            "⚠️ The text is very short. Predictions on very short inputs "
            "may be unreliable. Consider providing more context."
        )
        result = predict_single(user_text)
        # Still show prediction with caveat
        if result["label"] == "REAL":
            st.markdown(
                '<div class="result-card result-real">'
                '<h2>✅ REAL</h2>'
                '<p>The model predicts this text is <strong>likely real news</strong>.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-card result-fake">'
                '<h2>🚫 FAKE</h2>'
                '<p>The model predicts this text is <strong>likely fake news</strong>.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        if result.get("confidence") is not None:
            st.metric("Model Confidence", f"{result['confidence']:.2%}")
        st.info("ℹ️ Short inputs may produce unreliable predictions.")
    else:
        with st.spinner("Analyzing text patterns..."):
            result = predict_single(user_text)

        if result["label"] == "UNKNOWN":
            st.error(f"Could not make a prediction: {result.get('error', '')}")
        elif result["label"] == "REAL":
            st.markdown(
                '<div class="result-card result-real">'
                '<h2>✅ REAL</h2>'
                '<p>The model predicts this text is <strong>likely real news</strong> '
                'based on learned text patterns.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-card result-fake">'
                '<h2>🚫 FAKE</h2>'
                '<p>The model predicts this text is <strong>likely fake news</strong> '
                'based on learned text patterns.</p>'
                '</div>',
                unsafe_allow_html=True,
            )

        # Confidence
        if result.get("confidence") is not None:
            st.metric("Model Confidence", f"{result['confidence']:.2%}")

        # Expandable details
        with st.expander("📋 Prediction Details"):
            st.markdown(f"**Prediction:** {result['label']}")
            st.markdown(f"**Numeric label:** {result['label_int']} (0=Fake, 1=Real)")
            if result.get("confidence") is not None:
                st.markdown(f"**Confidence:** {result['confidence']:.4f}")
            st.markdown(f"**Cleaned text preview:** {result['cleaned'][:300]}...")

# --- Disclaimer ---
st.markdown(
    '<div class="disclaimer">'
    '⚠️ <strong>Disclaimer:</strong> This prediction is based on text patterns learned '
    'from training data. It is <strong>not</strong> a definitive statement about the '
    'truthfulness of this article. Always cross-check with trusted sources.'
    '</div>',
    unsafe_allow_html=True,
)

# --- Batch Prediction ---
st.markdown("---")
st.markdown("### 📂 Batch Prediction from CSV")

st.markdown(
    '<div class="info-box">'
    'Upload a CSV file with a <code>text</code> column (and optionally a '
    '<code>title</code> column). The system will predict each row.'
    '</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="batch_csv")

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)

        # Validate required column
        if "text" not in batch_df.columns:
            st.error(
                "❌ The uploaded CSV must have a `text` column. "
                f"Found columns: {list(batch_df.columns)}"
            )
        else:
            st.info(f"📄 Loaded {len(batch_df)} rows. Running predictions...")

            title_col = "title" if "title" in batch_df.columns else None
            with st.spinner("Processing batch predictions..."):
                result_df = predict_dataframe(batch_df, text_column="text",
                                              title_column=title_col)

            # Display results
            st.success(f"✅ Predictions complete for {len(result_df)} rows!")

            # Summary
            pred_counts = result_df["prediction"].value_counts()
            col_r, col_f = st.columns(2)
            with col_r:
                st.metric("REAL predictions", pred_counts.get("REAL", 0))
            with col_f:
                st.metric("FAKE predictions", pred_counts.get("FAKE", 0))

            st.dataframe(result_df[["text", "prediction", "confidence"]].head(20))

            # Download
            csv_buffer = io.StringIO()
            result_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "📥 Download Predictions CSV",
                csv_buffer.getvalue(),
                file_name="fake_news_predictions.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
