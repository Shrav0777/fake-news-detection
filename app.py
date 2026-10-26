"""
app.py — Streamlit Web Application for Fake News Detection
===========================================================
A polished, multi-page Streamlit app for demonstrating the
Fake News Detection AI system.

Pages:
    🏠 Home           — Project overview and how it works.
    🔍 Predict        — Enter news text and get a prediction.
    📊 Performance    — Model evaluation metrics and charts.
    📁 Dataset Info   — Dataset statistics and preprocessing details.

Usage:
    streamlit run app.py
"""

import os
import io
import json
import streamlit as st
import pandas as pd

# --- Path configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
METRICS_FILE = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
MODEL_META_FILE = os.path.join(MODELS_DIR, "model_metadata.json")


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
    section[data-testid="stSidebar"] .stMarkdown label {
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

    /* Feature card */
    .feature-card {
        background: white;
        border: 1px solid #e8e8e8;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    .feature-card h4 {
        color: #2c3e50;
        margin: 0 0 0.5rem 0;
    }
    .feature-card p {
        color: #666;
        font-size: 0.92rem;
        margin: 0;
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

@st.cache_data
def load_evaluation_metrics():
    """Load the saved evaluation metrics JSON."""
    if not os.path.isfile(METRICS_FILE):
        return None
    with open(METRICS_FILE) as f:
        return json.load(f)


@st.cache_data
def load_model_metadata():
    """Load model training metadata."""
    if not os.path.isfile(MODEL_META_FILE):
        return None
    with open(MODEL_META_FILE) as f:
        return json.load(f)


def check_model_ready() -> bool:
    """Check if model artifacts exist."""
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    return os.path.isfile(vec_path) and os.path.isfile(model_path)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR NAVIGATION                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown("## 📰 Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Home", "🔍 Predict", "📊 Performance", "📁 Dataset Info"],
        label_visibility="collapsed",
    )




# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: HOME                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if page == "🏠 Home":
    # Header
    st.markdown(
        '<div class="main-header">'
        '<h1>📰 Fake News Detection System</h1>'
        '<p>AI-powered analysis of news articles using Natural Language Processing</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Problem statement
    st.markdown("### 🎯 Problem Statement")
    st.markdown(
        "The spread of fake news is a significant challenge in the digital age. "
        "Misinformation can influence public opinion, disrupt elections, and cause "
        "real-world harm. This project uses **Artificial Intelligence** and "
        "**Natural Language Processing (NLP)** to automatically analyze news text "
        "and predict whether it is likely **real** or **fake** based on patterns "
        "learned from a large labelled dataset."
    )

    st.markdown("---")

    # How it works
    st.markdown("### ⚙️ How It Works")
    cols = st.columns(4)

    steps = [
        ("📝", "Input Text", "Enter a news headline or article body into the system."),
        ("🔧", "Text Processing", "The text is cleaned: lowercased, URLs/HTML removed, non-alphabetic characters stripped."),
        ("📊", "TF-IDF Features", "Text is converted into numerical features using Term Frequency–Inverse Document Frequency."),
        ("🤖", "ML Prediction", "A trained classifier analyzes the features and predicts REAL or FAKE."),
    ]

    for col, (emoji, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div class="feature-card">'
                f'<h4>{emoji} {title}</h4>'
                f'<p>{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Technology stack
    st.markdown("### 🛠️ Technology Stack")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "| Component | Technology |\n"
            "|---|---|\n"
            "| Language | Python |\n"
            "| Data Processing | Pandas, NumPy |\n"
            "| ML Framework | Scikit-learn |\n"
            "| NLP Features | TF-IDF Vectorization |\n"
        )
    with col2:
        st.markdown(
            "| Component | Technology |\n"
            "|---|---|\n"
            "| Classifiers | Logistic Regression, Naive Bayes, Passive Aggressive |\n"
            "| Visualization | Matplotlib, Seaborn |\n"
            "| Web Interface | Streamlit |\n"
            "| Model Storage | Joblib |\n"
        )

    # Disclaimer
    st.markdown(
        '<div class="disclaimer">'
        '⚠️ <strong>Important Disclaimer:</strong> This system provides predictions '
        'based on text patterns learned from training data. It is <strong>not</strong> '
        'a definitive fact-checker. Always verify news through multiple trusted sources.'
        '</div>',
        unsafe_allow_html=True,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: PREDICT                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

elif page == "🔍 Predict":
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


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: PERFORMANCE                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

elif page == "📊 Performance":
    st.markdown(
        '<div class="main-header">'
        '<h1>📊 Model Performance</h1>'
        '<p>Evaluation metrics and visualizations from the trained models</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    metrics = load_evaluation_metrics()
    if metrics is None:
        st.error(
            "⚠️ **Evaluation metrics not found!** "
            "Run `python train_model.py` first to generate results."
        )
        st.stop()

    model_results = metrics.get("model_results", {})
    best_model = metrics.get("best_model", "Unknown")

    # --- Best model highlight ---
    st.markdown(f"### 🏆 Best Model: **{best_model}**")

    if best_model in model_results:
        bm = model_results[best_model]
        cols = st.columns(4)
        metric_items = [
            ("Accuracy", bm.get("accuracy", 0)),
            ("Precision", bm.get("precision", 0)),
            ("Recall", bm.get("recall", 0)),
            ("F1-Score", bm.get("f1_score", 0)),
        ]
        for col, (name, value) in zip(cols, metric_items):
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h3>{name}</h3>'
                    f'<p class="value">{value:.4f}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        if "roc_auc" in bm:
            st.markdown(
                f'<div class="info-box">ROC-AUC Score: <strong>{bm["roc_auc"]:.4f}</strong></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # --- Metrics explanation ---
    st.markdown("### 📖 Understanding the Metrics")
    with st.expander("Click to learn about evaluation metrics", expanded=False):
        st.markdown("""
        | Metric | What It Means (Simple Explanation) |
        |---|---|
        | **Accuracy** | Out of all predictions, how many were correct? |
        | **Precision** | When the model says "FAKE", how often is it actually fake? |
        | **Recall** | Out of all actually fake articles, how many did the model catch? |
        | **F1-Score** | The balance between Precision and Recall (harmonic mean). |
        | **ROC-AUC** | How well the model separates REAL from FAKE across all thresholds. 1.0 is perfect, 0.5 is random guessing. |
        | **Confusion Matrix** | A table showing correct and incorrect predictions for each class. |
        """)

    st.markdown("---")

    # --- All models comparison table ---
    st.markdown("### 📋 All Models — Detailed Comparison")

    table_data = []
    for name, res in model_results.items():
        row = {
            "Model": name + (" ★" if name == best_model else ""),
            "Accuracy": f"{res.get('accuracy', 0):.4f}",
            "Precision": f"{res.get('precision', 0):.4f}",
            "Recall": f"{res.get('recall', 0):.4f}",
            "F1-Score": f"{res.get('f1_score', 0):.4f}",
        }
        if "roc_auc" in res:
            row["ROC-AUC"] = f"{res['roc_auc']:.4f}"
        if "train_time_sec" in res:
            row["Train Time (s)"] = f"{res['train_time_sec']:.1f}"
        table_data.append(row)

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- Visualization plots ---
    st.markdown("### 📈 Visualizations")

    plot_files = {
        "Model Accuracy Comparison": "model_comparison_accuracy.png",
        "Precision / Recall / F1 Comparison": "model_comparison_metrics.png",
        "Confusion Matrix (Best Model)": "confusion_matrix.png",
        "ROC Curve": "roc_curve.png",
    }

    for title, filename in plot_files.items():
        fpath = os.path.join(RESULTS_DIR, filename)
        if os.path.isfile(fpath):
            st.markdown(f"#### {title}")
            st.image(fpath, use_container_width=True)
        else:
            st.caption(f"_{title} — plot not available_")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: DATASET INFO                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

elif page == "📁 Dataset Info":
    st.markdown(
        '<div class="main-header">'
        '<h1>📁 Dataset Information</h1>'
        '<p>Details about the WELFake dataset and preprocessing pipeline</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    metrics = load_evaluation_metrics()

    # --- Dataset overview ---
    st.markdown("### 📄 WELFake Dataset")
    st.markdown(
        "The **WELFake (WEL**l-known **Fake** news) dataset is a merged collection "
        "of four popular news datasets: Kaggle, McIntire, Reuters, and BuzzFeed Political. "
        "It was published in IEEE Transactions on Computational Social Systems."
    )
    st.markdown(
        "**Source:** [Kaggle — Fake News Classification]"
        "(https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)"
    )

    st.markdown("#### Dataset Columns")
    st.markdown("""
    | Column | Description |
    |---|---|
    | `Unnamed: 0` | Serial number (dropped during preprocessing) |
    | `title` | News article headline |
    | `text` | Full article body text |
    | `label` | Binary label: **0 = Fake**, **1 = Real** |
    """)

    # --- Actual stats from training ---
    if metrics and "dataset_info" in metrics:
        ds = metrics["dataset_info"]
        st.markdown("---")
        st.markdown("### 📊 Dataset Statistics (After Preprocessing)")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f'<div class="metric-card">'
                f'<h3>Total Samples</h3>'
                f'<p class="value">{ds.get("total_samples", "N/A"):,}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="metric-card" style="background: linear-gradient(135deg, #e74c3c, #c0392b);">'
                f'<h3>Fake Articles</h3>'
                f'<p class="value">{ds.get("fake_count", "N/A"):,}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f'<div class="metric-card" style="background: linear-gradient(135deg, #27ae60, #2ecc71);">'
                f'<h3>Real Articles</h3>'
                f'<p class="value">{ds.get("real_count", "N/A"):,}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Class distribution plot
        dist_plot = os.path.join(RESULTS_DIR, "class_distribution.png")
        if os.path.isfile(dist_plot):
            st.markdown("#### Class Distribution")
            st.image(dist_plot, use_container_width=True)

        # Sample rows
        if "sample_rows" in ds and ds["sample_rows"]:
            st.markdown("#### Sample Data (First 5 Rows)")
            st.dataframe(
                pd.DataFrame(ds["sample_rows"]),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info(
            "ℹ️ Dataset statistics will appear here after you run `python train_model.py`."
        )

    # --- Preprocessing explanation ---
    st.markdown("---")
    st.markdown("### 🔧 Preprocessing Pipeline")
    st.markdown("""
    The raw dataset goes through these cleaning steps before training:

    1. **Drop index column** — The `Unnamed: 0` serial number is removed.
    2. **Validate labels** — Only rows with labels 0 (Fake) or 1 (Real) are kept.
    3. **Handle missing text** — NaN values in `title` and `text` are replaced with empty strings.
    4. **Combine fields** — `title` and `text` are merged into a single `content` column.
    5. **Text cleaning** — Lowercase, remove URLs, HTML tags, non-alphabetic characters, collapse whitespace.
    6. **Remove empty rows** — Articles with no content after cleaning are dropped.
    7. **Remove duplicates** — Exact duplicate articles (same text + label) are removed.
    """)

    # --- TF-IDF explanation ---
    st.markdown("---")
    st.markdown("### 📊 TF-IDF — Feature Extraction")
    st.markdown("""
    **TF-IDF** (Term Frequency – Inverse Document Frequency) converts text into numbers
    that machine learning algorithms can understand.

    **Simple explanation:**
    - **TF (Term Frequency):** How often a word appears in ONE article.
    - **IDF (Inverse Document Frequency):** How rare or unique a word is across ALL articles.
    - **TF-IDF = TF × IDF:** Words that are frequent in one article but rare overall get
      higher scores. Common words like "the" get very low scores.

    **Example:**
    > In an article about "election fraud", the word "fraud" might appear frequently (high TF)
    > and be relatively rare across all articles (high IDF), giving it a **high TF-IDF score**.
    > Meanwhile, "the" appears everywhere (low IDF), so it gets a **very low TF-IDF score**.

    This means the model focuses on meaningful, distinguishing words rather than common ones.
    """)

    # Model metadata
    meta = load_model_metadata()
    if meta:
        st.markdown("---")
        st.markdown("### 🤖 Model Information")
        st.json(meta)
