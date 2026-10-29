# 📰 Fake News Detection System


An AI-powered Fake News Detection System utilizing Natural Language Processing (NLP) and Machine Learning. The system analyzes news headlines and article body text to classify articles as either **REAL** or **FAKE** based on linguistic and lexical patterns learned from a benchmark dataset of over 72,000 news articles.

> ⚠️ **Important Disclaimer:** This system provides probabilistic predictions based on textual patterns learned from training data. It is **not** a definitive fact-checking agency. Always cross-verify news articles across multiple trusted journalistic sources.

---

## 🎯 Problem Statement & Background

The proliferation of misinformation and fake news in the digital era poses critical challenges to society. False narratives can rapidly distort public perception, destabilize democratic processes, manipulate financial markets, and incite real-world harm. 

Traditional manual fact-checking is labor-intensive and cannot scale to match the velocity of digital publishing and social media diffusion. This project implements an automated, machine-learning-driven text classification pipeline to detect deceptive and fabricated news content using statistical language representations and supervised classification models.

---

## ⚙️ System Architecture & Workflow

The end-to-end detection system operates through four core stages:

```
[ Input News Text / Headline ]
              │
              ▼
[ 1. Text Preprocessing & Cleaning ]
  • Lowercasing
  • URL & HTML stripping
  • Non-alphabetic character removal
  • Whitespace collapse
              │
              ▼
[ 2. Feature Extraction (TF-IDF) ]
  • Vocabulary: 50,000 features
  • Unigrams & Bigrams (1, 2)
  • Inverse document frequency weighting
              │
              ▼
[ 3. Classifier Inference ]
  • Passive Aggressive Classifier (Champion model)
  • Linear decision boundary evaluation
              │
              ▼
[ Prediction Output: REAL / FAKE + Confidence % ]
```

1. **Text Input:** News headlines, full article bodies, or uploaded CSV files are ingested.
2. **Text Normalization:** Unstructured raw text is stripped of web markup, punctuation, numbers, and noise.
3. **TF-IDF Vectorization:** Cleaned text is transformed into high-dimensional numerical feature vectors.
4. **Machine Learning Classifier:** A trained classifier predicts the binary class label (0 = Fake, 1 = Real) along with a prediction confidence score.

---

## 🛠️ Technology Stack

| Component | Technology | Description |
|---|---|---|
| **Language** | Python 3.9+ | Core programming runtime |
| **Data Processing** | Pandas, NumPy | Data manipulation and tabular analysis |
| **Machine Learning** | Scikit-learn | ML estimators, vectorizers, and evaluation metrics |
| **Feature Extraction** | TF-IDF Vectorizer | Sublinear term frequency-inverse document frequency |
| **Classifiers** | Logistic Regression, Naive Bayes, Passive Aggressive | Supervised classification algorithms |
| **Visualization** | Matplotlib, Seaborn | Metric plotting, confusion matrices, and ROC curves |
| **Web Interface** | Streamlit | Responsive, real-time prediction application |
| **Model Persistence** | Joblib | Fast disk serialization of models and vectorizers |

---

## 📁 Dataset Information & Schema

The system is trained and evaluated on the **WELFake (WELl-known Fake news) dataset**, a consolidated benchmark collection merging four recognized misinformation datasets: Kaggle, McIntire, Reuters, and BuzzFeed Political. The dataset was published in *IEEE Transactions on Computational Social Systems*.

- **Dataset Source:** [Kaggle — Fake News Classification (WELFake Dataset)](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

### Dataset Columns

| Column | Data Type | Description |
|---|---|---|
| `Unnamed: 0` | Integer | Serial index (dropped during preprocessing) |
| `title` | String | Headline of the news article |
| `text` | String | Full article body text |
| `label` | Integer | Ground truth binary label: **`0` = FAKE**, **`1` = REAL** |

### Dataset Statistics (After Preprocessing)

| Metric | Value |
|---|---|
| **Total Cleaned Samples** | **63,303** |
| **Fake Articles (Label 0)** | 34,519 (54.5%) |
| **Real Articles (Label 1)** | 28,784 (45.5%) |
| **Training Split (80%)** | 50,642 samples |
| **Testing Split (20%)** | 12,661 samples |
| **TF-IDF Vocabulary Size** | 50,000 features |

The class distribution visualization is saved at [`results/class_distribution.png`](file:///c:/Users/Shravani/fake%20news%20detection/results/class_distribution.png).

---

## 🔧 Data Preprocessing Pipeline

Raw news articles contain syntax noise, formatting artifacts, and irrelevant characters that degrade classification accuracy. The dataset undergoes a rigorous 7-stage preprocessing pipeline:

1. **Index Dropping:** Strips unnecessary serial identifiers (`Unnamed: 0`).
2. **Label Sanitization:** Retains exclusively valid binary labels (0 for Fake, 1 for Real), discarding invalid rows.
3. **Handling Missing Values:** Fills missing `NaN` values in `title` and `text` with empty strings to prevent null pointer exceptions.
4. **Field Combination:** Concatenates `title` and `text` into a unified `content` column, ensuring the model benefits from both headline cues and full article context.
5. **Text Normalization & Cleaning:**
   - Converts all text to lowercase.
   - Removes URLs and hyperlinks (`http\S+|www\.\S+`).
   - Strips HTML markup (`<.*?>`).
   - Filters out non-alphabetic characters (`[^a-z\s]`), eliminating special punctuation and digits.
   - Collapses contiguous whitespace into single spaces and strips outer whitespace.
6. **Empty Row Pruning:** Drops any entries that result in empty strings after normalization.
7. **Deduplication:** Removes exact duplicate articles sharing identical text and labels to prevent data leakage and bias.

---

## 📊 Feature Extraction — TF-IDF Theory

Machine learning models require numerical feature vectors. **TF-IDF (Term Frequency – Inverse Document Frequency)** quantifies how important a term is to a document relative to an entire corpus.

### Theoretical Breakdown

1. **Term Frequency (TF):** Measures how frequently a word $t$ appears in a specific document $d$:
   $$\text{TF}(t, d) = \frac{\text{Count of } t \text{ in } d}{\text{Total terms in } d}$$

2. **Inverse Document Frequency (IDF):** Measures the rarity of word $t$ across the entire collection of documents $D$:
   $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
   Words that occur across almost every document (such as "the", "is", "article") yield low IDF values, whereas domain-specific or sensationalist words receive higher weights.

3. **Composite TF-IDF Weight:**
   $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

### Concrete Example
- In an article reporting *"alien invasion confirmed by officials"*, the words *"alien"* and *"invasion"* appear multiple times (high TF) and are relatively rare in mainstream journalism (high IDF), yielding a **high TF-IDF score**.
- Conversely, ubiquitous stop words like *"the"* appear everywhere (low IDF), yielding near-zero scores.
- This allows the classifiers to focus on predictive vocabulary signals characteristic of fabricated stories (e.g., hyperbole, unverified claims) versus credible reporting styles.

---

## 🤖 Machine Learning Models & Theory

Three complementary classification algorithms were trained, benchmarked, and evaluated:

### 1. Logistic Regression
- **Mechanism:** A linear model that estimates the posterior probability $P(Y=1|X)$ using the standard logistic (sigmoid) function:
  $$P(Y=1|X) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
- **Strengths:** Robust baseline, highly interpretable feature coefficients, well-calibrated probabilistic outputs, and resilient to overfitting in high-dimensional sparse spaces.

### 2. Multinomial Naive Bayes
- **Mechanism:** A probabilistic classifier based on Bayes' Theorem under the naive assumption that feature occurrences are conditionally independent given the class label:
  $$P(C_k | \mathbf{x}) \propto P(C_k) \prod_{i=1}^n P(x_i | C_k)$$
- **Strengths:** Extremely fast training time ($<0.1$s) and effective for categorical word count and frequency distributions.

### 3. Passive Aggressive Classifier (Champion Model)
- **Mechanism:** An online, margin-based classification algorithm specifically designed for large-scale learning:
  - **Passive:** If the incoming example is classified correctly with a margin exceeding 1 ($\ell_t = 0$), model weights remain untouched.
  - **Aggressive:** If a misclassification or margin violation occurs ($\ell_t > 0$), weights are updated immediately with a step size proportional to the loss:
    $$\mathbf{w}_{t+1} = \mathbf{w}_t + \text{sign}(y_t) \tau_t \mathbf{x}_t, \quad \text{where } \tau_t = \frac{\ell_t}{\|\mathbf{x}_t\|^2 + \frac{1}{2C}}$$
- **Strengths:** Exceptional convergence speed on high-dimensional text datasets and highest overall empirical accuracy.

### Model Selection Criteria
The final deployment model is selected based on the highest **weighted F1-Score** on the held-out test split (12,661 samples), ensuring a balanced trade-off between identifying true misinformation (Recall) and minimizing false accusations against legitimate journalism (Precision).

---

## 📈 Evaluation Metrics & Theoretical Definitions

| Metric | Mathematical Formula | Practical Interpretation |
|---|---|---|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall fraction of correct predictions across both classes. |
| **Precision** | $\frac{TP}{TP + FP}$ | When the model flags an article as FAKE, how frequently is it truly fake? |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | Out of all actually fake articles in the test set, what fraction did the model catch? |
| **F1-Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean balancing Precision and Recall. |
| **ROC-AUC** | Area under the ROC curve | Diagnostic capability to distinguish REAL from FAKE across all classification thresholds ($1.0 = \text{perfect}, 0.5 = \text{random}$). |
| **Confusion Matrix** | $\begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix}$ | Detailed breakdown of true positives, false positives, true negatives, and false negatives. |

---

## 🏆 Model Evaluation Results & Performance Benchmarks

The models were evaluated on an independent test split of **12,661 news articles**. The results are summarized below:

| Model | Accuracy | Precision | Recall | Weighted F1-Score | ROC-AUC | Training Time |
|---|---|---|---|---|---|---|
| **Passive Aggressive (Best)** 🏆 | **96.52%** | **96.52%** | **96.52%** | **0.9652** | **0.9951** | ~0.5 s |
| **Logistic Regression** | 95.51% | 95.50% | 95.51% | 0.9550 | 0.9926 | ~1.1 s |
| **Multinomial Naive Bayes** | 89.81% | 89.90% | 89.81% | 0.8983 | 0.9614 | ~0.1 s |

### Performance Visualizations

Generated evaluation plots from training and testing are preserved in the `results/` directory:
- **Accuracy Comparison:** [`results/model_comparison_accuracy.png`](file:///c:/Users/Shravani/fake%20news%20detection/results/model_comparison_accuracy.png)
- **Multi-Metric Comparison (Precision, Recall, F1):** [`results/model_comparison_metrics.png`](file:///c:/Users/Shravani/fake%20news%20detection/results/model_comparison_metrics.png)
- **Confusion Matrix (Champion Model):** [`results/confusion_matrix.png`](file:///c:/Users/Shravani/fake%20news%20detection/results/confusion_matrix.png)
- **Receiver Operating Characteristic (ROC) Curve:** [`results/roc_curve.png`](file:///c:/Users/Shravani/fake%20news%20detection/results/roc_curve.png)

---

## 📁 Project Structure

```
fake-news-detection/
├── app.py                  # Streamlit web application (Predict Page & Batch Analysis)
├── train_model.py          # Data pipeline, TF-IDF extraction, training, model selection
├── evaluate_model.py       # Standalone evaluation & plot regeneration
├── predict.py              # Core inference module (used by app.py & CLI)
├── requirements.txt        # Python package dependencies
├── README.md               # Complete project theory, documentation & benchmarks
├── .gitignore              # Ignored files (virtual environments, caches)
├── data/
│   ├── README.md           # Dataset download instructions
│   ├── WELFake_Dataset.csv # Raw benchmark dataset (downloaded from Kaggle)
│   └── sample_test.csv     # Synthetic sample CSV for batch testing
├── models/                 # Saved model artifacts
│   ├── tfidf_vectorizer.joblib
│   ├── best_model.joblib
│   └── model_metadata.json
└── results/                # Evaluation reports & visualization figures
    ├── evaluation_metrics.json
    ├── class_distribution.png
    ├── model_comparison_accuracy.png
    ├── model_comparison_metrics.png
    ├── confusion_matrix.png
    └── roc_curve.png
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9 or higher
- `pip` package manager
- ~300 MB free disk space for dataset and serialized models

### 1. Clone / Open Project
```bash
cd "c:/Users/Shravani/fake news detection"
```

### 2. Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Dataset Setup (If re-training)
Place `WELFake_Dataset.csv` into the `data/` folder (downloaded from [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)).

### 5. Model Training (Optional — pre-trained artifacts included)
```bash
python train_model.py
```

### 6. Launch Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 📖 Web Application Usage Guide

The Streamlit web interface is dedicated entirely to the **Predict** experience:

### 1. Single Article Prediction
1. Open the web interface.
2. Choose your preferred input mode:
   - **Combined text:** Paste a combined news headline and body.
   - **Separate headline & article:** Input the headline and body in separate dedicated fields.
3. Click **🔎 Predict**.
4. The system will display:
   - Visual result card: **`✅ REAL`** (Emerald) or **`🚫 FAKE`** (Crimson).
   - **Confidence Score:** Percentage confidence of the classifier.
   - **Prediction Details:** Expandable drawer displaying numeric label (`0` or `1`), raw confidence value, and cleaned text preview.
5. Use **🗑️ Clear** to reset inputs.

### 2. Batch Prediction via CSV
1. Scroll down to **Batch Prediction from CSV**.
2. Upload a CSV file containing at least a `text` column (optional `title` column).
3. The system predicts all rows, displays real-time summary statistics, displays a preview table, and provides a **📥 Download Predictions CSV** button.
4. You can test this feature using the provided [`data/sample_test.csv`](file:///c:/Users/Shravani/fake%20news%20detection/data/sample_test.csv).

---

## 🧪 Testing & Verification

### Command-Line Prediction Test
Run an instant test without opening the browser:
```bash
python -c "from predict import predict_single; print(predict_single('Scientists discover new deep-sea species near Hawaii'))"
```

### Validation Test Cases

| Scenario | Input | Expected Output |
|---|---|---|
| **Standard Article** | 200+ word verified news story | Predicted as `REAL` with high confidence |
| **Sensationalist Story** | Conspiratorial/clickbait headline | Predicted as `FAKE` |
| **Headline Only** | "President signs new legislative bill" | Prediction provided with short input advisory |
| **Very Short Text** | Less than 10 characters | Warning notification regarding input length |
| **Empty Input** | Blank string / whitespace | Warning prompting for text input |
| **Batch CSV** | CSV with `text` column | Batch predictions with downloadable results |
| **Invalid CSV** | CSV without `text` column | Clear error prompt indicating missing column |

---

## 🔧 Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| `ModuleNotFoundError` | Dependencies missing | Run `pip install -r requirements.txt` |
| `Model not found!` | Model artifacts missing | Run `python train_model.py` to generate `models/` artifacts |
| Port conflict on 8501 | Streamlit port in use | Run `streamlit run app.py --server.port 8502` |
| Uploaded CSV error | Missing required header | Ensure your CSV file includes a column named `text` |

---

## 📄 License & References

- **License:** Educational and research use (AI Laboratory — Experiment 10).
- **Dataset License:** CC BY 4.0 (WELFake Dataset).
- **Academic Citation:**
  > Verma, P. K., Agrawal, P., Amorim, I., & Prodan, R. (2021). "WELFake: Word Embedding Over Linguistic Features for Fake News Detection." *IEEE Transactions on Computational Social Systems*, 8(4), 881-893. DOI: [10.1109/TCSS.2021.3068519](https://doi.org/10.1109/TCSS.2021.3068519).
