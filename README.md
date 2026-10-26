# 📰 Fake News Detection System

**AI Laboratory — Experiment No. 10**

An AI-powered Fake News Detection System using Natural Language Processing (NLP) and Machine Learning. The system analyzes news headlines and article text to predict whether news is likely **REAL** or **FAKE** based on patterns learned from a labelled dataset of 72,000+ news articles.

> ⚠️ **Disclaimer:** This system provides model predictions based on learned text patterns. It is **not** a definitive fact-checker. Always verify news through multiple trusted sources.

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.9+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| NLP Features | TF-IDF Vectorization |
| Classifiers | Logistic Regression, Multinomial Naive Bayes, Passive Aggressive Classifier |
| Visualization | Matplotlib, Seaborn |
| Web Interface | Streamlit |
| Model Persistence | Joblib |

---

## 📁 Project Structure

```
fake-news-detection/
├── app.py                  # Streamlit web application (main interface)
├── train_model.py          # Dataset loading, cleaning, training, model selection
├── evaluate_model.py       # Standalone model evaluation & plot regeneration
├── predict.py              # Reusable prediction module (used by app.py & CLI)
├── requirements.txt        # Python package dependencies
├── README.md               # This file
├── .gitignore              # Files excluded from version control
├── data/
│   ├── README.md           # Dataset download instructions
│   ├── WELFake_Dataset.csv # ← YOU download this from Kaggle (not in repo)
│   └── sample_test.csv     # Small synthetic sample for testing the interface
├── models/                 # ← Generated after running train_model.py
│   ├── tfidf_vectorizer.joblib
│   ├── best_model.joblib
│   └── model_metadata.json
├── results/                # ← Generated after running train_model.py
│   ├── evaluation_metrics.json
│   ├── class_distribution.png
│   ├── model_comparison_accuracy.png
│   ├── model_comparison_metrics.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
└── notebooks/              # Optional exploration notebooks
    └── exploration.ipynb   # (optional — for Google Colab)
```

### Files You Create vs. Generated Files

| File | Source |
|---|---|
| `app.py`, `train_model.py`, `evaluate_model.py`, `predict.py` | **You create** (provided in this project) |
| `requirements.txt`, `README.md`, `.gitignore` | **You create** (provided in this project) |
| `data/WELFake_Dataset.csv` | **You download** from Kaggle |
| `models/*`, `results/*` | **Auto-generated** by `train_model.py` |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- ~300 MB free disk space (for dataset + model artifacts)

### Step 1: Clone or Create the Project
```bash
# Create project directory (if not already done)
mkdir fake-news-detection
cd fake-news-detection

# Place all source files in this directory
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download the Dataset
1. Visit: **https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification**
2. Click **Download** (requires a free Kaggle account).
3. Extract the ZIP file.
4. Place `WELFake_Dataset.csv` in the `data/` folder.

**Alternative — Kaggle API:**
```bash
pip install kaggle
# Place your kaggle.json API key in ~/.kaggle/
kaggle datasets download -d saurabhshahane/fake-news-classification -p data/
# Then unzip:
cd data && tar -xf fake-news-classification.zip && cd ..
```

### Step 5: Train the Model
```bash
python train_model.py
```

This will:
- Load and clean the dataset
- Extract TF-IDF features
- Train 3 classifiers
- Evaluate and select the best model
- Save model artifacts to `models/`
- Save evaluation plots and metrics to `results/`

**Expected time:** 2–5 minutes depending on your hardware.

### Step 6: Launch the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📖 Usage Guide

### Single Prediction
1. Go to the **🔍 Predict** page.
2. Enter a news headline or article text.
3. Click **🔎 Predict**.
4. View the result (REAL or FAKE) with confidence score.

### Batch Prediction
1. On the **🔍 Predict** page, scroll to **Batch Prediction from CSV**.
2. Upload a CSV file with a `text` column (optionally, a `title` column).
3. View predictions and download the results as CSV.

### View Performance
- Go to **📊 Performance** to see evaluation metrics, comparison charts, and the confusion matrix.

### Dataset Info
- Go to **📁 Dataset Info** for dataset statistics, preprocessing details, and TF-IDF explanation.

---

## 🧪 Testing

### Quick Command-Line Test
```bash
python predict.py "Scientists discover new species of deep-sea fish near Hawaii"
```

### Test Cases

| # | Input | Expected Behavior |
|---|---|---|
| 1 | A well-formatted news article (~200 words) | Prediction displayed as REAL or FAKE |
| 2 | A headline only (e.g., "President signs new bill") | Prediction displayed (may warn about short input) |
| 3 | Empty input | Warning: "Please enter some text" |
| 4 | Very short text (< 10 chars) | Warning about unreliable prediction |
| 5 | A long article (1000+ words) | Prediction displayed normally |
| 6 | CSV without `text` column | Error: "must have a text column" |
| 7 | Missing model artifacts | Error: "run train_model.py first" |
| 8 | Borderline/uncertain text | Prediction shown with disclaimer |

### Using the Sample Test File
```bash
# Test batch prediction with the provided sample file
# Upload data/sample_test.csv through the Streamlit interface
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| "Dataset not found" | Download and place `WELFake_Dataset.csv` in `data/` |
| "Model not found" | Run `python train_model.py` |
| Streamlit won't start | Check if port 8501 is free; try `streamlit run app.py --server.port 8502` |
| `UnicodeDecodeError` | The dataset should be UTF-8. Re-download if corrupted. |
| Very slow training | Normal on older hardware. ~72K articles × 50K features takes time. |

---

## 📊 About the Models

### Logistic Regression
A linear classifier that predicts the probability of a class (REAL/FAKE) using a logistic function. Works well for text classification because high-dimensional sparse features (like TF-IDF) are naturally suited to linear models.

### Multinomial Naive Bayes
Based on Bayes' theorem with the assumption that features (word frequencies) are independent. Very fast and often surprisingly effective for text classification.

### Passive Aggressive Classifier
An online learning algorithm that stays "passive" when the prediction is correct but becomes "aggressive" when it makes an error, updating the model weights significantly. Good for large-scale text classification.

### Model Selection
The final model is selected automatically based on the highest **weighted F1-score** on the test set. F1-score balances precision (avoiding false alarms) and recall (catching all fake news), making it a robust metric for this task.

---

## 📄 License

This project is for educational purposes (AI Laboratory — Experiment 10).
The WELFake dataset is licensed under CC BY 4.0.

## 📚 References

1. Verma, P.K., et al. "WELFake: Word Embedding Over Linguistic Features for Fake News Detection." *IEEE Transactions on Computational Social Systems*, 2021. DOI: 10.1109/TCSS.2021.3068519.
2. Kaggle Dataset: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
