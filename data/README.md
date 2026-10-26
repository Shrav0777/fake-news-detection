# Fake News Detection - Data Directory

## Purpose
This directory stores the dataset files used for training and evaluation.

## Dataset: WELFake (Fake News Classification)

**Kaggle URL:** https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

### How to Download

#### Method 1: Manual Download (Recommended)
1. Visit the dataset URL above.
2. Click the **"Download"** button (you need a free Kaggle account).
3. Extract the downloaded ZIP file.
4. Place the CSV file (`WELFake_Dataset.csv`) in this `data/` directory.

#### Method 2: Kaggle API
```bash
pip install kaggle
# Configure ~/.kaggle/kaggle.json with your API key
kaggle datasets download -d saurabhshahane/fake-news-classification
# Extract to data/ directory
```

### Expected File
After download, this directory should contain:
```
data/
└── WELFake_Dataset.csv
```

### Dataset Columns
| Column      | Description                                   |
|-------------|-----------------------------------------------|
| Unnamed: 0  | Serial number / row index (will be dropped)   |
| title       | News article headline                         |
| text        | Full article body text                        |
| label       | 0 = Fake News, 1 = Real News                  |

### Dataset Statistics
- **Total entries:** ~72,134 usable rows (78,098 in CSV, some inaccessible)
- **Real news:** ~35,028
- **Fake news:** ~37,106
- **License:** CC BY 4.0

### Important Notes
- Do NOT commit the CSV file to version control (it is in `.gitignore`).
- The training script will validate the dataset before proceeding.
