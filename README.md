# 🍔 Restaurant Rating Prediction — IITM MLP Kaggle Assignment

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange?logo=scikit-learn)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-63.3%25-blue)
![Course](https://img.shields.io/badge/IIT%20Madras-MLP%20Term--2%202025-darkblue)

> **Predict the star rating (1–5) a customer gives a restaurant based on their text review and metadata.**  
> Part of the IIT Madras BS Data Science — Machine Learning Practice (MLP) course, Term-2 2025.

---

## 📌 Problem Statement

Given a dataset of restaurant reviews from McDonald's locations across the US, predict the **rating (1–5 stars)** a customer assigned based on:
- Their written review text
- Store metadata (location, category, rating count)
- Review timing

**Dataset:** 26,500 training rows · 7,000 test rows · 10 features

---

## 📁 Repository Structure

```
restaurant-rating-prediction/
│
├── notebook.ipynb       # Full notebook: EDA → Feature Engineering → Modelling → Submission
├── README.md            # This file
└── submission.csv       # Final predictions in required Kaggle format
```

---

## 🔬 Approach

### 1. Exploratory Data Analysis

- Numerical distributions: latitude, longitude, rating_count, review_time, rating
- Categorical analysis: store_name, category, store_address, review
- Scatter plot of store locations across the US
- Rating distribution (doughnut chart) — revealed class imbalance
- Review length vs sentiment score scatter, coloured by rating
- Pairplot of numerical features by rating class

### 2. Data Cleaning & Preprocessing

| Step | Detail |
|------|--------|
| Missing values | 524 missing lat/lon in train, 136 in test — imputed with **mean** |
| Duplicate removal | 3,328 duplicates removed → 23,172 clean rows |
| Outlier removal | IQR method on rating_count and review_time → 22,703 final rows |
| review_time parsing | Converted "3 months ago", "a year ago" etc. → integer months |
| rating_count cleaning | Removed commas ("5,468" → 5468) and cast to float |

### 3. Feature Engineering

| Feature Type | Details |
|-------------|---------|
| **Text cleaning** | Lowercased, removed punctuation/digits, stripped whitespace |
| **Sentiment score** | Custom lexicon: positive − negative word count per review |
| **Review length** | Character count of cleaned review |
| **TF-IDF** | 5,000 word-level features on cleaned reviews |
| **SVD reduction** | Truncated SVD → 100 dense components from TF-IDF matrix |
| **Numerical scaling** | StandardScaler on latitude, longitude, rating_count, review_time |
| **Category encoding** | LabelEncoder on restaurant category (fitted on train + test combined) |

**Final feature matrix: 22,703 × 105**

### 4. Models Benchmarked (Base)

| Model | Validation Accuracy |
|-------|-------------------|
| **Random Forest** | **0.6294** ⭐ best base |
| Extra Trees | 0.6237 |
| LightGBM | 0.6230 |
| Gradient Boosting | 0.6228 |
| Logistic Regression | 0.5807 |
| AdaBoost | 0.5345 |
| Gaussian Naive Bayes | 0.4959 |
| K-Nearest Neighbors | 0.4873 |

### 5. Hyperparameter Tuning

Tuned top 3 models using GridSearchCV / RandomizedSearchCV (3-fold CV):

| Model | Best Params | Tuned Accuracy |
|-------|-------------|---------------|
| **ExtraTrees** | n_estimators=150, max_depth=None, min_samples_split=5 | **0.6327** ⭐ |
| RandomForest | n_estimators=150, max_depth=None, min_samples_split=4 | 0.6320 |
| LightGBM | num_leaves=63, n_estimators=100, learning_rate=0.05 | 0.6303 |

---

## 📊 Final Results

| Metric | Value |
|--------|-------|
| **Best Validation Accuracy** | **63.3% (ExtraTrees Tuned)** |
| Final submission model | LightGBM (tuned) |
| Training size (after cleaning) | 22,703 rows |
| Feature dimensions | 105 (4 numerical + 1 categorical + 100 SVD text) |

---

## 🛠️ Tech Stack

```python
pandas, numpy
scikit-learn   # RF, ExtraTrees, LightGBM, GradientBoosting, LR, KNN, NB
               # TF-IDF, TruncatedSVD, StandardScaler, LabelEncoder
               # GridSearchCV, RandomizedSearchCV, Pipeline
lightgbm
matplotlib, seaborn
```

---

## 🚀 How to Run

1. Clone this repo
2. Download the dataset from the [Kaggle competition page](https://www.kaggle.com/competitions/mlp-term-2-2025-kaggle-assignment-3)
3. Place CSVs at `/kaggle/input/mlp-term-2-2025-kaggle-assignment-3/`
4. Run `notebook.ipynb` top to bottom — `submission.csv` will be generated

---

## 💡 Key Learnings

- **Text + metadata fusion matters** — combining TF-IDF embeddings (via SVD) with numerical features outperformed text-only or metadata-only approaches
- **Tree-based ensembles dominate** on tabular+NLP tasks when feature space is moderate (105 dims); deep models would need far more data to compete
- **Outlier removal improved stability** — IQR cleaning on rating_count reduced noise significantly
- **Sentiment lexicons are weak signals** — correlation between custom sentiment score and rating was only −0.111, showing that raw text embeddings (TF-IDF) carry far more signal
- **Hyperparameter tuning gave modest but consistent gains** — ExtraTrees improved from 0.6237 → 0.6327 after GridSearch

---

## 📚 Citation

```
iitmbscs2008p. MLP | Term-2 | 2025 Kaggle Assignment-3.
https://kaggle.com/competitions/mlp-term-2-2025-kaggle-assignment-3, 2025. Kaggle.
```

---

## 👤 Author

**Mohammad Afnan Shamsi**  
BS Data Science & AI — IIT Madras (3rd Year)  
[Kaggle](https://www.kaggle.com/mohammadafnanshamsi) · [LinkedIn](https://linkedin.com/in/afnan-shamsi-044a00283) · [GitHub](https://github.com/23f3004266)
