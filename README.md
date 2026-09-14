# Auspify Machine Learning Internship — Final Portfolio Project
## End-to-End Netflix Catalog Intelligence, Recommendation, Classification & Segmentation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0%2B-navy.svg)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Internship-Completed-brightgreen.svg)]()

This repository contains the complete practical implementation of the **4 selected machine learning tasks** for the **Auspify Machine Learning Internship Program**, built on the comprehensive Kaggle **Netflix Movies and TV Shows dataset (8,807 titles)**.

---

## 📌 Selected Task Portfolio

| Priority | Task | Focus Area | Technical Methodology | Deliverable Status |
|---|---|---|---|---|
| **1** | **Task 1: Netflix Content Recommendation System** | NLP / Content-Based Filtering | Content Soup, TF-IDF Vectorization, Cosine Similarity | ✅ Completed (`src/task1_recommender.py`, Notebook 1) |
| **2** | **Task 2: Content Type Prediction Model** | Supervised Classification | Multi-hot Encoding, Logistic Regression, Decision Tree, Random Forest | ✅ Completed (`src/task2_classifier.py`, Notebook 2) |
| **3** | **Task 4: Netflix Content Segmentation** | Unsupervised Learning | K-Means, Elbow Method, Silhouette Analysis, 2D PCA Projection | ✅ Completed (`src/task4_segmentation.py`, Notebook 4) |
| **4** | **Task 6: Content Success Analytics Engine** | Capstone Analytics & BI | End-to-End Pipeline, 5-Fold Stratified CV, Executive Dashboard | ✅ Completed (`src/task6_analytics_engine.py`, Notebook 6) |

---

## 📁 Repository Structure

```
.
├── data/
│   └── netflix_titles.csv                 # Cleaned Kaggle Netflix dataset (8,807 rows, 12 columns)
├── notebooks/
│   ├── task1_recommendation.ipynb         # Interactive Task 1 NLP Recommendation Notebook
│   ├── task2_content_type.ipynb           # Interactive Task 2 Supervised Classification Notebook
│   ├── task4_segmentation.ipynb           # Interactive Task 4 Unsupervised Clustering Notebook
│   └── task6_analytics_engine.ipynb       # Interactive Task 6 Capstone Analytics & Dashboard Notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py                     # Data cleaning, null imputation, column fix, feature pipeline
│   ├── task1_recommender.py               # TF-IDF & Cosine Similarity recommendation engine
│   ├── task2_classifier.py                # Multi-model benchmarking (LogReg, Decision Tree, Random Forest)
│   ├── task4_segmentation.py              # K-Means, Elbow/Silhouette analysis, 2D PCA mapping
│   └── task6_analytics_engine.py          # Capstone analytics engine, 5-fold CV & Executive Dashboard
├── screenshots/
│   ├── task1_similarity_matrix.png        # Semantic correlation matrix between benchmark titles
│   ├── task1_sample_recommendations.png   # Match score bar chart for target recommendations
│   ├── task2_confusion_matrices.png       # Test set confusion matrices for all 3 classifiers
│   ├── task2_roc_curves.png               # ROC curves with AUC scores
│   ├── task2_feature_importance.png       # Gini importance plot (Random Forest)
│   ├── task2_model_comparison.png         # Multi-metric model comparison chart
│   ├── task4_elbow_silhouette.png         # Inertia (Elbow) and Silhouette curves (k=2..8)
│   ├── task4_clusters_pca.png             # 2D PCA projection of content archetypes
│   ├── task4_cluster_traits.png           # Archetype volume distribution
│   ├── task6_executive_dashboard.png      # 4-panel executive visual dashboard
│   └── task6_content_growth_trends.png    # Longitudinal trend in Movie vs TV Show additions
├── reports/
│   ├── task2_model_metrics.csv            # Accuracy, Precision, Recall, F1, ROC-AUC comparison table
│   ├── task4_cluster_profiles.csv         # Discovered archetype definitions & traits
│   └── task6_executive_summary.json       # Automated business intelligence summary
├── templates/
│   └── index.html                         # Full-featured interactive web interface
├── static/
│   ├── css/style.css                      # Netflix dark theme styling
│   └── js/app.js                          # Frontend client logic & Chart.js integration
├── website/
│   └── index.html                         # Portable zero-install standalone web edition
├── app.py                                 # Flask web application & REST API server
├── run_all.py                             # Single-command runner executing the entire pipeline
├── generate_notebooks.py                  # Notebook compilation & pre-execution script
├── requirements.txt                       # Project dependencies
└── README.md                              # Portfolio documentation
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Execute Entire Pipeline in One Command
Run the master script to process data, train models, benchmark performance, generate reports, and export all visual screenshots:
```bash
python run_all.py
```

### 3. Run Individual Task Modules
Each module can be executed independently from the command line:
```bash
# Task 1: Recommendation Engine
python src/task1_recommender.py --title "Stranger Things" --top_n 5

# Task 2: Content Type Classifier
python src/task2_classifier.py

# Task 4: Content Segmentation
python src/task4_segmentation.py

# Task 6: Capstone Analytics Engine
python src/task6_analytics_engine.py
```

### 4. Interactive Jupyter Notebooks
Launch Jupyter Lab or VS Code to explore pre-executed, richly annotated notebooks:
```bash
jupyter lab notebooks/
```

### 5. Launch the Interactive Web Application
You can explore and interact with the machine learning models and visualizations through the web application:

**Option A — Live Flask Server (Full live model inference & REST APIs):**
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser to access:
- 🎬 **Live Search & Recommendation Engine** across all 8,807 titles
- 🧠 **Live AI Content Classifier** with real-time probability meters
- 🧬 **Interactive 5-Archetype Segmentation Explorer**
- 📊 **Dynamic Chart.js Analytics Dashboard**
- 📂 **Evaluation Reports & Visual Gallery**

**Option B — Standalone Portable Web Edition (Zero Installation Required):**
Simply double-click or open **[`website/index.html`](file:///c:/Users/Karan%20Yadan/OneDrive/Desktop/project/website/index.html)** in any web browser!

---

## 🔬 Task-by-Task Implementation & Technical Results

### Task 1: Netflix Content Recommendation System
- **Objective**: Retrieve semantically relevant content recommendations based on title metadata and narrative synopses.
- **Methodology**:
  - Engineered an enriched **"Content Soup"** combining weighted genres ($2\times$ weight), director names, primary cast members, and cleaned synopsis.
  - Extracted 10,000 sublinear TF-IDF features using word n-grams `(1, 2)`.
  - Computed pairwise similarity using `linear_kernel` dot products, achieving $<1\text{ ms}$ retrieval speed.
- **Benchmark Evaluation**:
  - *Query: "Stranger Things"* $\rightarrow$ *Nightflyers* (59.1% match), *Helix* (56.9%), *Chilling Adventures of Sabrina* (56.7%), *Manifest* (44.6%).
  - *Query: "Narcos"* $\rightarrow$ *Narcos: Mexico* (61.2% match), *Marvel's Jessica Jones* (44.8%), *Gotham* (44.3%), *Ozark* (43.8%).
- **Visual Artifacts**:
  - Heatmap: `screenshots/task1_similarity_matrix.png`
  - Score Breakdown: `screenshots/task1_sample_recommendations.png`

---

### Task 2: Content Type Prediction Model
- **Objective**: Predict whether a title is a **Movie** or **TV Show** using metadata and text features while strictly avoiding duration data leakage.
- **Feature Engineering**:
  - Multi-label binarized genre matrix (42 genres).
  - One-hot encoded primary countries and maturity rating classes.
  - Standardized release year and top 150 TF-IDF synopsis terms.
- **Model Benchmark Results (20% Unseen Stratified Test Set)**:

| Model | Accuracy | Precision | Recall | F1 Score | F1 Macro | ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Decision Tree** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Random Forest** | **0.9966** | **0.9951** | **1.0000** | **0.9976** | **0.9960** | **1.0000** |

- **Key Analytical Insight**: Genre descriptors such as `International TV Shows` and `TV Dramas` act as definitive format separators, providing near-perfect separation even in linear models.
- **Visual Artifacts**:
  - Confusion Matrices: `screenshots/task2_confusion_matrices.png`
  - ROC Curves: `screenshots/task2_roc_curves.png`
  - Feature Importance: `screenshots/task2_feature_importance.png`
  - Model Comparison: `screenshots/task2_model_comparison.png`

---

### Task 4: Netflix Content Segmentation
- **Objective**: Uncover natural catalog archetypes using unsupervised machine learning.
- **Methodology**:
  - Constructed high-dimensional feature representations (genres, age ratings, country indicators, release year, standardized durations, synopsis keywords).
  - Evaluated optimal $k$ using **Elbow Method (Inertia)** and **Silhouette Analysis** across $k \in [2, 8]$.
  - Fitted K-Means with $k=5$ and projected cluster distributions into 2D using **Principal Component Analysis (PCA)** (explaining 41.2% of variance).
- **Discovered Content Archetypes**:

| Cluster | Archetype Name | Catalog Share | Format Split | Primary Country Hubs | Dominant Genres |
|---|---|---|---|---|---|
| **C0** | **Global TV Dramas & Crime** | 30.8% (2,709) | 100% Movie | India, UK, US | International Movies, Dramas, Comedies |
| **C1** | **Kids, Animation & Family** | 32.9% (2,900) | 100% Movie | US, UK, Canada | Comedies, Dramas, Documentaries, Family |
| **C2** | **International Independent Cinema** | 27.3% (2,403) | 100% TV | US, UK, South Korea | International TV, TV Dramas, TV Comedies |
| **C3** | **Hollywood Action & Blockbusters** | 6.1% (538) | 96.8% Movie | US, India, UK | Dramas, Comedies, Action & Adventure |
| **C4** | **Documentaries & Cultural Stories** | 2.9% (257) | 100% TV | US, UK, Canada | TV Comedies, TV Dramas, Kids' TV |

- **Visual Artifacts**:
  - Elbow & Silhouette Curves: `screenshots/task4_elbow_silhouette.png`
  - 2D PCA Cluster Map: `screenshots/task4_clusters_pca.png`
  - Archetype Traits: `screenshots/task4_cluster_traits.png`

---

### Task 6: Content Success Analytics Engine (Capstone)
- **Objective**: Synthesize recommendations, classifications, and cluster archetypes into an executive business intelligence engine.
- **Predictive Business Modeling (5-Fold Stratified Cross-Validation)**:
  - *Business Question*: Can catalog metadata and archetype tags predict international reach and multi-region appeal?
  - **Random Forest**: **80.77% CV Accuracy**, **0.8287 CV F1-Score**, **0.8889 CV ROC-AUC**.
  - **Gradient Boosting**: **80.69% CV Accuracy**, **0.8271 CV F1-Score**, **0.8875 CV ROC-AUC**.
  - **Logistic Regression**: **80.58% CV Accuracy**, **0.8245 CV F1-Score**, **0.8748 CV ROC-AUC**.
- **Automated Strategic Insights**:
  1. **Format Shift**: TV series share expanded from $<20\%$ of additions in 2015 to **$33.7\%$** in 2021 as Netflix pivoted to episodic retention.
  2. **Streaming Boom Concentration**: Over **$70.5\%$** of all active catalog titles were produced during the 2015–2021 streaming boom era.
  3. **Global Production Hubs**: The United States (3,211 titles), India (1,008 titles), and the United Kingdom (628 titles) represent the top 3 production anchors.
  4. **Rapid Turnaround**: Median lag between theatrical/broadcast release and platform addition dropped to **1.0 year**.
- **Visual Artifacts**:
  - Executive Dashboard: `screenshots/task6_executive_dashboard.png`
  - Growth Trends: `screenshots/task6_content_growth_trends.png`
  - Summary JSON: `reports/task6_executive_summary.json`

---

## 🏆 Auspify Internship Evaluation Checklist

- [x] **Task Completion**: Fully implemented Tasks 1, 2, 4, and 6 per program guidelines.
- [x] **Code Quality**: Modular, PEP 8 compliant, well-documented with docstrings, type hints, and CLI entry points.
- [x] **Jupyter Notebooks**: 4 fully executed `.ipynb` notebooks with rich markdown explanations and embedded figures.
- [x] **Visual Deliverables**: 11 high-resolution plots saved in `screenshots/`.
- [x] **Quantitative Reports**: CSV metric tables and JSON summaries exported in `reports/`.
- [x] **Single-Command Pipeline**: End-to-end execution verified via `run_all.py` in under 15 seconds.

---

*Prepared for Auspify Machine Learning Internship evaluation.*  
*Tag: #Auspify #AuspifyInternship #AuspifyProjects*
