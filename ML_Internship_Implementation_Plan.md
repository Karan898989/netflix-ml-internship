# Auspify Machine Learning Internship — Implementation Plan
### 4-Week Practical Program (Complete any 4 of 6 tasks)

---

## 1. Recommended Task Selection

All 6 tasks use the **Netflix titles dataset** (genres, cast, country, rating, release year, description, etc.), so one clean dataset can power your whole internship. To show breadth (recommendation, classification, clustering, forecasting) while staying realistic in 4 weeks, this plan selects:

| Priority | Task | Difficulty | Why |
|---|---|---|---|
| 1 | Task 1 – Netflix Content Recommendation System | Easy | Core NLP/similarity skill, strong portfolio demo |
| 2 | Task 2 – Content Type Prediction Model | Easy | Fast win, builds classification foundation for Task 3 |
| 3 | Task 4 – Netflix Content Segmentation | Medium | Different ML paradigm (unsupervised), visually impressive |
| 4 | Task 6 – Netflix Content Success Analytics Engine | Advanced | Capstone that reuses/combines everything above — best "final showcase" project |

*(Optional stretch, if time allows: Task 3 – Rating Classification, or Task 5 – Trend Forecasting.)*

You can swap any of these — the workflow below is written so the environment setup and dataset prep are shared, and each task section is self-contained.

---

## 2. Environment & Tools Setup (Day 1)

- **Language**: Python 3.10+
- **Core libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`/`seaborn`, `nltk` or `scikit-learn TfidfVectorizer`
- **Dev environment**: Jupyter Notebook / Google Colab (fastest for internship pace) or VS Code
- **Version control**: Git + GitHub repo — one repo, one folder per task
- **Excel**: for quick data checks / evaluation logs (as listed on the task PDF)

**Suggested repo structure:**
```
netflix-ml-internship/
├── data/
│   └── netflix_titles.csv
├── notebooks/
│   ├── task1_recommendation.ipynb
│   ├── task2_content_type.ipynb
│   ├── task4_segmentation.ipynb
│   └── task6_analytics_engine.ipynb
├── screenshots/
├── README.md
└── requirements.txt
```

**Dataset**: Use the public "Netflix Movies and TV Shows" dataset (Kaggle) — it has the exact columns needed (title, type, director, cast, country, date_added, release_year, rating, duration, listed_in/genres, description).

---

## 3. 4-Week Schedule

### Week 1 — Setup + Task 1 (Recommendation System)
- Day 1–2: Environment setup, load & clean dataset (handle nulls in `director`, `cast`, `country`)
- Day 3: Feature prep — combine `listed_in`, `description`, `cast` into a single "content soup" text field
- Day 4: Convert text to vectors with `TfidfVectorizer`
- Day 5: Compute cosine similarity matrix; build `get_recommendations(title)` function
- Day 6: Test on 5–10 sample titles, tune weighting between genre vs. description
- Day 7: Evaluate qualitatively (do results make sense?), write short README, screenshot outputs

### Week 2 — Task 2 (Content Type Prediction)
- Day 1: Select features (`rating`, `duration`, `listed_in`, `country`, `release_year`)
- Day 2: Encode categorical variables (One-Hot / Label Encoding), handle missing values
- Day 3: Train/test split; train Logistic Regression + Decision Tree baseline
- Day 4: Train Random Forest; compare accuracy, precision, recall, F1
- Day 5: Confusion matrix + feature importance plot
- Day 6: Pick best model, document why
- Day 7: Clean notebook, commit to GitHub, screenshot results

### Week 3 — Task 4 (Content Segmentation)
- Day 1: Feature engineering — encode genre lists (multi-label binarization), scale numeric features (`release_year`, duration)
- Day 2: Determine optimal cluster count (Elbow Method / Silhouette Score)
- Day 3: Run K-Means clustering
- Day 4: Reduce dimensions (PCA or t-SNE) for 2D visualization
- Day 5: Plot clusters, label them (e.g., "International Dramas," "Kids Comedy," etc.)
- Day 6: Interpret each cluster's defining traits
- Day 7: Write findings summary, screenshot visualizations

### Week 4 — Task 6 (Content Success Analytics Engine — Capstone)
- Day 1: Advanced feature engineering — combine outputs/insights from Tasks 1, 2, 4 (e.g., cluster label as a feature)
- Day 2–3: Build 2–3 models addressing a business question (e.g., "predict content popularity proxy" or "predict genre trend by country/year")
- Day 4: Compare model performance (cross-validation, metrics table)
- Day 5: Generate automated insights (top genres by country, growth trends, cluster-to-type relationships)
- Day 6: Build a simple visual report (matplotlib/seaborn dashboard or a one-page summary)
- Day 7: Final polish — README, demo video (optional), submission packaging

---

## 4. Per-Task Technical Checklist

**Task 1 – Recommendation System**
- [ ] Clean text fields, build "content soup"
- [ ] TF-IDF vectorization
- [ ] Cosine similarity matrix
- [ ] Top-N recommendation function
- [ ] Sample outputs + brief evaluation notes

**Task 2 – Content Type Prediction**
- [ ] Feature selection + encoding
- [ ] ≥2 classification models trained
- [ ] Accuracy/precision/recall/F1 reported
- [ ] Confusion matrix visual

**Task 4 – Content Segmentation**
- [ ] Feature scaling + encoding
- [ ] Elbow/silhouette analysis for k
- [ ] K-Means clustering
- [ ] 2D visualization (PCA/t-SNE)
- [ ] Cluster interpretation write-up

**Task 6 – Analytics Engine (Capstone)**
- [ ] End-to-end pipeline (load → clean → feature engineer → model → insight)
- [ ] Multiple models compared
- [ ] Automated/summarized insights
- [ ] Final visual report

---

## 5. Submission & Evaluation Prep

Per the program's requirements, prepare for each completed task:
- Source code (notebook or `.py`)
- Screenshots of key outputs (recommendations list, confusion matrix, cluster plot, final report)
- GitHub repo link
- (Optional) short screen-recorded demo walkthrough

Keep in mind the evaluation criteria: task completion, code quality, creativity/implementation, and professional presentation — so comment your code, write a short README per task, and keep folders organized.

---

## 6. Weekly Time Budget (approx., for a ~10–12 hr/week pace)

| Week | Focus | Est. Hours |
|---|---|---|
| 1 | Setup + Task 1 | 10–12 |
| 2 | Task 2 | 8–10 |
| 3 | Task 4 | 10–12 |
| 4 | Task 6 (Capstone) | 12–14 |

---

*Tip: Share progress on LinkedIn using #Auspify #AuspifyInternship #AuspifyProjects as you complete each task — this builds the "professional showcase" the program encourages.*
