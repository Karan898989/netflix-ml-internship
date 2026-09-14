"""
Task 2: Content Type Prediction Model
Supervised machine learning to classify whether a Netflix title is a 'Movie' or 'TV Show'
based on genre composition, maturity rating, country of origin, release year, and synopsis features.
Benchmarking: Logistic Regression vs Decision Tree vs Random Forest.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

from src.data_loader import clean_netflix_data


def prepare_classification_features(df):
    """
    Builds non-leaking feature matrix from metadata & text:
    - Multi-hot genres (from listed_in)
    - One-hot top countries
    - One-hot content ratings
    - Standardized release year
    - TF-IDF synopsis features
    """
    print("[Task 2] Engineering feature matrix...")
    df = df.copy()

    # 1. Target variable: Movie = 1, TV Show = 0
    y = (df['type'] == 'Movie').astype(int)

    feature_dfs = []

    # 2. Multi-label Binarization for Genres
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df['genres_list'])
    genre_feature_names = [f"genre_{g.replace(' ', '_').lower()}" for g in mlb.classes_]
    df_genres = pd.DataFrame(genre_matrix, columns=genre_feature_names, index=df.index)
    feature_dfs.append(df_genres)

    # 3. Top Countries One-Hot Encoding
    top_countries = df['primary_country'].value_counts().head(12).index.tolist()
    country_series = df['primary_country'].apply(lambda c: c if c in top_countries else 'Other_Country')
    df_country = pd.get_dummies(country_series, prefix='country', dtype=int)
    feature_dfs.append(df_country)

    # 4. Rating One-Hot Encoding
    top_ratings = df['rating'].value_counts().head(8).index.tolist()
    rating_series = df['rating'].apply(lambda r: r if r in top_ratings else 'Other_Rating')
    df_rating = pd.get_dummies(rating_series, prefix='rating', dtype=int)
    feature_dfs.append(df_rating)

    # 5. Scaled Release Year
    scaler = StandardScaler()
    scaled_year = scaler.fit_transform(df[['release_year']])
    df_year = pd.DataFrame(scaled_year, columns=['scaled_release_year'], index=df.index)
    feature_dfs.append(df_year)

    # 6. TF-IDF on Description (top 150 terms)
    tfidf = TfidfVectorizer(max_features=150, stop_words='english')
    desc_matrix = tfidf.fit_transform(df['description']).toarray()
    desc_feature_names = [f"tfidf_{w}" for w in tfidf.get_feature_names_out()]
    df_desc = pd.DataFrame(desc_matrix, columns=desc_feature_names, index=df.index)
    feature_dfs.append(df_desc)

    X = pd.concat(feature_dfs, axis=1)
    print(f"[Task 2] Feature matrix constructed: {X.shape[0]} rows, {X.shape[1]} features.")
    return X, y, X.columns.tolist()


class ContentTypeClassifier:
    """Orchestrates model training, benchmarking, and visual report generation."""

    def __init__(self, df=None):
        if df is None:
            df = clean_netflix_data()
        self.df = df
        self.X, self.y, self.feature_names = prepare_classification_features(self.df)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=12, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=16, random_state=42, n_jobs=-1)
        }
        self.results = {}
        self.predictions = {}
        self.probabilities = {}

    def train_and_evaluate(self):
        """Fits all benchmark models and computes evaluation metrics."""
        print("\n--- Task 2: Model Training & Benchmarking ---")
        metrics_list = []

        for name, model in self.models.items():
            print(f"[Task 2] Training {name}...")
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

            self.predictions[name] = y_pred
            self.probabilities[name] = y_prob

            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred)
            rec = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            f1_macro = f1_score(self.y_test, y_pred, average='macro')
            roc_auc = roc_auc_score(self.y_test, y_prob)

            metrics = {
                'Model': name,
                'Accuracy': round(acc, 4),
                'Precision': round(prec, 4),
                'Recall': round(rec, 4),
                'F1 Score': round(f1, 4),
                'F1 Macro': round(f1_macro, 4),
                'ROC AUC': round(roc_auc, 4)
            }
            metrics_list.append(metrics)
            print(f"  -> Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

        self.results_df = pd.DataFrame(metrics_list)
        return self.results_df

    def save_reports_and_plots(self, screenshots_dir='screenshots', reports_dir='reports'):
        """Saves evaluation table to CSV and plots confusion matrices, ROC curves, feature importances."""
        os.makedirs(screenshots_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        # 1. Save CSV metrics report
        csv_path = os.path.join(reports_dir, 'task2_model_metrics.csv')
        self.results_df.to_csv(csv_path, index=False)
        print(f"[Task 2] Saved metrics report to {csv_path}")

        # 2. Confusion Matrices Plot
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for idx, (name, y_pred) in enumerate(self.predictions.items()):
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['TV Show', 'Movie'],
                yticklabels=['TV Show', 'Movie'],
                cbar=False
            )
            axes[idx].set_title(f'{name}\n(Acc: {self.results_df.loc[self.results_df["Model"]==name, "Accuracy"].values[0]:.3f})', fontweight='bold')
            axes[idx].set_xlabel('Predicted Label')
            axes[idx].set_ylabel('True Label')
        plt.suptitle('Task 2: Confusion Matrices Across Benchmark Models', fontsize=14, fontweight='bold', y=1.03)
        plt.tight_layout()
        cm_path = os.path.join(screenshots_dir, 'task2_confusion_matrices.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Task 2] Saved confusion matrices to {cm_path}")

        # 3. ROC Curves Plot
        plt.figure(figsize=(8, 6))
        for name, y_prob in self.probabilities.items():
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            auc_val = roc_auc_score(self.y_test, y_prob)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_val:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guessing')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title('Task 2: Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold', pad=12)
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        roc_path = os.path.join(screenshots_dir, 'task2_roc_curves.png')
        plt.savefig(roc_path, dpi=300)
        plt.close()
        print(f"[Task 2] Saved ROC curves to {roc_path}")

        # 4. Feature Importance (Random Forest)
        rf_model = self.models['Random Forest']
        importances = rf_model.feature_importances_
        feat_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        bars = plt.barh(feat_df['Feature'], feat_df['Importance'], color='#1f77b4')
        plt.xlabel('Gini Importance', fontweight='bold')
        plt.ylabel('Engineered Feature', fontweight='bold')
        plt.title('Task 2: Top 15 Most Informative Features (Random Forest)', fontsize=13, fontweight='bold', pad=12)
        plt.gca().invert_yaxis()
        for bar in bars:
            w = bar.get_width()
            plt.text(w + 0.001, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va='center', fontsize=9)
        plt.tight_layout()
        fi_path = os.path.join(screenshots_dir, 'task2_feature_importance.png')
        plt.savefig(fi_path, dpi=300)
        plt.close()
        print(f"[Task 2] Saved feature importance plot to {fi_path}")

        # 5. Model Metric Comparison Bar Chart
        melted_df = pd.melt(
            self.results_df, id_vars=['Model'],
            value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
            var_name='Metric', value_name='Score'
        )
        plt.figure(figsize=(10, 5))
        sns.barplot(data=melted_df, x='Metric', y='Score', hue='Model', palette='viridis')
        plt.ylim(0.7, 1.0)
        plt.title('Task 2: Comprehensive Model Benchmark Comparison', fontsize=13, fontweight='bold', pad=12)
        plt.ylabel('Score (0-1.0)', fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        comp_path = os.path.join(screenshots_dir, 'task2_model_comparison.png')
        plt.savefig(comp_path, dpi=300)
        plt.close()
        print(f"[Task 2] Saved model comparison chart to {comp_path}")


def main():
    classifier = ContentTypeClassifier()
    metrics_df = classifier.train_and_evaluate()
    print("\nBenchmark Summary Table:")
    print(metrics_df.to_string(index=False))
    classifier.save_reports_and_plots()


if __name__ == '__main__':
    main()
