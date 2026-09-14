"""
Task 6: Netflix Content Success Analytics Engine (Capstone)
End-to-end business intelligence engine combining recommendation indices,
content classification, and cluster archetypes.
Features automated strategic insights, multi-model predictive benchmarking,
and an executive visual analytics dashboard.
"""

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data_loader import clean_netflix_data
from src.task4_segmentation import ContentSegmenter


class NetflixAnalyticsEngine:
    """Capstone executive analytics engine integrating all ML components."""

    def __init__(self, df=None):
        print("\n--- Task 6 Capstone: Initializing Analytics Engine ---")
        if df is None:
            self.df = clean_netflix_data()
        else:
            self.df = df.copy()
            
        self._enrich_pipeline()

    def _enrich_pipeline(self):
        """Runs segmentation to attach cluster tags and engineers executive KPIs."""
        print("[Task 6] Enriching catalog with cluster archetypes from Task 4...")
        segmenter = ContentSegmenter(self.df, optimal_k=5)
        segmenter.fit_clusters(k=5)
        segmenter.profile_clusters()
        self.df['cluster'] = segmenter.df['cluster']
        self.archetype_map = dict(zip(segmenter.profiles_df['cluster'], segmenter.profiles_df['archetype_name']))
        self.df['archetype_name'] = self.df['cluster'].map(self.archetype_map)

        # Content release era
        def assign_era(year):
            if year < 2000:
                return 'Classic (Pre-2000)'
            elif year <= 2014:
                return 'Modern Era (2000-2014)'
            else:
                return 'Streaming Boom (2015-2021)'

        self.df['release_era'] = self.df['release_year'].apply(assign_era)

        # Global reach proxy: multi-country production or international genre tag
        self.df['is_international'] = self.df['listed_in'].str.contains('International', case=False) | (self.df['country'].str.contains(','))
        self.df['target_global_appeal'] = self.df['is_international'].astype(int)

        # Catalog turnaround lag (years between release and addition to Netflix)
        self.df['addition_lag_years'] = (self.df['year_added'] - self.df['release_year']).clip(lower=0)

        print(f"[Task 6] Dataset enriched. Shape: {self.df.shape}")

    def benchmark_predictive_models(self):
        """
        Builds and compares 3 models to address an executive business question:
        'Can metadata and genre composition predict Global Appeal / International Reach?'
        Uses 5-Fold Stratified Cross-Validation.
        """
        print("\n--- Task 6: Predictive Modeling Benchmark (Global Appeal Likelihood) ---")

        # Feature preparation
        top_genres = ['Dramas', 'Comedies', 'Action & Adventure', 'Documentaries', 'Kids\' TV', 'Romantic Movies']
        for g in top_genres:
            self.df[f'has_{g.lower().replace(" ", "_")}'] = self.df['listed_in'].str.contains(g, regex=False).astype(int)

        X_cols = [f'has_{g.lower().replace(" ", "_")}' for g in top_genres] + ['release_year', 'addition_lag_years', 'cluster']
        X = pd.get_dummies(self.df[X_cols], columns=['cluster'], drop_first=True)
        y = self.df['target_global_appeal']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        models = {
            'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        benchmark_records = []

        for name, model in models.items():
            scores = cross_validate(
                model, X_scaled, y, cv=cv,
                scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            )
            rec = {
                'Model': name,
                'CV Accuracy': round(scores['test_accuracy'].mean(), 4),
                'CV Precision': round(scores['test_precision'].mean(), 4),
                'CV Recall': round(scores['test_recall'].mean(), 4),
                'CV F1': round(scores['test_f1'].mean(), 4),
                'CV ROC-AUC': round(scores['test_roc_auc'].mean(), 4)
            }
            benchmark_records.append(rec)
            print(f"  {name:20s} | Acc: {rec['CV Accuracy']:.4f} | F1: {rec['CV F1']:.4f} | ROC-AUC: {rec['CV ROC-AUC']:.4f}")

        self.cv_results_df = pd.DataFrame(benchmark_records)
        return self.cv_results_df

    def generate_automated_insights(self):
        """Synthesizes high-level business intelligence insights."""
        print("\n--- Task 6: Generating Automated Strategic Insights ---")
        
        # 1. Catalog Growth and Shift
        yearly_type = self.df[self.df['year_added'] >= 2012].groupby(['year_added', 'type']).size().unstack(fill_value=0)
        recent_year = yearly_type.index.max()
        tv_share_recent = (yearly_type.loc[recent_year, 'TV Show'] / yearly_type.loc[recent_year].sum()) * 100

        # 2. Country Leaders
        top_countries = self.df[self.df['primary_country'] != 'Unknown']['primary_country'].value_counts().head(5)

        # 3. Dominant Archetypes
        top_archetype = self.df['archetype_name'].value_counts().idxmax()
        top_arch_pct = (self.df['archetype_name'].value_counts().max() / len(self.df)) * 100

        # 4. Content Freshness (Lag between release and platform addition)
        median_lag = self.df['addition_lag_years'].median()

        insights = {
            "total_titles_analyzed": len(self.df),
            "catalog_composition": {
                "movies_count": int((self.df['type'] == 'Movie').sum()),
                "tv_shows_count": int((self.df['type'] == 'TV Show').sum()),
                "movies_pct": round(float((self.df['type'] == 'Movie').mean() * 100), 2),
                "tv_shows_pct": round(float((self.df['type'] == 'TV Show').mean() * 100), 2)
            },
            "recent_trends": {
                "recent_year_evaluated": int(recent_year),
                "tv_series_share_recent_pct": round(float(tv_share_recent), 2),
                "median_catalog_addition_lag_years": float(median_lag),
                "streaming_boom_titles_pct": round(float((self.df['release_era'] == 'Streaming Boom (2015-2021)').mean() * 100), 2)
            },
            "geographic_distribution": {
                "top_5_production_nations": top_countries.to_dict()
            },
            "segmentation_summary": {
                "largest_archetype": top_archetype,
                "largest_archetype_catalog_share_pct": round(float(top_arch_pct), 2),
                "total_clusters_formed": len(self.archetype_map)
            }
        }

        print(json.dumps(insights, indent=2))
        self.insights = insights
        return insights

    def save_reports_and_dashboard(self, screenshots_dir='screenshots', reports_dir='reports'):
        """Builds and exports executive visual dashboard and JSON summary report."""
        os.makedirs(screenshots_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        # 1. Save JSON Report
        json_path = os.path.join(reports_dir, 'task6_executive_summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.insights, f, indent=4)
        print(f"[Task 6] Saved executive insights JSON to {json_path}")

        # 2. Executive Analytics Dashboard (4-panel figure)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        sns.set_style("whitegrid")

        # Panel A: Catalog Growth Over Time (2012-2021)
        yearly_type = self.df[(self.df['year_added'] >= 2012) & (self.df['year_added'] <= 2021)].groupby(['year_added', 'type']).size().unstack(fill_value=0)
        yearly_type.plot(kind='bar', stacked=True, color=['#E50914', '#221F1F'], ax=axes[0, 0], width=0.7)
        axes[0, 0].set_title('A: Netflix Catalog Additions by Year (Movies vs TV Shows)', fontsize=12, fontweight='bold', pad=10)
        axes[0, 0].set_xlabel('Year Added', fontweight='bold')
        axes[0, 0].set_ylabel('Number of Titles', fontweight='bold')
        axes[0, 0].legend(['Movie', 'TV Show'])
        axes[0, 0].tick_params(axis='x', rotation=30)

        # Panel B: Top 10 Content Producing Countries
        top_countries = self.df[self.df['primary_country'] != 'Unknown']['primary_country'].value_counts().head(10)
        sns.barplot(x=top_countries.values, y=top_countries.index, palette='rocket', hue=top_countries.index, legend=False, ax=axes[0, 1])
        axes[0, 1].set_title('B: Top 10 Production Hubs Globally', fontsize=12, fontweight='bold', pad=10)
        axes[0, 1].set_xlabel('Total Titles in Catalog', fontweight='bold')
        axes[0, 1].set_ylabel('Country', fontweight='bold')

        # Panel C: Content Rating Distribution (Target Audience Demographics)
        rating_counts = self.df['rating'].value_counts().head(7)
        axes[1, 0].pie(
            rating_counts.values,
            labels=rating_counts.index,
            autopct='%1.1f%%',
            startangle=140,
            colors=sns.color_palette('pastel', len(rating_counts))
        )
        axes[1, 0].set_title('C: Target Maturity Rating Share', fontsize=12, fontweight='bold', pad=10)

        # Panel D: Archetype Share (Task 4 Segmentation Integration)
        arch_counts = self.df['archetype_name'].value_counts()
        sns.barplot(x=arch_counts.values, y=arch_counts.index, palette='mako', hue=arch_counts.index, legend=False, ax=axes[1, 1])
        axes[1, 1].set_title('D: Content Segmentation Archetype Distribution', fontsize=12, fontweight='bold', pad=10)
        axes[1, 1].set_xlabel('Total Titles', fontweight='bold')
        axes[1, 1].set_ylabel('Discovered Archetype', fontweight='bold')

        plt.suptitle('Task 6: Netflix Executive Content Success Analytics Dashboard', fontsize=16, fontweight='bold', y=0.99)
        plt.tight_layout()
        dashboard_path = os.path.join(screenshots_dir, 'task6_executive_dashboard.png')
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Task 6] Saved executive dashboard to {dashboard_path}")

        # 3. Longitudinal Trends Figure
        plt.figure(figsize=(10, 5))
        trend_df = self.df[(self.df['year_added'] >= 2010) & (self.df['year_added'] <= 2021)].groupby(['year_added', 'type']).size().unstack(fill_value=0)
        trend_df_pct = trend_df.div(trend_df.sum(axis=1), axis=0) * 100
        plt.plot(trend_df_pct.index, trend_df_pct['Movie'], marker='o', lw=2.5, color='#E50914', label='Movies (%)')
        plt.plot(trend_df_pct.index, trend_df_pct['TV Show'], marker='s', lw=2.5, color='#1f77b4', label='TV Shows (%)')
        plt.title('Task 6: Structural Shift in Netflix Library: Movies vs TV Shows Proportion', fontsize=13, fontweight='bold', pad=12)
        plt.xlabel('Year Added to Platform', fontweight='bold')
        plt.ylabel('Share of New Additions (%)', fontweight='bold')
        plt.ylim(0, 100)
        plt.legend(loc='center left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        trends_path = os.path.join(screenshots_dir, 'task6_content_growth_trends.png')
        plt.savefig(trends_path, dpi=300)
        plt.close()
        print(f"[Task 6] Saved longitudinal trend chart to {trends_path}")


def main():
    engine = NetflixAnalyticsEngine()
    engine.benchmark_predictive_models()
    engine.generate_automated_insights()
    engine.save_reports_and_dashboard()


if __name__ == '__main__':
    main()
