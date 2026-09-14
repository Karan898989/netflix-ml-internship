"""
Task 1: Netflix Content Recommendation System
Uses Natural Language Processing (TF-IDF Vectorization) and Cosine Similarity
to deliver content-based recommendations based on genres, cast, director, and synopsis.
"""

import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.data_loader import clean_netflix_data, build_content_soup


class NetflixRecommender:
    """Content-Based Recommendation Engine for Netflix catalog."""

    def __init__(self, df=None):
        if df is None:
            raw_df = clean_netflix_data()
            self.df = build_content_soup(raw_df)
        else:
            self.df = df
        
        self.indices = pd.Series(self.df.index, index=self.df['title'].str.lower()).drop_duplicates()
        self.tfidf = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self._fit_engine()

    def _fit_engine(self):
        """Fits TF-IDF vectorizer and calculates cosine similarity matrix."""
        print("[Task 1] Building TF-IDF matrix from content soup...")
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_features=10000,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['content_soup'])
        print(f"[Task 1] TF-IDF shape: {self.tfidf_matrix.shape}")
        
        print("[Task 1] Computing Cosine Similarity matrix...")
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        print(f"[Task 1] Similarity matrix shape: {self.cosine_sim.shape}")

    def get_recommendations(self, title, top_n=10, filter_type=None):
        """
        Returns top_n recommended titles for a given movie/show.
        
        Parameters:
        - title (str): Title of the movie or show.
        - top_n (int): Number of recommendations to return.
        - filter_type (str, optional): 'Movie' or 'TV Show' to filter output.
        """
        title_clean = str(title).strip().lower()
        if title_clean not in self.indices:
            # Fuzzy fallback: check if title contains substring
            matches = [t for t in self.indices.index if title_clean in t]
            if matches:
                title_clean = matches[0]
            else:
                return pd.DataFrame()

        idx = self.indices[title_clean]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Exclude itself
        sim_scores = [s for s in sim_scores if s[0] != idx]

        recommended_indices = []
        scores = []
        for i, score in sim_scores:
            row_type = self.df.iloc[i]['type']
            if filter_type is None or row_type == filter_type:
                recommended_indices.append(i)
                scores.append(round(score * 100, 2))
            if len(recommended_indices) == top_n:
                break

        res = self.df.iloc[recommended_indices][['title', 'type', 'listed_in', 'director', 'release_year', 'description']].copy()
        res.insert(1, 'match_score_pct', scores)
        return res

    def evaluate_sample_benchmarks(self, sample_titles=None, output_dir='screenshots'):
        """Runs qualitative evaluation on sample titles and saves visual artifacts."""
        os.makedirs(output_dir, exist_ok=True)
        if sample_titles is None:
            sample_titles = [
                'Stranger Things',
                'Inception',
                'Breaking Bad',
                'The Crown',
                'Narcos',
                'Dark'
            ]

        print("\n--- Task 1: Sample Recommendation Benchmarks ---")
        evaluation_records = []
        for title in sample_titles:
            recs = self.get_recommendations(title, top_n=5)
            if not recs.empty:
                print(f"\nTarget Title: '{title}'")
                for i, row in recs.reset_index().iterrows():
                    print(f"  {i+1}. [{row['type']}] {row['title']} (Match: {row['match_score_pct']}%) - Genres: {row['listed_in']}")
                    evaluation_records.append({
                        'query_title': title,
                        'rank': i + 1,
                        'recommended_title': row['title'],
                        'type': row['type'],
                        'match_score': row['match_score_pct'],
                        'genres': row['listed_in']
                    })

        # Save visual comparison chart
        self._plot_similarity_heatmap(sample_titles, output_dir)
        self._plot_sample_recommendations(evaluation_records, output_dir)
        return pd.DataFrame(evaluation_records)

    def _plot_similarity_heatmap(self, sample_titles, output_dir):
        """Generates and saves a heatmap of pairwise similarities between sample titles."""
        valid_titles = []
        valid_indices = []
        for t in sample_titles:
            t_low = t.lower()
            if t_low in self.indices:
                idx = self.indices[t_low]
                if isinstance(idx, pd.Series):
                    idx = idx.iloc[0]
                valid_titles.append(t)
                valid_indices.append(idx)

        if len(valid_indices) < 2:
            return

        sub_matrix = self.cosine_sim[np.ix_(valid_indices, valid_indices)]

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            sub_matrix,
            annot=True,
            fmt=".3f",
            cmap="mako",
            xticklabels=valid_titles,
            yticklabels=valid_titles,
            cbar_kws={'label': 'Cosine Similarity Score'}
        )
        plt.title('Task 1: Semantic Content Similarity Between Benchmark Titles', fontsize=12, fontweight='bold', pad=12)
        plt.xticks(rotation=30, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'task1_similarity_matrix.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Task 1] Saved similarity heatmap to {save_path}")

    def _plot_sample_recommendations(self, records, output_dir):
        """Visual bar chart of top recommendations match scores for target query."""
        if not records:
            return
        df_recs = pd.DataFrame(records)
        top_query = records[0]['query_title']
        subset = df_recs[df_recs['query_title'] == top_query]

        plt.figure(figsize=(9, 5))
        bars = plt.barh(subset['recommended_title'], subset['match_score'], color='#E50914')
        plt.xlabel('Match Score (%)', fontweight='bold')
        plt.ylabel('Recommended Title', fontweight='bold')
        plt.title(f"Task 1: Top Recommendations for '{top_query}'", fontsize=13, fontweight='bold', pad=12)
        plt.xlim(0, 100)
        plt.gca().invert_yaxis()

        for bar in bars:
            w = bar.get_width()
            plt.text(w + 1, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va='center', fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(output_dir, 'task1_sample_recommendations.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Task 1] Saved sample recommendation chart to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Netflix Content Recommendation Engine")
    parser.add_argument('--title', type=str, default='Stranger Things', help='Title to get recommendations for')
    parser.add_argument('--top_n', type=int, default=5, help='Number of recommendations')
    args = parser.parse_args()

    recommender = NetflixRecommender()
    recs = recommender.get_recommendations(args.title, top_n=args.top_n)
    print(f"\nRecommendations for '{args.title}':")
    print(recs[['title', 'match_score_pct', 'type', 'listed_in']])
    recommender.evaluate_sample_benchmarks()


if __name__ == '__main__':
    main()
