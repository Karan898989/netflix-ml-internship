"""
Task 4: Netflix Content Segmentation
Unsupervised clustering using K-Means to discover natural content segments and archetypes.
Evaluates optimal k using Elbow method and Silhouette scores.
Projects high-dimensional space into 2D using Principal Component Analysis (PCA).
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

from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.data_loader import clean_netflix_data


def prepare_clustering_features(df):
    """
    Creates scaled high-dimensional feature representations for clustering:
    - Multi-hot genres
    - Content type (Movie / TV Show)
    - Rating category groupings
    - Primary country groups
    - Scaled release year and normalized duration
    - TF-IDF thematic keywords from synopses
    """
    print("[Task 4] Preparing segmentation feature vectors...")
    df = df.copy()

    # 1. Multi-hot genre encoding
    mlb = MultiLabelBinarizer()
    genre_mat = mlb.fit_transform(df['genres_list'])
    genre_cols = [f"g_{g.replace(' ', '_').lower()}" for g in mlb.classes_]
    df_genres = pd.DataFrame(genre_mat, columns=genre_cols, index=df.index)

    # 2. Content Type binary
    df_type = pd.DataFrame({'is_movie': (df['type'] == 'Movie').astype(int)}, index=df.index)

    # 3. Rating groupings
    def group_rating(r):
        r = str(r).upper()
        if r in ['TV-Y', 'TV-Y7', 'G', 'TV-G', 'PG', 'TV-PG']:
            return 'Kids_Family'
        elif r in ['PG-13', 'TV-14']:
            return 'Teens_General'
        elif r in ['R', 'TV-MA', 'NC-17']:
            return 'Mature_Adult'
        return 'Other_Rating'

    rating_group = df['rating'].apply(group_rating)
    df_rating_grp = pd.get_dummies(rating_group, prefix='age', dtype=int)

    # 4. Top Countries
    top_c = df['primary_country'].value_counts().head(8).index.tolist()
    c_series = df['primary_country'].apply(lambda x: x if x in top_c else 'Other_Country')
    df_country = pd.get_dummies(c_series, prefix='c', dtype=int)

    # 5. Scaled numeric features
    scaler = StandardScaler()
    scaled_nums = scaler.fit_transform(df[['release_year', 'duration_min', 'seasons']].fillna(0))
    df_nums = pd.DataFrame(scaled_nums, columns=['num_year', 'num_duration', 'num_seasons'], index=df.index)

    # 6. TF-IDF synopsis features (top 80 keywords)
    tfidf = TfidfVectorizer(max_features=80, stop_words='english')
    tfidf_mat = tfidf.fit_transform(df['description']).toarray()
    tfidf_cols = [f"txt_{w}" for w in tfidf.get_feature_names_out()]
    df_text = pd.DataFrame(tfidf_mat, columns=tfidf_cols, index=df.index)

    # Combine all feature spaces
    X = pd.concat([df_genres, df_type, df_rating_grp, df_country, df_nums, df_text], axis=1)
    print(f"[Task 4] Segmentation feature space built: {X.shape[0]} titles, {X.shape[1]} features.")
    return X


class ContentSegmenter:
    """Manages clustering, optimal k search, PCA projection, and cluster profiling."""

    def __init__(self, df=None, optimal_k=5):
        if df is None:
            df = clean_netflix_data()
        self.df = df.copy()
        self.X = prepare_clustering_features(self.df)
        self.optimal_k = optimal_k
        self.kmeans = None
        self.cluster_labels = None
        self.pca_2d = None
        self.elbow_results = {}

    def find_optimal_k(self, k_range=range(2, 11), sample_size=3000):
        """Calculates inertia and silhouette scores across a range of k."""
        print("\n--- Task 4: Evaluating Optimal Cluster Count (k) ---")
        inertias = []
        silhouettes = []
        
        # Subsample for faster silhouette evaluation if dataset is large
        if len(self.X) > sample_size:
            sample_indices = np.random.RandomState(42).choice(len(self.X), sample_size, replace=False)
            X_eval = self.X.iloc[sample_indices]
        else:
            X_eval = self.X

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_eval)
            inertias.append(km.inertia_)
            sil = silhouette_score(X_eval, labels)
            silhouettes.append(sil)
            print(f"  k={k} -> Inertia: {km.inertia_:.1f} | Silhouette Score: {sil:.4f}")

        self.elbow_results = {
            'k_values': list(k_range),
            'inertias': inertias,
            'silhouettes': silhouettes
        }
        return self.elbow_results

    def fit_clusters(self, k=None):
        """Fits K-Means with chosen k and computes 2D PCA coordinates."""
        if k is not None:
            self.optimal_k = k

        print(f"\n[Task 4] Fitting K-Means with k={self.optimal_k}...")
        self.kmeans = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=15)
        self.cluster_labels = self.kmeans.fit_predict(self.X)
        self.df['cluster'] = self.cluster_labels

        print("[Task 4] Performing PCA dimensionality reduction to 2D...")
        pca = PCA(n_components=2, random_state=42)
        self.pca_2d = pca.fit_transform(self.X)
        self.df['pca_x'] = self.pca_2d[:, 0]
        self.df['pca_y'] = self.pca_2d[:, 1]
        print(f"[Task 4] Explained variance by 2 PCA components: {pca.explained_variance_ratio_.sum()*100:.2f}%")

    @staticmethod
    def _generate_archetype_name(cluster_id, movie_pct, top_genres, top_countries):
        """Auto-generates a descriptive archetype name from actual cluster content."""
        # Determine content type label
        if movie_pct >= 90:
            type_label = "Movies"
        elif movie_pct <= 10:
            type_label = "TV Series"
        else:
            type_label = "Mixed Content"

        # Pick the most descriptive genre (skip overly generic ones)
        generic_genres = {'International Movies', 'International TV Shows', 'TV Shows', 'Movies'}
        descriptive_genres = [g for g in top_genres if g not in generic_genres]
        if descriptive_genres:
            genre_label = " & ".join(descriptive_genres[:2])
        elif top_genres:
            genre_label = " & ".join(top_genres[:2])
        else:
            genre_label = "General"

        # Add regional flavor if top country is not US/Unknown
        country = top_countries[0] if top_countries else "Unknown"
        if country not in ("United States", "Unknown"):
            return f"{country} {genre_label} ({type_label})"
        return f"{genre_label} ({type_label})"

    def profile_clusters(self):
        """Generates detailed business archetype profiles for each cluster."""
        profiles = []

        print("\n--- Task 4: Content Cluster Archetypes ---")
        for c in range(self.optimal_k):
            c_df = self.df[self.df['cluster'] == c]
            count = len(c_df)
            pct = (count / len(self.df)) * 100
            movie_pct = (c_df['type'] == 'Movie').mean() * 100
            tv_pct = 100 - movie_pct

            # Top genres
            all_genres = [g for sublist in c_df['genres_list'] for g in sublist]
            top_genres = pd.Series(all_genres).value_counts().head(4).index.tolist()

            # Top countries
            top_countries = c_df['primary_country'].value_counts().head(3).index.tolist()

            # Examples
            examples = c_df['title'].sample(min(3, count), random_state=42).tolist()

            # Auto-generate archetype name from actual cluster characteristics
            archetype = self._generate_archetype_name(c, movie_pct, top_genres, top_countries)

            print(f"\n[Cluster {c}] {archetype}")
            print(f"  Titles: {count:,} ({pct:.1f}% of catalog) | Movies: {movie_pct:.1f}% vs TV: {tv_pct:.1f}%")
            print(f"  Dominant Genres: {', '.join(top_genres)}")
            print(f"  Primary Countries: {', '.join(top_countries)}")
            print(f"  Sample Titles: {', '.join(examples)}")

            profiles.append({
                'cluster': c,
                'archetype_name': archetype,
                'title_count': count,
                'pct_catalog': round(pct, 2),
                'movie_pct': round(movie_pct, 1),
                'tv_show_pct': round(tv_pct, 1),
                'dominant_genres': '; '.join(top_genres),
                'primary_countries': '; '.join(top_countries),
                'sample_titles': '; '.join(examples)
            })

        self.profiles_df = pd.DataFrame(profiles)
        return self.profiles_df

    def save_reports_and_plots(self, screenshots_dir='screenshots', reports_dir='reports'):
        """Saves visual cluster plots and CSV cluster profiles."""
        os.makedirs(screenshots_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        # 1. Save Profiles CSV
        csv_path = os.path.join(reports_dir, 'task4_cluster_profiles.csv')
        self.profiles_df.to_csv(csv_path, index=False)
        print(f"[Task 4] Saved cluster profiles to {csv_path}")

        # 2. Save Elbow & Silhouette Plot
        if self.elbow_results:
            fig, ax1 = plt.subplots(figsize=(9, 5))
            k_vals = self.elbow_results['k_values']
            color = '#1f77b4'
            ax1.set_xlabel('Number of Clusters (k)', fontweight='bold')
            ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', color=color, fontweight='bold')
            ax1.plot(k_vals, self.elbow_results['inertias'], marker='o', color=color, lw=2)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()
            color = '#ff7f0e'
            ax2.set_ylabel('Silhouette Score', color=color, fontweight='bold')
            ax2.plot(k_vals, self.elbow_results['silhouettes'], marker='s', color=color, lw=2, linestyle='--')
            ax2.tick_params(axis='y', labelcolor=color)

            plt.title('Task 4: Optimal Cluster Selection via Elbow Method & Silhouette Score', fontsize=12, fontweight='bold', pad=12)
            plt.axvline(x=self.optimal_k, color='red', linestyle=':', label=f'Chosen k={self.optimal_k}')
            fig.tight_layout()
            elbow_path = os.path.join(screenshots_dir, 'task4_elbow_silhouette.png')
            plt.savefig(elbow_path, dpi=300)
            plt.close()
            print(f"[Task 4] Saved elbow/silhouette plot to {elbow_path}")

        # 3. Save 2D PCA Cluster Scatter Plot
        plt.figure(figsize=(11, 7))
        palette = sns.color_palette('bright', self.optimal_k)
        for c in range(self.optimal_k):
            sub = self.df[self.df['cluster'] == c]
            arch = self.profiles_df.loc[self.profiles_df['cluster'] == c, 'archetype_name'].values[0]
            plt.scatter(sub['pca_x'], sub['pca_y'], s=16, alpha=0.6, label=f"C{c}: {arch}", color=palette[c])

        plt.title('Task 4: 2D Principal Component Analysis (PCA) of Netflix Content Clusters', fontsize=13, fontweight='bold', pad=12)
        plt.xlabel('Principal Component 1', fontweight='bold')
        plt.ylabel('Principal Component 2', fontweight='bold')
        plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=True, fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        pca_path = os.path.join(screenshots_dir, 'task4_clusters_pca.png')
        plt.savefig(pca_path, dpi=300)
        plt.close()
        print(f"[Task 4] Saved 2D PCA cluster map to {pca_path}")

        # 4. Save Cluster Breakdown Bar Chart
        plt.figure(figsize=(10, 5))
        sns.barplot(data=self.profiles_df, x='archetype_name', y='title_count', hue='archetype_name', palette='crest', legend=False)
        plt.title('Task 4: Content Volume Distribution Across Discovered Archetypes', fontsize=13, fontweight='bold', pad=12)
        plt.xlabel('Content Archetype', fontweight='bold')
        plt.ylabel('Total Titles in Catalog', fontweight='bold')
        plt.xticks(rotation=20, ha='right')
        for i, row in self.profiles_df.iterrows():
            plt.text(i, row['title_count'] + 40, f"{row['title_count']:,}\n({row['pct_catalog']}%)", ha='center', fontsize=9, fontweight='bold')
        plt.tight_layout()
        traits_path = os.path.join(screenshots_dir, 'task4_cluster_traits.png')
        plt.savefig(traits_path, dpi=300)
        plt.close()
        print(f"[Task 4] Saved cluster traits breakdown to {traits_path}")


def main():
    segmenter = ContentSegmenter(optimal_k=5)
    segmenter.find_optimal_k(range(2, 9))
    segmenter.fit_clusters(k=5)
    segmenter.profile_clusters()
    segmenter.save_reports_and_plots()


if __name__ == '__main__':
    main()
