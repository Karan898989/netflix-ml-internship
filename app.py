"""
Flask Web Application for Auspify Machine Learning Internship Project.
Provides interactive web interface and RESTful APIs for:
- Content Recommendation Engine
- Live Content Type Classifier
- Segmentation & Archetypes Explorer
- Executive Analytics Dashboard
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import clean_netflix_data, build_content_soup
from src.task1_recommender import NetflixRecommender
from src.task2_classifier import ContentTypeClassifier

app = Flask(__name__)

# Global cache for ML models and catalog data
DATA = {}


def initialize_app_data():
    """Initializes models and datasets on application startup."""
    print("[Flask] Loading catalog data and initializing ML engines...")
    df = clean_netflix_data()
    df_soup = build_content_soup(df)
    
    # Task 1 Recommender
    recommender = NetflixRecommender(df=df_soup)
    
    # Task 2 Classifier
    classifier = ContentTypeClassifier(df=df)
    classifier.train_and_evaluate()
    
    # Task 4 Cluster Profiles
    cluster_profiles_path = os.path.join(PROJECT_ROOT, 'reports', 'task4_cluster_profiles.csv')
    if os.path.exists(cluster_profiles_path):
        cluster_df = pd.read_csv(cluster_profiles_path)
        clusters = cluster_df.to_dict(orient='records')
    else:
        clusters = []

    # Task 6 Executive Summary
    analytics_path = os.path.join(PROJECT_ROOT, 'reports', 'task6_executive_summary.json')
    if os.path.exists(analytics_path):
        with open(analytics_path, 'r', encoding='utf-8') as f:
            analytics = json.load(f)
    else:
        analytics = {}

    # Yearly trends data for dynamic charts
    yearly = df[(df['year_added'] >= 2012) & (df['year_added'] <= 2021)].groupby(['year_added', 'type']).size().unstack(fill_value=0)
    yearly_data = {
        'years': [int(y) for y in yearly.index],
        'movies': [int(m) for m in yearly['Movie']],
        'tv_shows': [int(t) for t in yearly['TV Show']]
    }

    # Top countries
    top_c = df[df['primary_country'] != 'Unknown']['primary_country'].value_counts().head(10)
    country_data = {
        'labels': top_c.index.tolist(),
        'counts': [int(c) for c in top_c.values]
    }

    # Rating shares
    top_ratings = df['rating'].value_counts().head(6)
    rating_data = {
        'labels': top_ratings.index.tolist(),
        'counts': [int(r) for r in top_ratings.values]
    }

    DATA['df'] = df
    DATA['recommender'] = recommender
    DATA['classifier'] = classifier
    DATA['clusters'] = clusters
    DATA['analytics'] = analytics
    DATA['yearly_data'] = yearly_data
    DATA['country_data'] = country_data
    DATA['rating_data'] = rating_data
    DATA['all_titles'] = sorted(df['title'].dropna().unique().tolist())

    print(f"[Flask] ML engines loaded successfully! Catalog has {len(df):,} titles.")


@app.route('/')
def index():
    """Serves the main application homepage."""
    return render_template(
        'index.html',
        total_titles=len(DATA['df']),
        analytics=DATA.get('analytics', {}),
        clusters=DATA.get('clusters', [])
    )


@app.route('/api/titles')
def get_titles():
    """Returns matching titles for frontend search autocomplete."""
    q = request.args.get('q', '').strip().lower()
    if not q:
        # Return popular default titles
        sample_popular = ['Stranger Things', 'Breaking Bad', 'Narcos', 'The Crown', 'Inception', 'Dark', 'Ozark', 'Money Heist', 'Squid Game', 'Black Mirror']
        return jsonify([t for t in sample_popular if t in DATA['all_titles']][:10])
    
    matches = [t for t in DATA['all_titles'] if q in t.lower()][:15]
    return jsonify(matches)


@app.route('/api/recommend')
def recommend():
    """Returns top-N content recommendations for a given title."""
    title = request.args.get('title', '').strip()
    top_n = int(request.args.get('top_n', 6))
    filter_type = request.args.get('type', None)
    if filter_type in ['All', '']:
        filter_type = None

    if not title:
        return jsonify({'error': 'Title parameter is required'}), 400

    recs_df = DATA['recommender'].get_recommendations(title, top_n=top_n, filter_type=filter_type)
    if recs_df.empty:
        return jsonify({'error': f"Title '{title}' not found in catalog.", 'results': []}), 404

    # Convert to records dictionary
    results = recs_df.to_dict(orient='records')
    return jsonify({
        'query_title': title,
        'results_count': len(results),
        'recommendations': results
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Performs live content type prediction for custom metadata inputs."""
    payload = request.get_json() or {}
    genres = payload.get('genres', ['Dramas'])
    country = payload.get('country', 'United States')
    rating = payload.get('rating', 'TV-MA')
    release_year = float(payload.get('release_year', 2021))
    description = payload.get('description', '')

    # Prepare single-row feature dictionary matching training schema exactly
    classifier = DATA['classifier']
    row_dict = {col: 0.0 for col in classifier.feature_names}

    # 1. Multi-label genres
    for g in genres:
        key = f"genre_{g.replace(' ', '_').lower()}"
        if key in row_dict:
            row_dict[key] = 1.0

    # 2. Country one-hot
    country_key = f"country_{country}"
    if country_key in row_dict:
        row_dict[country_key] = 1.0
    elif 'country_Other_Country' in row_dict:
        row_dict['country_Other_Country'] = 1.0

    # 3. Rating one-hot
    rating_key = f"rating_{rating}"
    if rating_key in row_dict:
        row_dict[rating_key] = 1.0
    elif 'rating_Other_Rating' in row_dict:
        row_dict['rating_Other_Rating'] = 1.0

    # 4. Scaled year
    row_dict['scaled_release_year'] = (release_year - 2014.18) / 8.82

    # 5. TF-IDF synopsis words
    desc_words = set(description.lower().split())
    for word in desc_words:
        w_key = f"tfidf_{word}"
        if w_key in row_dict:
            row_dict[w_key] = 1.0

    input_df = pd.DataFrame([row_dict])[classifier.feature_names]

    # Predict with Random Forest
    rf_model = classifier.models['Random Forest']
    probs = rf_model.predict_proba(input_df)[0]
    movie_prob = round(float(probs[1]) * 100, 2)
    tv_prob = round(float(probs[0]) * 100, 2)
    prediction = 'Movie' if movie_prob >= 50.0 else 'TV Show'

    return jsonify({
        'prediction': prediction,
        'movie_probability': movie_prob,
        'tv_show_probability': tv_prob,
        'confidence': max(movie_prob, tv_prob)
    })


@app.route('/api/clusters')
def get_clusters():
    """Returns the 5 discovered archetype profiles."""
    return jsonify(DATA.get('clusters', []))


@app.route('/api/analytics')
def get_analytics():
    """Returns executive analytics metrics and chart data series."""
    return jsonify({
        'summary': DATA.get('analytics', {}),
        'yearly': DATA.get('yearly_data', {}),
        'countries': DATA.get('country_data', {}),
        'ratings': DATA.get('rating_data', {})
    })


@app.route('/screenshots/<path:filename>')
def serve_screenshot(filename):
    """Serves high-resolution plot images."""
    screenshots_dir = os.path.join(PROJECT_ROOT, 'screenshots')
    return send_from_directory(screenshots_dir, filename)


@app.route('/reports/<path:filename>')
def serve_report(filename):
    """Serves generated report documents and CSVs."""
    reports_dir = os.path.join(PROJECT_ROOT, 'reports')
    return send_from_directory(reports_dir, filename)


# Initialize data at module import so Gunicorn / Flask CLI has it ready
initialize_app_data()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n=======================================================")
    print(f"🎬 Netflix ML Web App running at: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host='0.0.0.0', port=port, debug=False)
