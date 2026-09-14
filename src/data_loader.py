"""
Data Loader and Preprocessing Module for Netflix ML Internship Tasks.
Provides centralized loading, missing value imputation, text cleaning,
and feature engineering functions.
"""

import os
import re
import pandas as pd
import numpy as np


def get_default_data_path():
    """Returns the default path to the netflix_titles.csv dataset."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'data', 'netflix_titles.csv')


def load_raw_data(filepath=None):
    """Loads raw Netflix dataset from CSV."""
    if filepath is None:
        filepath = get_default_data_path()
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    return df


def clean_netflix_data(df=None, filepath=None):
    """
    Cleans Netflix dataset:
    - Resolves known column shift where duration is stored in rating for select records
    - Imputes missing values for director, cast, country, and rating
    - Parses date_added into datetime and extracts year_added, month_added
    - Parses duration into duration_numeric (minutes for movies, seasons for TV shows)
    - Normalizes genres and primary production countries
    """
    if df is None:
        df = load_raw_data(filepath)
    df = df.copy()

    # 1. Fix Kaggle dataset misalignment where rating contains 'min'
    shifted_mask = df['rating'].astype(str).str.contains('min', na=False)
    if shifted_mask.any():
        df.loc[shifted_mask & df['duration'].isna(), 'duration'] = df.loc[shifted_mask, 'rating']
        df.loc[shifted_mask, 'rating'] = 'NR'

    # 2. Impute missing values
    df['director'] = df['director'].fillna('Unknown').astype(str).str.strip()
    df['cast'] = df['cast'].fillna('Unknown').astype(str).str.strip()
    df['country'] = df['country'].fillna('Unknown').astype(str).str.strip()
    df['rating'] = df['rating'].fillna('TV-MA').astype(str).str.strip()
    df['description'] = df['description'].fillna('').astype(str).str.strip()
    df['listed_in'] = df['listed_in'].fillna('').astype(str).str.strip()

    # 3. Parse date_added
    df['date_added_clean'] = df['date_added'].astype(str).str.strip()
    df['date_added_dt'] = pd.to_datetime(df['date_added_clean'], format='%B %d, %Y', errors='coerce')
    
    # Fill missing dates with release_year-01-01
    missing_dates = df['date_added_dt'].isna()
    df.loc[missing_dates, 'date_added_dt'] = pd.to_datetime(
        df.loc[missing_dates, 'release_year'].astype(str) + '-01-01', errors='coerce'
    )
    df['year_added'] = df['date_added_dt'].dt.year.fillna(df['release_year']).astype(int)
    df['month_added'] = df['date_added_dt'].dt.month.fillna(1).astype(int)

    # 4. Parse primary country (first listed country if comma-separated)
    df['primary_country'] = df['country'].apply(
        lambda c: c.split(',')[0].strip() if c != 'Unknown' else 'Unknown'
    )

    # 5. Extract duration metrics
    def parse_duration_min(row):
        val = str(row['duration'])
        if 'min' in val:
            try:
                return float(re.findall(r'\d+', val)[0])
            except (IndexError, ValueError):
                return np.nan
        return np.nan

    def parse_seasons(row):
        val = str(row['duration'])
        if 'Season' in val:
            try:
                return float(re.findall(r'\d+', val)[0])
            except (IndexError, ValueError):
                return np.nan
        return np.nan

    df['duration_min'] = df.apply(parse_duration_min, axis=1)
    df['seasons'] = df.apply(parse_seasons, axis=1)

    # Median duration imputation for movies
    movie_median_min = df.loc[df['type'] == 'Movie', 'duration_min'].median()
    df.loc[(df['type'] == 'Movie') & (df['duration_min'].isna()), 'duration_min'] = movie_median_min
    df['seasons'] = df['seasons'].fillna(0)

    # 6. Parse genre list
    df['genres_list'] = df['listed_in'].apply(lambda g: [x.strip() for x in g.split(',') if x.strip()])

    return df


def build_content_soup(df):
    """
    Creates an enriched textual representation ('content soup') for recommendation.
    Combines:
    - Cleaned genres (weighted x2)
    - Director name (cleaned to treat as single entity)
    - Top 3 Cast members (cleaned to treat as single entities)
    - Cleaned synopsis description
    """
    def sanitize(text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', str(text)).lower()

    def clean_person_names(text, max_items=3):
        if not text or text == 'Unknown':
            return ''
        names = [x.strip() for x in text.split(',')[:max_items]]
        # Remove spaces so "Tom Hanks" becomes "tomhanks"
        return ' '.join([re.sub(r'\s+', '', n).lower() for n in names if n])

    soups = []
    for _, row in df.iterrows():
        # Genres with weight (repeated twice)
        genres_clean = ' '.join([re.sub(r'\s+', '', g).lower() for g in row['genres_list']])
        genres_weighted = f"{genres_clean} {genres_clean}"

        # Director
        director_clean = clean_person_names(row['director'], max_items=1)

        # Top Cast
        cast_clean = clean_person_names(row['cast'], max_items=3)

        # Clean description
        desc_clean = sanitize(row['description'])

        soup = f"{genres_weighted} {director_clean} {cast_clean} {desc_clean}".strip()
        soups.append(soup)

    df_copy = df.copy()
    df_copy['content_soup'] = soups
    return df_copy
