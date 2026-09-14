"""
Master Pipeline Runner for Auspify Machine Learning Internship Project.
Executes all 4 tasks sequentially:
- Task 1: Netflix Content Recommendation System
- Task 2: Content Type Prediction Model
- Task 4: Netflix Content Segmentation
- Task 6: Netflix Content Success Analytics Engine (Capstone)
"""

import os
import sys
import time

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import clean_netflix_data
from src.task1_recommender import NetflixRecommender
from src.task2_classifier import ContentTypeClassifier
from src.task4_segmentation import ContentSegmenter
from src.task6_analytics_engine import NetflixAnalyticsEngine


def run_full_internship_pipeline():
    start_time = time.time()
    print("=" * 70)
    print("AUSPIFY MACHINE LEARNING INTERNSHIP — FULL PIPELINE EXECUTION")
    print("=" * 70)

    # 0. Data Load & Sanity Verification
    print("\n[Step 0] Loading and verifying Netflix dataset...")
    df = clean_netflix_data()
    print(f"-> Successfully loaded {len(df):,} catalog titles across {df['type'].nunique()} types.")

    # 1. Task 1: Recommendation Engine
    print("\n" + "=" * 50)
    print("EXECUTING TASK 1: Content Recommendation System")
    print("=" * 50)
    recommender = NetflixRecommender(df=None)
    eval_df = recommender.evaluate_sample_benchmarks()
    print(f"-> Task 1 Complete: Evaluated {len(eval_df)} recommendations across benchmark titles.")

    # 2. Task 2: Content Type Classification
    print("\n" + "=" * 50)
    print("EXECUTING TASK 2: Content Type Prediction Model")
    print("=" * 50)
    classifier = ContentTypeClassifier(df=df)
    metrics_df = classifier.train_and_evaluate()
    classifier.save_reports_and_plots()
    print("-> Task 2 Complete. Performance Benchmark:")
    print(metrics_df.to_string(index=False))

    # 3. Task 4: Content Segmentation
    print("\n" + "=" * 50)
    print("EXECUTING TASK 4: Netflix Content Segmentation")
    print("=" * 50)
    segmenter = ContentSegmenter(df=df, optimal_k=5)
    segmenter.find_optimal_k(k_range=range(2, 9))
    segmenter.fit_clusters(k=5)
    profiles_df = segmenter.profile_clusters()
    segmenter.save_reports_and_plots()
    print(f"-> Task 4 Complete: Segmented catalog into {len(profiles_df)} distinct archetypes.")

    # 4. Task 6: Capstone Analytics Engine
    print("\n" + "=" * 50)
    print("EXECUTING TASK 6: Content Success Analytics Engine (Capstone)")
    print("=" * 50)
    engine = NetflixAnalyticsEngine(df=df)
    cv_df = engine.benchmark_predictive_models()
    insights = engine.generate_automated_insights()
    engine.save_reports_and_dashboard()
    print("-> Task 6 Complete. Cross-validation Benchmark:")
    print(cv_df.to_string(index=False))

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"ALL 4 TASKS SUCCESSFULLY EXECUTED IN {elapsed:.1f} SECONDS!")
    print("=" * 70)
    print("\nArtifacts Summary:")
    print("Screenshots Directory (screenshots/):")
    for f in sorted(os.listdir('screenshots')):
        size = os.path.getsize(os.path.join('screenshots', f)) / 1024
        print(f"  - screenshots/{f} ({size:.1f} KB)")

    print("\nReports Directory (reports/):")
    for f in sorted(os.listdir('reports')):
        size = os.path.getsize(os.path.join('reports', f)) / 1024
        print(f"  - reports/{f} ({size:.1f} KB)")


if __name__ == '__main__':
    run_full_internship_pipeline()
