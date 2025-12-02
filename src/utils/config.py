import os

# Project Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# File Paths
PAPERS_PARQUET = os.path.join(PROCESSED_DATA_DIR, 'papers.parquet')
AUTHORS_PARQUET = os.path.join(PROCESSED_DATA_DIR, 'authorships.parquet')
CITATIONS_PARQUET = os.path.join(PROCESSED_DATA_DIR, 'citations.parquet')

# Configuration
RANDOM_SEED = 42
MAX_FEATURES_TFIDF = 10000
