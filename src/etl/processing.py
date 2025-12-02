import pandas as pd
from src.utils.config import PROCESSED_DATA_DIR

def clean_papers(df):
    """
    Applies cleaning logic:
    - Drop missing IDs or titles
    - Normalize venues
    - Handle missing abstracts (fallback to title)
    """
    # TODO: Implement cleaning logic
    return df

def normalize_authors(df):
    """
    Heuristics for author name normalization.
    """
    # TODO: Implement author normalization (lowercase, strip accents)
    return df

def save_to_parquet(df, filename):
    """
    Saves DataFrame to Parquet in the processed directory.
    """
    path = f"{PROCESSED_DATA_DIR}/{filename}"
    df.to_parquet(path, index=False)
    print(f"Saved to {path}")
