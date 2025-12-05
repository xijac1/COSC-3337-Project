import re
import unicodedata
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from src.utils.config import MAX_FEATURES_TFIDF

def clean_text(text):
    """
    Clean and normalize text for NLP processing.
    
    Steps:
    - Convert to lowercase
    - Remove URLs, emails, special characters
    - Remove extra whitespace
    - Remove Unicode accents
    - Keep only alphanumeric characters and spaces
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str) or not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits (keep letters and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove Unicode accents
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def generate_tfidf_features(text_series):
    """
    Generates TF-IDF features from a Series of text (titles/abstracts).
    """
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES_TFIDF,
        stop_words='english',
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(text_series)
    return tfidf_matrix, vectorizer

def reduce_dimensions(tfidf_matrix, n_components=100):
    """
    Applies PCA to reduce dimensionality of TF-IDF vectors.
    """
    pca = PCA(n_components=n_components)
    reduced_matrix = pca.fit_transform(tfidf_matrix.toarray())
    return reduced_matrix
