from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from src.utils.config import MAX_FEATURES_TFIDF

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
