import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tqdm import tqdm

def train_topic_model(tfidf_matrix, n_topics=10, algorithm='lda', random_state=42, verbose=True):
    """
    Trains a topic model (LDA or NMF) with progress tracking.
    
    Args:
        tfidf_matrix: TF-IDF sparse matrix
        n_topics (int): Number of topics to extract
        algorithm (str): 'lda' or 'nmf'
        random_state (int): Random seed for reproducibility
        verbose (bool): Show training progress
        
    Returns:
        Trained topic model
    """
    if algorithm == 'lda':
        max_iter = 10
        model = LatentDirichletAllocation(
            n_components=n_topics, 
            random_state=random_state,
            max_iter=max_iter,
            learning_method='online',  # Online is faster than batch
            batch_size=512,  # Process in batches
            verbose=0,  # We'll use tqdm instead
            n_jobs=-1  # Use all CPU cores
        )
        
        if verbose:
            print(f"Training LDA with {n_topics} topics, {max_iter} iterations...")
            with tqdm(total=max_iter, desc="LDA iterations", unit="iter") as pbar:
                # Monkey-patch to track progress
                original_transform = model._e_step
                def tracked_e_step(*args, **kwargs):
                    result = original_transform(*args, **kwargs)
                    pbar.update(1)
                    return result
                model._e_step = tracked_e_step
                model.fit(tfidf_matrix)
        else:
            model.fit(tfidf_matrix)
    else:
        max_iter = 200
        model = NMF(
            n_components=n_topics, 
            random_state=random_state,
            max_iter=max_iter,
            init='nndsvda',
            verbose=0
        )
        
        if verbose:
            print(f"Training NMF with {n_topics} topics, max {max_iter} iterations...")
            with tqdm(total=max_iter, desc="NMF iterations", unit="iter") as pbar:
                # NMF doesn't have easy hooks, just fit with sklearn's verbose
                model.verbose = 1
                model.fit(tfidf_matrix)
                pbar.update(max_iter)
        else:
            model.fit(tfidf_matrix)
    
    return model

def get_top_topics(model, vectorizer, n_words=10):
    """
    Extract top words for each topic from a trained topic model.
    
    Args:
        model: Trained LDA or NMF model
        vectorizer: Fitted TfidfVectorizer
        n_words (int): Number of top words per topic
        
    Returns:
        dict: Topic number -> list of top words
    """
    feature_names = vectorizer.get_feature_names_out()
    topics_dict = {}
    
    for topic_idx, topic in enumerate(model.components_):
        # Get indices of top words
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topics_dict[topic_idx] = top_words
    
    return topics_dict

def display_topics(topics_dict, n_words=10):
    """
    Display topics in a readable format.
    
    Args:
        topics_dict: Dictionary from get_top_topics()
        n_words (int): Number of words to display per topic
    """
    for topic_idx, words in topics_dict.items():
        print(f"Topic {topic_idx}: {', '.join(words[:n_words])}")

def compute_coherence(model, tfidf_matrix, vectorizer):
    """
    Compute topic coherence score (simplified UMass coherence).
    
    Args:
        model: Trained topic model
        tfidf_matrix: TF-IDF sparse matrix
        vectorizer: Fitted TfidfVectorizer
        
    Returns:
        float: Average coherence score
    """
    # Get document-topic distributions
    doc_topic_dist = model.transform(tfidf_matrix)
    
    # Compute perplexity for LDA (lower is better)
    if hasattr(model, 'perplexity'):
        perplexity = model.perplexity(tfidf_matrix)
        return -perplexity  # Negative so higher is better
    
    # For NMF, compute reconstruction error
    reconstruction = np.dot(doc_topic_dist, model.components_)
    error = np.linalg.norm(tfidf_matrix.toarray() - reconstruction, 'fro')
    return -error  # Negative so higher is better

def get_document_topics(model, tfidf_matrix, top_n=1):
    """
    Get the top N topics for each document.
    
    Args:
        model: Trained topic model
        tfidf_matrix: TF-IDF sparse matrix
        top_n (int): Number of top topics to return per document
        
    Returns:
        np.ndarray: Document-topic assignments
    """
    doc_topic_dist = model.transform(tfidf_matrix)
    
    if top_n == 1:
        # Return dominant topic for each document
        return doc_topic_dist.argmax(axis=1)
    else:
        # Return top N topics for each document
        return np.argsort(doc_topic_dist, axis=1)[:, -top_n:][:, ::-1]

def train_citation_predictor(X, y):
    """
    Trains a classifier to predict citation impact.
    Splits data based on time (pre-2010 train, post-test) ideally, 
    but here takes X, y directly.
    """
    # TODO: Implement model training (LogReg, XGBoost, RF)
    # model = xgb.XGBClassifier()
    # model.fit(X_train, y_train)
    pass
