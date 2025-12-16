import joblib
import json
import os
from gensim.models import Word2Vec
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re

MODEL_DIR = "resume_parser_models"

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_data(text):
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)

    # Remove stopwords & non-alphabetic tokens
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]

    # reduce word into root form
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

def whitespace_tokenizer(text):
    return text.split()

def load_models():
    W2V_PATH = os.path.join(MODEL_DIR, "word2vec_model.bin")
    loaded_w2v_model = Word2Vec.load(W2V_PATH)

    KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans_model.joblib")
    loaded_kmeans_model = joblib.load(KMEANS_PATH)

    TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
    loaded_tfidf_vectorizer = joblib.load(TFIDF_PATH)
    word_idf = dict(zip(loaded_tfidf_vectorizer.get_feature_names_out(), loaded_tfidf_vectorizer.idf_))
    
    PARAMS_PATH = os.path.join(MODEL_DIR, "Scaling_params.json")
    with open(PARAMS_PATH, 'r') as f:
        loaded_scaling_params = json.load(f)
        
    MAPPING_PATH = os.path.join(MODEL_DIR, "cluster_mapping.json")
    with open(MAPPING_PATH, 'r') as f:
        loaded_mapping = json.load(f)

    return loaded_w2v_model,loaded_kmeans_model,word_idf,loaded_scaling_params,loaded_mapping

def score_single_resume(resume_text,w2v_model,kmeans_model,word_idf,params):
    tokens = preprocess_data(resume_text)
    
    word_vectors = []
    for word in tokens:
        if word in w2v_model.wv.key_to_index:
            word_vectors.append(w2v_model.wv[word])
    
    if not word_vectors:
        return 0.0, "Skipped (No meaningful tokens found)"

    doc_vector = np.mean(word_vectors, axis=0).reshape(1, -1)
    
    cluster_label = kmeans_model.predict(doc_vector)[0]
    centroid = kmeans_model.cluster_centers_[cluster_label]
    distance = np.linalg.norm(doc_vector - centroid, axis=1)[0]
    
    normalized_dist = (distance - params['min_dist']) / (params['max_dist'] - params['min_dist'])
    base_score = 1 - normalized_dist
    
    raw_bonus_score = sum(word_idf.get(word, 0) for word in tokens)
    min_bonus = params['min_bonus']
    max_bonus = params['max_bonus']
    normalized_bonus_score = (raw_bonus_score - min_bonus) / (max_bonus - min_bonus) * 0.5
    
    raw_final_score = base_score + normalized_bonus_score
    final_score = (raw_final_score / params['max_raw_score']) * 10
    
    return final_score, cluster_label

REQUIRED_RESOURCES = ['stopwords', 'punkt', 'wordnet', 'omw-1.4']

if __name__ == "__main__":
    for resource in REQUIRED_RESOURCES:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            print(f"NLTK resource '{resource}' not found. Downloading...")
            try:
                nltk.download(resource)
            except Exception as e:
                print(f"Error downloading NLTK resource '{resource}': {e}")
    
    w2v_model, kmeans_model, word_idf, scaling_params, cluster_mapping = load_models()
    test_resume_good = """
    Software Engineer specializing in Python, machine learning, and cloud deployment. 
    Expert in TensorFlow, PyTorch, and using Docker for large-scale production. 
    Proven track record in agile scrum environments and model optimization.
    """
    
    score, cluster = score_single_resume(test_resume_good, w2v_model, kmeans_model, word_idf, scaling_params)
    print("\n--- NEW CANDIDATE SCORING ---")
    print("Candidate 1 (Expert):")
    print(f"Assigned Cluster: {cluster} ({cluster_mapping.get(str(cluster), 'Unknown Category')})")
    print(f"Final Score (0-10): {score:.2f}")
    
    test_resume_bad = """
    Energetic sales professional. Responsible for generating leads and managing client relationships. 
    Highly motivated team player with strong communication skills. Experience with Microsoft Office.
    """
    
    score_bad, cluster_bad = score_single_resume(test_resume_bad, w2v_model, kmeans_model, word_idf, scaling_params)
    
    print("\nCandidate 2 (General):")
    print(f"Assigned Cluster: {cluster_bad} ({cluster_mapping.get(str(cluster_bad), 'Unknown Category')})")
    print(f"Final Score (0-10): {score_bad:.2f}")