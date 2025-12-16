import kagglehub
import pandas as pd
import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet,stopwords
import re
from gensim.models import Word2Vec
import gensim
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import json

nltk.download("punkt_tab")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

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

def vectorization(df):
    vector_list = []
    for token in df["CleanResume"]:
        word_vector = []
        for word in token:
            if word in model.wv.key_to_index:
                word_vector.append(model.wv[word])
        if len(word_vector) > 0:
            vector_list.append(np.mean(word_vector,axis = 0))
        else:
            vector_list.append(np.zeros(model.vector_size))
            
    document_list = np.array(vector_list)
    return document_list

def rarity_score(tokens,word_idf):
    raw_score = 0
    for word in tokens:
        if word in word_idf:
            raw_score += word_idf[word]
    return raw_score

def whitespace_tokenizer(text):
    return text.split()

MODEL_DIR = "./resume_parser_models"  
if __name__ == "__main__":
    path = kagglehub.dataset_download("gauravduttakiit/resume-dataset")
    file_path = os.path.join(path,"UpdatedResumeDataSet.csv")
    
    try:
        df = pd.read_csv(file_path)
        
        # cleaning data and create new column
        df["CleanCategory"] = df["Category"].apply(preprocess_data)
        df["CleanResume"] = df["Resume"].apply(preprocess_data)
        
        # skip-gram model
        model = gensim.models.Word2Vec(df["CleanResume"],min_count=1, vector_size=100,window=2,sg=1)
        vectors = vectorization(df)
        
        # fit model
        k = df["Category"].nunique()
        kmeans = KMeans(n_clusters=k, init="k-means++",random_state=0,n_init=10)
        kmeans.fit(vectors)
        y_kmeans = kmeans.predict(vectors)
        
        # cluster mapping
        df["ClusterLabel"] = y_kmeans
        cluster_analysis = pd.crosstab(df["ClusterLabel"],df["Category"])
        print("Cluster Analysis \n")
        print(cluster_analysis)
        
        # find the percentage dominance
        cluster_percentage = pd.crosstab(df["ClusterLabel"],df["Category"], normalize="index")*100
        print("Cluster Percentage \n")
        print(cluster_percentage.round(1))
        
        cluster_mapping = cluster_percentage.idxmax(axis=1).to_dict()
        
        os.makedirs(MODEL_DIR,exist_ok=True)
        mapping_path = os.path.join(MODEL_DIR, "cluster_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(cluster_mapping, f)
        print(f"Saved cluster mapping to {mapping_path}")
        
        # scoring
        cluster_center = kmeans.cluster_centers_
        labels = df["ClusterLabel"].to_numpy()
        resume_centroids = cluster_center[labels]
        combined = np.hstack([vectors,resume_centroids])
        distance = np.linalg.norm(vectors - resume_centroids, axis=1)
        min_dist = distance.min()
        max_dist = distance.max()
        normalized_dist = (distance - min_dist) / (max_dist - min_dist)
        
        base_score = 1 - normalized_dist
        
        # calculate the IDF weights
        corpus = [" ".join(tokens) for tokens in df["CleanResume"]]
        tfidf = TfidfVectorizer(stop_words=None, tokenizer=whitespace_tokenizer)
        tfidf.fit(corpus)
        word_idf = dict(zip(tfidf.get_feature_names_out(),tfidf.idf_))
        
        # calculate the resume rarity score
        raw_bonus_score = df["CleanResume"].apply(lambda x: rarity_score(x,word_idf))
        min_bonus = raw_bonus_score.min()
        max_bonus = raw_bonus_score.max()
        normalized_bonus_score =  (raw_bonus_score - min_bonus) / (max_bonus - min_bonus)
        
        # scoring combination and scaling
        raw_score = base_score + normalized_bonus_score
        final_score = (raw_score / 1.5) * 10
        df["FinalScore"] = final_score
        
        # save model
        os.makedirs(MODEL_DIR,exist_ok=True)
        
        model_path_w2v = os.path.join(MODEL_DIR,"word2vec_model.bin")
        model.save(model_path_w2v)
        
        model_path_kmeans = os.path.join(MODEL_DIR,"kmeans_model.joblib")
        joblib.dump(kmeans,model_path_kmeans)
        
        model_path_tfidf = os.path.join(MODEL_DIR,"tfidf_vectorizer.joblib")
        joblib.dump(tfidf,model_path_tfidf)
        
        scaling_params = {
            "min_dist": float(min_dist),
            "max_dist": float(max_dist),
            "min_bonus": float(min_bonus),
            "max_bonus": float(max_bonus),
            "max_raw_score": 1.5
        }
        params_path = os.path.join(MODEL_DIR,"Scaling_params.json")
        with open(params_path,"w") as f:
            json.dump(scaling_params,f)
    except FileNotFoundError:
        print("File Not Found")
    except Exception as e:
        print(f"Error: {e}")


