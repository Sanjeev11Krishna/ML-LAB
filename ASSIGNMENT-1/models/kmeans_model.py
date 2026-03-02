from sklearn.cluster import KMeans
import joblib
import numpy as np

def train_kmeans(X_train, y_train):
    # Use a sample for KMeans to speed up training
    from sklearn.utils import resample
    X_sample, y_sample = resample(X_train, y_train, n_samples=min(10000, len(X_train)), random_state=42)
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X_sample)

    cluster_means = {}
    for i in range(5):
        cluster_means[i] = y_sample[clusters == i].mean()

    joblib.dump((kmeans, cluster_means), "models/kmeans_model.pkl")
    return kmeans

def load_kmeans():
    return joblib.load("models/kmeans_model.pkl")

def predict_kmeans(input_data):
    kmeans, cluster_means = load_kmeans()
    cluster = kmeans.predict(input_data)[0]
    return cluster_means[cluster]