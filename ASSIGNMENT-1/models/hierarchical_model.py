from sklearn.cluster import AgglomerativeClustering
import joblib
import numpy as np

def train_hierarchical(X_train, y_train):
    # Use a sample for hierarchical clustering to avoid memory issues
    from sklearn.utils import resample
    X_sample, y_sample = resample(X_train, y_train, n_samples=min(1000, len(X_train)), random_state=42)
    
    model = AgglomerativeClustering(n_clusters=5)
    clusters = model.fit_predict(X_sample)

    cluster_means = {}
    for i in range(5):
        cluster_means[i] = y_sample[clusters == i].mean()

    joblib.dump(cluster_means, "models/hierarchical_model.pkl")
    return cluster_means

def load_hierarchical():
    return joblib.load("models/hierarchical_model.pkl")

def predict_hierarchical(input_data):
    cluster_means = load_hierarchical()
    # Simple nearest cluster approach
    return np.mean(list(cluster_means.values()))