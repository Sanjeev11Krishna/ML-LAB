from sklearn.ensemble import RandomForestRegressor
import joblib

def train_random_forest(X_train, y_train):
    from sklearn.utils import resample
    X_sample, y_sample = resample(X_train, y_train, n_samples=min(5000, len(X_train)), random_state=42)
    
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=8,
        random_state=42
    )
    model.fit(X_sample, y_sample)
    joblib.dump(model, "models/rf_model.pkl")
    return model

def load_random_forest():
    return joblib.load("models/rf_model.pkl")