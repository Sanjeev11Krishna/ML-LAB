from sklearn.ensemble import GradientBoostingRegressor
import joblib

def train_gradient_boosting(X_train, y_train):
    from sklearn.utils import resample
    X_sample, y_sample = resample(X_train, y_train, n_samples=min(5000, len(X_train)), random_state=42)
    
    model = GradientBoostingRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    model.fit(X_sample, y_sample)
    joblib.dump(model, "models/gb_model.pkl")
    return model

def load_gradient_boosting():
    return joblib.load("models/gb_model.pkl")