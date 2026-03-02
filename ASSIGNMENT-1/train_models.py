import pandas as pd
from sklearn.model_selection import train_test_split
from feature_engineering import create_features
from models.random_forest import train_random_forest
from models.gradient_boosting import train_gradient_boosting
from models.kmeans_model import train_kmeans
from models.hierarchical_model import train_hierarchical

df = pd.read_csv("data/ipl_data.csv")

df, venue_mapping = create_features(df)

features = ['current_score','wickets','overs',
            'current_run_rate','balls_remaining','wickets_remaining', 'venue_encoded']

X = df[features]
y = df['final_total']

# Save venue mapping for use in app
import joblib
joblib.dump(venue_mapping, "models/venue_mapping.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_random_forest(X_train, y_train)
train_gradient_boosting(X_train, y_train)
train_kmeans(X_train, y_train)
train_hierarchical(X_train, y_train)

print("All models trained successfully!")