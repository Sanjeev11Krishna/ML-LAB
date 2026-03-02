import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def preprocess_data(df, training=True):

    categorical_features = [
        'ground',
        'match_type',          # Day/Night
        'batting_order'        # First/Second
    ]

    numerical_features = [
        'current_score',
        'wickets',
        'overs',
        'current_run_rate',
        'balls_remaining',
        'wickets_remaining'
    ]

    if training:

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        X = preprocessor.fit_transform(df)
        joblib.dump(preprocessor, "models/preprocessor.pkl")

        return X

    else:
        preprocessor = joblib.load("models/preprocessor.pkl")
        X = preprocessor.transform(df)
        return X