import pandas as pd

def create_features(df):
    df['current_score'] = df['innings_score']
    df['wickets'] = df['innings_wickets']
    df['overs'] = df['over'] + (df['ball'] / 6)
    df['current_run_rate'] = df['current_score'] / df['overs'].replace(0, 1)
    df['balls_remaining'] = (20 - df['overs']) * 6
    df['wickets_remaining'] = 10 - df['wickets']
    
    # Encode venue as a categorical feature (using label encoding)
    venue_mapping = {venue: idx for idx, venue in enumerate(df['venue'].dropna().unique())}
    df['venue_encoded'] = df['venue'].map(venue_mapping).fillna(-1)
    
    # Create target variable - final total score for the match
    df['final_total'] = df.groupby('match_id')['innings_score'].transform('max')
    
    return df, venue_mapping