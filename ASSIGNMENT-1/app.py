import streamlit as st
import numpy as np
import pandas as pd
import joblib
from models.random_forest import load_random_forest
from models.gradient_boosting import load_gradient_boosting
from models.kmeans_model import predict_kmeans
from models.hierarchical_model import predict_hierarchical

# Load venue data and mapping
df = pd.read_csv("data/ipl_data.csv")
unique_venues = sorted(df['venue'].dropna().unique())
venue_mapping = joblib.load("models/venue_mapping.pkl")

st.title("🏏 IPL Final Score Predictor")

# Venue selection
venue = st.selectbox("Select Venue", unique_venues)

current_score = st.number_input("Current Score", min_value=0)
wickets = st.number_input("Wickets Lost", min_value=0, max_value=10)
overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0)

run_rate = current_score / overs if overs > 0 else 0
balls_remaining = (20 - overs) * 6
wickets_remaining = 10 - wickets

# Encode venue using the mapping
venue_encoded = venue_mapping.get(venue, -1)

input_data = np.array([[current_score, wickets, overs,
                        run_rate, balls_remaining,
                        wickets_remaining, venue_encoded]])

if st.button("Predict Final Score"):
    rf_model = load_random_forest()
    gb_model = load_gradient_boosting()

    rf_pred = rf_model.predict(input_data)[0]
    gb_pred = gb_model.predict(input_data)[0]
    km_pred = predict_kmeans(input_data)
    hc_pred = predict_hierarchical(input_data)

    st.subheader("📊 Predicted Final Scores")
    
    st.write(f"**Venue:** {venue}")
    st.write(f"Random Forest: {int(rf_pred)}")
    st.write(f"Gradient Boosting: {int(gb_pred)}")
    st.write(f"K-Means: {int(km_pred)}")
    st.write(f"Hierarchical: {int(hc_pred)}")