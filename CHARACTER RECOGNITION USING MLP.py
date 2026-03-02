import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

st.set_page_config(page_title="Character Recognition using MLP", layout="wide")
st.title("Handwritten Character Recognition (A–Z)")

@st.cache_data
def load_data():
    data = pd.read_csv("A_Z Handwritten Data.csv", nrows=50000).astype("float32")
    return data

data = load_data()

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

letters = [chr(i) for i in range(65, 91)]
y_letters = np.array([letters[int(label)] for label in y])

model_exists = os.path.exists("model.pkl")

if not model_exists:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_letters, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(128,),
                        activation='relu',
                        solver='adam',
                        max_iter=100,
                        random_state=42)

    mlp.fit(X_train, y_train)

    joblib.dump((mlp, scaler), "model.pkl")

model, scaler = joblib.load("model.pkl")

uploaded_file = st.file_uploader("Upload Character Image",
                                 type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))
    image_array = np.array(image)

    image_flat = image_array.reshape(1, -1)
    image_scaled = scaler.transform(image_flat)

    if st.button("Analyze Character"):
        prediction = model.predict(image_scaled)
        st.image(image_array, width=200)
        st.success(f"Predicted Character: {prediction[0]}")