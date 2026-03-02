import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# ---------------------------
# Load and Train Model
# ---------------------------

@st.cache_data
def load_and_train():
    df = pd.read_csv("Iris.csv")

    if "Id" in df.columns:
        df = df.drop("Id", axis=1)

    X = df.drop("Species", axis=1)
    y = df["Species"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    model = SVC(kernel='linear')
    model.fit(X_train_pca, y_train)

    y_pred = model.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, scaler, pca, le, acc, cm, X_train_pca, y_train


model, scaler, pca, le, acc, cm, X_train_pca, y_train = load_and_train()

# ---------------------------
# UI Layout
# ---------------------------

st.title("🌸 Iris Flower Classification using SVM")

st.sidebar.header("Enter Flower Measurements")

sepal_length = st.sidebar.number_input("Sepal Length", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.number_input("Sepal Width", 2.0, 5.0, 3.5)
petal_length = st.sidebar.number_input("Petal Length", 1.0, 7.0, 1.4)
petal_width = st.sidebar.number_input("Petal Width", 0.1, 2.5, 0.2)

# ---------------------------
# Prediction
# ---------------------------

if st.sidebar.button("Predict"):

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)

    prediction = model.predict(input_pca)
    species = le.inverse_transform(prediction)

    st.subheader("Prediction Result")
    st.success(f"Predicted Species: {species[0]}")

# ---------------------------
# Show Model Accuracy
# ---------------------------

st.subheader("Model Accuracy")
st.write(f"Accuracy: {acc:.2f}")

# ---------------------------
# Show Confusion Matrix
# ---------------------------

st.subheader("Confusion Matrix")

fig_cm, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=le.classes_)
disp.plot(ax=ax)
st.pyplot(fig_cm)

# ---------------------------
# Decision Boundary Plot
# ---------------------------

st.subheader("SVM Decision Boundary (PCA Space)")

x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots()
ax.contourf(xx, yy, Z, alpha=0.3)
ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, edgecolors='k')

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title("SVM Decision Boundary")

st.pyplot(fig)