# 🌸 Iris Flower Classification using SVM with PCA (Streamlit UI)

## 📌 Project Description

This project implements a Support Vector Machine (SVM) classifier on the Iris dataset with Principal Component Analysis (PCA) for dimensionality reduction.  

A Streamlit-based web application allows users to input flower measurements and get real-time predictions of the flower species.

The system also displays:
- Model Accuracy
- Confusion Matrix
- SVM Decision Boundary Visualization

---

## 🎯 Objective

The objective of this project is to:

- Classify iris flowers into three species:
  - Iris-setosa
  - Iris-versicolor
  - Iris-virginica
- Apply Support Vector Machine (SVM) for classification
- Use PCA to reduce dimensionality from 4 features to 2
- Provide an interactive web interface using Streamlit

---

## 📂 Dataset Information

The project uses the Iris dataset, which contains 150 samples with the following features:

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

Target variable:
- Species (3 classes)

---

## 🧠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit

---

## ⚙️ Machine Learning Workflow

1. Load the dataset
2. Remove unnecessary columns (Id)
3. Encode species labels into numeric format
4. Split data into training and testing sets
5. Standardize features using StandardScaler
6. Apply PCA to reduce 4D data to 2D
7. Train a Linear SVM classifier
8. Evaluate the model using:
   - Accuracy
   - Confusion Matrix
   - Classification Report
9. Visualize the SVM decision boundary
10. Build interactive UI using Streamlit

---

## 🖥️ User Interface Features

The Streamlit application allows users to:

- Enter:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- Click the **Predict** button
- View:
  - Predicted species
  - Model accuracy
  - Confusion matrix
  - Decision boundary plot

---

## 🚀 Installation and Execution

### Step 1: Install Required Libraries

```bash
pip install streamlit pandas numpy scikit-learn matplotlib
```
### Step 2: Place Files

Ensure the following files are in the same directory:
```bash
app.py
Iris.csv
```

### Step 3: Run the Application
```bash
streamlit run app.py
```
The application will open automatically in your default web browser.