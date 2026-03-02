# =====================================================
# SVM WITH PCA + HYPERPLANE + CONFUSION MATRIX
# IRIS DATASET
# =====================================================

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# =====================================================
# 2️⃣ Load Dataset
# =====================================================

df = pd.read_csv("Iris.csv")

# Drop Id column if present
if "Id" in df.columns:
    df = df.drop("Id", axis=1)

# =====================================================
# 3️⃣ Separate Features and Target
# =====================================================

X = df.drop("Species", axis=1)
y = df["Species"]

# Convert labels to numbers
le = LabelEncoder()
y = le.fit_transform(y)

# =====================================================
# 4️⃣ Train-Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# 5️⃣ Feature Scaling
# =====================================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================================
# 6️⃣ PCA (4D → 2D for Visualization)
# =====================================================

pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# =====================================================
# 7️⃣ Train Linear SVM
# =====================================================

model = SVC(kernel='linear')
model.fit(X_train, y_train)

# =====================================================
# 8️⃣ Predictions
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# 9️⃣ Print Evaluation
# =====================================================

print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# =====================================================
# 🔟 Plot Hyperplane (Decision Boundary)
# =====================================================

# Create mesh grid
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.01),
    np.arange(y_min, y_max, 0.01)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k')

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("SVM Decision Boundary (Hyperplane) after PCA")
plt.show()

# =====================================================
# 1️⃣1️⃣ Plot Confusion Matrix Graph
# =====================================================

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=le.classes_)
disp.plot()
plt.title("Confusion Matrix - SVM with PCA")
plt.show()