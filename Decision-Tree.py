import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ======================
# LOAD IRIS DATASET
# ======================

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
class_names = iris.target_names

# ======================
# SPLIT DATA
# ======================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# TRAIN MODEL
# ======================

model = DecisionTreeClassifier(random_state=42, max_depth=4)
model.fit(X_train, y_train)

# ======================
# EVALUATION
# ======================

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nModel Trained Successfully ✅")
print(f"\nAccuracy: {accuracy:.4f}\n")
print("Classification Report:\n")
print(report)

# ======================
# DISPLAY DECISION TREE
# ======================

plt.figure(figsize=(15,8))
plot_tree(model,
          feature_names=X.columns,
          class_names=class_names,
          filled=True)
plt.title("Decision Tree Visualization - Iris Dataset")
plt.show()

# ======================
# USER INPUT SECTION
# ======================

print("\nEnter Flower Details to Predict:\n")

try:
    sepal_length = float(input("Sepal Length: "))
    sepal_width = float(input("Sepal Width: "))
    petal_length = float(input("Petal Length: "))
    petal_width = float(input("Petal Width: "))

    input_data = pd.DataFrame([[ 
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]], columns=X.columns)

    prediction = model.predict(input_data)[0]

    print("\nPredicted Flower Type:", class_names[prediction])

except:
    print("\nInvalid input! Please enter numeric values only.")
