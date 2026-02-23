import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
class DecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None
    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else self.n_features
        self.root = self._grow_tree(X, y)
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            return Node(value=self._most_common_label(y))
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        if best_feat is None:
            return Node(value=self._most_common_label(y))
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr
        return split_idx, split_thresh
    def _information_gain(self, y, X_column, threshold):
        parent_gini = self._gini(y)
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        gini_l = self._gini(y[left_idxs])
        gini_r = self._gini(y[right_idxs])
        child_gini = (n_l/n)*gini_l + (n_r/n)*gini_r
        return parent_gini - child_gini
    def _gini(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return 1 - np.sum(ps**2)
    def _split(self, X_column, threshold):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs
    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
class RandomForest:
    def __init__(self, n_trees=3, max_depth=3):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                                n_features=int(np.sqrt(X.shape[1])))
            idxs = np.random.choice(len(X), len(X), replace=True)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        final_preds = []
        for preds in tree_preds:
            final_preds.append(Counter(preds).most_common(1)[0][0])
        return np.array(final_preds), tree_preds
def plot_tree(node, ax, x=0.5, y=1, dx=0.25, dy=0.15, class_names=None):
    if node.value is not None:
        ax.text(x, y, class_names[node.value],
                ha='center', va='center',
                bbox=dict(boxstyle="round"))
        return
    ax.text(x, y, f"X{node.feature} ≤ {node.threshold:.2f}",
            ha='center', va='center',
            bbox=dict(boxstyle="round"))
    ax.plot([x, x-dx], [y-0.05, y-dy], 'k-')
    ax.plot([x, x+dx], [y-0.05, y-dy], 'k-')
    plot_tree(node.left, ax, x-dx, y-dy, dx/1.5, dy, class_names)
    plot_tree(node.right, ax, x+dx, y-dy, dx/1.5, dy, class_names)
iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
rf = RandomForest(n_trees=3, max_depth=3)
rf.fit(X_train, y_train)
y_pred, _ = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=class_names)
print("\nModel Trained Successfully ✅")
print(f"\nAccuracy: {accuracy:.4f}\n")
print("Classification Report:\n")
print(report)
for i, tree in enumerate(rf.trees):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    plot_tree(tree.root, ax, class_names=class_names)
    plt.title(f"Tree {i+1}")
    plt.show()
def predict():
    try:
        values = [
            float(entry1.get()), float(entry2.get()), float(entry3.get()), float(entry4.get())
        ]
        sample = np.array(values).reshape(1, -1)
        final_pred, tree_preds = rf.predict(sample)
        result_text = ""
        for i, p in enumerate(tree_preds[0]):
            result_text += f"Tree {i+1}: {class_names[p]}\n"
        result_text += f"\nMajority Vote: {class_names[final_pred[0]]}"
        result_label.config(text=result_text)
    except:
        messagebox.showerror("Error", "Enter valid numeric values")
root = tk.Tk()
root.title("Random Forest (From Scratch) - Iris")
root.geometry("450x500")
tk.Label(root, text="Sepal Length").pack()
entry1 = tk.Entry(root)
entry1.pack()
tk.Label(root, text="Sepal Width").pack()
entry2 = tk.Entry(root)
entry2.pack()
tk.Label(root, text="Petal Length").pack()
entry3 = tk.Entry(root)
entry3.pack()
tk.Label(root, text="Petal Width").pack()
entry4 = tk.Entry(root)
entry4.pack()
tk.Button(root, text="Predict", command=predict).pack(pady=10)
result_label = tk.Label(root, text="", justify="left")
result_label.pack()
root.mainloop()
