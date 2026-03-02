import numpy as np
from collections import Counter

class DecisionTreeNode:
    """Node class for Decision Tree"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    """Decision Tree Classifier implemented from scratch"""
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def fit(self, X, y):
        """Build the decision tree"""
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        """Recursively build the tree"""
        n_samples = len(y)
        num_labels = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            num_labels == 1 or 
            n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)
        
        # Find best split
        feature_idx, threshold = self._best_split(X, y)
        
        # Split the data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        # Check if split is valid
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)
        
        # Recursively build left and right subtrees
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionTreeNode(feature=feature_idx, threshold=threshold, 
                               left=left, right=right)
    
    def _best_split(self, X, y):
        """Find the best split using Gini impurity"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        current_gini = self._gini_index(y)
        n_samples = len(y)
        
        for feature_idx in range(self.n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                gain = self._information_gain(y, current_gini, X[:, feature_idx], threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _gini_index(self, y):
        """Calculate Gini impurity"""
        gini = 1.0
        for label in np.unique(y):
            prob = np.sum(y == label) / len(y)
            gini -= prob ** 2
        return gini
    
    def _information_gain(self, y, current_gini, column, threshold):
        """Calculate information gain from a split"""
        left_mask = column <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
        
        n = len(y)
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        
        # Weighted average of child Ginis
        child_gini = (n_left / n) * self._gini_index(y[left_mask]) + \
                     (n_right / n) * self._gini_index(y[right_mask])
        
        return current_gini - child_gini
    
    def _most_common_label(self, y):
        """Return the most common label in y"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        predictions = [self._predict_sample(sample, self.root) for sample in X]
        return np.array(predictions)
    
    def _predict_sample(self, sample, node):
        """Predict class label for a single sample"""
        if node.is_leaf():
            return node.value
        
        if sample[node.feature] <= node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)


class RandomForest:
    """Random Forest Classifier implemented from scratch"""
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=2, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
    
    def fit(self, X, y):
        """Build the random forest using bootstrap aggregating (bagging)"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.trees = []
        n_samples = X.shape[0]
        
        print(f"Training {self.n_trees} decision trees...")
        
        for i in range(self.n_trees):
            # Bootstrap sampling (sample with replacement)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Create and train a decision tree
            tree = DecisionTree(max_depth=self.max_depth, 
                              min_samples_split=self.min_samples_split)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            
            # Calculate out-of-bag error for this tree
            oob_indices = np.setdiff1d(np.arange(n_samples), np.unique(indices))
            if len(oob_indices) > 0:
                oob_pred = tree.predict(X[oob_indices])
                oob_accuracy = np.mean(oob_pred == y[oob_indices])
                print(f"  Tree {i+1}/{self.n_trees} - OOB Accuracy: {oob_accuracy:.4f}")
            else:
                print(f"  Tree {i+1}/{self.n_trees} trained")
    
    def predict(self, X):
        """Predict class labels using majority voting"""
        # Collect predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Majority voting
        final_predictions = []
        for i in range(X.shape[0]):
            votes = tree_predictions[:, i]
            # Return the most common vote
            prediction = Counter(votes).most_common(1)[0][0]
            final_predictions.append(prediction)
        
        return np.array(final_predictions), tree_predictions
    
    def score(self, X, y):
        """Calculate accuracy on given test data"""
        predictions, _ = self.predict(X)
        return np.mean(predictions == y)