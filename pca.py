import numpy as np
from sklearn.preprocessing import StandardScaler

# Each row = one person, each column = one feature
# Features: Height, Arm Length
X = np.array([
    [150, 60],
    [160, 65],
    [170, 70],
    [180, 75]
])

print("Original Dataset:\n", X)

# Step 1: Standardization
X_std = StandardScaler().fit_transform(X)
print("\nStandardized Data:\n", X_std)

# Step 2: Covariance Matrix
cov_matrix = np.cov(X_std.T)
print("\nCovariance Matrix:\n", cov_matrix)

# Step 3: Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)

# Step 4: Sort eigenvalues in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("\nSorted Eigenvalues:\n", eigenvalues)
print("\nSorted Eigenvectors:\n", eigenvectors)

# Step 5: Select top k principal components
k = 1
principal_components = eigenvectors[:, :k]
print("\nPrincipal Component Matrix:\n", principal_components)

# Step 6: Transform the data
X_reduced = np.dot(X_std, principal_components)
print("\nReduced Dataset (After PCA):\n", X_reduced)

# Step 7: Variance ratio
variance_ratio = eigenvalues / np.sum(eigenvalues)
print("\nVariance Percentage:\n", variance_ratio * 100)
