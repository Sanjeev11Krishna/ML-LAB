import numpy as np
from sklearn.preprocessing import StandardScaler

X=np.array([[78,85,80,82],[65,70,68,72],[90,92,88,91],[72,75,70,74],[85,88,84,86]])
print("Original Dataset:\n",X)

X_std=StandardScaler().fit_transform(X)
print("\nStandardized Data:\n",X_std)

cov_matrix=np.cov(X_std.T)
print("\nCovariance Matrix:\n",cov_matrix)

eigenvalues,eigenvectors=np.linalg.eig(cov_matrix)
print("\nEigenvalues:\n",eigenvalues)
print("\nEigenvectors:\n",eigenvectors)

idx=np.argsort(eigenvalues)[::-1]
eigenvalues=eigenvalues[idx]
eigenvectors=eigenvectors[:,idx]
print("\nSorted Eigenvalues:\n",eigenvalues)
print("\nSorted Eigenvectors:\n",eigenvectors)

k=1
principal_components=eigenvectors[:,:k]
print("\nPrincipal Component Matrix:\n",principal_components)

X_reduced=np.dot(X_std,principal_components)
print("\nReduced Dataset (After PCA):\n",X_reduced)

variance_ratio=eigenvalues/np.sum(eigenvalues)
print("\nVariance Percentage:\n",variance_ratio*100)
