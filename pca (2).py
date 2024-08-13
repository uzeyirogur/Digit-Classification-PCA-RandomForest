import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load and visualize the digits dataset
digits = datasets.load_digits()
x = digits.data
y = digits.target

plt.figure(figsize=(16,6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x[i, :].reshape([8, 8]), cmap="gray")

"""
Each data point represents a digit drawn on an 8x8 pixel grid.
This makes each image 64 pixels in total. The images are flattened into a 64-dimensional vector.
Pixel values range from 0 (black) to 255 (white).
"""

# Reduce dimensionality with PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

# 2D visualization after PCA
plt.figure(figsize=(12, 10))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, edgecolors="none", alpha=0.7, s=40, cmap=plt.cm.get_cmap("nipy_spectral", 10))
plt.colorbar()
plt.show()

"""
PCA is often used to reduce the dimensionality of the data, making it easier to visualize and process.
Here, we reduce the 64-dimensional data to 2 dimensions.
"""

# Variance explained by each component
pca2 = PCA().fit(x)
plt.figure(figsize=(10, 7))
plt.plot(np.cumsum(pca2.explained_variance_ratio_), color="k", lw=2)
plt.xlabel("Number of Principal Components")
plt.ylabel("Total Explained Variance")
plt.xlim(0, 63)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.axvline(21, c="b")
plt.axhline(0.9, c="r")
plt.show()

"""
In practice, the number of components is chosen based on the total variance they explain.
For example, selecting enough components to explain 90% of the variance.
"""

# Apply PCA with selected number of components
pca3 = PCA(n_components=21)
x_pca2 = pca3.fit_transform(x)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x_pca2, y, random_state=0, test_size=0.2)

# Train Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=400, max_depth=4, criterion="entropy", random_state=0)
rfc.fit(x_train, y_train)
rfc_predict = rfc.predict(x_test)

# Calculate accuracy
acc = accuracy_score(y_test, rfc_predict) * 100
print(f"Accuracy: {acc:.2f}%")
