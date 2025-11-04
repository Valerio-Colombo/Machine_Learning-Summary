import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Define the path to save images
images_dir = os.path.join(os.path.dirname(__file__), '../images')

# Ensure the images directory exists
os.makedirs(images_dir, exist_ok=True)

# Generate random data points for three classes
np.random.seed(0)
n_samples = 50

# Class 1
mean1 = [2, 2]
cov1 = [[0.8, 0.2], [0.2, 0.8]]
X1 = np.random.multivariate_normal(mean1, cov1, n_samples)
y1 = np.zeros(n_samples)

# Class 2
mean2 = [6, 6]
cov2 = [[0.8, -0.2], [-0.2, 0.8]]
X2 = np.random.multivariate_normal(mean2, cov2, n_samples)
y2 = np.ones(n_samples)

# Class 3
mean3 = [10, 2]
cov3 = [[0.8, 0.2], [0.2, 0.8]]
X3 = np.random.multivariate_normal(mean3, cov3, n_samples)
y3 = np.full(n_samples, 2)

# Combine the data
X = np.vstack((X1, X2, X3))
y = np.hstack((y1, y2, y3))

# Fit LDA model
lda = LDA()
lda.fit(X, y)

# Plot the data points with different markers
plt.scatter(X1[:, 0], X1[:, 1], label='Class 1', alpha=1, color='red', marker='x')
plt.scatter(X2[:, 0], X2[:, 1], label='Class 2', alpha=1, color='green', marker='+')
plt.scatter(X3[:, 0], X3[:, 1], label='Class 3', alpha=1, color='blue', marker='o')

# Plot decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(3) - 0.5, colors=['#ffcccc', '#ccffcc', '#ccccff'], zorder=-1)
plt.colorbar(ticks=[0, 1, 2], format='Class %d')

plt.legend()
plt.axis('off')  # Turn off the axis

# Save the figure
image_path = os.path.join(images_dir, 'lda_decision_boundaries.png')
plt.savefig(image_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()