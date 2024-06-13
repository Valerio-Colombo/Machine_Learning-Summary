import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import os

# Define the data points and their labels
X = np.array([
    [2, 3], [3, 3], [3, 4], [5, 4], [6, 4], [7, 4], [7, 5], [8, 5], [8, 6], [9, 6],
    [1, 1], [2, 2], [3, 1], [4, 2], [5, 1], [6, 2], [7, 1], [8, 2], [9, 1], [10, 2]
])
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

# Create and fit the SVM models
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X, y)

svm_rbf = SVC(kernel='rbf', C=1.0)
svm_rbf.fit(X, y)

# Define the path to save images
images_dir = os.path.join(os.path.dirname(__file__), '../images')

# Ensure the images directory exists
os.makedirs(images_dir, exist_ok=True)

# Plotting function with margin lines and saving to file
def plot_svm(X, y, model, title, filename):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')

    # Create grid to evaluate model
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 500), np.linspace(ylim[0], ylim[1], 500))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and margins
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.title(title)
    plt.legend()
    plt.axis('off')  # Turn off the axis

    # Save the plot to a file
    plt.savefig(os.path.join(images_dir, filename), bbox_inches='tight')
    plt.close()

# Plot and save the results
plot_svm(X, y, svm_linear, 'SVM with Linear Kernel', 'SVM_linear.svg')
plot_svm(X, y, svm_rbf, 'SVM with RBF Kernel', 'SVM_rbf.svg')

# Print the number of support vectors
print(f"Number of support vectors (linear kernel): {len(svm_linear.support_vectors_)}")
print(f"Number of support vectors (RBF kernel): {len(svm_rbf.support_vectors_)}")

