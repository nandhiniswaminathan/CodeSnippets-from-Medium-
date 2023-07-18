import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

# Fetching a dataset of handwritten digits from OpenML
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
X, y = shuffle(X, y)

# Select one digit from the dataset
original_image = X[0].reshape(28, 28)

# Create a noisy version of the image by adding random noise
noisy_image = original_image + np.random.normal(0, 50, (28, 28))

# Parameters
beta = 1  # strength of the interaction between neighboring pixels
eta = 1  # weighting of the data term

# Gibbs Sampling
restored_image = np.copy(noisy_image)
for iter in range(10):  # 10 iterations
    for i in range(28):
        for j in range(28):
            # Compute the conditional distribution
            neighbors_sum = (
                restored_image[i - 1, j]
                + restored_image[i, j - 1]
                + restored_image[(i + 1) % 28, j]
                + restored_image[i, (j + 1) % 28]
            )
            mean = (eta * noisy_image[i, j] + beta * neighbors_sum) / (eta + 4 * beta)
            restored_image[i, j] = np.random.normal(mean, 1)

# Plotting the original, noisy, and restored images
plt.figure(figsize=(12, 5))
plt.subplot(131)
plt.title('Original Image')
plt.imshow(original_image, cmap='gray')

plt.subplot(132)
plt.title('Noisy Image')
plt.imshow(noisy_image, cmap='gray')

plt.subplot(133)
plt.title('Restored Image')
plt.imshow(restored_image, cmap='gray')

plt.show()
