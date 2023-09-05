import cv2
import numpy as np
from scipy import ndimage

# Read image and convert to grayscale
img = cv2.imread("test_image.jpg", cv2.IMREAD_GRAYSCALE)


# Define the Gaussian kernel function
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size : size + 1, -size : size + 1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
    return g


# Apply the Gaussian filter
kernel_size = 20  # Adjust the kernel size as needed
sigma = 5  # Adjust the sigma value as needed
gaussian_kernel = gaussian_kernel(kernel_size, sigma)
filtered_image = cv2.filter2D(img, -1, gaussian_kernel)
filtered_img_resized = cv2.resize(filtered_image, (400, 550))

# Resize the original image to match the dimensions of the filtered image
desired_height, desired_width = filtered_image.shape[:2]
original_img_resized = cv2.resize(img, (400, 550))

# Show the original and filtered images
cv2.imshow("Original Image", original_img_resized)
cv2.imshow("Filtered Image", filtered_img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
