import cv2
import numpy as np


# Define the Gaussian kernel function
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size : size + 1, -size : size + 1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
    return g


# sobel_x and sobel_y are used to calculate the gradient in x and y direction respectively
def sobel_x():
    gaussian_1d = np.array([1, 2, 1], np.float32)
    x_derivative = np.array([-1, 0, 1], np.float32)
    s_x = np.outer(gaussian_1d, x_derivative)

    return s_x


def sobel_y():
    gaussian_1d = np.array([1, 2, 1], np.float32)
    x_derivative = np.array([-1, 0, 1], np.float32)
    s_y = np.outer(x_derivative, gaussian_1d)

    return s_y


# padding is used to make the image size same as the filter size
def padding(image):
    padded_image = np.pad(image, ((1, 1), (1, 1)), "constant", constant_values=(0, 0))

    return padded_image


# conv2d is used to convolve the image with the filter
def conv2d(image, ftr):
    image = padding(image)
    s = ftr.shape + tuple(np.subtract(image.shape, ftr.shape) + 1)
    sub_image = np.lib.stride_tricks.as_strided(
        image, shape=s, strides=image.strides * 2
    )
    return np.einsum("ij,ijkl->kl", ftr, sub_image)


def sobel(gray_image):
    G_x = conv2d(gray_image, sobel_x())
    G_y = conv2d(gray_image, sobel_y())
    M = np.sqrt(np.power(G_x, 2) + np.power(G_y, 2))

    theta = np.arctan2(G_y, G_x)

    return M, theta


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def threshold(img, lowThresholdRatio, highThresholdRatio):
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong


def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak:
                try:
                    if (
                        (img[i + 1, j - 1] == strong)
                        or (img[i + 1, j] == strong)
                        or (img[i + 1, j + 1] == strong)
                        or (img[i, j - 1] == strong)
                        or (img[i, j + 1] == strong)
                        or (img[i - 1, j - 1] == strong)
                        or (img[i - 1, j] == strong)
                        or (img[i - 1, j + 1] == strong)
                    ):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


if __name__ == "__main__":
    img = cv2.imread("test_image.jpg", cv2.IMREAD_GRAYSCALE)
    # image = cv2.resize(img, (400, 550))

    # Apply the Gaussian filter
    kernel_size = 15  # Adjust the kernel size as needed
    sigma = 4  # Adjust the sigma value as needed
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

    # apply sobel
    M, theta = sobel(filtered_img_resized)

    # display the result
    cv2.imshow("step 2", np.uint8(M))  # what is np.uint8 ?
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    Z = non_max_suppression(M, theta)

    cv2.imshow("step 3", np.uint8(Z))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # threshold the image to get the edges using the threshold function defined above in the code
    res, weak, strong = threshold(Z, lowThresholdRatio=0.05, highThresholdRatio=0.09)

    # display the result
    cv2.imshow("step 4", np.uint8(res))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    canny = hysteresis(res, weak, strong=255)
    cv2.imshow("step 5", np.uint8(canny))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
