import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, generic_filter
from skimage import io
def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy_image = np.copy(image)
    total_pixels = image.size
    num_salt = int(total_pixels * salt_prob)
    num_pepper = int(total_pixels * pepper_prob)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    for channel in range(image.shape[2]):
        noisy_image[coords[0], coords[1], channel] = 1
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    for channel in range(image.shape[2]):
        noisy_image[coords[0], coords[1], channel] = 0
    return noisy_image
def add_gaussian_noise(image, mean=0, var=0.01):
    row, col, ch = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch)) * 255
    return np.clip(image + gauss, 0, 255).astype(np.uint8)
def add_salt_pepper_and_gaussian_noise(image, salt_prob=0.01, pepper_prob=0.01, mean=0, var=0.01):
    return add_gaussian_noise(add_salt_pepper_noise(image, salt_prob, pepper_prob), mean, var)
def median_filter_image(image, size=3):
    filtered_image = np.zeros_like(image)
    for channel in range(image.shape[2]):
        filtered_image[:, :, channel] = median_filter(image[:, :, channel], size=size)
    return filtered_image
def statistical_sorting_filter(image, size=3):
    def stat_sort_filter(values):
        return np.sort(values)[len(values) // 2]
    filtered_image = np.zeros_like(image)
    for channel in range(image.shape[2]):
        filtered_image[:, :, channel] = generic_filter(image[:, :, channel], stat_sort_filter, size=size)
    return filtered_image
image = io.imread('picture/4.2.02.tiff')
salt_pepper_image = add_salt_pepper_noise(image)
gaussian_image = add_gaussian_noise(image)
salt_pepper_gaussian_image = add_salt_pepper_and_gaussian_noise(image)
median_filtered_salt_pepper = median_filter_image(salt_pepper_image)
median_filtered_gaussian = median_filter_image(gaussian_image)
median_filtered_salt_pepper_gaussian = median_filter_image(salt_pepper_gaussian_image)
stat_sort_filtered_salt_pepper = statistical_sorting_filter(salt_pepper_image)
stat_sort_filtered_gaussian = statistical_sorting_filter(gaussian_image)
stat_sort_filtered_salt_pepper_gaussian = statistical_sorting_filter(salt_pepper_gaussian_image)
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
axs[0, 0].imshow(salt_pepper_image)
axs[0, 0].set_title('Salt & Pepper Noise')
axs[0, 0].axis('off')
axs[0, 1].imshow(gaussian_image)
axs[0, 1].set_title('Gaussian Noise')
axs[0, 1].axis('off')
axs[0, 2].imshow(salt_pepper_gaussian_image)
axs[0, 2].set_title('Salt & Pepper + Gaussian Noise')
axs[0, 2].axis('off')
axs[1, 0].imshow(median_filtered_salt_pepper)
axs[1, 0].set_title('Median Filtered Salt & Pepper')
axs[1, 0].axis('off')
axs[1, 1].imshow(median_filtered_gaussian)
axs[1, 1].set_title('Median Filtered Gaussian')
axs[1, 1].axis('off')
axs[1, 2].imshow(median_filtered_salt_pepper_gaussian)
axs[1, 2].set_title('Median Filtered Salt & Pepper + Gaussian')
axs[1, 2].axis('off')
axs[2, 0].imshow(stat_sort_filtered_salt_pepper)
axs[2, 0].set_title('Statistical Sorting Filtered Salt & Pepper')
axs[2, 0].axis('off')
axs[2, 1].imshow(stat_sort_filtered_gaussian)
axs[2, 1].set_title('Statistical Sorting Filtered Gaussian')
axs[2, 1].axis('off')
axs[2, 2].imshow(stat_sort_filtered_salt_pepper_gaussian)
axs[2, 2].set_title('Statistical Sorting Filtered Salt & Pepper + Gaussian')
axs[2, 2].axis('off')
plt.tight_layout()
plt.show()
