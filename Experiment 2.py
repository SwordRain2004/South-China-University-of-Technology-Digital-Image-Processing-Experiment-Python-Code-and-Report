import cv2
import numpy as np
import matplotlib.pyplot as plt
def gamma_correction(image, gamma):
    image = image / 255.0
    corrected_image = np.power(image, gamma)
    corrected_image = np.uint8(corrected_image * 255)
    return corrected_image
def plot_brightness_histogram(image, title):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.plot(hist_gray, color='black', label='Brightness')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
image_path = 'picture/4.2.02.tiff'
image = cv2.imread(image_path)
gamma_values = [0.5, 1.0, 1.5, 2.0, 2.2]
corrected_images = [gamma_correction(image, gamma) for gamma in gamma_values]
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
for i, gamma in enumerate(gamma_values):
    plt.subplot(2, 3, i + 2)
    plt.title(f'Gamma = {gamma}')
    plt.imshow(cv2.cvtColor(corrected_images[i], cv2.COLOR_BGR2RGB))
    plt.axis('off')
plt.tight_layout()
plt.show()
plot_brightness_histogram(image, 'Original Image Brightness Histogram')
for i, gamma in enumerate(gamma_values):
    plot_brightness_histogram(corrected_images[i], f'Gamma = {gamma} Brightness Histogram')
