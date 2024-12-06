import cv2
import numpy as np
import matplotlib.pyplot as plt
def plot_histograms(image, title):
    b, g, r = cv2.split(image)
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.plot(hist_b, color='blue', label='Blue')
    plt.plot(hist_g, color='green', label='Green')
    plt.plot(hist_r, color='red', label='Red')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
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
def display_contrast(image, title):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.text(0.5, 0.5, f'Contrast: {contrast:.2f}', fontsize=20, ha='center')
    plt.axis('off')
    plt.show()
image = cv2.imread('picture/4.2.02.tiff')
b, g, r = cv2.split(image)
b_eq = cv2.equalizeHist(b)
g_eq = cv2.equalizeHist(g)
r_eq = cv2.equalizeHist(r)
equalized_image = cv2.merge([b_eq, g_eq, r_eq])
hist_b, bins_b = np.histogram(b.flatten(), 256, [0, 256])
hist_g, bins_g = np.histogram(g.flatten(), 256, [0, 256])
hist_r, bins_r = np.histogram(r.flatten(), 256, [0, 256])
cdf_b = hist_b.cumsum()
cdf_g = hist_g.cumsum()
cdf_r = hist_r.cumsum()
cdf_b = (cdf_b - cdf_b.min()) * 255 / (cdf_b.max() - cdf_b.min())
cdf_g = (cdf_g - cdf_g.min()) * 255 / (cdf_g.max() - cdf_g.min())
cdf_r = (cdf_r - cdf_r.min()) * 255 / (cdf_r.max() - cdf_r.min())
cdf_b = cdf_b.astype(np.uint8)
cdf_g = cdf_g.astype(np.uint8)
cdf_r = cdf_r.astype(np.uint8)
specified_image = cv2.merge([cdf_b[b], cdf_g[g], cdf_r[r]])
plot_histograms(image, 'Original Image Histogram')
plot_brightness_histogram(image, 'Original Image Brightness Histogram')
display_contrast(image, 'Original Image Contrast')
plot_histograms(equalized_image, 'Equalized Image Histogram')
plot_brightness_histogram(equalized_image, 'Equalized Image Brightness Histogram')
display_contrast(equalized_image, 'Equalized Image Contrast')
plot_histograms(specified_image, 'Specified Image Histogram')
plot_brightness_histogram(specified_image, 'Specified Image Brightness Histogram')
display_contrast(specified_image, 'Specified Image Contrast')
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 2)
plt.title('Equalized Image')
plt.imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 3)
plt.title('Specified Image')
plt.imshow(cv2.cvtColor(specified_image, cv2.COLOR_BGR2RGB))
plt.show()
