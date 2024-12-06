import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift
from skimage import io, color
def low_pass_filter(fshift, cutoff):
    rows, cols = fshift.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1
    fshift_filtered = fshift * mask
    return fshift_filtered
def butterworth_high_pass_filter(fshift, cutoff, order=2):
    rows, cols = fshift.shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(v - ccol, u - crow)
    D = np.sqrt(U**2 + V**2)
    H = 1 / (1 + (cutoff / (D + 1e-5))**(2 * order))
    fshift_filtered = fshift * H
    return fshift_filtered, H
image = io.imread('picture/4.2.02.tiff')
if len(image.shape) == 3:
    image = color.rgb2gray(image)
f = fft2(image)
fshift = fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))
cutoff_low, cutoff, order = 30, 30, 2
fshift_low_pass = low_pass_filter(fshift, cutoff_low)
f_low_pass = fftshift(fshift_low_pass)
image_low_pass = ifft2(f_low_pass)
image_low_pass = np.abs(image_low_pass)
fshift_high_pass, H = butterworth_high_pass_filter(fshift, cutoff, order)
f_high_pass = fftshift(fshift_high_pass)
image_high_pass = ifft2(f_high_pass)
image_high_pass = np.abs(image_high_pass)
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')
axs[1, 0].imshow(magnitude_spectrum, cmap='gray')
axs[1, 0].set_title('Magnitude Spectrum')
axs[1, 0].axis('off')
axs[0, 1].imshow(image_low_pass, cmap='gray')
axs[0, 1].set_title('Low Pass Filtered Image')
axs[0, 1].axis('off')
axs[1, 1].imshow(np.abs(fshift_low_pass), cmap='gray')
axs[1, 1].set_title('Low Pass Filtered Spectrum')
axs[1, 1].axis('off')
axs[0, 2].imshow(image_high_pass, cmap='gray')
axs[0, 2].set_title('High Pass Filtered Image')
axs[0, 2].axis('off')
axs[1, 2].imshow(np.abs(fshift_high_pass), cmap='gray')
axs[1, 2].set_title('High Pass Filtered Spectrum')
axs[1, 2].axis('off')
plt.tight_layout()
plt.show()
