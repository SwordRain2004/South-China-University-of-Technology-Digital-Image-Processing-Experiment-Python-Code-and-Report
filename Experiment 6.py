import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from skimage import io, color, restoration
def gaussian_blur(image, sigma=1.0):
    return gaussian_filter(image, sigma=sigma)
def motion_blur(image, size=15, angle=45):
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    kernel_motion_blur = np.rot90(kernel_motion_blur, k=angle // 90)
    return convolve2d(image, kernel_motion_blur, mode='same', boundary='wrap')
def wiener_filter(image, psf, K=0.01):
    psf = psf + np.finfo(float).eps
    return restoration.wiener(image, psf, K)
def create_gaussian_psf(size, sigma):
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    psf /= psf.sum()
    return psf
image = io.imread('picture/4.2.02.tiff')
if len(image.shape) == 3:
    image = color.rgb2gray(image)
blurred_gaussian = gaussian_blur(image, sigma=2.0)
blurred_motion = motion_blur(image, size=15, angle=45)
psf_gaussian = create_gaussian_psf(size=15, sigma=2.0)
psf_motion = np.zeros((15, 15))
psf_motion[int((15-1)/2), :] = np.ones(15)
psf_motion = psf_motion / 15
psf_motion = np.rot90(psf_motion, k=45 // 90)
psf_motion /= psf_motion.sum()
restored_gaussian = wiener_filter(blurred_gaussian, psf_gaussian, K=0.01)
restored_motion = wiener_filter(blurred_motion, psf_motion, K=0.01)
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')
axs[0, 1].imshow(blurred_gaussian, cmap='gray')
axs[0, 1].set_title('Gaussian Blurred Image')
axs[0, 1].axis('off')
axs[0, 2].imshow(restored_gaussian, cmap='gray')
axs[0, 2].set_title('Restored Gaussian Image')
axs[0, 2].axis('off')
axs[1, 1].imshow(blurred_motion, cmap='gray')
axs[1, 1].set_title('Motion Blurred Image')
axs[1, 1].axis('off')
axs[1, 2].imshow(restored_motion, cmap='gray')
axs[1, 2].set_title('Restored Motion Image')
axs[1, 2].axis('off')
plt.tight_layout()
plt.show()
