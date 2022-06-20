from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
import scipy.ndimage as ndi
import cv2

sigma = 1
deltax = -21
deltay = 1.84

img1 = np.random.rand(500, 500)
img1 = gaussian_filter(img1, sigma=sigma)
img2 = ndi.shift(img1, (deltay, deltax))

Image.fromarray((img1 * 255).astype(np.uint8)).save('im1.bmp')
Image.fromarray((img2 * 255).astype(np.uint8)).save('im2.bmp')


#=== 1d
sig1 = np.random.rand(700)
sig1 = gaussian_filter1d(sig1, sigma=sigma)
sig2 = ndi.shift(sig1, deltax)

np.savez_compressed('x1.npz', x1 = sig1)
np.savez_compressed('x2.npz', x2 = sig2)


