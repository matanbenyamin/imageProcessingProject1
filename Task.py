
# load numpy array from npy file
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy import load
from skimage.transform import downscale_local_mean
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sklearn
from skimage.io import imshow, imread

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.ndimage import interpolation
from scipy.ndimage import gaussian_filter1d
import scipy.ndimage as ndi




import sys
!{sys.executable} -m pip install sklearn



def solve_2d(im1, im2):
    # ==============================#
    # Caculates registration with lsq
    # Output:
    #   dr -  shift in each dimensiotn
    #   im2 - shifted image based on dr
    # ==============================#

    imx = im1[1:, 1:] - im1[1:, :-1]
    imy = im1[1:, 1:] - im1[:-1, 1:]

    A = np.transpose([imx.flatten(), imy.flatten()])
    At = np.transpose(A)

    y = im2[1:, 1:] - im1[1:, 1:]
    y = y.flatten()

    s = np.linalg.pinv(At @ A)
    dr = s @ (At @ y)
    im2_shifted = ndi.shift(im2, dr)

    return dr, im2_shifted


def solve_1d(sig1, sig2):
    # ==============================#
    # Caculates registration with lsq
    # Output:
    #   dr -  shift in each dimensiotn
    #   sig2 - shifted signal based on dr
    # ==============================#

    # calculate derivative
    A = (sig1[2:] - sig1[:-2]) / 2

    # add zeros colum to use pinv
    A = np.c_[A, np.zeros_like(A)]

    At = np.transpose(A)
    y = sig2 - sig1
    y = y[1:-1]

    s = np.linalg.pinv(At @ A)
    dx = s @ np.transpose(At @ y)

    dr = dx[0]
    sig2_shifted = ndi.shift(sig2, dr)

    return dr, sig2_shifted

 # load array
delta = load('Delta.npy')
delta = 2.25
x1 = load('x1.npy')
x2 = load('x2.npy')
# Iterative one-dimensional registration
print('a.Iterative one-dimensional registration ')
print('Two one Dimensional signal ')
val1 = input("Enter  1st : ")
val2 = input("Enter 2nd : ")
plt.rcParams['figure.figsize']= (val1,val2)
X = x1
Y = x2
plt.scatter(X, Y)
plt.show()
# Building the model
m = delta
c = 0

L = 0.0001  # The learning Rate
epochs = 1000  # The number of iterations to perform gradient

n = float(len(X))  # Number of elements in X

# Performing Gradient
for i in range(epochs):
    x1, x2 = solve_1d(x1, x2)
    Y_pred = m * X + c  # The current predicted value of Y
    D_m = (-2 / n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2 / n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c

print(m, c)
# Y_pred = m*X + c
# plt.scatter(X, Y)
# plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')
# plt.show()
# .....................................


# b. One-dimensional multi-scale registration
print('b. One-dimensional multi-scale registration')
print('Two one Dimensional signal ')
# a = input("Enter  1st : ")

imge = imread('img2.jpeg')
# b = input("Enter 2nd : ")
factors = 3**np.arange(1, 5)
figure, axis = plt.subplots(1, len(factors), figsize=(20, 6))
for factor, ax in zip(factors, axis):
    image = downscale_local_mean(imge ,
                 factors=(factor, factor, 1)).astype(int)
    imge, image = solve_2d(imge, image)

    # ax.imshow(image)
    # ax.set_title('$N={}$'.format(image.shape[0]))
    # plt.imshow(image)
    # plt.show()
# ..................................................
#
# c. Iterative two-dimensional registration
# d. One-dimensional multi-scale registration
print('c. Iterative two-dimensional registration')

# dx1 = input("Enter  dx : ")
# dy2=input("Enter  dy : ")
import  cv2
from PIL.Image import Image
# Find the Gaussian pyramid of the two images and the mask
def gaussian_pyramid(img, num_levels):
    lower = img.copy()
    gaussian_pyr = [lower]
    for i in range(num_levels):
        lower = cv2.pyrDown(lower)
        gaussian_pyr.append(np.float32(lower))
    return gaussian_pyr


# Then calculate the Laplacian pyramid
def laplacian_pyramid(gaussian_pyr):
    laplacian_top = gaussian_pyr[-1]
    num_levels = len(gaussian_pyr) - 1

    laplacian_pyr = [laplacian_top]
    for i in range(num_levels, 0, -1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
        laplacian = np.subtract(gaussian_pyr[i - 1], gaussian_expanded)
        laplacian_pyr.append(laplacian)
    return laplacian_pyr


def blend(laplacian_A, laplacian_B, mask_pyr):
    LS = []
    for la, lb, mask in zip(laplacian_A, laplacian_B, mask_pyr):
        ls = lb * mask + la * (1.0 - mask)
        LS.append(ls)
    return LS


def reconstruct(laplacian_pyr):
    laplacian_top = laplacian_pyr[0]
    laplacian_lst = [laplacian_top]
    num_levels = len(laplacian_pyr) - 1
    for i in range(num_levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        laplacian_expanded = cv2.pyrUp(laplacian_top, dstsize=size)
        laplacian_top = cv2.add(laplacian_pyr[i + 1], laplacian_expanded)
        laplacian_lst.append(laplacian_top)
    return laplacian_lst
def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    print('c. Iterative two-dimensional registration')

    dx = input("Enter  dx : ")
    dy = input("Enter  dy : ")
    # padding=dx
    # strides=dy
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output
def processImage(image):
  image = cv2.imread(image)
  image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
  return image

if __name__ == '__main__':
    # Step-1
    # Load the two images
    img1 = cv2.imread('img2.jpeg')
    img1 = cv2.resize(img1, (1800, 1000))
    img2 = cv2.imread('img3.jpeg')
    img2 = cv2.resize(img2, (1800, 1000))

    # Create the mask
    mask = np.zeros((1000, 1800, 3), dtype='float32')
    mask[250:500, 640:1440, :] = (1, 1, 1)
    delta = load("Delta.npy")
    num_levels = delta

    # For image-1, calculate Gaussian and Laplacian
    gaussian_pyr_1 = gaussian_pyramid(img1, num_levels)
    laplacian_pyr_1 = laplacian_pyramid(gaussian_pyr_1)
    # For image-2, calculate Gaussian and Laplacian
    gaussian_pyr_2 = gaussian_pyramid(img2, num_levels)
    laplacian_pyr_2 = laplacian_pyramid(gaussian_pyr_2)
    # Calculate the Gaussian pyramid for the mask image and reverse it.
    mask_pyr_final = gaussian_pyramid(mask, num_levels)
    mask_pyr_final.reverse()

    add_laplace = blend(laplacian_pyr_1, laplacian_pyr_2, mask_pyr_final)

    final = reconstruct(add_laplace)
    # Save the final image to the disk
    cv2.imwrite('result.jpeg', final[num_levels])
    plt.imshow(mpimg.imread('result.jpeg'))
    plt.show()
    image = processImage('img2.jpeg')

    # Edge Detection Kernel
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    # Convolve and Save Output
    output = convolve2D(image, kernel, padding=2)
    cv2.imwrite('2DConvolved.jpg', output)


    # plt.imshow(mpimg.imread('2DConvolved.jpg'))
    # plt.show()