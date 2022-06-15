import cv2
import argparse
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import gaussian_filter

# !!! INPUT FILES - Change with your file PATH !!!
img_path = 'img2.jpeg'
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
    im2_shifted = ndi.shift(im2, (dr[1], dr[0]))

    return dr, im2_shifted


def solve_iter_2d(img1, img2, max_num_iter=150):
    # ==============================#
    # Calculates registration with lsq
    # Input:
    #   img1 - image to be registered
    #   img2 - image to be registered to
    #   max_num_iter - maximum number of iterations(default: 1500)
    # Output:
    #   dr -  shift in each dimensiotn
    #   im2_shifted - shifted image based on dr
    #   dx_vec - vector of shifts over convergence
    # ==============================#
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    dr = 0

    # if smoothness < 1.5e-3:
    #     img1_s = gaussian_filter(img1, 15)
    #     img2_s = gaussian_filter(img2, 15)
    #
    #     dr = solve_2d(img1_s, img2_s)
    #     dr = dr[0]
    #     print(dr)
    #     img2 = ndi.shift(img2, (dr[1], dr[0]))
    #     dri = (np.ceil(abs(dr))).astype(int)
    #     img1 = img1[dri[1]:-dri[1], dri[0]:-dri[0]]
    #     img2 = img2[dri[1]:-dri[1], dri[0]:-dri[0]]




    # initialize
    cumul_dx =dr
    dx_vec = []
    img2_shifted = img2
    for i in range(150):
        dr, img2_shifted = solve_2d(img1, img2_shifted)

        cumul_dx += dr
        dx_vec.append(dr)

        #dri is the shift in each dimension integerm used to crop img1 and img2
        dri = (np.ceil(abs(dr))).astype(int)

        if dri[0] > 0 and dri[1] > 0:
            img1 = img1[dri[1]:-dri[1], dri[0]:-dri[0]]
            img2_shifted = img2_shifted[dri[1]:-dri[1], dri[0]:-dri[0]]
        if i > 1:
            # dr doesn't change much, converged
            if np.max(np.abs(dr)) < 0.00001:
                print('converged at iteration: ', i)
                break


    return cumul_dx, img2_shifted,  dx_vec


def get_downscaled_img_2d(img, scale, stridex=0, stridey=0):
    # ==============================#
    # Downscales image by scale    #
    # ==============================#
    if scale < 1:
        # gaussian filter img
        img_downscaled = gaussian_filter(img, 2/scale)
    else:
        img_downscaled = img
    # assert stride<1/scale, 'Stride is too large'
    return img_downscaled[stridex::int(1 / scale), stridey::int(1 / scale)]


def register_multiscale_2d(img1, img2, scale_list=[0.25, 0.5, 1]):
    # ==============================#
    # Caculates registration with lsq
    # with multiscale registration
    # Output:
    #   cumul_dx - shift after convergence
    #   img2 - shifted image based on cumul_dx
    #   dx_vec - vector of shifts over convergence
    # ==============================#

    cumul_dx = 0
    dx_vec = []
    img2_shifted = img2

    for scale in scale_list:
        img1_downscaled = get_downscaled_img_2d(img1, scale)
        img2_downscaled = get_downscaled_img_2d(img2_shifted, scale)

        # match lengths of img1 and img2
        img1_downscaled = img1_downscaled[:np.min([len(img1_downscaled), len(img2_downscaled)])]
        img2_downscaled = img2_downscaled[:np.min([len(img1_downscaled), len(img2_downscaled)])]

        dr = solve_iter_2d(img1_downscaled, img2_downscaled)
        dr = dr[0]
        dr = dr / scale
        cumul_dx += dr
        dx_vec.append(dr)
        dri = (np.ceil(abs(dr))).astype(int)
        img2_shifted = ndi.shift(img2_shifted, [dr[1], dr[0]])
        if dri[0] > 0 and dri[1] > 0:
            img1 = img1[dri[1]:-dri[1], dri[0]:-dri[0]]
            img2_shifted = img2_shifted[dri[1]:-dri[1], dri[0]:-dri[0]]

    return cumul_dx, dx_vec, img2_shifted

def test_2d(sigma = None, delta = None, img1 = None):
    # ==============================#
    # Tests 2d registration        #
    # ==============================#
    # img1 = cv2.imread(img_path, 0)
    # generate random image

    if img1 is None:
        img1 = np.random.rand(500, 500)

    img1  = cv2.imread(img_path, 0)
    if delta is None:
        delta = int(input('Enter max delta: '))
        deltax = 2*delta*np.random.rand()-delta
        deltay = 2*delta*np.random.rand()-delta
    else:
        deltax = delta
        deltay = delta
    if delta is None:
        sigma = int(input('Enter sigma: '))



    if sigma<9:
        print('Sigma>7 is recommended')

    # smooth image
    img1 = gaussian_filter(img1, sigma=sigma)
    img2 = ndi.shift(img1, (deltay, deltax))

    print('Real:',deltax, deltay)

    dr = solve_2d(img1, img2)
    print('single: ', dr[0])

    dr = solve_iter_2d(img1, img2)
    print('iterative', dr[0])

    dr = register_multiscale_2d(img1, img2)
    print('multiscale', dr[0])
