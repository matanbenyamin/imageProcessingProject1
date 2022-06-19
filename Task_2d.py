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

    # initialize
    cumul_dx = dr
    dx_vec = []
    error_vec = []
    img2_shifted = img2
    i = 0
    while i < (max_num_iter):

        # cut images to overlap zone, based on max shift of 15% of image size
        if i == 0:
            shift = np.ceil(np.min(img1.shape) * 0.25).astype(int)
            img1 = img1[shift:-shift, shift:-shift]
            img2_shifted = img2_shifted[shift:-shift, shift:-shift]
            img2 = img2[shift:-shift, shift:-shift]

        dr, img2_shifted = solve_2d(img1, img2_shifted)

        cumul_dx += dr
        dx_vec.append(dr)

        # dri is the shift in each dimension integerm used to crop img1 and img2
        dri = (np.ceil([abs(2 * x) for x in dr])).astype(int)

        # dri = (np.ceil(abs(cumul_dx))).astype(int)
        # img2_shifted = ndi.shift(img2, [cumul_dx[1], cumul_dx[0]])

        if dri[0] > 0 and dri[1] > 0:
            img1 = img1[dri[1]:-dri[1], dri[0]:-dri[0]]
            img2_shifted = img2_shifted[dri[1]:-dri[1], dri[0]:-dri[0]]
            img2 = img2[dri[1]:-dri[1], dri[0]:-dri[0]]

        # normalized_error = np.sum(np.abs(img1 - img2_shifted)) / np.sum(np.abs(img1))
        # error_vec.append(normalized_error)

        # check convergence after minimal number of iterations
        if i > 1350:
            # dr doesn't change much, converged
            if np.max(np.abs(dr)) < 0.005:
                # print('converged at iteration: ', i, ' with dr: ', dr)
                break

        i += 1

    if i == max_num_iter - 1:
        print('did not converge')

    return cumul_dx, img2_shifted, dx_vec


def get_downscaled_img_2d(img, scale, stridex=0, stridey=0):
    # ==============================#
    # Downscales image by scale    #
    # ==============================#
    if scale < 1:
        # gaussian filter img
        img_downscaled = gaussian_filter(img, 2 / scale)
    else:
        img_downscaled = img
    # assert stride<1/scale, 'Stride is too large'
    # return img_downscaled[stridex::int(1 / scale), stridey::int(1 / scale)]
    return img_downscaled


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

        if dr[1] > 0.25 * img1.shape[0] or dr[0] > 0.25 * img1.shape[0]:
            print('calculated delta is quite large, consider reducing scale, increase sigma or increase image size')

        # dr = dr / scale

        # update vecotrs
        cumul_dx += dr
        dx_vec.append(dr)

        # cut img2 to prevent edge effects
        dri = (np.ceil(abs(dr))).astype(int)
        img2_shifted = ndi.shift(img2_shifted, [dr[1], dr[0]])

        dri = (np.ceil(abs(cumul_dx))).astype(int)
        img2_shifted = ndi.shift(img2, [cumul_dx[1], cumul_dx[0]])

        if dri[0] * dri[1] > 0 and np.max(dri) < 0.25 * np.min(img1.shape):
            img1 = img1[dri[1]:-dri[1], dri[0]:-dri[0]]
            img2_shifted = img2_shifted[dri[1]:-dri[1], dri[0]:-dri[0]]
            img2 = img2[dri[1]:-dri[1], dri[0]:-dri[0]]
        else:
            # dr is too large and completely outside of img1
            img2_shifted = img2
            cumul_dx -= dr
            dx_vec.pop()
            prev_err = np.linalg.norm(img1 / np.max(img1) - img2_shifted / np.max(img2_shifted))
            continue

        # prevent non useful iterations that do not get the images closer and may cause divergence
        if np.min(img1.shape) > 0:
            curr_err = np.linalg.norm(img1 / np.max(img1) - img2_shifted / np.max(img2_shifted))
            if scale > scale_list[0]:
                if curr_err > 1 * prev_err:
                    img2_shifted = img2
                    cumul_dx -= dr
                    dx_vec.pop()
                    print('skip scale: ', scale, ' because of error: ', curr_err)
        else:
            curr_err = 1e5
        prev_err = curr_err

    return cumul_dx, dx_vec, img2_shifted


def test_2d(sigma=None, delta=None, img1=None):
    # ==============================#
    # Tests 2d registration        #
    # ==============================#
    # img1 = cv2.imread(img_path, 0)
    # generate random image

    if img1 is None:
        img1 = np.random.rand(500, 500)

    img1 = cv2.imread(img_path, 0)
    if delta is None:
        delta = int(input('Enter max delta: '))
        deltax = 2 * delta * np.random.rand() - delta
        deltay = 2 * delta * np.random.rand() - delta
    else:
        deltax = delta
        deltay = delta
    if delta is None:
        sigma = int(input('Enter sigma: '))

    if sigma < 9:
        print('Sigma>7 is recommended')

    # smooth image
    img1 = gaussian_filter(img1, sigma=sigma)
    img2 = ndi.shift(img1, (deltay, deltax))

    print('Real:', deltax, deltay)

    dr = solve_2d(img1, img2)
    print('single: ', dr[0])

    dr = solve_iter_2d(img1, img2)
    print('iterative', dr[0])

    dr = register_multiscale_2d(img1, img2)
    print('multiscale', dr[0])


def sigma_optimizer_2d(img1, img2, method, sigma_list=list(range(1, 75, 6))):
    # ==============================#
    # Optimizes sigma for 2d        #
    # ==============================#
    # Input:
    #   img1 - image 1
    #   img2 - image 2
    #   method - method to use for registration
    #       'iterative' - iterative method
    #       'multiscale' - multiscale method
    #   sigma_list - list of sigmas to test

    # Output:
    #   dr for optimal sigma
    #   sigma - optimal sigma
    # ==============================#

    # assumption: better registration results in lower error

    error_vec = []
    dr_list = []
    for sigma in sigma_list:
        im1 = gaussian_filter(img1, sigma=sigma)
        im2 = gaussian_filter(img2, sigma=sigma)

        if method == 'iterative':
            dr = solve_iter_2d(im1, im2)
        elif method == 'multiscale':
            dr = register_multiscale_2d(im1, im2)
        dr_list.append(dr[0])

        im2_t = ndi.shift(img2, dr[0])
        dri = (np.ceil(abs(dr[0]))).astype(int)
        im1_t = img1[dri[1]:-dri[1], dri[0]:-dri[0]]
        im2_t = im2_t[dri[1]:-dri[1], dri[0]:-dri[0]]

        if np.max(dr[0]) < 0.25 * np.min(img1.shape):
            error = np.nanmedian(np.abs(im1_t - im2_t))
        else:
            error = 1e8

        if error == 0:
            error = 1e5
        # print('sigma: ', sigma, ' error: ', error, ' dr: ', dr[0])
        error_vec.append(error)

    return dr_list[np.nanargmin(error_vec)], sigma_list[np.nanargmin(error_vec)]
