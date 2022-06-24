import cv2
import argparse
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import gaussian_filter

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


def solve_iter_2d(img1, img2, max_num_iter=250):
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

    # save original copies since eahc shift deletes some of the image
    oimg1 = img1.copy()
    oimg2 = img2.copy()

    # initialize
    cumul_dx = dr
    dx_vec = []
    error_vec = []
    img2_shifted = img2
    i = 0
    while i < (max_num_iter):

        # cut images to overlap zone, based on max shift of 15% of image size
        # if i == 0:
        #     shift = np.ceil(np.min(img1.shape) * 0.15).astype(int)
        #     img1 = img1[shift:-shift, shift:-shift]
        #     img2_shifted = img2_shifted[shift:-shift, shift:-shift]
        #     img2 = img2[shift:-shift, shift:-shift]

        dr = solve_2d(img1, img2_shifted)
        dr = dr[0]
        cumul_dx += dr
        dx_vec.append(dr)

        # dri is the shift in each dimension integerm used to crop img1 and img2
        dri = (np.ceil(abs(4*cumul_dx))).astype(int)

        # get the original copies in the original size to prevent the image from getting smaller
        img1 = oimg1
        img2 = oimg2
        img2_shifted = ndi.shift(img2, [cumul_dx[1], cumul_dx[0]])

        if dri[0] > 0 and dri[1] > 0:
            img1 = img1[dri[1]:-dri[1], dri[0]:-dri[0]]
            img2_shifted = img2_shifted[dri[1]:-dri[1], dri[0]:-dri[0]]
            img2 = img2[dri[1]:-dri[1], dri[0]:-dri[0]]


        # break if img is too small because it drags bad calculation
        if np.min(img1.shape) < 70:
            break

        # check convergence after minimal number of iterations
        if i > 150:
            # dr doesn't change much, converged
            if np.max(np.abs(dr)) < 0.005:
                # print('converged at iteration: ', i, ' with dr: ', dr)
                break

        i += 1

    if i == max_num_iter:
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
    # gives better results to keep the image in the original size and just adjst the sigma by scale
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
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    # save original copies since eahc shift deletes some of the image
    oimg1 = img1.copy()
    oimg2 = img2.copy()
    cumul_dx = 0
    dx_vec = []
    img2_shifted = img2

    for scale in scale_list:
        img1_downscaled = get_downscaled_img_2d(img1, scale)
        img2_downscaled = get_downscaled_img_2d(img2_shifted, scale)

        # match lengths of img1 and img2
        img1_downscaled = img1_downscaled[:np.min([len(img1_downscaled), len(img2_downscaled)])]
        img2_downscaled = img2_downscaled[:np.min([len(img1_downscaled), len(img2_downscaled)])]

        # continue if img is too small because it drags bad calculation
        if np.min(img1_downscaled.shape) < 150:
            continue

        dr = solve_iter_2d(img1_downscaled, img2_downscaled)
        dr = dr[0]

        if dr[1] > 0.25 * img1.shape[0] or dr[0] > 0.25 * img1.shape[0]:
            print('calculated delta is quite large, consider reducing scale, increase sigma or increase image size')

        # gives better results to keep the image in the original size and just adjst the sigma by scale
        # dr = dr / scale

        # update vecotrs
        cumul_dx += dr
        dx_vec.append(dr)

        # cut img2 to prevent edge effects
        dri = (np.ceil(abs(2*cumul_dx))).astype(int)

        # get the original copies in the original size to prevent the image from getting smaller
        img1 = oimg1
        img2 = oimg2

        # ==== shift the image with bilinear interpolation
        # img2_shifted = shift_img_bilinear(img2, cumul_dx[1], cumul_dx[0])
        # === Results are the same but it is much faster with ndi.shift
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
        # if np.min(img1.shape) > 0:
        #     curr_err = np.linalg.norm(img1 / np.max(img1) - img2_shifted / np.max(img2_shifted))
        #     if scale > scale_list[0]:
        #         if curr_err > 1 * prev_err:
        #             img2_shifted = img2
        #             cumul_dx -= dr
        #             dx_vec.pop()
        #             print('skip scale: ', scale, ' because of error: ', curr_err)
        # else:
        #     curr_err = 1e5
        # prev_err = curr_err

    return cumul_dx, dx_vec, img2_shifted


def bilinear_subpixel_interpolation(img, x, y):
    # ==============================#
    # Bilinear interpolation        #
    # ==============================#
    x = np.round(x,3)
    y = np.round(y,3)
    x = np.clip(x, 0, img.shape[1]-1)
    y = np.clip(y, 0, img.shape[0]-1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    x0 = np.clip(x0, 0, img.shape[1]-1)
    y0 = np.clip(y0, 0, img.shape[0]-1)
    x1 = np.clip(x1, 0, img.shape[1]-1)
    y1 = np.clip(y1, 0, img.shape[0]-1)
    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def shift_img_bilinear(img, dx, dy):
    # ==============================#
    # Bilinear interpolation        #
    # ==============================#
    img_shifted = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_shifted[i, j] = bilinear_subpixel_interpolation(img, j - dx, i - dy)

    return img_shifted

# import plotly.graph_objects as go
# import plotly.express as px
# import plotly.io as pio
# pio.renderers.default = "browser"
# y = np.sum(shift_img_bilinear(img1[:50,:50],-0.5,-0.5), axis=0)
# y2 = np.sum(ndi.shift(img1[:50,:50],(-0.5,-0.5)), axis=0)
# fig = px.line(y=y)
# fig.add_trace(go.Scatter(x=np.arange(len(y)), y=y2, name='shifted'))
# fig.show()
#

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
