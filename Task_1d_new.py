import cv2
import argparse
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d

# !!! INPUT FILES - Change with your file PATH !!!
img_path = 'img2.jpeg'


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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




def solve_iter_1d(sig1, sig2, max_num_iter=100):
    # ==============================#
    # Caculates registration with lsq
    # Output:
    #   dr -  shift
    #   sig2 - shifted signal based on dr
    # ==============================#

    # calculate derivative
    sig1 = sig1.astype(float)
    sig2 = sig2.astype(float)

    # initialize
    cumul_dx = 0
    dx_vec = []
    error_vec = []
    sig2_shifted = sig2
    i = 0
    while i < (max_num_iter):

        # cut signal to overlap zone, based on max shift of 15% of image size
        if i == 0:
            shift = np.ceil(len(sig1) * 0.25).astype(int)
            sig1 = sig1[shift:-shift]
            sig2 = sig2[shift:-shift]
            sig2_shifted = sig2

        dr, sig2_shifted = solve_1d(sig1, sig2_shifted)

        cumul_dx += dr
        dx_vec.append(dr)

        # dri is the shift in each dimension integerm used to crop img1 and img2
        dri = np.ceil(abs(dr)).astype(int)


        if dri > 0:
            # crop sig1 and sig2
            sig1 = sig1[dri:-dri]
            sig2 = sig2[dri:-dri]
            sig2_shifted = sig2_shifted[dri:-dri]

        # check convergence after minimal number of iterations
        if i > 1350:
            # dr doesn't change much, converged
            if abs(dr) < 0.005:
                print('converged at iteration: ', i, ' with dr: ', dr)
                break

        i += 1

    if i == max_num_iter - 1:
        print('did not converge')

    return cumul_dx, sig2_shifted, dx_vec


def get_downscaled_sig(sig, scale, stride=0):
    # ==============================#
    # Downscales image by scale    #
    # ==============================#
    if scale < 1:
        # gaussian filter img
        sig_downscaled = gaussian_filter1d(sig, sigma=scale)
    else:
        sig_downscaled = sig
    # assert stride<1/scale, 'Stride is too large'
    # return img_downscaled[stridex::int(1 / scale), stridey::int(1 / scale)]
    return sig_downscaled


def register_multiscale_1d(sig1, sig2, scale_list = [0.25,0.5,1]):

    cumul_dx = 0
    dx_vec = []
    sig2_shifted = sig2

    for scale in scale_list:
        sig1_downscaled = get_downscaled_sig(sig1, scale)
        sig2_downscaled = get_downscaled_sig(sig2_shifted, scale)


        # match lengths of sig1 and sig2
        sig1_downscaled = sig1_downscaled[:np.min([len(sig1_downscaled), len(sig2_downscaled)])]
        sig2_downscaled = sig2_downscaled[:np.min([len(sig1_downscaled), len(sig2_downscaled)])]

        dr = solve_iter_1d(sig1_downscaled, sig2_downscaled)[0]
        # dr = dr[0]

        # dr = dr / scale

        # update vecotrs
        cumul_dx += dr
        dx_vec.append(dr)

        # cut img2 to prevent edge effects
        # dri = (np.ceil(abs(dr))).astype(int)
        # img2_shifted = ndi.shift(img2_shifted, [dr[1], dr[0]])

        dri = (np.ceil(abs(cumul_dx))).astype(int)
        sig2_shifted = ndi.shift(sig2, cumul_dx)

        if dri > 0:
            sig1 = sig1[dri:-dri]
            sig2 = sig2[dri:-dri]
            sig2_shifted = sig2_shifted[dri:-dri]

    return cumul_dx, dx_vec, sig2_shifted

