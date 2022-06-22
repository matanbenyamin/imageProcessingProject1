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
    # Input:
    #   sig1 - signal 1

    # Caculates registration with lsq
    # Output:
    #   dr -  shift in each dimensiotn
    #   sig2 - shifted signal based on dr
    # ==============================#

    # calculate derivative
    A = (sig1[2:] - sig1[:-2]) / 2

    # add zeros colum to use pinv
    A = np.c_[A, np.zeros_like(A)]

    #realize least squares
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
    # relaizes iterative least squares using solve_1d
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

        # store dr
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
        if i > 135:
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
    # Downscales image by scale     #
    # This actually just smoothes each scale by a gaussian filter
    # ==============================#
    if scale < 1:
        # gaussian filter img
        sig_downscaled = gaussian_filter1d(sig, sigma=scale)
    else:
        sig_downscaled = sig
    # assert stride<1/scale, 'Stride is too large'
    # return sig_downscaled[stride::int(1/scale)]
    return sig_downscaled


def register_multiscale_1d(sig1, sig2, scale_list = [0.25,0.5,1]):
    # ==============================#
    # performs registration of sig1 to sig2
    # in several scales (increasing)
    # Input:
    #   sig1 - signal 1
    #   sig2 - signal 2
    #   scale_list - list of scales to downscale
    # Output:
    #   dr - list of shifts in each dimension
    #   sig2_shifted - list of shifted signals
    # ==============================#


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

        # dr = dr / scale

        # update vecotrs
        cumul_dx += dr
        dx_vec.append(dr)

        # cut img2 to prevent edge effects
        dri = (np.ceil(abs(cumul_dx))).astype(int)
        sig2_shifted = ndi.shift(sig2, cumul_dx)

        if dri > 0:
            sig1 = sig1[dri:-dri]
            sig2 = sig2[dri:-dri]
            sig2_shifted = sig2_shifted[dri:-dri]

    return cumul_dx, dx_vec, sig2_shifted

def subsample_shift_by_interpolation(sig, dx):
    # ==============================#
    # shifts signal by dx using linear interpolation
    # Input:
    #   sig - signal
    #   dx - shift
    # Output:
    #   sig_shifted - shifted signal
    # ==============================#
    sig_shifted = interp1d(np.arange(len(sig)), sig, kind='linear')(np.arange(len(sig)) + dx)

    return sig_shifted

def sigma_optimizer_1d(sig1, sig2, method, sigma_list = list(range(1,75,4))):
    # ==============================#
    # Optimizes sigma for 1d        #
    # ==============================#
    # Input:
    #   sig1 - signal 1
    #   sig2 - signal 2
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
        x1 = gaussian_filter1d(sig1, sigma=sigma)
        x2 = gaussian_filter1d(sig2, sigma=sigma)

        if method == 'iterative':
            dr = solve_iter_1d(x1, x2)
        elif method == 'multiscale':
            dr = register_multiscale_1d(x1, x2)

        dr_list.append(dr[0])
        x2 = ndi.shift(sig2, dr[0])
        dri = int((np.ceil(abs(dr[0]))))
        sig1_t = sig1[dri:-dri]
        sig2_t = x2[dri:-dri]
        error_vec.append(np.nanmedian(np.abs(sig1_t - sig2_t)))
        # print('sigma: ', sigma, ' error: ', error_vec[-1], ' dr: ', dr_list[-1])
    return dr_list[np.nanargmin(error_vec)], sigma_list[np.nanargmin(error_vec)]