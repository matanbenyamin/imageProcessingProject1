import argparse
import numpy as np
import plotly.io as pio
import scipy.ndimage as ndi

pio.renderers.default = "browser"
from scipy.ndimage import gaussian_filter1d

# !!! INPUT FILES - Change with your files PATH !!!
x1file = 'x1.npy'
x2file = 'x2.npy'
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


def solve_iter(sig1, sig2, max_num_iter=100):
    # ==============================#
    # Caculates registration with lsq
    # Output:
    #   cumul_dx - shift after convergence
    #   sig2 - shifted signal based on cumul_dx
    #   dx_vec - vector of shifts over convergence

    cumul_dx = 0
    dx_vec = []
    x2 = sig2

    for i in range(max_num_iter):
        dr, sig2_shifted = solve_1d(sig1, x2)
        cumul_dx += dr
        dx_vec.append(dr)
        dri = int(np.ceil(abs(dr)))
        # cut off the edges to avoid edge effects of the shift
        x2 = sig2_shifted[dri:-dri]
        sig1 = sig1[dri:-dri]

        # stop automatically if dx_vec doesnt change
        if i > 1:
            if abs(dx_vec[i - 1] - dx_vec[i - 2]) < 0.01:
                # print('Converged after {} iterations'.format(i))
                break

    return cumul_dx, sig2_shifted, dx_vec


def get_downscaled_sig(sig, scale, stride=0):
    # ==============================#
    # Downscales signal by scale    #
    # ==============================#
    if scale < 1:
        sig_downscaled = gaussian_filter1d(sig, scale)
    else:
        sig_downscaled = sig
    # assert stride<1/scale, 'Stride is too large'
    return sig_downscaled[stride::int(1 / scale)]


def register_multiscale(sig1, sig2, scale_list):
    # ==============================#
    # Caculates registration with lsq
    # with multiscale registration
    # Output:
    cumul_dx = 0
    dx_vec = []
    sig2_shifted = sig2
    scale_list = [0.25, 0.5, 1]

    for scale in scale_list:
        sig1_downscaled = get_downscaled_sig(sig1, scale)
        sig2_downscaled = get_downscaled_sig(sig2_shifted, scale)

        # match lengths of sig1 and sig2
        sig1_downscaled = sig1_downscaled[:np.min([len(sig1_downscaled), len(sig2_downscaled)])]
        sig2_downscaled = sig2_downscaled[:np.min([len(sig1_downscaled), len(sig2_downscaled)])]

        dr, abc = solve_1d(sig1_downscaled, sig2_downscaled)
        dr, abd, cde = solve_iter(sig1_downscaled, sig2_downscaled)

        dr = dr / scale
        cumul_dx += dr
        dx_vec.append(dr)
        dri = int(dr)

        # cut off the edges to avoid edge effects of the shift
        sig2_shifted = ndi.shift(sig2_shifted, dr)
        sig2_shifted = sig2_shifted[dri:-dri]
        sig1 = sig1[dri:-dri]

        # shorten signal to the same length as sig2_shifted
        sig1 = sig1[:np.min([len(sig1), len(sig2_shifted)])]
        sig2_shifted = sig2_shifted[:np.min([len(sig1), len(sig2_shifted)])]

    return cumul_dx, sig2_shifted, dx_vec

def test_1d():
    # ==============================#
    # Tests 1d registration        #
    # ==============================#
    sig1 = np.load(x1file)
    sig2 = np.load(x2file)

    sig1 = np.random.rand(500)
    delta = int()

x1 = np.load(x1file)
x2 = np.load(x2file)

sigma = 3
x1 = gaussian_filter1d(x1, sigma=sigma)
x2 = gaussian_filter1d(x2, sigma=sigma)

dr, sig2_shifted = solve_1d(x1, x2)
dr_iter, sig2_shifted, dx_vec = solve_iter(x1, x2, max_num_iter=525)
dr_multiscale, sig2_shifted, dx_vec = register_multiscale(x1, x2, [0.25, 0.5, 1])

print('signle:', np.round(dr, 3))
print('iterations', np.round(dr_iter, 3))
print('multiscale', np.round(dr_multiscale, 3))
print()

# ==============================-#
# ===== Solve for synthetic signal
# ==============================#

x = 200 * np.random.uniform(0, 1, 1150)
sigma = 95
x = np.convolve(x, np.ones(sigma) / sigma, mode='same')

# shift signal
Downsample = 5
Delta = 12
sig1 = x[0::Downsample]
sig2 = x[Delta::Downsample]
dx = Delta / Downsample

# shorten signal
sig1 = sig1[:np.min([len(sig1), len(sig2)])]
sig2 = sig2[:np.min([len(sig1), len(sig2)])]

sigma = 1

# higher sigma will help convergence but will end un in less accurate result
# large shifts, where linear assumption is not valid, will require relatively high sigma

smooth_sig1 = gaussian_filter1d(sig1, sigma=sigma)
smooth_sig2 = gaussian_filter1d(sig2, sigma=sigma)

smooth_sig1 = smooth_sig1[sigma:-sigma]
smooth_sig2 = smooth_sig2[sigma:-sigma]

dr, sig2_shifted = solve_1d(smooth_sig1, smooth_sig2)
dr_iter, sig2_shifted, dx_vec = solve_iter(smooth_sig1, smooth_sig2, max_num_iter=525)
dr_multiscale, sig2_shifted, dx_vec = register_multiscale(smooth_sig1, smooth_sig2, [0.25, 0.5, 1])

print('original:', np.round(dx, 3))
print('signle:', np.round(dr, 3))
print('iterations', np.round(dr_iter, 3))
print('multiscale', np.round(dr_multiscale, 3))

# plots for 1d registration
# fig = px.line(dx_vec)
# fig.show()
#
# fig = px.line(smooth_sig1)
# fig.add_trace(go.Scatter(y=sig2_shifted))
# fig.show()
