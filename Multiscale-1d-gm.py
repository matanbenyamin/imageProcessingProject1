from Task_1d import *
import numpy as np
import sys

# data was saved like this:
# np.savez_compressed('data1d.npz',x=x, Downsample=Downsample, Delta=Delta, x1=x1, x2=x2)


# load npz file
sig1 = np.load(sys.argv[1])['x1']
sig2 = np.load(sys.argv[2])['x2']

# validate signal lengths
sig1 = sig1[:np.min([len(sig1), len(sig2)])]
sig2 = sig2[:np.min([len(sig1), len(sig2)])]

osig1 = sig1
osig2 = sig2

# smooth
sigma = 9
sig1 = gaussian_filter1d(sig1, sigma=sigma)
sig2 = gaussian_filter1d(sig2, sigma=sigma)




if len(sys.argv) > 3 and sys.argv[3] == 'auto':
    dx = sigma_optimizer_1d(osig1, osig2, method = 'multiscale')
    print(np.round(-dx[0], 3))
    print('sigma', np.round(dx[1], 3))
else:
    # solve iterative
    dx = register_multiscale_1d(sig1, sig2)[0]
    print(np.round(-dx, 3))
