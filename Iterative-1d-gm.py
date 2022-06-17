from Task_1d import *
import numpy as np
import sys

# data was saved like this:
# np.savez_compressed('data1d.npz',x=x, Downsample=Downsample, Delta=Delta, x1=x1, x2=x2)


# load npz file
sig1 = np.load(sys.argv[1])['x1']
sig2 = np.load(sys.argv[2])['x2']

# smooth
sigma = 1
sig1 = gaussian_filter1d(sig1, sigma=sigma)
sig2 = gaussian_filter1d(sig2, sigma=sigma)

# validate signal lengths
sig1 = sig1[:np.min([len(sig1), len(sig2)])]
sig2 = sig2[:np.min([len(sig1), len(sig2)])]

# solve iterative
dx = solve_iter_1d(sig1, sig2, max_num_iter=125)[0]
print('dx (iterative registration):', np.round(dx, 3))
