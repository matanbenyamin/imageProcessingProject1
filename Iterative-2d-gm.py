from Task_2d import *
import numpy as np
import sys

# load npz file
img1 = cv2.imread(sys.argv[1], 0)
img2 = cv2.imread(sys.argv[2], 0)

# smooth
sigma = 3
img1 = gaussian_filter(img1, sigma=sigma)
img2 = gaussian_filter(img2, sigma=sigma)


# validate image lengths
img1 = img1[:np.min([img1.shape[0], img2.shape[0]]), :np.min([img1.shape[1], img2.shape[1]])]
img2 = img2[:np.min([img1.shape[0], img2.shape[0]]), :np.min([img1.shape[1], img2.shape[1]])]


dx = solve_iter_2d(img1, img2, max_num_iter=125)[0]
print('dx, dy (iterative):', np.round(dx, 3))

