from Task_2d import *
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



# solve multiscale
dx = register_multiscale_2d(img1, img2,scale_list = [0.25, 0.5, 1])[0]
print('dx, dy (multiscale):', np.round(dx, 3))
