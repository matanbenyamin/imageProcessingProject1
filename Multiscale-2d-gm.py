from Task_2d import *
import sys

# load npz file
img1 = cv2.imread(sys.argv[1], 0)
img2 = cv2.imread(sys.argv[2], 0)


# validate image lengths
img1 = img1[:np.min([img1.shape[0], img2.shape[0]]), :np.min([img1.shape[1], img2.shape[1]])]
img2 = img2[:np.min([img1.shape[0], img2.shape[0]]), :np.min([img1.shape[1], img2.shape[1]])]

#save original images
oimg1 = img1
oimg2 = img2




if len(sys.argv) > 3 and sys.argv[3] == 'auto':
    dx = sigma_optimizer_2d(oimg1,oimg2 , method = 'multiscale')
    print(np.round(-dx[0], 3))
    print('sigma', np.round(dx[1], 3))
else:
    # solve iterative
    dx = register_multiscale_2d(img1, img2, scale_list=[ 0.25, 0.5, 1])[0]
    print(np.round(-dx, 3))

