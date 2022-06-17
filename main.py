from Task_2d import *
from Task_1d import *


img1 = np.random.rand(500, 500)
deltax = 75
deltay = 75
sigma = 35

img1 = np.random.rand(500, 500)



img1 = cv2.imread(img_path, 0)
#extrapolate image
# img1 = cv2.copyMakeBorder(img1, 2*deltax, 2*deltax, 2*deltay, 2*deltay, cv2.BORDER_REPLICATE)

deltax = 17
deltay = 2.5
sigma = 5


# img1 = gaussian_filter(img1, sigma=sigma)
img2 = ndi.shift(img1, (deltay, deltax))

from PIL import Image
Image.fromarray((img1 * 255).astype(np.uint8)).save('im1.bmp')
Image.fromarray((img2 * 255).astype(np.uint8)).save('im2.bmp')


img1 = cv2.imread('im1.bmp', 0)
img1 = img1[:-2, :-2]
img2 = cv2.imread('im2.bmp', 0)

dr = solve_2d(img1, img2)
print('single: ', dr[0])
dr = solve_iter_2d(img1, img2)
print('iterative', dr[0])
dr = register_multiscale_2d(img1, img2,scale_list = [0.25, 0.5, 1])
print('multiscale', dr[0])




import plotly.express as px
import plotly.io as pio


dr = solve_iter_2d(img1, img2)
print('iterative', dr[0])
pio.renderers.default = "browser"
fig = px.line(dr[2])
fig.show()


# test 1d
sigma = 25
delta = 12.45
sig1 = np.random.rand(500)
sig1 = gaussian_filter1d(sig1, sigma=sigma)
sig2 = ndi.shift(sig1, delta)


# sig1 = np.load('x1.npy')
# sig2 = np.load('x2.npy')


dr = solve_1d(sig1, sig2)
print('single: ', dr[0])
dr = solve_iter_1d(sig1, sig2)
print('iterative', dr[0])
dr = register_multiscale_1d(sig1, sig2)
print('multiscale', dr[0])