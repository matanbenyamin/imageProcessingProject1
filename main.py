
img1 = np.random.rand(500, 500)
deltax = 75
deltay = 75
sigma = 35

img1 = np.random.rand(500, 500)



img1 = cv2.imread(img_path, 0)
#extrapolate image
# img1 = cv2.copyMakeBorder(img1, 2*deltax, 2*deltax, 2*deltay, 2*deltay, cv2.BORDER_REPLICATE)

deltax = 15
deltay = 5.258
sigma = 3


img1 = gaussian_filter(img1, sigma=sigma)
img2 = ndi.shift(img1, (deltay, deltax))

dr = solve_2d(img1, img2)
print('single: ', dr[0])
dr = solve_iter_2d(img1, img2)
print('iterative', dr[0])
dr = register_multiscale_2d(img1, img2)
print('multiscale', dr[0])

fig = px.imshow(img1, title='img1')
fig.show()
fig = px.imshow(dr[2], title='img2')
fig.show()


import plotly.express as px
import plotly.io as pio


dr = solve_iter_2d(img1, img2)
print('iterative', dr[0])
pio.renderers.default = "browser"
fig = px.line(dr[2])
fig.show()
