
img1 = np.random.rand(500, 500)
deltax = 75
deltay = 75
sigma = 35


deltax = -105
deltay = 0.958
sigma = 1


img1 = gaussian_filter(img1, sigma=sigma)
img2 = ndi.shift(img1, (deltay, deltax))


dr = solve_2d(img1, img2)
print('single: ', dr[0])

dr = solve_iter_2d(img1, img2)
print('iterative', dr[0])

dr = register_multiscale_2d(img1, img2)
print('multiscale', dr[0])


import plotly.express as px
import plotly.io as pio


dr = solve_iter_2d(img1, img2)
print('iterative', dr[0])
pio.renderers.default = "browser"
fig = px.line(dr[2])
fig.show()
