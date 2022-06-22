from Task_2d import *
from Task_1d import *
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

# =============================================================================
# This test
# outputs analysis of shift estimation, for different sigma and shifts


# 2d test

# Performance analysis
oimg1= np.random.rand(500, 500)

# here yo may enter your own image
#img path = '../data/images/lena.png'
#oimg1 = cv2.imread(img_path, 0)

sigma = np.arange(1, 45,5)
delta = np.arange(1, (0.25*np.min(oimg1.shape)).astype(int),15)

# Iterative
error_iterative = np.zeros((len(sigma), len(delta)))
for i, s in enumerate(sigma):
    for j, d in enumerate(delta):
        img1 = gaussian_filter(oimg1, sigma=s)
        img2 = ndi.shift(img1, (d, d))
        dr = solve_iter_2d(img1, img2)
        error_iterative[i, j] = np.linalg.norm(np.abs(-dr[0]-[delta[j], delta[j]]))

# Multiscale
error_ms = np.zeros((len(sigma), len(delta)))
for i, s in enumerate(sigma):
    for j, d in enumerate(delta):
        img1 = gaussian_filter(oimg1, sigma=s)
        img2 = ndi.shift(img1, (d,d))
        dr = register_multiscale_2d(img1, img2, scale_list=[0.25, 0.5, 1])
        error_ms[i,j] = np.linalg.norm(np.abs(-dr[0]-[delta[j], delta[j]]))



# plot error for both methods seperately in two subplotss


fig = make_subplots(rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=("Iterative", "Multiscale"))


fig.add_trace(go.Heatmap(
    z=error_iterative,
    x=delta,
    y=sigma,
    colorscale='Viridis',
    showscale=True,
    colorbar=dict(title='Error')), row=1, col=1)
fig.add_trace(go.Heatmap(
    z=error_ms,
    x=delta,
    y=sigma,
    colorscale='Viridis',
    showscale=True,
    colorbar=dict(title='Error')), row=2, col=1)
fig.update_yaxes(title_text="Sigma", row=1, col=1)
fig.update_yaxes(title_text="Sigma", row=2, col=1)
fig.update_xaxes(title_text="Delta", row=1, col=1)
fig.update_xaxes(title_text="Delta", row=2, col=1)
fig.update_layout(title_text="Error between real and estimated dx for Iterative and Multiscale")
fig.show()
fig.write_html('2D.html')




# =============================================================================
# 1d test

# Performance analysis
osig1 = np.random.rand(500)

# here yo may enter your own signal
#sig path = '../data/images/lena.png'
# sig1 = np.load(sig_path)['x1']

sigma = np.arange(1, 45,5)
delta = np.arange(1, int(0.25*len(osig1)),15)


# Iterative
error_iterative = np.zeros((len(sigma), len(delta)))
for i, s in enumerate(sigma):
    for j, d in enumerate(delta):
        sig1 = gaussian_filter1d(osig1, sigma=s)
        sig2 = ndi.shift(sig1, d)
        dr = solve_iter_1d(sig1, sig2)
        error_iterative[i, j] = np.linalg.norm(np.abs(-dr[0]-[delta[j]]))

# Multiscale
error_ms = np.zeros((len(sigma), len(delta)))
for i, s in enumerate(sigma):
    for j, d in enumerate(delta):
        sig1 = gaussian_filter1d(osig1, sigma=s)
        sig2 = ndi.shift(sig1, d)
        dr = register_multiscale_1d(sig1, sig2, scale_list=[0.25, 0.5, 1])
        error_ms[i,j] = np.linalg.norm(np.abs(-dr[0]-[delta[j]]))

# plot error for both methods seperately in two subplotss


fig = make_subplots(rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=("Iterative", "Multiscale"))


fig.add_trace(go.Heatmap(
    z=error_iterative,
    x=delta,
    y=sigma,
    colorscale='Viridis',
    showscale=True,
    colorbar=dict(title='Error')), row=1, col=1)
fig.add_trace(go.Heatmap(
    z=error_ms,
    x=delta,
    y=sigma,
    colorscale='Viridis',
    showscale=True,
    colorbar=dict(title='Error')), row=2, col=1)
fig.update_yaxes(title_text="Sigma", row=1, col=1)
fig.update_yaxes(title_text="Sigma", row=2, col=1)
fig.update_xaxes(title_text="Delta", row=1, col=1)
fig.update_xaxes(title_text="Delta", row=2, col=1)
fig.update_layout(title_text="Error between real and estimated dx for Iterative and Multiscale")
fig.show()
fig.write_html("1D.html")


