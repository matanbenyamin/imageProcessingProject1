from scipy.ndimage import gaussian_filter


def solve_2d(im1, im2):
    # ==============================#
    # Caculates registration with lsq
    # Output:
    #   dr -  shift in each dimensiotn
    #   im2 - shifted image based on dr
    # ==============================#

    imx = im1[1:, 1:] - im1[1:, :-1]
    imy = im1[1:, 1:] - im1[:-1, 1:]

    A = np.transpose([imx.flatten(), imy.flatten()])
    At = np.transpose(A)

    y = im2[1:, 1:] - im1[1:, 1:]
    y = y.flatten()

    s = np.linalg.pinv(At @ A)
    dr = s @ (At @ y)
    im2_shifted = ndi.shift(im2, (dr[1],dr[0]))

    return dr, im2_shifted

def solve_iter_2d(img1, img2, max_num_iter = 1500):
    # ==============================#
    # Calculates registration with lsq
    # Input:
    #   img1 - image to be registered
    #   img2 - image to be registered to
    #   max_num_iter - maximum number of iterations(default: 1500)
    # Output:
    #   dr -  shift in each dimensiotn
    #   im2_shifted - shifted image based on dr
    #   dx_vec - vector of shifts over convergence
    # ==============================#

    cumul_dx = 0
    dx_vec = []
    img2_shifted = img2
    for i in range(150):
        dr, img2_shifted = solve_2d(img1, img2_shifted)
        cumul_dx += dr
        dx_vec.append(dr)
        # prevent overshooting
        dri = (np.ceil(abs(dr))).astype(int)
        if dri[0] > 0 and dri[1] > 0:
            img1 = img1[dri[1]:-dri[1], dri[0]:-dri[0]]
            img2_shifted = img2_shifted[dri[1]:-dri[1], dri[0]:-dri[0]]
            curr_err = np.mean(abs(np.abs(img1 - img2_shifted)))
        if i > 1:
            if err < np.max(np.abs(dr)) < 0.01:
                print('converged at iteration: ', i)
                break
    assert i < max_num_iter, 'Did not converge'
    return cumul_dx, dx_vec, img2_shifted

def get_downscaled_img_2d(img, scale, stridex = 0, stridey = 0):
    # ==============================#
    # Downscales image by scale    #
    # ==============================#
    if scale<1:
        #gaussian filter img
        img_downscaled  = gaussian_filter(img, scale)
    else:
        img_downscaled = img
    # assert stride<1/scale, 'Stride is too large'
    return img_downscaled[stridex::int(1/scale), stridey::int(1/scale)]

def register_multiscale_2d(img1, img2, scale_list = [0.25, 0.5, 1]):
    # ==============================#
    # Caculates registration with lsq
    # with multiscale registration
    # Output:
    #   cumul_dx - shift after convergence
    #   img2 - shifted image based on cumul_dx
    #   dx_vec - vector of shifts over convergence
    # ==============================#

    cumul_dx = 0
    dx_vec = []
    img2_shifted = img2

    for scale in scale_list:
        img1_downscaled = get_downscaled_img_2d(img1, scale)
        img2_downscaled = get_downscaled_img_2d(img2_shifted, scale)

        # match lengths of img1 and img2
        img1_downscaled = img1_downscaled[:np.min([len(img1_downscaled), len(img2_downscaled)])]
        img2_downscaled = img2_downscaled[:np.min([len(img1_downscaled), len(img2_downscaled)])]

        dr = solve_iter_2d(img1_downscaled, img2_downscaled)
        dr = dr[0]
        dr = dr / scale
        cumul_dx += dr
        dx_vec.append(dr)
        dri = (np.ceil(abs(dr))).astype(int)
        img2_shifted = ndi.shift(img2_shifted, [dr[1], dr[0]])
        if dri[0] > 0 and dri[1] > 0:
            img1 = img1[dri[1]:-dri[1], dri[0]:-dri[0]]
            img2_shifted = img2_shifted[dri[1]:-dri[1], dri[0]:-dri[0]]


    return cumul_dx, dx_vec, img2_shifted




# generate random image
img1 = np.random.rand(300, 300)
#smooth image
img1 = gaussian_filter(img1, sigma=25)

#shift image
deltax = np.random.randint(1, 10)
deltay = np.random.randint(1, 10)
print(deltax, deltay)
img2 = ndi.shift(img1, (deltax, deltay))

img1 = img1[deltax:-deltax, deltay:-deltay]
img2 = img2[deltax:-deltax, deltay:-deltay]


dr,a  = solve_2d(img1, img2)
print(dr)



# regiter 2d with iterations

# generate random image
img1 = np.random.rand(500, 500)
img1 = gaussian_filter(img1, sigma=15)
deltax = np.random.randint(1, 55)
deltay = np.random.randint(1, 55)
print(deltax, deltay)
img2 = ndi.shift(img1, (deltax, deltay))
img1 = img1[deltax:-deltax, deltay:-deltay]
img2 = img2[deltax:-deltax, deltay:-deltay]
cumul_dx = 0
dx_vec = []
img2_shifted = img2
oimg1 = img1
oimg2 = img2


dr, ee = solve_2d(img1, img2_shifted)
print('single: ', dr)


for i in range(150):
    dr, img2_shifted = solve_2d(img1, img2_shifted)
    cumul_dx += dr
    dx_vec.append(dr)
    # prevent overshooting
    dri = (np.ceil(abs(dr))).astype(int)
    if dri[0]>0 and dri[1]>0:
        img1 = img1[dri[1]:-dri[1], dri[0]:-dri[0]]
        img2_shifted = img2_shifted[dri[1]:-dri[1], dri[0]:-dri[0]]
        curr_err = np.mean(abs(np.abs(img1-img2_shifted)))
    if i>1:
        if err<np.max(np.abs(dr))<0.01:
            print('converged at iteration: ', i)
            break
print(cumul_dx)


# fig = px.line(np.sum(img1, axis=0))
# fig.add_trace(px.line(np.sum(img2_shifted, axis=0), color_discrete_sequence=['red']).data[0])
# fig.show()


# Multiscale registration

img1 = oimg1
img2 = oimg2
cumul_dx = 0
dx_vec = []
img2_shifted = img2
scale_list = [0.25, 0.5, 1]

for scale in scale_list:
    img1_downscaled = get_downscaled_img_2d(img1, scale)
    img2_downscaled = get_downscaled_img_2d(img2_shifted, scale)

    # match lengths of img1 and img2
    img1_downscaled = img1_downscaled[:np.min([len(img1_downscaled), len(img2_downscaled)])]
    img2_downscaled = img2_downscaled[:np.min([len(img1_downscaled), len(img2_downscaled)])]

    dr, abc = solve_2d(img1_downscaled, img2_downscaled)
    dr = dr / scale
    cumul_dx += dr
    dx_vec.append(dr)
    dri = (np.ceil(abs(dr))).astype(int)
    img2_shifted = ndi.shift(img2_shifted, [dr[1], dr[0]])
    if dri[0]>0 and dri[1]>0:
        img1 = img1[dri[1]:-dri[1], dri[0]:-dri[0]]
        img2_shifted = img2_shifted[dri[1]:-dri[1], dri[0]:-dri[0]]

print(cumul_dx)




# generate random image
img1 = np.random.rand(500, 500)
img1 = gaussian_filter(img1, sigma=5)
deltax = np.random.randint(1, 100)
deltay = np.random.randint(1, 100)
print(deltax, deltay)
img2 = ndi.shift(img1, (deltax, deltay))
img1 = img1[deltax:-deltax, deltay:-deltay]
img2 = img2[deltax:-deltax, deltay:-deltay]
cumul_dx = 0
dx_vec = []
img2_shifted = img2


dr = solve_2d(img1, img2)
print('single: ', dr[0])

dr = solve_iter_2d(img1, img2)
print('iterative', dr[0])

dr = register_multiscale_2d(img1, img2)
print('multiscale',dr[0])
