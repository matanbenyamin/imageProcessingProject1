import cv2


im = cv2.imread('newplot.png')
im = im[:,:,::-1]
im = im[68:757,144:1730]

fig = px.imshow(im)
fig.show()

# im  = im[:,:,-1]

color = [99,110,250]
# loo for each column and find indices of color in image
indices = []
for i in range(im.shape[1]):
    col = im[:,i]
    #look for color indices in col
    for j in range(len(col)):
        if col[j][0] == color[0] and col[j][1] == color[1] and col[j][2] == color[2]:
            indices.append(j)
            break




# find index of max in each column of im
max_ind = np.argmax(im, axis=0)


fig = px.line(indices)
fig.show()