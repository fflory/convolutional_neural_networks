import numpy as np
import math

np.set_printoptions(edgeitems = 32)

x = np.array([
    [2,3,7,4,6,2,9],
    [6,6,9,8,7,4,3],
    [3,4,8,3,8,9,7],
    [7,8,3,6,6,3,4],
    [4,2,1,8,3,4,6],
    [3,2,4,1,9,8,3],
    [0,1,3,9,2,1,4]
    ],
    dtype = int)
# image size n by n
n = x.shape[0]

# np.pad(x, (1, 1), 'constant', constant_values = (0, 0))

# padding
p = 0
x_p = np.pad(x, (p), 'constant', constant_values = (0))

vertical_edge_filter = np.array(
    [[3,4,4],
     [1,0,2],
    [-1,0,3]])
# filter size f by f
f = vertical_edge_filter.shape[0]

res_shape_par = x_p.shape[0] - vertical_edge_filter.shape[0] + 1

# stride
s = 2

strides1 = np.arange(0, res_shape_par, step = stride, dtype = int)
strides2 = np.arange(0, res_shape_par, step = stride, dtype = int)

conv_res = np.array([
    np.sum(x[i:i+3, j:j+3] * vertical_edge_filter)
    # without flipping in math this is called cross-correlation
    # in math textbook convolution would:
    # (not flipping)
    #  ... * vertical_edge_filter.T
    for i in strides1
    for j in strides2])
conv_res

out_dim = int(math.floor(((n + 2*p -f) / s) + 1))

vertical_edge_detector = np.array(conv_res).reshape(
    out_dim, out_dim)
vertical_edge_detector
