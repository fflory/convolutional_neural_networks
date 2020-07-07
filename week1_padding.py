import numpy as np

# from keras.preprocessing.sequence import pad_sequences
# from keras.layers import ZeroPadding2D
# from keras.backend import spatial_2d_padding

np.set_printoptions(edgeitems = 32)       #, linewidth = 150

x = np.array([
    [3, 0, 1, 2, 7, 4],
    [1, 5, 8, 9, 3, 1],
    [2, 7, 2, 5, 1, 3],
    [0, 1, 3, 1, 7, 8],
    [4, 2, 1, 6, 2, 8],
    [2, 4, 5, 2, 3, 9]],
    dtype = int)
n = x.shape[0]

# np.pad(x, (1, 1), 'constant', constant_values = (0, 0))

p = 1
x_p = np.pad(x, (p), 'constant', constant_values = (0))

vertical_edge_filter = np.array(
    [[1, 0, -1],
     [1, 0, -1],
    [1, 0, -1]])
f = vertical_edge_filter.shape[0]

res_shape_par = x_p.shape[0] - vertical_edge_filter.shape[0] + 1

conv_res = np.array([
    np.sum(x_p[i:i+3, j:j+3] * vertical_edge_filter)
    for i in np.arange(0, res_shape_par)
    for j in np.arange(0, res_shape_par)])

vertical_edge_detector = np.array(conv_res).reshape(res_shape_par, res_shape_par)


num_weights = (n + 2*p - f + 1)**2
num_weights

# 'valid' convolution: no padding: (n - f + 1)**2
# 'same' convolution: output size is same as input size
#     (n + 2*p - f + 1)**2 <-> p = (f-1)/2
#     f is usually an odd number (by convention)
