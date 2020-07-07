import numpy as np

np.set_printoptions(edgeitems = 32)       #, linewidth = 150

x = np.array([
    [3, 0, 1, 2, 7, 4],
    [1, 5, 8, 9, 3, 1],
    [2, 7, 2, 5, 1, 3],
    [0, 1, 3, 1, 7, 8],
    [4, 2, 1, 6, 2, 8],
    [2, 4, 5, 2, 3, 9]],
    dtype = np.uint8)

vertical_edge_filter = np.array(
    [[1, 0, -1],
     [1, 0, -1],
    [1, 0, -1]])

res_shape_par = x.shape[0] - vertical_edge_filter.shape[0] + 1

conv_res = np.array([
    np.sum(x[i:i+3, j:j+3] * vertical_edge_filter)
    for i in np.arange(0, res_shape_par)
    for j in np.arange(0, res_shape_par)])

vertical_edge_detector = np.array(conv_res).reshape(res_shape_par, res_shape_par)

vertical_edge_detector

# python: conv_forward
# tensorflow: tf.nn.conv2d
# keras: conv2D

ex2 = np.hstack(
    (np.ones((6,3), dtype=int)*10,
     np.zeros((6,3), dtype=int)))

ex2_res = np.array([
    np.sum(ex2[i:i+3, j:j+3] * vertical_edge_filter)
    for i in np.arange(0, res_shape_par)
    for j in np.arange(0, res_shape_par)])

np.array(ex2_res).reshape(res_shape_par, res_shape_par)

ex3 = np.hstack(
    (np.zeros((6,3), dtype=int),
     np.ones((6,3), dtype=int)*10))

ex3_res = np.array([
    np.sum(ex3[i:i+3, j:j+3] * vertical_edge_filter)
    for i in np.arange(0, res_shape_par)
    for j in np.arange(0, res_shape_par)])

np.array(ex3_res).reshape(res_shape_par, res_shape_par)
