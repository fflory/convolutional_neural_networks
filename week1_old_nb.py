# Databricks notebook source
# MAGIC %md 
# MAGIC # Edge Detection Example

# COMMAND ----------

import numpy as np
import math

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

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Padding

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC # Stridded Convolutions

# COMMAND ----------

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

strides1 = np.arange(0, res_shape_par, step = s, dtype = int)
strides2 = np.arange(0, res_shape_par, step = s, dtype = int)

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

# COMMAND ----------

# MAGIC %md
# MAGIC # Pooling Layers
# MAGIC 
# MAGIC pooling is easy
# MAGIC hyperparameters are:
# MAGIC f = filter size (common value: 2)
# MAGIC s = stride size (common value: 2)
# MAGIC 
# MAGIC no parameters to learn!

# COMMAND ----------

# formula also works for pooling
int(math.floor(((n + 2*p - f) / s) + 1))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # One Layer of a CNN

# COMMAND ----------

np.set_printoptions(edgeitems = 32)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 10 filters of 3 by 3 by 3
# MAGIC each fiter of 3 by 3 by 3 parameters get only one bias parameters
# MAGIC 27 paramters per filter plus bias = 28 parameters
# MAGIC (27 + 1) * 100 = 280
# MAGIC 
# MAGIC f^[l]: filter size
# MAGIC p^[l]: padding
# MAGIC s^[l]
# MAGIC n_c^[l] = number of filters
# MAGIC 
# MAGIC Input: n_h^[l-1] x n_w^[l-1] x n_c^[l-1]
# MAGIC 
# MAGIC Output: n_H^[l] x n_w^[l] x n_c^[l]
# MAGIC 
# MAGIC n_H^[l] = math.floor(((n + 2*p -f) / s) + 1)
# MAGIC 
# MAGIC 
# MAGIC Each filter is f^[l] x f^[l] x 

# COMMAND ----------

# MAGIC %md
# MAGIC # A simple conv Network

# COMMAND ----------

# MAGIC %md
# MAGIC 39 x 39 x 3
# MAGIC n_h^[0] = n_w^[0] = 39
# MAGIC n_c^[0] = 3
# MAGIC 
# MAGIC f[1] = 3
# MAGIC s[1] = 1
# MAGIC p[1] = 0
# MAGIC 10 filters
# MAGIC 
# MAGIC ((n+2*p-f) / s) + 1
# MAGIC 37 x 37 x 10
# MAGIC 
# MAGIC int(math.floor(((39+2*0-3) / 1) + 1))
# MAGIC 
# MAGIC n_H^[1] = 37
# MAGIC n_c^[1] = 10
