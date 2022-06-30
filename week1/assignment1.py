# Databricks notebook source
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC # Exercise 2 - conv_single_step

# COMMAND ----------

np.random.seed(1)
a_slice_prev = np.random.randn(4, 4, 3)
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)

# COMMAND ----------

# np.multiply(a_slice_prev, W)
a_slice_prev * W

# COMMAND ----------

np.sum(a_slice_prev * W) + float(b)

# COMMAND ----------

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """
    
    #(≈ 1 line)
    # X_pad = None
    # YOUR CODE STARTS HERE
    X_pad = np.pad(X, 
                   ((0,0), (pad, pad), (pad, pad), (0,0)), 
                   mode='constant', constant_values = (0))
    
    # YOUR CODE ENDS HERE
    
    return X_pad

# COMMAND ----------

def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """

    #(≈ 3 lines of code)
    # Element-wise product between a_slice_prev and W. Do not add the bias yet.
    # s = None
    # Sum over all entries of the volume s.
    # Z = None
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    # Z = None
    # YOUR CODE STARTS HERE
    s = np.multiply(a_slice_prev, W) # a * b also works
    Z = np.sum(s)
    Z = Z + float(b)
    # YOUR CODE ENDS HERE

    return Z

# COMMAND ----------

# MAGIC %md
# MAGIC #Exercise 3 - conv_forward

# COMMAND ----------

np.random.seed(1)
A_prev = np.random.randn(2, 5, 7, 4)       # shape (m, n_H_prev, n_W_prev, n_C_prev)
W = np.random.randn(3, 3, 4, 8)            # shape (f, f, n_C_prev, n_C)
b = np.random.randn(1, 1, 1, 8)            # shape (1, 1, 1, n_C)
hparameters = {"pad": 1, "stride": 2}

# COMMAND ----------

# Retrieve dimensions from A_prev's shape (≈1 line)  
(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
# Retrieve dimensions from W's shape (≈1 line)
(f, f, n_C_prev, n_C) = W.shape
# Retrieve information from "hparameters" (≈2 lines)
stride = hparameters['stride']
pad = hparameters['pad']
# Compute the dimensions of the CONV output volume using the formula given above. 
# Hint: use int() to apply the 'floor' operation. (≈2 lines)
n_H = int(((n_H_prev - f + 2 * pad)/stride)) + 1
n_W = int(((n_W_prev - f + 2 * pad)/stride)) + 1
# Initialize the output volume Z with zeros. (≈1 line)
Z = np.zeros((m, n_H, n_W, n_C))
# Create A_prev_pad by padding A_prev
A_prev_pad = zero_pad(A_prev, pad)

# COMMAND ----------

print(Z.shape)
print(A_prev_pad.shape)
print("n_H =", n_H, ", n_W =", n_W)

# COMMAND ----------

i=0
h=2
w=1
c=0
expected = -2.17796037
a_prev_pad = A_prev_pad[i, :,:,:]
vert_start = h
vert_end = h + f
horiz_start = w * stride
horiz_end = w * stride + f
a_slice_prev = a_prev_pad[
                        # i, # ith training example already selected
                        vert_start:vert_end, 
                        horiz_start:horiz_end, 
                        :]
weights = W[:,:,:,c]
biases = b[:,:,:,c]
# Z[i, h, w, c] = 
conv_single_step(a_slice_prev, weights, biases)

# COMMAND ----------

print(A_prev[0,:,:,].shape)
print(A_prev_pad[0,:,:,].shape)
for j in range(n_C_prev):
  print("-"*50)
  print(A_prev_pad[0,:,:,j].round(1))

# COMMAND ----------

weights[:,:,0].round(1)

# COMMAND ----------

weights = W[:,:,:, 0]
for c in range(n_C_prev):
  print("-"*20)
  print(weights[:,:,c].round(1))

# COMMAND ----------

a_prev_pad = A_prev_pad[0, :,:,:]
a_slice_prev = a_prev_pad[
                        # i, # ith training example already selected
                        0:3, # TODO 2:5, 4:7 
                        0:3, # TODO 2:5, 4:7, 6:9
                        :]
print(a_slice_prev.shape)
for c in range(n_C_prev):
  print("-"*4, "c=",  c, "-"*4)
  print(a_slice_prev[:,:,c].round(1))

# COMMAND ----------

biases = b[:,:,:,0]
conv_single_step(a_slice_prev, weights, biases)

# COMMAND ----------



# COMMAND ----------

for i in range(m-1):
    # i = 0
    a_prev_pad = A_prev_pad[i, :, :, :]               # Select ith training example's padded activation
    for h in range(n_H):           # loop over vertical axis of the output volume
        # h = 2
        # Find the vertical start and end of the current "slice" (≈2 lines)
        vert_start = h * stride
        vert_end = h * stride + f
        print("h =", h, ", vert_start =", vert_start, "vert_end =", vert_end)

# COMMAND ----------

for i in range(m-1):
    # i = 0
    a_prev_pad = A_prev_pad[i, :, :, :]               # Select ith training example's padded activation
    for w in range(n_W):           # loop over vertical axis of the output volume
        # w = 1
        # Find the vertical start and end of the current "slice" (≈2 lines)
        horiz_start = w * stride
        horiz_end = w * stride + f
        print("w =", w, ", horiz_start =", horiz_start, "horiz_end =", horiz_end)

# COMMAND ----------

np.array(range(5))[0:3]


# COMMAND ----------

# for i in range(m):
i = 0
a_prev_pad = A_prev_pad[i, :, :, :]               # Select ith training example's padded activation
# for h in range(n_H):           # loop over vertical axis of the output volume
h = 2
# Find the vertical start and end of the current "slice" (≈2 lines)
vert_start = h * stride
vert_end = h * stride + f
# for w in range(n_W):       # loop over horizontal axis of the output volume
w = 1
# Find the horizontal start and end of the current "slice" (≈2 lines)
horiz_start = w * stride
horiz_end = w * stride + f
# for c in range(n_C):   # loop over channels (= #filters) of the output volume
c = 1
# Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
a_slice_prev = a_prev_pad[
  # i, # ith training example already selected
  vert_start:(vert_end+1), 
  horiz_start:(horiz_end+1), 
  :]
# Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈3 line)
weights = W[:,:,:,c]
biases = b[:,:,:,c]
Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)

# COMMAND ----------

print(a_slice_prev)
print(weights.shape)
print(Z[i, :, :, c])

# COMMAND ----------

print("Z =", Z[i, h, w, :])

# COMMAND ----------

print(a_slice_prev.shape)
print(weights.shape)
print(biases.shape)

# COMMAND ----------


