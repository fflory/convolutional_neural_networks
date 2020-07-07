import numpy as np
import math

np.set_printoptions(edgeitems = 32)

10 filters of 3 by 3 by 3
each fiter of 3 by 3 by 3 parameters get only one bias parameters
27 paramters per filter plus bias = 28 parameters
(27 + 1) * 100 = 280

f^[l]: filter size
p^[l]: padding
s^[l]
n_c^[l] = number of filters

Input: n_h^[l-1] x n_w^[l-1] x n_c^[l-1]

Output: n_H^[l] x n_w^[l] x n_c^[l]

n_H^[l] = math.floor(((n + 2*p -f) / s) + 1)


Each filter is f^[l] x f^[l] x 
