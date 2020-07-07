pooling is easy
hyperparameters are:
f = filter size (common value: 2)
s = stride size (common value: 2)

no parameters to learn!

# formula also works for pooling
int(math.floor(((n + 2*p - f) / s) + 1))
