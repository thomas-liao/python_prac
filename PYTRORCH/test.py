import torch
import numpy as np

M = 10
N = 10


i = np.arange(M).reshape(1, M)
# j = np.arange(N).reshape(N, 1)[::-1]
j = np.arange(N).reshape(N, 1)[::-1]

print((i+j) < 10)


