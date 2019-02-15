import numpy as np

temp = [0, 1, 2, 3]

temp = temp + [i+18 for i in temp]
print(temp)