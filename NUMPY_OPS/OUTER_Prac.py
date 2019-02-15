# -*- coding:utf8 -*-


import numpy as np

### 1, column - oupter product
print('test column')
t1 = np.array([[1], [4], [2]]) # shape (3, 1)

t1_squeeze = np.squeeze(t1, axis=1) # shape (3,)

t2 = np.array([[1], [2], [3]])
t2_squeeze = np.squeeze(t2, axis=1)


res_fix = np.outer(t1_squeeze, t2_squeeze, out=None)
print(res_fix)
res_fix = np.outer(t1, t2, out=None)
print(res_fix) # they are the same

print(res_fix) # "以 列 为模板的， 向右照比例拓展的matrix"
#
# [[ 1  2  3]
#  [ 4  8 12]
#  [ 2  4  6]]


#### 2, raw - oupter product
print('\n\ntest raw\n\n')
t1 = np.array([1,4,2])
t2 = np.array([1,2,3])
res = np.outer(t1, t2)
print(res) # 居然跟1是一样的。。。。。

