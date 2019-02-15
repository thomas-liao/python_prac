import cv2
import os

import numpy as np

np.random.seed(101)
# temp = np.load('/Users/admin/Downloads/catts.npy')

lst = os.listdir('/Users/admin/Desktop/target_domain')
print(len(lst))

chosen_idx = np.random.choice(113, 100)

for i in chosen_idx:
    path = os.path.join('/Users/admin/Desktop/target_domain', lst[i])
    img = cv2.imread(path)
    cv2.imshow('temp', img)
    cv2.waitKey(0)
    #
    # img = cv2.resize(img, (72, 72))
    # cv2.imwrite('/Users/admin/PycharmProjects/python_prac/UNCATEGORIZED/target_cats/{}.png'.format(i), img)
    #
