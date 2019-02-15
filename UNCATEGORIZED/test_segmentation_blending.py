# import numpy as np
#
# obj_mask = np.zeros((5, 5))
#
# obj_mask[1:4, 1:4] = 1
# obj_mask = np.stack([obj_mask]*3, axis=-1)
# obj_mask = obj_mask == 1
# obj_mask = obj_mask.astype(np.float32)
#
# #
# #
# # #
# print('\nnow start the real test\n')
# blend_bg_img = 10 * np.ones((5, 5, 3))
# obj_img = 5 * np.ones((5, 5, 3))
#
# print(obj_img * obj_mask + ((obj_mask == 0).astype(np.float32)) * blend_bg_img)
#
#
#
# #
# #
# #

import cv2
import numpy as np

mask_img = cv2.imread('/Users/admin/Desktop/del_test_background_blending/00000008.png')
bkgd_img = cv2.imread('/Users/admin/Desktop/scene.jpeg')


img_h, img_w = mask_img.shape[:2]

bkgd_img = cv2.resize(bkgd_img, (img_w, img_h))

seg_color = [255, 127, 0]
print(mask_img[0, 0, :])


def get_obj_mask(seg_im, color):

    seg_mask = np.array(seg_im[:, :, 0] * (256 ** 2) + seg_im[:, :, 1] * 256 + seg_im[:, :, 2])
    if isinstance(color, list):
        R, G, B = color
    if isinstance(color, dict):
        R, G, B = color['R'], color['G'], color['B']

    val = R*(256 ** 2) + G * 256 + B

    obj_mask = np.equal(seg_mask, val)
    return obj_mask


obj_mask = get_obj_mask(mask_img, seg_color)
rev_obj_mask = ~obj_mask

obj_mask = obj_mask.astype(np.uint8)
rev_obj_mask = rev_obj_mask.astype(np.uint8)
#
obj_mask = np.stack([obj_mask]*3, axis=-1)
rev_obj_mask = np.stack([rev_obj_mask]*3, axis=-1)
#
blended_img = rev_obj_mask * bkgd_img
#
cv2.imshow('temp', blended_img)
cv2.waitKey(0)
cv2.destroyAllWindows()







#
# cv2.imshow('temp', 255* (~obj_mask).astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()







