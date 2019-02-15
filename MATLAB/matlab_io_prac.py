#

import scipy.io as spio
import scipy
import numpy as np
import cv2
import os
import pickle
from shutil import copyfile


#
def modified_loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list



    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

## sanity check
# temp = modified_loadmat('/Users/admin/Downloads/PASCAL3D+_release1.1/Annotations/car_imagenet/n02814533_10251.mat')
# print(temp)
#
# print(temp['record']['objects']['bbox'])
# print(temp['record']['objects']['anchors'])
#
#
#
# # mini test
# test_img = cv2.imread('/Users/admin/Downloads/PASCAL3D+_release1.1/Images/car_imagenet/n02814533_10251.JPEG')
#
# bbx = temp['record']['objects']['bbox']
# kp_names = temp['record']['objects']['anchors'].keys()
# print(kp_names)
#
#
# x1, y1, x2, y2 = map(int, bbx)
#
# cv2.rectangle(test_img, (x1, y1), (x2, y2), (255,0,0), 2)
#
# for kp_name in kp_names:
#     if len(temp['record']['objects']['anchors'][kp_name]['location']) == 2:
#         x, y = map(int, temp['record']['objects']['anchors'][kp_name]['location'])
#
#         cv2.circle(test_img, (x, y), 3, [0, 255, 128], -1)
#
#
# cv2.imshow("whatever", test_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # todo: 1. prepare all validation data  2. set up PCK test  3. run experiment and comparison


###
val_img_name = '/Users/admin/Downloads/PASCAL3D+_release1.1/Image_sets/car_imagenet_val.txt'
img_root_dir = '/Users/admin/Downloads/PASCAL3D+_release1.1/Images/car_imagenet'
annot_root_dir = '/Users/admin/Downloads/PASCAL3D+_release1.1/Annotations/car_imagenet'

img_combined_dir = []
annot_combined_dir = []

file_obj = open(val_img_name, 'r')

for line in file_obj:
    img_combined = os.path.join(img_root_dir, line)[:-1] + '.JPEG'
    img_combined_dir.append(img_combined)

    annot_combined = os.path.join(annot_root_dir, line)[:-1] + '.mat'
    annot_combined_dir.append(annot_combined)

# temp = modified_loadmat('/Users/admin/Downloads/PASCAL3D+_release1.1/Annotations/car_imagenet/n02814533_10251.mat')

i = 0
j = 0
for annot in annot_combined_dir:
    try:
        temp = modified_loadmat(annot)
        # print(temp['record']['objects'])
        bbx = map(int, temp['record']['objects']['bbox'])
        anchor = temp['record']['objects']['anchors']
        i += 1
        # print(anchor)
    except:
        j += 1
        continue

temp_img_names = []

with open('pascal3d_imagenet_val_annot.pickle', 'rb') as fin:
    det = pickle.load(fin)
    det['anchors_name'] = ['left_front_wheel', 'left_back_wheel', 'right_front_wheel', 'right_back_wheel', 'upper_left_windshield', 'upper_right_windshield', 'upper_left_rearwindow', 'upper_right_rearwindow', 'left_front_light', 'right_front_light', 'left_back_trunk', 'right_back_trunk']

    # reset beginning
    file_obj = open(val_img_name, 'r')

    for f in file_obj:
        try:
            img_name = f[:-1]
            img_annot = os.path.join(annot_root_dir, f[:-1]) + '.mat'
            temp = modified_loadmat(img_annot)
            temp_dict = {}
            temp_dict['anchors'] = temp['record']['objects']['anchors']
            temp_dict['bbox'] = temp['record']['objects']['bbox']
            det[img_name] = temp_dict
            temp_img_names.append(img_name) #
        except:
            continue

    with open('pascal3d_imagenet_val_annot.pickle', 'wb') as fout:
        pickle.dump(det, fout)




for line in temp_img_names:
    src_ = os.path.join(img_root_dir, line) + '.JPEG' # already left out the last '\n', no need line[:-1]
    dst_ = os.path.join('pascal3d_val', line) + '.JPEG'
    copyfile(src_, dst_)





















