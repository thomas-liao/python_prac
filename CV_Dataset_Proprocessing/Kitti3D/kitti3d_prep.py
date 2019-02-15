import os.path as osp
import os
from shutil import copyfile


root_dir = '/Users/admin/Downloads/kitti3d'
gt_img_dir = '/Users/admin/Downloads/kitti3d'
dest_root_dir = '/Users/admin/Downloads/kitti3d/tliao4'

files = ['det_occ0.txt', 'det_occ1.txt', 'det_occ2.txt', 'det_occ3.txt']
additional_post_fix = ['.2d', '.3d', '.yaw']
for f in files:
    file = osp.join(root_dir, f).rstrip()
    for name in open(file, 'r'):
        src_ = osp.join(gt_img_dir, name).rstrip()
        dest_dir = osp.join(dest_root_dir, f[:-4] + '_images')
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        #
        dd = src_.replace('det_bbx', 'tliao4/' + f[:-4] + '_images')
        # print(src_)
        # print(dd)
        # print('\n')
        copyfile(src_, dd)
        for p_fix in additional_post_fix:
            src_ad = src_.replace('.png', p_fix)
            dest_ad = dd.replace('.png', p_fix)
            try:
                copyfile(src_ad, dest_ad)
            except FileNotFoundError:
                continue




