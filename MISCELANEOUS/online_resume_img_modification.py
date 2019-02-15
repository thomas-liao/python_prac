# import cv2
#
# img = cv2.imread('/Users/admin/Desktop/vehicle_pw4.png')
# # img = cv2.imread('/Users/admin/Desktop/test/thomas-liao.github.io/assets/images/Messenger_5_8_bak.png')
#
#
# # img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2) )
#
# # cv2.imshow('temp', img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
#
#
# img = cv2.resize(img, (1982, 1208))
#
# cv2.imwrite('Messenger_13_16_fake.png', img)
#
# # a = 1
#

# # shape: 1208, 1982, 3




import cv2
img_name = '/Users/admin/Desktop/dm/img1.png'
img = cv2.imread(img_name)
img = cv2.resize(img, (1982, 1208))
cv2.imwrite(img_name, img)




#
#
# import cv2
# import os
#
# image_folder = '/Users/admin/Desktop/vehicle'
# video_name = 'video.avi'
#
# images = ['img1.png','img1.png','img1.png','img2.png','img2.png','img2.png',  'img3.png','img3.png','img3.png', 'img4.png','img4.png','img4.png']
#
# # images = ['img1.png','img1.png','img2.png','img2.png', 'img3.png', 'img3.png','img4.png','img4.png']
#
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape
#
# video = cv2.VideoWriter(video_name, 0, 4, (width, height))
#
# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))
#
#
# cv2.destroyAllWindows()
# video.release()
# #