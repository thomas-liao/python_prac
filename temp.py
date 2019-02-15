import cv2

img = cv2.imread('new_img2.jpg')

cv2.imshow('temp', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

h, w, _ = img.shape # 8, 59, 88
#
for i in range(h):
    for j in range(w):
        if 78 <= img[i, j, 0] <= 98 and 49 <= img[i, j, 1] <= 69 and 2 <= img[i, j, 2] <= 18:
            img[i, j, 0] = img[i, j, 1] = img[i, j, 2] = 255




cv2.imshow('temp', img)
# cv2.waitKey(9999)
# cv2.destroyAllWindows()
cv2.imwrite('new_img2.jpg', img)