import cv2
import numpy as np

img = cv2.imread("../imori.jpg").astype(np.float)
b = img[:, :, 0].copy()
g = img[:, :, 1].copy()
r = img[:, :, 2].copy()

gray = 0.2126*r + 0.7152*g + 0.0722*b
gray = gray.astype(np.uint8)

gray[gray < 128] = 0
gray[gray >= 128] = 255

cv2.imshow("Q3", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
