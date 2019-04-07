import cv2
import numpy as np

img = cv2.imread("../imori.jpg").astype(np.float)
r = img[:, :, 2].copy()
g = img[:, :, 1].copy()
b = img[:, :, 0].copy()

gray = 0.2126*r + 0.7152*g + 0.0722*b
gray = gray.astype(np.uint8)

cv2.imshow("Q2", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()