import cv2

img = cv2.imread("../imori.jpg")
img = img[:, :, (2, 1, 0)]
cv2.imshow("Q1", img)
cv2.waitKey(0)
cv2.destroyAllWindows()