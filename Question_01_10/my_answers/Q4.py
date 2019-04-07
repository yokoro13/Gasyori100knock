import cv2
import numpy as np

img = cv2.imread("../imori.jpg").astype(np.float)

H, W, C = img.shape

gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
gray = gray.astype(np.uint8)

max_sigma = 0
max_t = 0

for _t in range(1, 255):
    v0 = gray[np.where(gray < _t)]
    m0 = np.mean(v0) if len(v0) > 0 else 0.
    w0 = len(v0) / (H * W)
    v1 = gray[np.where(gray >= _t)]
    m1 = np.mean(v1) if len(v1) > 0 else 0.
    w1 = len(v1) / (H * W)
    sigma = w0 * w1 * ((m0 - m1) ** 2)
    if sigma > max_sigma:
        max_sigma = sigma
        max_t = _t

gray[gray < max_t] = 0
gray[gray >= max_t] = 255

cv2.imshow("Q4", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()