import cv2
import numpy as np

img = cv2.imread("../imori.jpg").astype(np.float)
H, W, C = img.shape

out = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
out = out.astype(np.uint8)

max_sigma = 0
max_t = 0

for _t in range(1, 255):
    v0 = out[np.where(out < _t)]
    m0 = np.mean(v0) if len(v0) > 0 else 0.
    w0 = len(v0) / (H*W)
    v1 = out[np.where(out >= _t)]
    m1 = np.mean(v1) if len(v1) > 0 else 0.
    w1 = len(v1) / (H*W)

    sigma = w0 * w1 * ((m0 - m1) ** 2)
    if sigma > max_sigma:
        max_sigma = sigma
        max_t = _t

out[out < max_t] = 0
out[out >= max_t] = 255

time = 2

MF = np.array(((0, 1, 0), (1, 0, 1), (0, 1, 0)), dtype=np.int)
for i in range(time):
    tmp = np.pad(out, (1, 1), "edge")
    for y in range(1, H+1):
        for x in range(1, W+1):
            if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) >= 255:
                out[y-1, x-1] = 255

cv2.imshow("Q47", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
