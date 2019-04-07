import cv2
import numpy as np

img = cv2.imread("../imori_noise.jpg")
H, W, C = img.shape

K_size = 3

pad = K_size//2
out = np.zeros((H + pad*2, W + pad*2, C), dtype=np.float)
out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)

tmp = out.copy()
for h_ in range(H):
    for w_ in range(W):
        for c_ in range(C):
            out[pad + h_, pad+w_, c_] = np.median(tmp[h_:h_+K_size, w_:w_+K_size, c_])

out = out[pad:pad+H, pad:pad+W].astype(np.uint8)

cv2.imshow("Q10", out)
cv2.waitKey(0)
cv2.destroyAllWindows()