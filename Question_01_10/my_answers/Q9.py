import cv2
import numpy as np

img = cv2.imread("../imori_noise.jpg")
H, W, C = img.shape

K_size = 3
sigma = 1.3

# ゼロパディング
pad = K_size//2
out = np.zeros((H + pad*2, W + pad*2, C), dtype=np.float)
out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)

K = np.zeros((K_size, K_size), dtype=np.float)

for x in range(-pad, -pad+K_size):
    for y in range(-pad, -pad+K_size):
        K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))

K /= sigma * np.sqrt(2 * np.pi)
K /= K.sum()

tmp = out.copy()

for h_ in range(H):
    for w_ in range(W):
        for c_ in range(C):
            out[pad+h_, pad+w_, c_] = np.sum(K * tmp[h_:h_+K_size, w_:w_+K_size, c_])

out = out[pad:pad+H, pad:pad+W].astype(np.uint8)
cv2.imshow("Q9", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
