import cv2
import numpy as np

img = cv2.imread("../imori.jpg")

out = img.copy()
H, W, C = img.shape
G = 8
Nh = int(H/G)
Nw = int(W/G)

for h_ in range(Nh):
    for w_ in range(Nw):
        for c_ in range(C):
            out[G*h_:G*(h_+1), G*w_:G*(w_+1), c_] = np.mean(out[G*h_:G*(h_+1), G*w_:G*(w_+1), c_]).astype(np.int)

cv2.imshow("Q7", out)
cv2.waitKey(0)
cv2.destroyAllWindows()