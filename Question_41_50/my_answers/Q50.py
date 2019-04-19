import cv2
import numpy as np

img = cv2.imread("../imori.jpg").astype(np.float)
H, W, C = img.shape

gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
gray = gray.astype(np.uint8)
K_size = 5
s = 1.4

pad = K_size//2
gau = np.zeros((H + pad*2, W + pad*2), dtype=np.float32)
gau = np.pad(gray, (pad, pad), 'edge')
tmp = gau.copy()

K = np.zeros((K_size, K_size), dtype=np.float32)

for y in range(-pad, -pad+K_size):
    for x in range(-pad, -pad+K_size):
        K[y+pad, x+pad] = np.exp(-(x**2 + y**2) / (2*(s**2)))

K /= (s * np.sqrt(2*np.pi))
K /= K.sum()

for y in range(H):
    for x in range(W):
        gau[pad + y, pad + x] = np.sum(K * tmp[y:y + K_size, x:x + K_size])

# Sobel vertical
KSV = np.array(((-1., -2., -1.), (0., 0., 0.), (1., 2., 1.)), dtype=np.float32)
# Sobel horizontal
KSH = np.array(((-1., 0., 1.), (-2., 0., 2.), (-1., 0., 1.)), dtype=np.float32)

gau = gau[pad - 1:H + pad + 1, pad - 1:W + pad + 1]
fy = np.zeros_like(gau, dtype=np.float32)
fx = np.zeros_like(gau, dtype=np.float32)
K_size = 3
pad = K_size // 2

for y in range(H):
    for x in range(W):
        fy[pad + y, pad + x] = np.sum(KSV * gau[y:y + K_size, x:x + K_size])
        fx[pad + y, pad + x] = np.sum(KSH * gau[y:y + K_size, x:x + K_size])

fx = fx[pad:pad + H, pad:pad + W]
fy = fy[pad:pad + H, pad:pad + W]

edge = np.sqrt(np.power(fx, 2) + np.power(fy, 2))
fx[fx == 0] = 1e-5
tan = np.arctan(fy/fx)

angle = np.zeros_like(tan, dtype=np.uint8)
angle[np.where((-0.4142 < tan) & (tan <= 0.4142))] = 0
angle[np.where((0.4142 < tan) & (tan < 2.4142))] = 45
angle[np.where((tan >= 2.4142) | (tan <= -2.4142))] = 95
angle[np.where((-2.4142 < tan) & (tan <= -0.4142))] = 135

dx1, dy1, dx2, dy2 = 0, 0, 0, 0

for y in range(H):
    for x in range(W):
        if angle[y, x] == 0:
            dx1, dy1, dx2, dy2 = -1, 0, 1, 0
        elif angle[y, x] == 45:
            dx1, dy1, dx2, dy2 = -1, 1, 1, -1
        elif angle[y, x] == 90:
            dx1, dy1, dx2, dy2 = 0, -1, 0, 1
        elif angle[y, x] == 135:
            dx1, dy1, dx2, dy2 = -1, -1, 1, 1
        if x == 0:
            dx1 = max(dx1, 0)
            dx2 = max(dx2, 0)
        if y == 0:
            dy1 = max(dy1, 0)
            dy2 = max(dx2, 0)
        if x == W-1:
            dx2 = min(dx1, 0)
            dx2 = min(dx2, 0)
        if y == H-1:
            dy1 = min(dy1, 0)
            dy2 = min(dy2, 0)
        if max(edge[y, x], edge[y+dy1, x+dx1], edge[y+dy2, x+dx2]) != edge[y, x]:
            edge[y, x] = 0

HT = 100
LT = 30
edge[edge >= HT] = 255
edge[edge <= LT] = 0


_edge = np.zeros((H+2, W+2), dtype=np.float32)
_edge[1:1+H, 1:1+W] = edge

# 周り８画素を取り出す
nn = np.array(((1., 1., 1.), (1., 0., 1.), (1., 1., 1.)), dtype=np.float32)

for y in range(1, H+2):
    for x in range(1, W+2):
        if _edge[y, x] < LT or _edge[y, x] > HT:
            continue
        if np.max(_edge[y-1:y+2, x-1:x+2] * nn) >= HT:
            _edge[y, x] = 255
        else:
            _edge[y, x] = 0

tmp = _edge.astype(np.float)

out = np.zeros_like(edge, dtype=np.float)

N = 1

MF = np.array(((0, 1, 0), (1, 0, 1), (0, 1, 0)), dtype=np.int)


for i in range(N):
    # tmp = np.pad(out, (1, 1), "edge")
    for y in range(1, H+1):
        for x in range(1, W+1):
            if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) >= 255:
                out[y-1, x-1] = 255

for i in range(N):
    tmp = np.pad(out, (1, 1), "edge")
    for y in range(1, H + 1):
        for x in range(1, W + 1):
            if np.sum(MF * tmp[y - 1:y + 2, x - 1:x + 2]) < 255 * 4:
                out[y - 1, x - 1] = 0

out = out.astype(np.uint8)
cv2.imshow("Q50", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
