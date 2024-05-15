import cv2
import numpy as np
import math

M = 480
L = 256

imgin = np.zeros((M, L), dtype=np.uint8) + 255
a = 0.032
b = 3
p = np.zeros(L, dtype=np.float32)

for z in range(L-1):
    p[z] = math.pow(a, b) * math.pow(z, b-1) * math.exp(-a*z) / math.factorial(b-1)
    print('%.10f' % p[z])
print('max = ', np.max(p))
print('vi tri max = ', np.argmax(p))
print('sum = ', np.sum(p))

scale = 10000
for z in range(L-1):
    cv2.line(imgin, (z, M-1), (z, M - 1 - int(p[z]*scale)), 0)

cv2.imshow("Image", imgin)
cv2.waitKey(0)

q = np.zeros(L, dtype=np.float32)
for v in range(L):
    z = L - 1 -v
    q[z] = p[v]

scale = 10000
for z in range(L):
    cv2.line(imgin, (z, M-1), (z, M - 1 - int(q[z]*scale)), 0)
