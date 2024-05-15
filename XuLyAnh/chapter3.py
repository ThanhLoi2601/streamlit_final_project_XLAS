import numpy as np
import cv2

L = 256

def Negative(imgin):
    M, N = imgin.shape
    # imgout = np.zeros((M, N), dtype=np.uint8) + 255 - imgin
    imgout = np.zeros((M, N), dtype=np.uint8) + 255
    for i in range(M):
        for j in range(N):
            r = imgin[i, j]
            s = L - 1 - r
            imgout[i, j] = s
    return imgout

def NegativeColor(imgin):
    M, N, C = imgin.shape
    imgout = np.zeros((M, N, C), dtype=np.uint8) + 255 - imgin
    # imgout = np.zeros((M, N, C), dtype=np.uint8) +255
    # for i in range(M):
    #     for j in range(N):
    #         b = imgin[i, j, 0]
    #         g = imgin[i, j, 1]
    #         r = imgin[i, j, 2]

    #         b = L - 1 - b
    #         g = L - 1 - g
    #         r = L - 1 - r

    #         imgout[i, j, 0] = b
    #         imgout[i, j, 1] = g
    #         imgout[i, j, 2] = r

    return imgout

def Logarit(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), dtype=np.uint8) + 255
    c = (L - 1) / np.log(1.0*L)
    for i in range(M):
        for j in range(N):
            r = imgin[i, j]
            s = c * np.log(1.0 + r)
            imgout[i, j] = s
    return imgout

def Power(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), dtype=np.uint8) + 255
    gamma = 5.0
    c =np.power(L-1, 1.0 - gamma)
    for i in range(M):
        for j in range(N):
            r = imgin[i, j]
            s = c * np.power(1.0*r,gamma)
            imgout[i, j] = s
    return imgout

def PiecewiseLinear(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), dtype=np.uint8) + 255
    rmin,rmax, _, _ = cv2.minMaxLoc(imgin)
    r1 = rmin
    s1 = 0
    r2 = rmax
    s2 = L - 1
    for i in range(M):
        for j in range(N):
            r = imgin[i, j]
            if r < r1:
                s = s1 / r1 * r
            elif r < r2:
                s = (s2 - s1) / (r2 - r1) * (r - r1) + s1
            else:
                # check divide by zero
                if r2 == L - 1:
                    s = L - 1
                else:
                    s = (L - 1 - s2) / (L - 1 - r2) * (r - r2) + s2
            imgout[i, j] = s
    return imgout

def Histogram(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, L), np.uint8) + 255
    h = np.zeros(L, dtype=np.int32)
    for i in range(M):
        for j in range(N):
            r = imgin[i, j]
            h[r] += 1
    p = h / (M * N)
    scale = 3000
    for r in range(L):
        cv2.line(imgout, (r, M), (r, M - int(scale*p[r])), (0, 0, 0))
    return imgout

def HistEqual(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8) + 255
    h = np.zeros(L, dtype=np.int32)
    for i in range(M):
        for j in range(N):
            r = imgin[i, j]
            h[r] += 1
    p = h / (M * N)
    s = np.zeros(L, dtype=np.float32)
    for k in range(L):
        for j in range(k+1):
            s[k] += p[j]
        s[k] = (L - 1) * s[k]

    for i in range(M):
        for j in range(N):
            r = imgin[i, j]
            imgout[i, j] = np.uint8(s[r])
    return imgout