
from scipy.ndimage import gaussian_filter
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math as M


# split and save rgb values with cv2.split


# Luminance intensity image


# def splitRGB(image, num):
#     Lr = []
#     Lg = []
#     Lb = []
#     for i in range(len(image)):
#         for j in range(len(image[i])):
#             (r, g, b) = image[i][j]
#             Lr.append(r)
#             Lg.append(g)
#             Lb.append(b)
#     if num == 1:
#         return Lr
#     elif num == 2:
#         return Lg
#     elif num == 3:
#         return Lb
image = cv2.imread("CIS_465\html\images\sampleimg.png", 1)
resize = cv2.resize(image, [600, 400], interpolation=cv2.INTER_AREA)


def NTSCgreyscale(image, v):
    fill = 0
    list1 = []
    list2 = []
    while fill < len(image):
        list1.append([])
        list2.append([])
        fill += 1
    for i in range(0, len(image)):
        for j in range(0, len(image[i])):
            (r, g, b) = image[i][j]
            x1 = 76.245*r
            x2 = 149.685*g
            x3 = 29.071*b
            x = x1+x2+x3/255
            list1[i].append(x/255)
            list2[i].append(x)

    if v == 0:
        return np.array(list1)
    else:
        return np.array(list2)

# Adaptive luminance enhancement


def findzvalues(image):
    fill = 0
    bilist = []
    while fill < len(image):
        bilist.append([])
        fill += 1
    L = findLvalue(image)
    z = 1
    if L <= 50:
        z = 0
    elif L > 50 and L <= 150:
        z = (L-50)/100
    elif L > 150:
        z = 1
    for i in range(len(image)):
        for j in range(len(image[i])):
            x = image[i][j]
            l = (M.pow(x, (0.75*z)+0.25))
            m = (1-x)*(0.4)*(1-z)
            n = (M.pow(x, 2-z))
            o = l+m+n
            bilist[i].append(o/2)

    return np.array(bilist)


def findLvalue(image):
    row, col = image.shape
    max = (row*col)/10
    z = 0
    L = 0
    x = cv2.calcHist([image], [0], None, [256], [0, 255])
    for i in x:
        z = z+i
        L = L + 1
        if z >= max:
            break
    return L


def contEnhance(image, img):

    fill = 0
    output = []

    while fill < len(image):
        output.append([])

        fill += 1

    im = NTSCgreyscale(img, 1)

    Z = np.std(image)
    p = 0
    gauss = gaussian_filter(im, sigma=35)
    if Z <= 3:
        p = 3
    elif Z > 3 and Z < 10:
        p = ((27 - (2*Z))/7)
    else:
        p = 1
    ibar = findzvalues(image)
    for k in range(0, len(image)):
        for l in range(0, len(image[k])):
            try:
                a = int(gauss[k][l])/((int(image[k][l])*255))
            except ZeroDivisionError:
                a = int(gauss[k][l]/1)
            E = M.pow(a, p)
            print(a)
            S = M.pow(255*ibar[k][l], E)
            output[k].append(S)

    return np.array(output)


def restore(image, L1, L2, L3):
    fill = 0
    bilist = []
    while fill < len(image):
        bilist.append([])
        fill += 1
    S = contEnhance(image)
    lamb = 0.5
    num = 0
    for i in range(len(image)):
        for j in range(len(image[i])):
            x = S[i][j]
            y = image[i][j] + 1
            Sr = x*(L1[num]/y)*lamb
            Sg = x*(L2[num]/y)*lamb
            Sb = x*(L3[num]/y)*lamb

            bilist[i].append([Sr, Sg, Sb])
            num += 1
    return np.array(bilist)


# can use built in histogram and CDF functions



Lr = splitRGB(resize, 1)
Lg = splitRGB(resize, 2)
Lb = splitRGB(resize, 3)

resize = resize.astype(np.uint8)
newimg = NTSCgreyscale(resize, 1)
newimg = newimg.astype(np.uint8)

output = contEnhance(newimg, resize)
print(output)
#output = restore(newimg, Lr, Lg, Lb)


output = output.astype(np.uint8)

cv2.imshow('Transformed', output)
cv2.imshow('Original', newimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
