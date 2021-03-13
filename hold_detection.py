import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


test_img = cv2.imread("./images/hold_test.png",0)
edges = cv2.Canny(test_img, 100,200)

print(edges)

plt.subplot(121),plt.imshow(test_img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
