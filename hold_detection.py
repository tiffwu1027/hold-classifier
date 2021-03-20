import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import random as rng


def detectHolds(img):
    # Canny edge detector to get the edges
    gaussian_img = cv2.GaussianBlur(img, (5,5), 0)
    # median_img = cv2.medianBlur(img, 5)
    gray = cv2.cvtColor(gaussian_img,cv2.COLOR_BGR2GRAY)
    otsu, _ = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    edges = cv2.Canny(gaussian_img, otsu,otsu * 2, L2gradient = True)
    plotCanny(edges, gaussian_img)

    # Get contours from edge detection and get objects with convex hull
     # Find contours
    contours,_ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find the convex hull object for each contour
    # hulls = map(cv2.convexHull, contours)
    # hulls = np.array(list(hulls))
    
    # mask = np.zeros(img.shape,np.uint8)


    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)

    # cv2.drawContours(mask,hulls,-1,[255,255,255],-1)
    # # Draw contours + hull results
    drawing = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (255, 255, 255)
        cv2.drawContours(drawing, contours, i, color)
        cv2.drawContours(drawing, hull_list, i, color)
    # Show in a window
    # cv2.imshow('Contours', drawing)
    # cv2.waitKey(0)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 255

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 100

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.05

    detector = cv2.SimpleBlobDetector_create(params)

    # detector = cv2.SimpleBlobDetector()
    keypoints = detector.detect(drawing)

    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)



def plotCanny(edges, img):
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()


rng.seed(12345)
test_img = cv2.imread("./images/hold_test_2.png",1)
hold_w_person_img = cv2.imread("./images/hold_with_person.jpg",1)
hold_w_person2_img = cv2.imread("./images/hold_with_person2.jpg",1)
hold_depth_img = cv2.imread("./images/depth.jpg",1)

detectHolds(test_img)
detectHolds(hold_w_person_img)
detectHolds(hold_w_person2_img)
detectHolds(hold_depth_img)
