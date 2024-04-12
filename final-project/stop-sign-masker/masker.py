import cv2
import numpy as np


def filter_img(path):    
    # Load RGB image
    rgb_image = cv2.imread(path)  

    # Convert RGB image to HSV
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

    # Filter image based on min HSV and max HSV values
    lower_thresh = np.array([0,100,50])
    upper_thresh = np.array([10,255,255])
    return cv2.inRange(hsv_image, lower_thresh, upper_thresh)

def edge_detection(filtered_img):    
    median = cv2.medianBlur(filtered_img, 5)
    closed = cv2.morphologyEx(median, cv2.MORPH_CLOSE, (3, 3))
    closed = cv2.morphologyEx(median, cv2.MORPH_CLOSE, (5, 5))
    closed = cv2.morphologyEx(median, cv2.MORPH_CLOSE, (7, 7))
    return cv2.Canny(closed, 50, 150)  # Using Canny edge detection

# filtered_image = cv2.resize(filtered_image, (0, 0), fx = 0.5, fy = 0.5)

paths = ["stop1.jpg", "stop2.jpg", "stop3.jpg", "stop4.jpg", "stop5.jpg"]

for path in paths:
    # Display the filtered RGB image
    filtered = filter_img(path)
    cv2.imshow('Filtered Image', filtered)

    sobel = edge_detection(filtered)
    cv2.imshow('Canny Image', sobel)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
