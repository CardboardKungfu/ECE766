import cv2
import numpy as np


def filter_img(path):    
    # Load RGB image
    rgb_image = cv2.imread(path)  

    # Convert RGB image to HSV
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

    # Filter image based on min HSV and max HSV values
    return cv2.inRange(hsv_image, (0, 100, 0), (10, 255, 255))

# filtered_image = cv2.resize(filtered_image, (0, 0), fx = 0.5, fy = 0.5)

paths = ["stop1.jpg", "stop2.jpg", "stop3.jpg", "organic-stop.jpeg"]

for path in paths:
    # Display the filtered RGB image
    cv2.imshow('Filtered Image', filter_img(path))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
