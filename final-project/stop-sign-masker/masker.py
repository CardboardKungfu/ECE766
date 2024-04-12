import cv2
import numpy as np

def filter_img(path):    
    # Load RGB image
    rgb_image = cv2.imread(path)  

    # Convert RGB image to HSV
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

    # Filter image based on min HSV and max HSV values
    lower_thresh = np.array([0,100,50])
    upper_thresh = np.array([12,255,255])
    return cv2.inRange(hsv_image, lower_thresh, upper_thresh)

def edges(filtered_img):    
    median = cv2.medianBlur(filtered_img, 5)
    closed = cv2.morphologyEx(median, cv2.MORPH_CLOSE, (3, 3))
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, (5, 5))
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, (7, 7))
    return cv2.Canny(closed, 50, 150)  # Using Canny edge detection

def draw_hough_lines(image, edges):
    # Perform Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    # Draw the detected lines on a copy of the original image
    image_with_lines = np.copy(image)
    if lines is not None:
        # Sort the lines by their length (rho parameter in Hough space)
        lines = sorted(lines, key=lambda x: x[0][0], reverse=True)[:8]
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            image_with_lines = cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 10)

    return image_with_lines

paths = ["stop1.jpg", "stop2.jpg", "stop3.jpg", "stop4.jpg", "stop5.jpg"]

for path in paths:
    image = cv2.imread(path)
    
    # Display the filtered RGB image
    cv2.imshow("Original Image", image)
    
    filtered = filter_img(path)
    # cv2.imshow('Filtered Image', filtered)

    edge_img = edges(filtered)
    cv2.imshow('Canny Image', edge_img)

    lines = draw_hough_lines(image, edge_img)
    cv2.imshow("Hough Lines", lines)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
