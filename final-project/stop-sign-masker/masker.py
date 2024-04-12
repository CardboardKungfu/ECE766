import cv2
import numpy as np

def rgb_to_sbh(rgb_image):
    # Convert RGB image to HSV
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

    return hsv_image

# Load RGB image
rgb_image = cv2.imread('stop3.jpg')  

# Convert RGB image to HSV
hsv_image = rgb_to_sbh(rgb_image)

# Filter HSV image based on conditions
min_hue = 0        # Minimum hue (red)
max_hue = 10       # Maximum hue (red)
min_sat = 100      # Minimum saturation
max_sat = 255      # Maximum saturation
min_val = 100      # Minimum value (brightness)
max_val = 255      # Maximum value (brightness)

filtered_image = cv2.inRange(hsv_image, (min_hue, min_sat, min_val), (max_hue, max_sat, max_val))
# filtered_image = cv2.resize(filtered_image, (0, 0), fx = 0.5, fy = 0.5)
# # Convert filtered HSV image to RGB
# filtered_rgb_image = cv2.cvtColor(filtered_image, cv2.COLOR_HSV2RGB)

# Display the filtered RGB image
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
