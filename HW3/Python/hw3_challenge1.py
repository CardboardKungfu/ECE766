from PIL import Image, ImageDraw
import numpy as np
import scipy

theta_width = 1
rho_width = 1

def generateHoughAccumulator(edge_image: np.ndarray, theta_num_bins: int, rho_num_bins: int) -> np.ndarray:
    '''
    Generate the Hough accumulator array.
    Arguments:
        edge_image: the edge image.
        theta_num_bins: the number of bins in the theta dimension.
        rho_num_bins: the number of bins in the rho dimension.
    Returns:
        hough_accumulator: the Hough accumulator array.
    '''
    hough_acc = np.zeros((rho_num_bins, theta_num_bins))
    
    height = edge_image.shape[0]
    width = edge_image.shape[1]

    for r in range(height):
        for c in range(width):
            if edge_image[r,c] != 0:
                for theta in range(0, 360, theta_width):
                    rho = c * np.sin(np.deg2rad(theta)) + r * np.cos(np.deg2rad(theta))
                    hough_acc[int(rho / rho_width), int(theta / theta_width)] += 1
                    # if int(rho / rho_width) == 0 or int(theta / theta_width) == 0 or int(rho / rho_width) == hough_acc.shape[0] - 1 or int(theta / theta_width) == hough_acc.shape[1] - 1:
                    #     hough_acc[int(rho / rho_width), int(theta / theta_width)] += 1
                    # else:
                    #     hough_acc[int(rho / rho_width), int(theta / theta_width)] += 1
                    #     hough_acc[int(rho / rho_width) + 1, int(theta / theta_width)] += 0.2
                    #     hough_acc[int(rho / rho_width) - 1, int(theta / theta_width)] += 0.2
                    #     hough_acc[int(rho / rho_width), int(theta / theta_width) + 1] += 0.2
                    #     hough_acc[int(rho / rho_width), int(theta / theta_width) - 1] += 0.2
                    #     hough_acc[int(rho / rho_width) + 1, int(theta / theta_width) + 1] += 0.2
                    #     hough_acc[int(rho / rho_width) + 1, int(theta / theta_width) - 1] += 0.2
                    #     hough_acc[int(rho / rho_width) - 1, int(theta / theta_width) + 1] += 0.2
                    #     hough_acc[int(rho / rho_width) - 1, int(theta / theta_width) - 1] += 0.2

    # scale it back to 255
    max_val = np.max(hough_acc)
    hough_acc = hough_acc * (255 / max_val)

    return hough_acc

def lineFinder(orig_img: np.ndarray, hough_img: np.ndarray, hough_threshold: float):
    '''
    Find the lines in the image.
    Arguments:
        orig_img: the original image.
        hough_img: the Hough image.
        hough_threshold: the threshold for the Hough accumulator array.
    Returns: 
        line_img: PIL image with lines drawn.

    '''
    hough_peaks = hough_img > hough_threshold

    height = hough_peaks.shape[0]
    width = hough_peaks.shape[1]
    line_img = Image.fromarray(orig_img.astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(line_img)
    
    blank_w_lines = Image.fromarray(np.zeros(orig_img.shape).astype(np.uint8)).convert('RGB')
    blank_draw = ImageDraw.Draw(blank_w_lines)

    for r in range(height):
        for c in range(width):
            if hough_peaks[r, c] != 0:
                rho = r * rho_width
                if c == 0: # avoid dividing by zero when theta is zero
                    xp0, yp0, xp1, yp1 = 0, rho, orig_img.shape[1], rho
                else:
                    theta = np.deg2rad(c * theta_width)
                    xp0 = rho / np.sin(theta)
                    yp0, xp1 = 0, 0
                    yp1 = rho / np.cos(theta)
                    if yp1 < 0:
                        xp1 = (rho - (orig_img.shape[0] * np.cos(theta))) / np.sin(theta)
                        yp1 = orig_img.shape[0]

                    if xp0 < 0: 
                        xp0 = (rho - (orig_img.shape[0] * np.cos(theta))) / np.sin(theta)
                        yp0 = orig_img.shape[0]
                        
                draw.line((xp0, yp0, xp1, yp1), fill=128, width=2)
                blank_draw.line((xp0, yp0, xp1, yp1), fill=128, width=2)

    line_img.show()
    return  line_img, blank_w_lines

def lineSegmentFinder(orig_img: np.ndarray, edge_arr: np.ndarray, hough_img: np.ndarray, hough_threshold: float, blank_w_lines: np.ndarray):
    '''
    Find the line segments in the image.
    Arguments:
        orig_img: the original image.
        edge_arr: the edge ndarray.
        hough_img: the Hough image.
        hough_threshold: the threshold for the Hough accumulator array.
        blank_w_lines: hough lines on black background
    Returns:
        line_segement_img: PIL image with line segments drawn.
    '''
    
    # Dilate edge image to capture all lines
    edge_arr = scipy.ndimage.binary_dilation(edge_arr)
    
    # # Increase mask dimensions to match RBG image
    # edge_arr_expanded = np.expand_dims(edge_arr, axis=-1)
    # edge_arr_expanded = np.repeat(edge_arr_expanded, 3, axis=-1)
        
    # # PIL expects a PIL image for a mask. It also doesn't accept bool types
    # masked_lines = edge_arr_expanded * blank_w_lines
    # masked_lines = Image.fromarray(masked_lines)
    # masked_lines.show()

    edge_mask = Image.fromarray(edge_arr)

    blank_w_lines = Image.fromarray(blank_w_lines)

    line_segment_img = Image.composite(orig_img, blank_w_lines, edge_mask)
    # line_segment_img.paste(masked_lines, (0, 0), mask=masked_lines)
    line_segment_img.show()
    return line_segment_img