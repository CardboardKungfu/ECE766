from PIL import Image, ImageDraw
import numpy as np

theta_width = 1
rho_width = 2

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
    hough_acc = np.zeros((rho_num_bins, theta_num_bins), dtype=np.uint8)

    height = edge_image.shape[0]
    width = edge_image.shape[1]

    for r in range(height):
        for c in range(width):
            if edge_image[r,c] != 0:
                for theta in range(0, 180, theta_width):
                    rho = c * np.sin(np.deg2rad(theta)) + r * np.cos(np.deg2rad(theta))
                    hough_acc[int(rho / rho_width), int(theta / theta_width)] += 1

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
    
    for r in range(height):
        for c in range(width):
            if hough_peaks[r, c] != 0:
                if c == 0: # avoid dividing by zero when theta is zero
                    c += 1
                theta = np.deg2rad(c * theta_width)
                rho = r * rho_width
                xp0 = rho / np.sin(theta)
                yp0 = 0
                xp1 = 0
                yp1 = rho / np.cos(theta)
                if yp1 < 0:
                    xp1 = (rho - (orig_img.shape[0] * np.cos(theta))) / np.sin(theta)
                    yp1 = orig_img.shape[0]

                if xp0 < 0: 
                    xp0 = (rho - (orig_img.shape[0] * np.cos(theta))) / np.sin(theta)
                    yp0 = orig_img.shape[0]
                    
                draw.line((xp0, yp0, xp1, yp1), fill=128)

    line_img.show()
    return  line_img

def lineSegmentFinder(orig_img: np.ndarray, edge_img: np.ndarray, hough_img: np.ndarray, hough_threshold: float):
    '''
    Find the line segments in the image.
    Arguments:
        orig_img: the original image.
        edge_img: the edge image.
        hough_img: the Hough image.
        hough_threshold: the threshold for the Hough accumulator array.
    Returns:
        line_segement_img: PIL image with line segments drawn.
    '''
    raise NotImplementedError
