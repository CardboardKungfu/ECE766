import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from PIL import Image
import argparse
import numpy as np
from runTests import run_tests

from skimage import feature, filters

theta_width = 1
rho_width = 1

def runHw3():
    # runHw3 is the "main" interface that lets you execute all the 
    # walkthroughs and challenges in this homework. It lists a set of 
    # functions corresponding to the problems that need to be solved.
    #
    # Note that this file also serves as the specifications for the functions 
    # you are asked to implement. In some cases, your submissions will be 
    # auto-graded.  Thus, it is critical that you adhere to all the specified 
    # function signatures.
    #
    # Before your submission, make sure you can run runHw3('all') 
    # without any error.
    #
    # Usage:
    # python runHw3.py                  : list all the registered functions
    # python runHw3.py 'function_name'  : execute a specific test
    # python runHw3.py 'all'            : execute all the registered functions
    parser = argparse.ArgumentParser(
        description='Execute a specific test or all tests.')
    parser.add_argument(
        'function_name', type=str, nargs='?', default='all',
        help='Name of the function to test or "all" to execute all the registered functions')
    args = parser.parse_args()

    # Call test harness
    fun_handles = {'honesty': honesty, 
                   'walkthrough1': walkthrough1,
                   'challenge1a': challenge1a,
                   'challenge1b': challenge1b,
                   'challenge1c': challenge1c,
                   'challenge1d': challenge1d}
    run_tests(args.function_name, fun_handles)


# Academic Honesty Policy
def honesty():
    from signAcademicHonestyPolicy import sign_academic_honesty_policy
    # Type your full name and uni (both in string) to state your agreement 
    # to the Code of Academic Integrity.
    sign_academic_honesty_policy('Jed Pulley', '9085816834')


# Test for Walkthrough 1: Image processing
def walkthrough1():
    from hw3_walkthrough1 import hw3_walkthrough1
    hw3_walkthrough1()


# Tests for Challenge 1: Hough transform
def challenge1a():
    img_list = ['hough_1.png', 'hough_2.png', 'hough_3.png']
    for i, fn in enumerate(img_list):
        img = Image.open(f"data/{fn}")

        # Apply edge detection to grayscale image
        gray_img = img.convert('L')
        gray_img = np.array(gray_img)
        edge_img = feature.canny(gray_img, sigma=2, low_threshold=0.04*np.max(gray_img), high_threshold=0.08*np.max(gray_img))

        # Save the edge detected image
        edge_img = Image.fromarray((edge_img * 255).astype(np.uint8))
        edge_img.save(f'outputs/edge_{fn}')

def challenge1b():
    from hw3_challenge1 import generateHoughAccumulator
    
    ##############
    def scale_min_max(image, part_size):
        a, b = part_size
        # Calculate the number of parts along each dimension
        num_parts_x = image.shape[0] // a
        num_parts_y = image.shape[1] // b
        
        # Initialize the scaled image
        scaled_image = np.zeros_like(image)
        
        for i in range(num_parts_x):
            for j in range(num_parts_y):
                # Extract the current part
                part = image[i*a:(i+1)*a, j*b:(j+1)*b]
                
                # Find the min and max values in the current part
                min_val = np.min(part)
                max_val = np.max(part)
                
                # Avoid division by zero in case min_val == max_val
                if min_val == max_val:
                    scaled_part = np.zeros_like(part)
                else:
                    # Scale the part
                    scaled_part = (part - min_val) / (max_val - min_val) * 255
                
                # Place the scaled part back into the scaled image
                scaled_image[i*a:(i+1)*a, j*b:(j+1)*b] = scaled_part
        
        return scaled_image
        ##############

    img_list = ['hough_1.png', 'hough_2.png', 'hough_3.png']

    theta_num_bins = int(360 / theta_width)

    hough_threshold = [65, 50, 50]

    for i, fn in enumerate(img_list):
        # Load the edge image from challenge1a
        img = Image.open(f'outputs/edge_{fn}')
        img = np.array(img.convert('L'))  # Convert the image to grayscale
        
        # determine number of rho bins based on diagonal length of img
        rho_num_bins = int(np.sqrt(img.shape[0]**2 + img.shape[1]**2) / rho_width)

        hough_accumulator = generateHoughAccumulator(img, theta_num_bins, rho_num_bins)

        hough2 = np.where(hough_accumulator > hough_threshold[i], hough_accumulator, 0)
        part_size = (int(hough2.shape[0]/20),int(hough2.shape[1]/10))  # Size of the parts to split into
        hough_scaled = scale_min_max(hough2, part_size)

        hough_accumulator = Image.fromarray(hough_accumulator.astype(np.uint8))
        hough_accumulator.save(f'outputs/accumulator_{fn}')

        hough_scaled = Image.fromarray(hough_scaled.astype(np.uint8))
        hough_scaled.save(f'outputs/thresh_accumulator_{fn}')

def challenge1c():
    from hw3_challenge1 import lineFinder
    
    hough_threshold = [142, 165, 245]

    img_list = ['hough_1.png', 'hough_2.png', 'hough_3.png']

    for i, fn in enumerate(img_list):
        orig_img = Image.open(f"data/{fn}")
        orig_img = np.array(orig_img.convert('L'))  

        # hough_img = Image.open(f'outputs/accumulator_{fn}')
        hough_img = Image.open(f'outputs/thresh_accumulator_{fn}')
        hough_img = np.array(hough_img.convert('L'))

        line_img, blank_w_lines = lineFinder(orig_img, hough_img, hough_threshold[i])
        line_img.save(f'outputs/line_{fn}')
        blank_w_lines.save(f'outputs/blank_w_lines_{fn}')

def challenge1d():
    from hw3_challenge1 import lineSegmentFinder
    img_list = ['hough_1.png', 'hough_2.png', 'hough_3.png']
    
    hough_threshold = [110, 100, 150]

    for i, fn in enumerate(img_list):
        orig_img = Image.open(f"data/{fn}")
        orig_img = np.array(orig_img.convert('L'))  # Convert the image to grayscale

        # Reapply edge detection for thicker masking
        # edge_img = feature.canny(orig_img, sigma=2.4, low_threshold=22, high_threshold=25)
        edge_img = Image.open(f'outputs/edge_{fn}')
        edge_img = np.array(edge_img.convert('L'))  # Convert the image to grayscale
        
        # Bring in hough lines on black background
        blank_w_lines = Image.open(f'outputs/blank_w_lines_{fn}')
        blank_w_lines = np.array(blank_w_lines)

        hough_img = Image.open(f'outputs/accumulator_{fn}')
        hough_img = np.array(hough_img.convert('L'))  # Convert the image to grayscale

        line_segement_img = lineSegmentFinder(orig_img, edge_img, hough_img, hough_threshold[i], blank_w_lines)
        line_segement_img.save(f'outputs/croppedline_{fn}')

if __name__ == '__main__':
    runHw3()