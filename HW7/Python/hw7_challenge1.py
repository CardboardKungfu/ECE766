from PIL import Image
import numpy as np
from typing import Union, Tuple, List

def computeFlow(img1: np.ndarray, img2: np.ndarray, win_radius: int, template_radius: int, grid_MN: List[int]) -> np.ndarray:
    # Compute optical flow using template matching
    # Input:
    #   img1 - HxW matrix of the first image
    #   img2 - HxW matrix of the second image
    #   win_radius - half size of the search window
    #   template_radius - half size of the template window
    #   grid_MN - 1x2 vector for the number of rows and cols in the grid
    # Output:
    #   result - HxWx2 matrix of optical flow vectors
    #     result[:,:,1] is the x component of the flow vectors
    #     result[]:,:,2] is the y component of the flow vectors
    raise NotImplementedError
