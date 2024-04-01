from PIL import Image
import numpy as np
from typing import Union, Tuple, List

from scipy import ndimage

def generateIndexMap(gray_list: List[np.ndarray], w_size: int) -> np.ndarray:
    # Generate an index map for the refocusing application
    # Input:
    #   gray_list - List of K gray-scale images
    #   w_size - half window size used for focus measure computation
    # Output:
    #   index_map - mxn index map
    #               index_map(i, j) is the index of the image that is in focus
    #               at pixel (i, j)
    
    height, width = gray_list[0].shape
    index_map = np.empty((height, width))
    gray_stack = np.stack(gray_list, axis=2)

    # for i, j in np.ndindex((height, width)):
    for j in range(width):
        for i in range(height):
            # Clip window so it doesn't fall out of bounds
            left = np.clip(i - w_size, 0, width - 1)
            right = np.clip(i + w_size, 0, width - 1)
            top = np.clip(j - w_size, 0, height - 1)
            bottom = np.clip(j + w_size, 0, height - 1)
            
            # Check if window size is non-zero
            if left < right and top < bottom:
                # Slice the ixj column of the 3d stack
                window_column = gray_stack[left:right, top:bottom, :]
                
                # Find the index of the maximum laplacian value over each of the layers in our window column
                # index_map[i, j] = np.argmax(np.apply_over_axes(ndimage.laplace, window_column, [0,1]))
                index_map[i, j] = np.argmax(np.apply_along_axis(ndimage.laplace, axis=2, arr=window_column))

    # raise NotImplementedError
    return index_map


def loadFocalStack(focal_stack_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    # Load the focal stack
    # Input:
    #   focal_stack_dir - directory of the focal stack
    # Output:
    #   rgb_list - List of RGB images for varying focal lengths
    #   gray_list - List of gray-scale images for varying focal lengths
    
    import glob
    rgb_list = []
    gray_list = []
    for filename in glob.glob(focal_stack_dir + '/*.jpg'): #assuming gif
        img = Image.open(filename)
        rgb_list.append(np.array(img))
        gray_list.append(np.array(img.convert('L')))
    
    # raise NotImplementedError
    return (rgb_list, gray_list)


def refocusApp(rgb_list: List[np.ndarray], depth_map: np.ndarray) -> None:
    # Refocusing application
    # Input:
    #   rgb_list - List of RGB images for varying focal lengths
    #   depth_map - mxn index map
    #               depth_map(i, j) is the index of the image that is in focus
    #               at pixel (i, j)
    raise NotImplementedError
