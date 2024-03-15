from PIL import Image
import numpy as np
from typing import Union, Tuple, List

def findSphere(img: np.ndarray) -> Tuple[np.ndarray, float]:
    # Find the center and radius of the sphere
    # Input:
    #   img - the image of the sphere
    # Output:
    #   center - 2x1 vector of the center of the sphere
    #   radius - radius of the sphere
    from skimage import filters, measure

    # Binarize the image using Otsu's thresholding method
    threshold_value = filters.threshold_otsu(img)
    binary_img = img > threshold_value
    
    # Find contours in the binary image
    contours = measure.find_contours(binary_img, 0.5)
    
    # Assuming the largest contour corresponds to the sphere
    largest_contour = max(contours, key=len)
    
    # Compute the centroid of the largest contour
    centroid = np.mean(largest_contour, axis=0).astype(np.int)
    
    # Compute the radius of the sphere from its area
    area = measure.regionprops(binary_img.astype(int))[0].area
    radius = np.sqrt(area / np.pi)
    
    # raise NotImplementedError
    return centroid, radius

def computeLightDirections(center: np.ndarray, radius: float, images: List[np.ndarray]) -> np.ndarray:
    # Compute the light source directions
    # Input:
    #   center - 2x1 vector of the center of the sphere
    #   radius - radius of the sphere
    #   images - list of N images
    # Output:
    #   light_dirs_5x3 - 5x3 matrix of light source directions
    raise NotImplementedError

def computeMask(images: List[np.ndarray]) -> np.ndarray:
    # Compute the mask of the object
    # Input:
    #   images - list of N images
    # Output:
    #   mask - HxW binary mask
    raise NotImplementedError

def computeNormals(light_dirs: np.ndarray, images: List[np.ndarray], mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Compute the surface normals and albedo of the object
    # Input:
    #   light_dirs - Nx3 matrix of light directions
    #   images - list of N images
    #   mask - binary mask
    # Output:
    #   normals - HxWx3 matrix of surface normals
    #   albedo_img - HxW matrix of albedo values
    raise NotImplementedError

