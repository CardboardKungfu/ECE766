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
    
    # Find contours in the binary image, assuming the largest contour corresponds to the sphere
    contours = measure.find_contours(binary_img, 0.5)
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
    
     # Compute the light source directions
    light_dirs_5x3 = np.zeros((5, 3))
    
    for i, img in enumerate(images):
        # Compute the point's coordinates in the image plane
        y_pix, x_pix = np.unravel_index(np.argmax(img), img.shape)
        
        # Compute the corresponding point's coordinates in 3D space
        x = (x_pix - center[1]) / radius
        y = (y_pix - center[0]) / radius
        z = np.sqrt(1 - x**2 - y**2)  # Assuming sphere is centered at (0, 0, 0)

        # # Normalize the vector to get the normal direction
        normal_vector = np.array([x, y, z])
        normal_vector /= np.linalg.norm(normal_vector)
        
        # # Assign the normalized vector to the corresponding row
        light_dirs_5x3[i] = normal_vector * img.max()  # Scale by the brightest pixel value
    
    # raise NotImplementedError
    return light_dirs_5x3

def computeMask(images: List[np.ndarray]) -> np.ndarray:
    # Compute the mask of the object
    # Input:
    #   images - list of N images
    # Output:
    #   mask - HxW binary mask

    bin_imgs = np.stack([img > 0 for img in images], axis=-1)
    mask = np.sum(bin_imgs, axis=-1)
    
    # raise NotImplementedError
    return mask

def computeNormals(light_dirs: np.ndarray, images: List[np.ndarray], mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Compute the surface normals and albedo of the object
    # Input:
    #   light_dirs - Nx3 matrix of light directions
    #   images - list of N images
    #   mask - binary mask
    # Output:
    #   normals - HxWx3 matrix of surface normals
    #   albedo_img - HxW matrix of albedo values

    height, width = mask.shape
    normals = np.zeros((height, width, 3), dtype=np.float64)
    albedo_img = np.zeros((height, width), dtype=np.float64)

    stacked_imgs = np.stack(images, axis=-1)

    for row in range(height):
        for col in range(width):
            if mask[row, col]: # use mask to tell if point is foreground
                I = stacked_imgs[row, col, :].astype(np.float64)
                N = np.linalg.pinv(light_dirs) @ I
                albedo = np.linalg.norm(N)
                normal = N / albedo
                normals[row, col] = normal
                albedo_img[row, col] = albedo

    # raise NotImplementedError
    return normals, albedo_img
