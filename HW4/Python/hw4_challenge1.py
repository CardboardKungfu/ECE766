from PIL import Image, ImageDraw
import numpy as np
from typing import Union, Tuple, List


def computeHomography(src_pts_nx2: np.ndarray, dest_pts_nx2: np.ndarray) -> np.ndarray:
    '''
    Compute the homography matrix.
    Arguments:
        src_pts_nx2: the coordinates of the source points (nx2 numpy array).
        dest_pts_nx2: the coordinates of the destination points (nx2 numpy array).
    Returns:
        H_3x3: the homography matrix (3x3 numpy array).
    '''

    # Split points by column then flatten into a single list
    x_s, y_s = np.hsplit(src_pts_nx2, 2)
    x_s, y_s = x_s.flatten(), y_s.flatten()

    x_d, y_d = np.hsplit(dest_pts_nx2, 2)
    x_d, y_d = x_d.flatten(), y_d.flatten()

    # A = np.array([
    #     [x_s[0], y_s[0], 1, 0, 0, 0, -x_d[0] * x_s[0], -x_d[0] * y_s[0], -x_d[0]],
    #     [0, 0, 0, x_s[0], y_s[0], 1, -y_d[0] * x_s[0], -y_d[0] * y_s[0], -y_d[0]],
    #     [x_s[1], y_s[1], 1, 0, 0, 0, -x_d[1] * x_s[1], -x_d[1] * y_s[1], -x_d[1]],
    #     [0, 0, 0, x_s[1], y_s[1], 1, -y_d[1] * x_s[1], -y_d[1] * y_s[1], -y_d[1]],
    #     [x_s[2], y_s[2], 1, 0, 0, 0, -x_d[2] * x_s[2], -x_d[2] * y_s[2], -x_d[2]],
    #     [0, 0, 0, x_s[2], y_s[2], 1, -y_d[2] * x_s[2], -y_d[2] * y_s[2], -y_d[2]],
    #     [x_s[3], y_s[3], 1, 0, 0, 0, -x_d[3] * x_s[3], -x_d[3] * y_s[3], -x_d[3]],
    #     [0, 0, 0, x_s[3], y_s[3], 1, -y_d[3] * x_s[3], -y_d[3] * y_s[3], -y_d[3]]
    #     ])

    A = np.zeros((8, 9))
    j = 0
    for i in range(src_pts_nx2.shape[0]):
        A[j,:] = [x_s[i], y_s[i], 1, 0, 0, 0, -x_d[i] * x_s[i], -x_d[i] * y_s[i], -x_d[i]]
        A[j+1,:] = [0, 0, 0, x_s[i], y_s[i], 1, -y_d[i] * x_s[i], -y_d[i] * y_s[i], -y_d[i]]
        j += 2

    eig_vals, eig_vecs = np.linalg.eig(A.T @ A)
    min_index = np.argmin(eig_vals)
    H = eig_vecs[:, min_index].reshape((3,3))
    
    H = H / H[2,2] # normalize homography matrix

    return H


def applyHomography(H_3x3: np.ndarray, src_pts_nx2: np.ndarray) ->  np.ndarray:
    '''
    Apply the homography matrix to the source points.
    Arguments:
        H_3x3: the homography matrix (3x3 numpy array).
        src_pts_nx2: the coordinates of the source points (nx2 numpy array).
    Returns:
        dest_pts_nx2: the coordinates of the destination points (nx2 numpy array).
    '''
    homo_src = np.insert(src_pts_nx2, 2, 1, axis=1)
    dest_mat = H_3x3 @ homo_src.T
    z_tilde = dest_mat[-1]
    dest_points_homo = (dest_mat / z_tilde).T
    dest_points = dest_points_homo[:, 0:2]

    # raise NotImplementedError
    return dest_points


def showCorrespondence(img1: Image.Image, img2: Image.Image, pts1_nx2: np.ndarray, pts2_nx2: np.ndarray) -> Image.Image:
    '''
    Show the correspondences between the two images.
    Arguments:
        img1: the first image.
        img2: the second image.
        pts1_nx2: the coordinates of the points in the first image (nx2 numpy array).
        pts2_nx2: the coordinates of the points in the second image (nx2 numpy array).
    Returns:
        result: image depicting the correspondences.
    '''

    width = img1.width + img2.width
    height = max(img1.height, img2.height)

    new_img = Image.new("RGB", (width, height))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))

    draw = ImageDraw.Draw(new_img)
    for src_pt, dest_pt in zip(pts1_nx2, pts2_nx2):
        draw.line([tuple(src_pt), (dest_pt[0] + img1.width, dest_pt[1])], fill="red", width=1)
    
    new_img.show()
    # raise NotImplementedError
    return new_img

def backwardWarpImg(src_img: np.ndarray, destToSrc_H: np.ndarray, canvas_shape: Union[Tuple, List]) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Backward warp the source image to the destination canvas based on the
    homography given by destToSrc_H. 
    Arguments:
        src_img: the source image.
        destToSrc_H: the homography that maps points from the destination
            canvas to the source image.
        canvas_shape: shape of the destination canvas (height, width).
    Returns:
        dest_img: the warped source image.
        dest_mask: a mask indicating sourced pixels. pixels within the
            source image are 1, pixels outside are 0.
    '''
    canvas = np.zeros(canvas_shape, dtype=src_img.dtype)
    src_height, src_width = src_img.shape[:2]
    for x_d in range(canvas.shape[0]):
        for y_d in range(canvas.shape[1]):
            homo_coords = destToSrc_H @ np.array([y_d, x_d, 1]).reshape((3,1))
            y_s, x_s, _ = (homo_coords / homo_coords[2]).flatten()
            # Ignore if transformed pixel leads us outside of our original image, 
            if x_s < 0 or y_s < 0 or x_s > src_height or y_s > src_width:
                continue
            canvas[x_d, y_d] = src_img[x_s.astype(int), y_s.astype(int)]

    blank_mask = canvas > 0
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(canvas)
    # plt.title('Van Gogh in Osaka')
    plt.show()

    return blank_mask, canvas
    raise NotImplementedError


def blendImagePair(img1: List[Image.Image], mask1: List[Image.Image], img2: Image.Image, mask2: Image.Image, mode: str) -> Image.Image:
    '''
    Blend the warped images based on the masks.
    Arguments:
        img1: list of source images.
        mask1: list of source masks.
        img2: destination image.
        mask2: destination mask.
        mode: either 'overlay' or 'blend'
    Returns:
        out_img: blended image.
    '''
    raise NotImplementedError

def runRANSAC(src_pt: np.ndarray, dest_pt: np.ndarray, ransac_n: int, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Run the RANSAC algorithm to find the inliers between the source and
    destination points.
    Arguments:
        src_pt: the coordinates of the source points (nx2 numpy array).
        dest_pt: the coordinates of the destination points (nx2 numpy array).
        ransac_n: the number of iterations to run RANSAC.
        eps: the threshold for considering a point to be an inlier.
    Returns:
        inliers_id: the indices of the inliers (kx1 numpy array).
        H: the homography matrix (3x3 numpy array).
    '''
    raise NotImplementedError

def stitchImg(*args: Image.Image) -> Image.Image:
    '''
    Stitch a list of images.
    Arguments:
        args: a variable number of input images.
    Returns:
        stitched_img: the stitched image.
    '''
    raise NotImplementedError
