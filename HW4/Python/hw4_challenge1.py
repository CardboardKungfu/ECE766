from PIL import Image, ImageDraw
import numpy as np
from scipy import ndimage
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

    A = np.zeros((src_pts_nx2.shape[0] * 2, 9))
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
    height_d, width_d = canvas_shape
    canvas = np.zeros((height_d, width_d, 3), dtype=np.float32)
    canvas_mask = np.zeros((height_d, width_d), dtype=np.uint8)
    src_height, src_width = src_img.shape[:2]

    for y_d in range(height_d):
        for x_d in range(width_d):
            homo_coords = destToSrc_H @ np.array([x_d, y_d, 1])
            homo_coords /= homo_coords[2]
            x_s, y_s = homo_coords[:2]
            # Ignore if transformed pixel leads us outside of our original image
            if 0 <= x_s < src_width and 0 <= y_s < src_height:
                canvas[y_d, x_d] = src_img[int(y_s), int(x_s)]
                canvas_mask[y_d, x_d] = 1

    # raise NotImplementedError
    return canvas_mask, canvas

def blendImagePair(img1: np.ndarray, mask1: np.ndarray, img2: np.ndarray, mask2: np.ndarray, mode: str) -> np.ndarray:
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
    out_img = np.zeros(img1.shape)
    mask1 = mask1 / 255
    mask2 = mask2 / 255

    if mode == "overlay":
        out_img = np.where(mask2[:, :, None] > 0, img2, img1)        
    elif mode == "blend":
        img1_weighted = ndimage.distance_transform_edt(mask1)
        img1_weighted[img1_weighted == 0] = 0.000001

        img2_weighted = ndimage.distance_transform_edt(mask2)
        img2_weighted[img2_weighted == 0] = 0.000001

        image_blended = (img1 * img1_weighted[..., np.newaxis] + img2 * img2_weighted[..., np.newaxis]) / (img1_weighted + img2_weighted)[..., np.newaxis]
        out_img = image_blended.astype(np.uint8)
    else:
        raise ValueError("Invalid blending mode. Choose 'overlay' or 'blend'.")

    # raise NotImplementedError
    return out_img

def runRANSAC(src_pts: np.ndarray, dest_pts: np.ndarray, ransac_n: int, eps: float) -> Tuple[np.ndarray, np.ndarray]:
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

    # Stabilize the ransac results
    np.random.seed(1)

    sample_size = 4
    best_H = None
    max_inliers = 0
    inlier_indices = None
    sample_indices = np.arange(len(src_pts))
    for _ in range(ransac_n):
        indices = np.random.choice(sample_indices, sample_size, replace=False)
        src_sample = src_pts[indices]
        dest_sample = dest_pts[indices]

        cand_H = computeHomography(src_sample, dest_sample)
        cand_dest = applyHomography(cand_H, src_pts)

        distances = np.linalg.norm(cand_dest - dest_pts, axis=1)

        inliers = np.where(distances < eps)[0]
        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            inlier_indices = inliers
            best_H = cand_H

    # raise NotImplementedError
    return inlier_indices, best_H

def stitchImg(*args: np.ndarray) -> np.ndarray:
    '''
    Stitch a list of images.
    Arguments:
        args: a variable number of input images.
    Returns:
        stitched_img: the stitched image.
    '''

    '''
    Start with two images
    Run RANSAC to get best homography
    Pass in inverse homography to backwardwarp
    Overlap images based on points in common
    Blend two images together
    '''
    from helpers import genSIFTMatches
    
    start_img = args[0]
    index = (start_img == 0).all(axis=-1)
    # Alter zero elements to a low value
    start_img[index] = 1/255.0

    for curr_img in args[1:]:
        
        index = (curr_img == 0).all(axis=-1)
        curr_img[index] = 1/255.0

        x_s, x_d = genSIFTMatches(curr_img, start_img)
        # genSIFT returns (y,x) not (x,y), so flip them around
        x_s, x_d = x_s[:, [1, 0]], x_d[:, [1, 0]]
        # Compute homography from the new image to the current image
        _, H = runRANSAC(x_s, x_d, 300, 1)
        # _, H = runRANSAC(x_s, x_d, 1000, 0.6)

        # Find corner points to determine after-warped size of canvas
        width, height = curr_img.shape[1], curr_img.shape[0]
        corners = np.array([
            [0        , 0         ],
            [width - 1, 0         ],
            [width - 1, height - 1],
            [0        , height - 1]
        ])

        corners_warped = applyHomography(H, corners)

        min_x = np.min(corners_warped[:, 0])
        min_y = np.min(corners_warped[:, 1])
        tl_x = 0 if min_x >= 0 else -min_x
        tl_y = 0 if min_y >= 0 else -min_y
        new_start_x = int(np.round(tl_x))
        new_start_y = int(np.round(tl_y))
        max_width  = max(int(np.ceil(np.max(corners_warped[:, 0]) + new_start_x)), start_img.shape[1] + new_start_x) 
        max_height = max(int(np.ceil(np.max(corners_warped[:, 1]) + new_start_y)), start_img.shape[0] + new_start_y)

        canvas_shape = (max_height, max_width, 3)
        curr_canvas = np.zeros(canvas_shape)
        curr_canvas[new_start_y:start_img.shape[0] + new_start_y, new_start_x:start_img.shape[1] + new_start_x, :] = start_img

        canvas_mask = np.any(curr_canvas != 0, axis=-1).astype(int)
        new_H = update_homography(H, new_start_x, new_start_y)
        warped_mask, warped_img = backwardWarpImg(curr_img, np.linalg.inv(new_H), canvas_shape[:2])

        canvas_mask = canvas_mask.squeeze()

        img_blended = blendImagePair((curr_canvas * 255).astype(np.uint8), (canvas_mask * 255).astype(np.uint8), (warped_img * 255).astype(np.uint8), (warped_mask * 255).astype(np.uint8), mode='blend')
        start_img = img_blended / 255.0
        index = (img_blended == 1).all(axis=-1)
        img_blended[index] = 0
    
    out_img = Image.fromarray(img_blended.astype(np.uint8))
    # raise NotImplementedError
    return out_img

def update_homography(H, tx, ty):
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]])
    new_H = np.dot(T, H)
    return new_H