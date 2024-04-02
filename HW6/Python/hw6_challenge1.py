from PIL import Image
import numpy as np
from typing import Union, Tuple, List

from scipy import signal


def generateIndexMap(gray_list: List[np.ndarray], w_size: int) -> np.ndarray:
    # Generate an index map for the refocusing application
    # Input:
    #   gray_list - List of K gray-scale images
    #   w_size - half window size used for focus measure computation
    # Output:
    #   index_map - mxn index map
    #               index_map(i, j) is the index of the image that is in focus
    #               at pixel (i, j)
    
    laplace_ker = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ])
    ones_ker = np.ones((w_size, w_size))

    lap_imgs = [np.square(signal.convolve2d(img, laplace_ker, mode='same', boundary='symm')) for img in gray_list]
    lap_imgs = [signal.convolve2d(img, ones_ker, mode='same', boundary='symm') for img in lap_imgs]

    avg_ker = ones_ker / np.sum(ones_ker)
    lap_imgs = [signal.convolve2d(img, avg_ker, mode='same', boundary='symm') for img in lap_imgs]

    lap_stack = np.stack(lap_imgs, axis=2)
    index_map = np.empty(lap_stack.shape[:2])

    index_map = np.argmax(lap_stack, axis=2)

    # raise NotImplementedError
    return index_map


def loadFocalStack(focal_stack_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    # Load the focal stack
    # Input:
    #   focal_stack_dir - directory of the focal stack
    # Output:
    #   rgb_list - List of RGB images for varying focal lengths
    #   gray_list - List of gray-scale images for varying focal lengths
    
    rgb_list = []
    gray_list = []
    for i in range(25):
        filename = f"data\\stack\\frame{i+1}.jpg"
        img = Image.open(filename)
        rgb_list.append(np.array(img))
        gray_list.append(np.array(img.convert('L')))
    
    # raise NotImplementedError
    return (rgb_list, gray_list)


def refocusApp(rgb_list: List[np.ndarray], depth_map: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    im = ax.imshow(rgb_list[0])  # Display the first image in the list

    while True:
        # Ask user to choose a scene point
        point = plt.ginput(1, timeout=-1, show_clicks=False)
        if not point:  # If user clicks outside the image, exit the loop
            break
            
        print(point)
        scene_point = tuple(map(int, point[0]))

        # Check if the scene point falls within the image dimensions
        if (0 <= scene_point[0] < depth_map.shape[1]) and (0 <= scene_point[1] < depth_map.shape[0]):
            # Refocus the image to the chosen scene point
            i, j = scene_point
            focal_index = depth_map[j, i]

            # Return the image focused at the scene point
            refocused_image = rgb_list[int(focal_index)]

            # Update the displayed image
            im.set_data(refocused_image)
            plt.draw()
            plt.pause(0.1)  # Pause to update the plot
        else:
            print("Selected point is outside the image.")

    plt.close()