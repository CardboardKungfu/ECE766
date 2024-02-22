import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

from PIL import Image
import numpy as np

from skimage import measure

def generateLabeledImage(gray_img: np.ndarray, threshold: float) -> np.ndarray:
    '''
    Generates a labeled image from a grayscale image by assigning unique labels to each connected component.
    Arguments:
        gray_img: grayscale image.
        threshold: threshold for the grayscale image.
    Returns:
        labeled_img: the labeled image.
    '''

    bin_img = gray_img > threshold

    labeled = measure.label(bin_img)

    return labeled

def compute2DProperties(labeled_img: np.ndarray) ->  np.ndarray:
    '''
    Compute the 2D properties of each object in labeled image.
    Arguments:
        orig_img: the original image.
        labeled_img: the labeled image.
    Returns:
        obj_db: the object database, where each row contains the properties
            of one object.
    '''
    
    # find number of objects in image based on labeled number
    obj_count = np.max(labeled_img)
    obj_db = np.zeros((obj_count, 6))

    # mask the image to get a single object stand alone to work on
    for count in range(obj_count):
        obj_arr = labeled_img == (count + 1)
        # Image.fromarray(obj_arr).show()

        rows = np.shape(obj_arr)[0]
        cols = np.shape(obj_arr)[1]
        
        # find area, x center, and y center
        area = 0
        x_tmp = 0
        y_tmp = 0

        for i in range(rows):
            for j in range(cols):
                area += obj_arr[i][j]
                x_tmp += j * obj_arr[i][j]
                y_tmp += i * obj_arr[i][j]
        x_center = np.ceil(x_tmp / area)
        y_center = np.ceil(y_tmp / area)
       
        # find a, b, and c
        a = 0
        b = 0
        c = 0

        for i in range(rows):
            for j in range(cols):
                if obj_arr[i][j]:
                    a += (i - y_center)**2
                    b += 2 * (j - x_center) * (i - y_center)
                    c += (j - x_center)**2
        
        # determine orientation, minimum moment of inertia, and roundness
        theta1 = np.arctan2(b, (a - c)) / 2
        theta2 = theta1 + (np.pi / 2)

        tmp_e1 = a * (np.sin(theta1)**2) - b * np.sin(theta1) * np.cos(theta1) + c * (np.cos(theta1)**2)
        tmp_e2 = a * (np.sin(theta2)**2) - b * np.sin(theta2) * np.cos(theta2) + c * (np.cos(theta2)**2)

        if tmp_e1 < tmp_e2:
            E_min = tmp_e1
            E_max = tmp_e2
            orientation = theta1
        else:
            E_max = tmp_e1
            E_min = tmp_e2
            orientation = theta2

        roundness = E_min / E_max

        obj_db[count, 0] = count + 1
        obj_db[count, 1] = x_center
        obj_db[count, 2] = y_center
        obj_db[count, 3] = E_min
        obj_db[count, 4] = orientation
        obj_db[count, 5] = roundness

    return obj_db

def recognizeObjects(orig_img: np.ndarray, labeled_img: np.ndarray, obj_db: np.ndarray, output_fn: str):
    '''
    Recognize the objects in the labeled image and save recognized objects to output_fn
    Arguments:
        orig_img: the original image.
        labeled_img: the labeled image.
        obj_db: the object database, where each row contains the properties 
            of one object.
        output_fn: filename for saving output image with the objects recognized.
    '''
    # create figure
    fig, ax = plt.subplots()
    plt.axis(False)
    ax.imshow(orig_img, cmap='gray')
    
    new_obj_db = compute2DProperties(labeled_img)

    # threshold percentages for second moment of inertia and roundness
    moment_thresh = 0.155
    round_thresh = 0.055

    for old_obj in obj_db:
        for new_obj in new_obj_db:
            if abs((old_obj[3] - new_obj[3]) / old_obj[3]) < moment_thresh and abs((old_obj[5] - new_obj[5]) / old_obj[5]) < round_thresh:
                x = new_obj[1]
                y = new_obj[2]
                ax.plot(x, y)
                
                angle = new_obj[4]
                length = 50

                end_x = x + length * np.sin(angle)
                end_y = y + length * np.cos(angle)
                plt.plot([x, end_x], [y, end_y], color='blue')
                plt.scatter(x, y, color='blue')
    
    plt.savefig(output_fn)
    plt.show()

def hw2_challenge1a():
    import matplotlib.cm as cm
    from skimage.color import label2rgb
    from hw2_challenge1 import generateLabeledImage
    img_list = ['two_objects.png', 'many_objects_1.png', 'many_objects_2.png']
    # Using good ol' fashioned visual inspection, I found 0.5 to perform adequately for all images
    threshold_list = [0.5, 0.5, 0.5]   # You need to find the right thresholds

    for i in range(len(img_list)):
        orig_img = Image.open(f"data/{img_list[i]}")
        orig_img = np.array(orig_img.convert('L')) / 255.
        labeled_img = generateLabeledImage(orig_img, threshold_list[i])
        Image.fromarray(labeled_img.astype(np.uint8)).save(
            f'outputs/labeled_{img_list[i]}')
        
        cmap = np.array(cm.get_cmap('Set1').colors)
        rgb_img = label2rgb(labeled_img, colors=cmap, bg_label=0)
        Image.fromarray((rgb_img * 255).astype(np.uint8)).save(
            f'outputs/rgb_labeled_{img_list[i]}')

def hw2_challenge1b():
    labeled_two_obj = Image.open('outputs/labeled_two_objects.png')
    labeled_two_obj = np.array(labeled_two_obj)
    orig_img = Image.open('data/two_objects.png')
    orig_img = np.array(orig_img.convert('L')) / 255.
    obj_db  = compute2DProperties(labeled_two_obj)
    np.save('outputs/obj_db.npy', obj_db)
    
    labeled_many_objects_1 = Image.open('outputs/labeled_many_objects_1.png')
    labeled_many_objects_1 = np.array(labeled_many_objects_1)
    many_objects_img = Image.open('data/many_objects_1.png')
    many_objects_img = np.array(many_objects_img.convert('L')) / 255.
    many_obj_db  = compute2DProperties(labeled_many_objects_1)
    np.save('outputs/many_obj_db.npy', many_obj_db)

    # TODO: Plot the position and orientation of the objects
    # Use a dot or star to annotate the position and a short line segment originating from the dot for orientation
    # Refer to demoTricksFun.py for examples to draw dots and lines. 
    fig, (ax, ax2) = plt.subplots(1, 2)
    ax.axis(False)
    ax.imshow(orig_img, cmap='gray')
    for i in range(obj_db.shape[0]):
        # plot the position
        x = obj_db[i, 1]
        y = obj_db[i, 2]
        
        ax.plot(x, y)
        
        # plot the orientation
        angle = obj_db[i, 4]
        length = 50

        end_x = x + length * np.sin(angle)
        end_y = y + length * np.cos(angle)
        ax.plot([x, end_x], [y, end_y], color='blue')
        ax.scatter(x, y, color='blue')
    
    ax2.axis(False)
    ax2.imshow(many_objects_img, cmap='gray')
    for i in range(many_obj_db.shape[0]):
        # plot the position
        x = many_obj_db[i, 1]
        y = many_obj_db[i, 2]
        
        ax2.plot(x, y)
        
        # plot the orientation
        angle = many_obj_db[i, 4]
        length = 50

        end_x = x + length * np.sin(angle)
        end_y = y + length * np.cos(angle)
        ax2.plot([x, end_x], [y, end_y], color='blue')
        ax2.scatter(x, y, color='blue')

    fig.savefig('outputs/two_and_many.png')
    plt.show()

def hw2_challenge1c():
    obj_db = np.load('outputs/obj_db.npy')    
    many_obj_db = np.load('outputs/many_obj_db.npy')
    img_list = ['many_objects_1.png', 'many_objects_2.png']

    for i in range(len(img_list)):
        labeled_img = Image.open(f'outputs/labeled_{img_list[i]}')
        labeled_img = np.array(labeled_img)
        orig_img = Image.open(f"data/{img_list[i]}")
        orig_img = np.array(orig_img.convert('L')) / 255.

        # scan through two objects database
        recognizeObjects(orig_img, labeled_img, obj_db,
                         f'outputs/testing1c_{img_list[i]}')
        
        # scan through many objects 1 database
        recognizeObjects(orig_img, labeled_img, many_obj_db,
                         f'outputs/testing1c_many_1_{img_list[i]}')
        
