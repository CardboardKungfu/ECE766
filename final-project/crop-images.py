from PIL import Image, ImageDraw
import json
import numpy as np
from scipy import ndimage

coco_json_path = "dataset\\_annotations.coco.json"
with open(coco_json_path, "r") as file:
    coco_data = json.load(file)

images = coco_data["images"]
annotations = coco_data["annotations"]

count = 0
for i, annotation in enumerate(annotations):
    if count > 250:
        break

    count += 1
    for j, img in enumerate(images):
        if annotation["image_id"] == img["id"]:
            filename = img["file_name"]
            height = img["height"]
            width = img["width"]
            bbox = annotation["bbox"]
            # segmentation = annotation["segmentation"]
            img_PIL = Image.open("dataset\\" + filename)
            background = Image.new("RGB", (width, height), color="black")

            # Create a mask for the object
            mask = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(mask)
            # for segment in segmentation:
            #     draw.polygon(segment, fill=255)
            x1, y1, width, height = bbox
            x2 = x1 + width
            y2 = y1 + height
            draw.rectangle([x1, y1, x2, y2], fill=255) # draw bounding box, not octagon
            
            padded_mask = np.array(mask)
            padded_mask = ndimage.binary_dilation(padded_mask, iterations=5)
            padded_mask = Image.fromarray(padded_mask)
            # Paste the object onto the black background
            background.paste(img_PIL, mask=padded_mask)
            background.save(f"test_mask\\cropped_{filename}")