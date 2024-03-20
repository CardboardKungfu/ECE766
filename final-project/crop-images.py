from PIL import Image, ImageDraw
import json
import numpy as np

coco_json_path = "dataset\\_annotations.coco.json"
with open(coco_json_path, "r") as file:
    coco_data = json.load(file)

images = coco_data["images"]
annotations = coco_data["annotations"]

for i, annotation in enumerate(annotations):
    for j, img in enumerate(images):
        if annotation["image_id"] == img["id"]:
            filename = img["file_name"]
            height = img["height"]
            width = img["width"]
            bbox = annotation["bbox"]
            segmentation = annotation["segmentation"]
            img_PIL = Image.open("dataset\\" + filename)
            background = Image.new("RGB", (width, height), color="black")

            # Create a mask for the object
            mask = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(mask)
            for segment in segmentation:
                draw.polygon(segment, fill=255)
            
            # Paste the object onto the black background
            background.paste(img_PIL, mask=mask)
            background.save(f"cropped-dataset\\cropped_{filename}")