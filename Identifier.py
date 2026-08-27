import torch
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def load_objectifier():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    print("Loading SAM 2...")

    predictor = SAM2ImagePredictor.from_pretrained(
        "facebook/sam2-hiera-large"
    )

    print("Model loaded!")
    return predictor,device
def object_detector(image,device,predictor):
    image_np = np.array(image)

    mask_generator = SAM2AutomaticMaskGenerator(
        predictor.model
    )

    print("Finding objects...")

    masks = mask_generator.generate(image_np)

    print("Found:", len(masks), "masks")

    plt.figure(figsize=(12, 8))
    plt.imshow(image)

    ax = plt.gca()

    for x,mask_data in enumerate(masks):

        mask = mask_data["segmentation"]

        x1, y1, w, h = mask_data["bbox"]
        x2 = w+x1
        y2 = h+y1
        image
        res = image.crop([
        max(x1 - 10, 0),
        max(y1 - 10, 0),
        min(x2 + 10, image.width),
        min(y2 + 10, image.height)])
        res.save(f"object/{x}.jpg")
        rect = plt.Rectangle(
            (x1, y1),
            w,
            h,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )

        ax.add_patch(rect)

    plt.savefig("detections.png", bbox_inches="tight")
    plt.close()
    return len(masks)