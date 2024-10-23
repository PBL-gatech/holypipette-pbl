import torch
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import os

class PipetteSegmenter:
    def __init__(self, model_path, device='mps'):
        self.sam = sam_model_registry["vit_h"](checkpoint=model_path).to(device)
        self.predictor = SamPredictor(self.sam)
        self.device = device

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        self.predictor.set_image(image)
        return image

    def get_mask(self, input_point):
        input_label = np.array([1])
        masks, scores, _ = self.predictor.predict(
            point_coords=np.array([input_point]),
            point_labels=input_label,
            multimask_output=False
        )
        return masks[0], scores[0]

    def save_mask(self, mask, output_path):
        cv2.imwrite(output_path, (mask * 255).astype(np.uint8))

    def overlay_mask(self, image, mask):
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = [0, 255, 0]  # Green mask
        return cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

if __name__ == "__main__":
    model_path = "sam_vit_h_4b8939.pth"
    image_path = "../../test_images/P_DET_IMAGES/1test.jpg"
    output_mask_path = "../../test_images/P_DET_IMAGES/1test_mask.jpg"
    output_overlay_path = "../../test_images/P_DET_IMAGES/1test.jpg_overlay.jpg"
    input_point = (100, 200)

    segmenter = PipetteSegmenter(model_path)
    image = segmenter.load_image(image_path)
    mask, score = segmenter.get_mask(input_point)

    segmenter.save_mask(mask, output_mask_path)
    overlay_image = segmenter.overlay_mask(image, mask)
    cv2.imwrite(output_overlay_path, overlay_image)

    print(f"Mask saved to: {output_mask_path}")
    print(f"Overlay image saved to: {output_overlay_path}")
    print(f"Mask Score: {score}")
