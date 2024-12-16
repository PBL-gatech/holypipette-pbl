import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import yaml

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class cellSegmentor:
    def __init__(self, sam_checkpoint, model_cfg, device=None):
        # Enable MPS fallback for unsupported operations
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Warnings for non-CUDA devices
        if self.device.type == "mps":
            print(
                "\nWarning: SAM2 is optimized for CUDA devices. "
                "Using MPS may result in degraded performance or numerically different results."
            )
        elif self.device.type == "cpu":
            print("\nWarning: Running on CPU may significantly impact performance.")

        self.sam_checkpoint = sam_checkpoint
        self.model_cfg = model_cfg
        self.sam = None
        self.predictor = None
        self.image = None

        # Load the SAM2 model
        self._load_model()

    def _load_model(self):
        # Configure for CUDA optimizations
        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        # Load SAM2 model with error handling
        try:
            # Explicitly load the config
            with open(self.model_cfg, 'r') as f:
                config = yaml.safe_load(f)

            # Load SAM2 model
            self.sam = build_sam2(self.model_cfg, self.sam_checkpoint, device=self.device)
        except Exception as e:
            print(f"Error loading SAM2 model: {e}")
            raise

        # Initialize predictor
        self.predictor = SAM2ImagePredictor(self.sam)

    def load_image(self, image_path):
        # Load and convert the image to RGB
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Error: Could not load image from {image_path}. Please check the file path and ensure the file exists.")
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def set_image(self):
        # Set image to SAM2 model for embedding
        if self.image is not None:
            self.predictor.set_image(self.image)
        else:
            raise ValueError("No image loaded. Please load an image first.")

    def predict_mask(self, input_point, input_label, multimask_output=True):
        # Predict masks using the input point and label
        if self.image is None:
            raise ValueError("Image has not been set. Please call set_image() before predicting.")

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=multimask_output,
        )
        return masks, scores

    def predict_mask_box(self, input_box, multimask_output=True):
        # Predict masks using the input box
        if self.image is None:
            raise ValueError("Image has not been set. Please call set_image() before predicting.")

        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=multimask_output,
        )
        return masks, scores

    def single_prediction(self, input_point, input_label, multimask_output=False):
        # Get a single prediction using the input point and label
        if self.image is None:
            raise ValueError("Image has not been set. Please call set_image() before predicting.")

        masks, scores = self.predict_mask(input_point, input_label, multimask_output)
        return masks[0]  # Return the first mask

    def show_image(self):
        # Show the loaded image
        if self.image is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(self.image)
            plt.axis('on')
            plt.show()
        else:
            raise ValueError("No image loaded. Please load an image first.")

    def show_mask(self, mask, ax, random_color=False, borders=True):
        # Utility method to visualize mask on image
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        if borders:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)

        ax.imshow(mask_image)

    def show_points(self, coords, labels, ax, marker_size=375):
        # Utility method to visualize input points
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)

    def show_box(self, box, ax):
        # Utility method to visualize bounding box
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    def visualize_prediction(self, input_point=None, input_label=None, input_box=None, multimask_output=False, borders=True):
        # Visualize masks with the given input point or box
        if input_point is None and input_box is None:
            raise ValueError("Please provide either input_point or input_box.")
        if input_point is not None and input_box is not None:
            raise ValueError("Please provide only one of input_point or input_box.")

        if input_point is not None:
            masks, scores = self.predict_mask(input_point, input_label, multimask_output)
            for i, (mask, score) in enumerate(zip(masks, scores)):
                plt.figure(figsize=(10, 10))
                plt.imshow(self.image)
                self.show_mask(mask, plt.gca(), borders=borders)
                self.show_points(input_point, input_label, plt.gca())
                plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
                plt.axis('off')
                plt.show()
        else:
            masks, scores = self.predict_mask_box(input_box, multimask_output)
            for i, (mask, score) in enumerate(zip(masks, scores)):
                plt.figure(figsize=(10, 10))
                plt.imshow(self.image)
                self.show_mask(mask, plt.gca(), borders=borders)
                self.show_box(input_box, plt.gca())
                plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
                plt.axis('off')
                plt.show()

if __name__ == "__main__":
    # Example usage
    sys.path.append(r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\cellModel\sam2")
    
    # Adjust paths to ensure they are correct
    sam_checkpoint = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\cellModel\sam2\checkpoints\sam2.1_hiera_tiny.pt" 
    model_cfg = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\cellModel\sam2\sam2\configs\sam2.1\sam2.1_hiera_t.yaml"
    
    # Create object segmentation instance
    obj_seg = cellSegmentor(sam_checkpoint, model_cfg)

    # Load an example image
    # image_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\cellModel\sam2\notebooks\images\truck.jpg"
    # image_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_12_10-15_17\camera_frames\178605_1733864203.246073.webp" # pipette
    image_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\cellModel\example pictures\before.tiff" # cell
    obj_seg.load_image(image_path)
    obj_seg.set_image()

    # Predict using a single point
    input_point = np.array([[500, 375]]) # cell or truck
    # input_point = np.array([[400, 900]]) # pipette
    input_point = np.array([[600, 550]]) # cell

    input_label = np.array([1])
    obj_seg.visualize_prediction(input_point=input_point, input_label=input_label)

    # Predict using a bounding box
    input_box = np.array([425, 600, 700, 875])
    obj_seg.visualize_prediction(input_box=input_box)

    # test single prediction speed
    import time
    start_time = time.time()
    obj_seg.single_prediction(input_point, input_label)
    elapsed_time = time.time() - start_time
    elapsed_time_ms = elapsed_time * 1000
    # print(f'Single prediction time: {elapsed_time:.4f} seconds')
    print(f'Single prediction time: {elapsed_time_ms:.4f} ms')