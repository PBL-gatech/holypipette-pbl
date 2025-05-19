import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import yaml
import time
import logging

# current_dir = os.path.dirname(os.path.abspath(__file__))
# repo_root = os.path.abspath(os.path.join(current_dir, '..'))
# logging.debug(f"repo root: {repo_root}")
# sys.path.append(os.path.join(repo_root, "holypipette", "deepLearning", "cellModel", "MobileSAM"))
# from mobile_sam import sam_model_registry, SamPredictor

#sam2 imports
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, '..'))
# logging.info(f"repo root: {repo_root}")
checkpoint = os.path.join(repo_root,  "deepLearning", "cellModel", "sam2", "checkpoints", "sam2.1_hiera_tiny.pt")
model_cfg = os.path.join(repo_root,  "deepLearning", "cellModel", "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_t.yaml")
# logging.info(f"checkpoint: {checkpoint}")
# logging.info(f"model_cfg: {model_cfg}")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class CellSegmentor:  
    def __init__(self, sam_checkpoint = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\cellModel\MobileSAM\weights\mobile_sam.pt", model_type="vit_t", device="cuda"):
        self.device = device
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.sam = None
        self.predictor = None
        self.image = None
        
        # Initialize the SAM model
        self._load_model()

    def _load_model(self):
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

    def load_image(self, image_path=None, image=None):
        # Load and convert the image to RGB
        if image_path is not None:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Error: Could not load image from {image_path}. Please check the file path and ensure the file exists.")
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image is not None:
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Either image_path or image must be provided.")
    def set_image(self):
        # Set image to SAM model for embedding
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

    def segment(self, image = None,input_point=None, input_label=None, input_box=None, multimask_output=False):

        if image is not None:
            self.load_image(image = image)
            self.set_image()
        else: 
            if self.image is None:
                raise ValueError("No image loaded. Please load an image first.")
        # Segment the image with the given input point or box
        if input_point is None and input_box is None:
            raise ValueError("Please provide either input_point or input_box.")
        if input_point is not None and input_box is not None:
            raise ValueError("Please provide only one of input_point or input_box.")

        if input_point is not None: 
            # use single predict method
            mask = self.single_prediction(input_point, input_label, multimask_output)
            if mask is None:
                logging.error("No mask found")
                return None
            else:
                return mask
        else:
            # use box predict method
            masks, scores = self.predict_mask_box(input_box, multimask_output)
            if masks is None:
                logging.error("No masks found")
                return None
            else:
                return masks[0]
class CellSegmentor2:
    def __init__(self, sam_checkpoint=checkpoint , model_cfg = model_cfg , device=None):
        logging.info("Initializing SAM2 CellSegmentor...")
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

    def load_image(self, image_path=None, image=None):
        if image_path is not None:
            image = cv2.imread(image_path)
            ...
            if len(image.shape) == 2:
                # Expand grayscale -> BGR
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image is not None:
            # Check if grayscale
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Either image_path or image must be provided.")


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

    def segment(self, image = None,input_point=None, input_label=None, input_box=None, multimask_output=False):

        if image is not None:

            self.load_image(image = image)
            self.set_image()

        else: 
            if self.image is None:
                raise ValueError("No image loaded. Please load an image first.")
        # Segment the image with the given input point or box
        if input_point is None and input_box is None:
            raise ValueError("Please provide either input_point or input_box.")
        if input_point is not None and input_box is not None:
            raise ValueError("Please provide only one of input_point or input_box.")

        if input_point is not None: 
            # use single predict method
            mask = self.single_prediction(input_point, input_label, multimask_output)
            return mask
        else:
            # use box predict method
            masks, scores = self.predict_mask_box(input_box, multimask_output)
            return masks[0]

import os
import numpy as np
import onnxruntime as ort
import cv2
import matplotlib.pyplot as plt

class CellSegmentor3:
    def __init__(
        self,
        encoder_path: str = None,
        decoder_path: str = None,
        device: str = "cuda",
    ):
        """
        Initialize the ONNX-based SAM2 segmentor.
        If no paths are provided, defaults to the cellModel/sam2/onnx folder.
        Args:
            encoder_path: path to sam2.1_tiny_preprocess.onnx
            decoder_path: path to sam2.1_tiny.onnx
            device: 'cpu' or 'cuda'
        """
        # Determine default paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(current_dir, '..'))
        default_dir = os.path.join(
            repo_root,
            'deepLearning',
            'cellModel'
        )

        # Use provided paths or fall back to defaults
        if encoder_path is None:
            encoder_path = os.path.join(default_dir, 'sam2.1_tiny_preprocess.onnx')
        if decoder_path is None:
            decoder_path = os.path.join(default_dir, 'sam2.1_tiny.onnx')

        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.device = device

        # Configure ONNX Runtime providers
        providers = (
            ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if device == "cuda"
            else ['CPUExecutionProvider']
        )

        # Load ONNX sessions
        self.enc_session = ort.InferenceSession(self.encoder_path, providers=providers)
        self.dec_session = ort.InferenceSession(self.decoder_path, providers=providers)

        # Cache input names for robustness
        self.enc_input_name = self.enc_session.get_inputs()[0].name
        self.dec_input_names = {
            inp.name for inp in self.dec_session.get_inputs()
        }

        self.image = None
        self.embedding = None


    def load_image(self, image_path=None, image=None):
        """
        Load image from disk or use provided numpy array.
        Converts BGR to RGB and resizes/pads to 1024x1024.
        """
        if image_path:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Unable to load image: {image_path}")
        elif image is not None:
            img = image.copy()
        else:
            raise ValueError("Provide image_path or image array.")

        # Convert to RGB
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize and pad to 1024x1024
        h, w = img_rgb.shape[:2]
        scale = 1024 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img_rgb, (new_w, new_h))
        canvas = np.zeros((1024, 1024, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        self.image = canvas.astype(np.float32) / 255.0  # normalize to [0,1]

    def set_image(self):
        """
        Runs encoder to compute embeddings.
        """
        if self.image is None:
            raise ValueError("No image loaded")
        inp = self.image.transpose(2, 0, 1)[None].astype(np.float32)
        # Older models may expect a different input key (e.g. "input")
        self.embedding = self.enc_session.run(
            None, {self.enc_input_name: inp}
        )[0]

    def predict_mask(
        self,
        input_point: np.ndarray = None,
        input_label: np.ndarray = None,
        input_box: np.ndarray = None,
        multimask_output: bool = True,
    ):
        """
        Run decoder with the given prompts.
        Accepts either point prompts or a single box prompt.
        """
        if self.embedding is None:
            raise ValueError("Call set_image() first")

        # Prepare input dictionary based on available input names
        if "embeddings" in self.dec_input_names:
            inputs = {"embeddings": self.embedding}
        else:
            # fall back to first input name
            first_name = next(iter(self.dec_input_names))
            inputs = {first_name: self.embedding}

        if input_point is not None and input_label is not None:
            if "point_coords" in self.dec_input_names:
                inputs["point_coords"] = input_point.astype(np.float32)[None]
            if "point_labels" in self.dec_input_names:
                inputs["point_labels"] = input_label.astype(np.float32)[None]

        if input_box is not None:
            if "box" in self.dec_input_names:
                inputs["box"] = input_box.astype(np.float32)[None]

        outputs = self.dec_session.run(None, inputs)
        masks, scores = outputs[0], outputs[1]
        return masks, scores

    def predict_mask_box(self, input_box: np.ndarray, multimask_output: bool = True):
        """
        Convenience wrapper around predict_mask for box prompts.
        """
        return self.predict_mask(input_box=input_box, multimask_output=multimask_output)

    def single_prediction(self, input_point, input_label):
        masks, scores = self.predict_mask(input_point=input_point, input_label=input_label, multimask_output=False)
        return masks[0]

    def show_image(self):
        if self.image is None:
            raise ValueError("No image to show")
        plt.figure(figsize=(8,8)); plt.imshow(self.image); plt.axis('off'); plt.show()

    def show_mask(self, mask, ax=None, alpha=0.6, borders=True):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,8))
        color = np.array([1, 0, 0, alpha])
        h, w = mask.shape[-2:]
        mask_img = mask[0] if mask.ndim == 3 else mask
        overlay = mask_img.reshape(h, w, 1) * color.reshape(1,1,4)
        ax.imshow(self.image)
        ax.imshow(overlay)
        if borders:
            cnts, _ = cv2.findContours(mask_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in cnts:
                ax.plot(c[:,0,0], c[:,0,1], color='white', linewidth=2)
        ax.axis('off')

    def show_points(self, coords, labels, ax, marker_size=100):
        pos = coords[labels==1]; neg = coords[labels==0]
        ax.scatter(pos[:,0], pos[:,1], c='green', s=marker_size, edgecolor='white')
        ax.scatter(neg[:,0], neg[:,1], c='red', s=marker_size, edgecolor='white')

    def show_box(self, box, ax):
        x0,y0,x1,y1 = box
        ax.add_patch(plt.Rectangle((x0,y0), x1-x0, y1-y0, fill=False, edgecolor='green', lw=2))

    def visualize_prediction(
        self,
        input_point=None,
        input_label=None,
        input_box=None,
        multimask_output=False,
        borders=True,
    ):
        masks, scores = self.predict_mask(input_point=input_point, input_label=input_label,
                                         input_box=input_box, multimask_output=multimask_output)
        for i, (mask, score) in enumerate(zip(masks, scores)):
            fig, ax = plt.subplots(figsize=(8,8))
            self.show_mask(mask, ax=ax, borders=borders)
            if input_point is not None:
                self.show_points(input_point, input_label, ax)
            if input_box is not None:
                self.show_box(input_box, ax)
            ax.set_title(f"Mask {i+1}, Score: {score:.3f}")
            plt.show()

    def segment(
        self,
        image=None,
        input_point=None,
        input_label=None,
        input_box=None,
        multimask_output=False,
    ):
        if image is not None:
            self.load_image(image=image)
            self.set_image()
        if input_point is not None:
            return self.single_prediction(input_point, input_label)
        if input_box is not None:
            masks, _ = self.predict_mask(input_box=input_box, multimask_output=multimask_output)
            return masks[0]
        raise ValueError("Provide input_point or input_box.")





import sys, time
import numpy as np
import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, 
    QHBoxLayout, QPushButton, QFileDialog  # <-- NEW IMPORTS
)
from PyQt5.QtGui import QPixmap, QImage

# ---------------------------------------------------------
# Suppose you have already defined CellSegmentor2 here somewhere
# class CellSegmentor2:
#     def segment(...):
#         ...
# ---------------------------------------------------------

class CameraWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, fps=30, parent=None):
        super().__init__(parent)
        self.fps = fps
        self.keep_running = True
        self.image_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\cellModel\example pictures\before.tiff"
        # self.image_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\devices\camera\FakeMicroscopeImgs\cellsegtest.png"
        bgr = cv2.imread(self.image_path)
        if bgr is None:
            raise FileNotFoundError(f"Could not load {self.image_path}")
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        self.points = []
        self.labels = []
        self.segmentor = CellSegmentor3()

    def run(self):
        while self.keep_running:
            display_img = self.rgb.copy()
            if self.points:
                mask = self.segmentor.segment(
                    image=display_img,
                    input_point=np.array(self.points),
                    input_label=np.array(self.labels),
                    multimask_output=False
                )
                if mask is not None:
                    mask_bool = mask.astype(bool)
                    display_img[mask_bool] = [255, 0, 0]
                    for (x, y) in self.points:
                        cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)

            self.frame_ready.emit(display_img)
            time.sleep(1 / self.fps)

    def stop(self):
        self.keep_running = False
        self.quit()
        self.wait()

    def add_point(self, x, y):
        self.points.append([x, y])
        self.labels.append(1)

    def clear_points(self):
        self.points.clear()
        self.labels.clear()

class SamCAM(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SamCAM")
        self.image_label = QLabel(alignment=Qt.AlignCenter)
        
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        # ----------------------------------------------------
        # 1) Create a bottom-right button layout and button
        button_layout = QHBoxLayout()
        button_layout.addStretch()  # pushes the button to the right
        self.save_button = QPushButton("Save segmentation?")
        button_layout.addWidget(self.save_button)
        layout.addLayout(button_layout)
        # 2) Connect button to a method that saves the image
        self.save_button.clicked.connect(self.save_segmentation)
        # ----------------------------------------------------
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Create worker thread for “camera”
        self.worker = CameraWorker(fps=5)
        self.worker.frame_ready.connect(self.update_image)
        self.worker.start()

        # Will store the latest displayed frame here
        self.latest_frame = None

    def update_image(self, rgb):
        """Receive new frames and display them."""
        # Store the latest image so we can save it
        self.latest_frame = rgb.copy()
        
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            label_pos = self.image_label.mapFromParent(event.pos())
            pm = self.image_label.pixmap()
            if pm:
                scale_w = pm.width() / self.image_label.width()
                scale_h = pm.height() / self.image_label.height()
                x_img = int(label_pos.x() * scale_w)
                y_img = int(label_pos.y() * scale_h)
                self.worker.add_point(x_img, y_img)
        elif event.button() == Qt.RightButton:
            self.worker.clear_points()

    def closeEvent(self, event):
        self.worker.stop()
        super().closeEvent(event)

    # ----------------------------------------------------
    def save_segmentation(self):
        """Re-run segmentation on original image and save only the cut-out region (RGBA)."""
        if not self.worker.points:
            print("No points set, nothing to segment.")
            return

        # Re-run segmentation on the original
        rgb_copy = self.worker.rgb.copy()
        mask = self.worker.segmentor.segment(
            image=rgb_copy,
            input_point=np.array(self.worker.points),
            input_label=np.array(self.worker.labels),
            multimask_output=False
        )
        if mask is None:
            print("Segmentation returned None, nothing to save.")
            return

        # Convert mask to boolean
        mask_bool = (mask > 0)

        # Build an RGBA image (4 channels)
        h, w, _ = rgb_copy.shape
        segmented_rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Copy over the original RGB where the mask is True
        segmented_rgba[mask_bool, 0:3] = rgb_copy[mask_bool]

        # Set alpha channel to 255 where the mask is True
        segmented_rgba[mask_bool, 3] = 255

        # Let user pick the save path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Segmented PNG", "", 
            "PNG Files (*.png);;All Files (*)"
        )
        if not file_path:
            return  # user canceled

        # Finally save as PNG (4-channels) - the alpha channel is preserved
        cv2.imwrite(file_path, segmented_rgba)
        print(f"Saved RGBA cut-out to: {file_path}")

    # ----------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SamCAM()
    w.show()
    sys.exit(app.exec_())

    # #### test segmentor 
    # # obj_seg = CellSegmentor()
    # obj_seg = CellSegmentor2()

    # # Load an example image
    # # # image_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\cellModel\sam2\notebooks\images\truck.jpg"
    # # # image_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_12_10-15_17\camera_frames\178605_1733864203.246073.webp" # pipette

    # # image_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\cellModel\example pictures\before.tiff" # cell
    # image_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\devices\camera\FakeMicroscopeImgs\background.tif"

     
    # # load and set image
    # # obj_seg.load_image(r"C:\Users\sa-forest\Documents\GitHub\patchability\code\segment-anything\notebooks\images\truck.jpg")
    # obj_seg.load_image(image_path)
    # obj_seg.set_image()
    # # obj_seg.show_image()

    # # # Predict using a single point
    # # input_point = np.array([[500, 375]]) # cell or truck
    # # # input_point = np.array([[400, 900]]) # pipette
  
    # input_point = np.array([600, 400])

    # input_label = np.array([1])
    # obj_seg.visualize_prediction(input_point=input_point, input_label=input_label)

    # # # # # Predict using a bounding box
    # # input_box = np.array([425, 600, 700, 875])
    # # obj_seg.visualize_prediction(input_box=input_box)

    # # perform a  single prediction
    # #obj_seg.single_prediction(input_point, input_label)

    # # # test segment speed
    # # start_time = time.time()

    # # image = cv2.imread(image_path)
    # # total_time = 0
    # # num  = 1000
    # # time_array = np.zeros(num)
    # # for i in range(num):
    # #     start_time = time.time()
    # #     mask = obj_seg.segment(image=image, input_point=input_point, input_label=input_label)
    # #     elapsed_time = time.time() - start_time
    # #     elapsed_time_ms = elapsed_time * 1000
    # #     # total_time += elapsed_time
    # #     time_array[i] = elapsed_time_ms
    
    # # time_array = time_array[3:]
    # # average_time = np.mean(time_array)
    # # print(f'Average segment time: {average_time:.4f}  milliseconds')
    # # # remove first few points

    # # # plot time_array
    # # plt.plot(time_array)
    # # plt.xlabel('Iteration')
    # # plt.ylabel('Time (ms)')
    # # plt.title('Segment Time')
    # # plt.show()
