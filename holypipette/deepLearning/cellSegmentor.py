import os
import sys
import cv2
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import logging
from abc import ABC, abstractmethod

# ================= BASE SEGMENTOR =================
class BaseSegmentor(ABC):
    """Common functionality for cell image segmentation across different SAM back‑ends."""

    def __init__(self, device: str | None = None):
        self.device = self._resolve_device(device)
        self.sam = None
        self.predictor = None
        self.image = None
        self._load_model()

    @staticmethod
    def _resolve_device(explicit: str | None):
        if explicit is not None:
            return torch.device(explicit)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @abstractmethod
    def _load_model(self):
        ...

    def load_image(self, image_path=None, image=None):
        if image_path is not None:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Error: Could not load image from {image_path}. Please check the file path and ensure the file exists.")
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image is not None:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Either image_path or image must be provided.")

    def set_image(self):
        if self.image is not None:
            self.predictor.set_image(self.image)
        else:
            raise ValueError("No image loaded. Please load an image first.")

    def predict_mask(self, input_point, input_label, multimask_output=True):
        if self.image is None:
            raise ValueError("Image has not been set. Please call set_image() before predicting.")

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=multimask_output,
        )
        return masks, scores

    def predict_mask_box(self, input_box, multimask_output=True):
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
        if self.image is None:
            raise ValueError("Image has not been set. Please call set_image() before predicting.")

        masks, scores = self.predict_mask(input_point, input_label, multimask_output)
        return masks[0]  # Return the first mask

    def show_image(self):
        if self.image is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(self.image)
            plt.axis('on')
            plt.show()
        else:
            raise ValueError("No image loaded. Please load an image first.")

    @staticmethod
    def show_mask(mask, ax, random_color=False, borders=True):
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

    @staticmethod
    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)

    @staticmethod
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    def visualize_prediction(self, input_point=None, input_label=None, input_box=None, multimask_output=False, borders=True):
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

    def segment(self, image=None, input_point=None, input_label=None, input_box=None, multimask_output=False):
        if image is not None:
            self.load_image(image=image)
            self.set_image()
        else:
            if self.image is None:
                raise ValueError("No image loaded. Please load an image first.")
        if input_point is None and input_box is None:
            raise ValueError("Please provide either input_point or input_box.")
        if input_point is not None and input_box is not None:
            raise ValueError("Please provide only one of input_point or input_box.")

        if input_point is not None:
            mask = self.single_prediction(input_point, input_label, multimask_output)
            if mask is None:
                logging.error("No mask found")
                return None
            else:
                return mask
        else:
            masks, scores = self.predict_mask_box(input_box, multimask_output)
            if masks is None:
                logging.error("No masks found")
                return None
            else:
                return masks[0]


# ================= MOBILE SAM SEGMENTOR (CellSegmentor1) =================
class CellSegmentor1(BaseSegmentor):
    """Segmentor powered by MobileSAM (Tiny‑ViT)."""

    def __init__(self, sam_checkpoint=None, model_type="vit_t", device=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(current_dir, ".."))
        sys.path.append(os.path.join(repo_root, "holypipette", "deepLearning", "cellModel", "MobileSAM"))
        default_ckpt = os.path.join(repo_root, "deepLearning", "cellModel", "MobileSAM", "weights", "mobile_sam.pt")
        self.sam_checkpoint = sam_checkpoint or default_ckpt
        self.model_type = model_type
        super().__init__(device)

    def _load_model(self):
        from mobile_sam import sam_model_registry, SamPredictor
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)


# ================= SAM2 SEGMENTOR (CellSegmentor2) =================
class CellSegmentor2(BaseSegmentor):
    """Segmentor powered by SAM2 (v2.1 Hiera‑Tiny)."""

    def __init__(self, sam_checkpoint=None, model_cfg=None, device=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(current_dir, ".."))
        default_ckpt = os.path.join(repo_root, "deepLearning", "cellModel", "sam2", "checkpoints", "sam2.1_hiera_tiny.pt")
        default_cfg = os.path.join(repo_root, "deepLearning", "cellModel", "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_t.yaml")
        self.sam_checkpoint = sam_checkpoint or default_ckpt
        self.model_cfg = model_cfg or default_cfg
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        super().__init__(device)

    def _enable_cuda_tricks(self):
        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def _load_model(self):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self._enable_cuda_tricks()
        try:
            with open(self.model_cfg, "r") as f:
                yaml.safe_load(f)
            self.sam = build_sam2(self.model_cfg, self.sam_checkpoint, device=self.device)
        except Exception as exc:
            logging.error(f"Error loading SAM2 model: {exc}")
            raise
        self.predictor = SAM2ImagePredictor(self.sam)

import os
import numpy as np
import onnxruntime as ort
import cv2
import matplotlib.pyplot as plt
import torch
from typing import Optional

class CellSegmentor3(BaseSegmentor):
    """ONNX-based SAM2 segmentor, refactored to match BaseSegmentor interface and sam-cpp-macos reference."""
    def __init__(self, encoder_path: Optional[str] = None, decoder_path: Optional[str] = None, device: Optional[str] = None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(current_dir, ".."))
        default_dir = os.path.join(repo_root, "deepLearning", "cellModel")

        if encoder_path is None:
            encoder_path = os.path.join(default_dir, "sam2.1_tiny_preprocess.onnx")
        if decoder_path is None:
            decoder_path = os.path.join(default_dir, "sam2.1_tiny.onnx")

        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        super().__init__(device=device)

    def _load_model(self):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device.type == "cuda"
            else ["CPUExecutionProvider"]
        )
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # Suppress most warnings
        self.enc_session = ort.InferenceSession(self.encoder_path, providers=providers, sess_options=sess_options)
        self.dec_session = ort.InferenceSession(self.decoder_path, providers=providers, sess_options=sess_options)
        self.enc_input_name = self.enc_session.get_inputs()[0].name
        self.dec_input_names = [inp.name for inp in self.dec_session.get_inputs()]
        self.image_embeddings = None
        self.high_res_features1 = None
        self.high_res_features2 = None
        self._orig_shape = None
        self._pad_shape = None
        self.input_size = (1024, 1024)  # default model input
        self._pad_offset = (0, 0)

    def load_image(self, image_path=None, image=None):
        if image_path is not None:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Could not load image from {image_path}.")
        elif image is not None:
            img = image.copy()
        else:
            raise ValueError("Either image_path or image must be provided.")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]
        self._orig_shape = (orig_h, orig_w)
        scale = min(1024 / orig_h, 1024 / orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        resized = cv2.resize(img_rgb, (new_w, new_h))
        canvas = np.zeros((1024, 1024, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        norm_img = (canvas.astype(np.float32) / 255.0 - mean) / std
        self.image = norm_img
        self.input_size = (1024, 1024)
        self._pad_shape = (new_h, new_w)
        self._pad_offset = (0, 0)
        # Debug info
        print(f"[DEBUG] Original image shape: {orig_h}x{orig_w}")
        print(f"[DEBUG] Resized image shape: {new_h}x{new_w}")
        print(f"[DEBUG] Padding offset: {self._pad_offset}")

    def set_image(self):
        if self.image is None:
            raise ValueError("No image loaded. Please load an image first.")
        inp = self.image.transpose(2, 0, 1)[None].astype(np.float32)
        result = self.enc_session.run(None, {self.enc_input_name: inp})
        self.image_embeddings = result[0]
        self.high_res_features1 = result[1]
        self.high_res_features2 = result[2]
        # Debug info
        print(f"[DEBUG] Encoder output shapes: {[x.shape for x in result]}")

    def _scale_points(self, points):
        points = np.asarray(points)
        orig_h, orig_w = self._orig_shape
        pad_h, pad_w = self._pad_shape
        scale = min(1024 / orig_h, 1024 / orig_w)
        scaled = points * scale
        scaled = scaled + np.array([self._pad_offset[1], self._pad_offset[0]])
        # Debug info
        print(f"[DEBUG] Scaling points {points} from orig shape {self._orig_shape} -> resized shape {self._pad_shape}")
        print(f"[DEBUG] Resulting scaled points: {scaled}")
        return scaled.astype(np.float32)

    def predict_mask(self, input_point=None, input_label=None, multimask_output=True):
        if self.image_embeddings is None:
            raise ValueError("Image has not been set. Please call set_image() before predicting.")
        orig_h, orig_w = self._orig_shape
        if input_point is not None and len(input_point):
            scaled_points = self._scale_points(input_point)
            point_coords = scaled_points[None, ...]
            point_labels = np.asarray(input_label, dtype=np.float32)[None]
        else:
            point_coords = np.zeros((1, 0, 2), dtype=np.float32)
            point_labels = np.zeros((1, 0), dtype=np.float32)
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.array([0], dtype=np.float32)
        orig_im_size = np.array([orig_h, orig_w], dtype=np.int64)
        decoder_inputs = {
            "image_embeddings": self.image_embeddings,
            "high_res_features1": self.high_res_features1,
            "high_res_features2": self.high_res_features2,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_input": mask_input,
            "has_mask_input": has_mask_input,
            "orig_im_size": orig_im_size
        }
        onnx_inputs = {k: decoder_inputs[k] for k in self.dec_input_names}
        # Debug info
        print(f"[DEBUG] Decoder input shapes:")
        for k, v in onnx_inputs.items():
            print(f"    {k}: {getattr(v, 'shape', None)} dtype={getattr(v, 'dtype', None)}")
        outputs = self.dec_session.run(None, onnx_inputs)
        masks, scores = outputs[0], outputs[1]
        print(f"[DEBUG] Decoder output masks shape: {masks.shape}")
        print(f"[DEBUG] Decoder output scores shape: {scores.shape}")
        best_idx = np.argmax(scores[0])
        mask = masks[0, best_idx]
        mask_bin = (mask > 0).astype(np.uint8)
        print(f"[DEBUG] Best mask index: {best_idx} (score={scores[0, best_idx]:.4f})")
        print(f"[DEBUG] Mask output shape: {mask_bin.shape}")
        return mask_bin, scores[0, best_idx]

    def predict_mask_box(self, input_box, multimask_output=True):
        if input_box is None or len(input_box) != 4:
            raise ValueError("Box must be [x0,y0,x1,y1]")
        points = np.array([[input_box[0], input_box[1]], [input_box[2], input_box[3]]], dtype=np.float32)
        labels = np.array([2, 3], dtype=np.float32)
        return self.predict_mask(input_point=points, input_label=labels, multimask_output=multimask_output)

    def single_prediction(self, input_point, input_label, multimask_output=False):
        mask, score = self.predict_mask(input_point, input_label, multimask_output)
        return mask

    def show_image(self):
        if self.image is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow((self.image * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])).clip(0,1))
            plt.axis('on')
            plt.show()
        else:
            raise ValueError("No image loaded. Please load an image first.")

    @staticmethod
    def show_mask(mask, ax, random_color=False, borders=True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_img = mask.astype(np.uint8)
        mask_image = mask_img.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
        ax.imshow(mask_image)

    @staticmethod
    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    @staticmethod
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    def visualize_prediction(self, input_point=None, input_label=None, input_box=None, multimask_output=False, borders=True):
        if input_point is None and input_box is None:
            raise ValueError("Please provide either input_point or input_box.")
        if input_point is not None and input_box is not None:
            raise ValueError("Please provide only one of input_point or input_box.")
        if input_point is not None:
            masks, score = self.predict_mask(input_point, input_label, multimask_output)
            plt.figure(figsize=(10, 10))
            plt.imshow((self.image * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])).clip(0,1))
            self.show_mask(masks, plt.gca(), borders=borders)
            self.show_points(input_point, input_label, plt.gca())
            plt.title(f"Mask, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()
        else:
            masks, score = self.predict_mask_box(input_box, multimask_output)
            plt.figure(figsize=(10, 10))
            plt.imshow((self.image * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])).clip(0,1))
            self.show_mask(masks, plt.gca(), borders=borders)
            self.show_box(input_box, plt.gca())
            plt.title(f"Mask, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()

    def segment(self, image=None, input_point=None, input_label=None, input_box=None, multimask_output=False):
        if image is not None:
            self.load_image(image=image)
            self.set_image()
        else:
            if self.image is None:
                raise ValueError("No image loaded. Please load an image first.")
        if input_point is None and input_box is None:
            raise ValueError("Please provide either input_point or input_box.")
        if input_point is not None and input_box is not None:
            raise ValueError("Please provide only one of input_point or input_box.")
        if input_point is not None:
            mask = self.single_prediction(input_point, input_label, multimask_output)
            if mask is None:
                print("[DEBUG] No mask found")
                return None
            else:
                return mask
        else:
            mask, score = self.predict_mask_box(input_box, multimask_output)
            if mask is None:
                print("[DEBUG] No masks found")
                return None
            else:
                return mask




import sys, time
import numpy as np
import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, 
    QHBoxLayout, QPushButton, QFileDialog  # <-- NEW IMPORTS
)
from PyQt5.QtGui import QPixmap, QImage

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
