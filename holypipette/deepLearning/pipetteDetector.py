import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import importlib.util


def _import_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


class PipetteDetector(ABC):
    """Abstract base class for pipette detectors."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def detect_pipette(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        """Locate the pipette tip in the provided image."""
        raise NotImplementedError

    @staticmethod
    def _ensure_color(img: np.ndarray) -> np.ndarray:
        """Ensure the image has three channels for models that expect color input."""
        if img is None:
            return img
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    @staticmethod
    def _ensure_grayscale(img: np.ndarray) -> np.ndarray:
        """Convert a color image to grayscale if required."""
        if img is None:
            return img
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


class PipetteDetector1(PipetteDetector):
    """ONNX-based pipette detector (original implementation)."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        super().__init__()
        cur_file = Path(__file__).parent.absolute()
        default_model = cur_file / "pipetteModel" / "pipetteDetectorNetnano2.onnx"
        self.model_path = Path(model_path) if model_path is not None else default_model
        self.yolo_net = cv2.dnn.readNetFromONNX(str(self.model_path))
        layer_names = self.yolo_net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.yolo_net.getUnconnectedOutLayers()]
        self.pipette_class = 0

    def detect_pipette(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        """Return the (x, y) position of the pipette tip or None if not detected."""
        img = self._ensure_color(img)
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        self.yolo_net.setInput(blob)
        outs = self.yolo_net.forward(self.output_layers)

        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                idx = np.argmax(detection[4, :])
                detection = detection[:, idx]
                x, y, width, height, objectness = tuple(detection)
                if objectness < 0.20:
                    continue

                boxes.append([x, y])
                confidences.append(float(objectness))

        if len(boxes) == 0:
            return None

        confidences = np.array(confidences)
        best_x, best_y = boxes[confidences.argmax()]
        if np.isnan(best_x) or np.isnan(best_y):
            return None

        best_x = (best_x / 640) * img.shape[1]
        best_y = (best_y / 640) * img.shape[0]

        return int(best_x), int(best_y)


class PipetteDetector2(PipetteDetector):
    """DINO-based pipette detector using the transformer pipeline."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None) -> None:
        super().__init__()
        src_dir = Path(__file__).parent / "pipetteModel" / "holypipette_pipette_detection" / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        model_module = _import_module_from_path("holypipette_pipette_detection.model", src_dir / "model.py")
        pipeline_module = _import_module_from_path("holypipette_pipette_detection.pipeline", src_dir / "pipeline.py")

        DINOPipetteDetector = model_module.DINOPipetteDetector
        get_pipeline = pipeline_module.get_pipeline

        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DINOPipetteDetector()

        default_model = (
            Path(__file__).parent
            / "pipetteModel"
            / "holypipette_pipette_detection"
            / "models"
            / "DINOPipetteDetector.pt"
        )
        self.model_path = Path(model_path) if model_path is not None else default_model

        state_dict = torch.load(self.model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

        self.pipeline = get_pipeline(self.model, self.device)
        self._last_z: Optional[float] = None

    def detect_pipette(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        """Return the (x, y) position of the pipette tip or None if not detected."""
        if img is None:
            return None

        gray = self._ensure_grayscale(img)
        try:
            xy_pred, z_coord_pred = self.pipeline.get_model_prediction(gray)
        except Exception:
            return None

        xy_tensor = xy_pred[0].detach().cpu().numpy()
        x_offset, y_offset = float(xy_tensor[0]), float(xy_tensor[1])
        height, width = gray.shape

        x_pred = int(round(x_offset + width // 2))
        y_pred = int(round(y_offset + height // 2))

        z_tensor = z_coord_pred.detach().cpu().numpy()
        self._last_z = float(z_tensor[0]) if z_tensor.size else None

        if not (0 <= x_pred < width and 0 <= y_pred < height):
            return None

        return x_pred, y_pred

    @property
    def last_depth_prediction(self) -> Optional[float]:
        """Return the most recent z-coordinate prediction, if available."""
        return self._last_z


if __name__ == '__main__':
    detector = PipetteDetector1()
    path = r"C:\Users\sa-forest\GaTech Dropbox\Benjamin Magondu\YOLOretrainingdata\Pipette CNN Training Data\20191016\3654098923.png"
    img = cv2.imread(path)

    start = time.time()
    result = detector.detect_pipette(img)

    if result is not None:
        x, y = result
        print(f'framerate: {1 / (time.time() - start)}')
        cv2.circle(img, (x, y), 3, (0, 255, 0))
        cv2.imshow("pipette detection test", img)
        cv2.waitKey(0)
    else:
        print("No pipette detected")
