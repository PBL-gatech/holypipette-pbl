from __future__ import annotations
import logging, os, cv2, numpy as np
from typing import Optional, Tuple
from holypipette.deepLearning.cellSegmentor import CellSegmentor2

__all__ = ["CellTrackHelper"]


class CellTrackHelper:
    """Locate a single cell in *image* and return its centroid (x, y)."""

    # ------------------------------------------------------------------ #
    def __init__(self, stage, camera) -> None:
        self.stage  = stage
        self.camera = camera
        self.width  = int(getattr(camera, "width",  None) or 0)
        self.height = int(getattr(camera, "height", None) or 0)
        self.segmentor = CellSegmentor2()

    def find_centroid(
        self, ref_image:np.ndarray, image: np.ndarray,input_point: Tuple[int, int] | Tuple[float, float],) -> Optional[np.ndarray]:
        """Return `[cx, cy]` in *pixels* if segmentation succeeds, else `None`."""
        seed = self._validate_seed(input_point)
        if seed is None:
            return None                               # bad seed - already logged

        prepped = self._prepare_image(image)
        if prepped is None:
            return None                               # unsupported image shape

        mask = self._run_segmentation(prepped, seed)
        if mask is None:
            return None                               # segmentation failed

        return self._compute_centroid(mask)

    def _validate_seed(
        self, point: Tuple[int, int] | Tuple[float, float]
    ) -> Optional[np.ndarray]:
        """Ensure the seed is numeric and in `float32` shape (1, 2)."""
        try:
            x, y = float(point[0]), float(point[1])
            # logging.debug("CellTrackHelper: seed (%.2f, %.2f) px", x, y)
            return np.array([[x, y]], dtype=np.float32)
        except Exception as exc:
            logging.error("CellTrackHelper: bad seed (%s) – %s", point, exc)
            return None

    # 2. ---------------------------------------------------------------- #
    def _prepare_image(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert to 3-channel `float32` in [0, 1] without altering colour order.
        *Important*: `CellSegmentor2` expects **BGR** (OpenCV default).
        """
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):    # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.ndim != 3 or img.shape[2] != 3:
            logging.error("CellTrackHelper: unsupported image shape %s", img.shape)
            return None

        if img.dtype != np.float32:
            img = (img.astype(np.float32) / 255.0)

        return img

    # 3. ---------------------------------------------------------------- #
    def _run_segmentation(
        self, rgb: np.ndarray, seed: np.ndarray
    ) -> Optional[np.ndarray]:
        """Call SAM-2 and handle mask bookkeeping/debug‐dump."""
        mask = self.segmentor.segment(rgb, seed, np.array([1], dtype=np.int32))
        if mask is None:
            logging.warning("CellTrackHelper: segmentation failed – no mask")
            return None

        # SAM-2 may return (1, H, W)
        if mask.ndim == 3:
            mask = mask[0]

        # # optional debug dump
        # try:
        #     os.makedirs("testmasks", exist_ok=True)
        #     dbg = (mask * 255).astype(np.uint8).copy()

        #     cv2.circle(dbg, (int(seed[0, 0]), int(seed[0, 1])), 5, 128, -1)
        #     cx, cy = int(seed[0, 0]), int(seed[0, 1])
        #     cv2.circle(dbg, (cx, cy), 5, 128, -1)
        #     cv2.imwrite("testmasks/mask.png", dbg)
        # except Exception as exc:
        #     logging.debug("CellTrackHelper: mask dump failed – %s", exc)

        return mask

    # 4. ---------------------------------------------------------------- #
    def _compute_centroid(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Return centroid `[cx, cy]` (float32) or `None` if mask is empty."""
        M = cv2.moments(mask.astype(np.uint8))
        if M["m00"] == 0:
            logging.warning("CellTrackHelper: zero-area mask – centroid undefined")
            return None
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        logging.debug("CellTrackHelper: centroid (%.2f, %.2f) px", cx, cy)
        return np.array([cx, cy], dtype=np.float32)
