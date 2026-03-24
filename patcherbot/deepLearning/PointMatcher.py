import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except ImportError:  # matplotlib is optional
    plt = None

# Ensure the vendored LightGlue package is importable even when not installed globally.
_LIGHTGLUE_PARENT = Path(__file__).parent / "cellModel" / "LightGlue"
if str(_LIGHTGLUE_PARENT) not in sys.path:
    sys.path.insert(0, str(_LIGHTGLUE_PARENT))

try:
    from lightglue import ALIKED, DISK, DoGHardNet, LightGlue, SIFT, SuperPoint
    from lightglue.utils import load_image, match_pair
    _LIGHTGLUE_AVAILABLE = True
    _LIGHTGLUE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    _LIGHTGLUE_AVAILABLE = False
    _LIGHTGLUE_IMPORT_ERROR = exc
    ALIKED = DISK = DoGHardNet = LightGlue = SIFT = SuperPoint = None
    load_image = None
    match_pair = None

_EXTRACTOR_MAP = (
    {
        "superpoint": SuperPoint,
        "disk": DISK,
        "aliked": ALIKED,
        "sift": SIFT,
        "doghardnet": DoGHardNet,
    }
    if _LIGHTGLUE_AVAILABLE
    else {}
)

PathLike = Union[str, Path]


class PointMatcher:
    """
    High-level interface to measure LightGlue matching latency.
    """

    def __init__(
        self,
        *,
        features: str = "superpoint",
        device: Optional[str] = None,
        extractor_conf: Optional[Dict] = None,
        matcher_conf: Optional[Dict] = None,
    ) -> None:
        """
        Initialize the PointMatcher with specified keypoint features and device.

        Args:
            features (str): Feature extractor to use ('superpoint', 'disk', 'aliked',
                            'sift', 'doghardnet').
            device (Optional[str]): Torch device to run models ('cuda' or 'cpu').
            extractor_conf (Optional[Dict]): Keyword arguments for the feature extractor.
            matcher_conf (Optional[Dict]): Keyword arguments for the LightGlue matcher.

        Raises:
            NotImplementedError: If LightGlue is not installed or cannot be imported.
            ValueError: If an unsupported feature extractor is specified.
        """
        if not _LIGHTGLUE_AVAILABLE:
            detail = f" (import error: {_LIGHTGLUE_IMPORT_ERROR})" if _LIGHTGLUE_IMPORT_ERROR else ""
            raise NotImplementedError(f"LightGlue is not available{detail}.")
        self.features = features.lower()
        if self.features not in _EXTRACTOR_MAP:
            supported = ", ".join(sorted(_EXTRACTOR_MAP))
            raise ValueError(f"Unsupported features '{features}'. Choose from: {supported}.")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        extractor_kwargs = dict(extractor_conf or {})
        if "max_num_keypoints" not in extractor_kwargs and self.features == "superpoint":
            extractor_kwargs["max_num_keypoints"] = 2048

        extractor_cls = _EXTRACTOR_MAP[self.features]
        self.extractor = extractor_cls(**extractor_kwargs).eval().to(self.device)

        matcher_kwargs = dict(matcher_conf or {})
        self.matcher = LightGlue(features=self.features, **matcher_kwargs).eval().to(self.device)

    def match(
        self,
        image0: Union[PathLike, torch.Tensor],
        image1: Union[PathLike, torch.Tensor],
        *,
        load_conf: Optional[Dict] = None,
        **preprocess,
    ) -> Dict[str, object]:
        """Match two images and report latency.

        Args:
            image0/image1: Either file paths or pre-loaded ``torch.Tensor`` images.
            load_conf: Extra kwargs forwarded to ``lightglue.utils.load_image`` when the
                inputs are file paths. Example: ``{\"resize\": 1024}``.
            **preprocess: Additional preprocessing configuration for the extractor.

        Returns:
            Dict containing ``feats0``, ``feats1``, ``matches`` and ``latency``.
        """
        image0_tensor = self._prepare_image(image0, load_conf)
        image1_tensor = self._prepare_image(image1, load_conf)

        start = time.perf_counter()
        feats0, feats1, matches = match_pair(
            self.extractor, self.matcher, image0_tensor, image1_tensor, device=self.device, **preprocess
        )
        latency = time.perf_counter() - start

        keypoints_pair = self._get_matched_keypoints(feats0, feats1, matches)
        center_shift = self._calculate_shift(keypoints_pair, feats0, feats1)
        overlay = self._match_patch(image0_tensor, image1_tensor, keypoints_pair)

        return {
            "feats0": feats0,
            "feats1": feats1,
            "matches": matches,
            "latency": latency,
            "center_shift": center_shift,
            "overlay": overlay,
        }

    def _prepare_image(
        self, image: Union[PathLike, torch.Tensor], load_conf: Optional[Dict]
    ) -> torch.Tensor:
        """
        Convert an input image (file path or tensor) to a torch.Tensor on the correct device.

        Args:
            image (Union[PathLike, torch.Tensor]): Input image.
            load_conf (Optional[Dict]): Extra arguments for image loading.

        Returns:
            torch.Tensor: Image tensor on the configured device.
        """
        if isinstance(image, torch.Tensor):
            tensor = image
        else:
            load_kwargs = dict(load_conf or {})
            tensor = load_image(str(image), **load_kwargs)
        return tensor.to(self.device)

    def _get_matched_keypoints(
        self,
        feats0: Dict[str, torch.Tensor],
        feats1: Dict[str, torch.Tensor],
        matches: Dict[str, torch.Tensor],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract matched keypoints from features and match indices.

        Args:
            feats0 (Dict[str, torch.Tensor]): Feature dict for image0.
            feats1 (Dict[str, torch.Tensor]): Feature dict for image1.
            matches (Dict[str, torch.Tensor]): Match indices dict.

        Returns:
            Optional[Tuple[torch.Tensor, torch.Tensor]]: Tuple of matched keypoints
            (points0, points1), or None if no valid matches.
        """
        match_indices = matches.get("matches")
        if match_indices is None or match_indices.ndim != 2 or match_indices.shape[1] != 2:
            return None
        if match_indices.numel() == 0:
            return None

        valid = (match_indices >= 0).all(dim=1)
        if not valid.any().item():
            return None

        match_indices = match_indices[valid].to(device=feats0["keypoints"].device, dtype=torch.long)
        if match_indices.numel() == 0:
            return None

        points0 = feats0["keypoints"][match_indices[:, 0]]
        points1 = feats1["keypoints"][match_indices[:, 1]]
        if points0.shape[0] == 0 or points1.shape[0] == 0:
            return None

        return points0, points1

    def _calculate_shift(
        self,
        keypoints_pair: Optional[Tuple[torch.Tensor, torch.Tensor]],
        feats0: Dict[str, torch.Tensor],
        feats1: Dict[str, torch.Tensor],
    ) -> Optional[Dict[str, object]]:
        """
        Estimate translation between image centers from matched keypoints.
        
        Args:
            keypoints_pair (Optional[Tuple[torch.Tensor, torch.Tensor]]): Matched keypoints.
            feats0 (Dict[str, torch.Tensor]): Features for image0.
            feats1 (Dict[str, torch.Tensor]): Features for image1.

        Returns:
            Optional[Dict[str, object]]: Dictionary containing:
                - 'center_dx', 'center_dy': Mean translation in pixels.
                - 'num_matches': Number of valid matches.
                - 'center0', 'center1': Original image centers (if available).
                - 'center_displacement': Displacement vector after alignment.
                Returns None if keypoints_pair is None or invalid.
        """
        if keypoints_pair is None:
            return None

        points0, points1 = keypoints_pair
        translation = (points1 - points0).mean(dim=0)
        if not torch.isfinite(translation).all():
            return None

        size0 = feats0.get("image_size")
        size1 = feats1.get("image_size")
        if size0 is not None and size1 is not None:
            size0 = size0.to(translation)
            size1 = size1.to(translation)
            center0 = (size0 - 1.0) / 2.0
            center1 = (size1 - 1.0) / 2.0
        else:
            center0 = center1 = None

        result = {
            "center_dx": float(translation[0].item()),
            "center_dy": float(translation[1].item()),
            "num_matches": int(points0.shape[0]),
        }

        if center0 is not None and center1 is not None:
            aligned_center0 = center0 + translation
            displacement_vec = center1 - aligned_center0
            displacement = {
                "dx": float(displacement_vec[0].item()),
                "dy": float(displacement_vec[1].item()),
            }
            center0_tuple = tuple(map(float, center0.tolist()))
            center1_tuple = tuple(map(float, center1.tolist()))
            result.update(
                {
                    "center0": center0_tuple,
                    "center1": center1_tuple,
                    "center_displacement": displacement,
                }
            )

        return result

    def _match_patch(
        self,
        image0: torch.Tensor,
        image1: torch.Tensor,
        keypoints_pair: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        """
        Overlay image0 onto image1 by aligning matched keypoints via translation.
        
        Args:
            image0 (torch.Tensor): Source image to warp.
            image1 (torch.Tensor): Target image.
            keypoints_pair (Optional[Tuple[torch.Tensor, torch.Tensor]]): Matched keypoints.

        Returns:
            Optional[torch.Tensor]: Warped overlay image or None if keypoints_pair is None.

        """
        if keypoints_pair is None:
            return None

        points0, points1 = keypoints_pair
        translation = (points1 - points0).mean(dim=0)
        if not torch.isfinite(translation).all():
            return None

        if image0.shape[0] != image1.shape[0]:
            return None

        target_hw = image1.shape[-2], image1.shape[-1]
        aligned_image0 = self._apply_translation(image0, translation, target_hw=target_hw)
        overlay = 0.5 * aligned_image0 + 0.5 * image1.to(aligned_image0)
        return overlay.clamp(0.0, 1.0).detach().cpu()

    def _apply_translation(
        self,
        image: torch.Tensor,
        translation: torch.Tensor,
        *,
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Translate `image` by `translation` (dx, dy) using bilinear sampling.
        
        Args:
            image (torch.Tensor): Input image tensor of shape (C, H, W).
            translation (torch.Tensor): Translation vector (dx, dy).
            target_hw (Optional[Tuple[int, int]]): Target height and width. Defaults
                to source image size.

        Returns:
            torch.Tensor: Translated image tensor of shape (C, H, W).

        Raises:
            ValueError: If the input image tensor does not have 3 dimensions.
        """
        if image.dim() != 3:
            raise ValueError("Expected image tensor with shape (C, H, W).")

        _, src_h, src_w = image.shape
        tgt_h, tgt_w = target_hw if target_hw is not None else (src_h, src_w)

        if src_h <= 1 or src_w <= 1:
            return image.clone()

        device = image.device
        dtype = image.dtype

        # Create target pixel coordinate grid.
        ys = torch.arange(tgt_h, device=device, dtype=dtype)
        xs = torch.arange(tgt_w, device=device, dtype=dtype)
        if hasattr(torch, "meshgrid"):
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        else:
            grid_y, grid_x = torch.meshgrid(ys, xs)

        # Shift coordinates by translation to sample from source image.
        src_x = grid_x - translation[0]
        src_y = grid_y - translation[1]

        if src_w > 1:
            src_x_norm = (src_x / (src_w - 1)) * 2 - 1
        else:
            src_x_norm = src_x * 0
        if src_h > 1:
            src_y_norm = (src_y / (src_h - 1)) * 2 - 1
        else:
            src_y_norm = src_y * 0

        grid = torch.stack((src_x_norm, src_y_norm), dim=-1).unsqueeze(0)
        image_batch = image.unsqueeze(0)

        warped = F.grid_sample(
            image_batch,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return warped.squeeze(0)

if __name__ == "__main__":
    # IMAGE0_PATH = Path(r"C:\Users\sa-forest\Documents\GitHub\LightGlue\ex_data\cell_4.webp")
    # IMAGE1_PATH = Path(r"C:\Users\sa-forest\Documents\GitHub\LightGlue\ex_data\36712_1760473097.769556.webp")
    IMAGE0_PATH = Path(r"C:\Users\sa-forest\Documents\GitHub\LightGlue\ex_data\88602_1760469071.915733.webp")
    IMAGE1_PATH = Path(r"C:\Users\sa-forest\Documents\GitHub\LightGlue\ex_data\106826_1760469696.866317.webp")
    matcher = PointMatcher(features="superpoint")
    result = matcher.match(IMAGE0_PATH, IMAGE1_PATH)

    latency_ms = result["latency"] * 1e3
    print(f"Latency: {latency_ms:.2f} ms")
    shift = result.get("center_shift")
    if shift is None:
        print("Center shift: unavailable (no valid matches).")
    else:
        print(
            f"Center shift: dx={shift['center_dx']:.2f} px, dy={shift['center_dy']:.2f} px "
            f"(from {shift['num_matches']} matches)"
        )
        displacement = shift.get("center_displacement")
        if displacement is not None:
            print(
                f"Aligned center displacement: dx={displacement['dx']:.2f} px, "
                f"dy={displacement['dy']:.2f} px"
            )

    overlay = result.get("overlay")
    if overlay is None:
        print("Overlay: unavailable.")
    else:
        print(f"Overlay image shape: {tuple(overlay.shape)}")
        if plt is None:
            print("Matplotlib not installed; cannot display overlay.")
        else:
            overlay_np = overlay.permute(1, 2, 0).numpy()
            plt.figure("LightGlue Overlay")
            plt.imshow(overlay_np)
            plt.title("Overlay: image0 warped onto image1")
            plt.axis("off")
            plt.show()
