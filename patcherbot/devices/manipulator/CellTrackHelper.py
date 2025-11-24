from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

try:
    from patcherbot.deepLearning.PatchMatcher import (
        ImageInput,
        MatcherConfig,
        PatchMatcher,
    )
    from patcherbot.deepLearning.cellSegmentor import CellSegmentor2
except ModuleNotFoundError:  # pragma: no cover - allow standalone execution
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from patcherbot.deepLearning.PatchMatcher import (
        ImageInput,
        MatcherConfig,
        PatchMatcher,
    )
    from patcherbot.deepLearning.cellSegmentor import CellSegmentor2

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None

__all__ = ["CellTrackHelper"]


class CellTrackHelper:
    """
    Fuse LightGlue keypoint matching with SAM2 segmentation to recover a robust
    cell centroid in the current camera frame.

    Workflow:
        1. Segment the template (reference) image to estimate the true cell centroid.
        2. Use LightGlue (PatchMatcher) to stitch the template onto the current frame.
        3. Offset the matched target by the reference centroid offset.
        4. Re-run segmentation around the LightGlue prediction (with jittered seeds)
           and return the most plausible centroid. Fall back to the coarse target if
           segmentation fails.
    """

    # Offsets (in px) applied around the LightGlue prediction. Keep small so the
    # segmentor does not drift to unrelated structures during long moves.
    _DEFAULT_SEGMENTATION_OFFSETS: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.0),
        (18.0, 0.0),
        (-18.0, 0.0),
        (0.0, 18.0),
        (0.0, -18.0),
    )

    def __init__(self, stage, camera, **matcher_kwargs: object) -> None:
        self.stage = stage
        self.camera = camera
        self.width = int(getattr(camera, "width", 0) or 0)
        self.height = int(getattr(camera, "height", 0) or 0)
        self.segmentor = CellSegmentor2()
        self._matcher = PatchMatcher(**matcher_kwargs)

    # ------------------------------------------------------------------ #
    def find_centroid(
        self,
        reference_image: ImageInput,
        image: ImageInput,
        use_centroid: bool = True,
        *,
        prompt_point: Optional[Tuple[float, float]] = None,
        expected_point: Optional[Tuple[float, float]] = None,
        load_conf: Optional[MatcherConfig] = None,
        offsets: Optional[Sequence[Tuple[float, float]]] = None,
        max_refine_distance: float = 120.0,
        **preprocess: object,
    ) -> Optional[np.ndarray]:
        """
        Return the estimated cell centroid (in pixels) for *image*.

        Args:
            reference_image: Template captured when the user queued the cell.
            image: Current camera frame.
            use_centroid: When ``True`` (default) run SAM segmentation to recover a
                refined centroid. When ``False`` skip segmentation and return the
                LightGlue match directly.
            prompt_point: Optional positive point (x, y) used to seed the template
                segmentation (when enabled). Defaults to the template centre.
            expected_point: Optional prediction (x, y) of the cell position in the
                current frame, typically derived from stage bookkeeping. Used to
                gate LightGlue outliers and as an extra seed for refinement.
            load_conf: Configuration forwarded to ``PatchMatcher``.
            offsets: Extra seed offsets (px) for the refinement segmentation step.
            max_refine_distance: Reject segmentation results that stray further than
                this distance (px) from the LightGlue target.
            **preprocess: Forwarded to ``PatchMatcher.find_target``.
        """
        tmpl_np = self._ensure_numpy(reference_image)
        curr_np = self._ensure_numpy(image)
        if tmpl_np is None or curr_np is None:
            logging.error("CellTrackHelper: expected numpy arrays for matching.")
            return None

        tmpl_height, tmpl_width = tmpl_np.shape[:2]
        curr_height, curr_width = curr_np.shape[:2]
        tmpl_center = np.array([tmpl_width / 2.0, tmpl_height / 2.0], dtype=np.float32)
        if prompt_point is not None:
            prompt_raw = np.array(prompt_point, dtype=np.float32).reshape(-1)
            if prompt_raw.size < 2:
                logging.error(
                    "CellTrackHelper: prompt_point must have at least two values, got %s",
                    prompt_raw,
                )
                return None
            prompt = prompt_raw[:2]
        else:
            prompt = tmpl_center.copy()

        expected: Optional[np.ndarray] = None
        if expected_point is not None:
            expected_raw = np.array(expected_point, dtype=np.float32).reshape(-1)
            if expected_raw.size < 2:
                logging.error(
                    "CellTrackHelper: expected_point must have at least two values, got %s",
                    expected_raw,
                )
                return None
            expected = self._clamp_point(expected_raw[:2], curr_width, curr_height)

        # Step 1: Segment template to obtain the true centroid.
        reference_centroid: Optional[np.ndarray] = (
            self._segment_centroid(tmpl_np, prompt) if use_centroid else None
        )
        if reference_centroid is None:
            reference_centroid = prompt

        ref_offset = reference_centroid - tmpl_center

        # Step 2: LightGlue matching to locate the template in the live frame.
        current_center = np.array([curr_width / 2.0, curr_height / 2.0], dtype=np.float32)
        current_center_tuple = (float(current_center[0]), float(current_center[1]))
        try:
            match = self._matcher.find_target(
                self._prepare_match_tensor(tmpl_np),
                self._prepare_match_tensor(curr_np),
                current_center=current_center_tuple,
                load_conf=load_conf,
                **preprocess,
            )
        except Exception as exc:  # pragma: no cover - external dependency
            logging.error("CellTrackHelper: LightGlue matching failed: %s", exc)
            return None

        translation = match.get("translation")
        if translation is not None:
            translation = np.asarray(translation, dtype=np.float32)
            coarse_point = reference_centroid + translation
        else:
            target_point = match.get("target_point")
            if target_point is None:
                logging.error("CellTrackHelper: LightGlue result missing 'target_point'.")
                return None
            coarse_point = np.asarray(target_point, dtype=np.float32) + ref_offset
        coarse_point = self._clamp_point(coarse_point, curr_width, curr_height)

        offsets_to_use: Iterable[Tuple[float, float]] = (
            offsets if offsets is not None else self._DEFAULT_SEGMENTATION_OFFSETS
        )
        dynamic_offsets: list[Tuple[float, float]] = list(offsets_to_use)

        if expected is not None:
            delta_expected = float(np.linalg.norm(coarse_point - expected))
            if delta_expected > max_refine_distance:
                logging.warning(
                    "CellTrackHelper: LightGlue prediction deviates from stage estimate by %.1f px; falling back to stage prediction.",
                    delta_expected,
                )
                coarse_point = expected.copy()
            else:
                offset_vec = expected - coarse_point
                if np.any(np.abs(offset_vec) > 1e-3):
                    dynamic_offsets.insert(0, (float(offset_vec[0]), float(offset_vec[1])))

        offsets_for_refine: Tuple[Tuple[float, float], ...] = tuple(dynamic_offsets)

        # Step 3: Re-run segmentation around the coarse prediction if requested.
        if use_centroid:
            refined = self._refine_with_segmentation(
                curr_np,
                coarse_point,
                offsets_for_refine,
                max_refine_distance=max_refine_distance,
            )
            final_point = refined if refined is not None else coarse_point
        else:
            final_point = coarse_point
        return final_point.astype(np.float32)

    # ------------------------------------------------------------------ #
    def _refine_with_segmentation(
        self,
        image: np.ndarray,
        coarse_point: np.ndarray,
        offsets: Iterable[Tuple[float, float]],
        *,
        max_refine_distance: float,
    ) -> Optional[np.ndarray]:
        """Try segmentation with jittered seeds and keep the closest valid centroid."""
        try:
            self._prime_segmentor(image)
        except Exception as exc:
            logging.error("CellTrackHelper: unable to prepare segmentor: %s", exc)
            return None

        best_candidate: Optional[np.ndarray] = None
        best_distance: Optional[float] = None

        for dx, dy in offsets:
            seed = coarse_point + np.array([dx, dy], dtype=np.float32)
            if not self._point_within(seed, image.shape[1], image.shape[0]):
                continue

            centroid = self._segment_from_seed(seed)
            if centroid is None:
                continue

            distance = float(np.linalg.norm(centroid - coarse_point))
            if distance > max_refine_distance:
                continue

            if best_distance is None or distance < best_distance:
                best_candidate = centroid
                best_distance = distance

        return best_candidate

    # ------------------------------------------------------------------ #
    def _segment_centroid(
        self,
        image: np.ndarray,
        seed_point: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Convenience wrapper for single-pass segmentation on *image*."""
        try:
            self._prime_segmentor(image)
        except Exception as exc:
            logging.error("CellTrackHelper: could not load template for segmentation: %s", exc)
            return None
        return self._segment_from_seed(seed_point)

    def _prime_segmentor(self, image: np.ndarray) -> None:
        """Load *image* into the SAM2 predictor."""
        prepped = self._prepare_for_segmentation(image)
        self.segmentor.load_image(image=prepped)
        self.segmentor.set_image()

    def _segment_from_seed(self, seed_point: np.ndarray) -> Optional[np.ndarray]:
        """Run SAM2 with a single positive seed and return the centroid."""
        point = np.asarray(seed_point, dtype=np.float32)
        if point.shape != (2,):
            point = point.reshape(2)

        pts = np.array([point], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)

        try:
            mask = self.segmentor.single_prediction(
                input_point=pts,
                input_label=labels,
                multimask_output=False,
            )
        except Exception as exc:  # pragma: no cover - external dependency
            logging.error("CellTrackHelper: SAM2 segmentation failed: %s", exc)
            return None

        if mask is None:
            logging.warning("CellTrackHelper: SAM2 returned no mask.")
            return None


        return self._compute_centroid(mask)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_centroid(mask: np.ndarray) -> Optional[np.ndarray]:
        """Return centroid `[cx, cy]` (float32) or `None` if mask is empty."""
        if mask.ndim == 3:
            mask = mask[0]

        mask_bool = np.asarray(mask > 0)
        mask_bool = np.squeeze(mask_bool)
        if mask_bool.ndim != 2:
            logging.warning("CellTrackHelper: unexpected mask shape %s.", mask_bool.shape)
            return None
        size = int(np.count_nonzero(mask_bool))
        if size == 0:
            logging.warning("CellTrackHelper: segmentation mask has zero area.")
            return None
        if size < 5:
            logging.warning("CellTrackHelper: SAM2 mask too small (%d px).", size)
            return None

        height, width = mask_bool.shape[:2]
        image_area = height * width
        if image_area > 0 and size > image_area * 0.25:
            percentage = 100.0 * size / float(image_area)
            # logging.warning(
            #     "CellTrackHelper: SAM2 mask too large (%d px, %.1f%% of image).",
            #     size,
            #     percentage,
            # )
            return None

        mask_uint8 = mask_bool.astype(np.uint8)
        moments = cv2.moments(mask_uint8)
        if moments["m00"] == 0.0:
            logging.warning("CellTrackHelper: segmentation mask has zero area.")
            return None
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        return np.array([cx, cy], dtype=np.float32)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _prepare_match_tensor(image: np.ndarray) -> torch.Tensor:
        """Convert image to a 3xHxW float tensor in [0, 1] for LightGlue."""
        rgb = CellTrackHelper._ensure_rgb(image)
        arr = rgb.astype(np.float32)
        if arr.max() > 1.0:
            dtype_max = {
                np.uint8: 255.0,
                np.uint16: 65535.0,
            }.get(arr.dtype.type, float(arr.max()))
            arr /= dtype_max if dtype_max > 0 else 255.0
        arr = np.clip(arr, 0.0, 1.0)
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))  # C, H, W
        return tensor

    @staticmethod
    def _prepare_for_segmentation(image: np.ndarray) -> np.ndarray:
        """Return a uint8 BGR image suitable for SAM2."""
        bgr = CellTrackHelper._ensure_bgr(image)
        if bgr.dtype == np.uint16:
            bgr = (bgr / 257.0).astype(np.uint8)
        elif bgr.dtype != np.uint8:
            arr = bgr.astype(np.float32)
            if arr.max() > 1.0:
                arr /= arr.max()
            bgr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return bgr

    @staticmethod
    def _ensure_numpy(image: ImageInput) -> Optional[np.ndarray]:
        """Convert supported image inputs into numpy arrays."""
        if isinstance(image, np.ndarray):
            return image
        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim == 3:
                tensor = tensor.permute(1, 2, 0)
            elif tensor.ndim == 2:
                tensor = tensor.unsqueeze(-1)
            return tensor.numpy()
        logging.error("CellTrackHelper: unsupported image type %s", type(image))
        return None

    @staticmethod
    def _ensure_rgb(image: np.ndarray) -> np.ndarray:
        """Ensure *image* is an RGB array (H, W, 3)."""
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.ndim == 3 and image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.ndim == 3 and image.shape[2] == 3:
            # Assume input already RGB/BGR; LightGlue is agnostic to channel order.
            return image
        raise ValueError(f"Unsupported image shape for RGB conversion: {image.shape}")

    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        """Ensure *image* is a BGR array (H, W, 3)."""
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        raise ValueError(f"Unsupported image shape for BGR conversion: {image.shape}")

    @staticmethod
    def _clamp_point(point: np.ndarray, width: int, height: int) -> np.ndarray:
        """Clamp a point to image bounds."""
        x = float(np.clip(point[0], 0.0, max(width - 1, 0)))
        y = float(np.clip(point[1], 0.0, max(height - 1, 0)))
        return np.array([x, y], dtype=np.float32)

    @staticmethod
    def _point_within(point: np.ndarray, width: int, height: int) -> bool:
        """Check that *point* lies inside image bounds."""
        x, y = float(point[0]), float(point[1])
        return 0.0 <= x < width and 0.0 <= y < height


class CellTrackTester:
    """Minimal harness to visualise CellTrackHelper on a pair of images."""

    class _MockStage:
        """Placeholder stage; CellTrackHelper does not use it in offline tests."""

        def __init__(self) -> None:
            self.name = "MockStage"

    class _MockCamera:
        """Simple camera stub exposing image dimensions."""

        def __init__(self, width: int, height: int) -> None:
            self.width = width
            self.height = height

    def __init__(
        self,
        reference_image: Path | str,
        current_image: Path | str,
        *,
        features: str = "superpoint",
        offsets: Optional[Sequence[Tuple[float, float]]] = None,
        max_refine_distance: float = 120.0,
    ) -> None:
        self.reference_path = Path(reference_image)
        self.current_path = Path(current_image)
        self.reference_image = self._load_image(self.reference_path)
        self.current_image = self._load_image(self.current_path)

        height, width = self.current_image.shape[:2]
        self._camera = self._MockCamera(width, height)
        self._stage = self._MockStage()

        matcher_kwargs = {}
        if features:
            matcher_kwargs["features"] = features

        self.helper = CellTrackHelper(self._stage, self._camera, **matcher_kwargs)
        self.offsets = tuple(offsets) if offsets is not None else None
        self.max_refine_distance = max_refine_distance

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Execute LightGlue + SAM2 pipeline and plot overlays."""
        prompt = np.array(
            [self.reference_image.shape[1] / 2.0, self.reference_image.shape[0] / 2.0],
            dtype=np.float32,
        )

        ref_mask, ref_centroid = self._segment_template(prompt)
        if ref_centroid is None:
            logging.warning("CellTrackTester: template segmentation failed.")
        else:
            logging.info("Template centroid: (%.2f, %.2f)", ref_centroid[0], ref_centroid[1])

        ref_centre = ref_centroid if ref_centroid is not None else prompt
        match_info = self._compute_lightglue(ref_mask, ref_centre)
        coarse_point = match_info["coarse_point"]
        logging.info("Coarse target (LightGlue + offset): (%.2f, %.2f)", coarse_point[0], coarse_point[1])

        current_mask, refined_centroid = self._segment_current(coarse_point)
        if refined_centroid is not None:
            logging.info("Refined centroid: (%.2f, %.2f)", refined_centroid[0], refined_centroid[1])
        else:
            logging.warning("CellTrackTester: current-frame segmentation failed.")

        self._display_results(
            reference_mask=ref_mask,
            warped_mask=match_info.get("warped_mask"),
            current_mask=current_mask,
            overlay_image=match_info.get("overlay"),
            reference_centroid=ref_centroid,
            coarse_point=coarse_point,
            refined_centroid=refined_centroid,
        )

    # ------------------------------------------------------------------ #
    def _compute_lightglue(
        self,
        reference_mask: Optional[np.ndarray],
        reference_centroid: np.ndarray,
    ) -> dict:
        tmpl_np = self.reference_image
        curr_np = self.current_image

        tmpl_height, tmpl_width = tmpl_np.shape[:2]
        curr_height, curr_width = curr_np.shape[:2]
        tmpl_center = np.array([tmpl_width / 2.0, tmpl_height / 2.0], dtype=np.float32)

        ref_offset = reference_centroid - tmpl_center

        current_center = np.array([curr_width / 2.0, curr_height / 2.0], dtype=np.float32)
        current_center_tuple = (float(current_center[0]), float(current_center[1]))
        match = self.helper._matcher.find_target(
            self.helper._prepare_match_tensor(tmpl_np),
            self.helper._prepare_match_tensor(curr_np),
            current_center=current_center_tuple,
        )

        translation_vec = match.get("translation")
        if translation_vec is not None:
            translation_arr = np.asarray(translation_vec, dtype=np.float32)
            coarse_point = reference_centroid + translation_arr
            translation = translation_arr
        else:
            target_point = np.asarray(match["target_point"], dtype=np.float32)
            coarse_point = target_point + ref_offset
            translation = coarse_point - reference_centroid

        coarse_point = self.helper._clamp_point(coarse_point, curr_width, curr_height)
        warp_matrix = np.float32(
            [
                [1.0, 0.0, float(translation[0])],
                [0.0, 1.0, float(translation[1])],
            ]
        )
        warped_template = cv2.warpAffine(
            tmpl_np,
            warp_matrix,
            (curr_width, curr_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        overlay = (
            0.5 * curr_np.astype(np.float32) + 0.5 * warped_template.astype(np.float32)
        )
        overlay = np.clip(overlay, 0.0, 255.0).astype(np.uint8)

        warped_mask = None
        if reference_mask is not None:
            if reference_mask.ndim == 3:
                reference_mask = reference_mask[0]
            translation = coarse_point - reference_centroid
            warp_matrix = np.float32([[1.0, 0.0, translation[0]], [0.0, 1.0, translation[1]]])
            warped_mask = cv2.warpAffine(
                reference_mask.astype(np.uint8),
                warp_matrix,
                (curr_width, curr_height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        return {
            "match": match,
            "coarse_point": coarse_point,
            "overlay": overlay,
            "warped_mask": warped_mask,
        }

    def _segment_template(
        self,
        prompt_point: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Segment the template image to recover mask + centroid."""
        try:
            self.helper._prime_segmentor(self.reference_image)
        except Exception as exc:  # pragma: no cover - external dependency
            logging.error("CellTrackTester: failed to prepare template for segmentation: %s", exc)
            return None, None

        pts = np.array([prompt_point], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)

        mask = self.helper.segmentor.single_prediction(
            input_point=pts,
            input_label=labels,
            multimask_output=False,
        )
        if mask is None:
            return None, None

        centroid = self.helper._compute_centroid(mask)
        return mask, centroid

    def _segment_current(
        self,
        coarse_point: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Segmentation refinement around the LightGlue target in the current frame."""
        try:
            self.helper._prime_segmentor(self.current_image)
        except Exception as exc:  # pragma: no cover - external dependency
            logging.error("CellTrackTester: failed to prepare current image for segmentation: %s", exc)
            return None, None

        offsets = self.offsets or self.helper._DEFAULT_SEGMENTATION_OFFSETS
        best_mask: Optional[np.ndarray] = None
        best_centroid: Optional[np.ndarray] = None
        best_distance: Optional[float] = None

        curr_height, curr_width = self.current_image.shape[:2]
        for dx, dy in offsets:
            seed = coarse_point + np.array([dx, dy], dtype=np.float32)
            if not self.helper._point_within(seed, curr_width, curr_height):
                continue

            pts = np.array([seed], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)
            mask = self.helper.segmentor.single_prediction(
                input_point=pts,
                input_label=labels,
                multimask_output=False,
            )
            if mask is None:
                continue

            centroid = self.helper._compute_centroid(mask)
            if centroid is None:
                continue

            distance = float(np.linalg.norm(centroid - coarse_point))
            if distance > self.max_refine_distance:
                continue

            if best_distance is None or distance < best_distance:
                best_mask = mask
                best_centroid = centroid
                best_distance = distance

        return best_mask, best_centroid

    # ------------------------------------------------------------------ #
    def _display_results(
        self,
        *,
        reference_mask: Optional[np.ndarray],
        warped_mask: Optional[np.ndarray],
        current_mask: Optional[np.ndarray],
        overlay_image: Optional[np.ndarray],
        reference_centroid: Optional[np.ndarray],
        coarse_point: np.ndarray,
        refined_centroid: Optional[np.ndarray],
    ) -> None:
        template_overlay = self._overlay_mask(self.reference_image.copy(), reference_mask, color=(0, 165, 255))
        if reference_centroid is not None:
            self._draw_marker(template_overlay, reference_centroid, color=(0, 140, 255))
        overlay_with_mask = None
        if overlay_image is not None:
            overlay_bgr = overlay_image.copy()
            overlay_with_mask = self._overlay_mask(overlay_bgr, warped_mask, color=(0, 255, 0))
            self._draw_marker(overlay_with_mask, coarse_point, color=(255, 200, 0))
            if refined_centroid is not None:
                self._draw_marker(overlay_with_mask, refined_centroid, color=(0, 255, 255))

        current_overlay = self._overlay_mask(self.current_image.copy(), current_mask, color=(0, 255, 0))
        self._draw_marker(current_overlay, coarse_point, color=(255, 200, 0))
        if refined_centroid is not None:
            self._draw_marker(current_overlay, refined_centroid, color=(0, 255, 255))

        print("--- CellTrackTester ---")
        print(f"Template path: {self.reference_path}")
        print(f"Current path : {self.current_path}")
        print(f"Coarse centroid : {coarse_point}")
        if refined_centroid is not None:
            print(f"Refined centroid: {refined_centroid}")
        if reference_centroid is not None:
            print(f"Reference centroid: {reference_centroid}")

        if plt is None:
            logging.warning("Matplotlib not available; skipping visualisation.")
            return

        images = [
            ("Template + mask", template_overlay),
            ("Stitched overlay", overlay_with_mask if overlay_with_mask is not None else template_overlay),
            ("Current + refined mask", current_overlay),
        ]

        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for ax, (title, img) in zip(axes, images):
            ax.imshow(self._bgr_to_rgb(img))
            ax.set_title(title)
            ax.axis("off")

        summary = f"Coarse: ({coarse_point[0]:.1f}, {coarse_point[1]:.1f})"
        if refined_centroid is not None:
            summary += f" | Refined: ({refined_centroid[0]:.1f}, {refined_centroid[1]:.1f})"
        fig.suptitle(summary)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _overlay_mask(
        image: np.ndarray,
        mask: Optional[np.ndarray],
        *,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Overlay *mask* onto *image* using the provided colour."""
        output = image.copy()
        if mask is None:
            return output

        if mask.ndim == 3:
            mask_2d = mask[0]
        else:
            mask_2d = mask

        if mask_2d.dtype != np.uint8:
            mask_bool = mask_2d.astype(np.float32) > 0.0
        else:
            mask_bool = mask_2d > 0

        if not np.any(mask_bool):
            return output

        colour_layer = np.zeros_like(output, dtype=np.float32)
        colour_layer[:] = color

        blended = output.astype(np.float32)
        blended[mask_bool] = (
            blended[mask_bool] * (1.0 - alpha) + colour_layer[mask_bool] * alpha
        )
        blended = np.clip(blended, 0.0, 255.0)
        return blended.astype(np.uint8)

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Unable to load image from {path}")
        return img

    @staticmethod
    def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _draw_marker(image: np.ndarray, point: Optional[np.ndarray], color: Tuple[int, int, int]) -> None:
        if point is None:
            return
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.drawMarker(
            image,
            (x, y),
            color,
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=16,
            thickness=2,
            line_type=cv2.LINE_AA,
        )


if __name__ == "__main__":
    # DEFAULT_CUR = Path(r"C:\Users\sa-forest\Documents\GitHub\LightGlue\ex_data\88602_1760469071.915733.webp")
    # DEFAULT_CUR = Path(r"C:\Users\sa-forest\Documents\GitHub\LightGlue\ex_data\106826_1760469696.866317.webp")

    DEFAULT_REF = Path(r"C:\Users\sa-forest\Documents\GitHub\LightGlue\ex_data\cell_4.webp")
    DEFAULT_CUR = Path(r"C:\Users\sa-forest\Documents\GitHub\LightGlue\ex_data\36712_1760473097.769556.webp")

    tester = CellTrackTester(DEFAULT_REF, DEFAULT_CUR, features="superpoint")
    tester.run()
